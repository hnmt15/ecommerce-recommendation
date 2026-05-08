from flask import Flask, jsonify, request
from flask_cors import CORS
import sys
import pandas as pd
from pathlib import Path

sys.path.append(r"C:\TTNT\ecommerce-recommendation\recommendation_system")

from utils.data_loader import DataLoader
from utils.preprocess import Preprocessor
from models.content_model import ContentBasedRecommender
from models.collab_filtering_model import CollabRecommender
from models.hybrid_model import HybridRecommender

app = Flask(__name__)
CORS(app)

# ══════════════════════════════════════════════════════════════════════
# LOAD MODEL KHI KHỞI ĐỘNG SERVER
# ══════════════════════════════════════════════════════════════════════
print("⏳ Đang load model...")

loader = DataLoader()
data = loader.load_all_data()
df = loader.merge_reviews_with_products(data)

preprocessor = Preprocessor()
clean_df, product_df = preprocessor.preprocess(
    df, min_user_ratings=3, min_item_ratings=2
)

cb = ContentBasedRecommender()
MODELS_PATH = r"C:\TTNT\ecommerce-recommendation\recommendation_system\models"

cb.load_model(f'{MODELS_PATH}/content_model.pkl')

cf = CollabRecommender(
    model_path=f'{MODELS_PATH}/collaborative_model.pth',
    map_path=f'{MODELS_PATH}/maps.pkl'
)

hybrid = HybridRecommender(
    content_recommender=cb,
    collab_recommender=cf,
    cb_weight=0.4,
    cf_weight=0.6,
    cold_start_threshold=3
)


# Tính sẵn top sản phẩm phổ biến (cho trang khách)
# Dựa trên order_items (mua nhiều nhất) + rating trung bình
def compute_popular(top_n=20):
    try:
        # Đếm số lượt mua từ order_items
        if data.get('order_items') is not None and data.get('orders') is not None:
            order_items = data['order_items']
            orders = data['orders']

            # Join order_items với orders để lấy product_id
            if 'product_id' in order_items.columns:
                buy_counts = order_items.groupby('product_id').size().reset_index(name='buy_count')
            else:
                buy_counts = pd.DataFrame(columns=['product_id', 'buy_count'])
        else:
            buy_counts = pd.DataFrame(columns=['product_id', 'buy_count'])

        # Rating trung bình từ reviews
        if data.get('reviews') is not None:
            avg_ratings = (
                data['reviews']
                .groupby('product_id')['rating']
                .agg(['mean', 'count'])
                .reset_index()
                .rename(columns={'mean': 'avg_rating', 'count': 'n_reviews'})
            )
        else:
            avg_ratings = pd.DataFrame(columns=['product_id', 'avg_rating', 'n_reviews'])

        # Gộp với thông tin sản phẩm
        popular = product_df[['product_id', 'product_name', 'category']].drop_duplicates('product_id')

        if not buy_counts.empty:
            popular = popular.merge(buy_counts, on='product_id', how='left')
        else:
            popular['buy_count'] = 0

        if not avg_ratings.empty:
            popular = popular.merge(avg_ratings, on='product_id', how='left')
        else:
            popular['avg_rating'] = 0
            popular['n_reviews'] = 0

        popular['buy_count'] = popular['buy_count'].fillna(0).astype(int)
        popular['avg_rating'] = popular['avg_rating'].fillna(0).round(2)

        # Sắp xếp: mua nhiều nhất + rating cao
        popular['score'] = popular['buy_count'] * 0.7 + popular['avg_rating'] * 0.3
        popular = popular.sort_values('score', ascending=False).head(top_n)

        return popular[['product_name', 'category', 'buy_count', 'avg_rating']].to_dict(orient='records')

    except Exception as e:
        print(f"Lỗi compute_popular: {e}")
        # Fallback: lấy sản phẩm theo rating cao nhất
        fallback = (
            product_df[['product_name', 'category']]
            .drop_duplicates()
            .head(top_n)
        )
        fallback['buy_count'] = 0
        fallback['avg_rating'] = 0
        return fallback.to_dict(orient='records')


POPULAR_PRODUCTS = compute_popular(top_n=20)
print(f"✅ Model sẵn sàng! Top {len(POPULAR_PRODUCTS)} sản phẩm phổ biến đã tính.")


# ══════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ══════════════════════════════════════════════════════════════════════

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


@app.route('/api/popular', methods=['GET'])
def popular():
    """
    Trả về top sản phẩm phổ biến (cho trang khách chưa đăng nhập)
    Sắp xếp theo: lượt mua × 0.7 + rating × 0.3
    """
    return jsonify({'products': POPULAR_PRODUCTS})


@app.route('/api/recommend', methods=['GET'])
def recommend():
    """
    Gợi ý sản phẩm cá nhân hóa cho user đã đăng nhập
    Params: user_id (string), top_n (int, default=10)
    """
    user_input = request.args.get('user_id', '').strip()
    top_n = int(request.args.get('top_n', 10))

    if not user_input:
        return jsonify({'error': 'Vui lòng nhập User ID'}), 400

    # Tìm user_id khớp
    all_user_ids = clean_df['user_id'].unique().tolist()
    matched = [u for u in all_user_ids if user_input in str(u)]

    if not matched:
        return jsonify({'error': f'Không tìm thấy user "{user_input}"'}), 404

    user_id = matched[0]

    # Lấy lịch sử
    user_history = loader.get_user_history(data, user_id=user_id)

    if user_history.empty:
        return jsonify({'error': 'User không có lịch sử tương tác'}), 404

    # Thông tin user
    n_interactions = len(user_history)
    is_cold = n_interactions < 3
    mode = 'Cold Start' if is_cold else 'Hybrid'
    source_counts = user_history['source'].value_counts().to_dict()

    # Gợi ý
    result = hybrid.recommend(
        user_id=user_id,
        user_history_df=user_history,
        product_df=product_df,
        top_n=top_n
    )

    if result.empty:
        return jsonify({'error': 'Không tìm được gợi ý'}), 404

    recommendations = []
    for i, row in result.iterrows():
        recommendations.append({
            'rank': i + 1,
            'product_name': row['product_name'],
            'category': row['category'],
            'score': round(float(row['final_score']), 4),
        })

    return jsonify({
        'user_id': user_id,
        'n_interactions': n_interactions,
        'mode': mode,
        'source_counts': source_counts,
        'recommendations': recommendations
    })


@app.route('/api/users', methods=['GET'])
def get_users():
    """Lấy mẫu user ID để thử nhanh"""
    sample = clean_df['user_id'].unique()[:10].tolist()
    return jsonify({'users': sample})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
