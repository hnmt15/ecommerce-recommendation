from flask import Flask, jsonify, request
from flask_cors import CORS
import sys
import pandas as pd
from pathlib import Path

sys.path.append(r"C:\TTNT\ecommerce-recommendation\recommendation_system")

from utils.data_loader import DataLoader
from utils.preprocess import Preprocessor
from utils.evaluator import Evaluator
from models.content_model import ContentBasedRecommender
from models.collab_filtering_model import CollabRecommender
from models.hybrid_model import HybridRecommender

app = Flask(__name__)
CORS(app)

print("⏳ Đang load model...")

loader = DataLoader()
data = loader.load_all_data()
df = loader.merge_reviews_with_products(data)

preprocessor = Preprocessor()
clean_df, product_df = preprocessor.preprocess(
    df, min_user_ratings=3, min_item_ratings=2
)

MODELS_PATH = r"C:\TTNT\ecommerce-recommendation\recommendation_system\models"

cb = ContentBasedRecommender()
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

evaluator = Evaluator()

_metrics_cache = None


def get_metrics():
    global _metrics_cache
    if _metrics_cache is not None:
        return _metrics_cache

    print("⏳ Đang tính metrics...")
    try:
        test_df = clean_df.groupby('user_id').tail(
            max(1, int(len(clean_df) * 0.2 / clean_df['user_id'].nunique()))
        )
        sample_users = clean_df['user_id'].unique()[:50].tolist()

        cb_results = evaluator.evaluate_content_based(cb, product_df, sample_size=100, k=10)
        cf_results = evaluator.evaluate_collaborative(cf, test_df, top_n=10)
        hybrid_results = evaluator.evaluate_hybrid(
            hybrid, loader, data, product_df, test_df,
            user_ids=sample_users, k=10
        )

        _metrics_cache = {
            'content_based': {
                'precision@10': round(cb_results.get('precision@10', 0), 4),
                'recall@10':    round(cb_results.get('recall@10', 0), 4),
                'ndcg@10':      round(cb_results.get('ndcg@10', 0), 4),
                'hit_rate':     round(cb_results.get('hit_rate', 0), 4),
            },
            'collaborative': {
                'precision@10': round(cf_results.get('precision', 0), 4),
                'recall@10':    round(cf_results.get('recall', 0), 4),
                'ndcg@10':      round(cf_results.get('ndcg@10', 0), 4),
                'rmse':         round(cf_results.get('rmse', 0), 4),
                'mae':          round(cf_results.get('mae', 0), 4),
                'hit_rate':     round(cf_results.get('hit_rate', 0), 4),
            },
            'hybrid': {
                'precision@10': round(hybrid_results.get('precision@10', 0), 4),
                'recall@10':    round(hybrid_results.get('recall@10', 0), 4),
                'ndcg@10':      round(hybrid_results.get('ndcg@10', 0), 4),
                'hit_rate':     round(hybrid_results.get('hit_rate', 0), 4),
            }
        }
        print("✅ Metrics đã tính xong!")
    except Exception as e:
        print(f"❌ Lỗi tính metrics: {e}")
        _metrics_cache = {'error': str(e)}

    return _metrics_cache


def compute_popular(top_n=20):
    try:
        if data.get('order_items') is not None:
            order_items = data['order_items']
            if 'product_id' in order_items.columns:
                buy_counts = order_items.groupby('product_id').size().reset_index(name='buy_count')
            else:
                buy_counts = pd.DataFrame(columns=['product_id', 'buy_count'])
        else:
            buy_counts = pd.DataFrame(columns=['product_id', 'buy_count'])

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

        popular = product_df[['product_id', 'product_name', 'category']].drop_duplicates('product_id')

        if not buy_counts.empty:
            popular = popular.merge(buy_counts, on='product_id', how='left')
        else:
            popular['buy_count'] = 0

        if not avg_ratings.empty:
            popular = popular.merge(avg_ratings, on='product_id', how='left')
        else:
            popular['avg_rating'] = 0

        popular['buy_count'] = popular['buy_count'].fillna(0).astype(int)
        popular['avg_rating'] = popular['avg_rating'].fillna(0).round(2)
        popular['score'] = popular['buy_count'] * 0.7 + popular['avg_rating'] * 0.3
        popular = popular.sort_values('score', ascending=False).head(top_n)

        return popular[['product_name', 'category', 'buy_count', 'avg_rating']].to_dict(orient='records')
    except Exception as e:
        print(f"Lỗi compute_popular: {e}")
        fallback = product_df[['product_name', 'category']].drop_duplicates().head(top_n)
        fallback['buy_count'] = 0
        fallback['avg_rating'] = 0
        return fallback.to_dict(orient='records')


POPULAR_PRODUCTS = compute_popular(top_n=20)
print(f"✅ Model sẵn sàng! Top {len(POPULAR_PRODUCTS)} sản phẩm phổ biến đã tính.")


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


@app.route('/api/popular', methods=['GET'])
def popular():
    return jsonify({'products': POPULAR_PRODUCTS})


@app.route('/api/metrics', methods=['GET'])
def metrics():
    result = get_metrics()
    if 'error' in result:
        return jsonify({'error': result['error']}), 500
    return jsonify(result)


@app.route('/api/recommend', methods=['GET'])
def recommend():
    user_input = request.args.get('user_id', '').strip()
    top_n      = int(request.args.get('top_n', 10))
    model_type = request.args.get('model', 'hybrid').lower()

    if not user_input:
        return jsonify({'error': 'Vui lòng nhập User ID'}), 400

    all_user_ids = clean_df['user_id'].unique().tolist()
    matched = [u for u in all_user_ids if user_input in str(u)]

    if not matched:
        return jsonify({'error': f'Không tìm thấy user "{user_input}"'}), 404

    user_id = matched[0]
    user_history = loader.get_user_history(data, user_id=user_id)

    if user_history.empty:
        return jsonify({'error': 'User không có lịch sử tương tác'}), 404

    n_interactions = len(user_history)
    is_cold = n_interactions < 3
    source_counts = user_history['source'].value_counts().to_dict()

    # ── Danh sách sản phẩm đã tương tác ──────────────────────────────
    interactions = []
    for _, row in user_history.iterrows():
        interactions.append({
            'product_name': str(row.get('product_name', f"ID: {row.get('product_id', '?')}")),
            'event_type':   str(row.get('source', row.get('event_type', 'view'))),
        })

    # ── Chọn model theo param ─────────────────────────────────────────
    recommendations = []

    if model_type == 'content':
        result = cb.recommend_for_user(
            user_history_df=user_history[['product_name', 'weight']],
            top_n=top_n,
            weight_col='weight'
        )
        mode = 'Content-Based'
        if not result.empty:
            for i, row in result.iterrows():
                recommendations.append({
                    'rank':         i + 1,
                    'product_name': row['product_name'],
                    'category':     row.get('category', ''),
                    'score':        round(float(row.get('score', row.get('similarity_score', 0))), 4),
                    'reason':       'cb',
                })

    elif model_type == 'collab':
        rec_ids = cf.recommend(user_id, top_n=top_n)
        mode = 'Collaborative'
        matched_products = product_df[product_df['product_id'].isin(rec_ids)].drop_duplicates('product_id')
        for rank, (_, row) in enumerate(matched_products.iterrows(), 1):
            recommendations.append({
                'rank':         rank,
                'product_name': row['product_name'],
                'category':     row.get('category', ''),
                'score':        0.0,
                'reason':       'cf',
            })

    else:  # hybrid (default)
        result = hybrid.recommend(
            user_id=user_id,
            user_history_df=user_history,
            product_df=product_df,
            top_n=top_n
        )
        mode = 'Cold Start' if is_cold else 'Hybrid'
        if not result.empty:
            for i, row in result.iterrows():
                if is_cold:
                    reason = 'cb'
                else:
                    cb_score = float(row.get('cb_score', 0))
                    cf_score = float(row.get('cf_score', 0))
                    reason = 'cb' if cb_score >= cf_score else 'cf'

                recommendations.append({
                    'rank':         i + 1,
                    'product_name': row['product_name'],
                    'category':     row.get('category', ''),
                    'score':        round(float(row['final_score']), 4),
                    'reason':       reason,
                })

    if not recommendations:
        return jsonify({'error': 'Không tìm được gợi ý'}), 404

    return jsonify({
        'user_id':         user_id,
        'n_interactions':  n_interactions,
        'mode':            mode,
        'model_type':      model_type,
        'source_counts':   source_counts,
        'interactions':    interactions,
        'recommendations': recommendations,
    })


@app.route('/api/users', methods=['GET'])
def get_users():
    sample = clean_df['user_id'].unique()[:10].tolist()
    return jsonify({'users': sample})


if __name__ == '__main__':
    app.run(debug=True, port=5000)