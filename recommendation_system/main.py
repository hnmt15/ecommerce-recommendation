import random
import pandas as pd
from utils.data_loader import DataLoader
from utils.preprocess import Preprocessor
from utils.evaluator import Evaluator
from models.content_model import ContentBasedRecommender


def main():
    print("STARTING HANDLE ECOMMERCE DATA")

    # 1. Load data
    loader = DataLoader()
    data = loader.load_all_data()
    df = loader.merge_reviews_with_products(data)

    if df is None:
        print("Can't load data!")
        return

    # 2. Preprocess (nhận về 2 DataFrame)
    preprocessor = Preprocessor()
    clean_df, product_df = preprocessor.preprocess(
        df,
        min_user_ratings=3,
        min_item_ratings=2
    )
    if product_df is None or clean_df is None:
        print("Preprocessing failed")

    preprocessor.save_processed(clean_df, product_df)


    # 4. Dùng clean_df cho collaborative
    train_interactions, test_interactions = preprocessor.split_for_collaborative(clean_df)

    print(f"\nDONE!")
    print(f"   Content-based: {len(product_df)} train, {len(product_df)} test products")
    print(f"   Collaborative: {len(train_interactions)} train, {len(test_interactions)} test interactions")

    # 6. RUN CONTENT-BASED MODEL
    print("\n1. RUN CONTENT-BASED MODEL")
    recommender = ContentBasedRecommender()
    recommender.train_model(df=product_df, text_column='combined_text', title_column='product_name')
    recommender.save_model('models/content_model.pkl')


    # random_product = random.choice(train_products['product_name'].unique())
    # recommender.recommend_and_display(random_product, top_n=5)

    #sample_products = product_df['product_name'].sample(n=min(5, len(product_df))).tolist()
    top_users = loader.get_top_users(data, n=10, aggregate=True)

    user_history_df = pd.DataFrame()  # Khởi tạo empty
    real_user_id = None

    if top_users:
        for user_id in top_users:
            user_history = loader.get_user_history(data, user_id=user_id, min_weight=2)

            # Lọc sản phẩm hợp lệ bằng cách kiểm tra sự tồn tại trong index model
            # Dùng isin để lọc nhanh hơn vòng lặp list
            mask = user_history['product_name'].isin(recommender.indices.index)
            valid_user_history = user_history[mask]

            if len(valid_user_history) >= 2:
                user_history_df = valid_user_history
                real_user_id = user_id
                break

    if not user_history_df.empty:
        print(f"\n Tìm thấy user: {real_user_id}")
        print(f"   Số sản phẩm hợp lệ trong kho: {len(user_history_df)}")

        # Gợi ý bằng hàm đã tối ưu hóa (Giả sử bạn đã update ContentBasedRecommender)
        recommendations = recommender.recommend_for_user(user_history_df, top_n=10, weight_col='weight')

        if not recommendations.empty:
            recommender.display_recommendations_with_reasons(recommendations, top_n=10)
    else:
        print("⚠️ Không tìm thấy user đủ điều kiện (có ít nhất 2 sản phẩm trong kho model).")

    # 5. Evaluate Model
    print("\n" + "=" * 60)
    print("📊 ĐÁNH GIÁ CONTENT-BASED MODEL")
    print("=" * 60)

    evaluator = Evaluator()
    metrics_to_eval = ['category', 'brand', 'both']

    for metric in metrics_to_eval:
        results = evaluator.evaluate_content_based(
            recommender=recommender,
            product_df=product_df,
            sample_size=200,
            k=10,
            ground_truth_col=metric
        )

        print(f"\n📈 Đánh giá theo {metric.upper()}:")
        print(f"   Precision@10: {results[f'precision@10']:.4f}")
        print(f"   Recall@10:    {results[f'recall@10']:.4f}")
        print(f"   NDCG@10:      {results[f'ndcg@10']:.4f}")
        if metric == 'category':
            print(f"   Coverage:     {results['coverage']:.2%}")

if __name__ == "__main__":
    main()