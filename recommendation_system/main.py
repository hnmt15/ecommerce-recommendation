import random
import pandas as pd
from utils.data_loader import DataLoader
from utils.preprocess import  Preprocessor
from utils.evaluator import Evaluator
from models.content_model import ContentBasedRecommender
from models.collab_filtering_model import CollabRecommender
import torch
import pickle
from models.hybrid_model import HybridRecommender


def main():
    # 1. Load & Preprocess (Giữ nguyên phần đầu của bạn)
    loader = DataLoader()
    data = loader.load_all_data()
    df = loader.merge_reviews_with_products(data)

    if df is None: return

    preprocessor = Preprocessor()
    clean_df, product_df = preprocessor.preprocess(df, min_user_ratings=3, min_item_ratings=2)
    train_interactions, test_interactions = preprocessor.split_for_collaborative(clean_df)
    preprocessor.save_train_test(train_interactions, test_interactions)

    # 2. Train Content-Based
    recommender = ContentBasedRecommender()
    recommender.train_model(df=product_df)
    recommender.save_model('models/content_model.pkl')

    # 3. Khởi tạo Evaluator
    evaluator = Evaluator()
    print("\n" + "=" * 60)
    print(" BÁO CÁO ĐÁNH GIÁ HỆ THỐNG")
    print("=" * 60)

    # --- PHẦN 1: Đánh giá khả năng hiểu sản phẩm (Item-Item) ---
    # Chỉ đánh giá theo Category để kiểm tra độ chính xác của TF-IDF
    cb_item_results = evaluator.evaluate_content_based(
        recommender=recommender,
        product_df=product_df,
        sample_size=200,
        k=10,
    )

    print(f"\n[1] Khả năng gợi ý sản phẩm tương đương (Item-Item Similarity):")
    print(f"    - Precision@10: {cb_item_results['precision@10']:.4f}")
    print(f"    - Recall@10:    {cb_item_results['recall@10']:.4f}")
    print(f"    - NDCG@10:      {cb_item_results['ndcg@10']:.4f}")
    print(f"    - Coverage:     {cb_item_results['coverage']:.2%}")

    # --- PHẦN 2: Đánh giá khả năng hiểu người dùng (User-centric) ---
    # TEST VỚI 100 USERS TRONG MẪU TEST

    # top_user_ids = loader.get_top_users(data, n=100)  # Lấy mẫu 100 user tích cực
    #
    # user_results = evaluator.evaluate_multiple_users(
    #     recommender=recommender,
    #     data_loader=loader,
    #     data=data,
    #     user_ids=top_user_ids,
    #     product_df=product_df,
    #     k=10
    # )
    #
    # print(f"\n[2] Khả năng dự đoán hành vi người dùng (User-centric Evaluation):")
    # print(f"    - Số lượng User thử nghiệm: {user_results['Total Users Evaluated']}")
    # print(f"    - Avg Precision@10: {user_results['Avg Precision@10']:.2%}")
    # print(f"    - Avg Recall@10:    {user_results['Avg Recall@10']:.2%}")
    #
    # print("-" * 60)



    # 4.GỢI Ý CONTENT_BASED CHO MỘT USER CỤ THỂ
    # user_history = loader.get_user_history(data, user_id="U003023")
    # if user_history is not None and not user_history.empty:
    #     # 2. Lọc bỏ các sản phẩm không có trong kho của Content-Model
    #     mask = user_history['product_name'].isin(recommender.indices.index)
    #     valid_history = user_history[mask]
    #
    #     if not valid_history.empty:
    #         # 3. Gọi hàm gợi ý đã tối ưu
    #         recommendations = recommender.recommend_for_user(
    #             valid_history,
    #             top_n=10,
    #             weight_col='weight'
    #         )
    #
    #         if not recommendations.empty:
    #             print(f" Tìm thấy {len(valid_history)} sản phẩm hợp lệ trong lịch sử.")
    #             recommender.display_recommendations_with_reasons(recommendations, top_n=10)
    #         else:
    #             print(" Không thể tạo gợi ý (có thể do dữ liệu quá thưa).")
    #     else:
    #         print("⚠ Sản phẩm trong lịch sử User này không tồn tại trong kho huấn luyện.")
    # else:
    #     print(f"!")


    # 5. GỢI Ý VÀ CHO COLLAB
    # recommender = CollabRecommender()
    # recommender.get_detailed_recommendations("U003669", clean_df, 5)
    #
    # evaluator = Evaluator()
    # results = evaluator.evaluate_collaborative(
    #     model= recommender,
    #     test_df=test_interactions,
    #     top_n=10
    # )
    #
    # print(f"\n KẾT QUẢ TỔNG HỢP:")
    # print(f"   RMSE: {results['rmse']:.4f}")
    # print(f"   Hit Rate@10: {results['hit_rate']:.2f}%")
    # print(f"   Precision@10: {results['precision']:.4f}")
    # print(f"   Recall@10: {results['recall']:.4f}")

    # 5. HYBRID + ĐÁNH GIÁ SO SÁNH
    cf_recommender = CollabRecommender()

    cf_results = evaluator.evaluate_collaborative(
        model=cf_recommender,
        test_df=test_interactions,
        top_n=10
    )
    # Khởi tạo HybridRecommender
    hybrid = HybridRecommender(
        content_recommender=recommender,
        collab_recommender=cf_recommender,
        cb_weight=0.4,
        cf_weight=0.6,
        cold_start_threshold=3
    )
    # Đánh giá Hybrid trên 100 user mẫu từ tập test
    sample_users = test_interactions['user_id'].unique()[:100].tolist()

    hybrid_results = evaluator.evaluate_hybrid(
        hybrid_recommender=hybrid,
        data_loader=loader,
        data=data,
        product_df=product_df,
        test_df=test_interactions,
        user_ids=sample_users,
        k=10
    )
    # In bảng so sánh 3 model ra console
    evaluator.print_comparison_table(cb_item_results, cf_results, hybrid_results, k=10)

    # Vẽ biểu đồ so sánh và lưu vào models/evaluation_comparison.png
    evaluator.plot_comparison(
        cb_results=cb_item_results,
        cf_results=cf_results,
        hybrid_results=hybrid_results,
        k=10,
        save_path='models/evaluation_comparison.png'
    )

if __name__ == "__main__":
    main()