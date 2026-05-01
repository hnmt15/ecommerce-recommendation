import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd


class Evaluator:
    def __init__(self):
        pass

    def precision_recall_ndcg(self, recommended, actual, k=10):
        """Tính toán bộ 3 chỉ số cơ bản trong một lần gọi"""
        if not recommended or not actual:
            return 0.0, 0.0, 0.0

        rec_k = recommended[:k]
        actual_set = set(actual)
        hits = len(set(rec_k) & actual_set)

        precision = hits / k
        recall = hits / len(actual)

        # Tính NDCG
        dcg = sum([1 / np.log2(i + 2) for i, item in enumerate(rec_k) if item in actual_set])
        idcg = sum([1 / np.log2(i + 2) for i in range(min(len(actual), k))])
        ndcg = dcg / idcg if idcg > 0 else 0.0

        return precision, recall, ndcg

    def evaluate_content_based(self, recommender, product_df, sample_size=100, k=10):
        """Đánh giá nhanh gọn theo Category"""
        sample = product_df.sample(n=min(sample_size, len(product_df)), random_state=42)
        results = []

        for _, row in sample.iterrows():
            # Ground truth: Các sản phẩm cùng danh mục
            actual = product_df[(product_df['category'] == row['category']) &
                                (product_df['product_id'] != row['product_id'])]['product_id'].tolist()

            if not actual: continue

            # Lấy gợi ý từ model
            rec_result = recommender.get_recommendation(row['product_name'], top_n=k)
            recommended = rec_result['product_id'].tolist() if not rec_result.empty else []

            results.append(self.precision_recall_ndcg(recommended, actual, k))

        # Tính trung bình
        avg_metrics = np.mean(results, axis=0)
        return {
            'precision@10': avg_metrics[0],
            'recall@10': avg_metrics[1],
            'ndcg@10': avg_metrics[2],
            'coverage': len(results) / sample_size
        }

    def evaluate_content_based_for_user(self, recommender, user_history_df,
                                        product_df, k=10,
                                        ground_truth_col='category'):
        """
        Đánh giá gợi ý cho user (dùng hold-out: giấu 1 sản phẩm)

        Args:
            recommender: ContentBasedRecommender đã train
            user_history_df: Lịch sử user (có 'product_name', 'weight')
            product_df: DataFrame sản phẩm
            k: Số lượng gợi ý
            ground_truth_col: Cột dùng làm ground truth

        Returns:
            dict: precision@k, recall@k
        """
        if len(user_history_df) < 2:
            return {'precision@k': 0.0, 'recall@k': 0.0, 'message': 'Not enough history'}

        # Giấu 1 sản phẩm cuối cùng để làm ground truth
        test_product = user_history_df.iloc[-1]
        train_history = user_history_df.iloc[:-1]

        # Gợi ý dựa trên train_history
        recommendations = recommender.recommend_for_user(
            user_history_df=train_history[['product_name', 'weight']],
            top_n=k,
            weight_col='weight'
        )

        if recommendations.empty:
            return {'precision@k': 0.0, 'recall@k': 0.0, 'message': 'No recommendations'}

        # Ground truth: sản phẩm thực tế user đã tương tác (test_product)
        actual_product = test_product['product_name']

        # Hoặc ground truth mở rộng: các sản phẩm cùng category/brand
        if ground_truth_col == 'category' and 'category' in test_product:
            actual_products = product_df[
                (product_df['category'] == test_product['category']) &
                (product_df['product_name'] != actual_product)
                ]['product_name'].tolist()
            actual_products.append(actual_product)
        else:
            actual_products = [actual_product]

        recommended_names = recommendations['product_name'].tolist()

        # Tính precision/recall
        hits = len(set(recommended_names[:k]) & set(actual_products))

        return {
            f'precision@{k}': hits / k,
            f'recall@{k}': hits / len(actual_products) if actual_products else 0,
            'recommended': recommended_names[:5],
            'actual': actual_product
        }

    def evaluate_multiple_users(self, recommender, data_loader, data, user_ids, product_df, k=10):
        """
        Đánh giá Content-Based trên danh sách User để lấy chỉ số trung bình.
        """
        all_results = []

        for user_id in user_ids:
            # Lấy lịch sử thực tế của User
            user_history = data_loader.get_user_history(data, user_id=user_id)

            # Chỉ đánh giá những User có đủ lịch sử (>= 2 món)
            if len(user_history) >= 2:
                res = self.evaluate_content_based_for_user(
                    recommender, user_history, product_df, k=k
                )
                if 'message' not in res:
                    all_results.append([res[f'precision@{k}'], res[f'recall@{k}']])

        if not all_results:
            return {"Precision": 0, "Recall": 0}

        # Tính trung bình cộng của tất cả User đã đánh giá
        avg_metrics = np.mean(all_results, axis=0)
        return {
            f"Avg Precision@{k}": avg_metrics[0],
            f"Avg Recall@{k}": avg_metrics[1],
            "Total Users Evaluated": len(all_results)
        }







    def rmse(self, true_ratings, predicted_ratings):
        """Dự đoán rating có chính xác không?"""
        return mean_squared_error(true_ratings, predicted_ratings, squared=False)

    def mae(self, true_ratings, predicted_ratings):
        """Sai số tuyệt đối trung bình"""
        return np.mean(np.abs(np.array(true_ratings) - np.array(predicted_ratings)))

    def evaluate_collaborative(self, model, test_df, top_n=10):
        """
        Đánh giá TOÀN DIỆN collaborative model
        - RMSE/MAE: Dự đoán rating có chính xác không?
        - Hit Rate/Precision/Recall: Gợi ý có đúng không?
        """


        # ========== 1. ĐÁNH GIÁ DỰ ĐOÁN RATING (RMSE/MAE) ==========
        predictions = []
        true_ratings = []

        for _, row in test_df.iterrows():
            user_id = row['user_id']
            item_id = row['product_id']
            true_rating = row['rating']

            pred_rating = model.predict(user_id, item_id)

            predictions.append(pred_rating)
            true_ratings.append(true_rating)

        rmse = np.sqrt(mean_squared_error(true_ratings, predictions))
        mae = np.mean(np.abs(np.array(true_ratings) - np.array(predictions)))

        # ========== 2. ĐÁNH GIÁ GỢI Ý (Hit Rate, Precision, Recall) ==========
        test_users = test_df['user_id'].unique()

        if hasattr(model, 'user_map'):
            test_users = [u for u in test_users if u in model.user_map]

        hits = 0
        precisions = []
        recalls = []

        for user_id in test_users:
            # Sản phẩm thực tế user đã mua
            actual_items = set(test_df[test_df['user_id'] == user_id]['product_id'].values)

            if not actual_items:
                continue

            # Gợi ý từ model
            predicted_items = model.recommend(user_id, top_n=top_n)

            # Hit Rate: có ít nhất 1 đúng không?
            if any(item in actual_items for item in predicted_items):
                hits += 1

            # Precision/Recall
            hits_count = len(set(predicted_items) & actual_items)
            precisions.append(hits_count / top_n)
            recalls.append(hits_count / len(actual_items))

        hit_rate = (hits / len(test_users)) * 100 if test_users else 0

        print("\n" + "=" * 60)
        print(" COLLABORATIVE FILTERING EVALUATION")
        print("=" * 60)
        print(f"\n DỰ ĐOÁN RATING:")
        print(f"   RMSE: {rmse:.4f} (càng nhỏ càng tốt, 0 là hoàn hảo)")
        print(f"   MAE: {mae:.4f} (sai số trung bình ~{mae:.2f} sao)")

        print(f"\n GỢI Ý SẢN PHẨM (top-{top_n}):")
        print(f"   Hit Rate: {hit_rate:.2f}% ({hits}/{len(test_users)} users)")
        print(f"   Precision@{top_n}: {np.mean(precisions):.4f}")
        print(f"   Recall@{top_n}: {np.mean(recalls):.4f}")

        return {
            'rmse': rmse,
            'mae': mae,
            'hit_rate': hit_rate,
            'precision': np.mean(precisions) if precisions else 0,
            'recall': np.mean(recalls) if recalls else 0,
            'total_users': len(test_users),
            'hits': hits
        }
