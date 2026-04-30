import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd


class Evaluator:
    def __init__(self):
        pass

    # ========== ĐÁNH GIÁ CONTENT-BASED (Item-Item Similarity) ==========

    def precision_at_k(self, recommended, actual, k=10):
        if not recommended or not actual:
            return 0.0
        recommended_k = recommended[:k]
        hits = len(set(recommended_k) & set(actual))
        return hits / k

    def recall_at_k(self, recommended, actual, k=10):
        if not recommended or not actual:
            return 0.0
        recommended_k = recommended[:k]
        hits = len(set(recommended_k) & set(actual))
        return hits / len(actual)

    def ndcg_at_k(self, recommended, actual, k=10):
        if not recommended or not actual:
            return 0.0

        rel_map = {item: 1 for item in actual}
        dcg = 0.0
        for i, item in enumerate(recommended[:k]):
            rel = rel_map.get(item, 0)
            dcg += rel / np.log2(i + 2)

        ideal = sorted(actual, key=lambda x: 1, reverse=True)[:k]
        idcg = 0.0
        for i, item in enumerate(ideal):
            idcg += 1 / np.log2(i + 2)

        return dcg / idcg if idcg > 0 else 0.0

    def evaluate_content_based(self, recommender, product_df, sample_size=200, k=10,
                               ground_truth_col='category'):
        """
        Đánh giá content-based model trên toàn bộ product_df

        Args:
            recommender: ContentBasedRecommender đã train
            product_df: DataFrame chứa sản phẩm (đã gộp)
            sample_size: Số lượng sản phẩm lấy mẫu để đánh giá
            k: Số lượng gợi ý
            ground_truth_col: Cột dùng làm ground truth ('category', 'brand', hoặc 'both')

        Returns:
            dict: precision@k, recall@k, ndcg@k, coverage
        """
        # Lấy mẫu sản phẩm để đánh giá (tránh chạy quá lâu)
        if sample_size and sample_size < len(product_df):
            sample_products = product_df.sample(n=sample_size, random_state=42)
        else:
            sample_products = product_df

        recommendations_dict = {}
        actual_dict = {}

        # Thống kê
        total_products = len(sample_products)
        no_recommendation_count = 0

        for _, row in sample_products.iterrows():
            product_id = row['product_id']
            product_name = row['product_name']

            # Lấy ground truth theo tiêu chí
            if ground_truth_col == 'category':
                relevant_products = product_df[
                    (product_df['category'] == row['category']) &
                    (product_df['product_id'] != product_id)
                    ]['product_id'].tolist()
            elif ground_truth_col == 'brand':
                relevant_products = product_df[
                    (product_df['brand'] == row['brand']) &
                    (product_df['product_id'] != product_id)
                    ]['product_id'].tolist()
            else:  # both: category AND brand
                relevant_products = product_df[
                    (product_df['category'] == row['category']) &
                    (product_df['brand'] == row['brand']) &
                    (product_df['product_id'] != product_id)
                    ]['product_id'].tolist()

            # Bỏ qua nếu không có ground truth
            if not relevant_products:
                continue

            actual_dict[product_id] = relevant_products

            # Lấy recommendations
            result = recommender.get_recommendation(product_name, top_n=k)

            if result is not None and not result.empty:
                recommendations_dict[product_id] = result['product_id'].tolist()
            else:
                recommendations_dict[product_id] = []
                no_recommendation_count += 1

        # Tính metrics
        precisions = []
        recalls = []
        ndcgs = []

        for product_id, recs in recommendations_dict.items():
            actual = actual_dict.get(product_id, [])
            if not actual:
                continue

            precisions.append(self.precision_at_k(recs, actual, k))
            recalls.append(self.recall_at_k(recs, actual, k))
            ndcgs.append(self.ndcg_at_k(recs, actual, k))

        # Tính coverage (tỷ lệ sản phẩm có recommendations)
        coverage = (total_products - no_recommendation_count) / total_products if total_products > 0 else 0

        return {
            f'precision@{k}': np.mean(precisions) if precisions else 0.0,
            f'recall@{k}': np.mean(recalls) if recalls else 0.0,
            f'ndcg@{k}': np.mean(ndcgs) if ndcgs else 0.0,
            'coverage': coverage,
            'num_samples': len(precisions),
            'total_products': total_products,
            'no_recommendation': no_recommendation_count
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

    # ========== ĐÁNH GIÁ COLLABORATIVE (Dự đoán rating) ==========

    def rmse(self, true_ratings, predicted_ratings):
        """Root Mean Square Error"""
        return mean_squared_error(true_ratings, predicted_ratings, squared=False)

    def mae(self, true_ratings, predicted_ratings):
        """Mean Absolute Error"""
        return np.mean(np.abs(np.array(true_ratings) - np.array(predicted_ratings)))