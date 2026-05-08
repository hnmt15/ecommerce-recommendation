import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')


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

    def evaluate_hybrid(self, hybrid_recommender, data_loader, data,
                        product_df, test_df, user_ids=None, k=10):
        """
        Đánh giá HybridRecommender trên danh sách user.

        Ground truth dùng CÙNG CÁCH với CB (theo category) để so sánh công bằng:
        - Giấu 1 sản phẩm cuối trong lịch sử user (hold-out)
        - Ground truth = tất cả sản phẩm cùng category với sản phẩm bị giấu
        - Kiểm tra xem Hybrid có gợi ý đúng category không

        Args:
            hybrid_recommender: HybridRecommender đã khởi tạo
            data_loader:        DataLoader để lấy lịch sử user
            data:               dict data từ load_all_data()
            product_df:         DataFrame sản phẩm (có product_id, category)
            test_df:            DataFrame test (user_id, product_id, rating)
            user_ids:           Danh sách user cần đánh giá (None = lấy từ test_df)
            k:                  Top-K

        Returns:
            dict: precision@k, recall@k, ndcg@k, hit_rate, total_users
        """
        if user_ids is None:
            user_ids = test_df['user_id'].unique().tolist()

        precisions, recalls, ndcgs = [], [], []
        hits = 0
        evaluated = 0

        print(f"\n⏳ Đánh giá Hybrid trên {len(user_ids)} users (k={k})...")

        for user_id in user_ids:
            # Lấy toàn bộ lịch sử user (train + events)
            user_history = data_loader.get_user_history(data, user_id=user_id)
            if user_history.empty or len(user_history) < 2:
                continue

            # Hold-out: giấu sản phẩm cuối cùng làm ground truth
            # (giống cách CB đánh giá để so sánh công bằng)
            test_item_row = user_history.iloc[-1]
            train_history = user_history.iloc[:-1]

            # Lấy category của sản phẩm bị giấu
            test_product_id = test_item_row.get('product_id')
            test_category = test_item_row.get('category', None)

            if not test_category or not test_product_id:
                continue

            # Ground truth = tất cả sản phẩm cùng category (trừ chính nó)
            # Đây là cách CB dùng => so sánh công bằng
            actual_items = product_df[
                (product_df['category'] == test_category) &
                (product_df['product_id'] != test_product_id)
                ]['product_id'].tolist()

            # Thêm chính sản phẩm bị giấu vào ground truth
            actual_items.append(test_product_id)

            if not actual_items:
                continue

            # Gợi ý từ hybrid dựa trên train_history (không có sản phẩm bị giấu)
            recommended_ids = hybrid_recommender.recommend_ids(
                user_id=user_id,
                user_history_df=train_history,
                product_df=product_df,
                top_n=k
            )

            if not recommended_ids:
                continue

            # Tính chỉ số
            p, r, ndcg = self.precision_recall_ndcg(recommended_ids, actual_items, k)
            precisions.append(p)
            recalls.append(r)
            ndcgs.append(ndcg)

            if any(item in set(actual_items) for item in recommended_ids):
                hits += 1

            evaluated += 1

        if evaluated == 0:
            print("⚠️  Không có user nào được đánh giá!")
            return {f'precision@{k}': 0, f'recall@{k}': 0,
                    f'ndcg@{k}': 0, 'hit_rate': 0, 'total_users': 0}

        hit_rate = (hits / evaluated) * 100

        result = {
            f'precision@{k}': np.mean(precisions),
            f'recall@{k}': np.mean(recalls),
            f'ndcg@{k}': np.mean(ndcgs),
            'hit_rate': hit_rate,
            'total_users': evaluated
        }

        print(f"\n{'=' * 60}")
        print(f" HYBRID RECOMMENDER EVALUATION")
        print(f"{'=' * 60}")
        print(f"   Users đánh giá:  {evaluated}")
        print(f"   Precision@{k}:    {result[f'precision@{k}']:.4f}")
        print(f"   Recall@{k}:       {result[f'recall@{k}']:.4f}")
        print(f"   NDCG@{k}:         {result[f'ndcg@{k}']:.4f}")
        print(f"   Hit Rate:        {hit_rate:.2f}%")

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # VẼ BIỂU ĐỒ SO SÁNH (MỚI)
    # ─────────────────────────────────────────────────────────────────────────

    def plot_comparison(self, cb_results, cf_results, hybrid_results,
                        k=10, save_path='models/evaluation_comparison.png'):
        """
        Vẽ biểu đồ so sánh 3 model: CB vs CF vs Hybrid
        Dùng 2 loại biểu đồ thầy dạy:
          - Subplot 1: Bar chart (cột nhóm) — so sánh từng chỉ số
          - Subplot 2: Line chart (đường)   — thấy xu hướng tổng thể

        Args:
            cb_results:     dict từ evaluate_content_based()
            cf_results:     dict từ evaluate_collaborative()
            hybrid_results: dict từ evaluate_hybrid()
            k:              giá trị k dùng trong evaluation
            save_path:      đường dẫn lưu ảnh
        """
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # ── Chuẩn bị dữ liệu ────────────────────────────────────────────────
        def get(d, *keys, default=0.0):
            for key in keys:
                if key in d:
                    return d[key]
            return default

        # 3 chỉ số chính để so sánh, đưa về cùng thang 0-1
        metrics = [f'Precision@{k}', f'Recall@{k}', 'Hit Rate']

        cb_vals = [
            get(cb_results, f'precision@{k}', 'precision'),
            get(cb_results, f'recall@{k}', 'recall'),
            get(cb_results, 'hit_rate', default=0.0) / 100,
        ]
        cf_vals = [
            get(cf_results, 'precision', f'precision@{k}'),
            get(cf_results, 'recall', f'recall@{k}'),
            get(cf_results, 'hit_rate') / 100,
        ]
        hybrid_vals = [
            get(hybrid_results, f'precision@{k}', 'precision'),
            get(hybrid_results, f'recall@{k}', 'recall'),
            get(hybrid_results, 'hit_rate') / 100,
        ]

        # ── Vẽ 2 biểu đồ trong 1 figure ─────────────────────────────────────
        fig, bieudo = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(f'So Sánh Hiệu Suất 3 Model (k={k})',
                     fontsize=13, fontweight='bold')

        # ── SUBPLOT 1: BIỂU ĐỒ CỘT (BAR CHART) ─────────────────────────────
        # Vẽ nhiều cột trên cùng biểu đồ để so sánh (giống ví dụ thầy dạy)
        X_axis = np.arange(len(metrics))
        width = 0.25

        bieudo[0].bar(X_axis - width, cb_vals, width=width, label='Content-Based', color='b', alpha=0.8)
        bieudo[0].bar(X_axis, cf_vals, width=width, label='Collaborative', color='r', alpha=0.8)
        bieudo[0].bar(X_axis + width, hybrid_vals, width=width, label='Hybrid', color='g', alpha=0.8)

        bieudo[0].set_xticks(X_axis)
        bieudo[0].set_xticklabels(metrics)
        bieudo[0].set_xlabel('Chỉ số đánh giá')
        bieudo[0].set_ylabel('Giá trị (0-1)')
        bieudo[0].set_title('Biểu đồ cột so sánh 3 Model')
        bieudo[0].legend()
        bieudo[0].set_ylim(0, max(max(cb_vals), max(cf_vals), max(hybrid_vals)) * 1.3 + 0.01)

        # Ghi số lên đầu mỗi cột cho dễ đọc
        for bars, vals in zip(
                [X_axis - width, X_axis, X_axis + width],
                [cb_vals, cf_vals, hybrid_vals]
        ):
            for x_pos, val in zip(bars, vals):
                bieudo[0].text(x_pos, val + 0.005, f'{val:.3f}',
                               ha='center', va='bottom', fontsize=7.5)

        # ── SUBPLOT 2: BIỂU ĐỒ ĐƯỜNG (LINE CHART) ──────────────────────────
        # Thấy được xu hướng model nào ổn định hơn qua các chỉ số
        bieudo[1].plot(metrics, cb_vals,
                       color='b', marker='o', linestyle='-',
                       linewidth=2, markersize=8, label='Content-Based')
        bieudo[1].plot(metrics, cf_vals,
                       color='r', marker='s', linestyle='--',
                       linewidth=2, markersize=8, label='Collaborative')
        bieudo[1].plot(metrics, hybrid_vals,
                       color='g', marker='*', linestyle='-.',
                       linewidth=2, markersize=10, label='Hybrid')

        bieudo[1].set_xlabel('Chỉ số đánh giá')
        bieudo[1].set_ylabel('Giá trị (0-1)')
        bieudo[1].set_title('Biểu đồ đường xu hướng 3 Model')
        bieudo[1].legend()
        bieudo[1].set_ylim(0, max(max(cb_vals), max(cf_vals), max(hybrid_vals)) * 1.3 + 0.01)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ Biểu đồ so sánh đã lưu tại: {save_path}")
        plt.close()

        return save_path

    def print_comparison_table(self, cb_results, cf_results, hybrid_results, k=10):
        """In bảng so sánh gọn gàng ra console"""

        def get(d, *keys, default=0.0):
            for key in keys:
                if key in d:
                    return d[key]
            return default

        print("\n" + "=" * 65)
        print(f"  BẢNG SO SÁNH 3 MODEL (k={k})")
        print("=" * 65)
        print(f"{'Chỉ số':<20} {'Content-Based':>15} {'Collaborative':>15} {'Hybrid':>12}")
        print("-" * 65)

        rows = [
            (f'Precision@{k}',
             get(cb_results, f'precision@{k}', 'precision'),
             get(cf_results, 'precision', f'precision@{k}'),
             get(hybrid_results, f'precision@{k}', 'precision')),
            (f'Recall@{k}',
             get(cb_results, f'recall@{k}', 'recall'),
             get(cf_results, 'recall', f'recall@{k}'),
             get(hybrid_results, f'recall@{k}', 'recall')),
            (f'NDCG@{k}',
             get(cb_results, f'ndcg@{k}', 'ndcg'),
             get(cf_results, f'ndcg@{k}', 'ndcg'),
             get(hybrid_results, f'ndcg@{k}', 'ndcg')),
            ('Hit Rate (%)',
             get(cb_results, 'hit_rate'),
             get(cf_results, 'hit_rate'),
             get(hybrid_results, 'hit_rate')),
        ]

        for name, cb_v, cf_v, hy_v in rows:
            # Đánh dấu giá trị tốt nhất
            best = max(cb_v, cf_v, hy_v)
            cb_str = f"{cb_v:.4f}" + (" ✓" if cb_v == best else "  ")
            cf_str = f"{cf_v:.4f}" + (" ✓" if cf_v == best else "  ")
            hy_str = f"{hy_v:.4f}" + (" ✓" if hy_v == best else "  ")
            print(f"{name:<20} {cb_str:>15} {cf_str:>15} {hy_str:>12}")

        print("=" * 65)
        print("  ✓ = Model tốt nhất ở chỉ số đó")
        print("=" * 65)
