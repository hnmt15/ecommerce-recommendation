import pandas as pd
import numpy as np


class HybridRecommender:
    """
    Hybrid Recommender kết hợp Content-Based + Collaborative Filtering.

    Logic Switch:
    - User MỚI (cold start, ít tương tác): Chỉ dùng Content-Based
    - User CÓ LỊCH SỬ DÀY: Dùng Hybrid (CB + CF weighted sum)
    """

    def __init__(self, content_recommender, collab_recommender,
                 cb_weight=0.4, cf_weight=0.6,
                 cold_start_threshold=3):
        """
        Args:
            content_recommender: ContentBasedRecommender đã train
            collab_recommender:  CollabRecommender đã load
            cb_weight:           Trọng số Content-Based (0-1)
            cf_weight:           Trọng số Collaborative (0-1), cb + cf = 1
            cold_start_threshold: Số tương tác tối thiểu để dùng Hybrid
                                  (< threshold => cold start => chỉ dùng CB)
        """
        assert abs(cb_weight + cf_weight - 1.0) < 1e-6, "cb_weight + cf_weight phải = 1"

        self.cb = content_recommender
        self.cf = collab_recommender
        self.cb_weight = cb_weight
        self.cf_weight = cf_weight
        self.cold_start_threshold = cold_start_threshold

    # ─────────────────────────────────────────────────────────────────────────
    # NORMALIZATION
    # ─────────────────────────────────────────────────────────────────────────

    def _normalize_min_max(self, scores: dict) -> dict:
        """Min-Max normalize một dict {product_id: score} về [0, 1]"""
        if not scores:
            return scores
        values = np.array(list(scores.values()), dtype=float)
        min_v, max_v = values.min(), values.max()
        if max_v == min_v:
            return {k: 1.0 for k in scores}
        return {k: (v - min_v) / (max_v - min_v) for k, v in scores.items()}

    def _get_cb_scores(self, user_history_df, all_product_ids, top_n=50) -> dict:
        """
        Lấy điểm Content-Based cho user, trả về dict {product_id: cb_score_normalized}
        CB trả similarity_score đã 0-1, nhưng chỉ top_n sản phẩm — ta pad 0 cho phần còn lại
        """
        recs = self.cb.recommend_for_user(
            user_history_df=user_history_df,
            top_n=top_n,
            weight_col='weight'
        )
        if recs.empty:
            return {}

        cb_scores = dict(zip(recs['product_id'], recs['similarity_score']))
        # similarity_score trong content_model đã được normalize 0-1 (max=1)
        return cb_scores

    def _get_cf_scores(self, user_id, all_product_ids) -> dict:
        """
        Lấy điểm CF cho user, trả về dict {product_id: cf_score_normalized}
        CF dùng sigmoid*5 nên range ~0-5, cần normalize về 0-1
        """
        import torch

        if user_id not in self.cf.user_map:
            return {}

        u_idx = self.cf.user_map[user_id]
        u_tensor = torch.LongTensor([u_idx]).repeat(len(self.cf.item_map))
        all_item_indices = torch.LongTensor(list(self.cf.item_map.values()))

        with torch.no_grad():
            predictions = self.cf.model(u_tensor, all_item_indices)

        raw_scores = dict(zip(
            list(self.cf.item_map.keys()),
            predictions.numpy().tolist()
        ))

        # Normalize CF scores (0-5 range) về 0-1
        cf_normalized = self._normalize_min_max(raw_scores)
        return cf_normalized

    # ─────────────────────────────────────────────────────────────────────────
    # COLD START DETECTION
    # ─────────────────────────────────────────────────────────────────────────

    def _is_cold_start(self, user_id, user_history_df) -> bool:
        """
        User bị coi là cold start nếu:
        1. Không tồn tại trong CF model (chưa train), HOẶC
        2. Lịch sử tương tác ít hơn ngưỡng cold_start_threshold
        """
        not_in_cf = (user_id not in self.cf.user_map)
        few_interactions = (len(user_history_df) < self.cold_start_threshold)
        return not_in_cf or few_interactions

    # ─────────────────────────────────────────────────────────────────────────
    # RECOMMEND
    # ─────────────────────────────────────────────────────────────────────────

    def recommend(self, user_id, user_history_df, product_df, top_n=10):
        """
        Hàm gợi ý chính.

        Args:
            user_id:          ID của user
            user_history_df:  DataFrame lịch sử user (product_name, weight, product_id, ...)
            product_df:       DataFrame toàn bộ sản phẩm
            top_n:            Số gợi ý muốn trả về

        Returns:
            DataFrame: [product_id, product_name, category, final_score, mode]
        """
        if user_history_df.empty:
            print(f"⚠️  User {user_id} không có lịch sử — không thể gợi ý.")
            return pd.DataFrame()

        cold_start = self._is_cold_start(user_id, user_history_df)
        all_product_ids = product_df['product_id'].unique().tolist()

        # ── COLD START: Chỉ dùng Content-Based ──────────────────────────────
        if cold_start:
            print(f"❄️  User {user_id}: COLD START ({len(user_history_df)} tương tác) "
                  f"→ Dùng Content-Based Only")

            cb_scores = self._get_cb_scores(user_history_df, all_product_ids, top_n=top_n * 3)
            if not cb_scores:
                return pd.DataFrame()

            scored = pd.DataFrame([
                {'product_id': pid, 'final_score': score}
                for pid, score in cb_scores.items()
            ])
            scored['mode'] = 'content_based'

        # ── HYBRID: CB + CF ─────────────────────────────────────────────────
        else:
            print(f"🔀  User {user_id}: HYBRID ({len(user_history_df)} tương tác) "
                  f"→ CB×{self.cb_weight} + CF×{self.cf_weight}")

            cb_scores = self._get_cb_scores(user_history_df, all_product_ids, top_n=top_n * 5)
            cf_scores = self._get_cf_scores(user_id, all_product_ids)

            # Union tất cả product_id từ cả 2 nguồn
            all_ids = set(cb_scores.keys()) | set(cf_scores.keys())

            hybrid_scores = {}
            for pid in all_ids:
                cb_s = cb_scores.get(pid, 0.0)
                cf_s = cf_scores.get(pid, 0.0)
                hybrid_scores[pid] = self.cb_weight * cb_s + self.cf_weight * cf_s

            scored = pd.DataFrame([
                {'product_id': pid, 'final_score': score}
                for pid, score in hybrid_scores.items()
            ])
            scored['mode'] = 'hybrid'

        # ── Loại bỏ sản phẩm đã tương tác ───────────────────────────────────
        interacted_ids = set(user_history_df['product_id'].values) \
            if 'product_id' in user_history_df.columns else set()
        scored = scored[~scored['product_id'].isin(interacted_ids)]

        # ── Lấy Top N ────────────────────────────────────────────────────────
        scored = scored.sort_values('final_score', ascending=False).head(top_n)

        # ── Gộp thông tin sản phẩm ───────────────────────────────────────────
        product_info = product_df[['product_id', 'product_name', 'category']].drop_duplicates('product_id')
        result = scored.merge(product_info, on='product_id', how='left')

        return result[['product_id', 'product_name', 'category', 'final_score', 'mode']].reset_index(drop=True)

    # ─────────────────────────────────────────────────────────────────────────
    # DISPLAY
    # ─────────────────────────────────────────────────────────────────────────

    def display(self, result: pd.DataFrame):
        if result.empty:
            print("⚠️  Không có gợi ý nào.")
            return

        mode = result['mode'].iloc[0]
        print(f"\n{'=' * 70}")
        print(f"GỢI Ý HYBRID — Mode: {mode.upper()}")
        print(f"{'=' * 70}")
        print(f"{'#':<4} {'Tên sản phẩm':<40} {'Score':<10} {'Danh mục'}")
        print("-" * 70)

        for i, row in result.iterrows():
            name = row['product_name']
            if len(str(name)) > 37:
                name = str(name)[:34] + "..."
            cat = row.get('category', 'N/A')
            print(f"{i + 1:<4} {name:<40} {row['final_score']:.4f}     {cat}")

        print(f"{'=' * 70}\n")

    # ─────────────────────────────────────────────────────────────────────────
    # HỖ TRỢ EVALUATOR (interface chung)
    # ─────────────────────────────────────────────────────────────────────────

    def recommend_ids(self, user_id, user_history_df, product_df, top_n=10) -> list:
        """Trả về list product_id (dùng trong Evaluator)"""
        result = self.recommend(user_id, user_history_df, product_df, top_n=top_n)
        if result.empty:
            return []
        return result['product_id'].tolist()