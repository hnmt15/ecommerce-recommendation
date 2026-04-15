import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    def __init__(self):
        self.tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        self.cosine_sim = None
        self.df = None
        self.indices = None

    def train_model(self, df, text_column='combined_text', title_column='product_name'):
        print("Đang tính toán độ tương đồng giữa các sản phẩm...")
        self.df = df.copy()
        self.df = self.df.reset_index(drop=True)

        tfidf_matrix = self.tfidf.fit_transform(self.df[text_column])

        self.cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

        self.indices = pd.Series(self.df.index, index=self.df[title_column]).drop_duplicates()
        print("Đã train xong Content-based Model!")

    def get_recommendation(self, product_name, top_n=5):
        if self.indices is None or product_name not in self.indices:
            return f"Không tìm thấy '{product_name}'"

        idx = self.indices[product_name]

        if isinstance(idx, pd.Series):
            idx = idx.iloc[0]

        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        recommended_indices = []
        # Lấy ID của sản phẩm hiện tại để không gợi ý lại chính nó
        current_product_id = self.df.iloc[idx].get('product_id') or self.df.iloc[idx].get('ProductId')
        seen_product_ids = {current_product_id}

        # Duyệt qua danh sách đã sắp xếp (bỏ qua phần tử đầu tiên vì là chính nó)
        for i, score in sim_scores[1:]:
            # Check ID để tránh lặp sản phẩm nếu data bị duplicate
            p_id = self.df.iloc[i].get('product_id') or self.df.iloc[i].get('ProductId')

            if p_id not in seen_product_ids:
                recommended_indices.append(i)
                seen_product_ids.add(p_id)

            if len(recommended_indices) >= top_n:
                break

        # Trả về đúng 3 cột mày yêu cầu
        # Chú ý: tên cột trong DataFrame của mày phải khớp (product_id vs ProductId)
        cols_to_return = ['product_id', 'product_name', 'combined_text']

        # Kiểm tra xem cột nào tồn tại trong df của mày để tránh lỗi key
        actual_cols = [c for c in cols_to_return if c in self.df.columns]

        return self.df.iloc[recommended_indices][actual_cols]
