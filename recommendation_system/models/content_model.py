import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedRecommender:
    def __init__(self):
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.cosine_sim = None
        self.df = None
        self.indices = None

    def train_model(self, df, text_column, title_column):
        print("Đang tính toán độ tương đồng giữa các sản phẩm...")
        self.df = df.copy()
        self.df = self.df.reset_index(drop=True)

        tfidf_matrix = self.tfidf.fit_transform(df[text_column])

        self.cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

        self.indices = pd.Series(self.df.index, index=self.df[title_column]).drop_duplicates()
        print("Đã đọc xong data!")

    def get_recommendation(self, product_name, top_n=5):
        if product_name not in self.indices:
            return f"Không tìm thấy '{product_name}'"

        idx = self.indices[product_name]

        if isinstance(idx, pd.Series):
            idx = idx.iloc[0]

        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:(top_n * 5) + 1]
        product_indices = [i[0] for i in sim_scores]

        result_df = self.df.iloc[product_indices].drop_duplicates(subset=['ProductId'])

        return result_df.head(top_n)