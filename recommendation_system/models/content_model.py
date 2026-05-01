import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

class ContentBasedRecommender:
    def __init__(self):
        self.tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=5000)
        self.cosine_sim = None
        self.df = None
        self.indices = None
        self.is_trained = False

    def train_model(self, df, text_column='combined_text', title_column='product_name'):
        print("Đang tính toán độ tương đồng giữa các sản phẩm...")
        #Copy & reset DataFrame
        self.df = df.copy()
        self.df = self.df.reset_index(drop=True)

        #TF-IDF vector hóa text
        tfidf_matrix = self.tfidf.fit_transform(self.df[text_column])
        print(f"   TF-IDF shape: {tfidf_matrix.shape}")

        #Tính cosine similarity matrix
        self.cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

        self.indices = pd.Series(self.df.index, index=self.df[title_column]).drop_duplicates()
        self.is_trained = True
        print("Hoàn thành train Content-based Model!")

    def get_recommendation(self, product_name, top_n=5):
        #Kiểm tra model đã train
        if not self.is_trained:
            raise ValueError("Model chưa được train!")
        #Kiểm tra sự tồn tại của sản phẩm
        if self.indices is None or product_name not in self.indices:
            return pd.DataFrame(columns=['product_id', 'product_name', 'similarity_score'])

        #Tạo mapping tên sản phẩm -> index
        idx = self.indices[product_name]
        if isinstance(idx, pd.Series):
            idx = idx.iloc[0]

        #Lấy similarity scores từ matrix
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        #Sắp xếp giảm dần
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # TIẾN HÀNH LẤY TOP N PRODUCTS
        recommended_indices = []
        top_scores = []
        seen_product_ids = set()
        # Lấy ID của sản phẩm hiện tại
        current_product_id = self.df.iloc[idx].get('product_id') or self.df.iloc[idx].get('ProductId')
        seen_product_ids.add(current_product_id)

        # Duyệt qua danh sách đã sắp xếp (bỏ qua phần tử đầu tiên vì là chính nó)
        for i, score in sim_scores[1:]:
            # Lọc bỏ sản phẩm trùng ID
            p_id = self.df.iloc[i].get('product_id') or self.df.iloc[i].get('ProductId')

            if p_id not in seen_product_ids:
                recommended_indices.append(i)
                top_scores.append(score)
                seen_product_ids.add(p_id)
            # Trả về products
            if len(recommended_indices) >= top_n:
                break

        result = pd.DataFrame({
            'product_id': self.df.iloc[recommended_indices]['product_id'].values,
            'product_name': self.df.iloc[recommended_indices]['product_name'].values,
            'similarity_score': top_scores
        })

        if 'category' in self.df.columns:
            result['category'] = self.df.iloc[recommended_indices]['category'].values

        if 'brand' in self.df.columns:
            result['brand'] = self.df.iloc[recommended_indices]['brand'].values

        return result

    def recommend_for_user(self, user_history_df, top_n=10, weight_col='weight'):
        if not self.is_trained or user_history_df.empty:
            return pd.DataFrame()

            # 1. Chuyển đổi tên sản phẩm sang Index một lần duy nhất
            # Loại bỏ các sản phẩm không tồn tại trong bộ huấn luyện
        valid_history = user_history_df[user_history_df['product_name'].isin(self.indices.index)]

        if valid_history.empty:
            return pd.DataFrame()

        # Lấy danh sách indices và weights
        user_indices = []
        for name in valid_history['product_name']:
            idx = self.indices[name]
            # Nếu idx là một Series (do trùng tên sản phẩm), lấy giá trị đầu tiên
            if isinstance(idx, pd.Series):
                user_indices.append(idx.iloc[0])
            else:
                user_indices.append(idx)
        weights = valid_history[weight_col].values

        # 2. Truy xuất trực tiếp các hàng từ ma trận Similarity
        # sim_matrix_rows shape: (số lượng sản phẩm đã xem, tổng số sản phẩm trong kho)
        sim_matrix_rows = self.cosine_sim[user_indices]

        # 3. Tính toán Vector tổng hợp bằng phép nhân ma trận (Matrix Multiplication)
        # Nhân trọng số và cộng dồn tất cả
        # weighted_sum shape: (tổng số sản phẩm trong kho,)
        weighted_sum = np.dot(weights, sim_matrix_rows)

        # 4. Sắp xếp Vector kết quả
        # Lấy index của các sản phẩm có điểm cao nhất (từ thấp đến cao)
        sorted_indices = np.argsort(weighted_sum)[::-1]

        # 5. Lọc bỏ sản phẩm đã xem và lấy Top N
        interacted_indices = set(user_indices)
        recommended_indices = []
        final_scores = []
        reasons = []

        for idx in sorted_indices:
            if idx not in interacted_indices:
                recommended_indices.append(idx)
                final_scores.append(weighted_sum[idx])

                match_in_history_idx = np.argmax(sim_matrix_rows[:, idx])
                reasons.append(valid_history.iloc[match_in_history_idx]['product_name'])
            if len(recommended_indices) >= top_n:
                break

        # 6. Tạo DataFrame kết quả
        result = self.df.iloc[recommended_indices].copy()
        result['score'] = final_scores
        result['based_on'] = reasons

        # Chuẩn hóa similarity_score 0-1
        if final_scores:
            max_s = max(final_scores)
            result['similarity_score'] = [s / max_s for s in final_scores]

        return result[['product_id', 'product_name', 'category', 'similarity_score', 'based_on']]

    def recommend_and_display(self, product_name, top_n=5):
        result = self.get_recommendation(product_name, top_n)

        if result.empty:
            print(f"\nCan find the product!: '{product_name}'")
            return result

        #Tìm index của product gốc trong ma trận
        idx = self.indices.get(product_name)
        if idx is not None:
            if isinstance(idx, pd.Series):
                idx = idx.iloc[0]
            original_id = self.df.iloc[idx].get('product_id') or self.df.iloc[idx].get('ProductId')
            original_cat = self.df.iloc[idx].get('category', 'N/A')

            print(f"\n{'=' * 70}")
            print(f"TEST RANDOM PRODUCT: '{product_name}'")
            print(f"   ID: {original_id}")
            print(f"   Tên: {product_name}")
            print(f"   Danh mục: {original_cat}")
        print(f"\n{'=' * 70}")

        print(f"GỢI Ý TOP {top_n} SẢN PHẨM TƯƠNG TỰ")
        print(f"\n{'#':<4} {'Tên sản phẩm':<40} {'Độ tương đồng':<15} {'Danh mục':<15}")
        #Đổ result có từ get_recommended() vào
        for i, row in result.iterrows():
            rank = i + 1
            name = row['product_name']
            if len(name) > 37:
                name = name[:34] + "..."
            similarity = f"{row['similarity_score']:.2%}"
            category = row.get('category', 'N/A')
            if len(category) > 14:
                category = category[:11] + "..."

            print(f"{rank:<4} {name:<40} {similarity:<15} {category:<15}")

        print(f"{'=' * 70}\n")

        return result

    def display_recommendations_with_reasons(self, recommendations, top_n=10):
        """
        Hiển thị gợi ý kèm giải thích (dùng cột based_on đã có)
        """
        if recommendations.empty:
            print("⚠️ Không có gợi ý nào!")
            return

        print(f"\n📋 GỢI Ý TOP {min(top_n, len(recommendations))} SẢN PHẨM (kèm giải thích):")
        print("=" * 80)

        for i in range(min(top_n, len(recommendations))):
            row = recommendations.iloc[i]
            rank = i + 1
            name = row['product_name']
            score = f"{row['similarity_score']:.2%}"
            category = row.get('category', 'N/A')
            based_on = row.get('based_on', 'sở thích của bạn')

            # Giới hạn độ dài based_on
            if len(based_on) > 60:
                based_on = based_on[:57] + "..."

            print(f"\n{rank}. {name}")
            print(f"    Danh mục: {category}")
            print(f"    Độ tương đồng: {score}")
            print(f"    Lý do: {based_on}")

        print("\n" + "=" * 80)

    def save_model(self, path='models/content_model.pkl'):
        import joblib
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'tfidf': self.tfidf,
            'cosine_sim': self.cosine_sim,
            'df': self.df,
            'indices': self.indices,
            'is_trained': self.is_trained
        }, path)
        #print(f"Model saved to {path}")

    def load_model(self, path='models/content_model.pkl'):
        import joblib
        data = joblib.load(path)
        self.tfidf = data['tfidf']
        self.cosine_sim = data['cosine_sim']
        self.df = data['df']
        self.indices = data['indices']
        self.is_trained = data['is_trained']
        #print(f"Model loaded from {path}")