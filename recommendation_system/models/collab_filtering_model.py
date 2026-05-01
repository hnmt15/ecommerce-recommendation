import torch
import torch.nn as nn
import pickle
import pandas as pd


# --- PHẦN 1: ĐỊNH NGHĨA CẤU TRÚC (MODEL) ---
class MatrixFactorization(nn.Module):
    def __init__(self, n_users, n_items, n_factors=30):
        super().__init__()
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.item_factors = nn.Embedding(n_items, n_factors)
        self.user_factors.weight.data.uniform_(0, 0.05)
        self.item_factors.weight.data.uniform_(0, 0.05)

    def forward(self, user, item):
        u = self.user_factors(user)
        i = self.item_factors(item)
        dot_product = (u * i).sum(dim=1)
        return torch.sigmoid(dot_product) * 5


# --- PHẦN 2: LỚP ĐIỀU KHIỂN (RECOMMENDER) ---
class CollabRecommender:
    def __init__(self, model_path='models/collaborative_model.pth', map_path='models/maps.pkl'):
        # 1. Nạp mapping
        with open(map_path, 'rb') as f:
            maps = pickle.load(f)
            self.user_map = maps['user_map']
            self.item_map = maps['item_map']
            self.inv_item_map = {v: k for k, v in self.item_map.items()}

        # 2. Khởi tạo model và nạp trọng số
        n_users = len(self.user_map)
        n_items = len(self.item_map)
        self.model = MatrixFactorization(n_users, n_items, n_factors=30)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def get_detailed_recommendations(self, target_user_id, df, top_n=5):
        self.model.eval()  # Sử dụng model trong self

        # 1. Lấy danh sách sản phẩm User đã mua
        items_already_bought = df[df['user_id'] == target_user_id]['product_id'].unique()

        # 2. Dự đoán (Sử dụng map trong self)
        if target_user_id not in self.user_map:
            print(f"Người dùng {target_user_id} chưa có trong hệ thống.")
            return

        u_idx = self.user_map[target_user_id]
        u_tensor = torch.LongTensor([u_idx]).repeat(len(self.item_map))
        all_item_indices = torch.LongTensor(list(self.item_map.values()))

        with torch.no_grad():
            predictions = self.model(u_tensor, all_item_indices)

        # 3. Tạo DataFrame kết quả
        res_df = pd.DataFrame({
            'product_id': list(self.item_map.keys()),
            'score': predictions.numpy()
        })

        # 4. Loại bỏ sản phẩm đã mua
        res_df = res_df[~res_df['product_id'].isin(items_already_bought)]

        # 5. Lấy Top N và Gộp thông tin
        top_res = res_df.sort_values(by='score', ascending=False).head(top_n)
        product_info = df[['product_id', 'product_name', 'category']].drop_duplicates('product_id')
        final_output = top_res.merge(product_info, on='product_id', how='left')

        # In kết quả
        print(f"=== GỢI Ý CHO USER: {target_user_id} ===")
        for i, row in final_output.iterrows():
            print(f"{i + 1}. {row['product_name']}")
            print(f"   - Danh mục: {row['category']}")
            print(f"   - Độ phù hợp: {row['score']:.2f}/5.0")
            print("-" * 30)

    def recommend(self, target_user_id, top_n=10):

        self.model.eval()

        # Kiểm tra user có tồn tại không
        if target_user_id not in self.user_map:
            return []

        u_idx = self.user_map[target_user_id]
        u_tensor = torch.LongTensor([u_idx]).repeat(len(self.item_map))
        all_item_indices = torch.LongTensor(list(self.item_map.values()))

        with torch.no_grad():
            predictions = self.model(u_tensor, all_item_indices)

        # Tạo DataFrame tạm để lọc
        res_df = pd.DataFrame({
            'product_id': list(self.item_map.keys()),
            'score': predictions.numpy()
        })

        # Lấy Top N ID sản phẩm
        top_ids = res_df.sort_values(by='score', ascending=False).head(top_n)['product_id'].tolist()
        return top_ids

    def predict(self, user_id, item_id):

        #Dự đoán rating cho một cặp user-item

        self.model.eval()

        # Kiểm tra user/item có tồn tại không
        if user_id not in self.user_map or item_id not in self.item_map:
            return 2.5  # Giá trị mặc định

        # Lấy index
        u_idx = self.user_map[user_id]
        i_idx = self.item_map[item_id]

        # Tạo tensor
        u_tensor = torch.LongTensor([u_idx])
        i_tensor = torch.LongTensor([i_idx])

        with torch.no_grad():
            pred = self.model(u_tensor, i_tensor)

        return pred.item()  # Trả về float