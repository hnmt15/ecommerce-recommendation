import os
import pandas as pd
from pathlib import Path


class DataLoader:
    def __init__(self):
        current_dir = Path(__file__).parent
        self.project_root = current_dir.parent
        self.data_dir = self.project_root / 'data' / 'raw'
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load_all_data(self):
        files = {
            'users': self.data_dir / 'users.csv',
            'products': self.data_dir / 'products.csv',
            'reviews': self.data_dir / 'reviews.csv',
            'orders': self.data_dir / 'orders.csv',
            'order_items': self.data_dir / 'order_items.csv',
            'events': self.data_dir / 'events.csv'
        }

        data = {}
        for name, path in files.items():
            if path.exists():
                data[name] = pd.read_csv(path)
                print(f" Đã đọc {name}.csv: {len(data[name])} dòng")
            else:
                print(f" Không tìm thấy: {path}")
                data[name] = None

        return data

    def merge_reviews_with_products(self, data):
        """Join reviews với products để lấy tên sản phẩm"""
        if data['reviews'] is None or data['products'] is None:
            print("Thiếu file reviews hoặc products")
            return None

        # Xác định tên cột product_id
        reviews = data['reviews']
        products = data['products']

        # Kiểm tra tên cột product_id
        product_id_col_reviews = 'product_id' if 'product_id' in reviews.columns else 'parent_asin'
        product_id_col_products = 'product_id' if 'product_id' in products.columns else 'parent_asin'

        # Chọn các cột cần từ products
        product_cols = [product_id_col_products]
        if 'product_name' in products.columns:
            product_cols.append('product_name')
        if 'category' in products.columns:
            product_cols.append('category')
        if 'price' in products.columns:
            product_cols.append('price')
        if 'brand' in products.columns:
            product_cols.append('brand')

        merged = reviews.merge(
            products[product_cols],
            left_on=product_id_col_reviews,
            right_on=product_id_col_products,
            how='inner'
        )

        # Xử lý users
        if data['users'] is not None:
            users = data['users']

            user_cols = ['user_id']

            optional_cols = ['name', 'email', 'gender', 'city', 'signup_date']
            for col in optional_cols:
                if col in users.columns:
                    user_cols.append(col)

            # Thực hiện merge
            try:
                merged = merged.merge(
                    users[user_cols],
                    on='user_id',
                    how='left'
                )
                print(f" Đã thêm thông tin user ({len(user_cols) - 1} trường)")
            except Exception as e:
                print(f" Lỗi khi merge user info: {e}")

        return merged

    def load_and_preview(self, sample_size=None):
        # Đọc tất cả file
        data = self.load_all_data()

        # Merge reviews với products
        df = self.merge_reviews_with_products(data)

        if df is None:
            return None

        # Giới hạn sample size
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
            print(f"\n Lấy mẫu {sample_size} dòng")


        # print(f"\n" + "=" * 50)
        # print("THÔNG TIN DATAFRAME SAU MERGE")
        # print("=" * 50)
        # print(f"Các cột: {df.columns.tolist()}")
        # print(f"Số dòng: {len(df):,}")
        # print(f"Users: {df['user_id'].nunique():,}")
        # print(f"Products: {df['product_id'].nunique():,}")
        # print(f"Rating range: {df['rating'].min()} - {df['rating'].max()}")

        return df

    def get_statistics(self, df):
        if df is None:
            return

        print("\n" + "=" * 50)
        print("CHI TIẾT")
        print("=" * 50)

        # Phân phối rating
        # print("\n📊 Phân phối Rating:")
        # rating_dist = df['rating'].value_counts().sort_index()
        # for rating, count in rating_dist.items():
        #     print(f"   {rating}⭐: {count:>8,} ({count / len(df) * 100:>5.1f}%)")
        #
        # # Top sản phẩm
        # print("\n🏆 Top 10 sản phẩm được đánh giá nhiều nhất:")
        # top_products = df['product_name'].value_counts().head(10)
        # for idx, (name, count) in enumerate(top_products.items(), 1):
        #     print(f"   {idx}. {name[:50]}: {count} reviews")
        #
        # # Top user
        # print("\n👥 Top 10 user đánh giá nhiều nhất:")
        # top_users = df['user_id'].value_counts().head(10)
        # for idx, (user, count) in enumerate(top_users.items(), 1):
        #     print(f"   {idx}. User {user}: {count} reviews")

        # Độ thưa (sparsity)
        n_users = df['user_id'].nunique()
        n_products = df['product_id'].nunique()
        n_interactions = len(df)
        sparsity = (1 - n_interactions / (n_users * n_products)) * 100
        print(f"\n Độ thưa của ma trận User-Item: {sparsity:.2f}%")



# Test nhanh
if __name__ == "__main__":
    loader = DataLoader()
    df = loader.load_and_preview(sample_size=10000)

    if df is not None:
        loader.get_statistics(df)