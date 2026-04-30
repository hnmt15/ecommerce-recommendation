import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


class Preprocessor:
    def __init__(self):
        current_dir = Path(__file__).parent
        self.project_root = current_dir.parent
        self.processed_dir = self.project_root / 'data' / 'processed'
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def preprocess(self, df, min_user_ratings=3, min_item_ratings=3,
                   rating_col='rating', user_col='user_id', item_col='product_id'):
        """
        df: DataFrame từ DataLoader (đã merge reviews + products)

        Returns:
            clean_df: DataFrame dạng user-item interactions (giữ nguyên user info)
            product_df: DataFrame dạng product features (đã gộp, cho content-based)
        """
        if df is None:
            return None, None

        # Tạo combined_text từ name, cate, brand
        text_parts = []

        if 'brand' in df.columns:
            # Chỉ thêm brand nếu nó khác tên sản phẩm
            brand_series = df['brand'].fillna('')
            product_name_series = df['product_name'].fillna('')

            # Kiểm tra: nếu brand khác product_name thì mới thêm
            brand_to_add = brand_series.copy()
            mask = brand_series == product_name_series
            brand_to_add[mask] = ''
            text_parts.append(brand_to_add)

        if 'product_name' in df.columns:
            text_parts.append(df['product_name'].fillna(''))

        if 'category' in df.columns:
            text_parts.append(df['category'].fillna(''))

        if text_parts:
            df['combined_text'] = text_parts[0]
            for part in text_parts[1:]:
                df['combined_text'] = df['combined_text'] + " " + part
        else:
            df['combined_text'] = ""

        # Loại cột null trong bảng reviews
        initial_len = len(df)
        df = df.dropna(subset=[user_col, item_col, rating_col])
        print(f"\n Loại bỏ null:")
        print(f"   Mất {initial_len - len(df)} dòng")

        # Loại duplicate
        initial_len = len(df)
        df = df.drop_duplicates(subset=[user_col, item_col])
        print(f"\n Loại bỏ duplicate (user+product):")
        print(f"   Mất {initial_len - len(df)} dòng")

        # Lọc user có ít nhất 3 ratings
        user_counts = df[user_col].value_counts()
        valid_users = user_counts[user_counts >= min_user_ratings].index
        initial_len = len(df)
        df = df[df[user_col].isin(valid_users)]
        print(f"\n Lọc users (ít nhất {min_user_ratings} ratings):")
        print(f"   Giữ lại {len(valid_users):,} users")
        print(f"   Mất {initial_len - len(df)} dòng")

        # Lọc sp có ít nhất 2 ratings
        item_counts = df[item_col].value_counts()
        valid_items = item_counts[item_counts >= min_item_ratings].index
        initial_len = len(df)
        df = df[df[item_col].isin(valid_items)]
        print(f"\n Lọc items (ít nhất {min_item_ratings} ratings):")
        print(f"   Giữ lại {len(valid_items):,} items")
        print(f"   Mất {initial_len - len(df)} dòng")

        # Các model ML (SVD, ALS, Neural Network) chỉ làm việc với số, không với string
        #nên chuẩn hóa id thành dạng category: gán stt cho mỗi giá trị duy nhất
        df['user_idx'] = df[user_col].astype('category').cat.codes
        df['item_idx'] = df[item_col].astype('category').cat.codes

        if rating_col != 'rating':
            df['rating'] = df[rating_col]
        else:
            df['rating'] = df[rating_col]

        # THỐNG KÊ
        n_users = df['user_idx'].nunique()
        n_items = df['item_idx'].nunique()
        n_interactions = len(df)
        sparsity = (1 - n_interactions / (n_users * n_items)) * 100

        print(f"\n" + "=" * 50)
        print("AFTER PREPROCESS (Interactions):")
        print("=" * 50)
        print(f"   Users: {n_users:,}")
        print(f"   Products: {n_items:,}")
        print(f"   Interactions: {n_interactions:,}")
        print(f"   Độ thưa: {sparsity:.2f}%")
        print(f"   Rating TB: {df['rating'].mean():.2f}")

        # Giữ lại df_clean cho collab
        clean_df = df.copy()

        # Chỉnh df mới cho model content_based
        product_df = self.aggregate_products(df)

        return clean_df, product_df

    def aggregate_products(self, df):
        #Gộp dữ liệu cho content_based: có nhiều user review sp -> sp xuất hiện nhiều lần -> gộp combined_text lại, cho sp chỉ xuất hiện 1 lần để tạo 1 vector duy nhất
        product_df = df.groupby('product_id').agg({
            'product_name': 'first',
            'brand': 'first',
            'category': 'first',
            'combined_text': 'first',
            'rating': 'mean',
        }).reset_index()

        print(f"\n   Gộp {len(df)} interactions → {len(product_df)} sản phẩm (cho content-based)")
        return product_df


    def split_for_collaborative(self, df, test_size=0.2, random_state=42):
        unique_users = df['user_idx'].unique()
        train_users, test_users = train_test_split(
            unique_users, test_size=test_size, random_state=random_state
        )

        train = df[df['user_idx'].isin(train_users)]
        test = df[df['user_idx'].isin(test_users)]

        print(f"\n   Collaborative split:")
        print(f"   Train users: {len(train_users):,}")
        print(f"   Test users: {len(test_users):,}")
        print(f"   Train interactions: {len(train):,}")
        print(f"   Test interactions: {len(test):,}")

        return train, test


    def save_processed(self, clean_df, product_df,
                       clean_filename='ecommerce_clean.csv',
                       product_filename='ecommerce_products.csv'):
        """Lưu cả 2 loại dữ liệu"""
        if clean_df is not None:
            clean_path = self.processed_dir / clean_filename
            clean_df.to_csv(clean_path, index=False)
            print(f"💾 Saved clean interactions: {clean_path}")

        if product_df is not None:
            product_path = self.processed_dir / product_filename
            product_df.to_csv(product_path, index=False)
            print(f"💾 Saved product features: {product_path}")

    def save_train_test(self, train, test, prefix='ecommerce'):
        if train is not None:
            train_path = self.processed_dir / f'{prefix}_train.csv'
            train.to_csv(train_path, index=False)
            print(f"💾 Saved: {train_path}")

        if test is not None:
            test_path = self.processed_dir / f'{prefix}_test.csv'
            test.to_csv(test_path, index=False)
            print(f"💾 Saved: {test_path}")

    def load_clean_data(self, filename='ecommerce_clean.csv'):
        """Load clean interactions data (cho collaborative)"""
        load_path = self.processed_dir / filename
        if not load_path.exists():
            print(f" Không tìm thấy file: {load_path}")
            return None
        df = pd.read_csv(load_path)
        print(f" Đã load {len(df):,} interactions từ {load_path}")
        return df

    def load_product_data(self, filename='ecommerce_products.csv'):
        """Load product features data (cho content-based)"""
        load_path = self.processed_dir / filename
        if not load_path.exists():
            print(f" Không tìm thấy file: {load_path}")
            return None
        df = pd.read_csv(load_path)
        print(f" Đã load {len(df):,} sản phẩm từ {load_path}")
        return df