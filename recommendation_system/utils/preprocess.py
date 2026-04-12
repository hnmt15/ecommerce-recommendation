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
        """
        if df is None:
            return None

        # Tạo text từ nhiều nguồn
        text_parts = []

        if 'review_text' in df.columns:
            text_parts.append(df['review_text'].fillna(''))
        elif 'Text' in df.columns:
            text_parts.append(df['Text'].fillna(''))

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

        # 2. LOẠI BỎ DỮ LIỆU NULL
        initial_len = len(df)
        df = df.dropna(subset=[user_col, item_col, rating_col])
        print(f"\n Loại bỏ null:")
        print(f"   Mất {initial_len - len(df)} dòng")

        # 3. LOẠI BỎ DUPLICATE (cùng user đánh giá cùng product nhiều lần)
        initial_len = len(df)
        df = df.drop_duplicates(subset=[user_col, item_col])
        print(f"\n Loại bỏ duplicate (user+product):")
        print(f"   Mất {initial_len - len(df)} dòng")

        # 4. LỌC USERS CÓ ÍT NHẤT min_user_ratings
        user_counts = df[user_col].value_counts()
        valid_users = user_counts[user_counts >= min_user_ratings].index
        initial_len = len(df)
        df = df[df[user_col].isin(valid_users)]
        # print(f"\n👤 Lọc users (ít nhất {min_user_ratings} ratings):")
        # print(f"   Giữ lại {len(valid_users):,} users")
        # print(f"   Mất {initial_len - len(df)} dòng")

        # 5. LỌC ITEMS CÓ ÍT NHẤT min_item_ratings
        item_counts = df[item_col].value_counts()
        valid_items = item_counts[item_counts >= min_item_ratings].index
        initial_len = len(df)
        df = df[df[item_col].isin(valid_items)]
        # print(f"\n📦 Lọc items (ít nhất {min_item_ratings} ratings):")
        # print(f"   Giữ lại {len(valid_items):,} items")
        # print(f"   Mất {initial_len - len(df)} dòng")

        # 6. CHUẨN HÓA ID
        # print(f"\n Chuẩn hóa ID cho model...")
        df['user_idx'] = df[user_col].astype('category').cat.codes
        df['item_idx'] = df[item_col].astype('category').cat.codes
        # print(f"   ✅ Đã tạo 'user_idx' (0-{df['user_idx'].max()}) và 'item_idx' (0-{df['item_idx'].max()})")

        # 7. CHUẨN HÓA RATING
        if rating_col != 'rating':
            df['rating'] = df[rating_col]
        else:
            df['rating'] = df[rating_col]

        # 8. THỐNG KÊ SAU PREPROCESS
        print(f"\n" + "=" * 50)
        print("AFTER PREPROCESS:")
        print("=" * 50)
        # print(f"   Số dòng: {len(df):,}")
        # print(f"   Users: {df['user_idx'].nunique():,}")
        # print(f"   Products: {df['item_idx'].nunique():,}")
        # print(f"   Rating TB: {df['rating'].mean():.2f} ± {df['rating'].std():.2f}")
        #
        # # Độ thưa
        n_users = df['user_idx'].nunique()
        n_items = df['item_idx'].nunique()
        n_interactions = len(df)
        sparsity = (1 - n_interactions / (n_users * n_items)) * 100
        print(f"   Độ thưa: {sparsity:.2f}%")

        return df
    def split_train_test(self, df, test_size=0.2, random_state=42):
        """Chia dữ liệu train/test theo thời gian hoặc ngẫu nhiên"""
        if df is None:
            return None, None

        # Thử chia theo thời gian nếu có cột date
        print("Split data:")
        train, test = train_test_split(
            df, test_size=test_size, random_state=random_state,
            stratify=df['rating']  # Giữ tỷ lệ rating
        )

        # Đảm bảo test chỉ chứa user và product có trong train
        valid_users = set(train['user_idx'].unique())
        valid_items = set(train['item_idx'].unique())

        test_before = len(test)
        test = test[
            test['user_idx'].isin(valid_users) &
            test['item_idx'].isin(valid_items)
            ]

        print(f"   Train: {len(train):,} interactions ({len(train) / len(df) * 100:.1f}%)")
        print(f"   Test:  {len(test):,} interactions ({len(test) / len(df) * 100:.1f}%)")
        print(f"   Test bị loại (user/item mới): {test_before - len(test)}")

        return train, test

    def save_processed(self, df, filename='ecommerce_clean.csv'):

        if df is None:
            return

        save_path = self.processed_dir / filename
        df.to_csv(save_path, index=False)

        file_size = save_path.stat().st_size / (1024 * 1024)


    def save_train_test(self, train, test, prefix='ecommerce'):

        if train is not None:
            train_path = self.processed_dir / f'{prefix}_train.csv'
            train.to_csv(train_path, index=False)


        if test is not None:
            test_path = self.processed_dir / f'{prefix}_test.csv'
            test.to_csv(test_path, index=False)


    def load_processed(self, filename='ecommerce_clean.csv'):

        load_path = self.processed_dir / filename

        if not load_path.exists():
            print(f" Không tìm thấy file: {load_path}")
            return None

        df = pd.read_csv(load_path)
        print(f" Đã load {len(df):,} rows từ {load_path}")
        return df

    # def get_sample_text(self, df, n=3):
    #
    #     if df is None:
    #         return
    #
    #     print("\n📝 MẪU TEXT SAU PREPROCESS:\n")
    #     for i in range(min(n, len(df))):
    #         print(f"\n{i + 1}. User {df.iloc[i]['user_id']} -> Product {df.iloc[i]['product_id']}")
    #         print(f"   Rating: {df.iloc[i]['rating']}⭐")
    #         if 'combined_text' in df.columns:
    #             text_preview = df.iloc[i]['combined_text'][:200]
    #             print(f"   Text: {text_preview}...")