import pandas as pd
from pathlib import Path

class Preprocessor:

    def __init__(self):

        current_dir = Path(__file__).parent
        self.project_root = current_dir.parent

        self.processed_dir = self.project_root / 'data' / 'processed'
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def preprocess(self, df, min_user_ratings=3, min_item_ratings=3):
        """
            df: DataFrame thô từ DataLoader
            min_user_ratings: Users có ít nhất số ratings này mới giữ
            min_item_ratings: Items có ít nhất số ratings này mới giữ

        """
        if df is None:
            return None
        print("BẮT ĐẦU PREPROCESS...")

        # 1. TẠO CỘT TEXT CHO CONTENT-BASED
        print("\nTạo cột combined_text cho Content-based...")
        df['combined_text'] = df['Summary'].fillna('') + " " + df['Text'].fillna('')
        print(f"---> Đã tạo cột 'combined_text'")

        # 2. LOẠI BỎ DỮ LIỆU NULL
        initial_len = len(df)
        df = df.dropna(subset=['UserId', 'ProductId', 'Score'])
        print(f"\nLoại bỏ null:")
        print(f"---> Mất {initial_len - len(df)} dòng")

        # 3. LOẠI BỎ DUPLICATE (cùng user đánh giá cùng product nhiều lần)
        initial_len = len(df)
        df = df.drop_duplicates(subset=['UserId', 'ProductId'])
        print(f"\nLoại bỏ duplicate:")
        print(f"---> Mất {initial_len - len(df)} dòng")

        # 4. LỌC USERS CÓ ÍT NHẤT min_user_ratings
        user_counts = df['UserId'].value_counts()
        valid_users = user_counts[user_counts >= min_user_ratings].index
        initial_len = len(df)
        df = df[df['UserId'].isin(valid_users)]
        print(f"\nLọc users (ít nhất {min_user_ratings} ratings):")
        print(f"---> Giữ lại {len(valid_users)} users")
        print(f"---> Mất {initial_len - len(df)} dòng")

        # 5. LỌC ITEMS CÓ ÍT NHẤT min_item_ratings
        item_counts = df['ProductId'].value_counts()
        valid_items = item_counts[item_counts >= min_item_ratings].index
        initial_len = len(df)
        df = df[df['ProductId'].isin(valid_items)]
        print(f"\nLọc items (ít nhất {min_item_ratings} ratings):")
        print(f"---> Giữ lại {len(valid_items)} items")
        print(f"---> Mất {initial_len - len(df)} dòng")

        # 6. CHUẨN HÓA ID (chuyển thành số 0,1,2... cho model)
        print(f"\nChuẩn hóa ID cho model...")
        df['user_idx'] = df['UserId'].astype('category').cat.codes
        df['item_idx'] = df['ProductId'].astype('category').cat.codes
        print(f"---> Đã tạo 'user_idx' và 'item_idx'")

        # 7. THỐNG KÊ SAU PREPROCESS
        print(f"\n" + "=" * 50)
        print("THỐNG KÊ SAU PREPROCESS:")
        print("=" * 50)
        print(f"   Số dòng: {len(df):,}")
        print(f"   Users: {df['user_idx'].nunique():,}")
        print(f"   Products: {df['item_idx'].nunique():,}")
        print(f"   Rating TB: {df['Score'].mean():.2f}")

        # Độ thưa của ma trận
        n_users = df['user_idx'].nunique()
        n_items = df['item_idx'].nunique()
        n_interactions = len(df)
        sparsity = (1 - n_interactions / (n_users * n_items)) * 100
        print(f"   Độ thưa: {sparsity:.2f}%")

        return df

    def save_processed(self, df, filename='amazon_clean.csv'):
        if df is None:
            return

        save_path = self.processed_dir / filename
        df.to_csv(save_path, index=False)
        print(f"\nĐã lưu vào: {save_path}")
        # print(f"Kích thước: {save_path.stat().st_size / (1024 * 1024):.2f} MB")

    def load_processed(self, filename='amazon_clean.csv'):
        load_path = self.processed_dir / filename

        if not load_path.exists():
            print(f"Không tìm thấy file: {load_path}")
            return None

        df = pd.read_csv(load_path)
        print(f"Đã load {len(df)} rows từ {load_path}")
        return df

    def get_sample_text(self, df, n=3):
        """Hiển thị mẫu text để kiểm tra"""
        if df is None:
            return

        print("\nMẪU TEXT SAU PREPROCESS:\n")
        for i in range(min(n, len(df))):
            print(f"\n{i + 1}. Summary: {df.iloc[i]['Summary'][:100]}...")
            print(f"   Text: {df.iloc[i]['Text'][:150]}...")
            print(f"   Rating: {df.iloc[i]['Score']}⭐")