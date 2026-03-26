import os
import pandas as pd
from pathlib import Path

class DataLoader:
    def __init__(self):
        current_dir = Path(__file__).parent

        # Lên 1 cấp để về recommendation_system (vì utils nằm trong đó)
        self.project_root = current_dir.parent

        # data/raw cùng cấp với utils
        self.data_dir = self.project_root / 'data' / 'raw'
        self.data_dir.mkdir(parents=True, exist_ok=True)


    def download_amazon_reviews(self, sample_size=100000):
        csv_path = self.data_dir / 'amazon_reviews.csv'
        print("CSV PATH:", csv_path.resolve())
        if csv_path.exists():
            print(f"Đã tìm thấy file CSV: {csv_path}")
            # Đọc với giới hạn số dòng
            df = pd.read_csv(csv_path, nrows=sample_size)
            print(f"Loaded {len(df)} reviews (giới hạn {sample_size} dòng)")
            return df

        print("Đang tải dataset từ Kaggle...")
        try:
            os.system(f"kaggle datasets download -d arhamrumi/amazon-product-reviews -p {self.data_dir} --unzip")
            extracted_files = list(self.data_dir.glob("*.csv"))
            if extracted_files:
                csv_file = extracted_files[0]
                csv_file.rename(csv_path)
                print(f"Đã tải và giải nén vào: {csv_path}")
                # Đọc với giới hạn số dòng
                df = pd.read_csv(csv_path, nrows=sample_size)
                print(f"Loaded {len(df)} reviews (giới hạn {sample_size} dòng)")
                return df
            else:
                print("Không tìm thấy file CSV sau khi giải nén")
                return None
        except Exception as e:
            print(f"Lỗi khi tải dataset: {e}")
            return None

    def load_and_preview(self, sample_size=100000):
        df = self.download_amazon_reviews(sample_size=sample_size)

        if df is not None:
            print(f"\n" + "=" * 50)
            print("THÔNG TIN DATAFRAME")
            print(f"=" * 50)
            print(f"Các cột: {df.columns.tolist()}")
            print(f"Số dòng: {len(df)}")
            print(f"Users: {df['UserId'].nunique()}")
            print(f"Products: {df['ProductId'].nunique()}")

            return df
        return None

#
# if __name__ == "__main__":
#     loader = DataLoader()
#     df = loader.load_and_preview(sample_size=10000)