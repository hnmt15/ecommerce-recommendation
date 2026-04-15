import random

from utils.data_loader import DataLoader
from utils.preprocess import Preprocessor
from models.content_model import ContentBasedRecommender


def main():
    print("STARTING HANDLE ECOMMERCE DATA")

    # 1. Load dữ liệu
    loader = DataLoader()
    df_raw = loader.load_and_preview(sample_size=None)  # None = load all

    if df_raw is None:
        print(" Không thể load dữ liệu!")
        return

    # 2. Thống kê raw data
    loader.get_statistics(df_raw)

    # 3. Preprocess
    preprocessor = Preprocessor()
    df_clean = preprocessor.preprocess(
        df_raw,
        min_user_ratings=3,  # User phải có ít nhất 3 đánh giá
        min_item_ratings=2  # Product phải có ít nhất 2 đánh giá
    )

    # 4. Chia train/test
    train, test = preprocessor.split_train_test(df_clean, test_size=0.2)

    # 5. Lưu dữ liệu
    preprocessor.save_processed(df_clean, 'ecommerce_clean.csv')
    preprocessor.save_train_test(train, test, 'ecommerce')

    # 6. Kiểm tra mẫu
    #preprocessor.get_sample_text(df_clean, n=3)

    # print("\n" + "=" * 60)
    # print("✅ PIPELINE HOÀN TẤT!")
    # print("=" * 60)
    # print("Dữ liệu đã được lưu trong: data/processed/")
    # print("  - ecommerce_clean.csv: Dữ liệu đã clean")
    # print("  - ecommerce_train.csv: Tập train")
    # print("  - ecommerce_test.csv: Tập test")

    # 6. RUN CONTENT-BASED MODEL
    print("\nSTEP 6: RUN CONTENT-BASED MODEL")

    recommender = ContentBasedRecommender()

    recommender.train_model(df=train, text_column='combined_text', title_column='product_id')

    test_item = random.choice(train['product_id'].unique())

    print(f"\n Gợi ý 5 món đồ có nội dung giống với ID: '{test_item}'")

    print("=" * 60)

    result = recommender.get_recommendation(product_name=test_item, top_n=5)

    if isinstance(result, str):
        print(result)
    else:
        print(result[['product_id', 'product_name', 'combined_text']].to_string(index=False))

if __name__ == "__main__":
    main()