import random

from utils.data_loader import DataLoader
from utils.preprocess import Preprocessor
from models.content_model import ContentBasedRecommender


def main():

    print("=" * 60)
    print("E-COMMERCE RECOMMENDATION SYSTEM")
    print("=" * 60)

    # 1. LOAD DATA
    print("\nSTEP 1: LOAD DATA")
    loader = DataLoader()
    df_raw = loader.load_and_preview(sample_size=100000)

    if df_raw is None:
        print("Error: Cannot load data")
        return

    # 2. PREPROCESS
    print("\nSTEP 2: PREPROCESS DATA")
    preprocessor = Preprocessor()

    df_clean = preprocessor.preprocess(
        df_raw,
        min_user_ratings=3,
        min_item_ratings=3
    )

    if df_clean is None:
        print("Error: Preprocess failed")
        return

    # 3. SAVE PROCESSED DATA
    print("\nSTEP 3: SAVE DATA")
    preprocessor.save_processed(df_clean)

    # 4. LOAD PROCESSED DATA
    print("\nSTEP 4: LOAD PROCESSED DATA")
    df_loaded = preprocessor.load_processed()

    # 5. SHOW SAMPLE TEXT
    print("\nSTEP 5: SAMPLE TEXT")
    preprocessor.get_sample_text(df_loaded)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED")
    print("=" * 60)

    # 6. RUN CONTENT-BASED MODEL
    print("\nSTEP 6: RUN CONTENT-BASED MODEL")

    recommender = ContentBasedRecommender()
    recommender.train_model(df=df_loaded, text_column='combined_text', title_column='ProductId')

    test_item = random.choice(df_loaded['ProductId'].unique())
    print(f"\n Gợi ý 5 món đồ có nội dung giống với '{test_item}'")
    print("=" * 60)

    result = recommender.get_recommendation(product_name = test_item, top_n = 5)

    print(result[['ProductId', 'Score', 'Summary']])

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    main()
