from utils.data_loader import DataLoader
from utils.preprocess import Preprocessor



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


if __name__ == "__main__":
    main()
