import pandas as pd
from pathlib import Path


def quick_check():
    """Kiểm tra nhanh cấu trúc của các file CSV"""
    data_dir = Path(__file__).parent.parent / 'data' / 'raw'

    # Kiểm tra users
    users_path = data_dir / 'users.csv'
    if users_path.exists():
        users = pd.read_csv(users_path)
        print("\n" + "=" * 50)
        print("📄 users.csv")
        print("=" * 50)
        print(f"Các cột: {users.columns.tolist()}")
        print(f"Số dòng: {len(users)}")
        print("\n3 dòng đầu:")
        print(users.head(3))

    # Kiểm tra reviews
    reviews_path = data_dir / 'reviews.csv'
    if reviews_path.exists():
        reviews = pd.read_csv(reviews_path)
        print("\n" + "=" * 50)
        print("📄 reviews.csv")
        print("=" * 50)
        print(f"Các cột: {reviews.columns.tolist()}")
        print(f"Số dòng: {len(reviews)}")
        print("\n3 dòng đầu:")
        print(reviews.head(3))

    # Kiểm tra products
    products_path = data_dir / 'products.csv'
    if products_path.exists():
        products = pd.read_csv(products_path)
        print("\n" + "=" * 50)
        print("📄 products.csv")
        print("=" * 50)
        print(f"Các cột: {products.columns.tolist()}")
        print(f"Số dòng: {len(products)}")
        print("\n3 dòng đầu:")
        print(products.head(3))


if __name__ == "__main__":
    quick_check()