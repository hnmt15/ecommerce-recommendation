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

    def build_user_history(self, data, user_id=None, include_events=True, include_reviews=True):
        """
        Xây dựng lịch sử tương tác của user từ reviews và events

        Args:
            data: Dictionary chứa các DataFrame (từ load_all_data)
            user_id: ID của user cần lấy (None = lấy tất cả users)
            include_events: Có bao gồm events không
            include_reviews: Có bao gồm reviews không

        Returns:
            DataFrame với lịch sử tương tác
        """
        interactions = []

        # ========== 1. TỪ REVIEWS (explicit feedback) ==========
        if include_reviews and data['reviews'] is not None:
            reviews = data['reviews'].copy()

            # Lọc theo user_id nếu có
            if user_id:
                reviews = reviews[reviews['user_id'] == user_id]

            # Thêm thông tin sản phẩm
            if data['products'] is not None:
                products = data['products'][['product_id', 'product_name', 'category', 'brand']]
                reviews = reviews.merge(products, on='product_id', how='left')

            for _, row in reviews.iterrows():
                interactions.append({
                    'user_id': row['user_id'],
                    'product_id': row['product_id'],
                    'product_name': row.get('product_name', 'Unknown'),
                    'category': row.get('category', ''),
                    'brand': row.get('brand', ''),
                    'rating': row.get('rating', 0),  # rating thật 1-5
                    'weight': row.get('rating', 1),  # trọng số = rating
                    'source': 'review',
                    'timestamp': row.get('timestamp', None)
                })

        # ========== 2. TỪ EVENTS (implicit feedback) ==========
        if include_events and data['events'] is not None:
            events = data['events'].copy()

            # Lọc theo user_id nếu có
            if user_id:
                events = events[events['user_id'] == user_id]

            # Thêm thông tin sản phẩm
            if data['products'] is not None:
                products = data['products'][['product_id', 'product_name', 'category', 'brand']]
                events = events.merge(products, on='product_id', how='left')

            # Định nghĩa trọng số cho từng loại event
            event_weights = {
                'view': 1,  # Xem: quan tâm nhẹ
                'cart': 3,  # Thêm giỏ: quan tâm nhiều
                'wishlist': 4,  # Yêu thích: quan tâm rất nhiều
                'purchase': 5  # Mua: quan tâm cao nhất
            }

            for _, row in events.iterrows():
                event_type = row['event_type']
                weight = event_weights.get(event_type, 1)

                interactions.append({
                    'user_id': row['user_id'],
                    'product_id': row['product_id'],
                    'product_name': row.get('product_name', 'Unknown'),
                    'category': row.get('category', ''),
                    'brand': row.get('brand', ''),
                    'rating': None,  # Không có rating thật
                    'weight': weight,  # Trọng số theo hành vi
                    'source': event_type,
                    'timestamp': row.get('timestamp', None)
                })

        if not interactions:
            print(f"️ Không có tương tác cho user {user_id if user_id else 'tất cả'}")
            return pd.DataFrame()

        # ========== 3. TỔNG HỢP VÀ XỬ LÝ TRÙNG LẶP ==========
        interactions_df = pd.DataFrame(interactions)

        # Nếu cùng user-product có nhiều tương tác, lấy weight cao nhất
        interactions_df = interactions_df.groupby(['user_id', 'product_id']).agg({
            'product_name': 'first',
            'category': 'first',
            'brand': 'first',
            'rating': 'max',  # Lấy rating cao nhất
            'weight': 'max',  # Lấy trọng số cao nhất
            'source': lambda x: '|'.join(sorted(set(x))),  # Gộp các nguồn
            'timestamp': 'max'
        }).reset_index()

        # Sắp xếp theo weight giảm dần (tương tác quan trọng nhất lên đầu)
        interactions_df = interactions_df.sort_values('weight', ascending=False)

        return interactions_df

    def get_user_history(self, data, user_id, min_weight=1):
        """
        Lấy lịch sử tương tác của một user cụ thể

        Args:
            data: Dictionary chứa các DataFrame
            user_id: ID của user cần lấy
            min_weight: Chỉ lấy tương tác có weight >= ngưỡng này

        Returns:
            DataFrame: Lịch sử tương tác của user
        """
        # Xây dựng interactions cho user
        interactions_df = self.build_user_history(data, user_id=user_id)

        if interactions_df.empty:
            return interactions_df

        # Lọc theo trọng số
        interactions_df = interactions_df[interactions_df['weight'] >= min_weight]

        print(f"\n📊 Lịch sử user {user_id}:")
        print(f"   Tổng số tương tác: {len(interactions_df)}")
        print(f"   Phân bố nguồn:")
        source_counts = interactions_df['source'].value_counts()
        for source, count in source_counts.items():
            print(f"      - {source}: {count}")
        print(f"   Weight trung bình: {interactions_df['weight'].mean():.2f}")

        return interactions_df

    def get_top_users(self, data, n=10, aggregate=True):
        """
        Lấy top N users có nhiều tương tác nhất
        """
        interactions_df = self.build_user_history(data, user_id=None)

        if interactions_df.empty:
            return []

        top_users = interactions_df['user_id'].value_counts().head(n)

        print(f"\n🏆 Top {n} users có nhiều tương tác nhất:")
        for idx, (user_id, count) in enumerate(top_users.items(), 1):
            print(f"   {idx}. User {user_id}: {count} tương tác")

        return top_users.index.tolist()

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


