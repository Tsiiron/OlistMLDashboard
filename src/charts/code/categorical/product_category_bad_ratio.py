import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_bad_ratio_by_category(orders_items_df, products_df, reviews_df, top_n=15):
    """
    Доля плохих отзывов по категориям товаров.
    top_n: показать топ-N категорий по количеству заказов
    """
    # Подтягиваем категории товаров
    df = orders_items_df.merge(
        products_df[['product_id', 'product_category_name']],
        on='product_id',
        how='left'
    )

    # Подтягиваем review_score
    df = df.merge(
        reviews_df[['order_id', 'review_score']],
        on='order_id',
        how='left'
    )

    df = df[df['review_score'].notna()]

    # Категоризация отзывов
    df['review_label'] = df['review_score'].apply(lambda x: 'bad' if x <= 2 else ('good' if x >= 4 else 'neutral'))
    df_plot = df[df['review_label'] != 'neutral']

    # Считаем долю плохих отзывов по категории
    counts = df_plot.groupby('product_category_name')['review_label'].value_counts().unstack().fillna(0)
    counts['total_orders'] = counts.sum(axis=1)
    counts['bad_ratio'] = counts['bad'] / counts['total_orders']
    counts = counts.sort_values('bad_ratio', ascending=False)

    # Берем топ-N категорий
    counts = counts.head(top_n)

    plt.figure(figsize=(10,6))
    sns.barplot(x='bad_ratio', y=counts.index, data=counts, color='#E74C3C')
    plt.xlabel('Share of Bad Reviews')
    plt.ylabel('Product Category')
    plt.title(f'Share of Bad Reviews by Product Category (Top {top_n})')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Путь к исходным CSV
    BASE_DIR = Path(__file__).resolve().parent.parents[3] / "data"

    orders_items_df = pd.read_csv(BASE_DIR / "olist_order_items_dataset.csv")
    products_df = pd.read_csv(BASE_DIR / "olist_products_dataset.csv")
    reviews_df = pd.read_csv(BASE_DIR / "olist_order_reviews_dataset.csv")

    plot_bad_ratio_by_category(orders_items_df, products_df, reviews_df, top_n=15)
