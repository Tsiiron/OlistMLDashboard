import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_bad_ratio_by_payment_method(orders_df, reviews_df, payments_df, min_orders=5):
    """
    Доля плохих отзывов по способам оплаты.
    Учитываются только способы оплаты с минимумом min_orders заказов.
    """
    # Подтягиваем review_score и payment_type к заказам
    df = orders_df.merge(
        reviews_df[['order_id', 'review_score']],
        on='order_id',
        how='left'
    ).merge(
        payments_df[['order_id', 'payment_type']],
        on='order_id',
        how='left'
    )

    # Убираем заказы без отзывов
    df = df[df['review_score'].notna()]

    # Категоризация отзывов
    df['review_label'] = df['review_score'].apply(lambda x: 'bad' if x <= 2 else ('good' if x >= 4 else 'neutral'))
    df_plot = df[df['review_label'] != 'neutral']

    # Считаем количество и долю плохих отзывов по payment_type
    counts = df_plot.groupby('payment_type')['review_label'].value_counts().unstack().fillna(0)
    counts['total_orders'] = counts.sum(axis=1)
    counts = counts[counts['total_orders'] >= min_orders]  # фильтр по минимуму заказов
    counts['bad_ratio'] = counts['bad'] / counts['total_orders']
    counts = counts.sort_values('bad_ratio', ascending=False)

    plt.figure(figsize=(10,6))
    sns.barplot(x='bad_ratio', y=counts.index, data=counts, color='#E74C3C')
    plt.xlabel('Share of Bad Reviews')
    plt.ylabel('Payment Method')
    plt.title(f'Share of Bad Reviews by Payment Method (min {min_orders} orders)')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Путь к исходным CSV
    BASE_DIR = Path(__file__).resolve().parent.parents[3] / "data"
    orders_df = pd.read_csv(BASE_DIR / "olist_orders_dataset.csv")
    reviews_df = pd.read_csv(BASE_DIR / "olist_order_reviews_dataset.csv")
    payments_df = pd.read_csv(BASE_DIR / "olist_order_payments_dataset.csv")

    plot_bad_ratio_by_payment_method(orders_df, reviews_df, payments_df, min_orders=5)
