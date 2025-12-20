import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

def plot_bad_ratio_by_seller(orders_df, top_n=15, min_orders=5):
    """
    Топ N продавцов по доле плохих отзывов.
    Учитываются только продавцы с >= min_orders заказов.
    """
    df = orders_df.copy()
    # Категоризация отзывов
    df['review_label'] = df['review_score'].apply(lambda x: 'bad' if x <=2 else ('good' if x>=4 else 'neutral'))
    df_plot = df[df['review_label'] != 'neutral']

    # Считаем количество и долю плохих отзывов по продавцам
    counts = df_plot.groupby('seller_id')['review_label'].value_counts().unstack().fillna(0)
    counts['total_orders'] = counts.sum(axis=1)
    counts = counts[counts['total_orders'] >= min_orders]  # фильтр по минимуму заказов
    counts['bad_ratio'] = counts['bad'] / counts['total_orders']

    # Берем топ N продавцов по доле плохих отзывов
    counts_top = counts.sort_values('bad_ratio', ascending=False).head(top_n)

    plt.figure(figsize=(12,6))
    sns.barplot(x='bad_ratio', y=counts_top.index, data=counts_top, color='#E74C3C')
    plt.xlabel('Share of Bad Reviews')
    plt.ylabel('Seller ID')
    plt.title(f'Top {top_n} Sellers by Share of Bad Reviews (min {min_orders} orders)')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    CURRENT_DIR = Path(__file__).resolve().parent
    SRC_DIR = CURRENT_DIR.parents[3] / "src"
    sys.path.insert(0, str(SRC_DIR))

    from analysis.init_data import read_datasets
    from analysis.data_processing import build_feature_table

    datasets = read_datasets()
    datasets = build_feature_table(datasets)

    orders_df = datasets['orders_enriched']

    plot_bad_ratio_by_seller(orders_df, top_n=15, min_orders=5)
