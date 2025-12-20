import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

def plot_bad_ratio_by_state(orders_df, customers_df, top_n=15):
    df = orders_df.merge(customers_df[['customer_id', 'customer_state']], on='customer_id', how='left')
    df['review_label'] = df['review_score'].apply(lambda x: 'bad' if x <=2 else ('good' if x>=4 else 'neutral'))
    df_plot = df[df['review_label'] != 'neutral']

    # считаем долю плохих отзывов
    counts = df_plot.groupby('customer_state')['review_label'].value_counts(normalize=True).unstack().fillna(0)
    counts = counts.sort_values('bad', ascending=False).head(top_n)

    plt.figure(figsize=(12,6))
    sns.barplot(x=counts['bad'], y=counts.index, color='#E74C3C')
    plt.xlabel('Share of Bad Reviews')
    plt.ylabel('Customer State')
    plt.title(f'Top {top_n} Customer States by Share of Bad Reviews')
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
    customers_df = datasets['olist_customers_dataset']

    plot_bad_ratio_by_state(orders_df, customers_df, top_n=15)
