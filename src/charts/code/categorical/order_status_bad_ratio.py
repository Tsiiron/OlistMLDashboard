import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

def plot_bad_ratio_by_order_status(orders_df):
    df = orders_df.copy()
    df['review_label'] = df['review_score'].apply(lambda x: 'bad' if x <=2 else ('good' if x>=4 else 'neutral'))
    df_plot = df[df['review_label'] != 'neutral']

    counts = df_plot.groupby('order_status')['review_label'].value_counts(normalize=True).unstack().fillna(0)
    counts = counts.sort_values('bad', ascending=False)

    plt.figure(figsize=(10,6))
    sns.barplot(x=counts['bad'], y=counts.index, color='#E74C3C')
    plt.xlabel('Share of Bad Reviews')
    plt.ylabel('Order Status')
    plt.title('Bad Review Share by Order Status')
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

    plot_bad_ratio_by_order_status(orders_df)
