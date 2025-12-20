import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def plot_correlation_heatmap(orders_df: pd.DataFrame):
    numeric_cols = [
        "review_score",
        "delivery_duration_days",   # delivery_time_days
        "approval_delay_days",      # processing_time_days
        "estimated_vs_actual_delay",# delivery_delay_days
        "order_revenue",            # total_payment
        "items_count"               # order_item_count
    ]

    # Берем только существующие колонки
    numeric_cols = [col for col in numeric_cols if col in orders_df.columns]
    data = orders_df[numeric_cols].dropna()

    corr = data.corr(method="pearson")

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0
    )
    plt.title("Корреляционная матрица числовых признаков")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parents[3] / "data"

    # Загружаем необходимые CSV
    orders_df = pd.read_csv(BASE_DIR / "olist_orders_dataset.csv")
    order_items_df = pd.read_csv(BASE_DIR / "olist_order_items_dataset.csv")
    payments_df = pd.read_csv(BASE_DIR / "olist_order_payments_dataset.csv")

    # --- создаем агрегированные числовые признаки как в orders_enriched ---
    # delivery_duration_days
    orders_df['order_purchase_timestamp'] = pd.to_datetime(orders_df['order_purchase_timestamp'])
    orders_df['order_approved_at'] = pd.to_datetime(orders_df['order_approved_at'])
    orders_df['order_delivered_customer_date'] = pd.to_datetime(orders_df['order_delivered_customer_date'])
    orders_df['order_estimated_delivery_date'] = pd.to_datetime(orders_df['order_estimated_delivery_date'])

    orders_df['delivery_duration_days'] = (orders_df['order_delivered_customer_date'] - orders_df['order_purchase_timestamp']).dt.days
    orders_df['approval_delay_days'] = (orders_df['order_approved_at'] - orders_df['order_purchase_timestamp']).dt.days
    orders_df['estimated_vs_actual_delay'] = (orders_df['order_delivered_customer_date'] - orders_df['order_estimated_delivery_date']).dt.days

    # order_revenue и items_count из order_items
    items_agg = order_items_df.groupby("order_id").agg({
        "price": "sum",
        "product_id": "count"
    }).rename(columns={"price": "order_revenue", "product_id": "items_count"}).reset_index()

    orders_df = orders_df.merge(items_agg, on="order_id", how="left")

    plot_correlation_heatmap(orders_df)
