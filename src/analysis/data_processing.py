import pandas as pd


def build_feature_table(datasets: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Подготавливает данные для аналитики и ML:
    очищает таблицы, агрегирует транзакции и формирует итоговую таблицу orders_enriched.
    """

    data = datasets.copy()

    # ---------- PRODUCTS ----------
    products_key = "olist_products_dataset"
    if products_key in data:
        data[products_key] = data[products_key].assign(
            product_category_name=lambda df: df["product_category_name"].fillna("undefined")
        )
        print("[PROCESS] products → категории очищены")

    # ---------- ORDERS ----------
    orders_key = "olist_orders_dataset"
    if orders_key in data:
        orders = data[orders_key]

        datetime_fields = [
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ]

        for field in datetime_fields:
            orders[field] = pd.to_datetime(orders[field], errors="coerce")

        data[orders_key] = orders
        print("[PROCESS] orders → временные поля приведены к datetime")

    # ---------- PAYMENTS ----------
    payments_key = "olist_order_payments_dataset"
    if payments_key in data:
        payments_summary = (
            data[payments_key]
            .groupby("order_id", as_index=False)
            .agg(
                payment_value_total=("payment_value", "sum"),
                payment_installments_max=("payment_installments", "max"),
            )
        )
        data[payments_key] = payments_summary
        print("[PROCESS] payments → агрегированы суммы и рассрочки")

    # ---------- ORDER ITEMS ----------
    items_key = "olist_order_items_dataset"
    if items_key in data:
        items_summary = (
            data[items_key]
            .groupby("order_id", as_index=False)
            .agg(
                order_revenue=("price", "sum"),
                items_count=("product_id", "count"),
                seller_id=("seller_id", "first"),
            )
        )
        data[items_key] = items_summary
        print("[PROCESS] items → рассчитана выручка и количество товаров")

    # ---------- REVIEWS ----------
    reviews_key = "olist_order_reviews_dataset"
    if reviews_key in data:
        data[reviews_key] = data[reviews_key][["order_id", "review_score"]]
        print("[PROCESS] reviews → выбраны ключевые поля")

    # ---------- FINAL MERGE ----------
    orders_enriched = (
        data[orders_key]
        .merge(data.get(reviews_key), on="order_id", how="left")
        .merge(data.get(payments_key), on="order_id", how="left")
        .merge(data.get(items_key), on="order_id", how="left")
    )

    # ---------- FEATURE ENGINEERING ----------
    orders_enriched = orders_enriched.assign(
        delivery_duration_days=lambda df: (
            df["order_delivered_customer_date"] - df["order_purchase_timestamp"]
        ).dt.days,
        approval_delay_days=lambda df: (
            df["order_approved_at"] - df["order_purchase_timestamp"]
        ).dt.days,
        estimated_vs_actual_delay=lambda df: (
            df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]
        ).dt.days,
    )

    numeric_cols = [
        "delivery_duration_days",
        "approval_delay_days",
        "estimated_vs_actual_delay",
        "items_count",
        "order_revenue",
    ]
    orders_enriched[numeric_cols] = orders_enriched[numeric_cols].fillna(0)

    data["orders_enriched"] = orders_enriched
    print("[PROCESS] orders_enriched → финальная таблица сформирована")

    return data
