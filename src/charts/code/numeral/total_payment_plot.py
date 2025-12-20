import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches

# --------------------------
# Функция построения графика
# --------------------------
def plot_total_payment_by_review(df: pd.DataFrame, max_payment=1000):
    """
    Гистограмма распределения общей суммы платежа (total_payment)
    по типу отзывов. Нейтральные отзывы исключены.
    Параметр max_payment ограничивает ось X.
    """
    df = df.copy()

    # --- Категоризация отзывов ---
    def label_review(score):
        if score <= 2:
            return "bad"
        elif score >= 4:
            return "good"
        else:
            return "neutral"

    df["review_label"] = df["review_score"].apply(label_review)
    df_plot = df[(df["review_label"] != "neutral") & (df["total_payment"].notna())]

    if df_plot.empty:
        print("Нет данных для построения графика.")
        return

    # --- Ограничение максимальной суммы ---
    df_plot = df_plot[df_plot["total_payment"] <= max_payment]

    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 6))

    # --- Бины для каждого целого значения или округленного интервала ---
    bins = 30  # можно увеличить/уменьшить для детализации

    sns.histplot(
        data=df_plot,
        x="total_payment",
        hue="review_label",
        bins=bins,
        palette={"bad": "#E74C3C", "good": "#27AE60"},
        multiple="dodge",
        edgecolor="#2C3E50",
        alpha=0.8
    )

    plt.title("Total Payment by Review Type", fontsize=16, weight="bold")
    plt.xlabel("Total Payment ($)", fontsize=13)
    plt.ylabel("Number of Orders", fontsize=13)

    # --- Ограничение оси X ---
    plt.xlim(0, max_payment)

    # --- Больше делений на оси X ---
    plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # --- Легенда вручную ---
    red_patch = mpatches.Patch(color="#E74C3C", label="bad")
    green_patch = mpatches.Patch(color="#27AE60", label="good")
    plt.legend(handles=[red_patch, green_patch], title="Review Type", fontsize=11, title_fontsize=12)

    plt.tight_layout()
    plt.show()


# ========================
# Исполняемый блок
# ========================
if __name__ == "__main__":
    # Добавляем src в sys.path
    CURRENT_DIR = Path(__file__).resolve().parent
    SRC_DIR = CURRENT_DIR.parents[3] / "src"
    sys.path.insert(0, str(SRC_DIR))

    # Импорты после sys.path
    from analysis.init_data import read_datasets
    from analysis.data_processing import build_feature_table

    # --- Загрузка и подготовка данных ---
    print("[RUN] Загрузка данных")
    datasets = read_datasets()

    print("[RUN] Предобработка и генерация признаков")
    datasets = build_feature_table(datasets)
    orders_df = datasets["orders_enriched"]

    # В новой таблице колонка с суммой платежей называется "order_revenue" или "payment_value_total"?
    # Если есть "order_revenue", используем её:
    orders_df["total_payment"] = orders_df["order_revenue"]

    print("[RUN] Построение графика общей суммы платежа по типу отзывов")
    plot_total_payment_by_review(orders_df, max_payment=1000)
