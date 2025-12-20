import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import matplotlib.patches as mpatches

# --------------------------
# Функция построения графика
# --------------------------
def plot_processing_time_by_review(df: pd.DataFrame):
    """
    Гистограмма распределения времени обработки заказа (approval_delay_days)
    по типу отзывов. Каждое уникальное значение времени имеет отдельный столбец.
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
    df_plot = df[(df["review_label"] != "neutral") & (df["approval_delay_days"].notna())]

    if df_plot.empty:
        print("Нет данных для построения графика.")
        return

    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 6))

    # --- Бины для каждого уникального значения ---
    unique_bins = sorted([x for x in df_plot["approval_delay_days"].unique() if x <= 7])
    bins = [x - 0.5 for x in unique_bins] + [unique_bins[-1] + 0.5]

    sns.histplot(
        data=df_plot,
        x="approval_delay_days",
        hue="review_label",
        bins=bins,
        palette={"bad": "#E74C3C", "good": "#27AE60"},
        multiple="dodge",
        edgecolor="#2C3E50",
        alpha=0.8
    )

    plt.title("Processing Time by Review Type", fontsize=16, weight="bold")
    plt.xlabel("Processing Time (days)", fontsize=13)
    plt.ylabel("Number of Orders", fontsize=13)

    # --- Ось X под уникальные значения ---
    plt.xticks(unique_bins)

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

    print("[RUN] Построение графика времени обработки заказов по типу отзывов")
    plot_processing_time_by_review(orders_df)
