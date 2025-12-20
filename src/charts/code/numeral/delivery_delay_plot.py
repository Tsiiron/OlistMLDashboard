import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import matplotlib.ticker as mticker

# --------------------------
# Функция построения графика
# --------------------------
def plot_delivery_vs_review(df: pd.DataFrame):
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
    df_plot = df[df["review_label"] != "neutral"]

    # --- Стиль графика ---
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 6))

    sns.histplot(
        data=df_plot,
        x="estimated_vs_actual_delay",
        hue="review_label",
        bins=25,
        palette={"bad": "#E74C3C", "good": "#27AE60"},
        multiple="dodge",
        edgecolor="#2C3E50",
        alpha=0.8
    )

    plt.title("Delivery Delay Distribution by Review Type", fontsize=16, weight="bold")
    plt.xlabel("Delivery Delay (days)", fontsize=13)
    plt.ylabel("Number of Orders", fontsize=13)

    # --- Ограничение оси X ---
    plt.xlim(-50, 50)

    # --- Вертикальная линия для on-time (сплошная) ---
    plt.axvline(0, color="#2980B9", linewidth=2, alpha=0.8, label="On-time")

    # --- Больше делений на оси X ---
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(5))  # каждые 5 дней
    plt.gca().xaxis.set_minor_locator(mticker.MultipleLocator(1))  # каждая 1 день как минорные деления

    # --- Легенда ---
    plt.legend(title="Review Type", fontsize=11, title_fontsize=12)

    plt.tight_layout()
    plt.show()


# ========================
# Исполняемый блок
# ========================
if __name__ == "__main__":
    # -----------------------
    # Добавляем src в sys.path
    # -----------------------
    CURRENT_DIR = Path(__file__).resolve().parent
    SRC_DIR = CURRENT_DIR.parents[3] / "src"
    sys.path.insert(0, str(SRC_DIR))

    # -----------------------
    # Импорты после sys.path
    # -----------------------
    from analysis.init_data import read_datasets
    from analysis.data_processing import build_feature_table

    # --- Загрузка и подготовка данных ---
    print("[RUN] Загрузка данных")
    datasets = read_datasets()

    print("[RUN] Предобработка и генерация признаков")
    datasets = build_feature_table(datasets)
    orders_df = datasets["orders_enriched"]

    print("[RUN] Построение графика задержки доставки")
    plot_delivery_vs_review(orders_df)
