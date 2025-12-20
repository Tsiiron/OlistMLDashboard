from pathlib import Path
import sys
import pandas as pd

# --------------------------------------------------
# Добавляем src в sys.path для корректных импортов
# --------------------------------------------------
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from analysis.init_data import read_datasets
from analysis.data_processing import build_feature_table


def export_for_datalens(
    datasets: dict[str, pd.DataFrame],
    table_key: str = "orders_enriched",
    output_name: str = "orders_analytics",
    index: bool = False
) -> None:
    """
    Сохраняет итоговую таблицу в папку datalens.
    """

    if table_key not in datasets:
        raise KeyError(f"Таблица '{table_key}' не найдена в datasets")

    datalens_dir = Path(__file__).resolve().parents[2] / "datalens"
    datalens_dir.mkdir(parents=True, exist_ok=True)

    output_path = datalens_dir / f"{output_name}.csv"
    datasets[table_key].to_csv(output_path, index=index)

    print(
        f"[EXPORT] CSV сохранён: datalens/{output_path.name} "
        f"(строк: {datasets[table_key].shape[0]})"
    )


if __name__ == "__main__":
    print("[RUN] Загрузка данных")
    datasets = read_datasets()

    print("[RUN] Обработка данных")
    datasets = build_feature_table(datasets)

    print("[RUN] Экспорт данных")
    export_for_datalens(datasets)
