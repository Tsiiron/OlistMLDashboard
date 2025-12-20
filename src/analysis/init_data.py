from pathlib import Path
import pandas as pd


def read_datasets(
    base_dir: str | Path = None,
    extensions: tuple[str, ...] = (".csv",),
    verbose: bool = True
) -> dict[str, pd.DataFrame]:
    """
    Считывает табличные файлы из директории data и возвращает словарь DataFrame'ов.

    Parameters
    ----------
    base_dir : str | Path | None
        Путь к папке с данными. Если None — используется папка /data в корне проекта.
    extensions : tuple[str, ...]
        Допустимые расширения файлов.
    verbose : bool
        Выводить ли информацию о загрузке файлов.
    """

    if base_dir is None:
        # src/analysis/init_data.py -> src -> корень проекта -> data
        base_dir = Path(__file__).resolve().parents[2] / "data"
    else:
        base_dir = Path(base_dir)

    if not base_dir.exists():
        raise FileNotFoundError(f"Каталог с данными не найден: {base_dir}")

    datasets: dict[str, pd.DataFrame] = {}

    for file_path in base_dir.iterdir():
        if file_path.suffix.lower() in extensions:
            key = file_path.stem
            df = pd.read_csv(file_path)
            datasets[key] = df

            if verbose:
                print(
                    f"[DATA] {key:<35} "
                    f"rows={df.shape[0]:>7} cols={df.shape[1]:>3}"
                )

    if not datasets:
        raise ValueError("В каталоге data не найдено подходящих файлов")

    return datasets
