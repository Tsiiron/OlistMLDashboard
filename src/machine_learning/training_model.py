import pandas as pd
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Добавляем путь к src/analysis
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "analysis"))

from init_data import read_datasets
from data_processing import build_feature_table


def train_ml_model():
    # --- Загрузка и предобработка данных ---
    dfs = read_datasets()  # читаем CSV из папки data
    dfs = build_feature_table(dfs)  # сборка финальной таблицы с признаками

    orders_full = dfs.get("orders_enriched")
    if orders_full is None:
        print(f"Ключи в dfs: {list(dfs.keys())}")
        raise KeyError("Не найден ключ с объединёнными данными.")

    # --- Целевая переменная ---
    orders_full["bad_review"] = orders_full["review_score"].apply(lambda x: 1 if x <= 2 else 0)

    # --- Признаки ---
    numeric_features = [
        "delivery_duration_days",
        "approval_delay_days",
        "estimated_vs_actual_delay",
        "items_count",
        "payment_value_total"
    ]
    categorical_features = [
        "payment_installments_max",
        "seller_id"
    ]

    # --- Числовые признаки ---
    X_num = orders_full[numeric_features].fillna(0)

    # --- Категориальные признаки ---
    X_cat = orders_full[categorical_features].fillna("unknown").astype(str)  # ВСЕ в строки
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_cat_encoded = pd.DataFrame(
        ohe.fit_transform(X_cat),
        columns=ohe.get_feature_names_out(categorical_features),
        index=X_cat.index
    )

    # --- Итоговый датасет ---
    X = pd.concat([X_num, X_cat_encoded], axis=1)
    y = orders_full["bad_review"]

    # --- Разделение на train/test ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Обучение модели ---
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # --- Оценка модели ---
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    return model, ohe, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    train_ml_model()
