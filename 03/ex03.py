"""
Exercicio 03 - Retail Black Friday Sales

Treina um sistema inteligente com tres classificadores, um para cada
indicacao solicitada:

1. Categoria do produto (product_category)
2. Forma de pagamento (payment_method)
3. Faixa etaria do comprador (age_group)

O CSV nao precisa estar no repositorio. Quando o arquivo local nao existe,
o script baixa a base publica com kagglehub:

    kagglehub.dataset_download("noopurbhatt/retail-black-friday-sales-dataset")
"""

from __future__ import annotations

from pathlib import Path
from pickle import dump
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from inferencia import demonstrar_inferencia


RANDOM_STATE = 42
DATASET_ID = "noopurbhatt/retail-black-friday-sales-dataset"
SCRIPT_DIR = Path(__file__).resolve().parent
LOCAL_DATA_PATH = SCRIPT_DIR / "retail_black_friday_sales_100k.csv"
MODEL_PATH = SCRIPT_DIR / "modelo_retail_black_friday.pkl"
METRICS_PATH = SCRIPT_DIR / "metricas_ex03.csv"

TARGETS = {
    "product_category": "categoria do produto",
    "payment_method": "forma de pagamento",
    "age_group": "faixa etaria",
}

CATEGORICAL_FEATURES = [
    "gender",
    "city",
    "customer_segment",
    "product_id",
]

NUMERIC_FEATURES = [
    "original_price",
    "discount_pct",
    "final_price",
    "quantity",
    "purchase_amount",
    "purchase_hour",
    "is_weekend",
    "is_black_friday",
    "purchase_month",
    "purchase_day",
    "purchase_dayofweek",
]

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
DROP_COLUMNS = ["transaction_id", "customer_id", "purchase_date"]
DERIVED_DATE_FEATURES = ["purchase_month", "purchase_day", "purchase_dayofweek"]


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def make_one_hot_encoder() -> OneHotEncoder:
    """Mantem compatibilidade com versoes recentes e antigas do scikit-learn."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def load_dataset() -> pd.DataFrame:
    if LOCAL_DATA_PATH.exists():
        print(f"Usando CSV local: {LOCAL_DATA_PATH}")
        return pd.read_csv(LOCAL_DATA_PATH)

    try:
        import kagglehub
    except ImportError as exc:
        raise SystemExit(
            "O pacote kagglehub nao esta instalado. Instale com:\n"
            "  pip install kagglehub pandas scikit-learn"
        ) from exc

    print(f"CSV local nao encontrado. Baixando dataset via KaggleHub: {DATASET_ID}")
    dataset_dir = Path(kagglehub.dataset_download(DATASET_ID))
    csv_files = sorted(dataset_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Nenhum CSV encontrado em {dataset_dir}")

    print(f"Usando arquivo baixado: {csv_files[0]}")
    return pd.read_csv(csv_files[0])


def prepare_dataset(raw_data: pd.DataFrame) -> pd.DataFrame:
    data = raw_data.copy()
    raw_feature_columns = [
        column for column in FEATURE_COLUMNS if column not in DERIVED_DATE_FEATURES
    ]
    required_columns = set(raw_feature_columns + list(TARGETS) + ["purchase_date"])
    missing = sorted(required_columns - set(data.columns))
    if missing:
        raise ValueError(f"Colunas obrigatorias ausentes no dataset: {missing}")

    purchase_date = pd.to_datetime(data["purchase_date"], errors="coerce")
    data["purchase_month"] = purchase_date.dt.month
    data["purchase_day"] = purchase_date.dt.day
    data["purchase_dayofweek"] = purchase_date.dt.dayofweek

    data = data.drop(columns=[col for col in DROP_COLUMNS if col in data.columns])
    return data


def build_pipeline() -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", make_one_hot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    classifier = SGDClassifier(
        loss="log_loss",
        alpha=0.0001,
        max_iter=1000,
        tol=1e-3,
        early_stopping=True,
        validation_fraction=0.10,
        n_iter_no_change=5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessamento", preprocessor),
            ("classificador", classifier),
        ]
    )


def metrics_from_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    labels: list[Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_table = pd.DataFrame(
        cm,
        index=[f"real_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
    )

    rows = []
    total = cm.sum()
    for index, label in enumerate(labels):
        tp = cm[index, index]
        fn = cm[index, :].sum() - tp
        fp = cm[:, index].sum() - tp
        tn = total - tp - fn - fp

        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        class_accuracy = sensitivity

        rows.append(
            {
                "classe": label,
                "total_real": int(tp + fn),
                "acertos": int(tp),
                "acuracia_classe": class_accuracy,
                "sensibilidade": sensitivity,
                "especificidade": specificity,
            }
        )

    return cm_table, pd.DataFrame(rows)


def train_target_model(
    data: pd.DataFrame,
    target: str,
    description: str,
) -> dict[str, Any]:
    x = data[FEATURE_COLUMNS]
    y = data[target]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = build_pipeline()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    labels = sorted(y.unique().tolist())
    cm_table, class_metrics = metrics_from_confusion_matrix(y_test, y_pred, labels)

    summary = {
        "alvo": target,
        "descricao": description,
        "acuracia_global": accuracy_score(y_test, y_pred),
        "sensibilidade_macro": class_metrics["sensibilidade"].mean(),
        "especificidade_macro": class_metrics["especificidade"].mean(),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }

    report = classification_report(y_test, y_pred, zero_division=0)

    return {
        "target": target,
        "description": description,
        "model": model,
        "summary": summary,
        "confusion_matrix": cm_table,
        "class_metrics": class_metrics,
        "classification_report": report,
    }


def print_dataset_summary(data: pd.DataFrame) -> None:
    print_section("1) Carregamento e analise inicial")
    print(f"Amostras: {len(data)}")
    print(f"Features usadas: {FEATURE_COLUMNS}")
    print(f"Alvos: {list(TARGETS)}")
    print(f"Valores ausentes nas features: {int(data[FEATURE_COLUMNS].isna().sum().sum())}")

    for target, description in TARGETS.items():
        print(f"\nDistribuicao de {description} ({target}):")
        distribution = data[target].value_counts().sort_index()
        for label, count in distribution.items():
            percent = count / len(data) * 100
            print(f"  {label:<18} {count:6d} ({percent:5.2f}%)")


def print_pipeline_summary() -> None:
    print_section("2) Pipeline ate o treinamento")
    print("Fluxo executado para cada uma das tres indicacoes:")
    print("  1. Baixar/carregar o CSV do KaggleHub.")
    print("  2. Criar features de calendario a partir de purchase_date.")
    print("  3. Remover identificadores e alvos da matriz de entrada.")
    print("  4. Dividir treino/teste com stratify no alvo avaliado.")
    print("  5. Aplicar preprocessamento dentro do Pipeline:")
    print(f"     - numericas: imputacao pela mediana + StandardScaler")
    print(f"     - categoricas: imputacao pela moda + OneHotEncoder")
    print("  6. Treinar SGDClassifier(loss='log_loss') com saida probabilistica.")
    print("  7. Avaliar no conjunto de teste separado.")


def print_result(result: dict[str, Any]) -> None:
    target = result["target"]
    description = result["description"]
    summary = result["summary"]

    print_section(f"3) Resultados - {description} ({target})")
    print(f"Acuracia global       : {summary['acuracia_global']:.4f}")
    print(f"Sensibilidade macro   : {summary['sensibilidade_macro']:.4f}")
    print(f"Especificidade macro  : {summary['especificidade_macro']:.4f}")
    print(f"F1-score macro        : {summary['f1_macro']:.4f}")
    print(f"F1-score weighted     : {summary['f1_weighted']:.4f}")

    print("\nMatriz de confusao:")
    print(result["confusion_matrix"].to_string())

    print("\nAcuracia por classe, sensibilidade e especificidade:")
    class_metrics = result["class_metrics"].copy()
    for col in ["acuracia_classe", "sensibilidade", "especificidade"]:
        class_metrics[col] = class_metrics[col].map(lambda value: f"{value:.4f}")
    print(class_metrics.to_string(index=False))

    print("\nRelatorio de classificacao:")
    print(result["classification_report"])


def save_artifacts(
    results: list[dict[str, Any]],
    demo_sale: dict[str, Any],
    demo_expected: dict[str, Any],
) -> None:
    artifacts = {
        "modelos": {result["target"]: result["model"] for result in results},
        "descricoes_alvos": TARGETS,
        "feature_columns": FEATURE_COLUMNS,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "metricas": {result["target"]: result["summary"] for result in results},
        "demo_sale": demo_sale,
        "demo_expected": demo_expected,
        "dataset_id": DATASET_ID,
    }

    with MODEL_PATH.open("wb") as file:
        dump(artifacts, file)

    metrics = pd.DataFrame([result["summary"] for result in results])
    metrics.to_csv(METRICS_PATH, index=False)

    print_section("4) Artefatos salvos")
    print(f"Modelo/pipelines: {MODEL_PATH}")
    print(f"Metricas resumo : {METRICS_PATH}")


def select_demo_sale(
    data: pd.DataFrame,
    results: list[dict[str, Any]],
    max_rows: int = 5000,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Escolhe um exemplo claro para demonstrar a inferencia apos o treino."""
    candidates = data.head(max_rows).copy()
    x_candidates = candidates[FEATURE_COLUMNS]
    model_by_target = {result["target"]: result["model"] for result in results}

    best_index = candidates.index[0]
    best_tuple = (-1, -1.0)

    for index, row in candidates.iterrows():
        sale_df = x_candidates.loc[[index]]
        correct_count = 0
        confidence_sum = 0.0

        for target, model in model_by_target.items():
            probabilities = model.predict_proba(sale_df)[0]
            classes = model.named_steps["classificador"].classes_
            best_position = int(np.argmax(probabilities))
            predicted = classes[best_position]
            confidence_sum += float(probabilities[best_position])
            if predicted == row[target]:
                correct_count += 1

        current_tuple = (correct_count, confidence_sum)
        if current_tuple > best_tuple:
            best_tuple = current_tuple
            best_index = index

        if correct_count == len(TARGETS) and confidence_sum >= 1.50:
            break

    demo_row = data.loc[best_index]
    demo_sale = demo_row[FEATURE_COLUMNS].to_dict()
    demo_expected = {target: demo_row[target] for target in TARGETS}
    return demo_sale, demo_expected


def main() -> None:
    raw_data = load_dataset()
    data = prepare_dataset(raw_data)

    print_dataset_summary(data)
    print_pipeline_summary()

    results = []
    for target, description in TARGETS.items():
        print_section(f"Treinando modelo para {description}")
        result = train_target_model(data, target, description)
        results.append(result)
        print_result(result)

    demo_sale, demo_expected = select_demo_sale(data, results)

    save_artifacts(results, demo_sale, demo_expected)

    print_section("5) Funcionamento do sistema inteligente")
    demonstrar_inferencia(MODEL_PATH)


if __name__ == "__main__":
    main()
