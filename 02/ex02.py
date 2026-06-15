"""
Exercicio 02 - Wine Quality

Tarefa:
Avaliar ao menos tres metaestimadores classificadores e selecionar aquele
com melhor desempenho.

O script demonstra:
1. Pipeline ate o treinamento do modelo.
2. Acuracia global, acuracia por classe via matriz de confusao e f1-score.
3. Justificativa do modelo mais adequado para possivel implantacao.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from pickle import dump

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_PATH = SCRIPT_DIR / "winequality_combined.csv"
MODEL_PATH = SCRIPT_DIR / "melhor_modelo_vinho.pkl"
RESULTS_PATH = SCRIPT_DIR / "resultados_ex02.csv"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: object
    param_grid: dict[str, list[object]]


def make_one_hot_encoder() -> OneHotEncoder:
    """Mantem compatibilidade com versoes novas e antigas do scikit-learn."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def print_section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset nao encontrado: {DATASET_PATH}")

    data = pd.read_csv(DATASET_PATH)

    if "quality" not in data.columns:
        raise ValueError("A coluna alvo 'quality' nao foi encontrada no dataset.")

    x = data.drop(columns="quality")
    y = data["quality"]
    return x, y


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = x.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = x.select_dtypes(exclude=np.number).columns.tolist()

    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_columns),
            ("categorical", make_one_hot_encoder(), categorical_columns),
        ],
        remainder="drop",
    )


def build_model_specs() -> list[ModelSpec]:
    return [
        ModelSpec(
            name="Random Forest",
            estimator=RandomForestClassifier(
                random_state=RANDOM_STATE,
                class_weight="balanced",
                n_jobs=-1,
            ),
            param_grid={
                "classifier__n_estimators": [150, 250],
                "classifier__max_depth": [None, 12],
                "classifier__min_samples_leaf": [1, 2],
                "classifier__max_features": ["sqrt"],
            },
        ),
        ModelSpec(
            name="Extra Trees",
            estimator=ExtraTreesClassifier(
                random_state=RANDOM_STATE,
                class_weight="balanced",
                n_jobs=-1,
            ),
            param_grid={
                "classifier__n_estimators": [150, 250],
                "classifier__max_depth": [None, 12],
                "classifier__min_samples_leaf": [1, 2],
                "classifier__max_features": ["sqrt"],
            },
        ),
        ModelSpec(
            name="Gradient Boosting",
            estimator=GradientBoostingClassifier(random_state=RANDOM_STATE),
            param_grid={
                "classifier__n_estimators": [100, 150],
                "classifier__learning_rate": [0.05, 0.10],
                "classifier__max_depth": [2, 3],
            },
        ),
        ModelSpec(
            name="AdaBoost",
            estimator=AdaBoostClassifier(random_state=RANDOM_STATE),
            param_grid={
                "classifier__n_estimators": [100, 200],
                "classifier__learning_rate": [0.50, 1.00],
            },
        ),
    ]


def class_accuracy_table(
    y_true: pd.Series,
    y_pred: np.ndarray,
    labels: list[int],
) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    totals = cm.sum(axis=1)
    correct = np.diag(cm)
    class_accuracy = np.divide(
        correct,
        totals,
        out=np.zeros_like(correct, dtype=float),
        where=totals != 0,
    )

    return pd.DataFrame(
        {
            "classe_quality": labels,
            "total_real": totals,
            "acertos": correct,
            "acuracia_classe": class_accuracy,
        }
    )


def evaluate_model(
    name: str,
    model: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    labels: list[int],
    best_cv_score: float,
    best_params: dict[str, object],
) -> dict[str, object]:
    y_pred = model.predict(x_test)
    report = classification_report(
        y_test,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    class_acc = class_accuracy_table(y_test, y_pred, labels)

    return {
        "nome": name,
        "modelo": model,
        "predicoes": y_pred,
        "matriz_confusao": cm,
        "acuracia_global": accuracy_score(y_test, y_pred),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "relatorio": report,
        "acuracia_por_classe": class_acc,
        "melhor_f1_cv": best_cv_score,
        "melhores_parametros": best_params,
    }


def train_and_evaluate_models(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    labels: list[int],
) -> list[dict[str, object]]:
    preprocessor = build_preprocessor(x_train)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    results: list[dict[str, object]] = []

    print_section("3) Treinamento e avaliacao dos metaestimadores")

    for spec in build_model_specs():
        print(f"\nTreinando {spec.name}...")
        pipeline = Pipeline(
            steps=[
                ("preprocessamento", preprocessor),
                ("classifier", spec.estimator),
            ]
        )

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=spec.param_grid,
            scoring="f1_weighted",
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
        search.fit(x_train, y_train)

        result = evaluate_model(
            name=spec.name,
            model=search.best_estimator_,
            x_test=x_test,
            y_test=y_test,
            labels=labels,
            best_cv_score=search.best_score_,
            best_params=search.best_params_,
        )
        results.append(result)

        print(f"  Melhor F1 weighted na validacao cruzada: {search.best_score_:.4f}")
        print(f"  Acuracia global no teste: {result['acuracia_global']:.4f}")
        print(f"  F1 weighted no teste: {result['f1_weighted']:.4f}")

    return results


def print_dataset_summary(x: pd.DataFrame, y: pd.Series) -> None:
    print_section("1) Carregamento e analise inicial")
    print(f"Arquivo: {DATASET_PATH.name}")
    print(f"Amostras: {len(x)}")
    print(f"Atributos de entrada: {x.shape[1]}")
    print(f"Valores ausentes: {int(x.isna().sum().sum() + y.isna().sum())}")
    print("\nDistribuicao da classe quality:")
    distribution = y.value_counts().sort_index()
    for quality, count in distribution.items():
        percent = count / len(y) * 100
        print(f"  Classe {quality}: {count:4d} amostras ({percent:5.2f}%)")


def print_pipeline_summary(x_train: pd.DataFrame, x_test: pd.DataFrame) -> None:
    numeric_columns = x_train.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = x_train.select_dtypes(exclude=np.number).columns.tolist()

    print_section("2) Pipeline ate o treinamento")
    print("Fluxo executado:")
    print("  1. Ler o CSV winequality_combined.csv.")
    print("  2. Separar X (atributos) e y (quality).")
    print("  3. Dividir treino/teste com stratify=y para preservar as classes.")
    print("  4. Aplicar preprocessamento dentro do Pipeline:")
    print(f"     - StandardScaler nas colunas numericas: {numeric_columns}")
    print(f"     - OneHotEncoder nas colunas categoricas: {categorical_columns}")
    print("  5. Treinar cada metaestimador com GridSearchCV e StratifiedKFold.")
    print("  6. Avaliar todos no mesmo conjunto de teste, mantido fora do treino.")
    print(f"\nTamanho do treino: {len(x_train)} amostras")
    print(f"Tamanho do teste : {len(x_test)} amostras")


def print_detailed_results(results: list[dict[str, object]], labels: list[int]) -> None:
    print_section("4) Resultados comparativos")
    summary = pd.DataFrame(
        [
            {
                "modelo": result["nome"],
                "f1_weighted_cv": result["melhor_f1_cv"],
                "acuracia_global_teste": result["acuracia_global"],
                "f1_weighted_teste": result["f1_weighted"],
                "f1_macro_teste": result["f1_macro"],
            }
            for result in results
        ]
    ).sort_values(
        by=["f1_weighted_teste", "acuracia_global_teste", "f1_macro_teste"],
        ascending=False,
    )
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    for result in results:
        print_section(f"5) Matriz de confusao e metricas por classe - {result['nome']}")
        cm = pd.DataFrame(
            result["matriz_confusao"],
            index=[f"real_{label}" for label in labels],
            columns=[f"pred_{label}" for label in labels],
        )
        print(cm.to_string())

        print("\nAcuracia por classe calculada pela matriz de confusao:")
        class_acc = result["acuracia_por_classe"].copy()
        class_acc["acuracia_classe"] = class_acc["acuracia_classe"].map(
            lambda value: f"{value:.4f}"
        )
        print(class_acc.to_string(index=False))


def print_classification_reports(
    results: list[dict[str, object]],
    y_test: pd.Series,
    labels: list[int],
) -> None:
    for result in results:
        print_section(f"Relatorio detalhado - {result['nome']}")
        print(
            classification_report(
                y_test,
                result["predicoes"],
                labels=labels,
                zero_division=0,
            )
        )


def choose_best_model(results: list[dict[str, object]]) -> dict[str, object]:
    return max(
        results,
        key=lambda result: (
            result["f1_weighted"],
            result["acuracia_global"],
            result["f1_macro"],
        ),
    )


def print_deployment_justification(best: dict[str, object]) -> None:
    print_section("6) Modelo selecionado e justificativa para implantacao")
    print(f"Modelo selecionado: {best['nome']}")
    print(f"Acuracia global no teste: {best['acuracia_global']:.4f}")
    print(f"F1-score weighted no teste: {best['f1_weighted']:.4f}")
    print(f"F1-score macro no teste: {best['f1_macro']:.4f}")
    print(f"Melhor F1 weighted na validacao cruzada: {best['melhor_f1_cv']:.4f}")
    print(f"Melhores parametros: {best['melhores_parametros']}")

    print("\nJustificativa:")
    print(
        "  O modelo foi escolhido porque obteve o maior F1-score weighted no "
        "conjunto de teste, usando a acuracia global e o F1 macro como "
        "criterios de desempate. O F1 weighted e adequado aqui porque o dataset "
        "e multiclasse e desbalanceado: as notas 5 e 6 concentram a maior parte "
        "das amostras, enquanto notas extremas possuem pouco suporte."
    )
    print(
        "  Para uma possivel implantacao, o artefato salvo contem o pipeline "
        "completo: preprocessamento e classificador. Isso reduz risco de "
        "diferenca entre treinamento e inferencia, pois novas amostras passam "
        "pelas mesmas transformacoes aprendidas no treino."
    )


def save_outputs(best: dict[str, object], results: list[dict[str, object]]) -> None:
    with MODEL_PATH.open("wb") as file:
        dump(
            {
                "modelo_pipeline": best["modelo"],
                "nome_modelo": best["nome"],
                "metricas": {
                    "acuracia_global": best["acuracia_global"],
                    "f1_weighted": best["f1_weighted"],
                    "f1_macro": best["f1_macro"],
                    "f1_weighted_cv": best["melhor_f1_cv"],
                },
                "melhores_parametros": best["melhores_parametros"],
            },
            file,
        )

    summary = pd.DataFrame(
        [
            {
                "modelo": result["nome"],
                "f1_weighted_cv": result["melhor_f1_cv"],
                "acuracia_global_teste": result["acuracia_global"],
                "f1_weighted_teste": result["f1_weighted"],
                "f1_macro_teste": result["f1_macro"],
            }
            for result in results
        ]
    )
    summary.to_csv(RESULTS_PATH, index=False)

    print(f"\nArtefato do melhor modelo salvo em: {MODEL_PATH}")
    print(f"Resumo das metricas salvo em: {RESULTS_PATH}")


def main() -> None:
    x, y = load_dataset()
    labels = sorted(y.unique().tolist())

    print_dataset_summary(x, y)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print_pipeline_summary(x_train, x_test)

    results = train_and_evaluate_models(x_train, x_test, y_train, y_test, labels)

    print_detailed_results(results, labels)
    print_classification_reports(results, y_test, labels)

    best = choose_best_model(results)
    print_deployment_justification(best)
    save_outputs(best, results)


if __name__ == "__main__":
    main()
