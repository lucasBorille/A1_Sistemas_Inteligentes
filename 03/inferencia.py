"""
Modulo de inferencia do Exercicio 03.

Carrega o artefato treinado por ex03.py e retorna, para uma circunstancia
de venda, as tres indicacoes exigidas pela tarefa com grau de certeza:

- product_category
- payment_method
- age_group
"""

from __future__ import annotations

from pathlib import Path
from pickle import load
from pprint import pprint
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = SCRIPT_DIR / "modelo_retail_black_friday.pkl"


def _load_artifacts(model_path: Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Artefato nao encontrado: {model_path}. Execute ex03.py primeiro."
        )

    with model_path.open("rb") as file:
        return load(file)


def _validate_sale(sale: dict[str, Any], expected_columns: list[str]) -> None:
    missing = [column for column in expected_columns if column not in sale]
    if missing:
        raise ValueError(f"Campos obrigatorios ausentes: {missing}")


def _top_probabilities(model: Any, sale_df: pd.DataFrame, limit: int = 3) -> list[dict]:
    probabilities = model.predict_proba(sale_df)[0]
    classes = model.named_steps["classificador"].classes_
    ordered_indexes = probabilities.argsort()[::-1][:limit]

    return [
        {
            "classe": str(classes[index]),
            "score": round(float(probabilities[index]), 4),
        }
        for index in ordered_indexes
    ]


def inferir_venda(
    sale: dict[str, Any],
    model_path: Path = DEFAULT_MODEL_PATH,
) -> dict[str, Any]:
    artifacts = _load_artifacts(model_path)
    feature_columns = artifacts["feature_columns"]
    _validate_sale(sale, feature_columns)

    sale_df = pd.DataFrame([sale])[feature_columns]
    predictions = {}

    for target, model in artifacts["modelos"].items():
        predicted_class = model.predict(sale_df)[0]
        top_probabilities = _top_probabilities(model, sale_df)

        predictions[target] = {
            "indicacao": str(predicted_class),
            "grau_de_certeza": top_probabilities[0]["score"],
            "top_probabilidades": top_probabilities,
        }

    return predictions


def demonstrar_inferencia(model_path: Path = DEFAULT_MODEL_PATH) -> None:
    artifacts = _load_artifacts(model_path)
    sale = artifacts["demo_sale"]
    expected = artifacts.get("demo_expected", {})
    descriptions = artifacts["descricoes_alvos"]
    predictions = inferir_venda(sale, model_path)

    print("Circunstancia de venda usada na demonstracao:")
    pprint(sale, sort_dicts=False)

    if expected:
        print("\nValores reais da linha usada como exemplo:")
        pprint(expected, sort_dicts=False)

    print("\nIndicacoes do sistema inteligente:")
    for target, result in predictions.items():
        print(f"\n{descriptions[target].title()} ({target})")
        print(f"  Indicacao        : {result['indicacao']}")
        print(f"  Grau de certeza  : {result['grau_de_certeza']:.4f}")
        print("  Top probabilidades:")
        for item in result["top_probabilidades"]:
            print(f"    - {item['classe']:<18} {item['score']:.4f}")


if __name__ == "__main__":
    demonstrar_inferencia()
