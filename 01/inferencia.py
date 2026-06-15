"""
=============================================================================
SISTEMAS INTELIGENTES — Heart Failure: Inferência para Novo Paciente
=============================================================================
Carrega os artefatos persistidos (modelo + scaler) e emite um laudo de
risco para um paciente desconhecido, aplicando exatamente o mesmo pipeline
de pré-processamento usado no treinamento.
=============================================================================
"""

from pickle import load
import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES DE REFERÊNCIA (espelham ex01.py)
# ─────────────────────────────────────────────────────────────────────────────
COLUNAS_BINARIAS   = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']
COLUNAS_CONTINUAS  = ['age', 'creatinine_phosphokinase', 'ejection_fraction',
                      'platelets', 'serum_creatinine', 'serum_sodium', 'time']


def _validar_entrada(dados_brutos: dict, colunas_esperadas: list) -> None:
    """
    Verifica se o dicionário de entrada possui todas as colunas esperadas
    e se as variáveis binárias contêm apenas valores 0 ou 1.
    Lança ValueError com mensagem descritiva em caso de problema.
    """
    faltantes = [c for c in colunas_esperadas if c not in dados_brutos]
    if faltantes:
        raise ValueError(f"Campos obrigatórios ausentes: {faltantes}")

    for col in COLUNAS_BINARIAS:
        val = dados_brutos.get(col)
        if val not in (0, 1):
            raise ValueError(
                f"Campo binário '{col}' deve ser 0 ou 1, recebido: {val!r}"
            )


def infere_paciente(dados_brutos: dict) -> dict:
    """
    Recebe um dicionário com os dados clínicos de um paciente, aplica o
    mesmo pipeline de pré-processamento do treinamento (padronização das
    variáveis contínuas, alinhamento de colunas) e retorna o laudo de risco.

    Parâmetros
    ----------
    dados_brutos : dict
        Dicionário com as 12 variáveis clínicas do paciente.

    Retorno
    -------
    dict com:
        grupo_predito          — descrição textual do grupo de risco
        score_risco_obito      — probabilidade estimada de óbito (0.0 – 1.0)
        nivel_de_urgencia      — BAIXO / MÉDIO / ALTO
        probabilidades_completas — probabilidades para cada classe
        meta_estimador_utilizado — nome do modelo carregado
        features_mais_relevantes — top-5 features do modelo para interpretação
    """

    # ── 1. Carregar artefatos persistidos ────────────────────────────────────
    try:
        artefatos = load(open('melhor_modelo_cardiologia.pkl', 'rb'))
    except FileNotFoundError:
        return {"erro": "Arquivo 'melhor_modelo_cardiologia.pkl' não encontrado. "
                        "Execute ex01.py para treinar e salvar o modelo."}

    modelo            = artefatos['modelo']
    scaler            = artefatos['scaler']
    nome_modelo       = artefatos['nome_modelo']
    colunas_esperadas = artefatos['colunas_features']

    # ── 2. Validação da entrada ───────────────────────────────────────────────
    try:
        _validar_entrada(dados_brutos, colunas_esperadas)
    except ValueError as e:
        return {"erro": str(e)}

    # ── 3. Construção do DataFrame e alinhamento de colunas ──────────────────
    df_paciente = pd.DataFrame([dados_brutos])[colunas_esperadas]

    # ── 4. Aplicar o MESMO pré-processamento do treino ───────────────────────
    #    4a. Padronização das variáveis contínuas com o scaler já fitado
    #        IMPORTANTE: usa scaler.transform() — não fit_transform()
    #        O scaler foi fitado no treino; re-fitar aqui causaria data leakage.
    df_paciente[COLUNAS_CONTINUAS] = scaler.transform(df_paciente[COLUNAS_CONTINUAS])

    #    4b. Garantia de integridade das colunas binárias
    #        (proteção extra: o usuário pode ter passado float acidentalmente)
    for col in COLUNAS_BINARIAS:
        df_paciente[col] = int(round(df_paciente[col].iloc[0]))

    # ── 5. Inferência ─────────────────────────────────────────────────────────
    classe_predita  = modelo.predict(df_paciente)[0]
    probabilidades  = modelo.predict_proba(df_paciente)[0]  # [P_0, P_1]
    score_risco     = probabilidades[1]

    # ── 6. Categorização clínica do nível de risco ────────────────────────────
    if score_risco < 0.35:
        nivel_risco = "BAIXO"
    elif score_risco < 0.65:
        nivel_risco = "MÉDIO"
    else:
        nivel_risco = "ALTO"

    resultado_grupo = (
        "Grupo de Alto Risco (Similar a Pacientes com Evolução Crítica)"
        if classe_predita == 1
        else "Grupo de Baixo Risco / Estável"
    )

    # ── 7. Top-5 features mais relevantes (contexto clínico) ──────────────────
    importancias = pd.Series(
        modelo.feature_importances_,
        index=colunas_esperadas
    ).sort_values(ascending=False)
    top5 = importancias.head(5).to_dict()

    return {
        'grupo_predito'             : resultado_grupo,
        'score_risco_obito'         : round(float(score_risco), 4),
        'nivel_de_urgencia'         : nivel_risco,
        'probabilidades_completas'  : {
            'Sobrevivência/Estável (0)' : round(float(probabilidades[0]), 4),
            'Óbito/Evolução Crítica (1)': round(float(probabilidades[1]), 4),
        },
        'meta_estimador_utilizado'  : nome_modelo,
        'features_mais_relevantes'  : {k: round(v, 4) for k, v in top5.items()},
    }


# ─────────────────────────────────────────────────────────────────────────────
# DEMONSTRAÇÃO: inferência para três perfis de paciente
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    from pprint import pprint

    # ── Paciente A: indicadores críticos (alto risco esperado) ───────────────
    paciente_critico = {
        'age'                     : 72.0,
        'anaemia'                 : 1,
        'creatinine_phosphokinase': 900,
        'diabetes'                : 0,
        'ejection_fraction'       : 22,   # Severo — normal ≥ 55%
        'high_blood_pressure'     : 1,
        'platelets'               : 324000.0,
        'serum_creatinine'        : 2.4,  # Função renal comprometida
        'serum_sodium'            : 132,  # Hiponatremia
        'sex'                     : 1,
        'smoking'                 : 1,
        'time'                    : 6
    }

    # ── Paciente B: perfil estável (baixo risco esperado) ────────────────────
    paciente_estavel = {
        'age'                     : 50.0,
        'anaemia'                 : 0,
        'creatinine_phosphokinase': 200,
        'diabetes'                : 0,
        'ejection_fraction'       : 60,   # Normal
        'high_blood_pressure'     : 0,
        'platelets'               : 260000.0,
        'serum_creatinine'        : 0.9,  # Função renal normal
        'serum_sodium'            : 138,
        'sex'                     : 0,
        'smoking'                 : 0,
        'time'                    : 200
    }

    # ── Paciente C: perfil limítrofe (risco intermediário) ───────────────────
    paciente_intermediario = {
        'age'                     : 65.0,
        'anaemia'                 : 1,
        'creatinine_phosphokinase': 400,
        'diabetes'                : 1,
        'ejection_fraction'       : 35,   # Moderadamente reduzido
        'high_blood_pressure'     : 1,
        'platelets'               : 230000.0,
        'serum_creatinine'        : 1.6,
        'serum_sodium'            : 135,
        'sex'                     : 1,
        'smoking'                 : 0,
        'time'                    : 90
    }

    pacientes = [
        ("A — Perfil Crítico",        paciente_critico),
        ("B — Perfil Estável",         paciente_estavel),
        ("C — Perfil Intermediário",   paciente_intermediario),
    ]

    for nome, paciente in pacientes:
        print("\n" + "=" * 60)
        print(f"  LAUDO — Paciente {nome}")
        print("=" * 60)
        print("Dados de entrada:")
        for k, v in paciente.items():
            print(f"  {k:<30} = {v}")
        print("\nResultado da inferência:")
        laudo = infere_paciente(paciente)
        pprint(laudo, sort_dicts=False)

    # ── Demonstração de validação de entrada (campo binário inválido) ─────────
    print("\n" + "=" * 60)
    print("  TESTE DE VALIDAÇÃO — entrada inválida")
    print("=" * 60)
    paciente_invalido = dict(paciente_critico)
    paciente_invalido['anaemia'] = 3    # valor inválido propositalmente
    resultado_erro = infere_paciente(paciente_invalido)
    print("Resultado com campo binário inválido (anaemia=3):")
    pprint(resultado_erro)
