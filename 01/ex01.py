"""
=============================================================================
SISTEMAS INTELIGENTES — Heart Failure: Treinamento do Modelo Preditivo
=============================================================================

METAESTIMADOR ESCOLHIDO: Random Forest Classifier (Ensemble de Árvores)

JUSTIFICATIVA TÉCNICA:
  1. Robustez a outliers: variáveis como creatinine_phosphokinase (23–7861)
     e serum_creatinine (0.5–9.4) possuem escalas muito díspares e outliers
     clínicos relevantes. O Random Forest, por ser baseado em divisões por
     limiar (não distância), é naturalmente robusto a isso — dispensando
     normalização obrigatória nas features, ao contrário de SVM ou KNN.

  2. Tratamento nativo de dados mistos: a base combina variáveis binárias
     puras (anaemia, diabetes, high_blood_pressure, sex, smoking) com
     contínuas (age, creatinine_phosphokinase, platelets…). O RF opera
     diretamente nessa mistura sem necessidade de One-Hot Encoding.

  3. Importância de features interpretável: em contexto clínico, saber
     quais variáveis mais contribuem para o risco é fundamental para
     o médico. O RF fornece feature_importances_ nativamente.

  4. Resiliência ao desbalanceamento com SMOTE: o ensemble de múltiplas
     árvores, após balanceamento, generaliza melhor do que um estimador
     único (ex: árvore de decisão simples), reduzindo overfitting.

  5. Otimização via RandomizedSearchCV + StratifiedKFold: garante
     avaliação honesta mesmo com N=299 amostras (dataset pequeno),
     usando F1-Score como métrica de otimização — equilíbrio entre
     precisão e recall para o evento raro (DEATH_EVENT=1, ~32% dos casos).
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0. IMPORTAÇÕES
# ─────────────────────────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score,
                             roc_auc_score, f1_score, classification_report)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from collections import Counter
from pickle import dump
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# 1. CARREGAMENTO E ANÁLISE EXPLORATÓRIA DOS DADOS
# ─────────────────────────────────────────────────────────────────────────────
dados = pd.read_csv('heart_failure_clinical_records_dataset.csv')

COLUNAS_BINARIAS   = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']
COLUNAS_CONTINUAS  = ['age', 'creatinine_phosphokinase', 'ejection_fraction',
                      'platelets', 'serum_creatinine', 'serum_sodium', 'time']

print("=" * 60)
print("  ANÁLISE EXPLORATÓRIA DOS DADOS")
print("=" * 60)
print(f"\nDimensões do dataset: {dados.shape[0]} pacientes x {dados.shape[1]} variáveis")
print(f"\nValores ausentes por coluna:\n{dados.isnull().sum()}")
print(f"\n{'─'*40}")
print("Distribuição original das classes:")
contagem = Counter(dados['DEATH_EVENT'])
print(f"  Sobrevivência (0): {contagem[0]} pacientes ({contagem[0]/len(dados)*100:.1f}%)")
print(f"  Óbito        (1): {contagem[1]} pacientes ({contagem[1]/len(dados)*100:.1f}%)")
print(f"\nRazão de desbalanceamento: 1 : {contagem[0]/contagem[1]:.2f}")

print(f"\n{'─'*40}")
print("Variáveis binárias — verificação de valores únicos:")
for col in COLUNAS_BINARIAS:
    print(f"  {col}: {sorted(dados[col].unique())} ✓")

print(f"\n{'─'*40}")
print("Variáveis contínuas — estatísticas descritivas:")
print(dados[COLUNAS_CONTINUAS].describe().round(2).to_string())

# Detecção de outliers com IQR (apenas informativo — não removidos pois
# valores extremos em creatinina/CPK são clinicamente válidos e relevantes)
print(f"\n{'─'*40}")
print("Outliers detectados (IQR × 1.5) — mantidos por relevância clínica:")
for col in COLUNAS_CONTINUAS:
    q1, q3 = dados[col].quantile(0.25), dados[col].quantile(0.75)
    iqr = q3 - q1
    n_out = ((dados[col] < q1 - 1.5*iqr) | (dados[col] > q3 + 1.5*iqr)).sum()
    if n_out > 0:
        print(f"  {col}: {n_out} outlier(s)")

# ─────────────────────────────────────────────────────────────────────────────
# 2. SEPARAÇÃO EM TREINO E TESTE (Evita Data Leakage)
#    stratify= garante a mesma proporção de classes (≈68%/32%) em ambos
#    os conjuntos, essencial para datasets desbalanceados.
# ─────────────────────────────────────────────────────────────────────────────
dados_atributos = dados.drop(columns=['DEATH_EVENT'])
dados_classe    = dados['DEATH_EVENT']

atributos_train, atributos_teste, classe_train, classe_teste = train_test_split(
    dados_atributos,
    dados_classe,
    test_size=0.25,
    random_state=42,        # Semente fixa — reprodutibilidade garantida
    stratify=dados_classe   # Mantém proporção clínica nos dois conjuntos
)

print(f"\n{'─'*40}")
print(f"Split treino/teste: {len(atributos_train)} / {len(atributos_teste)} amostras")

# ─────────────────────────────────────────────────────────────────────────────
# 3. PRÉ-PROCESSAMENTO — ETAPAS APLICADAS APENAS NO TREINO
#
#    Princípio fundamental: o conjunto de teste simula dados inéditos do mundo
#    real. Todo pré-processamento que "aprende" algo dos dados (scaler, SMOTE)
#    deve ser FITADO exclusivamente no treino e apenas TRANSFORMADO no teste.
# ─────────────────────────────────────────────────────────────────────────────

# 3a. PADRONIZAÇÃO DAS VARIÁVEIS CONTÍNUAS
#     Embora o Random Forest seja invariante à escala (splits por limiar),
#     a padronização é incluída como boa prática de pipeline reproducível
#     e para que os artefatos exportados possam ser reusados com outros
#     estimadores sensíveis à escala (SVM, KNN, Regressão Logística).
#     Os dados binários NÃO são padronizados — escala 0/1 já é adequada.
scaler = StandardScaler()

# Fit exclusivamente no treino → aprende média e desvio do treino
atributos_train[COLUNAS_CONTINUAS] = scaler.fit_transform(
    atributos_train[COLUNAS_CONTINUAS]
)
# Transform no teste → aplica os parâmetros aprendidos no treino (sem re-fit)
atributos_teste[COLUNAS_CONTINUAS] = scaler.transform(
    atributos_teste[COLUNAS_CONTINUAS]
)

print(f"\n{'─'*40}")
print("Padronização (StandardScaler) aplicada nas variáveis contínuas:")
print("  Médias aprendidas no treino:", dict(zip(COLUNAS_CONTINUAS,
      scaler.mean_.round(2))))

# 3b. BALANCEAMENTO COM SMOTE (Synthetic Minority Over-sampling Technique)
#     Aplicado SOMENTE no treino — jamais no teste.
#     O SMOTE gera amostras sintéticas interpolando vizinhos da classe
#     minoritária (DEATH_EVENT=1) no espaço das features. Como opera por
#     interpolação contínua, variáveis binárias (0/1) recebem valores
#     intermediários (ex: 0.73) — por isso a correção no passo 3c é obrigatória.
resampler = SMOTE(random_state=42)
atributos_train_b, classe_train_b = resampler.fit_resample(atributos_train, classe_train)

# 3c. CORREÇÃO CRÍTICA DAS COLUNAS BINÁRIAS PÓS-SMOTE
#     O SMOTE interpola continuamente entre vizinhos, podendo gerar valores
#     como anaemia=0.73 — que não possui significado clínico (anemia é
#     presente/ausente). O arredondamento restaura a semântica binária.
for col in COLUNAS_BINARIAS:
    atributos_train_b[col] = atributos_train_b[col].round().astype(int)

print(f"\n{'─'*40}")
print("Balanceamento SMOTE + correção das colunas binárias:")
contagem_b = Counter(classe_train_b)
print(f"  Após SMOTE — Sobrevivência (0): {contagem_b[0]}  |  Óbito (1): {contagem_b[1]}")

# Verificação da integridade binária pós-correção
for col in COLUNAS_BINARIAS:
    vals = sorted(atributos_train_b[col].unique())
    assert vals == [0, 1], f"ERRO: coluna {col} tem valores inválidos: {vals}"
print("  Integridade das colunas binárias verificada ✓")

# ─────────────────────────────────────────────────────────────────────────────
# 4. OTIMIZAÇÃO DE HIPERPARÂMETROS DO METAESTIMADOR
#    RandomizedSearchCV + StratifiedKFold (5 folds):
#    - Amostra aleatoriamente 20 combinações do espaço de hiperparâmetros
#    - Cada combinação é avaliada em 5 folds estratificados (preserva
#      proporção de classes em cada fold, crítico para dataset pequeno)
#    - Métrica de otimização: F1-Score — penaliza igualmente falsos positivos
#      e negativos; ideal para contexto clínico com desbalanceamento residual
# ─────────────────────────────────────────────────────────────────────────────
rf_base = RandomForestClassifier(random_state=42)

param_dist = {
    'n_estimators':     [50, 100, 200, 300],
    'max_depth':        [None, 5, 10, 15],
    'min_samples_split':[2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion':        ['gini', 'entropy']
}

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
random_search = RandomizedSearchCV(
    estimator=rf_base,
    param_distributions=param_dist,
    n_iter=20,
    scoring='f1',
    cv=cv_strategy,
    random_state=42,
    n_jobs=-1
)

print(f"\n{'─'*40}")
print("Iniciando busca de hiperparâmetros (RandomizedSearchCV)...")
random_search.fit(atributos_train_b, classe_train_b)
melhor_modelo = random_search.best_estimator_
print(f"Busca concluída. Melhores parâmetros: {random_search.best_params_}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. AVALIAÇÃO NO CONJUNTO DE TESTE (Dados Inéditos)
# ─────────────────────────────────────────────────────────────────────────────
preditos = melhor_modelo.predict(atributos_teste)
probas   = melhor_modelo.predict_proba(atributos_teste)[:, 1]

acuracia     = accuracy_score(classe_teste, preditos)
f1           = f1_score(classe_teste, preditos)
auc          = roc_auc_score(classe_teste, probas)
tn, fp, fn, tp = confusion_matrix(classe_teste, preditos).ravel()
sensibilidade  = tp / (tp + fn) if (tp + fn) > 0 else 0
especificidade = tn / (tn + fp) if (tn + fp) > 0 else 0

print("\n" + "=" * 60)
print("  RELATÓRIO DE DESEMPENHO — CONJUNTO DE TESTE")
print("=" * 60)
print(f"  Melhores Parâmetros : {random_search.best_params_}")
print(f"  Acurácia Geral      : {acuracia:.4f}")
print(f"  F1-Score            : {f1:.4f}")
print(f"  ROC-AUC             : {auc:.4f}")
print(f"  Sensibilidade (VP)  : {sensibilidade:.4f}  ← detectar óbito")
print(f"  Especificidade (VN) : {especificidade:.4f}  ← detectar sobrevivência")
print(f"\n  Matriz de Confusão:")
print(f"            Predito 0   Predito 1")
print(f"  Real  0 :    {tn:>4}        {fp:>4}    (VN / FP)")
print(f"  Real  1 :    {fn:>4}        {tp:>4}    (FN / VP)")
print()
print("  Relatório detalhado por classe:")
print(classification_report(classe_teste, preditos,
      target_names=['Sobrevivência (0)', 'Óbito (1)']))

# Importância das features
importancias = pd.Series(
    melhor_modelo.feature_importances_,
    index=dados_atributos.columns
).sort_values(ascending=False)
print("  Importância das features (Random Forest):")
for feat, imp in importancias.items():
    bar = '█' * int(imp * 40)
    print(f"  {feat:<30} {imp:.4f}  {bar}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. EXPORTAÇÃO DOS ARTEFATOS
#    O scaler é salvo junto ao modelo para que a inferência aplique
#    exatamente a mesma transformação aprendida no treino.
# ─────────────────────────────────────────────────────────────────────────────
artefatos = {
    'modelo'          : melhor_modelo,
    'scaler'          : scaler,
    'nome_modelo'     : 'Random Forest Classifier (Ensemble)',
    'colunas_features': dados_atributos.columns.tolist(),
    'colunas_binarias': COLUNAS_BINARIAS,
    'colunas_continuas': COLUNAS_CONTINUAS,
}

with open('melhor_modelo_cardiologia.pkl', 'wb') as f:
    dump(artefatos, f)

print("\n[INFO] Artefatos salvos com sucesso em 'melhor_modelo_cardiologia.pkl'.")
print("       Conteúdo salvo: modelo, scaler, nomes das features e metadados.")
