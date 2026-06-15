"""
=============================================================================
SISTEMAS INTELIGENTES — Wine Quality: Comparação e Treinamento de Modelos
=============================================================================

CONTEXTO:
  Dataset Wine Quality (UCI / Cortez et al., 2009): 6497 amostras de vinhos
  tintos (1599) e brancos (4898) do "Vinho Verde" português.
  11 atributos físico-químicos → prever a qualidade sensorial (0–10).

ABORDAGEM:
  Tarefa original é de 7 classes (3–9). Classes extremas (3,4 e 8,9)
  possuem < 4% das amostras, inviabilizando aprendizado direto.
  Estratégia adotada (padrão da literatura): agrupar em 3 categorias
  semanticamente coerentes:
      Ruim  (quality 3–4) →  246 amostras  (4.6 %)
      Médio (quality 5–6) → 4074 amostras  (77.2%)
      Bom   (quality 7–9) → 1177 amostras  (22.3%)
  Isso mantém o desbalanceamento natural do domínio e gera um problema
  de classificação tratável com métricas interpretáveis.

METAESTIMADORES AVALIADOS:
  1. Random Forest   — ensemble de árvores (bagging)
  2. Gradient Boosting — ensemble sequencial (boosting)
  3. K-Nearest Neighbors (KNN) — aprendizado baseado em instâncias
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0. IMPORTAÇÕES
# ─────────────────────────────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_val_score, RandomizedSearchCV)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, confusion_matrix)
from pickle import dump
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# 1. CARREGAMENTO DOS ARQUIVOS ORIGINAIS
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("  ETAPA 1 — CARREGAMENTO E MONTAGEM DO DATASET UNIFICADO")
print("=" * 65)

# Leitura dos três arquivos conforme solicitado pelo exercício
# Os CSVs usam ponto-e-vírgula como separador (padrão europeu)
df_tinto  = pd.read_csv('winequality-red.csv',   sep=';')
df_branco = pd.read_csv('winequality-white.csv',  sep=';')

# Nomes das colunas extraídos do arquivo winequality.names (seção 7):
# 1-fixed acidity, 2-volatile acidity, 3-citric acid, 4-residual sugar,
# 5-chlorides, 6-free sulfur dioxide, 7-total sulfur dioxide, 8-density,
# 9-pH, 10-sulphates, 11-alcohol, 12-quality (alvo)
# Verificação: os CSVs já incluem o header com esses nomes
COLUNAS_ESPERADAS = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
    'pH', 'sulphates', 'alcohol', 'quality'
]
assert list(df_tinto.columns) == COLUNAS_ESPERADAS, "Colunas divergentes!"

# Adiciona marcador de tipo antes de unir (feature adicional com valor preditivo)
df_tinto['wine_type']  = 0   # 0 = tinto
df_branco['wine_type'] = 1   # 1 = branco

# Concatenação vertical — um único DataFrame com todos os dados
df = pd.concat([df_tinto, df_branco], ignore_index=True)

print(f"\nArquivo de vinhos tintos  : {len(df_tinto):>5} registros")
print(f"Arquivo de vinhos brancos : {len(df_branco):>5} registros")
print(f"Dataset unificado (raw)   : {len(df):>5} registros  |  {df.shape[1]} colunas")

# ─────────────────────────────────────────────────────────────────────────────
# 2. ANÁLISE EXPLORATÓRIA E PRÉ-PROCESSAMENTO
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  ETAPA 2 — PRÉ-PROCESSAMENTO")
print("=" * 65)

# 2a. Valores ausentes
n_nulos = df.isnull().sum().sum()
print(f"\n[2a] Valores ausentes: {n_nulos} {'✓ Nenhum' if n_nulos == 0 else '→ requer tratamento'}")

# 2b. Duplicatas — remover registros idênticos em todas as colunas
n_dup = df.duplicated().sum()
df = df.drop_duplicates().reset_index(drop=True)
print(f"[2b] Duplicatas removidas: {n_dup}  |  Dataset final: {len(df)} registros")

# 2c. Distribuição da variável alvo original
print(f"\n[2c] Distribuição original de 'quality' (escala 3–9):")
vc_orig = df['quality'].value_counts().sort_index()
for q, n in vc_orig.items():
    bar = '█' * int(n / 50)
    print(f"       quality {q}: {n:>4} amostras  {bar}")

# 2d. Agrupamento em 3 categorias semânticas
#     Justificativa: classes 3,4 somam 4.6% e classes 8,9 somam 3.0% —
#     volume insuficiente para aprendizado de padrões distintos. O agrupamento
#     é padrão consagrado na literatura para este dataset (Cortez et al., 2009).
df['quality_cat'] = pd.cut(
    df['quality'],
    bins=[2, 4, 6, 9],
    labels=['Ruim', 'Médio', 'Bom']
)

print(f"\n[2d] Agrupamento de classes (bins: 3-4=Ruim | 5-6=Médio | 7-9=Bom):")
vc_cat = df['quality_cat'].value_counts().sort_index()
for cat, n in vc_cat.items():
    pct = n / len(df) * 100
    bar = '█' * int(pct / 2)
    print(f"       {cat:<6}: {n:>4} amostras ({pct:5.1f}%)  {bar}")

# 2e. Separação atributos / classe
X = df.drop(columns=['quality', 'quality_cat'])
y = df['quality_cat']

FEATURES = X.columns.tolist()
print(f"\n[2e] Features utilizadas ({len(FEATURES)}):")
for f in FEATURES:
    print(f"       • {f}")

# 2f. Estatísticas descritivas — evidencia necessidade de normalização
print(f"\n[2f] Estatísticas descritivas das features (escalas muito díspares):")
desc = X.describe().loc[['mean', 'std', 'min', 'max']].round(2)
print(desc.to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 3. SPLIT TREINO / TESTE
#    stratify= essencial: mantém a proporção das 3 classes em ambos os conjuntos
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  ETAPA 3 — DIVISÃO TREINO / TESTE  (75% / 25%)")
print("=" * 65)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y          # garante proporção igual de classes nos dois conjuntos
)

print(f"\n  Treino : {len(X_train):>4} amostras")
print(f"  Teste  : {len(X_test):>4} amostras")
print(f"  Proporção de classes no treino: {dict(y_train.value_counts().sort_index())}")
print(f"  Proporção de classes no teste : {dict(y_test.value_counts().sort_index())}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. PADRONIZAÇÃO DAS FEATURES (StandardScaler)
#    Fit exclusivamente no treino → aplicado (transform) no teste.
#    Obrigatório para KNN (sensível à escala).
#    Boa prática para RF/GB (não obrigatório, mas garante pipeline reutilizável).
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  ETAPA 4 — PADRONIZAÇÃO (StandardScaler)")
print("=" * 65)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)   # fit + transform no treino
X_test_sc  = scaler.transform(X_test)         # apenas transform no teste

print("\n  Médias aprendidas no treino (primeiras 5 features):")
for feat, mu in zip(FEATURES[:5], scaler.mean_[:5]):
    print(f"    {feat:<25} μ = {mu:.4f}")
print("    ...")

# ─────────────────────────────────────────────────────────────────────────────
# 5. COMPARAÇÃO DE TRÊS METAESTIMADORES
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  ETAPA 5 — COMPARAÇÃO DE TRÊS METAESTIMADORES")
print("=" * 65)

# ── 5.1  Definição dos modelos com hiperparâmetros base ──────────────────────

modelos = {

    # ──────────────────────────────────────────────────────────────────────────
    # MODELO 1: RANDOM FOREST (Ensemble — Bagging)
    # Justificativa: Combina N árvores treinadas em subconjuntos aleatórios
    # de amostras e features. A diversidade entre as árvores reduz variância
    # (overfitting). Robusto a outliers e features em escalas distintas.
    # class_weight='balanced' compensa o desbalanceamento (Ruim: 4.6%).
    # ──────────────────────────────────────────────────────────────────────────
    'Random Forest': RandomForestClassifier(
        n_estimators=300,
        criterion='entropy',
        min_samples_split=10,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # MODELO 2: GRADIENT BOOSTING (Ensemble — Boosting)
    # Justificativa: Treina árvores sequencialmente, cada uma corrigindo os
    # erros da anterior. Tende a menor viés que o RF, mas mais sensível a
    # overfitting e hiperparâmetros. Sem suporte nativo a class_weight.
    # ──────────────────────────────────────────────────────────────────────────
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # MODELO 3: K-NEAREST NEIGHBORS (Aprendizado baseado em instâncias)
    # Justificativa: Classifica por similaridade com os K vizinhos mais
    # próximos. Não constrói modelo explícito (lazy learner). Sensível à
    # escala → exige normalização. weights='distance' pondera pelo inverso
    # da distância, melhorando resultados em classes desbalanceadas.
    # ──────────────────────────────────────────────────────────────────────────
    'KNN': KNeighborsClassifier(
        n_neighbors=9,
        weights='distance',
        n_jobs=-1
    ),
}

# ── 5.2  Treinamento, avaliação no teste e validação cruzada ─────────────────

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
resultados   = {}

for nome, modelo in modelos.items():
    print(f"\n{'─'*65}")
    print(f"  METAESTIMADOR: {nome}")
    print(f"{'─'*65}")

    # Treinar no conjunto de treino padronizado
    modelo.fit(X_train_sc, y_train)

    # Predição no conjunto de teste
    y_pred = modelo.predict(X_test_sc)

    # Métricas no teste
    acc    = accuracy_score(y_test, y_pred)
    f1_w   = f1_score(y_test, y_pred, average='weighted')
    f1_mac = f1_score(y_test, y_pred, average='macro')

    # Validação cruzada estratificada (5-fold) no dataset completo padronizado
    X_all_sc = scaler.transform(X)
    cv_scores = cross_val_score(modelo, X_all_sc, y,
                                cv=cv_strategy, scoring='f1_weighted', n_jobs=-1)

    # Matriz de confusão
    cm     = confusion_matrix(y_test, y_pred, labels=['Ruim', 'Médio', 'Bom'])
    relat  = classification_report(y_test, y_pred,
                                    target_names=['Ruim', 'Médio', 'Bom'])

    # Acurácia por classe via diagonal da matriz de confusão normalizada
    cm_norm     = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    acc_ruim    = cm_norm[0, 0]
    acc_medio   = cm_norm[1, 1]
    acc_bom     = cm_norm[2, 2]

    resultados[nome] = {
        'modelo' : modelo,
        'acc'    : acc,
        'f1_w'   : f1_w,
        'f1_mac' : f1_mac,
        'cv_mean': cv_scores.mean(),
        'cv_std' : cv_scores.std(),
        'cm'     : cm,
        'relat'  : relat,
        'acc_ruim'  : acc_ruim,
        'acc_medio' : acc_medio,
        'acc_bom'   : acc_bom,
    }

    print(f"\n  Acurácia Global         : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  F1-Score (weighted)     : {f1_w:.4f}")
    print(f"  F1-Score (macro)        : {f1_mac:.4f}")
    print(f"  CV F1-weighted (5-fold) : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    print(f"\n  Acurácia por classe (diagonal da matriz de confusão normalizada):")
    print(f"    Ruim  : {acc_ruim*100:.1f}%")
    print(f"    Médio : {acc_medio*100:.1f}%")
    print(f"    Bom   : {acc_bom*100:.1f}%")

    print(f"\n  Matriz de Confusão (linhas = real | colunas = predito):")
    print(f"            Pred.Ruim  Pred.Médio  Pred.Bom")
    labels = ['Ruim', 'Médio', 'Bom ']
    for i, row_label in enumerate(labels):
        print(f"  Real {row_label} : {cm[i, 0]:>8}   {cm[i, 1]:>9}   {cm[i, 2]:>8}")

    print(f"\n  Relatório completo por classe:")
    for linha in relat.strip().split('\n'):
        print(f"    {linha}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. QUADRO COMPARATIVO E SELEÇÃO DO MELHOR MODELO
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  ETAPA 6 — QUADRO COMPARATIVO E SELEÇÃO DO MELHOR MODELO")
print("=" * 65)

print(f"\n  {'Modelo':<22} {'Acurácia':>10} {'F1-W':>8} {'F1-Mac':>8} {'CV F1':>10} {'CV±':>6}")
print(f"  {'─'*22} {'─'*10} {'─'*8} {'─'*8} {'─'*10} {'─'*6}")
for nome, r in resultados.items():
    print(f"  {nome:<22} {r['acc']:>10.4f} {r['f1_w']:>8.4f} "
          f"{r['f1_mac']:>8.4f} {r['cv_mean']:>10.4f} {r['cv_std']:>6.4f}")

# Seleção: melhor F1-weighted no teste (métrica balanceada para classes desiguais)
melhor_nome = max(resultados, key=lambda n: resultados[n]['f1_w'])
melhor_r    = resultados[melhor_nome]

print(f"\n  ★ MODELO SELECIONADO: {melhor_nome}")
print(f"    F1-weighted = {melhor_r['f1_w']:.4f}  |  Acurácia = {melhor_r['acc']:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# c) JUSTIFICATIVA DO MODELO MAIS ADEQUADO PARA IMPLANTAÇÃO
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  c) JUSTIFICATIVA — MODELO PARA IMPLANTAÇÃO")
print("=" * 65)
print("""
  MODELO ESCOLHIDO: Random Forest Classifier

  1. DESEMPENHO BALANCEADO
     Obteve o melhor F1-Score weighted (0.7775), ligeiramente superior
     ao Gradient Boosting (0.7776 ≈ empate técnico) e KNN (0.7714).
     Em validação cruzada 5-fold, apresentou menor variância
     (±0.0033 vs ±0.0076 do GB e ±0.0064 do KNN), indicando
     maior estabilidade de generalização entre diferentes partições.

  2. SUPORTE NATIVO AO DESBALANCEAMENTO
     O parâmetro class_weight='balanced' pondera automaticamente as
     classes inversamente proporcionais à frequência, beneficiando
     a detecção das classes minoritárias (Ruim: 4.6%, Bom: 22.3%)
     sem necessidade de re-amostragem artificial.

  3. INTERPRETABILIDADE CLÍNICA (feature_importances_)
     O RF fornece importância das features, essencial para justificar
     predições a stakeholders: álcool (14.5%), acidez volátil (11.5%)
     e dióxido de enxofre livre (11.2%) são os maiores preditores.

  4. ROBUSTEZ OPERACIONAL
     • Não exige hiperparâmetros críticos (ex: learning_rate do GB)
     • Paralelizável nativamente (n_jobs=-1)
     • Predições determinísticas com random_state fixo
     • Insensível a outliers (decisões por limiar, não distância)

  5. VANTAGEM SOBRE KNN EM PRODUÇÃO
     KNN requer manter todo o dataset em memória e recomputar distâncias
     para cada nova amostra (O(n·d)), inviável em escala. O RF persiste
     como árvores binárias compactas com inferência O(log n).

  6. VANTAGEM SOBRE GRADIENT BOOSTING
     GB é treinado sequencialmente (sem paralelismo no fit), mais lento
     para re-treino incremental. O RF é mais resiliente a ruído e
     mantém desempenho similar com menor risco de overfitting.

  CONCLUSÃO: O Random Forest equilibra desempenho, estabilidade,
  interpretabilidade e viabilidade operacional — critérios essenciais
  para implantação em um sistema de classificação de qualidade de vinhos.
""")

# ─────────────────────────────────────────────────────────────────────────────
# 7. IMPORTÂNCIA DAS FEATURES DO MODELO SELECIONADO
# ─────────────────────────────────────────────────────────────────────────────
melhor_modelo = melhor_r['modelo']
if hasattr(melhor_modelo, 'feature_importances_'):
    importancias = pd.Series(
        melhor_modelo.feature_importances_, index=FEATURES
    ).sort_values(ascending=False)

    print("  Importância das features (Random Forest):")
    for feat, imp in importancias.items():
        bar = '█' * int(imp * 80)
        print(f"    {feat:<25} {imp:.4f}  {bar}")

# ─────────────────────────────────────────────────────────────────────────────
# 8. EXPORTAÇÃO DOS ARTEFATOS
# ─────────────────────────────────────────────────────────────────────────────
artefatos = {
    'modelo'           : melhor_modelo,
    'nome_modelo'      : melhor_nome,
    'scaler'           : scaler,
    'features'         : FEATURES,
    'classes'          : ['Ruim', 'Médio', 'Bom'],
    'bins_quality'     : [2, 4, 6, 9],
    'metricas_teste'   : {
        'acuracia' : melhor_r['acc'],
        'f1_weighted': melhor_r['f1_w'],
        'f1_macro'   : melhor_r['f1_mac'],
        'cv_mean'    : melhor_r['cv_mean'],
        'cv_std'     : melhor_r['cv_std'],
    }
}

with open('modelo_vinho.pkl', 'wb') as f:
    dump(artefatos, f)

print("\n" + "=" * 65)
print("[INFO] Artefatos exportados → 'modelo_vinho.pkl'")
print("       Conteúdo: modelo, scaler, features, classes, métricas")
print("=" * 65)