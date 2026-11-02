"""
Divisao treino/teste e validacao cruzada POR MUNICIPIO.

As features de contexto sao constantes dentro de cada municipio. Num split
aleatorio, alunos do mesmo municipio caem no treino e no teste: o modelo
reconhece o municipio pelos valores exatos (ex.: populacao) e aprende a
taxa de 2024 dos colegas, o que equivale a usar o resultado da prova do
mesmo ano (leakage). Separando por municipio, o teste mede a capacidade
de prever municipios que o modelo nunca viu.
"""
import pandas as pd
import time
from sklearn.base import clone
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, RandomizedSearchCV

from src.modeling.pipeline import SEMENTE


def dividir_treino_teste(X, y, w, g, n_partes=5):
    """Separa ~1/n_partes dos municipios para teste, estratificando pelo target."""
    sgkf = StratifiedGroupKFold(n_splits=n_partes, shuffle=True, random_state=SEMENTE)
    idx_treino, idx_teste = next(sgkf.split(X, y, groups=g))
    treino = (X.iloc[idx_treino], y.iloc[idx_treino], w.iloc[idx_treino], g.iloc[idx_treino])
    teste = (X.iloc[idx_teste], y.iloc[idx_teste], w.iloc[idx_teste], g.iloc[idx_teste])
    return treino, teste


def validar_modelo(modelo, X, y, w, cv, g=None):
    """Validacao cruzada: ajusta em cada fold de treino e mede no fold de validacao."""
    resultados = []
    for i, (idx_tr, idx_va) in enumerate(cv.split(X, y, groups=g)):
        m = clone(modelo)
        m.fit(X.iloc[idx_tr], y.iloc[idx_tr],
              modelo__sample_weight=w.iloc[idx_tr].to_numpy())
        proba = m.predict_proba(X.iloc[idx_va])[:, 1]
        y_va = y.iloc[idx_va]
        resultados.append({
            "fold": i,
            "roc_auc": roc_auc_score(y_va, proba),
            "recall_nao_alfab": recall_score(y_va, (proba >= 0.5).astype(int), pos_label=0),
        })
    return pd.DataFrame(resultados)

def comparar_modelos(modelos, X, y, w, cv, g=None, rotulo_split=""):
    """Roda validar_modelo para cada modelo e devolve um resumo por modelo."""
    linhas = []
    for nome, modelo in modelos.items():
        t0 = time.perf_counter()
        r = validar_modelo(modelo, X, y, w, cv, g)
        linhas.append({
            "modelo": nome,
            "split": rotulo_split,
            "roc_auc_media": r["roc_auc"].mean(),
            "roc_auc_dp": r["roc_auc"].std(),
            "recall_nao_alfab": r["recall_nao_alfab"].mean(),
            "segundos": time.perf_counter() - t0,
        })
    return pd.DataFrame(linhas)

def otimizar(modelo, espaco, X, y, w, g, cv, n_iter):
    """Busca aleatoria de hiperparametros com validacao cruzada por municipio."""
    busca = RandomizedSearchCV(
        modelo, espaco, n_iter=n_iter, scoring="roc_auc", cv=cv,
        random_state=SEMENTE, n_jobs=1, refit=False,
    )
    busca.fit(X, y, groups=g, modelo__sample_weight=w.to_numpy())
    resultados = pd.DataFrame(busca.cv_results_)
    colunas = ["params", "mean_test_score", "std_test_score"]
    return resultados.sort_values("rank_test_score")[colunas].reset_index(drop=True)