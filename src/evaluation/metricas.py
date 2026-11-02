"""Avaliacao do modelo final no conjunto de teste."""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, recall_score, roc_auc_score, roc_curve


def escolher_corte(y, proba):
    """Corte que maximiza o indice de Youden (sensibilidade + especificidade - 1)."""
    fpr, tpr, cortes = roc_curve(y, proba)
    return float(cortes[np.argmax(tpr - fpr)])


def avaliar(y, proba, corte):
    """Metricas no corte escolhido. Classe de interesse: nao alfabetizado (0)."""
    pred = (proba >= corte).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        "roc_auc": roc_auc_score(y, proba),
        "corte": corte,
        "recall_nao_alfab": recall_score(y, pred, pos_label=0),
        "precisao_nao_alfab": tn / (tn + fn),
        "recall_alfab": recall_score(y, pred, pos_label=1),
    }


def auc_por_uf(y, proba, uf):
    """ROC-AUC dentro de cada UF: mede se o modelo ordena bem o risco no estado."""
    df = pd.DataFrame({"y": y.to_numpy(), "p": proba, "uf": uf.to_numpy()})
    return (df.groupby("uf")
              .apply(lambda d: roc_auc_score(d["y"], d["p"]) if d["y"].nunique() > 1 else np.nan)
              .rename("roc_auc").sort_values())