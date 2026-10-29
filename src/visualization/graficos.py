"""Funcoes auxiliares de visualizacao."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

RAIZ = Path(__file__).resolve().parents[2]
DIR_IMAGES = RAIZ / "images"

sns.set_theme(style="whitegrid", palette="deep")


def salvar_figura(fig: plt.Figure, nome: str) -> Path:
    """Salva a figura em images/<nome>.png com resolucao para o README."""
    DIR_IMAGES.mkdir(parents=True, exist_ok=True)
    caminho = DIR_IMAGES / f"{nome}.png"
    fig.savefig(caminho, dpi=120, bbox_inches="tight")
    print(f"[OK] figura salva em images/{nome}.png")
    return caminho

def curva_por_decis(df, coluna, target="alfabetizado", ax=None, q=10, log_x=False):
    """Taxa media do target por decil de uma feature numerica."""
    faixas = pd.qcut(df[coluna], q=q, duplicates="drop")
    grupos = df.groupby(faixas, observed=True)
    x = grupos[coluna].median()
    y = grupos[target].mean() * 100
    corr = df[coluna].corr(df[target])

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x.values, y.values, marker="o", linewidth=2)
    ax.axhline(df[target].mean() * 100, color="gray", linestyle=":", label="Media geral")
    if log_x:
        ax.set_xscale("log")
    ax.set_title(f"{coluna}\ncorr {corr:.3f}")
    ax.set_xlabel("Mediana do decil" + (" (escala log)" if log_x else ""))
    ax.set_ylabel("% alfabetizados")
    return ax


def heatmap_correlacao(df, colunas, ax=None):
    """Heatmap de correlacao de Spearman entre colunas numericas."""
    corr = df[colunas].corr(method="spearman")
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, ax=ax)
    return ax