"""Funcoes auxiliares de visualizacao."""
from pathlib import Path

import matplotlib.pyplot as plt
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