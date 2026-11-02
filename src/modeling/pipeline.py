"""
Pipelines de modelagem: pre-processamento integrado a cada modelo.

Tudo que aprende com os dados (mediana do imputador, media/desvio do
scaler, categorias do encoder) fica DENTRO do Pipeline. Assim, em cada
fold da validacao cruzada, essas estatisticas sao calculadas so com os
dados de treino daquele fold: sem data leakage.
"""
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.preprocessing.features import FEATURES_CATEGORICAS, FEATURES_NUMERICAS

SEMENTE = 42

# Muito assimetricas (populacao vai de 833 a 11 milhoes): log so para o modelo linear
COLUNAS_ASSIMETRICAS = ["populacao_2022", "pib_per_capita_2022", "escola_alunos_2024"]


def criar_preprocessador(linear: bool) -> ColumnTransformer:
    """Monta o pre-processamento.

    linear=True  -> log nas assimetricas + padronizacao (Regressao Logistica).
    linear=False -> so imputacao (arvores nao dependem de escala).
    """
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    if linear:
        demais = [c for c in FEATURES_NUMERICAS if c not in COLUNAS_ASSIMETRICAS]
        transformadores = [
            ("num_log", Pipeline([
                ("imputar", SimpleImputer(strategy="median", add_indicator=True)),
                ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                ("escalar", StandardScaler()),
            ]), COLUNAS_ASSIMETRICAS),
            ("num", Pipeline([
                ("imputar", SimpleImputer(strategy="median", add_indicator=True)),
                ("escalar", StandardScaler()),
            ]), demais),
            ("cat", encoder, FEATURES_CATEGORICAS),
        ]
    else:
        transformadores = [
            ("num", SimpleImputer(strategy="median", add_indicator=True), FEATURES_NUMERICAS),
            ("cat", encoder, FEATURES_CATEGORICAS),
        ]

    prep = ColumnTransformer(transformadores, verbose_feature_names_out=False)
    return prep.set_output(transform="pandas")


def criar_modelos() -> dict:
    """Os 4 modelos em ordem de complexidade, cada um com seu pre-processamento."""
    return {
        "dummy": Pipeline([
            ("prep", criar_preprocessador(linear=False)),
            ("modelo", DummyClassifier(strategy="prior")),
        ]),
        "logistica": Pipeline([
            ("prep", criar_preprocessador(linear=True)),
            ("modelo", LogisticRegression(max_iter=1000)),
        ]),
        "random_forest": Pipeline([
            ("prep", criar_preprocessador(linear=False)),
            ("modelo", RandomForestClassifier(
                n_estimators=200, min_samples_leaf=50, max_features="sqrt",
                n_jobs=-1, random_state=SEMENTE,
            )),
        ]),
        "lightgbm": Pipeline([
            ("prep", criar_preprocessador(linear=False)),
            ("modelo", LGBMClassifier(
                n_estimators=500, learning_rate=0.05, num_leaves=31,
                min_child_samples=100, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.8, random_state=SEMENTE, verbose=-1,
            )),
        ]),
    }

# Espacos de busca de hiperparametros (prefixo "modelo__" = passo do Pipeline)
ESPACOS_BUSCA = {
    "logistica": {
        "modelo__C": [0.01, 0.1, 1.0, 10.0],
    },
    "lightgbm": {
        "modelo__n_estimators": [200, 500, 1000],
        "modelo__learning_rate": [0.02, 0.05, 0.1],
        "modelo__num_leaves": [15, 31, 63, 127],
        "modelo__min_child_samples": [50, 100, 300, 1000],
        "modelo__reg_lambda": [0.0, 1.0, 10.0],
    },
}