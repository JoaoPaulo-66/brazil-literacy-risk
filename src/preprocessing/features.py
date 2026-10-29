"""
Feature engineering e definicao das variaveis do modelo.

As transformacoes daqui sao linha a linha e nao aprendem nada da base
(nenhuma media, mediana ou contagem). Por isso podem rodar antes do split
treino/teste sem causar data leakage. O que aprende com os dados
(imputacao, encoding) fica dentro do Pipeline do sklearn.
"""
import pandas as pd

TARGET = "alfabetizado"
PESO = "peso_aluno"

FEATURES_CATEGORICAS = ["rede", "sigla_uf"]

FEATURES_NUMERICAS = [
    # Escola (2024, sem resultado de prova)
    "escola_alunos_2024",
    # Contexto municipal do indicador (2023)
    "muni_taxa_mesma_rede_2023",
    "muni_participacao_municipal_2023",
    # IBGE
    "populacao_2022",
    "pop_crescimento_2010_2022",
    "alfabetizacao_15mais_2022",
    "pib_per_capita_2022",
    "pib_agro_pct_2021",
    "pib_adm_publica_pct_2021",
]


def criar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria as features derivadas a partir da base de carregar_base()."""
    df = df.copy()

    # Taxa do municipio em 2023 na mesma rede do aluno (municipal ou
    # estadual). Sem dado da rede, usa a taxa da rede publica como fallback.
    df["muni_taxa_mesma_rede_2023"] = (
        df["muni_taxa_municipal_2023"]
        .where(df["rede"] == "Municipal", df["muni_taxa_estadual_2023"])
        .fillna(df["muni_taxa_publica_2023"])
    )
    return df


def separar_xy(df: pd.DataFrame):
    """Devolve X (features), y (target) e w (peso amostral)."""
    df = criar_features(df)
    X = df[FEATURES_CATEGORICAS + FEATURES_NUMERICAS]
    y = df[TARGET].astype(int)
    w = df[PESO]
    return X, y, w