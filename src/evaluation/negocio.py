"""Respostas as perguntas de negocio a partir do modelo final."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.modeling.pipeline import SEMENTE
from src.preprocessing.externos import baixar_json

URL_MUNICIPIOS = "https://servicodados.ibge.gov.br/api/v1/localidades/municipios"
CONTEXTO = ["populacao_2022", "alfabetizacao_15mais_2022", "pib_per_capita_2022",
            "pib_agro_pct_2021", "pib_adm_publica_pct_2021"]


def nomes_municipios():
    """Nome de cada municipio pela API de localidades do IBGE."""
    dados = baixar_json(URL_MUNICIPIOS)
    return pd.DataFrame({"id_municipio": [str(m["id"]) for m in dados],
                         "municipio": [m["nome"] for m in dados]})


def risco_por_municipio(X, y, w, g, proba, corte):
    """Agrega as previsoes por municipio, ponderando pelo peso amostral."""
    d = pd.DataFrame({"id_municipio": g.to_numpy(), "sigla_uf": X["sigla_uf"].to_numpy(),
                      "w": w.to_numpy(), "y": y.to_numpy(), "p": proba,
                      "risco": (proba < corte).astype(int)})
    d = d.assign(wy=d["w"] * d["y"], wp=d["w"] * d["p"], wr=d["w"] * d["risco"])
    m = d.groupby("id_municipio").agg(
        sigla_uf=("sigla_uf", "first"), alunos=("y", "size"),
        w=("w", "sum"), wy=("wy", "sum"), wp=("wp", "sum"), wr=("wr", "sum"))
    m["taxa_real_2024"] = m["wy"] / m["w"] * 100
    m["taxa_prevista_2024"] = m["wp"] / m["w"] * 100
    m["pct_alunos_em_risco"] = m["wr"] / m["w"] * 100
    return m.drop(columns=["w", "wy", "wp", "wr"]).reset_index()


def contexto_municipio(X, g):
    """Uma linha por municipio com o contexto socioeconomico e a taxa de 2023."""
    grupos = g.to_numpy()
    c = X[CONTEXTO].groupby(grupos).first()
    c["taxa_2023"] = X["muni_taxa_mesma_rede_2023"].groupby(grupos).mean()
    c.index.name = "id_municipio"
    return c.reset_index()


def agrupar_municipios(df, colunas, k=4):
    """KMeans sobre o perfil padronizado do municipio (log nas assimetricas)."""
    Z = df[colunas].copy()
    for col in ["populacao_2022", "pib_per_capita_2022"]:
        if col in Z:
            Z[col] = np.log(Z[col])
    Z = Z.fillna(Z.median())
    return KMeans(n_clusters=k, n_init=10, random_state=SEMENTE).fit_predict(
        StandardScaler().fit_transform(Z))

def taxa_municipal_prevista(modelo, X, w, g):
    """Taxa prevista da rede municipal de cada municipio (ponderada pelo peso)."""
    m = (X["rede"] == "Municipal").to_numpy()
    p = modelo.predict_proba(X[m])[:, 1]
    d = pd.DataFrame({"id_municipio": g.to_numpy()[m], "w": w.to_numpy()[m]})
    d["wp"] = d["w"] * p
    s = d.groupby("id_municipio")[["w", "wp"]].sum()
    return (s["wp"] / s["w"] * 100).rename("taxa_prevista")


def projetar_ano_seguinte(X, g, resultado_recente):
    """Troca a taxa municipal do ano anterior pelo resultado mais recente (rede municipal)."""
    Xn = X.copy()
    novo = g.map(resultado_recente)
    m = (Xn["rede"] == "Municipal") & novo.notna()
    Xn.loc[m, "muni_taxa_mesma_rede_2023"] = novo[m]
    return Xn