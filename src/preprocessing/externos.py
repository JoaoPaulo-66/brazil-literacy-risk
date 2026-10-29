"""Enriquecimento com dados externos do IBGE (API de agregados v3).

Baixa indicadores municipais (Censo 2022 e PIB dos Municipios 2022),
um registro por municipio, chave id_municipio (codigo IBGE de 7 digitos).
"""
import gzip
import json
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

BASE = "https://servicodados.ibge.gov.br/api/v3/agregados"
RAIZ = Path(__file__).resolve().parents[2]
ARQUIVO = RAIZ / "data" / "raw" / "ibge_municipios.parquet"

FONTES = [
    {
        "tabela": 4709,
        "periodo": 2022,
        "variaveis": {93: "populacao_2022", 10605: "pop_crescimento_2010_2022"},
    },
    {
        "tabela": 5938,
        "periodo": 2022,
        "variaveis": {37: "pib_2022"},
    },
    {
        # VAB por setor so esta disponivel por municipio ate 2021
        "tabela": 5938,
        "periodo": 2021,
        "variaveis": {
            498: "vab_total_2021",
            513: "vab_agro_2021",
            525: "vab_adm_publica_2021",
        },
    },
    {
        "tabela": 9543,
        "periodo": 2022,
        "variaveis": {2513: "alfabetizacao_15mais_2022"},
        # Totais de sexo, cor/raca e idade (15 anos ou mais)
        "classificacao": "2[6794]|86[95251]|287[100362]",
    },
]


def baixar_json(url):
    """Baixa um JSON da API do IBGE, descomprimindo gzip se vier comprimido."""
    url = urllib.parse.quote(url, safe=":/?&=")
    with urllib.request.urlopen(url, timeout=120) as r:
        dados = r.read()
    if dados[:2] == b"\x1f\x8b":
        dados = gzip.decompress(dados)
    return json.loads(dados.decode("utf-8"))


def baixar_tabela(tabela, periodo, variaveis, classificacao=None):
    """Baixa variaveis de uma tabela SIDRA para todos os municipios.

    Devolve um DataFrame com id_municipio e uma coluna por variavel.
    """
    ids = "|".join(str(v) for v in variaveis)
    url = f"{BASE}/{tabela}/periodos/{periodo}/variaveis/{ids}?localidades=N6[all]"
    if classificacao:
        url += f"&classificacao={classificacao}"

    resposta = baixar_json(url)

    colunas = []
    for var in resposta:
        nome = variaveis[int(var["id"])]
        linhas = [
            {"id_municipio": s["localidade"]["id"], nome: s["serie"].get(str(periodo))}
            for res in var["resultados"]
            for s in res["series"]
        ]
        colunas.append(pd.DataFrame(linhas).set_index("id_municipio"))

    df = pd.concat(colunas, axis=1).reset_index()
    # A API usa "-", "..." e "X" para valores ausentes ou sigilosos
    for nome in variaveis.values():
        df[nome] = pd.to_numeric(df[nome], errors="coerce")
    return df


def montar_ibge():
    """Junta todas as fontes numa tabela unica por municipio."""
    tabelas = [baixar_tabela(**fonte) for fonte in FONTES]
    df = tabelas[0]
    for t in tabelas[1:]:
        df = df.merge(t, on="id_municipio", how="outer", validate="one_to_one")

    # PIB vem em mil reais
    df["pib_per_capita_2022"] = df["pib_2022"] * 1000 / df["populacao_2022"]
    # Participacoes setoriais pela definicao oficial (VAB setor / VAB total).
    # Com VAB total <= 0 a proporcao nao tem significado: vira NaN e o imputador trata.
    vab_valido = df["vab_total_2021"] > 0
    df["pib_agro_pct_2021"] = (df["vab_agro_2021"] / df["vab_total_2021"] * 100).where(vab_valido)
    df["pib_adm_publica_pct_2021"] = (df["vab_adm_publica_2021"] / df["vab_total_2021"] * 100).where(vab_valido)

    return df.drop(columns=["pib_2022", "vab_total_2021", "vab_agro_2021", "vab_adm_publica_2021"])


if __name__ == "__main__":
    df = montar_ibge()
    print("Shape:", df.shape)
    print("IDs duplicados:", df["id_municipio"].duplicated().sum())
    print("\nNulos por coluna:")
    print(df.isna().sum())
    print("\nResumo:")
    print(df.describe().T.round(2))
    ARQUIVO.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ARQUIVO, index=False)
    print(f"\nSalvo em {ARQUIVO}")