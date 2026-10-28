"""
Extracao das bases do Athena para data/raw.

Executar a partir da raiz do projeto:
    .\\venv\\Scripts\\python.exe -m src.preprocessing.extracao
"""
from pathlib import Path

import awswrangler as wr
import boto3
import pandas as pd

REGIAO = "sa-east-1"
DATABASE = "brazil_literacy_lakehouse"
ATHENA_OUTPUT = "s3://brazil-literacy-lakehouse-joaopaulo/athena-results/"

RAIZ = Path(__file__).resolve().parents[2]
DIR_RAW = RAIZ / "data" / "raw"


def consultar(sql: str) -> pd.DataFrame:
    """Executa uma query no Athena e devolve um DataFrame."""
    boto3.setup_default_session(region_name=REGIAO)
    return wr.athena.read_sql_query(sql, database=DATABASE, s3_output=ATHENA_OUTPUT)


def salvar(df: pd.DataFrame, nome: str) -> Path:
    """Salva o DataFrame em data/raw/<nome>.parquet."""
    DIR_RAW.mkdir(parents=True, exist_ok=True)
    caminho = DIR_RAW / f"{nome}.parquet"
    df.to_parquet(caminho, index=False)
    print(f"[OK] {nome}: {len(df):,} linhas -> {caminho.relative_to(RAIZ)}")
    return caminho


# Populacao de treino: alunos de 2024 que fizeram a prova.
# Leakage tratado na fonte:
#   - proficiencia NAO e extraida (define o target pelo corte de 743)
#   - presenca/preenchimento so servem de filtro (falta = target 0 automatico)
SQL_ALUNOS_2024 = """
    SELECT
        id_aluno,
        id_escola,
        id_municipio,
        sigla_uf,
        rede,
        serie,
        caderno,
        peso_aluno,
        CAST(alfabetizado AS integer) AS alfabetizado
    FROM silver_alunos
    WHERE CAST(ano AS varchar) = '2024'
      AND CAST(presenca AS varchar) = '1'
      AND CAST(preenchimento_caderno AS varchar) = '1'
      AND rede IN ('Estadual', 'Municipal')
"""


def extrair_alunos_2024() -> Path:
    df = consultar(SQL_ALUNOS_2024)
    print(f"     proporcao de alfabetizados: {df['alfabetizado'].mean():.2%}")
    return salvar(df, "alunos_2024")

# Porte da escola no proprio ano de 2024.
# O id_escola e remascarado a cada ano (validado na EDA: so 2,4% dos IDs
# repetidos ficam no mesmo municipio), entao nao ha historico por escola.
# O porte e conhecido antes da prova, por isso nao vaza.
# Chave composta (escola + municipio) por seguranca contra colisao de IDs.
SQL_ESCOLAS_2024 = """
    SELECT
        CAST(id_escola AS varchar)    AS id_escola,
        CAST(id_municipio AS varchar) AS id_municipio,
        COUNT(*)                      AS escola_alunos_2024
    FROM silver_alunos
    WHERE CAST(ano AS varchar) = '2024'
      AND rede IN ('Estadual', 'Municipal')
    GROUP BY 1, 2
"""


def extrair_escolas_2024() -> Path:
    df = consultar(SQL_ESCOLAS_2024)
    assert not df.duplicated(["id_escola", "id_municipio"]).any(), "chave duplicada"
    print(f"     escolas: {len(df):,} | mediana de alunos: {df['escola_alunos_2024'].median():.0f}")
    return salvar(df, "escolas_2024")

# Contexto do municipio no ano anterior (2023), vindo da camada Gold.
# Uma linha por municipio, com as tres redes lado a lado.
# O rotulo da rede publica tem acento, entao e filtrado por LIKE com
# curinga '_' no lugar do 'u' acentuado, para nao digitar acento no codigo.
SQL_MUNICIPIOS_2023 = """
    WITH g AS (
        SELECT
            id_municipio,
            CASE
                WHEN rede = 'Municipal'    THEN 'municipal'
                WHEN rede = 'Estadual'     THEN 'estadual'
                WHEN rede LIKE 'P_blica%'  THEN 'publica'
            END AS rede_curta,
            CAST(taxa_alfabetizacao AS double)      AS taxa,
            CAST(media_portugues AS double)         AS media_pt,
            CAST(percentual_participacao AS double) AS participacao
        FROM gold_indicador_municipio
        WHERE CAST(ano AS varchar) = '2023'
    )
    SELECT
        id_municipio,
        MAX(CASE WHEN rede_curta = 'municipal' THEN taxa END)         AS muni_taxa_municipal_2023,
        MAX(CASE WHEN rede_curta = 'municipal' THEN media_pt END)     AS muni_media_pt_municipal_2023,
        MAX(CASE WHEN rede_curta = 'municipal' THEN participacao END) AS muni_participacao_municipal_2023,
        MAX(CASE WHEN rede_curta = 'estadual'  THEN taxa END)         AS muni_taxa_estadual_2023,
        MAX(CASE WHEN rede_curta = 'estadual'  THEN media_pt END)     AS muni_media_pt_estadual_2023,
        MAX(CASE WHEN rede_curta = 'estadual'  THEN participacao END) AS muni_participacao_estadual_2023,
        MAX(CASE WHEN rede_curta = 'publica'   THEN taxa END)         AS muni_taxa_publica_2023,
        MAX(CASE WHEN rede_curta = 'publica'   THEN media_pt END)     AS muni_media_pt_publica_2023,
        MAX(CASE WHEN rede_curta = 'publica'   THEN participacao END) AS muni_participacao_publica_2023
    FROM g
    GROUP BY id_municipio
"""


def extrair_municipios_2023() -> Path:
    df = consultar(SQL_MUNICIPIOS_2023)
    assert df["id_municipio"].is_unique, "id_municipio duplicado na base de municipios"
    cobertura = df.filter(like="muni_taxa_").notna().sum()
    print("     municipios com taxa por rede:")
    for coluna, n in cobertura.items():
        print(f"       {coluna}: {n:,}")
    return salvar(df, "municipios_2023")

if __name__ == "__main__":
    extrair_alunos_2024()
    extrair_escolas_2024()
    extrair_municipios_2023()