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


if __name__ == "__main__":
    extrair_alunos_2024()