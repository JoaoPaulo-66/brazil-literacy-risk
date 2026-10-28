"""
Montagem da base analitica: junta alunos 2024 com o contexto
de escola e municipio de 2023.
"""
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
DIR_RAW = RAIZ / "data" / "raw"


def carregar_base() -> pd.DataFrame:
    """Le as tres bases de data/raw e devolve uma linha por aluno."""
    alunos = pd.read_parquet(DIR_RAW / "alunos_2024.parquet")
    escolas = pd.read_parquet(DIR_RAW / "escolas_2023.parquet")
    municipios = pd.read_parquet(DIR_RAW / "municipios_2023.parquet")

    # Chaves como texto nas tres bases: Silver e Gold podem ter
    # gravado o mesmo codigo com tipos diferentes (int vs string).
    for df, chave in [(alunos, "id_escola"), (alunos, "id_municipio"),
                      (escolas, "id_escola"), (municipios, "id_municipio")]:
        df[chave] = df[chave].astype(str)

    base = alunos.merge(escolas, on="id_escola", how="left", validate="many_to_one")
    base = base.merge(municipios, on="id_municipio", how="left", validate="many_to_one")

    assert len(base) == len(alunos), "o join mudou o numero de alunos"
    return base