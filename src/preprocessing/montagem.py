"""
Montagem da base analitica: junta alunos 2024 com o porte da escola
(2024) e o contexto do municipio (2023).
"""
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
DIR_RAW = RAIZ / "data" / "raw"


def carregar_base() -> pd.DataFrame:
    """Le as bases de data/raw e devolve uma linha por aluno."""
    alunos = pd.read_parquet(DIR_RAW / "alunos_2024.parquet")
    escolas = pd.read_parquet(DIR_RAW / "escolas_2024.parquet")
    municipios = pd.read_parquet(DIR_RAW / "municipios_2023.parquet")

    # Chaves como texto em todas as bases: Silver e Gold podem ter
    # gravado o mesmo codigo com tipos diferentes (int vs string).
    for df, chaves in [(alunos, ["id_escola", "id_municipio"]),
                       (escolas, ["id_escola", "id_municipio"]),
                       (municipios, ["id_municipio"])]:
        for chave in chaves:
            df[chave] = df[chave].astype(str)

    base = alunos.merge(escolas, on=["id_escola", "id_municipio"],
                        how="left", validate="many_to_one")
    base = base.merge(municipios, on="id_municipio",
                      how="left", validate="many_to_one")

    assert len(base) == len(alunos), "o join mudou o numero de alunos"
    assert base["escola_alunos_2024"].notna().all(), "aluno sem escola no join"
    return base