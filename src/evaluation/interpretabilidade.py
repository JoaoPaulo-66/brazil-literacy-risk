"""Interpretabilidade do modelo final (Regressao Logistica)."""
import pandas as pd
import shap


def coeficientes(modelo):
    """Coeficientes da Logistica por coluna transformada (features padronizadas)."""
    nomes = modelo.named_steps["prep"].get_feature_names_out()
    coef = modelo.named_steps["modelo"].coef_[0]
    return pd.DataFrame({"feature": nomes, "coef": coef}).sort_values("coef")


def valores_shap(modelo, X, X_fundo):
    """SHAP exato para modelo linear, na escala de log-odds."""
    prep = modelo.named_steps["prep"]
    fundo = shap.maskers.Independent(prep.transform(X_fundo), max_samples=len(X_fundo))
    explainer = shap.LinearExplainer(modelo.named_steps["modelo"], fundo)
    return explainer(prep.transform(X))


def feature_original(coluna):
    """Junta as colunas criadas no pre-processamento na feature de origem."""
    for prefixo in ("sigla_uf", "rede"):
        if coluna.startswith(prefixo + "_"):
            return prefixo
    return coluna.replace("missingindicator_", "")


def importancia_agrupada(sv):
    """Media do |SHAP| somada por feature original (UF e rede viram uma barra cada)."""
    df = pd.DataFrame(abs(sv.values), columns=sv.feature_names)
    return df.T.groupby(feature_original).sum().T.mean().sort_values()