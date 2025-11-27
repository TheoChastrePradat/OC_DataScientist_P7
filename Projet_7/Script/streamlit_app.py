import os
import pandas as pd
import requests
import streamlit as st
from pathlib import Path

API_URL = st.secrets.get("API_URL", "https://credit-scoring-api-8j5p.onrender.com")

st.set_page_config(page_title="Credit Scoring App", page_icon="💳", layout="centered")
st.title("Credit Default Scoring - API")

c1, c2 = st.columns(2)
with c1:
    st.markdown("""Prediction de defaut de credit via FastAPI en local ou via l'URL render.""")
    if st.button("Ping API"):
        try:
            info = requests.get(f"{API_URL}/health", timeout=5).json()
            if info.get("status") == "ok":
                st.success(f"API OK · {info['model_version']} · seuil={info['threshold']}")
            else:
                st.error(f"API returned unexpected response: {info}")
        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to API: {e}")


@st.cache_data(show_spinner=False)
def load_features(path: str):
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)

# Chemin local au dataset de features prod
FEATURES_PATH = os.getenv("FEATURES_PATH", st.secrets.get("FEATURES_PATH", "features.parquet"))
df = load_features(FEATURES_PATH)

# on s’assure d’avoir un identifiant client
ID_COL = "SK_ID_CURR"
assert ID_COL in df.columns, f"Colonne id manquante: {ID_COL}"

if ID_COL in df.columns:
    if pd.api.types.is_float_dtype(df[ID_COL]) or pd.api.types.is_object_dtype(df[ID_COL]):
        df[ID_COL] = pd.to_numeric(df[ID_COL], errors="coerce").astype("Int64")
else:
    st.error(f"Colonne id manquante: {ID_COL}")

# Saisie de client_id
client_id = st.text_input("SK_ID_CURR", value="")
go = st.button("Prédire")

# Outils
def row_to_payload(row: pd.Series) -> dict:
    d = row.drop(labels=["SK_ID_CURR"]).to_dict()
    clean = {}
    for k, v in d.items():
        if pd.isna(v):
            clean[k] = None
        elif hasattr(v, "item"):
            clean[k] = v.item()
        elif isinstance(v, (bool, int, float)):
            clean[k] = v
        else:
            # dernier recours: tenter float, sinon None
            try:
                clean[k] = float(v)
            except Exception:
                clean[k] = None
    return {"features": clean}


if go:
    if client_id == "":
        st.error("Veuillez entrer un SK_ID_CURR.")
    else:
        
        row = None
        try:
            df[ID_COL] = df[ID_COL].astype(str).str.strip()
            client_id_norm = str(client_id).strip()
            row = df.loc[df[ID_COL] == int(client_id_norm)]
            if row.empty:
                row = df.loc[df[ID_COL].astype(str) == client_id_norm]
        except Exception:
            row = df.loc[df[ID_COL].astype(str) == client_id_norm]

        if row is None or row.empty:
            st.error("Client introuvable.")
        else:
            row = row.iloc[0]
            payload = row_to_payload(row)

            try:
                r = requests.post(f"{API_URL}/predict", json=payload, timeout=20)
                r.raise_for_status()
                res = r.json()

                st.subheader("Résultat")
                st.metric("Probabilité défaut", f"{res['probability']:.3f}")
                st.metric("Seuil métier", f"{res['threshold']:.3f}")
                st.write(f"**Décision** : {res['decision']}")
                if res.get("missing_features"):
                    st.info(f"Features manquantes remplies (NaN): {len(res['missing_features'])}")
                if res.get("extra_features"):
                    st.info(f"Features ignorées: {res['extra_features']}")
            except requests.HTTPError as e:
                # Affiche le corps renvoyé par FastAPI
                st.error(f"{e}\n{r.text}")
            except Exception as e: 
                st.error(e)

st.caption(f"API: {API_URL}/docs · Lignes: {len(df)} · Colonnes: {len(df.columns)}")


# rapport HTML data drift
st.header("Data Drift")

REPORT_PATH = Path(__file__).parent / "artifacts" / "evidently" / "evidently_data_drift_report.html"

if REPORT_PATH.exists():
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        html = f.read()
    st.components.v1.html(html, height=900, scrolling=True)
else:
    st.info("Aucun rapport trouvé. Lance d’abord:  `python generate_drift_report.py`")