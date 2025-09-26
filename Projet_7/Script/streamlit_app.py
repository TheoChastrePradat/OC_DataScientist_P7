import requests
import streamlit as st

API_URL_LOCAL = st.secrets.get("API_URL_LOCAL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Credit Scoring App", page_icon="💳", layout="centered")
st.title("Credit Default Scoring - API")

c1, c2 = st.columns(2)
with c1:
    st.markdown("""
    Cette application Streamlit utilise l'API FastAPI pour obtenir des prédictions de défaut de crédit.
    Teste de l'API en local ou via l'URL publique hébergée sur Render.com
    """)
    if st.button("Ping API"):
        try:
            info = requests.get(f"{API_URL_LOCAL}/health", timeout=5).json()
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

# Saisie de client_id
client_id = st.text_input("SK_ID_CURR", value="")
go = st.button("Prédire")

# Outils
def row_to_payload(row: pd.Series) -> dict:
    # on retire l'ID et on envoie toutes les features restantes
    payload = {"features": row.drop(labels=[ID_COL]).to_dict()}
    
    for k, v in list(payload["features"].items()):
        if pd.isna(v):
            payload["features"][k] = None  # l’API mettra NaN
        elif hasattr(v, "item"):          # numpy scalar
            payload["features"][k] = v.item()
    return payload


if go:
    if client_id == "":
        st.error("Veuillez entrer un SK_ID_CURR.")
    else:
        
        row = None
        try:
            row = df.loc[df[ID_COL] == int(client_id)]
            if row.empty:
                row = df.loc[df[ID_COL].astype(str) == str(client_id)]
        except Exception:
            row = df.loc[df[ID_COL].astype(str) == str(client_id)]

        if row is None or row.empty:
            st.error("Client introuvable.")
        else:
            row = row.iloc[0]
            payload = row_to_payload(row)

            try:
                r = requests.post(f"{API_URL_LOCAL}/predict", json=payload, timeout=20)
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
            except Exception as e:
                st.error(e)

st.caption(f"API: {API_URL_LOCAL} · Lignes: {len(df)} · Colonnes: {len(df.columns)}")