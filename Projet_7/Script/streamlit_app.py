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

st.subheader("Caractéristiques client (exemple minimal)")
ext1 = st.number_input("EXT_SOURCE_1", value=0.45)
ext2 = st.number_input("EXT_SOURCE_2", value=0.30)
ext3 = st.number_input("EXT_SOURCE_3", value=0.28)
days_emp = st.number_input("DAYS_EMPLOYED", value=-1200, step=1)
pay_rate = st.number_input("PAYMENT_RATE", value=0.03, format="%.5f")

if st.button("Prédire"):
    payload = {"features": {
        "EXT_SOURCE_1": ext1, "EXT_SOURCE_2": ext2, "EXT_SOURCE_3": ext3,
        "DAYS_EMPLOYED": days_emp, "PAYMENT_RATE": pay_rate
    }}
    try:
        response = requests.post(f"{API_URL_LOCAL}/predict", json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
        st.metric("Probabilité défaut", f"{result['probability']:.3f}")
        st.metric("Seuil métier", f"{result['threshold']:.3f}")
        st.subheader(f"Décision : {result['decision']}")
        if result["missing_features"]:
            st.warning(f"Caractéristiques manquantes : {', '.join(result['missing_features'])}")
        if result["extra_features"]:
            st.info(f"Caractéristiques ignorées : {', '.join(result['extra_features'])}")
    except requests.exceptions.RequestException as e:
        st.error(f"Error during prediction: {e}")