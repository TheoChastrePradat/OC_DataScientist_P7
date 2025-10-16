from fastapi import FastAPI

import os
import json
import joblib

import numpy as np
import pandas as pd

from pathlib import Path
from pydantic import BaseModel, Field
from fastapi import HTTPException
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

# chemins dossier artifacts
ART_DIR       = Path(os.getenv("ART_DIR", "artifacts"))
MODEL_PATH    = Path(os.getenv("MODEL_PATH", ART_DIR / "model.joblib"))
META_PATH     = Path(os.getenv("ARTIFACTS_PATH", ART_DIR / "artifacts.json"))
OVERRIDE_THR  = os.getenv("THRESHOLD")

# chargement des artefacts au démarrage
model = joblib.load(MODEL_PATH)

with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)

EXPECTED_FEATURES = meta["expected_features"]
THRESHOLD = float(OVERRIDE_THR) if OVERRIDE_THR is not None else float(meta["threshold"])
MODEL_VERSION = meta.get("model_version", "v1")
CLASS_MAPPING = meta.get("class_mapping", {"Accepter": 0, "Refuser": 1})

# FastAPI app
app = FastAPI(
    title="Credit Default Scoring API",
    version=MODEL_VERSION,
    description="Retourne la probabilité de défaut et la décision (Accepter/Refuser) selon le seuil métier."
)

# facilite les tests depuis Streamlit local
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# schémas d'E/S
class PredictRequest(BaseModel):
    
    features: dict = Field(
        ...,
        example={
            "EXT_SOURCE_1": 0.45,
            "EXT_SOURCE_2": 0.30,
            "EXT_SOURCE_3": 0.28,
            "DAYS_EMPLOYED": -1200,
            "PAYMENT_RATE": 0.03
        }
    )

class PredictBatchRequest(BaseModel):
    # Liste de lignes, chaque élément est un dict {feature: valeur}
    rows: list[dict]

class PredictResponse(BaseModel):
    probability: float
    threshold: float
    predicted_class: int # 1 = "Refuser", 0 = "Accepter"
    decision: str
    missing_features: list[str] = []
    extra_features: list[str] = []
    model_version: str

class PredictBatchResponse(BaseModel):
    results: list[PredictResponse]

# helpers
def prepare_dataframe(features: dict | list[dict]) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Aligne les features sur EXPECTED_FEATURES, ajoute les manquantes (NaN), ignore les extra."""
    if isinstance(features, dict):
        df = pd.DataFrame([features])
    else:
        df = pd.DataFrame(features)

    missing = [c for c in EXPECTED_FEATURES if c not in df.columns]
    extra   = [c for c in df.columns if c not in EXPECTED_FEATURES]

    for c in missing:
        # LGBM gère NaN nativement
        df[c] = np.nan

    # Réordonne exactement comme à l'entraînement
    df = df[EXPECTED_FEATURES]

    # Convertit tout en numérique (strings -> NaN), puis cast en float32
    df = df.apply(pd.to_numeric, errors="coerce").astype(np.float32)

    return df, missing, extra

def class_label(pred_int: int) -> str:
    # 1 -> "Refuser", 0 -> "Accepter"
    return "❌ Refuser" if pred_int == CLASS_MAPPING.get("Refuser", 1) else "✅ Accepter"

# route par défaut, redirige vers la doc interactive
@app.get("/")
def root():
    return RedirectResponse(url="/docs", status_code=302)


# endpoints API
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": MODEL_VERSION,
        "threshold": THRESHOLD,
        "n_features": len(EXPECTED_FEATURES),
        "class_mapping": CLASS_MAPPING,
        "model_path": str(MODEL_PATH),
        "meta_path": str(META_PATH),
    }


@app.get("/metadata")
def metadata():
    return {
        "model_version": MODEL_VERSION,
        "threshold": THRESHOLD,
        "n_features": len(EXPECTED_FEATURES),
        "expected_features": EXPECTED_FEATURES,  # ordre exact d’entraînement
        "class_mapping": CLASS_MAPPING,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        X, missing, extra = prepare_dataframe(req.features)
        proba_bad = float(model.predict_proba(X)[:, 1][0])   # colonne 1 = classe "Refuser"
        yhat = int(proba_bad >= THRESHOLD)                   # 1 = Refuser, 0 = Accepter
        return PredictResponse(
            probability=proba_bad,
            threshold=THRESHOLD,
            predicted_class=yhat,
            decision=class_label(yhat),
            missing_features=missing,
            extra_features=extra,
            model_version=MODEL_VERSION
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict_batch", response_model=PredictBatchResponse)
def predict_batch(req: PredictBatchRequest):
    try:
        X, missing, extra = prepare_dataframe(req.rows)
        probas = model.predict_proba(X)[:, 1]
        preds = (probas >= THRESHOLD).astype(int)

        results = []
        for p, y in zip(probas, preds):
            results.append(PredictResponse(
                probability=float(p),
                threshold=THRESHOLD,
                predicted_class=int(y),
                decision=class_label(int(y)),
                missing_features=missing,   # identiques pour tout le batch
                extra_features=extra,
                model_version=MODEL_VERSION
            ))
        return PredictBatchResponse(results=results)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))