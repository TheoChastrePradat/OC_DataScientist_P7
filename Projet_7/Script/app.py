# Imports
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware

import os
import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from pydantic import BaseModel, Field
from typing import Optional, Tuple, List, Dict, Any

# Config & chemins artefacts
ART_DIR      = Path(os.getenv("ART_DIR", "artifacts"))
MODEL_PATH   = Path(os.getenv("MODEL_PATH", ART_DIR / "model.joblib"))
META_PATH    = Path(os.getenv("ARTIFACTS_PATH", ART_DIR / "artifacts.json"))
OVERRIDE_THR = os.getenv("THRESHOLD")
SKIP_MODEL   = os.getenv("SKIP_MODEL_LOAD", "0") == "1"  # utile pour la CI

# État global (lazy)
_model: Optional[Any] = None
_meta: Dict[str, Any] = {}
EXPECTED_FEATURES: List[str] = []
THRESHOLD: float = 0.5
MODEL_VERSION: str = "v1"
CLASS_MAPPING: Dict[str, int] = {"Accepter": 0, "Refuser": 1}

def _load_meta_if_needed() -> None:
    """Charge les métadonnées (seuil, features, mapping) au premier accès."""
    global _meta, EXPECTED_FEATURES, THRESHOLD, MODEL_VERSION, CLASS_MAPPING
    if _meta:
        return
    try:
        with open(META_PATH, "r", encoding="utf-8") as f:
            _meta = json.load(f)
        EXPECTED_FEATURES = _meta["expected_features"]
        thr = float(_meta.get("threshold", 0.5))
        THRESHOLD = float(OVERRIDE_THR) if OVERRIDE_THR is not None else thr
        MODEL_VERSION = _meta.get("model_version", "v1")
        CLASS_MAPPING = _meta.get("class_mapping", {"Accepter": 0, "Refuser": 1})
    except Exception as e:
        if SKIP_MODEL:
            # Valeurs par défaut en CI si les artefacts ne sont pas présents
            _meta = {}
            EXPECTED_FEATURES = []
            THRESHOLD = float(OVERRIDE_THR) if OVERRIDE_THR is not None else 0.5
            MODEL_VERSION = "ci"
            CLASS_MAPPING = {"Accepter": 0, "Refuser": 1}
        else:
            raise

def _load_model_if_needed() -> None:
    """Charge le modèle au premier accès, sauf si SKIP_MODEL_LOAD=1."""
    global _model
    if _model is not None or SKIP_MODEL:
        return
    _model = joblib.load(MODEL_PATH)

# FastAPI app
app = FastAPI(
    title="Credit Default Scoring API",
    version=MODEL_VERSION,
    description="Retourne la probabilité de défaut et la décision (Accepter/Refuser) selon le seuil métier."
)

# CORS, utile pour Streamlit local
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# Schémas d'E/S

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
    rows: list[dict]  # chaque élément est un dict

class PredictResponse(BaseModel):
    # éviter le warning "model_" namespace protégé
    model_config = {"protected_namespaces": ()}

    probability: float
    threshold: float
    predicted_class: int            # 1 = "Refuser", 0 = "Accepter"
    decision: str                   # libellé lisible
    missing_features: list[str] = []  # colonnes manquantes
    extra_features: list[str] = []    # colonnes ignorées
    model_version: str

class PredictBatchResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    results: list[PredictResponse]

# Helpers

def prepare_dataframe(features: dict | list[dict]) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Aligne les features sur EXPECTED_FEATURES, ajoute les manquantes (NaN), ignore les extra."""
    _load_meta_if_needed()

    if isinstance(features, dict):
        df = pd.DataFrame([features])
    else:
        df = pd.DataFrame(features)

    missing = [c for c in EXPECTED_FEATURES if c not in df.columns]
    extra   = [c for c in df.columns if c not in EXPECTED_FEATURES]

    for c in missing:
        df[c] = np.nan  # LGBM gère NaN nativement

    # Réordonne exactement comme à l'entraînement
    if EXPECTED_FEATURES:
        df = df[EXPECTED_FEATURES]

    # Convertit tout en numérique, puis cast en float32
    df = df.apply(pd.to_numeric, errors="coerce").astype(np.float32)

    return df, missing, extra

def class_label(pred_int: int) -> str:
    _load_meta_if_needed()
    # 1 -> "Refuser", 0 -> "Accepter"
    return "❌ Refuser" if pred_int == CLASS_MAPPING.get("Refuser", 1) else "✅ Accepter"

# Routes

@app.get("/")
def root():
    # redirige vers la doc interactive Swagger
    return RedirectResponse(url="/docs", status_code=302)

@app.get("/health")
def health():
    _load_meta_if_needed()
    return {
        "status": "ok",
        "model_version": MODEL_VERSION,
        "threshold": THRESHOLD,
        "n_features": len(EXPECTED_FEATURES),
        "class_mapping": CLASS_MAPPING,
        "model_path": str(MODEL_PATH),
        "meta_path": str(META_PATH),
        "skip_model_load": SKIP_MODEL,
        "model_loaded": _model is not None,
    }

@app.get("/metadata")
def metadata():
    _load_meta_if_needed()
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
        _load_meta_if_needed()
        _load_model_if_needed()
        if _model is None:
            raise RuntimeError("Modèle non chargé (SKIP_MODEL_LOAD=1 ?)")
        X, missing, extra = prepare_dataframe(req.features)
        proba_bad = float(_model.predict_proba(X)[:, 1][0])   # colonne 1 = classe "Refuser"
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
        _load_meta_if_needed()
        _load_model_if_needed()
        if _model is None:
            raise RuntimeError("Modèle non chargé (SKIP_MODEL_LOAD=1 ?)")
        X, missing, extra = prepare_dataframe(req.rows)
        probas = _model.predict_proba(X)[:, 1]
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
