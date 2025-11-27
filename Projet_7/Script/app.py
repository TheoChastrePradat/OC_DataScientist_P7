# Imports
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware

import os
import shap
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
BG_PATH = ART_DIR / "shap_background.parquet"

_explainer = None

# Mode mock pour la CI
class _DummyModel:
    """
    Renvoie proba_refus = 0 pour simplifier les tests CI
    """
    def predict_proba(self, X):
        n = len(X)
        proba_refus = np.zeros(n, dtype=np.float32)
        return np.c_[1 - proba_refus, proba_refus]


class ExplainRequest(BaseModel):
    features: dict
    top_k: int = 10


class ExplainResponse(BaseModel):
    base_value: float
    prediction: float
    contribution: List[Dict]
    model_version: str
    threshold: float


# État global (lazy)
_model: Optional[Any] = None
_meta: Dict[str, Any] = {}
EXPECTED_FEATURES: List[str] = []
THRESHOLD: float = 0.5
MODEL_VERSION: str = "v1"
CLASS_MAPPING: Dict[str, int] = {"Accepter": 0, "Refuser": 1}

def _load_meta_if_needed() -> None:
    """
    Charge les métadonnées (seuil, features, mapping) au premier accès
    """
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
    except Exception:
        if SKIP_MODEL:
            # Valeurs par défaut en CI si les artefacts ne sont pas présents
            _meta = {}
            EXPECTED_FEATURES = ["EXT_SOURCE_1", "EXT_SOURCE_2", "PAYMENT_RATE"]
            THRESHOLD = float(OVERRIDE_THR) if OVERRIDE_THR is not None else 0.5
            MODEL_VERSION = "ci"
            CLASS_MAPPING = {"Accepter": 0, "Refuser": 1}
        else:
            raise

def _load_model_if_needed() -> None:
    """
    Charge le modèle au premier accès, ou un DummyModel si SKIP_MODEL_LOAD=1
    """
    global _model
    if _model is not None:
        return
    if SKIP_MODEL:
        _model = _DummyModel()
    else:
        _model = joblib.load(MODEL_PATH)

def _load_background_if_needed() -> pd.DataFrame:
    """
    Charge le dataset de background pour SHAP
    """
    if BG_PATH.exists():
        bg = pd.read_parquet(BG_PATH)

        for c in EXPECTED_FEATURES:
            if c not in bg.columns:
                bg[c] = np.nan
        bg = bg[EXPECTED_FEATURES].apply(pd.to_numeric, errors="coerce").astype(np.float32)
        return bg
    
    med = pd.Series({c: 0.0 for c in EXPECTED_FEATURES}, dtype=np.float32)
    return pd.DataFrame([med])[EXPECTED_FEATURES]

def _load_explainer_if_needed():
    """
    Charge l'explainer SHAP au premier acces
    """
    global _explainer
    _load_model_if_needed()
    _load_meta_if_needed()

    if _explainer is None and _model is not None:
        return

    bg = _load_background_if_needed()

    if EXPECTED_FEATURES:
        for c in EXPECTED_FEATURES:
            if c not in bg.columns:
                bg[c] = np.nan
        bg = bg[EXPECTED_FEATURES]
    bg = bg.apply(pd.to_numeric, errors="coerce").astype(np.float32).dropna(how="all")

    masker = shap.maskers.Independent(bg)
    # donne les valeurs en proba
    _explainer = shap.TreeExplainer(_model, masker=masker, algorithm="tree", model_output="probability", feature_perturbation="interventional")

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
    rows: list[dict] # chaque élément est un dict

class PredictResponse(BaseModel):
    # éviter le warning "model_" namespace protégé
    model_config = {"protected_namespaces": ()}

    probability: float
    threshold: float
    predicted_class: int # 1 = "Refuser", 0 = "Accepter"
    decision: str # libellé lisible
    missing_features: list[str] = [] # colonnes manquantes
    extra_features: list[str] = [] # colonnes ignorées
    model_version: str

class PredictBatchResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    results: list[PredictResponse]

# Helpers
def prepare_dataframe(features: dict | list[dict]) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Aligne les features sur EXPECTED_FEATURES, ajoute les manquantes (NaN), ignore les extra
    """
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
    # si on n’a pas encore accédé au modèle, tente un chargement paresseux
    try:
        _load_model_if_needed()
    except Exception:
        pass
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
        "expected_features": EXPECTED_FEATURES, # ordre exact d’entraînement
        "class_mapping": CLASS_MAPPING,
    }

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        _load_meta_if_needed()
        _load_model_if_needed()
        X, missing, extra = prepare_dataframe(req.features)
        proba_bad = float(_model.predict_proba(X)[:, 1][0])   # colonne 1 = classe "Refuser"
        yhat = int(proba_bad >= THRESHOLD) # 1 = Refuser, 0 = Accepter
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

@app.post("/explain", response_model=ExplainResponse)
def explain(req: ExplainRequest):
    try:
        _load_meta_if_needed()
        _load_model_if_needed()
        _load_explainer_if_needed()

        if _model is None:
            raise RuntimeError("Modèle non chargé, (SKIP_MODEL_LOAD=1 ?)")
        if _explainer is None:
            raise RuntimeError("Explainer non chargé")
        
        X, missing, extra = prepare_dataframe(req.features)

        # prédiction probabilité classe "Refuser" 1
        pred = float(_model.predict_proba(X)[:, 1][0])

        exp = _explainer(X, check_additivity=False)
        shap_row = np.array(exp.values)[0]
        base_value = float(np.array(exp.base_values)[0])

        feats = EXPECTED_FEATURES if EXPECTED_FEATURES else list(X.columns)
        vals = X.iloc[0].tolist()
        rows = []
        for f, v, s in zip(feats, vals, shap_row):
            rows.append({
                "feature": f,
                "value": float(v) if pd.notna(v) else None,
                "shap": float(s),
                "abs_shap": float(abs(s)),
            })
        rows.sort(key=lambda r: r["abs_shap"], reverse=True)
        rows = rows[:max(1, int(req.top_k))]

        return ExplainResponse(
            base_value=base_value,
            prediction=pred,
            contribution=rows,
            model_version=MODEL_VERSION,
            threshold=THRESHOLD,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))