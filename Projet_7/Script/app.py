# Imports
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware

import os
import time
import shap
import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from pydantic import BaseModel, Field
from typing import Optional, Tuple, List, Dict, Any

# Config & chemins artefacts
BASE_DIR = Path(__file__).resolve().parent
ART_DIR  = Path(os.getenv("ART_DIR", str(BASE_DIR / "artifacts")))
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(ART_DIR / "model.joblib")))
META_PATH  = Path(os.getenv("ARTIFACTS_PATH", str(ART_DIR / "artifacts.json")))
OVERRIDE_THR = os.getenv("THRESHOLD")
SKIP_MODEL   = os.getenv("SKIP_MODEL_LOAD", "0") == "1"  # utile pour la CI
BG_PATH = ART_DIR / "shap_background.parquet"
EVALUATION_PATH = ART_DIR / "shap_evaluation.parquet"

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
GLOBAL_MEAN_ABS = None
GLOBAL_MEAN_ABS_TSTAMP = 0.0
GLOBAL_TTL_SECONDS = 600
MAX_BG_FOR_SHAP = 200

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
        bg = pd.read_parquet(BG_PATH)[EXPECTED_FEATURES]

        for c in EXPECTED_FEATURES:
            if c not in bg.columns:
                bg[c] = np.nan
        bg = bg[EXPECTED_FEATURES].apply(pd.to_numeric, errors="coerce").astype(np.float32)
        if len(bg) > MAX_BG_FOR_SHAP:
            bg = bg.sample(n=MAX_BG_FOR_SHAP, random_state=42)
        return bg
    
    med = pd.Series({c: 0.0 for c in EXPECTED_FEATURES}, dtype=np.float32)
    return pd.DataFrame([med])[EXPECTED_FEATURES]


def _load_explainer_if_needed():
    """
    Charge l'explainer SHAP au premier accès.
    """
    global _explainer
    _load_model_if_needed()
    _load_meta_if_needed()

    if _explainer is not None:
        return

    _explainer = shap.TreeExplainer(_model)
    # bg = _load_background_if_needed()
    # masker = shap.maskers.Independent(bg)
    # _explainer = shap.TreeExplainer(
    #     _model,
    #     masker=masker,
    #     algorithm="tree",
    #     model_output="probability",
    #     feature_perturbation="interventional",
    # )


def _proba_refuser(X: pd.DataFrame) -> np.ndarray:
    """
    Retourne p(y=1) (classe 'Refuser'), en sélectionnant la bonne colonne
    selon model.classes_.
    """
    proba = _model.predict_proba(X)
    if hasattr(_model, "classes_"):
        idx1 = int(np.where(_model.classes_ == 1)[0][0])
    else:
        idx1 = 1  # fallback
    return proba[:, idx1]


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


def _shap_values_and_base(X: pd.DataFrame)-> tuple[np.ndarray, float]:
    """
    Return (sv, base_value) for class=1 in a version-agnostic way.

    sv : (n_samples, n_features) raw SHAP values
    base_value : scalar float (flattened)
    """
    # New API path
    try:
        exp = _explainer(X, check_additivity=False)
        vals = np.array(exp.values)
        base = exp.base_values

        # vals can be (n, d) or (n, d, C). If 3D, select class=1.
        if vals.ndim == 3:
            # shape: (n_samples, n_features, n_classes)
            cidx = 1 if vals.shape[-1] > 1 else 0
            vals = vals[:, :, cidx]

        sv = vals  # now (n, d)
        base_value = _flatten_scalar(base)  # robust scalar
        return sv, base_value
    except Exception:
        # Old API path
        sv_any = _explainer.shap_values(X)  # list per class or ndarray
        base_any = _explainer.expected_value

        if isinstance(sv_any, list):
            sv = np.array(sv_any[1] if len(sv_any) > 1 else sv_any[0])
        else:
            sv = np.array(sv_any)

        base_value = _flatten_scalar(base_any)
        return sv, base_value


def _sanitize_row_for_shap(X_row: pd.DataFrame) -> pd.DataFrame:
    X_row = X_row.apply(pd.to_numeric, errors="coerce").astype(np.float32)
    try:
        bg = _load_background_if_needed()
        med = bg.median(numeric_only=True)
        X_row = X_row.fillna(med)
    except Exception:
        X_row = X_row.fillna(0.0)
    return X_row


def _flatten_scalar(x) -> float:
    """
    Turn anything (scalar/list/np.ndarray) into a single float by flattening and taking [0].
    """
    arr = np.array(x)
    return float(arr.reshape(-1)[0])


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
    shap_ready = _explainer is not None
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
        "paths": {
            "base_dir": str(BASE_DIR),
            "art_dir": str(ART_DIR),
            "model_path": str(MODEL_PATH),
            "meta_path": str(META_PATH),
            "bg_path": str(BG_PATH),
        },
        "exists": {
            "art_dir": ART_DIR.exists(),
            "model": MODEL_PATH.exists(),
            "meta": META_PATH.exists(),
            "bg": BG_PATH.exists(),
        }
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
        proba_bad = float(_proba_refuser(X)[0])   # colonne 1 = classe "Refuser"
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
        probas = _proba_refuser(X)
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


@app.post("/explain", response_model=ExplainResponse, tags=["Scoring"])
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
        pred = float(_proba_refuser(X)[0])

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


@app.get("/explain_global", tags=["Scoring"])
def explain_global(top_k: int = 20):
    _load_meta_if_needed()
    _load_model_if_needed()

    top_k = max(1, int(top_k))
    now = time.time()

    # 0) Si on a un cache frais, renvoie-le instantanément
    global GLOBAL_MEAN_ABS, GLOBAL_MEAN_ABS_TSTAMP
    if GLOBAL_MEAN_ABS is not None and (now - GLOBAL_MEAN_ABS_TSTAMP) < GLOBAL_TTL_SECONDS:
        s = GLOBAL_MEAN_ABS.sort_values(ascending=False).head(top_k)
        return {"class_explained": 1,
                "importances": [{"feature": k, "mean_abs_shap": float(v)} for k, v in s.items()],
                "source": "cache"}

    # 1) precomputed s'il existe
    try:
        if EVALUATION_PATH.exists():
            df_eval = pd.read_parquet(EVALUATION_PATH)
            cols = {c.lower(): c for c in df_eval.columns}
            if "feature" in cols and ("mean_abs_shap" in cols or "mean_abs" in cols):
                key = cols.get("mean_abs_shap", cols.get("mean_abs"))
                df_eval = df_eval.rename(columns=str.lower)
                s = pd.Series(df_eval[key].values, index=df_eval["feature"].values).astype(float)
                GLOBAL_MEAN_ABS = s
                GLOBAL_MEAN_ABS_TSTAMP = time.time()
                s2 = s.sort_values(ascending=False).head(top_k)
                return {"class_explained": 1,
                        "importances": [{"feature": k, "mean_abs_shap": float(v)} for k, v in s2.items()],
                        "source": "precomputed"}
    except Exception:
        pass

    # 2) SHAP sur background réduit + mise en cache
    try:
        _load_explainer_if_needed()
        bg = _load_background_if_needed()  # déjà réduit à MAX_BG_FOR_SHAP
        sv_bg, _ = _extract_shap_for_class1(bg)  # robuste à la version SHAP
        mean_abs = np.abs(sv_bg).mean(axis=0)
        s = pd.Series(mean_abs, index=bg.columns)
        GLOBAL_MEAN_ABS = s
        GLOBAL_MEAN_ABS_TSTAMP = time.time()
        s2 = s.sort_values(ascending=False).head(top_k)
        return {"class_explained": 1,
                "importances": [{"feature": k, "mean_abs_shap": float(v)} for k, v in s2.items()],
                "source": "shap_background"}
    except Exception:
        pass

    # 3) Fallback instantané: feature_importances_
    try:
        if hasattr(_model, "feature_importances_"):
            imp = np.asarray(_model.feature_importances_, dtype=float)
            idx = EXPECTED_FEATURES if EXPECTED_FEATURES else list(range(len(imp)))
            s = pd.Series(imp, index=idx)
            GLOBAL_MEAN_ABS = s
            GLOBAL_MEAN_ABS_TSTAMP = time.time()
            s2 = s.sort_values(ascending=False).head(top_k)
            return {"class_explained": 1,
                    "importances": [{"feature": k, "mean_abs_shap": float(v)} for k, v in s2.items()],
                    "source": "model_feature_importances"}
    except Exception:
        pass

    return {"class_explained": 1, "importances": [], "source": "none"}


@app.post("/explain_local", tags=["Scoring"])
def explain_local(req: ExplainRequest):
    try:
        _load_meta_if_needed()
        _load_model_if_needed()
        _load_explainer_if_needed()

        X, _, _ = prepare_dataframe(req.features)
        X = _sanitize_row_for_shap(X)

        # Probability for class=1 (decision still based on threshold)
        proba = float(_proba_refuser(X)[0])

        # SHAP raw values (robust across versions)
        sv, base_value = _shap_values_and_base(X)   # sv: (1, n_features)
        shap_row = sv[0]                                # 1D length n_features

        feats = EXPECTED_FEATURES if EXPECTED_FEATURES else list(X.columns)
        vals = X.iloc[0].tolist()

        rows = []
        for f, v, s in zip(feats, vals, shap_row):
            s = float(s)
            rows.append({
                "feature": f,
                "value": None if pd.isna(v) else float(v),
                "shap_value": s,                         # + pushes to Refuser, - to Accepter
                "abs_shap": abs(s),
                "direction": "push_refuser" if s > 0 else "push_accepter",
            })
        rows.sort(key=lambda r: r["abs_shap"], reverse=True)
        rows = rows[:max(1, int(req.top_k))]

        return {
            "base_value": _flatten_scalar(base_value),   # safe scalar
            "probability": proba,
            "top_contributions": rows,
            "model_version": MODEL_VERSION,
            "threshold": THRESHOLD,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))