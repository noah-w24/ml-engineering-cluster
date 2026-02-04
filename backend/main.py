import os
import logging
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import mlflow
import pandas as pd
import numpy as np
import time
import re
import string
from sklearn.base import BaseEstimator, TransformerMixin
import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Violence Detection API")

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class TweetRequest(BaseModel):
    text: str

class ModelPrediction(BaseModel):
    model_name: str
    prediction: str
    confidence: float = 0.0 # Placeholder if probing probability is possible

class PredictionResponse(BaseModel):
    predictions: List[ModelPrediction]

loaded_models = {}
preprocessor = None
last_loaded_time = 0
RELOAD_INTERVAL = 300  # Default 5 minutes, but logic will check on request

def load_resources():
    """
    Loads end-to-end model pipelines from S3.
    Also loads the fitted preprocessor.
    """
    global loaded_models, preprocessor, last_loaded_time
    
    logger.info("Looking for final-model pipelines in experiment 11/")
    try:
        mlflow.set_tracking_uri("http://mlflow.mlops.svc.cluster.local:5000")
        preprocessor = mlflow.sklearn.load_model("s3://mlflow-artifacts/11/models/m-bfb9b230e0cc4e53b08d6b65fa8170de/artifacts")
        loaded_models["LogReg_balanced"]=mlflow.sklearn.load_model("s3://mlflow-artifacts/11/models/m-730ac1ef738a45488e5432682eb137e0/artifacts")
        loaded_models["RF_balanced"]=mlflow.sklearn.load_model("s3://mlflow-artifacts/11/models/m-9d4e688d042746c89450ad7e34fffff7/artifacts")
        loaded_models["HistGB"]=mlflow.sklearn.load_model("s3://mlflow-artifacts/11/models/m-37eb9fcd8c8c4b269c4bbae8bf2b48d7/artifacts")
    except Exception as e:
        logger.error(f"Error loading models: {e}")

    last_loaded_time = time.time()
    logger.info("Resources reload complete.")

@app.on_event("startup")
async def startup_event():
    load_resources()

def _split_camel_case(token: str) -> str:
    token = re.sub(r"([a-z])([A-Z])", r"\1 \2", token)
    token = re.sub(r"([A-Za-z])(\d)", r"\1 \2", token)
    token = re.sub(r"(\d)([A-Za-z])", r"\1 \2", token)
    token = token.replace("_", " ")
    return token


def normalize_tweet(text: str) -> str:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""

    text = str(text)
    text = _URL_RE.sub(" HTTPURL ", text)
    text = _USER_RE.sub(" @USER ", text)

    def _hashtag_repl(m: re.Match) -> str:
        tag = m.group(1)
        tag = _split_camel_case(tag)
        return f" {tag} "

    text = _HASHTAG_RE.sub(_hashtag_repl, text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_for_lexicon(text: str) -> str:
    t = normalize_tweet(text)
    t = t.lower().translate(_LEET_MAP)
    t = re.sub(r"(?<=\w)[^\w\s]+(?=\w)", "", t)
    return t


class ProfanityLexiconFeaturizer(BaseEstimator, TransformerMixin):
    """
    Outputs 3 numeric features:
      - profane_count
      - profane_ratio
      - has_profane
    """
    def __init__(self, lexicon=None, show_progress: bool = True):
        self.lexicon = lexicon or GER_PROFANITY
        self.show_progress = show_progress

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        s = X.iloc[:, 0] if isinstance(X, pd.DataFrame) else pd.Series(np.asarray(X).ravel())
        out = np.zeros((len(s), 3), dtype=np.float32)

        it = enumerate(s)
        if self.show_progress and len(s) >= 5000:
            it = tqdm(it, total=len(s), desc="lexicon features", unit="row", leave=False)

        for i, txt in it:
            t = normalize_for_lexicon(txt)
            tokens = _WORD_RE.findall(t)
            if not tokens:
                continue
            hits = sum(1 for w in tokens if w in self.lexicon)
            out[i, 0] = hits
            out[i, 1] = hits / max(len(tokens), 1)
            out[i, 2] = 1.0 if hits > 0 else 0.0

        return out


_PUNCT_SET = set(string.punctuation)


def punctuation_length_rate(text: str) -> float:
    t = normalize_tweet(text)
    if not t:
        return 0.0
    punct = sum(1 for ch in t if ch in _PUNCT_SET)
    return punct / max(len(t), 1)


def punctuation_length_rate_transform(X):
    s = X.iloc[:, 0] if isinstance(X, pd.DataFrame) else pd.Series(np.asarray(X).ravel())
    return s.apply(punctuation_length_rate).astype("float32").to_numpy().reshape(-1, 1)


def _log_stage(msg: str):
    print(f"\n=== {msg} ===", flush=True)

GER_PROFANITY = {
    "schlagen", "erschlagen", "töten", "ermorden", "verprügeln", "boxen",
    "kämpfen", "angreifen", "treten", "morden", "prügeln", "hauen",
}
_LEET_MAP = str.maketrans({
    "@": "a", "4": "a",
    "1": "i", "!": "i",
    "0": "o",
    "$": "s", "5": "s",
    "3": "e",
})

_WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)

_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", flags=re.IGNORECASE)
_USER_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")

def preprocess_text(text: str):
    global preprocessor
    if preprocessor is None:
        logger.warning("Preprocessor is not loaded! Falling back to raw text (likely to fail if model expects features).")
        raise HTTPException(status_code=503, detail="Preprocessor not initialized. Check MLflow connection and model URIs.")

    df = pd.DataFrame({
        "description": [text],
    })

    try:
        features = preprocessor.transform(df)
    except Exception as e:
        logger.error(f"Preprocessing transform failed. Input: '{text[:50]}...'. Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Preprocessing error: {str(e)}")
    
    return features

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: TweetRequest):
    global loaded_models, last_loaded_time
    
    # Reload logic: check every 60 seconds or if models are missing
    if (time.time() - last_loaded_time > 60) or (not loaded_models):
        logger.info("Triggering reload on request")
        load_resources()

    if not loaded_models:
        raise HTTPException(status_code=503, detail="No models available. Ensure final-model pipelines exist in S3.")

    results = []
    
    for name, model in loaded_models.items():
        try:
            logger.info(f"Predicting with model {name}")
            features = preprocess_text(request.text)
            
            # HistGradientBoostingClassifier does not support sparse input
            if "HistGB" in name and hasattr(features, "toarray"):
                features = features.toarray()
            
            print(f"LENGTH OF REQUEST: {features.shape}")
            pred = model.predict(features)[0]
            
            confidence = 0.0
            if hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba(features)
                    confidence = float(max(proba[0]))
                except Exception as e:
                    logger.warning(f"Could not get probability for {name}: {e}")

            results.append(ModelPrediction(
                model_name=name,
                prediction=str(pred),
                confidence=confidence
            ))
        except Exception as e:
            logger.error(f"Prediction error with model {name}: {e}", exc_info=True)
            results.append(ModelPrediction(
                model_name=name,
                prediction="Error",
                confidence=0.0
            ))

    return PredictionResponse(predictions=results)

@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": len(loaded_models)}
