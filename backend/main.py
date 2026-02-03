import os
import logging
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import mlflow.sklearn
import boto3
from urllib.parse import urlparse
import pandas as pd
import sys
import time

# Import custom preprocessing classes and perform hack for pickle compatibility
import preprocessing
sys.modules['preprocessing'] = preprocessing
# Also expose classes in main scope just in case
from preprocessing import (
    normalize_tweet,
    punctuation_length_rate_transform,
    ProfanityLexiconFeaturizer
)

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

# Configuration from Environment Variables
MINIO_ENDPOINT = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
BUCKET_NAME = "mlflow-artifacts"

# Initialize S3 Client
s3_client = boto3.client(
    "s3",
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
)

class TweetRequest(BaseModel):
    text: str

class ModelPrediction(BaseModel):
    model_name: str
    prediction: str
    confidence: float = 0.0 # Placeholder if probing probability is possible

class PredictionResponse(BaseModel):
    predictions: List[ModelPrediction]

class PredictionResponse(BaseModel):
    predictions: List[ModelPrediction]

loaded_models = {}
preprocessor = None
last_loaded_time = 0
RELOAD_INTERVAL = 300  # Default 5 minutes, but logic will check on request

def load_resources():
    """
    Loads models and preprocessor from S3.
    """
    global loaded_models, preprocessor, last_loaded_time
    
    # 1. Load Models (Last 3 modified in 11/models/)
    models_prefix = "11/models/"
    logger.info(f"Looking for models in {models_prefix}")

    try:
        # We need to find the last 3 modified folders.
        # S3 does not give timestamps for folders (CommonPrefixes).
        # We must look for a specific file inside to determine the "modification time" of the model.
        # We'll look for 'artifacts/MLmodel' inside each prefix.
        
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix=models_prefix)
        
        candidates = []
        
        for page in pages:
            if 'Contents' not in page:
                continue
            
            for obj in page['Contents']:
                key = obj['Key']
                # Check if this object is the 'MLmodel' file
                if key.endswith('/artifacts/MLmodel'):
                    # The key structure is assumed to be: 11/models/<model_id>/artifacts/MLmodel
                    # We want to extract the model_id (folder name) and the timestamp
                    
                    # Split path
                    parts = key.split('/')
                    # parts: ['11', 'models', 'm-xxxxx', 'artifacts', 'MLmodel']
                    # We need enough parts
                    if len(parts) >= 5:
                        model_name = parts[2] # m-xxxxx
                        # Construct the URI for loading: .../artifacts
                        # e.g. s3://bucket/11/models/m-xxxxx/artifacts
                        model_uri = f"s3://{BUCKET_NAME}/{'/'.join(parts[:-1])}" 
                        
                        candidates.append({
                            'name': model_name,
                            'uri': model_uri,
                            'last_modified': obj['LastModified']
                        })
        
        # Sort by LastModified descending
        candidates.sort(key=lambda x: x['last_modified'], reverse=True)
        
        # Take top 3
        top_candidates = candidates[:3]
        logger.info(f"Found {len(candidates)} models, loading top {len(top_candidates)}")

        # Clear existing models if we want a fresh state (or update intelligently - here we just replace)
        new_loaded_models = {}
        
        # Tell MLflow how to connect to S3
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = MINIO_ENDPOINT
        os.environ["AWS_ACCESS_KEY_ID"] = ACCESS_KEY
        os.environ["AWS_SECRET_ACCESS_KEY"] = SECRET_KEY

        for cand in top_candidates:
            model_name = cand['name']
            model_uri = cand['uri']
            try:
                # Avoid reloading if already loaded? MLflow load might be cached or fast enough.
                # But 'loaded_models' holds the object.
                logger.info(f"Loading model {model_name} (modified: {cand['last_modified']})")
                model = mlflow.sklearn.load_model(model_uri)
                new_loaded_models[model_name] = model
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
        
        loaded_models = new_loaded_models

    except Exception as e:
        logger.error(f"Error loading models: {e}")


    # 2. Load Preprocessor
    # We try to find 'artifacts/preprocessing' in similar locations or a fixed location
    # Strategy: Look for 'preprocessing' artifact in the LATEST valid run in '11/' if possible,
    # or just search for whre it is.
    # User script: mlflow.sklearn.log_model(preprocessor, name="preprocessing")
    # This usually places it at: 11/<run_id>/artifacts/preprocessing
    
    if not preprocessor:
        logger.info("Searching for preprocessor in 11/")
        try:
             # Scan 11/ for 'artifacts/preprocessing/MLmodel'
             # Since we don't know the exact run ID where preprocessing is strictly 'best',
             # we will take the LATEST one we find in the entire experiment 11.
             
            paginator = s3_client.get_paginator('list_objects_v2')
            # Limit scope? 11/ might have many runs.
            # We hope it's not too huge.
            pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix="11/")
            
            prep_candidates = []
            
            for page in pages:
                if 'Contents' not in page:
                    continue
                for obj in page['Contents']:
                     if obj['Key'].endswith('artifacts/preprocessing/MLmodel'):
                         parts = obj['Key'].split('/')
                         # Expected: 11/<run_id>/artifacts/preprocessing/MLmodel
                         if len(parts) >= 5:
                             uri = f"s3://{BUCKET_NAME}/{'/'.join(parts[:-1])}"
                             prep_candidates.append({
                                 'uri': uri,
                                 'last_modified': obj['LastModified']
                             })

            if prep_candidates:
                # usage latest
                prep_candidates.sort(key=lambda x: x['last_modified'], reverse=True)
                best_prep = prep_candidates[0]
                logger.info(f"Loading preprocessor from {best_prep['uri']}")
                preprocessor = mlflow.sklearn.load_model(best_prep['uri'])
            else:
                logger.warning("No preprocessor found in 11/")
                
        except Exception as e:
            logger.error(f"Error loading preprocessor: {e}")

    last_loaded_time = time.time()
    logger.info("Resources reload complete.")

@app.on_event("startup")
async def startup_event():
    load_resources()

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: TweetRequest):
    global loaded_models, preprocessor, last_loaded_time
    
    # Reload logic: "load automatically every few minutes or load once the user submits a tweet"
    # To satisfy "load once the user submits", we should reload here.
    # To avoid excessive S3 hits, maybe we debounce?
    # But user asked for it. We will try to reload if it's been more than, say, 10 seconds?
    # Or just always reload candidates list (fast) and only download (slow) if changed?
    # For now, we'll reload resources if > 60 seconds have passed, OR if no models loaded.
    # User said: "load once the user sumbmits a tweet".
    # I will allow re-checking every time but maybe optimize later. 
    # Let's do a short check interval.
    
    if (time.time() - last_loaded_time > 60) or (not loaded_models):
        logger.info("Triggering reload on request")
        load_resources()

    if not loaded_models:
         raise HTTPException(status_code=503, detail="No models loaded")
         
    if not preprocessor:
        # Try one last time?
        pass # We logged error already
        # raise HTTPException(status_code=503, detail="Preprocessor not available")
        # If we really can't find it, we will likely fail in transformation

    results = []
    
    # Preprocess
    try:
        if preprocessor:
            input_df = pd.DataFrame({'description': [request.text]})
            # Transform
            logger.info("Transforming input")
            input_features = preprocessor.transform(input_df)
        else:
            # Fallback: Assume model might be a Pipeline expecting a DataFrame
            logger.warning("No preprocessor loaded. Attempting to pass raw DataFrame to model.")
            input_features = pd.DataFrame({'description': [request.text]})
    except Exception as e:
         logger.error(f"Preprocessing failed: {e}")
         raise HTTPException(status_code=400, detail=f"Preprocessing failed: {str(e)}")
    
    for name, model in loaded_models.items():
        try:
            # Predict
            pred = model.predict(input_features)[0]
            
            # Confidence
            confidence = 0.0
            if hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba(input_features)
                    confidence = float(max(proba[0]))
                except:
                    pass

            results.append(ModelPrediction(
                model_name=name,
                prediction=str(pred),
                confidence=confidence
            ))
        except Exception as e:
            logger.error(f"Prediction error with model {name}: {e}")
            results.append(ModelPrediction(
                model_name=name,
                prediction="Error",
                confidence=0.0
            ))

    return PredictionResponse(predictions=results)

@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": len(loaded_models)}
