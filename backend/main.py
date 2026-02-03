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

loaded_models = {}
last_loaded_time = 0
RELOAD_INTERVAL = 300  # Default 5 minutes, but logic will check on request

def load_resources():
    """
    Loads end-to-end model pipelines from S3.
    Each pipeline includes preprocessing + classifier, so raw text can be passed directly.
    Looks for 'final-model' artifacts in the latest experiment runs.
    """
    global loaded_models, last_loaded_time
    
    logger.info("Looking for final-model pipelines in experiment 11/")

    try:
        # Tell MLflow how to connect to S3
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = MINIO_ENDPOINT
        os.environ["AWS_ACCESS_KEY_ID"] = ACCESS_KEY
        os.environ["AWS_SECRET_ACCESS_KEY"] = SECRET_KEY
        
        # Scan for 'final-model/MLmodel' artifacts in all runs under experiment 11/
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix="11/")
        
        candidates = []
        
        for page in pages:
            if 'Contents' not in page:
                continue
            
            for obj in page['Contents']:
                key = obj['Key']
                # Look for final-model artifacts: 11/<run_id>/final-model/MLmodel
                if 'final-model' in key and key.endswith('/MLmodel'):
                    # Extract run_id: 11/<run_id>/final-model/MLmodel
                    parts = key.split('/')
                    if len(parts) >= 4 and parts[2] != 'models':  # Avoid registered models folder
                        run_id = parts[1]
                        # Construct the URI: s3://bucket/11/<run_id>/final-model
                        model_uri = f"s3://{BUCKET_NAME}/{'/'.join(parts[:-1])}"
                        
                        candidates.append({
                            'run_id': run_id,
                            'uri': model_uri,
                            'last_modified': obj['LastModified']
                        })
        
        # Sort by LastModified descending and take top 3
        candidates.sort(key=lambda x: x['last_modified'], reverse=True)
        top_candidates = candidates[:3]
        logger.info(f"Found {len(candidates)} final-model pipelines, loading top {len(top_candidates)}")

        new_loaded_models = {}

        for cand in top_candidates:
            run_id = cand['run_id']
            model_uri = cand['uri']
            try:
                logger.info(f"Loading final-model from run {run_id} (modified: {cand['last_modified']})")
                # Load the end-to-end pipeline
                model = mlflow.sklearn.load_model(model_uri)
                # Use run_id as the model identifier (shows which training run produced it)
                new_loaded_models[run_id] = model
                logger.info(f"Successfully loaded final-model from {run_id}")
            except Exception as e:
                logger.error(f"Failed to load final-model from {run_id}: {e}")
        
        loaded_models = new_loaded_models
        
        if not loaded_models:
            logger.warning("No final-model pipelines loaded. Falling back to individual models from 11/models/")
            # Fallback: try loading old-style individual models
            models_prefix = "11/models/"
            pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix=models_prefix)
            fallback_candidates = []
            
            for page in pages:
                if 'Contents' not in page:
                    continue
                for obj in page['Contents']:
                    key = obj['Key']
                    if key.endswith('/artifacts/MLmodel'):
                        parts = key.split('/')
                        if len(parts) >= 5:
                            model_name = parts[2]  # m-xxxxx
                            model_uri = f"s3://{BUCKET_NAME}/{'/'.join(parts[:-1])}"
                            fallback_candidates.append({
                                'name': model_name,
                                'uri': model_uri,
                                'last_modified': obj['LastModified']
                            })
            
            fallback_candidates.sort(key=lambda x: x['last_modified'], reverse=True)
            for cand in fallback_candidates[:3]:
                try:
                    logger.info(f"Loading fallback model {cand['name']}")
                    model = mlflow.sklearn.load_model(cand['uri'])
                    new_loaded_models[cand['name']] = model
                except Exception as e:
                    logger.error(f"Failed to load fallback model {cand['name']}: {e}")
            
            loaded_models = new_loaded_models

    except Exception as e:
        logger.error(f"Error loading models: {e}")

    last_loaded_time = time.time()
    logger.info("Resources reload complete.")

@app.on_event("startup")
async def startup_event():
    load_resources()

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
            # Models are end-to-end pipelines that handle preprocessing internally
            # Pass raw text directly; the pipeline will preprocess and classify
            logger.info(f"Predicting with model {name}")
            pred = model.predict([request.text])[0]
            
            # Get confidence if available
            confidence = 0.0
            if hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba([request.text])
                    confidence = float(max(proba[0]))
                except Exception as e:
                    logger.debug(f"Could not get probability for {name}: {e}")

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
