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

def get_latest_experiment_id() -> str | None:
    """
    Finds the highest integer folder in the bucket root.
    """
    try:
        response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Delimiter='/')
        if 'CommonPrefixes' not in response:
            return None
        
        experiment_ids = []
        for prefix in response['CommonPrefixes']:
            # Prefix is like "9/"
            dir_name = prefix['Prefix'].strip('/')
            if dir_name.isdigit():
                experiment_ids.append(int(dir_name))
        
        if not experiment_ids:
            return None
            
        return str(max(experiment_ids))
    except Exception as e:
        logger.error(f"Error listing experiments: {e}")
        return None

def load_models():
    """
    Loads models from the latest experiment ID.
    Assumes structure: bucket/{experiment_id}/models/m-{hash}/
    """
    global loaded_models
    experiment_id = get_latest_experiment_id()
    if not experiment_id:
        logger.warning("No experiment folders found.")
        return

    models_prefix = f"{experiment_id}/models/"
    logger.info(f"Looking for models in {models_prefix}")

    try:
        # List "files" (folders) inside models/
        response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=models_prefix, Delimiter='/')
        
        if 'CommonPrefixes' not in response:
            logger.warning(f"No model folders found in {models_prefix}")
            return

        for prefix in response['CommonPrefixes']:
            # model_dir is like "9/models/m-46557.../"
            model_dir = prefix['Prefix']
            model_name = model_dir.strip('/').split('/')[-1] # e.g. m-46557...
            
            # Construct S3 URI for MLflow
            # Error log shows: "failures occurred while downloading... artifacts from s3://.../artifacts"
            # And "File model ... Not Found"
            # This means it's looking for a FOLDER named "model" inside the path we gave it.
            # But the user said: ".../artifacts/MLmodel"
            # This means the folder we are pointing to IS the model folder.
            
            # Previous attempt: model_uri = f"s3://{BUCKET_NAME}/{model_dir}artifacts/model"
            # That caused it to look for bucket/.../artifacts/model/MLmodel
            
            # The error "File model ... Not Found" suggests that mlflow is treating the last part of the path as the artifact name.
            # BUT wait, the user confirmed: ".../artifacts/MLmodel" exists directly inside `artifacts`.
            
            # So the correct "model uri" (the folder containing MLmodel) is just .../artifacts
            model_uri = f"s3://{BUCKET_NAME}/{model_dir}artifacts"
            
            try:
                logger.info(f"Loading model from {model_uri}")
                # Tell MLflow how to connect to S3
                os.environ["MLFLOW_S3_ENDPOINT_URL"] = MINIO_ENDPOINT
                os.environ["AWS_ACCESS_KEY_ID"] = ACCESS_KEY
                os.environ["AWS_SECRET_ACCESS_KEY"] = SECRET_KEY
                
                model = mlflow.sklearn.load_model(model_uri)
                loaded_models[model_name] = model
                logger.info(f"Successfully loaded {model_name}")
            except Exception as e:
                logger.error(f"Failed to load model from {model_uri}: {e}")

    except Exception as e:
        logger.error(f"Error exploring models directory: {e}")

@app.on_event("startup")
async def startup_event():
    load_models()

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: TweetRequest):
    if not loaded_models:
        # Try loading again if empty (maybe started before minio was ready)
        load_models()
        if not loaded_models:
            raise HTTPException(status_code=503, detail="No models available")

    results = []
    
    # Wrap text in a structure expected by sklearn pipelines (usually list or Series)
    input_data = [request.text]
    
    for name, model in loaded_models.items():
        try:
            # Predict
            # Note: This assumes the model is a Pipeline that accepts raw text.
            # If the model expects specific features, this will fail.
            pred = model.predict(input_data)[0]
            
            # Try to get probability if available
            confidence = 0.0
            if hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba(input_data)
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
