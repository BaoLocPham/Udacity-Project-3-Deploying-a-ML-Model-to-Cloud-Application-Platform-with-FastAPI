"""
This script for Rest API
"""
import logging
from fastapi import FastAPI, HTTPException, status, Request
import pandas as pd
import numpy as np
import joblib
from model.api_models import CensusRequest, PredictResponse
from starter.features import get_features
from starter import inference_model

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


# Create a new instance of the FastAPI class
app = FastAPI(title="MLOps app", version='1.0.0')


@app.on_event("startup")
async def get_models():
    """Get the model"""
    logger.info("Loading model...")
    model_path, encoder_path, lb_path = \
        ("./deploy/model.joblib",
         "./deploy/encoder.joblib",
         "./deploy/lb_encoder.joblib")
    # load model, encoder and label encoder
    app.model = joblib.load(model_path)
    app.encoder = joblib.load(encoder_path)
    app.lb_encoder = joblib.load(lb_path)


@app.get("/")
async def get_items():
    return {"message": "Hello from BaoLocPham - MLOps - Render API!"}


@app.post("/")
async def predict(user_request: CensusRequest, request: Request):
    """Predict method"""
    array = np.array([[
                     user_request.age,
                     user_request.hours_per_week,
                     user_request.workclass,
                     user_request.education,
                     user_request.marital_status,
                     user_request.occupation,
                     user_request.relationship,
                     user_request.race,
                     user_request.sex,
                     user_request.native_country
                     ]])
    temp_df = pd.DataFrame(data=array, columns=get_features())
    app = request.app
    preds, pred_labels = inference_model.run(
        temp_df,
        model=app.model,
        encoder=app.encoder,
        lb=app.lb_encoder)
    if preds is None or pred_labels is None:
        # If the prediction failed, return a 500 error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed"
        )

    return PredictResponse(
        prediction=preds,
        class_name=pred_labels,
        success=True,
    )
