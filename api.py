import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = 'artifacts/model.joblib'
FEATURE_COLS = [
    'num_transactions',
    'total_debit',
    'total_credit',
    'has_rent',
    'has_tesco',
    'has_restaurant',
    'has_salary',
    'has_utility',
    'has_transport',
]

app = FastAPI(title='Credit Risk Prediction API')

try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    raise RuntimeError(
        f"Model not found at '{MODEL_PATH}'. "
        "Run `python pipeline.py` then `python train_model.py` first."
    )


class PredictRequest(BaseModel):
    customer_id: str
    num_transactions: int
    total_debit: float
    total_credit: float
    has_rent: bool = False
    has_tesco: bool = False
    has_restaurant: bool = False
    has_salary: bool = False
    has_utility: bool = False
    has_transport: bool = False


class PredictResponse(BaseModel):
    customer_id: str
    probability: float
    prediction: int


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/predict', response_model=PredictResponse)
def predict(request: PredictRequest):
    features = np.array([[
        request.num_transactions,
        request.total_debit,
        request.total_credit,
        int(request.has_rent),
        int(request.has_tesco),
        int(request.has_restaurant),
        int(request.has_salary),
        int(request.has_utility),
        int(request.has_transport),
    ]])

    try:
        probability = float(model.predict_proba(features)[0][1])
        prediction = int(model.predict(features)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return PredictResponse(
        customer_id=request.customer_id,
        probability=round(probability, 4),
        prediction=prediction,
    )
