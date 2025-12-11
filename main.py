from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import lightgbm as lgb
import uvicorn

app = FastAPI()

# -------------------------------------------------------
# Load LightGBM model (with safe fallback)
# -------------------------------------------------------
import joblib

# -------------------------------------------------------
# Load LightGBM model (joblib-serialized Booster)
# -------------------------------------------------------
try:
    model = joblib.load("models/entry_lgbm.pkl")
    print("[ULTRON-ML-V2] Model loaded successfully from models/entry_lgbm.pkl")
except Exception as e:
    print("[ULTRON-ML-V2] WARNING: entry_lgbm.pkl not loaded — using placeholder model")
    print("Details:", e)
    model = None


# -------------------------------------------------------
# Feature schema
# -------------------------------------------------------
class FeaturesV2(BaseModel):
    features: list[float]   # expects length ~42


# -------------------------------------------------------
# Predict endpoint
# -------------------------------------------------------
@app.post("/predict")
def predict(payload: FeaturesV2):
    # Convert to numpy
    X = np.array(payload.features).reshape(1, -1)

    # ---------------------------------------------------
    # Placeholder inference if model is missing
    # ---------------------------------------------------
    if model is None:
        pred = 0.5
    else:
        y_pred = model.predict(X)
        pred = float(y_pred[0])

    # ---------------------------------------------------
    # Build response
    # ---------------------------------------------------
    return {
        "ok": True,
        "raw": pred,
        "probUp": max(0, min(1, pred)),
        "probDown": max(0, min(1, 1 - pred)),
        "expectedPnL": pred * 0.60,
        "expectedHoldTime": 20 + pred * 10,
        "agreementScore": pred,
    }


# -------------------------------------------------------
# Main runner
# -------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)
