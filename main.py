from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import uvicorn
import pandas as pd
import joblib
import os

app = FastAPI()

# --------------------------------------------------------------------
# LOAD LIGHTGBM MODEL
# --------------------------------------------------------------------
try:
    model = joblib.load("models/entry_lgbm.pkl")
    print("[ULTRON-ML-V2] Model loaded successfully from models/entry_lgbm.pkl")
except Exception as e:
    print("[ULTRON-ML-V2] WARNING: entry_lgbm.pkl not loaded — using placeholder model")
    print("Details:", e)
    model = None

# --------------------------------------------------------------------
# BASE REQUEST STRUCTURES
# --------------------------------------------------------------------
class EntryRequest(BaseModel):
    symbol: str
    side: str
    tf: str
    combinedScore: float
    qualityScore: float
    atrPct: float

# --------------------------------------------------------------------
# /predict (legacy endpoint – preserved for compatibility)
# --------------------------------------------------------------------
@app.post("/predict")
def predict(payload: dict):

    features = payload.get("features", [])

    if not isinstance(features, list):
        return {"ok": False, "error": "Invalid format"}

    X = np.array(features).reshape(1, -1)

    if model is None:
        pred = 0.5
    else:
        try:
            pred = float(model.predict(X)[0])
        except Exception as e:
            print("[ULTRON-ML-V2][PREDICT] ERROR:", e)
            pred = 0.5

    return {
        "ok": True,
        "probUp": pred,
        "probDown": 1 - pred,
        "expectedPnL": pred * 0.60,
        "expectedHoldTime": 20 + pred * 10,
        "agreementScore": pred,
    }

# --------------------------------------------------------------------
# /score — FULL ULTRON ML-V2 RECONSTRUCTION PIPELINE (Option B)
# --------------------------------------------------------------------
@app.post("/score")
async def score(req: EntryRequest):

    if model is None:
        return {
            "ok": True,
            "probUp": 0.5,
            "probDown": 0.5,
            "expectedPnL": 0.0,
            "expectedHoldTime": 0.0,
            "agreementScore": 0.0,
            "note": "placeholder model active"
        }

    # ---------------------------------------------------------------
    # DEFAULTS for missing upstream fields (Option B)
    # ---------------------------------------------------------------
    defaults = {
        "regimeScore": 0.0,
        "underlyingPriceLive": 0.0,
        "trendScore": 0.0,
        "uwFlowScore": 0.0,
        "uwCallPutRatio": 0.0,
        "uwNetPremium": 0.0,
        "twScore": 0.0,
        "directionBiasScore": 0.0,
        "directionBiasAligned": 0,

        "eventSeverity": 0.0,
        "eventMinutesToEvent": 999,
        "eventIsMacroDay": 0,
        "eventIsInWindow": 0,

        # categorical defaults
        "sector": "unknown",
        "mode": "paper",
        "regimeState": "neutral",
        "trendLabel": "neutral",
        "uwFlowBias": "neutral",
        "twBias": "neutral",
        "directionBiasLabel": "neutral",
        "eventTag": "none",
    }

    # ---------------------------------------------------------------
    # BUILD BASE RAW ROW
    # ---------------------------------------------------------------
    row = {
        "combinedScore": req.combinedScore,
        "qualityScore": req.qualityScore,
        "atrPct": req.atrPct,
        "side": req.side.lower(),
        "mode": defaults["mode"],
        "sector": defaults["sector"],
        **{k: v for k, v in defaults.items() if k not in ["mode", "sector"]},
    }

    df = pd.DataFrame([row])

    # ---------------------------------------------------------------
    # ONE-HOT ENCODE CATEGORICAL VARIABLES
    # ---------------------------------------------------------------
    categorical_cols = [
        "side", "mode", "sector", "regimeState",
        "trendLabel", "uwFlowBias", "twBias",
        "directionBiasLabel", "eventTag"
    ]

    for col in categorical_cols:
        dummies = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

    # ---------------------------------------------------------------
    # LOAD TRAINING FEATURE ORDER (47 FEATURES)
    # ---------------------------------------------------------------
    feature_path = "models/ultron_v3_entry_features.txt"
    feature_list = []

    with open(feature_path, "r") as f:
        for line in f:
            feature = line.strip()
            if feature:
                feature_list.append(feature)

    # ---------------------------------------------------------------
    # ALIGN DATAFRAME TO EXACT TRAINING FEATURE VECTOR
    # ---------------------------------------------------------------
    # Add missing columns
    for col in feature_list:
        if col not in df.columns:
            df[col] = 0

    # Remove extra columns
    df = df[feature_list]

    # ---------------------------------------------------------------
    # RUN LIGHTGBM PREDICTION
    # ---------------------------------------------------------------
    try:
        X = df.values
        pred = float(model.predict(X)[0])
    except Exception as e:
        print("[ULTRON-ML-V2][SCORE] ERROR:", e)
        return {"ok": False, "error": str(e)}

    probUp = pred
    probDown = 1 - pred
    expectedPnL = probUp - probDown
    expectedHoldTime = 30 * probUp
    agreementScore = probUp

    # ---------------------------------------------------------------
    # RETURN ML-V2 OUTPUT
    # ---------------------------------------------------------------
    return {
        "ok": True,
        "probUp": float(probUp),
        "probDown": float(probDown),
        "expectedPnL": float(expectedPnL),
        "expectedHoldTime": float(expectedHoldTime),
        "agreementScore": float(agreementScore),
    }


# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
