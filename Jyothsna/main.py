import os
import joblib
import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from Jyothsna.schemas import (
    FeeDefaultRequest, FeeDefaultResponse,
    AttendanceAnomalyRequest, AttendanceAnomalyResponse
)

app = FastAPI(
    title="Kalnet AI — Anomaly & Default Predictor API",
    description="API to wrap Fee Default and Attendance Anomaly models for the web dashboard.",
    version="1.0.0"
)

# Paths relative to the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEE_MODEL_PATH = os.path.join(BASE_DIR, "Are_Samhith", "models", "model_v2.pkl")
FEE_THRESHOLD_PATH = os.path.join(BASE_DIR, "Are_Samhith", "models", "threshold_v2.pkl")
ATTENDANCE_MODEL_PATH = os.path.join(BASE_DIR, "Om", "models", "attendance_model.pkl")

# Global variables for models
models = {}

@app.on_event("startup")
def load_models():
    """Load models into memory on startup."""
    try:
        if os.path.exists(FEE_MODEL_PATH):
            models["fee_model"] = joblib.load(FEE_MODEL_PATH)
            print(f"Loaded Fee Default model from {FEE_MODEL_PATH}")
        
        if os.path.exists(FEE_THRESHOLD_PATH):
            models["fee_threshold"] = joblib.load(FEE_THRESHOLD_PATH)
            print(f"Loaded Fee threshold from {FEE_THRESHOLD_PATH}")
        else:
            models["fee_threshold"] = 0.40 # Default fallback
            
        if os.path.exists(ATTENDANCE_MODEL_PATH):
            with open(ATTENDANCE_MODEL_PATH, "rb") as f:
                models["attendance_model"] = pickle.load(f)
            print(f"Loaded Attendance model from {ATTENDANCE_MODEL_PATH}")
            
    except Exception as e:
        print(f"Error during model loading: {e}")

@app.get("/")
async def root():
    return {
        "message": "Kalnet AI Predictor API is running",
        "models_loaded": list(models.keys())
    }

@app.post("/predict/fee-default", response_model=FeeDefaultResponse)
async def predict_fee_default(request: FeeDefaultRequest):
    if "fee_model" not in models:
        raise HTTPException(status_code=503, detail="Fee default model not loaded")
    
    # Convert Pydantic model to dict, then to DataFrame
    data_dict = request.dict()
    input_df = pd.DataFrame([data_dict])
    
    # Feature columns as expected by the model (Order matters!)
    FEATURE_COLS = [
        "income_encoded", "transport_user", "sibling_count",
        "current_status", "current_defaulted", "is_low_income",
        "has_many_siblings", "was_late_or_worse", "current_term"
    ]
    
    try:
        # Get probability for the "Default" class (index 1)
        probs = models["fee_model"].predict_proba(input_df[FEATURE_COLS])[:, 1]
        prob = float(probs[0])
        
        threshold = models.get("fee_threshold", 0.40)
        
        def get_category(p):
            if p >= 0.6: return "HIGH"
            elif p >= 0.3: return "MEDIUM"
            else: return "LOW"
        
        return FeeDefaultResponse(
            student_id=request.student_id,
            default_probability=round(prob * 100, 2),
            risk_category=get_category(prob),
            will_default_pred=int(prob >= threshold)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/attendance-anomaly", response_model=AttendanceAnomalyResponse)
async def predict_attendance_anomaly(request: AttendanceAnomalyRequest):
    if "attendance_model" not in models:
        raise HTTPException(status_code=503, detail="Attendance model not loaded")
    
    data_dict = request.dict()
    input_df = pd.DataFrame([data_dict])
    
    # Feature columns as expected by the model
    FEATURES = [
        "attendance_rate", "longest_absence_streak",
        "absence_in_last_30_days", "day_of_week_variance"
    ]
    
    try:
        raw = models["attendance_model"].predict(input_df[FEATURES])
        scores = models["attendance_model"].decision_function(input_df[FEATURES])
        
        # Isolation Forest: -1 is anomalous, 1 is normal
        predicted_anomaly = 1 if raw[0] == -1 else 0
        anomaly_score = float(-scores[0]) # Flipped: higher is more anomalous
        
        def get_risk_level(score):
            if score > 0.15: return "High"
            elif score > 0.05: return "Medium"
            else: return "Low"
            
        return AttendanceAnomalyResponse(
            student_id=request.student_id,
            predicted_anomaly=predicted_anomaly,
            anomaly_score=round(anomaly_score, 4),
            risk_level=get_risk_level(anomaly_score)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
