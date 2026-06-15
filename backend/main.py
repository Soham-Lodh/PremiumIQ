from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import os

app = FastAPI(title="Insurance Premium Predictor API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Load artifacts ───────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

try:
    model = joblib.load(ARTIFACTS_DIR / "model.joblib")
    scaler_obj = joblib.load(ARTIFACTS_DIR / "scaler.joblib")
    MODEL_LOADED = True
except Exception as e:
    print(f"WARNING: Could not load model artifacts: {e}")
    MODEL_LOADED = False


# ─── Schemas ──────────────────────────────────────────────────────────────────
class PredictionRequest(BaseModel):
    age: int = Field(..., ge=18, le=100)
    gender: str  # Male / Female
    marital_status: str  # Married / Unmarried
    number_of_dependants: int = Field(..., ge=0, le=10)
    income_lakhs: float = Field(..., ge=0, le=200)
    employment_status: str  # Salaried / Self-Employed / Freelancer
    bmi_category: str  # Normal / Overweight / Obesity / Underweight
    smoking_status: str  # No Smoking / Occasional / Regular
    physical_activity: str  # Low / Medium / High
    stress_level: str  # Low / Medium / High
    medical_history: str  # from the predefined list
    insurance_plan: str  # Bronze / Silver / Gold
    region: str  # Northeast / Northwest / Southeast / Southwest


class PredictionResponse(BaseModel):
    annual_premium: float
    monthly_premium: float
    risk_score: int
    risk_label: str
    risk_factors: list
    confidence_band: dict


# ─── Business Logic ───────────────────────────────────────────────────────────
RISK_SCORES = {
    "diabetes": 6,
    "heart disease": 8,
    "high blood pressure": 6,
    "thyroid": 5,
    "no disease": 0,
    "none": 0,
}

def calculate_total_risk(medical_history: str) -> int:
    diseases = medical_history.lower().split(" & ")
    return sum(RISK_SCORES.get(d.strip(), 0) for d in diseases)

def get_risk_label(score: int) -> str:
    if score == 0:
        return "Low"
    elif score <= 6:
        return "Moderate"
    elif score <= 11:
        return "High"
    else:
        return "Very High"

def get_risk_factors(req: PredictionRequest) -> list:
    factors = []
    risk_score = calculate_total_risk(req.medical_history)

    if req.smoking_status == "Regular":
        factors.append({"factor": "Regular Smoking", "impact": "high", "icon": "🚬"})
    elif req.smoking_status == "Occasional":
        factors.append({"factor": "Occasional Smoking", "impact": "medium", "icon": "🚬"})

    if req.bmi_category == "Obesity":
        factors.append({"factor": "Obesity (BMI)", "impact": "high", "icon": "⚖️"})
    elif req.bmi_category == "Overweight":
        factors.append({"factor": "Overweight (BMI)", "impact": "medium", "icon": "⚖️"})

    if req.medical_history != "No Disease":
        factors.append({"factor": f"Medical History: {req.medical_history}", "impact": "high" if risk_score >= 8 else "medium", "icon": "🏥"})

    if req.stress_level == "High":
        factors.append({"factor": "High Stress Level", "impact": "medium", "icon": "🧠"})

    if req.physical_activity == "Low":
        factors.append({"factor": "Sedentary Lifestyle", "impact": "medium", "icon": "🏃"})

    if req.age >= 50:
        factors.append({"factor": f"Age {req.age} (Senior)", "impact": "medium", "icon": "👤"})

    if req.number_of_dependants >= 3:
        factors.append({"factor": f"{req.number_of_dependants} Dependants", "impact": "low", "icon": "👨‍👩‍👧‍👦"})

    if not factors:
        factors.append({"factor": "No significant risk factors", "impact": "low", "icon": "✅"})

    return factors


def preprocess_input(req: PredictionRequest) -> pd.DataFrame:
    expected_columns = [
        'age', 'gender', 'marital_status', 'physical_activity',
        'stress_level', 'number_of_dependants', 'smoking_status',
        'income_lakhs', 'insurance_plan', 'total_risk_score',
        'region_Northwest', 'region_Southeast', 'region_Southwest',
        'bmi_category_Obesity', 'bmi_category_Overweight', 'bmi_category_Underweight',
        'employment_status_Salaried', 'employment_status_Self-Employed'
    ]

    df = pd.DataFrame(0, columns=expected_columns, index=[0])

    df['age'] = req.age
    df['number_of_dependants'] = req.number_of_dependants
    df['income_lakhs'] = req.income_lakhs
    df['total_risk_score'] = calculate_total_risk(req.medical_history)
    df['gender'] = 1 if req.gender == 'Male' else 0
    df['marital_status'] = 1 if req.marital_status == 'Married' else 0

    df['physical_activity'] = {"Low": 0, "Medium": 1, "High": 2}[req.physical_activity]
    df['stress_level'] = {"Low": 0, "Medium": 1, "High": 2}[req.stress_level]
    df['smoking_status'] = {"Regular": 0, "Occasional": 1, "No Smoking": 2}[req.smoking_status]
    df['insurance_plan'] = {"Bronze": 0, "Silver": 1, "Gold": 2}[req.insurance_plan]

    if req.region == 'Northwest':
        df['region_Northwest'] = 1
    elif req.region == 'Southeast':
        df['region_Southeast'] = 1
    elif req.region == 'Southwest':
        df['region_Southwest'] = 1

    if req.bmi_category == 'Obesity':
        df['bmi_category_Obesity'] = 1
    elif req.bmi_category == 'Overweight':
        df['bmi_category_Overweight'] = 1
    elif req.bmi_category == 'Underweight':
        df['bmi_category_Underweight'] = 1

    if req.employment_status == 'Salaried':
        df['employment_status_Salaried'] = 1
    elif req.employment_status == 'Self-Employed':
        df['employment_status_Self-Employed'] = 1

    cols_to_scale = scaler_obj['cols_to_scale']
    scaler = scaler_obj['scaler']
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    return df


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_LOADED}


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded. Place model.joblib and scaler.joblib in ./artifacts/")

    try:
        df = preprocess_input(req)
        raw_prediction = float(model.predict(df)[0])
        annual = round(raw_prediction, 2)
        monthly = round(annual / 12, 2)

        risk_score = calculate_total_risk(req.medical_history)
        risk_label = get_risk_label(risk_score)
        risk_factors = get_risk_factors(req)

        # Simple ±8% confidence band
        confidence_band = {
            "low": round(annual * 0.92, 2),
            "high": round(annual * 1.08, 2)
        }

        return PredictionResponse(
            annual_premium=annual,
            monthly_premium=monthly,
            risk_score=risk_score,
            risk_label=risk_label,
            risk_factors=risk_factors,
            confidence_band=confidence_band,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/")
def root():
    return {"message": "Insurance Premium Predictor API. POST to /predict"}
