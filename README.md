# PremiumIQ — Health Insurance Premium Predictor

> An end-to-end ML system that predicts annual health insurance premiums from 13 user inputs, served through a production-grade React + FastAPI application.

---

**[🌐 Live App](https://premiumiq.vercel.app/)
## What This Project Actually Does

Most ML projects stop at a Jupyter notebook. This one doesn't.

PremiumIQ trains a tuned XGBoost regression model on ~10,000 real-world insurance records, achieves **R² = 0.9944** on the test set, then wraps that model in a full-stack web application — multi-step form UI, FastAPI inference server and a results panel with confidence bands and risk factor breakdowns.

The entire stack, from raw data to deployed interface, is contained in this repository.

---

## Model Performance

| Model | Train R² | Test R² |
|---|---|---|
| Linear Regression (baseline) | 0.9554 | 0.9560 |
| Ridge Regression (α=15) | 0.9554 | 0.9559 |
| Random Forest (tuned) | — | 0.9778 (CV) |
| **XGBoost (tuned) ✓ Selected** | — | **0.9944 (CV)** |

**Extreme error rate (>10% off):** 5.42% on held-out test set (161 / 2,972 samples)  
**Dataset size after cleaning:** 9,905 rows | **Features used:** 18

XGBoost was selected because it outperformed all baselines. The near-identical Linear and Ridge scores confirmed no serious overfitting in the linear models — XGBoost's improvement comes from its ability to capture non-linear interactions between features like age, smoking status, and medical history.

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Pipeline | Python, pandas, scikit-learn, XGBoost, joblib |
| Backend | FastAPI + Uvicorn |
| Frontend | React 18, Recharts, CSS Variables |
| Notebooks | Jupyter (full pipeline with inline outputs) |

---

## Architecture

```
User (Browser)
     │
     ▼
React 18 Frontend  (multi-step form → result panel with charts)
     │  POST /predict
     ▼
FastAPI Backend
     │  loads artifacts on startup
     ├── model.joblib     (XGBoost, tuned via RandomizedSearchCV)
     └── scaler.joblib    (StandardScaler + col names for exact replay)


## ML Pipeline — What Was Done and Why

The full pipeline is documented cell-by-cell in `backend/Health_Care_Premium_Explained.ipynb`.

**1. Data Cleaning**
- Dropped ~34 rows with nulls (<0.4% of data) — too few to warrant imputation
- Removed outliers: age > 100 (data entry errors like age=1178), income > 100L (income=960L is unrealistic for the target population)
- Dropped `income_level` (bucketed string version of `income_lakhs`) — redundant and would introduce multicollinearity

**2. Encoding Strategy**
- **Ordinal encoding** for features with natural order: gender, marital_status, physical_activity, stress_level, smoking_status, insurance_plan
- **One-hot encoding** for nominal features: region, bmi_category, employment_status

**3. Feature Engineering**
- `medical_history` contains compound conditions like `"Diabetes & Heart disease"`. These were split on `" & "` and each condition mapped to a risk weight, producing a single `total_risk_score` feature. This mirrors the actuarial logic of risk accumulation.

**4. Multicollinearity Check (VIF)**
- All 18 features had VIF < 2.5 (well below the threshold of 5). No features were dropped on this basis.

**5. Hyperparameter Tuning**
- `RandomizedSearchCV` with 3-fold CV
- XGBoost: 15 iterations, best params: `subsample=0.7, reg_lambda=2, n_estimators=300, max_depth=3, learning_rate=0.2, colsample_bytree=0.9`
- Random Forest: 10 iterations, best CV R² = 0.9778 (XGBoost won at 0.9944)

**6. Artifact Persistence**
- `model.joblib` — serialized XGBoost estimator
- `scaler.joblib` — dict containing `{'scaler': StandardScaler, 'cols_to_scale': [...]}`. Storing column names alongside the scaler ensures the backend applies scaling to exactly the right columns at inference time, preventing silent bugs.

---

## Features the Model Uses (18 Total)

| # | Feature | Type | Encoding |
|---|---|---|---|
| 1 | age | numeric | scaled |
| 2 | gender | binary | ordinal |
| 3 | marital_status | binary | ordinal |
| 4 | physical_activity | ordinal | ordinal |
| 5 | stress_level | ordinal | ordinal |
| 6 | number_of_dependants | numeric | scaled |
| 7 | smoking_status | ordinal | ordinal |
| 8 | income_lakhs | numeric | scaled |
| 9 | insurance_plan | ordinal | ordinal |
| 10 | total_risk_score | engineered | scaled |
| 11–13 | region (NW, SE, SW) | nominal | one-hot |
| 14–16 | bmi_category (Obesity, Overweight, Underweight) | nominal | one-hot |
| 17–18 | employment_status (Salaried, Self-Employed) | nominal | one-hot |

---

## API Reference

### `POST /predict`

**Request body:**
```json
{
  "age": 35,
  "gender": "Male",
  "marital_status": "Married",
  "number_of_dependants": 2,
  "income_lakhs": 15,
  "employment_status": "Salaried",
  "bmi_category": "Normal",
  "smoking_status": "No Smoking",
  "physical_activity": "High",
  "stress_level": "Low",
  "medical_history": "No Disease",
  "insurance_plan": "Silver",
  "region": "Northeast"
}
```

**Response:**
```json
{
  "annual_premium": 12450.75,
  "monthly_premium": 1037.56,
  "risk_score": 0,
  "risk_label": "Low",
  "risk_factors": [
    { "factor": "No significant risk factors", "impact": "low", "icon": "✅" }
  ],
  "confidence_band": { "low": 11454.69, "high": 13446.81 }
}

## Project Structure

```
insurance-premium-predictor/
├── backend/
│   ├── main.py                          # FastAPI app — preprocessing + prediction endpoint
│   ├── requirements.txt
│   ├── runtime.txt
│   ├── Health_Care_Premium_Explained.ipynb  # Full ML pipeline, annotated
│   └── artifacts/
│       ├── model.joblib                 # Trained XGBoost model
│       └── scaler.joblib                # StandardScaler + column names
└── frontend/
    ├── nginx.conf                       # Proxies /predict → backend:8000
    ├── package.json
    ├── public/
    │   └── index.html
    └── src/
        ├── App.jsx                      # Multi-step form shell + state management
        ├── App.css                      # Design system (CSS variables, layout)
        ├── index.js
        └── components/
            ├── FormFields.jsx           # Reusable: slider, toggle, radio, stepper, select
            ├── ProgressBar.jsx          # Step progress indicator
            ├── StepPersonal.jsx         # Step 1: age, gender, marital status, region
            ├── StepHealth.jsx           # Step 2: BMI, smoking, activity, stress, conditions
            ├── StepFinancial.jsx        # Step 3: income, employment, dependants
            ├── StepPolicy.jsx           # Step 4: plan selection (Bronze / Silver / Gold)
            └── ResultPanel.jsx          # Output: premium, confidence band, risk breakdown
```

---

## Quick Start

**Backend:**
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
npm start
# Runs on http://localhost:3000
```

---

## Key Implementation Decisions

**Why ordinal encoding for smoking status and not one-hot?**  
`No Smoking → 0`, `Occasional → 1`, `Regular → 2` encodes a real-world risk gradient. One-hot encoding would treat these as unrelated categories and lose that signal.

**Why XGBoost over Random Forest despite both being tree ensembles?**  
XGBoost's sequential boosting (each tree corrects residuals of the previous) outperformed Random Forest's parallel bagging by a significant margin here (0.9944 vs 0.9778 CV R²). For this dataset size and feature structure, boosting is clearly the better fit.

**Why `max_depth=3` for XGBoost?**  
The tuning process selected shallow trees (depth 3) with more boosting rounds (300 estimators) and a moderate learning rate (0.2). This is a well-known sweet spot — deep trees overfit, shallow trees + more rounds generalize better.

**Medical history risk accumulation:**  
Compound conditions like `"Diabetes & Heart disease"` are split on `" & "`, each disease mapped to a risk weight, and accumulated into `total_risk_score`. This isn't just feature engineering for the model — it directly drives the `risk_factors` breakdown in the UI response.

---

## Limitations & Honest Caveats

- The dataset is synthetic/educational — the model should not be used for actual premium pricing decisions.
- The confidence band in the API response is a ±8% envelope approximation, not a statistically calibrated prediction interval.
- Extreme error rate of 5.42% (predictions off by >10%) suggests the model has edge cases, likely around compound medical conditions and high-income outlier profiles.

---

## Author

**Soham Lodh**  
[GitHub](https://github.com/soham-lodh)
