"""
FastAPI Backend for Bank Churn Prediction Model
WITH FEATURE ENGINEERING TO MATCH TRAINING DATA
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pickle
import numpy as np
import json
import os
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Bank Churn Prediction API",
    description="ML model to predict customer churn probability",
    version="1.0"
)

# ===== LOAD MODEL =====
model = None
try:
    paths_to_try = [
        'models/final_churn_model.pkl',
        '../models/final_churn_model.pkl',
    ]

    for path in paths_to_try:
        if os.path.exists(path):
            with open(path, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"✅ Model loaded from: {path}")
            break

    if model is None:
        logger.warning("❌ Model file not found")

except Exception as e:
    logger.error(f"❌ Error loading model: {str(e)}")

# ===== LOAD METRICS =====
metrics = {
    'accuracy': 0.75,
    'recall_churn': 0.78,
    'precision_churn': 0.45,
    'f1_churn': 0.57,
    'roc_auc': 0.85
}

try:
    paths_to_try = [
        'models/model_metrics.json',
        '../models/model_metrics.json',
    ]

    for path in paths_to_try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
            logger.info(f"✅ Metrics loaded from: {path}")
            break
except Exception as e:
    logger.warning(f"⚠️ Could not load metrics: {str(e)}")

# ===== MODELS =====
class CustomerData(BaseModel):
    age: int
    tenure: int
    credit_score: int
    balance: float
    estimated_salary: float
    num_of_products: int
    is_active_member: int
    has_cr_card: int
    geography: str
    gender: str

class PredictionResponse(BaseModel):
    churn_probability: float
    risk_segment: str
    recommendation: str
    model_confidence: float

# ===== FEATURE ENGINEERING FUNCTIONS =====

def encode_geography(geography: str) -> int:
    """Encode geography"""
    geography_map = {'France': 0, 'Germany': 1, 'Spain': 2}
    return geography_map.get(geography, 0)

def encode_gender(gender: str) -> int:
    """Encode gender"""
    gender_map = {'Female': 0, 'Male': 1}
    return gender_map.get(gender, 0)

def create_features(customer: CustomerData, row_number: int = 1) -> np.ndarray:
    """
    Create all 24 engineered features matching the training data

    Feature order from training data:
    1. RowNumber
    2. CreditScore
    3. Age
    4. Balance
    5. EstimatedSalary
    6. balance_salary_rate
    7. product_utilization_rate_by_year
    8. product_utilization_rate_by_estimated_salary
    9. tenure_rate_by_age
    10. credit_score_rate_by_age
    11. product_utilization_rate_by_salary
    12. credit_score_rate_by_salary
    13. mean_financials
    14. credit_score_squared
    15. age_tenure
    16. countries_monthly_average_salaries
    17. HasCrCard
    18. IsActiveMember
    19. NumOfProducts
    20. Tenure
    21. gender_category
    22. country_category
    23. credit_score_category
    24. age_category
    """

    # Encode categorical
    gender_code = encode_gender(customer.gender)
    country_code = encode_geography(customer.geography)

    # Create credit score category
    if customer.credit_score < 580:
        credit_score_category = 0
    elif customer.credit_score < 670:
        credit_score_category = 1
    elif customer.credit_score < 740:
        credit_score_category = 2
    elif customer.credit_score < 800:
        credit_score_category = 3
    else:
        credit_score_category = 4

    # Create age category
    if customer.age < 30:
        age_category = 0
    elif customer.age < 40:
        age_category = 1
    elif customer.age < 50:
        age_category = 2
    elif customer.age < 60:
        age_category = 3
    else:
        age_category = 4

    # Calculate engineered features
    balance_salary_rate = customer.balance / customer.estimated_salary if customer.estimated_salary > 0 else 0
    product_utilization_rate_by_year = customer.num_of_products / (customer.tenure + 1)  # Avoid division by zero
    product_utilization_rate_by_estimated_salary = customer.num_of_products / (customer.estimated_salary / 50000) if customer.estimated_salary > 0 else 0
    tenure_rate_by_age = customer.tenure / (customer.age - 18) if customer.age > 18 else 0
    credit_score_rate_by_age = customer.credit_score / customer.age if customer.age > 0 else 0
    product_utilization_rate_by_salary = customer.num_of_products / (customer.estimated_salary / 100000) if customer.estimated_salary > 0 else 0
    credit_score_rate_by_salary = customer.credit_score / (customer.estimated_salary / 50000) if customer.estimated_salary > 0 else 0
    mean_financials = (customer.balance + customer.estimated_salary) / 2
    credit_score_squared = customer.credit_score ** 2
    age_tenure = customer.age * customer.tenure

    # Countries monthly average salary (proxy based on geography)
    # From typical averages: France ~2500, Germany ~3500, Spain ~1800
    countries_monthly_salaries = {'France': 2500, 'Germany': 3500, 'Spain': 1800}
    countries_monthly_average_salaries = countries_monthly_salaries.get(customer.geography, 2500)

    # Build feature array in exact order
    features = np.array([[
        row_number,                                      # 1
        customer.credit_score,                           # 2
        customer.age,                                    # 3
        customer.balance,                                # 4
        customer.estimated_salary,                       # 5
        balance_salary_rate,                             # 6
        product_utilization_rate_by_year,               # 7
        product_utilization_rate_by_estimated_salary,   # 8
        tenure_rate_by_age,                              # 9
        credit_score_rate_by_age,                        # 10
        product_utilization_rate_by_salary,             # 11
        credit_score_rate_by_salary,                     # 12
        mean_financials,                                 # 13
        credit_score_squared,                            # 14
        age_tenure,                                      # 15
        countries_monthly_average_salaries,             # 16
        customer.has_cr_card,                            # 17
        customer.is_active_member,                       # 18
        customer.num_of_products,                        # 19
        customer.tenure,                                 # 20
        gender_code,                                     # 21
        country_code,                                    # 22
        credit_score_category,                           # 23
        age_category                                     # 24
    ]], dtype=np.float32)

    logger.info(f"✅ Created features shape: {features.shape}")
    logger.info(f"📈 Feature values: {features[0]}")

    return features

def assign_risk_segment(probability: float) -> str:
    if probability >= 0.70:
        return "High Risk"
    elif probability >= 0.40:
        return "Medium Risk"
    else:
        return "Low Risk"

def get_recommendation(risk_segment: str, geography: str, age: int, num_products: int) -> str:
    if risk_segment == "High Risk":
        rec = "🔴 IMMEDIATE ACTION REQUIRED\n"
        rec += "• Contact customer immediately via phone\n"
        rec += "• Offer retention incentive ($500-1,000)\n"
        rec += "• Priority support upgrade\n"

        if geography == "Germany":
            rec += "• Note: Germany region has elevated churn - consider region-specific offer\n"
        if age > 45:
            rec += "• Note: Older customers more likely to churn - emphasize service quality\n"
        if num_products <= 2:
            rec += "• Recommend cross-selling additional products\n"

    elif risk_segment == "Medium Risk":
        rec = "🟡 PROACTIVE ENGAGEMENT\n"
        rec += "• Send personalized email with targeted offers\n"
        rec += "• Recommend product upgrades\n"
        rec += "• Enroll in loyalty program\n"
        rec += "• Schedule optional account review\n"

    else:
        rec = "🟢 MAINTAIN RELATIONSHIP\n"
        rec += "• Standard retention activities\n"
        rec += "• Regular account check-ins\n"
        rec += "• Loyalty rewards program\n"
        rec += "• Cross-sell opportunities\n"

    return rec

# ===== ROUTES =====

@app.get("/", response_class=HTMLResponse)
async def get_home():
    try:
        paths_to_try = ['templates/index.html', './templates/index.html']
        for path in paths_to_try:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    return f.read()
        return "<h1>Error: index.html not found</h1>"
    except Exception as e:
        logger.error(f"Error loading HTML: {str(e)}")
        return f"<h1>Error: {str(e)}</h1>"

@app.get("/api/metrics")
async def get_model_metrics():
    return {
        "model_name": metrics.get('model_name', "XGBoost with SMOTE + Class Weight"),
        "accuracy": metrics.get('accuracy', 0.75),
        "recall": metrics.get('recall', 0.78),
        "precision": metrics.get('precision', 0.45),
        "f1_score": metrics.get('f1_score', 0.57),
        "roc_auc": metrics.get('roc_auc', 0.85),
        "description": "Model optimized for high recall to catch 78% of actual churners"
    }

@app.post("/api/predict", response_model=PredictionResponse)
async def predict_churn(customer: CustomerData):
    logger.info(f"📊 Received prediction request: {customer}")

    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Validate inputs
        if not 18 <= customer.age <= 100:
            raise HTTPException(status_code=400, detail="Age must be 18-100")
        if customer.num_of_products < 1 or customer.num_of_products > 4:
            raise HTTPException(status_code=400, detail="Products must be 1-4")
        if customer.geography not in ['France', 'Germany', 'Spain']:
            raise HTTPException(status_code=400, detail="Geography invalid")
        if customer.gender not in ['Male', 'Female']:
            raise HTTPException(status_code=400, detail="Gender invalid")

        logger.info("✅ Input validation passed")

        # Create engineered features
        features = create_features(customer)

        # Make prediction
        logger.info("🔮 Making prediction...")
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        logger.info(f"✅ Prediction: {prediction}, Probability: {probability}")

        # Assign risk segment
        risk_segment = assign_risk_segment(probability)

        # Get recommendation
        recommendation = get_recommendation(
            risk_segment, customer.geography, customer.age, customer.num_of_products
        )

        # Calculate confidence
        confidence = float(max(model.predict_proba(features)[0]))

        logger.info(f"✅ Result: {risk_segment}, Confidence: {confidence}")

        return PredictionResponse(
            churn_probability=float(probability),
            risk_segment=risk_segment,
            recommendation=recommendation,
            model_confidence=confidence
        )

    except HTTPException as e:
        logger.error(f"❌ HTTP Exception: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"❌ Prediction error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/api/sample-data")
async def get_sample_data():
    return {
        "high_risk_example": {
            "age": 48, "tenure": 2, "credit_score": 600, "balance": 50000,
            "estimated_salary": 75000, "num_of_products": 2, "is_active_member": 0,
            "has_cr_card": 1, "geography": "Germany", "gender": "Female"
        },
        "medium_risk_example": {
            "age": 35, "tenure": 5, "credit_score": 750, "balance": 100000,
            "estimated_salary": 90000, "num_of_products": 2, "is_active_member": 1,
            "has_cr_card": 1, "geography": "France", "gender": "Male"
        },
        "low_risk_example": {
            "age": 28, "tenure": 8, "credit_score": 850, "balance": 250000,
            "estimated_salary": 120000, "num_of_products": 3, "is_active_member": 1,
            "has_cr_card": 1, "geography": "France", "gender": "Female"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None, "api_version": "1.0"}

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*70)
    print("🚀 Starting FastAPI Server with Feature Engineering")
    print("="*70)
    print(f"\n✅ Model Loaded: {model is not None}")
    print("\n📱 API: http://localhost:8000")
    print("📊 Docs: http://localhost:8000/docs")
    print("\n" + "="*70 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")