from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Load trained pipeline
model = joblib.load("productivity_model_20251127_234333.pkl")
expected_features = list(model.feature_names_in_)

# Load classification model
try:
    classify_info = joblib.load("productivity_classifier.pkl")
    classify_model = classify_info["model"]
    classify_features = classify_info["feature_names"]
    classify_encoders = classify_info.get("label_encoders", {})
except:
    classify_model = None
    classify_features = []
    classify_encoders = {}

# Define input schema
class ProductivityInput(BaseModel):
    date: str
    quarter: str
    department: str
    day: str
    team: float
    targeted_productivity: float
    smv: float
    wip: float
    over_time: float
    incentive: float
    idle_time: float
    idle_men: float
    no_of_style_change: float
    no_of_workers: float

@app.get("/")
def root():
    return {"message": "API is running!"}


@app.post("/classify")
def classify(data: ProductivityInput):
    if classify_model is None:
        return {"error": "Classification model not loaded"}
    
    df = pd.DataFrame([data.dict()])
    
    # Normalize categorical columns
    for col in ["quarter", "department", "day"]:
        df[col] = df[col].str.lower()
    
    for col in classify_features:
        if col not in df.columns:
            df[col] = 0
    
    df_class = df[classify_features]
    for col, le in classify_encoders.items():
        if col in df_class:
            df_class[col] = le.transform(df_class[col].astype(str))
    
    pred_class = classify_model.predict(df_class)[0]
    return {"prediction_class": int(pred_class)}

@app.post("/predict")
def predict(data: ProductivityInput):
    # Convert JSON input to DataFrame
    df = pd.DataFrame([data.dict()])

    # Add missing columns (if any)
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0  

    # Reorder columns to match training
    df = df[expected_features]

    # Run prediction
    prediction = model.predict(df)[0]

    return {"prediction": float(prediction)}

