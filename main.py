from fastapi import FastAPI
import joblib
import numpy as np

# --- Create the API application ---
app = FastAPI(title="Loan Outcome Predictor API")

# --- Load the saved model (from the .pkl file) ---
ret_PESS_model = joblib.load("ret_PESS_model.pkl")
loan_status_model = joblib.load("loan_status_model.pkl")


# --- Endpoint 1: Home page (confirms the API is alive) ---
@app.get("/")
def home():
    return {"message": "Loan Outcome Predictor API is running"}

# --- Endpoint 2: Prediction (this is the one that does the work) ---
@app.post("/predict")
def predict(data: dict):
    # Pull the customer's features out of the incoming request
    features = np.array([[
        data["monthly_spend"],
        data["tenure_months"],
        data["support_calls"]
    ]])

    # Use the model to get a churn probability
    churn_prob = model.predict_proba(features)[0][1]

    return {
        "churn_probability": round(float(churn_prob), 4)
    }