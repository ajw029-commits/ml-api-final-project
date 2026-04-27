from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parent
REFERENCE_YEAR = 2020
DTI_MEDIAN = 17.73
REVOL_BAL_CAP = 96392.6
REVOL_UTIL_CAP = 150.0

MODEL_FEATURES = [
    "loan_amnt",
    "int_rate",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "term_num",
    "credit_age",
    "fico_avg",
    "home_ownership_MORTGAGE",
    "home_ownership_NONE",
    "home_ownership_OTHER",
    "home_ownership_OWN",
    "home_ownership_RENT",
    "verification_status_Source Verified",
    "verification_status_Verified",
    "purpose_credit_card",
    "purpose_debt_consolidation",
    "purpose_educational",
    "purpose_home_improvement",
    "purpose_house",
    "purpose_major_purchase",
    "purpose_medical",
    "purpose_moving",
    "purpose_other",
    "purpose_renewable_energy",
    "purpose_small_business",
    "purpose_vacation",
    "purpose_wedding",
    "emp_length_10+ years",
    "emp_length_2 years",
    "emp_length_3 years",
    "emp_length_4 years",
    "emp_length_5 years",
    "emp_length_6 years",
    "emp_length_7 years",
    "emp_length_8 years",
    "emp_length_9 years",
    "emp_length_< 1 year",
]


class LoanRequest(BaseModel):
    loan_amnt: float
    int_rate: float
    annual_inc: float
    dti: float
    delinq_2yrs: int
    open_acc: int
    pub_rec: int
    revol_bal: float
    revol_util: float
    term_num: int
    earliest_cr_line: str
    fico_range_low: float
    fico_range_high: float
    home_ownership: str
    verification_status: str
    purpose: str
    emp_length: str


app = FastAPI(title="Loan Outcome Predictor API")

ret_pess_model = joblib.load(BASE_DIR / "ret_PESS_model.pkl")
loan_status_model = joblib.load(BASE_DIR / "loan_status_model.pkl")
classifier_scaler = joblib.load(BASE_DIR / "classifier_scaler.pkl")

if hasattr(ret_pess_model, "n_jobs"):
    ret_pess_model.n_jobs = 1


def build_feature_frame(data: LoanRequest) -> pd.DataFrame:
    earliest_credit_line = pd.to_datetime(data.earliest_cr_line)
    credit_age = REFERENCE_YEAR - earliest_credit_line.year
    fico_avg = (data.fico_range_high + data.fico_range_low) / 2.0

    row = {feature: 0.0 for feature in MODEL_FEATURES}
    row.update(
        {
            "loan_amnt": data.loan_amnt,
            "int_rate": data.int_rate,
            "annual_inc": float(np.log1p(data.annual_inc)),
            "dti": DTI_MEDIAN if data.dti == 999 else data.dti,
            "delinq_2yrs": data.delinq_2yrs,
            "open_acc": data.open_acc,
            "pub_rec": data.pub_rec,
            "revol_bal": min(data.revol_bal, REVOL_BAL_CAP),
            "revol_util": min(data.revol_util, REVOL_UTIL_CAP),
            "term_num": data.term_num,
            "credit_age": float(credit_age),
            "fico_avg": float(fico_avg),
        }
    )

    categorical_values = {
        "home_ownership": data.home_ownership,
        "verification_status": data.verification_status,
        "purpose": data.purpose,
        "emp_length": data.emp_length,
    }

    for prefix, value in categorical_values.items():
        feature_name = f"{prefix}_{value}"
        if feature_name in row:
            row[feature_name] = 1.0

    return pd.DataFrame([row], columns=MODEL_FEATURES)


@app.get("/")
def home():
    return {"message": "Loan Outcome Predictor API is running"}


@app.post("/predict")
def predict(data: LoanRequest):
    if data.fico_range_high < data.fico_range_low:
        return {"error": "fico_range_high must be greater than or equal to fico_range_low"}

    features = build_feature_frame(data)
    scaled_features = classifier_scaler.transform(features)

    class_labels = list(loan_status_model.classes_)
    probability_map = dict(
        zip(class_labels, loan_status_model.predict_proba(scaled_features)[0])
    )

    default_probability = float(probability_map.get("Charged Off", 0.0))
    pessimistic_return = float(ret_pess_model.predict(features)[0])

    return {
        "default_probability": round(default_probability, 4),
        "fully_paid_probability": round(1.0 - default_probability, 4),
        "pessimistic_return_rate": round(pessimistic_return, 4),
    }
