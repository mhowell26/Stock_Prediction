import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt

# Load model + defaults
model = joblib.load("model.joblib")
X_train = pd.read_csv("X_train.csv")

with open("feature_defaults.json", "r") as f:
    defaults = json.load(f)

st.set_page_config(page_title="Loan Default Predictor", layout="wide")

st.title("🏦 Loan Default Risk Predictor")

st.write("""
This app predicts whether a borrower is likely to default on a loan.
Only a few key inputs are required — the rest are filled automatically.
""")

# 🔑 ONLY USE IMPORTANT FEATURES (professor requirement)
loan_amnt = st.number_input("Loan Amount", 1000.0, 35000.0, 12000.0)
term_months = st.selectbox("Loan Term", [36, 60])
int_rate = st.number_input("Interest Rate (%)", 1.0, 40.0, 13.5)
annual_inc = st.number_input("Annual Income", 10000.0, 300000.0, 75000.0)
dti = st.number_input("Debt-to-Income Ratio", 0.0, 60.0, 18.0)
fico_score = st.number_input("FICO Score", 300.0, 850.0, 700.0)
revol_util = st.number_input("Revolving Utilization (%)", 0.0, 150.0, 45.0)

home_ownership = st.selectbox(
    "Home Ownership",
    sorted(X_train["home_ownership"].dropna().astype(str).unique())
)

purpose = st.selectbox(
    "Loan Purpose",
    sorted(X_train["purpose"].dropna().astype(str).unique())
)

# Prediction button
if st.button("Predict Default Risk"):

    # Fill rest of columns using defaults
    applicant = defaults.copy()

    applicant.update({
        "loan_amnt": loan_amnt,
        "term_months": term_months,
        "int_rate": int_rate,
        "annual_inc": annual_inc,
        "dti": dti,
        "fico_score": fico_score,
        "revol_util": revol_util,
        "home_ownership": home_ownership,
        "purpose": purpose
    })

    input_df = pd.DataFrame([applicant])

    # Predict
    probability = model.predict_proba(input_df)[0, 1]
    prediction = model.predict(input_df)[0]

    # Risk label
    if probability >= 0.65:
        risk = "🔴 High Risk"
    elif probability >= 0.35:
        risk = "🟡 Medium Risk"
    else:
        risk = "🟢 Low Risk"

    st.subheader("Prediction Result")

    st.metric("Default Probability", f"{probability:.2%}")
    st.write(f"Prediction: **{prediction}**")
    st.write(f"Risk Level: {risk}")

    # Optional SHAP
    try:
        explainer = joblib.load("shap_explainer.joblib")
        feature_names = joblib.load("model_transformed_feature_names.joblib")

        transformed = model.named_steps["preprocess"].transform(
            model.named_steps["clean_feature_engineer"].transform(input_df)
        )

        shap_values = explainer.shap_values(transformed)

        if isinstance(shap_values, list):
            vals = shap_values[1][0]
        else:
            vals = shap_values[0]

        shap_df = pd.DataFrame({
            "feature": feature_names,
            "value": vals
        })

        shap_df["abs"] = shap_df["value"].abs()
        shap_df = shap_df.sort_values("abs", ascending=False).head(10)

        st.subheader("🔍 Model Explanation")

        fig, ax = plt.subplots()
        plot_df = shap_df.sort_values("abs")
        ax.barh(plot_df["feature"], plot_df["value"])
        ax.set_title("Top Feature Contributions")
        st.pyplot(fig)

    except:
        st.warning("SHAP explanation not available (missing files)")
