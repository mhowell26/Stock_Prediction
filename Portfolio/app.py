
import os, json
import numpy as np
import pandas as pd
import joblib
import boto3
import streamlit as st

st.set_page_config(page_title="JPM 5D Return Predictor", layout="wide")
st.title("JPMorgan (JPM) – 5-Trading-Day Forward Return Predictor")

mode = st.radio("Prediction mode", ["Local model (joblib)", "SageMaker endpoint"], horizontal=True)

FEATURES = ['range_pct', 'body_pct', 'vol_20', 'rsi_14', 'macd', 'macd_signal', 'macd_hist', 'atr_14_pct', 'is_month_end', 'is_quarter_end', 'SPY_logret_1d', 'XLF_logret_1d', 'BAC_logret_1d', '^TNX_logret_1d', 'UUP_logret_1d', 'EURUSD=X_logret_1d', '^VIX_logret_1d']

st.sidebar.header("Input features")
row = {}
for f in FEATURES:
    default = 0.0
    if f in ["is_month_end","is_quarter_end"]:
        default = 0
    row[f] = st.sidebar.number_input(f, value=float(default))

X_row = pd.DataFrame([row])[FEATURES]
st.subheader("Model input")
st.dataframe(X_row)

def predict_local(X):
    model = joblib.load("finalized_model.joblib")
    return float(model.predict(X)[0])

def predict_endpoint(X, endpoint_name):
    rt = boto3.client("sagemaker-runtime")
    payload = {"instances": X.values.tolist()}
    resp = rt.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload)
    )
    out = json.loads(resp["Body"].read().decode("utf-8"))
    return float(out["predictions"][0])

endpoint_name = "hw2-jpm-endpoint"
if mode == "SageMaker endpoint":
    endpoint_name = st.text_input("Endpoint name", value=endpoint_name)

if st.button("Predict"):
    if mode == "Local model (joblib)":
        if not os.path.exists("finalized_model.joblib"):
            st.error("finalized_model.joblib not found in app directory.")
        else:
            pred = predict_local(X_row)
            st.success(f"Predicted 5D forward log return: {pred:.6f}")
    else:
        pred = predict_endpoint(X_row, endpoint_name)
        st.success(f"Endpoint predicted 5D forward log return: {pred:.6f}")

st.caption("Tip: log return of 0.01 is roughly +1% over ~5 trading days.")
