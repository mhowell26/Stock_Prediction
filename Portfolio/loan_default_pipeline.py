
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class LoanDataCleanerFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, outlier_cols=None, lower_q=0.01, upper_q=0.99):
        self.outlier_cols = outlier_cols
        self.lower_q = lower_q
        self.upper_q = upper_q
        self.caps_ = {}

    def fit(self, X, y=None):
        X = self._basic_clean(X.copy())
        if self.outlier_cols is None:
            self.outlier_cols_ = [
                c for c in X.select_dtypes(include=[np.number]).columns
                if c != "loan_default"
            ]
        else:
            self.outlier_cols_ = list(self.outlier_cols)

        self.caps_ = {}
        for col in self.outlier_cols_:
            if col in X.columns:
                numeric = pd.to_numeric(X[col], errors="coerce")
                self.caps_[col] = (
                    float(numeric.quantile(self.lower_q)),
                    float(numeric.quantile(self.upper_q))
                )
        return self

    def transform(self, X):
        X = self._basic_clean(X.copy())

        for col, (lower, upper) in self.caps_.items():
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors="coerce").clip(lower, upper)

        X["loan_to_income"] = X["loan_amnt"] / (X["annual_inc"] + 1)
        X["installment_to_income"] = (X["installment"] * 12) / (X["annual_inc"] + 1)
        X["log_annual_inc"] = np.log1p(X["annual_inc"].clip(lower=0))
        X["log_loan_amnt"] = np.log1p(X["loan_amnt"].clip(lower=0))
        X["credit_burden"] = (X["dti"] * X["revol_util"]) / 100
        X["rate_x_term"] = X["int_rate"] * X["term_months"]
        X["payment_pressure"] = X["installment"] / ((X["annual_inc"] / 12) + 1)
        X["income_after_debt"] = X["annual_inc"] * (1 - (X["dti"] / 100))
        X["high_dti_flag"] = (X["dti"] > 25).astype(int)
        X["high_util_flag"] = (X["revol_util"] > 70).astype(int)
        X["long_term_flag"] = (X["term_months"] == 60).astype(int)
        X["rate_dti_interaction"] = X["int_rate"] * X["dti"]

        X["fico_bucket"] = pd.cut(
            X["fico_score"],
            bins=[0, 640, 700, 760, 900],
            labels=["poor", "fair", "good", "excellent"]
        ).astype(str)

        X["util_bucket"] = pd.cut(
            X["revol_util"],
            bins=[-1, 30, 60, 90, 200],
            labels=["low", "moderate", "high", "very_high"]
        ).astype(str)

        return X

    def _basic_clean(self, X):
        X.columns = [str(c).strip().lower().replace(" ", "_") for c in X.columns]
        X = X.drop_duplicates()

        for col in ["home_ownership", "purpose"]:
            if col in X.columns:
                X[col] = (
                    X[col].astype(str)
                    .str.strip()
                    .str.lower()
                    .replace({"nan": np.nan, "none": np.nan, "": np.nan})
                )

        numeric_cols = [
            "loan_amnt", "term_months", "int_rate", "installment",
            "annual_inc", "dti", "fico_score", "open_acc", "revol_util",
            "emp_length_years"
        ]

        for col in numeric_cols:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors="coerce")

        non_negative_cols = [
            "loan_amnt", "term_months", "int_rate", "installment",
            "annual_inc", "dti", "fico_score", "open_acc", "revol_util",
            "emp_length_years"
        ]

        for col in non_negative_cols:
            if col in X.columns:
                X.loc[X[col] < 0, col] = np.nan

        X = X.replace([np.inf, -np.inf], np.nan)

        return X
