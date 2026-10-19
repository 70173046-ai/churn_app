"""
Customer Churn Prediction - Flask backend
Assignment 2 (Artificial Intelligence)

Input-to-prediction flow (be ready to explain this in your viva):
  1. Browser sends the form  ->  POST /
  2. validate_form()   checks every field and converts text to numbers
  3. build_features()  repeats the SAME preprocessing used in Assignment 1:
                       - AvgCostPerMonth = TotalCharges / (tenure + 1)
                       - binary columns mapped to 0/1
                       - one-hot encoding (drop_first=True) in the SAME column order
  4. scaler.transform() applies the StandardScaler fitted on the training data
  5. model.predict()    gives Churn / No Churn (Logistic Regression)
  6. The result is shown on the page in plain language
"""

import os

import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load the saved artifacts (created by the last cell of the Assignment 1 notebook)
# Absolute paths are used so the app also works on PythonAnywhere.
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))        # Logistic Regression
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))      # StandardScaler
FEATURE_COLUMNS = joblib.load(os.path.join(BASE_DIR, "columns.pkl"))  # 31 training columns, in order

# ---------------------------------------------------------------------------
# Allowed values for every dropdown (the same categories as the dataset)
# ---------------------------------------------------------------------------
YES_NO = ["Yes", "No"]
ADDON_CHOICES = ["No", "Yes", "No internet service"]

CHOICES = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": ["0", "1"],
    "Partner": YES_NO,
    "Dependents": YES_NO,
    "PhoneService": YES_NO,
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ADDON_CHOICES,
    "OnlineBackup": ADDON_CHOICES,
    "DeviceProtection": ADDON_CHOICES,
    "TechSupport": ADDON_CHOICES,
    "StreamingTV": ADDON_CHOICES,
    "StreamingMovies": ADDON_CHOICES,
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": YES_NO,
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}

# Friendly names used in error messages
LABELS = {
    "gender": "Gender",
    "SeniorCitizen": "Senior citizen",
    "Partner": "Partner",
    "Dependents": "Dependents",
    "PhoneService": "Phone service",
    "MultipleLines": "Multiple lines",
    "InternetService": "Internet service",
    "OnlineSecurity": "Online security",
    "OnlineBackup": "Online backup",
    "DeviceProtection": "Device protection",
    "TechSupport": "Tech support",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming movies",
    "Contract": "Contract",
    "PaperlessBilling": "Paperless billing",
    "PaymentMethod": "Payment method",
    "tenure": "Tenure",
    "MonthlyCharges": "Monthly charges",
    "TotalCharges": "Total charges",
}

INTERNET_ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                   "TechSupport", "StreamingTV", "StreamingMovies"]

# Same lists as in the notebook (Assignment 1, Task 3)
BINARY_COLS = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
MULTI_COLS = ["MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
              "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
              "Contract", "PaymentMethod"]
BINARY_MAP = {"No": 0, "Yes": 1, "Female": 0, "Male": 1}


# ---------------------------------------------------------------------------
# Step 1: validation
# ---------------------------------------------------------------------------
def parse_number(raw, name, minimum, maximum, as_int=False):
    """Convert a text field to a number and check its range. Returns (value, error)."""
    raw = (raw or "").strip()
    if raw == "":
        return None, f"{LABELS[name]} is required."
    try:
        value = float(raw)
    except ValueError:
        return None, f"{LABELS[name]} must be a number."
    if value != value or value in (float("inf"), float("-inf")):   # NaN / infinity
        return None, f"{LABELS[name]} must be a valid number."
    if as_int:
        if value != int(value):
            return None, f"{LABELS[name]} must be a whole number of months."
        value = int(value)
    if not (minimum <= value <= maximum):
        return None, f"{LABELS[name]} must be between {minimum} and {maximum}."
    return value, None


def validate_form(form):
    """Check all inputs. Returns (clean_data, errors). errors is {field: message}."""
    data, errors = {}, {}

    # Numeric fields
    for name, lo, hi, as_int in [("tenure", 0, 72, True),
                                 ("MonthlyCharges", 0.01, 500, False),
                                 ("TotalCharges", 0, 50000, False)]:
        value, err = parse_number(form.get(name), name, lo, hi, as_int)
        if err:
            errors[name] = err
        else:
            data[name] = value

    # Dropdown fields (the browser can be bypassed, so values are checked again here)
    for name, allowed in CHOICES.items():
        value = (form.get(name) or "").strip()
        if value == "":
            # These two depend on other answers and are filled in below if possible
            if name == "MultipleLines" or name in INTERNET_ADDONS:
                continue
            errors[name] = f"Please choose a value for {LABELS[name]}."
        elif value not in allowed:
            errors[name] = f"'{value}' is not a valid option for {LABELS[name]}."
        else:
            data[name] = value

    # Dependent answers: no phone -> no multiple lines, no internet -> no add-ons
    if data.get("PhoneService") == "No":
        data["MultipleLines"] = "No phone service"
        errors.pop("MultipleLines", None)
    elif data.get("PhoneService") == "Yes":
        if data.get("MultipleLines") == "No phone service":
            errors["MultipleLines"] = "Choose Yes or No for multiple lines (the customer has phone service)."
        elif "MultipleLines" not in data and "MultipleLines" not in errors:
            errors["MultipleLines"] = "Please choose a value for Multiple lines."

    if data.get("InternetService") == "No":
        for col in INTERNET_ADDONS:
            data[col] = "No internet service"
            errors.pop(col, None)
    elif data.get("InternetService") in ("DSL", "Fiber optic"):
        for col in INTERNET_ADDONS:
            if data.get(col) == "No internet service":
                errors[col] = f"Choose Yes or No for {LABELS[col].lower()} (the customer has internet)."
            elif col not in data and col not in errors:
                errors[col] = f"Please choose a value for {LABELS[col]}."

    return data, errors


# ---------------------------------------------------------------------------
# Step 2: preprocessing identical to Assignment 1
# ---------------------------------------------------------------------------
def build_features(data):
    """Turn one validated customer into a 1-row DataFrame in the exact training layout."""
    row = {col: 0 for col in FEATURE_COLUMNS}

    # Numeric columns
    row["SeniorCitizen"] = int(data["SeniorCitizen"])
    row["tenure"] = data["tenure"]
    row["MonthlyCharges"] = data["MonthlyCharges"]
    row["TotalCharges"] = data["TotalCharges"]
    # Engineered feature from Assignment 1
    row["AvgCostPerMonth"] = data["TotalCharges"] / (data["tenure"] + 1)

    # Binary columns -> 0 / 1
    for col in BINARY_COLS:
        row[col] = BINARY_MAP[data[col]]

    # One-hot columns. The training data used get_dummies(drop_first=True), so the
    # first category of each feature has NO column (all zeros). We build the dummy
    # columns by hand because pd.get_dummies on a single row would drop the only
    # category and give wrong results.
    for col in MULTI_COLS:
        dummy_name = f"{col}_{data[col]}"
        if dummy_name in row:
            row[dummy_name] = 1

    return pd.DataFrame([row], columns=FEATURE_COLUMNS)   # same column order as training


# ---------------------------------------------------------------------------
# Step 3: prediction
# ---------------------------------------------------------------------------
def predict(data):
    features = build_features(data)
    scaled = scaler.transform(features)
    predicted_class = int(model.predict(scaled)[0])
    churn_probability = float(model.predict_proba(scaled)[0][1]) * 100

    if predicted_class == 1:
        label = "Churn"
        headline = "This customer is likely to leave."
        advice = ("Consider contacting them with a retention offer, such as a longer "
                  "contract discount or a service upgrade.")
    else:
        label = "No Churn"
        headline = "This customer is likely to stay."
        advice = "No immediate action is needed. Keep the service quality consistent."

    if churn_probability >= 50:
        risk = "High"
    elif churn_probability >= 30:
        risk = "Medium"
    else:
        risk = "Low"

    return {
        "label": label,
        "headline": headline,
        "advice": advice,
        "probability": round(churn_probability, 1),
        "risk": risk,
        "is_churn": predicted_class == 1,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result, errors, values = None, {}, {}

    if request.method == "POST":
        values = request.form.to_dict()           # keep what the user typed
        data, errors = validate_form(request.form)
        if not errors:
            try:
                result = predict(data)
            except Exception:                      # never show a crash page to the user
                app.logger.exception("Prediction failed")
                errors["_form"] = "Something went wrong while making the prediction. Please check the values and try again."

    return render_template("index.html", choices=CHOICES, values=values,
                           errors=errors, result=result)


if __name__ == "__main__":
    app.run(debug=True)