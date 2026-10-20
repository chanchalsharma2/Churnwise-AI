from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# File paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_model.joblib")
DATA_PATH = os.path.join(BASE_DIR, "data", "customer_churn.csv")

# Load trained model and dataset
model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

# Clean dataset values
if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )


def get_risk_factors(form):
    """Return possible churn warning factors from selected form values."""

    factors = []

    contract = form.get("Contract", "")
    monthly_charges = float(form.get("MonthlyCharges", 0) or 0)
    online_security = form.get("OnlineSecurity", "")
    tech_support = form.get("TechSupport", "")

    if contract == "Month-to-month":
        factors.append({
            "title": "Month-to-Month Contract",
            "description": (
                "Customers without long-term contracts may be more "
                "likely to discontinue the service."
            )
        })

    if monthly_charges >= 70:
        factors.append({
            "title": "High Monthly Charges",
            "description": (
                "Higher monthly bills may contribute to customer "
                "dissatisfaction and increased churn risk."
            )
        })

    if online_security == "No":
        factors.append({
            "title": "Online Security Unavailable",
            "description": (
                "The absence of online security services may affect "
                "the customer's service experience."
            )
        })

    if tech_support == "No":
        factors.append({
            "title": "Technical Support Unavailable",
            "description": (
                "Limited access to technical support may contribute "
                "to customer dissatisfaction."
            )
        })

    if not factors:
        factors.append({
            "title": "No Listed Warning Factors",
            "description": (
                "No listed warning factors were identified from the "
                "selected details. Churn risk may still exist."
            )
        })

    return factors


def get_advice(risk):
    """Provide retention advice based on estimated churn risk."""

    if risk >= 60:
        return (
            "High churn risk: Contact the customer, review their "
            "service experience, and offer relevant support or "
            "retention options."
        )

    elif risk >= 35:
        return (
            "Moderate churn risk: Review customer satisfaction and "
            "identify opportunities to improve the service experience."
        )

    else:
        return (
            "Low estimated churn risk: Maintain service quality and "
            "continue monitoring customer feedback."
        )


def get_dashboard_stats():
    """Calculate basic dashboard statistics."""

    total_customers = len(df)

    if "Churn" in df.columns and total_customers > 0:
        churned_customers = (
            df["Churn"].astype(str).str.strip().str.lower() == "yes"
        ).sum()

        churn_rate = round(
            (churned_customers / total_customers) * 100, 1
        )
    else:
        churned_customers = 0
        churn_rate = 0

    if "MonthlyCharges" in df.columns and total_customers > 0:
        avg_monthly_charges = round(
            pd.to_numeric(
                df["MonthlyCharges"], errors="coerce"
            ).mean(),
            2
        )
    else:
        avg_monthly_charges = 0

    return {
        "total_customers": total_customers,
        "churned_customers": int(churned_customers),
        "churn_rate": churn_rate,
        "avg_monthly_charges": avg_monthly_charges
    }


def make_prediction(form):
    """Prepare customer inputs and predict churn probability safely."""

    # Use a dataset row as a base for features not collected by the form.
    customer = df.drop(
        columns=["Churn", "customerID"], errors="ignore"
    ).iloc[[0]].copy()

    numeric_fields = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    for field in numeric_fields:
        if field in form and field in customer.columns:
            customer[field] = float(form[field] or 0)

    categorical_fields = [
        "Contract",
        "InternetService",
        "PaymentMethod",
        "OnlineSecurity",
        "TechSupport"
    ]

    for field in categorical_fields:
        if field in form and field in customer.columns:
            customer[field] = form[field]

    # Get the probability for the churn class safely without string-matching errors
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(customer)[0]
        
        # For standard binary classifiers, index 1 corresponds to positive/churn class
        if len(probabilities) >= 2:
            risk = round(float(probabilities[1]) * 100, 2)
        else:
            risk = round(float(probabilities[0]) * 100, 2)
    else:
        prediction_label = model.predict(customer)[0]
        risk = 100.0 if prediction_label in [1, "1", "yes", "True"] else 0.0

    if risk >= 60:
        risk_level = "High"
    elif risk >= 35:
        risk_level = "Moderate"
    else:
        risk_level = "Low"

    return risk, risk_level


@app.route("/", methods=["GET", "POST"])
def index():
    stats = get_dashboard_stats()

    result = None
    risk = None
    risk_level = None
    advice = None
    risk_factors = []

    if request.method == "POST":
        try:
            risk, risk_level = make_prediction(request.form)
            advice = get_advice(risk)
            risk_factors = get_risk_factors(request.form)

            result = {
                "risk": risk,
                "risk_level": risk_level,
                "advice": advice,
                "risk_factors": risk_factors
            }

        except Exception as error:
            result = {
                "error": (
                    "Unable to generate a prediction. Please check "
                    "the entered values and try again."
                )
            }
            app.logger.exception("Prediction failed: %s", error)

    return render_template(
        "index.html",
        stats=stats,
        total_customers=stats["total_customers"],
        churned_customers=stats["churned_customers"],
        churn_rate=stats["churn_rate"],
        avg_monthly_charges=stats["avg_monthly_charges"],
        result=result,
        risk=risk,
        risk_level=risk_level,
        advice=advice,
        risk_factors=risk_factors
    )


if __name__ == "__main__":
    app.run(debug=True)