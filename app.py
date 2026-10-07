from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import json
from model_results import MODEL_RESULTS

app = Flask(__name__)


model = joblib.load("customer_churn_model.pkl")



CHURN_THRESHOLD = 0.35


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analytics")
def analytics():

    with open("model_metadata.json", "r") as file:
        metadata = json.load(file)

    return render_template(
        "analytics.html",
        metadata=metadata,
    )


@app.route("/models")
def models():

    return render_template(
        "models.html",
        results=MODEL_RESULTS,
    )

@app.route("/predict", methods=["GET", "POST"])
def predict():

    if request.method == "GET":
        return render_template("predict.html")

    data = request.form

    customer = pd.DataFrame(
        [
            {
                "Gender": data["Gender"],
                "Senior Citizen": data["Senior Citizen"],
                "Partner": data["Partner"],
                "Dependents": data["Dependents"],
                "Phone Service": data["Phone Service"],
                "Multiple Lines": data["Multiple Lines"],
                "Internet Service": data["Internet Service"],
                "Online Security": data["Online Security"],
                "Online Backup": data["Online Backup"],
                "Device Protection": data["Device Protection"],
                "Tech Support": data["Tech Support"],
                "Streaming TV": data["Streaming TV"],
                "Streaming Movies": data["Streaming Movies"],
                "Contract": data["Contract"],
                "Paperless Billing": data["Paperless Billing"],
                "Payment Method": data["Payment Method"],
                "Tenure Months": float(data["Tenure Months"]),
                "Monthly Charges": float(data["Monthly Charges"]),
                "Total Charges": float(data["Total Charges"]),
                "CLTV": float(data["CLTV"]),
            }
        ]
    )

    probability = model.predict_proba(customer)[0][1]

    prediction = (
        "Likely to Churn"
        if probability >= CHURN_THRESHOLD
        else "Likely to Stay"
    )

    return render_template(
        "predict.html",
        prediction=prediction,
        probability=round(probability * 100, 2),
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():

    data = request.get_json()

    customer = pd.DataFrame([data])

    probability = model.predict_proba(customer)[0][1]

    prediction = (
        "Likely to Churn"
        if probability >= CHURN_THRESHOLD
        else "Likely to Stay"
    )

    return jsonify(
    {
        "prediction": prediction,
        "churn_probability": float(
            probability
        ),
        "threshold": float(
            CHURN_THRESHOLD
        ),
    }
)


@app.route("/health")
def health():

    return jsonify(
        {
            "status": "healthy",
            "model": "XGBoost",
        }
    )


if __name__ == "__main__":
    app.run(debug=True)