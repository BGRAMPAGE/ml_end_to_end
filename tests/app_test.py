from app import app


def test_home_page():
    client = app.test_client()

    response = client.get("/")

    assert response.status_code == 200
    assert b"Customer Churn" in response.data


def test_predict_page():
    client = app.test_client()

    response = client.get("/predict")

    assert response.status_code == 200


def test_models_page():
    client = app.test_client()

    response = client.get("/models")

    assert response.status_code == 200


def test_analytics_page():
    client = app.test_client()

    response = client.get("/analytics")

    assert response.status_code == 200


def test_health_endpoint():
    client = app.test_client()

    response = client.get("/health")

    assert response.status_code == 200

    data = response.get_json()

    assert data["status"] == "healthy"


def test_prediction_api():

    client = app.test_client()

    customer = {
        "Gender": "Male",
        "Senior Citizen": "No",
        "Partner": "Yes",
        "Dependents": "No",
        "Phone Service": "Yes",
        "Multiple Lines": "No",
        "Internet Service": "DSL",
        "Online Security": "No",
        "Online Backup": "Yes",
        "Device Protection": "No",
        "Tech Support": "No",
        "Streaming TV": "No",
        "Streaming Movies": "No",
        "Contract": "Month-to-month",
        "Paperless Billing": "Yes",
        "Payment Method": "Electronic check",
        "Tenure Months": 5,
        "Monthly Charges": 70.0,
        "Total Charges": 350.0,
        "CLTV": 4000,
    }

    response = client.post(
        "/api/predict",
        json=customer,
    )

    assert response.status_code == 200

    data = response.get_json()

    assert "prediction" in data
    assert "churn_probability" in data
    assert "threshold" in data

    assert data["prediction"] in [
        "Likely to Churn",
        "Likely to Stay",
    ]

    assert 0 <= data["churn_probability"] <= 1