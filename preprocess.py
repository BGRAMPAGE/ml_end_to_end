import pandas as pd

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = Path(__file__).resolve().parent / "Telco_customer_churn.xlsx"


def load_data(path: str) -> pd.DataFrame:
    return pd.read_excel(path)


def preprocess_data(df: pd.DataFrame):
    columns_to_drop = [
        "CustomerID",
        "Count",
        "Country",
        "State",
        "City",
        "Zip Code",
        "Lat Long",
        "Latitude",
        "Longitude",
        "Churn Value",
        "Churn Score",
        "Churn Reason",
    ]

    df = df.drop(columns=columns_to_drop)

    df["Total Charges"] = pd.to_numeric(
        df["Total Charges"],
        errors="coerce",
    )

    y = (df["Churn Label"] == "Yes").astype(int)

    X = df.drop(columns=["Churn Label"])

    X["Total Charges"] = X["Total Charges"].fillna(0)

    return X, y


CATEGORICAL_FEATURES = [
    "Gender",
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Contract",
    "Paperless Billing",
    "Payment Method",
]


preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            CATEGORICAL_FEATURES,
        ),
    ],
    remainder="passthrough",
)


def get_processed_data():
    df = load_data(DATA_PATH)

    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.transform(X_test)

    return X_train_encoded, X_test_encoded, y_train, y_test


if __name__ == "__main__":
    X_train_encoded, X_test_encoded, y_train, y_test = get_processed_data()

    print("Encoded training shape:", X_train_encoded.shape)
    print("Encoded testing shape:", X_test_encoded.shape)