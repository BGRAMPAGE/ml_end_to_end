import joblib

from sklearn.metrics import precision_recall_curve
from preprocess import (
    DATA_PATH,
    CATEGORICAL_FEATURES,
    preprocess_data,
)
import json
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from xgboost import XGBClassifier



df = pd.read_excel(DATA_PATH)

X, y = preprocess_data(df)



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)




NUMERICAL_FEATURES = [
    "Tenure Months",
    "Monthly Charges",
    "Total Charges",
    "CLTV",
]




categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ]
)


numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL_FEATURES,
        ),
        (
            "numerical",
            numerical_pipeline,
            NUMERICAL_FEATURES,
        ),
    ]
)




models = {

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        random_state=42,
    ),

    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        random_state=42,
        eval_metric="logloss",
    ),
}




results = []
trained_models = {}


for name, model in models.items():

    print("\n" + "=" * 60)
    print(f"Training {name}")
    print("=" * 60)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    
    cv_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
    )

    print(
        f"5-Fold CV ROC-AUC: "
        f"{cv_scores.mean():.4f} "
        f"+/- {cv_scores.std():.4f}"
    )

    
    pipeline.fit(X_train, y_train)

    
    y_pred = pipeline.predict(X_test)

    y_probability = pipeline.predict_proba(X_test)[:, 1]
    
    

    thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    print("\nThreshold Analysis")
    print("-" * 70)

    for threshold in thresholds:

        y_threshold_pred = (
            y_probability >= threshold
        ).astype(int)

        threshold_precision = precision_score(
            y_test,
            y_threshold_pred,
            zero_division=0,
        )

        threshold_recall = recall_score(
            y_test,
            y_threshold_pred,
            zero_division=0,
        )

        threshold_f1 = f1_score(
            y_test,
            y_threshold_pred,
            zero_division=0,
        )

        print(
            f"Threshold: {threshold:.2f} | "
            f"Precision: {threshold_precision:.4f} | "
            f"Recall: {threshold_recall:.4f} | "
            f"F1: {threshold_f1:.4f}"
        )

    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_probability)

    results.append(
        {
            "Model": name,
            "CV ROC-AUC": cv_scores.mean(),
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "ROC-AUC": roc_auc,
        }
    )

    trained_models[name] = pipeline




results_df = pd.DataFrame(results)

print("\n")
print("=" * 90)
print("MODEL COMPARISON")
print("=" * 90)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)



best_model_name = results_df.loc[
    results_df["CV ROC-AUC"].idxmax(),
    "Model",
]

best_model = trained_models[best_model_name]

best_probability = best_model.predict_proba(X_test)[:, 1]

best_prediction = (
    best_probability >= 0.35
).astype(int)

cm = confusion_matrix(
    y_test,
    best_prediction,
)

fpr, tpr, _ = roc_curve(
    y_test,
    best_probability,
)



metadata = {
    "best_model": best_model_name,

    "threshold": 0.35,

    "metrics": {
        "accuracy": accuracy_score(
            y_test,
            best_prediction,
        ),

        "precision": precision_score(
            y_test,
            best_prediction,
            zero_division=0,
        ),

        "recall": recall_score(
            y_test,
            best_prediction,
            zero_division=0,
        ),

        "f1": f1_score(
            y_test,
            best_prediction,
            zero_division=0,
        ),

        "roc_auc": roc_auc_score(
            y_test,
            best_probability,
        ),
    },

    "confusion_matrix": cm.tolist(),

    "roc_curve": {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
    },
}


with open(
    "model_metadata.json",
    "w",
) as file:

    json.dump(
        metadata,
        file,
        indent=4,
    )


print("\nModel metadata saved as model_metadata.json")
print("\n" + "=" * 60)
print(f"BEST MODEL: {best_model_name}")
print("=" * 60)




joblib.dump(
    best_model,
    "customer_churn_model.pkl",
)

print("\nModel saved as customer_churn_model.pkl")