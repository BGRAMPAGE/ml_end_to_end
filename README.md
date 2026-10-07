## END TO END MACHINE LEARNING PROJECT

# Customer Churn Prediction — ML Project with CI/CD

A machine learning project that predicts whether a telecom customer is likely to **churn (leave the company)**.

The project is being built step-by-step with the eventual goal of creating a **production-ready ML system** with:

* Machine Learning model
* Data preprocessing pipeline
* MLflow for experiment/model tracking
* FastAPI for model serving
* Docker for containerization
* Automated testing
* GitHub Actions for CI/CD
* Container Registry
* Cloud deployment
* Model monitoring

> **Current status:** We have completed the initial dataset understanding, target selection, feature selection, and basic preprocessing. Model training and deployment have **not** been implemented yet.

---

# 1. Project Goal

The goal is to build a system that takes information about a telecom customer and predicts:

```text
Will this customer churn?
```

The prediction will be:

```text
0 → Customer will not churn
1 → Customer will churn
```

For example:

```text
Customer information
        ↓
Preprocessing
        ↓
Machine Learning Model
        ↓
Churn Prediction
        ↓
0 or 1
```

Later, this model will be exposed through an API so that an application can send customer information and receive a prediction.

---

# 2. Dataset

We are using the **Telco Customer Churn** dataset.

The dataset contains information about telecom customers, including:

* Customer demographics
* Services used
* Contract information
* Payment information
* Monthly charges
* Total charges
* Churn information

The dataset contains approximately **7,000 customer records**.

The original dataset contains several columns that are useful for analysis but should not necessarily be given directly to the machine learning model.

---

# 3. Prediction Target

Our target variable is:

```text
Churn Label
```

It contains:

```text
Yes
No
```

We convert it into numerical values:

```text
Yes → 1
No  → 0
```

Therefore:

```text
y = Churn Label
```

where:

```text
1 = Customer churned
0 = Customer did not churn
```

---

# 4. Features Used for Prediction

After examining the dataset, we selected the following features for the initial model.

## Customer Information

```text
Gender
Senior Citizen
Partner
Dependents
```

## Service Information

```text
Phone Service
Multiple Lines
Internet Service
Online Security
Online Backup
Device Protection
Tech Support
Streaming TV
Streaming Movies
```

## Contract and Billing Information

```text
Contract
Paperless Billing
Payment Method
```

## Numerical Information

```text
Tenure Months
Monthly Charges
Total Charges
```

Therefore, our initial model uses **19 input features**.

### Complete feature list

```text
1.  Gender
2.  Senior Citizen
3.  Partner
4.  Dependents
5.  Tenure Months
6.  Phone Service
7.  Multiple Lines
8.  Internet Service
9.  Online Security
10. Online Backup
11. Device Protection
12. Tech Support
13. Streaming TV
14. Streaming Movies
15. Contract
16. Paperless Billing
17. Payment Method
18. Monthly Charges
19. Total Charges
```

The target is:

```text
Churn Label
```

---

# 5. Why Some Columns Were Removed

The original dataset contains additional columns that we are not currently using as model inputs.

## CustomerID

```text
CustomerID
```

This is simply an identifier.

For example:

```text
7590-VHVEG
```

The ID does not contain meaningful information about whether a customer will churn.

Therefore:

```text
CustomerID → removed
```

---

## Count

```text
Count
```

This column represents a count value and does not provide useful predictive information for our initial model.

Therefore:

```text
Count → removed
```

---

## Geographic Information

We are currently removing:

```text
Country
State
City
Zip Code
Lat Long
Latitude
Longitude
```

These fields describe the customer's geographic location.

For the first version of the model, we are not using geographic information because it introduces additional complexity and is not necessary for building our initial churn prediction system.

We can revisit these features later if model analysis shows that geographic information is useful.

---

# 6. Avoiding Data Leakage

Some columns contain information that should **not** be provided to the model because they reveal the answer or information generated after the churn outcome.

This is called **data leakage**.

We removed:

```text
Churn Value
Churn Score
Churn Reason
```

### Churn Value

`Churn Value` is essentially another representation of the churn target.

Our target is already:

```text
Churn Label
```

Giving `Churn Value` to the model would duplicate the answer.

Therefore:

```text
Churn Value → removed
```

---

### Churn Reason

`Churn Reason` describes why a customer churned.

For example, it may contain information that is only known **after the customer has actually churned**.

When making a real prediction, we would not know the customer's future churn reason.

Therefore:

```text
Churn Reason → removed
```

---

### Churn Score

`Churn Score` is already a churn-related score.

If we give this to our model, the model could rely on an existing churn prediction rather than learning from the actual customer information.

Therefore:

```text
Churn Score → removed
```

---

# 7. Current Data Cleaning

One important issue we found is the `Total Charges` column.

Although it represents a numerical value, the dataset stores it in a form that may be interpreted as text.

Therefore we convert it to a numerical datatype.

Conceptually:

```python
df["Total Charges"] = pd.to_numeric(
    df["Total Charges"],
    errors="coerce"
)
```

If an invalid value is encountered, it becomes:

```text
NaN
```

At this early stage, we temporarily handle missing `Total Charges` values.

Later, when we build the proper machine learning pipeline, we will use a more robust preprocessing strategy with scikit-learn.

---

# 8. Features vs Target

Machine learning separates the dataset into two main parts.

## X — Features

These are the pieces of information given to the model.

```text
X = customer information
```

For example:

```text
Gender
Tenure Months
Internet Service
Contract
Monthly Charges
Total Charges
...
```

## y — Target

This is what the model is trying to predict.

```text
y = Churn Label
```

So:

```text
X → Customer information
y → Churn
```

The model learns the relationship:

```text
X → y
```

---

# 9. Why We Need Encoding

Several of our features contain text categories.

For example:

```text
Gender

Male
Female
```

or:

```text
Contract

Month-to-month
One year
Two year
```

Machine learning algorithms generally require numerical input.

Therefore, these categorical values need to be converted into numerical representations.

This process is called:

```text
Encoding
```

For example, using one-hot encoding:

```text
Contract

Month-to-month → 1 0 0
One year       → 0 1 0
Two year       → 0 0 1
```

This allows the ML model to work with the categorical information without incorrectly assuming that:

```text
Two year > One year > Month-to-month
```

The actual encoding will be implemented as part of our scikit-learn preprocessing pipeline.

---

# 10. Train/Test Split

Before training a machine learning model, we need to divide the data into:

```text
Training data
Testing data
```

The training data is used to teach the model.

The testing data is kept separate so we can evaluate how well the model performs on data it has not seen before.

Our initial split is:

```text
80% → Training
20% → Testing
```

We also use:

```python
random_state=42
```

This makes the split reproducible.

We use:

```python
stratify=y
```

so that the proportion of churned and non-churned customers remains approximately consistent between the training and testing datasets.

---

# 11. Current Preprocessing Flow

So far, our preprocessing logic is conceptually:

```text
Raw Dataset
     ↓
Load Excel file
     ↓
Remove irrelevant columns
     ↓
Remove data-leakage columns
     ↓
Convert Total Charges to numeric
     ↓
Separate features (X) and target (y)
     ↓
Convert Churn Label
     ↓
Train/Test Split
```

The result is:

```text
X → 19 prediction features
y → Churn (0 or 1)
```

---

# 12. Important: Preprocessing Is Not Finished Yet

The current preprocessing is only the **initial cleaning stage**.

We have NOT yet implemented the complete production preprocessing pipeline.

The next important step is to create a proper scikit-learn pipeline that handles:

```text
Numerical features
        ↓
Imputation / cleaning
        ↓
Categorical features
        ↓
One-Hot Encoding
        ↓
Combined feature matrix
        ↓
ML Model
```

This is important because the exact same preprocessing must happen during:

```text
Training
```

and later during:

```text
API Prediction
```

We do not want the training code to process data differently from the production API.

---

# 13. Project Architecture — Planned

The final project will eventually look approximately like this:

```text
                    ┌─────────────────┐
                    │  Telco Dataset  │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │  Preprocessing  │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Feature Encoding│
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │  Model Training │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │    MLflow       │
                    │ Tracking/Model  │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │   Saved Model   │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │    FastAPI      │
                    │   Prediction    │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │     Docker      │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │   CI/CD         │
                    │ GitHub Actions  │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Cloud Deployment│
                    └─────────────────┘
```

This is the **planned final architecture**. We have not implemented all of these components yet.

---

# 14. Technology Stack

The project will use the following technologies.

| Technology      | Purpose                       |
| --------------- | ----------------------------- |
| Python          | Programming language          |
| Pandas          | Data loading and manipulation |
| Scikit-learn    | Preprocessing and ML          |
| MLflow          | Experiment/model tracking     |
| FastAPI         | Model serving API             |
| Pytest          | Testing                       |
| Ruff            | Linting/code quality          |
| Docker          | Containerization              |
| Git             | Version control               |
| GitHub          | Source code repository        |
| GitHub Actions  | CI/CD                         |
| Docker Registry | Store container images        |
| Cloud Platform  | Deployment                    |

---

# 15. Current Project Status

### Completed

* [x] Python environment created
* [x] Dataset selected
* [x] Dataset inspected
* [x] Target variable identified
* [x] Prediction features identified
* [x] Irrelevant columns identified
* [x] Data leakage columns identified
* [x] Initial preprocessing logic created
* [x] `Total Charges` conversion identified
* [x] Feature/target separation understood
* [x] Train/test split understood
* [x] Need for categorical encoding understood

### Not Yet Implemented

* [ ] Complete sklearn preprocessing pipeline
* [ ] One-hot encoding
* [ ] Model training
* [ ] Model evaluation
* [ ] Experiment tracking with MLflow
* [ ] Model saving/registry
* [ ] FastAPI API
* [ ] API testing
* [ ] Docker
* [ ] GitHub Actions
* [ ] Container registry
* [ ] Cloud deployment
* [ ] Monitoring

---

# 16. Current Project Flow

At our current stage, the project is:

```text
Dataset
   ↓
Understand columns
   ↓
Select target
   ↓
Select useful features
   ↓
Remove irrelevant/leaking information
   ↓
Basic preprocessing
   ↓
Train/Test Split
   ↓
NEXT:
Complete preprocessing + encoding
```

The **next step** is to build the proper `ColumnTransformer` + `Pipeline` that handles numerical and categorical features automatically.
