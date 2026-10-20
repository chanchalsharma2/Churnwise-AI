import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "IBM/telco-customer-churn-on-icp4d/master/"
    "data/Telco-Customer-Churn.csv"
)

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

print("Downloading customer dataset...")
df = pd.read_csv(DATA_URL)

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

df = df.dropna(subset=["Churn"]).copy()

target = "Churn"
y = df[target].map({"Yes": 1, "No": 0})

X = df.drop(columns=[target, "customerID"], errors="ignore")

numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training machine-learning model...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nModel evaluation")
print("----------------")
print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")
print(classification_report(
    y_test, predictions, target_names=["No Churn", "Churn"]
))

joblib.dump(model, "models/churn_model.joblib")
df.to_csv("data/customer_churn.csv", index=False)

print("\nSuccess! Model saved in models/churn_model.joblib")
print("Dataset saved in data/customer_churn.csv")