import os
import time
import certifi
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
from datetime import datetime
from pymongo import MongoClient, errors

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# ---------------- LOAD ENV ---------------- #
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")
DAGSHUB_REPO = os.getenv("DAGSHUB_REPO")

if not MONGO_URI:
    raise ValueError("MONGO_URI not found in .env")

# ---------------- MLFLOW DAGSHUB SETUP ---------------- #
if DAGSHUB_USERNAME and DAGSHUB_TOKEN and DAGSHUB_REPO:
    os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN
    tracking_uri = f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow"
    mlflow.set_tracking_uri(tracking_uri)

mlflow.set_experiment("aqi_prediction_experiment")


# ---------------- SAFE MONGO CONNECTION ---------------- #
def get_mongo_client(uri):
    for attempt in range(5):
        try:
            client = MongoClient(uri, tls=True, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=20000)
            client.admin.command("ping")
            print("✅ Connected to MongoDB")
            return client
        except errors.ServerSelectionTimeoutError:
            print(f"Retry {attempt+1}/5 MongoDB connection...")
            time.sleep(5)
    raise ConnectionError("❌ Could not connect to MongoDB")


client = get_mongo_client(MONGO_URI)
db = client["aqi_db"]
collection = db["aqi_data"]

# ---------------- FETCH DATA ---------------- #
print("Fetching data...")
df = pd.DataFrame(list(collection.find({}, {"_id": 0})))
if df.empty:
    raise ValueError("No data found in MongoDB")

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")


# ---------------- FEATURE ENGINEERING ---------------- #
def create_features(df):
    df = df.copy()

    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_month'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month

    for lag in [1,2,3,6,12,24]:
        df[f'aqi_lag_{lag}'] = df['aqi'].shift(lag)

    for window in [3,6,12,24]:
        df[f'aqi_rolling_mean_{window}'] = df['aqi'].rolling(window).mean()
        df[f'aqi_rolling_std_{window}'] = df['aqi'].rolling(window).std()

    df['aqi_diff_1'] = df['aqi'].diff(1)
    df['aqi_diff_3'] = df['aqi'].diff(3)

    if all(col in df.columns for col in ['pm25','pm10']):
        df['pm25_pm10_ratio'] = df['pm25'] / (df['pm10'] + 0.001)

    if all(col in df.columns for col in ['no2','o3']):
        df['nox_ratio'] = df['no2'] / (df['o3'] + 0.001)

    df = df.dropna()
    return df


df = create_features(df)

# ---------------- DEFINE FEATURES ---------------- #
exclude_cols = ['aqi','timestamp']
feature_cols = [col for col in df.columns if col not in exclude_cols]

X = df[feature_cols]
y = df['aqi'].astype(int)

min_class = y.min()
y = y - min_class

split_date = df['timestamp'].quantile(0.8, interpolation='nearest')

X_train = X[df['timestamp'] < split_date]
X_test = X[df['timestamp'] >= split_date]
y_train = y[df['timestamp'] < split_date]
y_test = y[df['timestamp'] >= split_date]


# ---------------- SCALE + FEATURE SELECTION ---------------- #
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

selector = SelectKBest(f_classif, k=min(20, len(feature_cols)))
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]


# ---------------- HYPERPARAMETER TUNING ---------------- #
param_grid = {
    'n_estimators':[50,100],
    'max_depth':[5,10],
    'min_samples_split':[5,10],
    'min_samples_leaf':[2,5]
}

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
rf_grid.fit(X_train_selected, y_train)


# ---------------- MODELS ---------------- #
models = {
    "RandomForest": RandomForestClassifier(**rf_grid.best_params_, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        min_samples_split=10,
        min_samples_leaf=5,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    ),
    "LogisticRegression": LogisticRegression(
        C=0.1,
        penalty='l2',
        solver='liblinear',
        max_iter=2000,
        random_state=42
    )
}


# ---------------- TRAIN + LOG EVERYTHING ---------------- #
for name, model in models.items():
    print(f"\n🚀 Training {name}...")

    with mlflow.start_run(run_name=name):

        model.fit(X_train_selected, y_train)
        y_pred = model.predict(X_test_selected)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="macro")
        recall = recall_score(y_test, y_pred, average="macro")
        f1 = f1_score(y_test, y_pred, average="macro")

        mlflow.log_param("model_name", name)
        mlflow.log_param("num_features", len(selected_features))

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision_macro", precision)
        mlflow.log_metric("recall_macro", recall)
        mlflow.log_metric("f1_macro", f1)
        #save preproccessing artifacts
        joblib.dump(scaler, "scaler.joblib")
        joblib.dump(selector, "selector.joblib")
        joblib.dump(feature_cols, "feature_columns.joblib")
        joblib.dump(selected_features, "selected_features.joblib")

        mlflow.log_artifact("scaler.joblib")
        mlflow.log_artifact("selector.joblib")
        mlflow.log_artifact("feature_columns.joblib")
        mlflow.log_artifact("selected_features.joblib")

        # Classification Report
        # report = classification_report(y_test, y_pred)
        # report_file = f"{name}_classification_report.txt"
        # with open(report_file, "w") as f:
        #     f.write(report)
        # mlflow.log_artifact(report_file)

        # # Confusion Matrix
        # cm = confusion_matrix(y_test, y_pred)
        # plt.figure()
        # sns.heatmap(cm, annot=True, fmt="d")
        # plt.title(f"{name} Confusion Matrix")
        # plt.xlabel("Predicted")
        # plt.ylabel("Actual")

        # cm_file = f"{name}_confusion_matrix.png"
        # plt.savefig(cm_file)
        # mlflow.log_artifact(cm_file)
        # plt.close()

        # Log model to registry
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="aqi_best_model"
        )

        print(f"✅ {name}")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")


client.close()
print("\n🎯 Training complete. All models logged to MLflow + Registry.")




