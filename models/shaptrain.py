import os
import time
import certifi
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import shap
from dotenv import load_dotenv
from datetime import datetime
from pymongo import MongoClient, errors

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

import matplotlib.pyplot as plt
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
            client = MongoClient(
                uri,
                tls=True,
                tlsCAFile=certifi.where(),
                serverSelectionTimeoutMS=20000
            )
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

# ---------------- FEATURES ---------------- #
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


# ---------------- SCALE + SELECT ---------------- #
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

selector = SelectKBest(f_classif, k=min(20, len(feature_cols)))
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]


# ---------------- HYPERPARAM TUNING ---------------- #
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
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    ),
    "LogisticRegression": LogisticRegression(
        C=0.1,
        max_iter=2000,
        multi_class='auto',
        solver='lbfgs',
        random_state=42
    )
}


# ---------------- TRAIN + LOG ---------------- #
for name, model in models.items():

    print(f"\n🚀 Training {name}...")

    with mlflow.start_run(run_name=name):

        model.fit(X_train_selected, y_train)
        y_pred = model.predict(X_test_selected)

        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="macro")
        recall = recall_score(y_test, y_pred, average="macro")
        f1 = f1_score(y_test, y_pred, average="macro")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision_macro", precision)
        mlflow.log_metric("recall_macro", recall)
        mlflow.log_metric("f1_macro", f1)

        # -------- SAVE PREPROCESSING -------- #
        joblib.dump(scaler, "scaler.joblib")
        joblib.dump(selector, "selector.joblib")
        joblib.dump(feature_cols, "feature_columns.joblib")
        joblib.dump(selected_features, "selected_features.joblib")

        mlflow.log_artifact("scaler.joblib")
        mlflow.log_artifact("selector.joblib")
        mlflow.log_artifact("feature_columns.joblib")
        mlflow.log_artifact("selected_features.joblib")

        # -------- EDA REPORT -------- #
        eda_text = []
        eda_text.append("=== EDA REPORT ===\n")
        eda_text.append(f"Dataset Shape: {df.shape}\n")
        eda_text.append("\nClass Distribution:\n")
        eda_text.append(y.value_counts().to_string())
        eda_text.append("\n\nFeature Summary:\n")
        eda_text.append(df.describe().to_string())

        corr = df[selected_features + ['aqi']].corr()['aqi'].abs().sort_values(ascending=False)
        eda_text.append("\n\nTop Correlated Features:\n")
        eda_text.append(corr.head(10).to_string())

        eda_file = f"{name}_eda_report.txt"
        with open(eda_file, "w") as f:
            f.write("\n".join(eda_text))

        mlflow.log_artifact(eda_file)

        # -------- SHAP ANALYSIS -------- #
        try:
            if name in ["RandomForest", "GradientBoosting"]:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test_selected)

                if isinstance(shap_values, list):
                    shap_array = np.mean(np.abs(np.array(shap_values)), axis=(0,1))
                else:
                    shap_array = np.mean(np.abs(shap_values), axis=0)

            else:
                explainer = shap.LinearExplainer(model, X_train_selected)
                shap_values = explainer.shap_values(X_test_selected)

                if isinstance(shap_values, list):
                    shap_array = np.mean(np.abs(np.array(shap_values)), axis=(0,1))
                else:
                    shap_array = np.mean(np.abs(shap_values), axis=0)

            shap_importance = pd.Series(shap_array, index=selected_features)
            shap_importance = shap_importance.sort_values(ascending=False)

            shap_text = ["=== SHAP FEATURE IMPORTANCE ===\n"]
            shap_text.append(shap_importance.head(15).to_string())

            shap_file = f"{name}_shap_summary.txt"
            with open(shap_file, "w") as f:
                f.write("\n".join(shap_text))

            mlflow.log_artifact(shap_file)

            # SHAP Plot
            plt.figure()
            shap_importance.head(15).plot(kind='barh')
            plt.title(f"{name} SHAP Feature Importance")
            plt.gca().invert_yaxis()

            shap_plot_file = f"{name}_shap_plot.png"
            plt.savefig(shap_plot_file, bbox_inches='tight')
            plt.close()

            mlflow.log_artifact(shap_plot_file)

        except Exception as e:
            print("SHAP failed:", e)

        # -------- LOG MODEL -------- #
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="aqi_best_model"
        )

        print(f"✅ {name}")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Score: {f1:.4f}")

client.close()
print("\n🎯 Training complete. All models logged with EDA + SHAP.")
