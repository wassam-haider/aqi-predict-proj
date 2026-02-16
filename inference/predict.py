import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pymongo import MongoClient
import certifi
from dotenv import load_dotenv

# ---------------- LOAD ENV & MONGO ---------------- #
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

client = MongoClient(MONGO_URI, tls=True, tlsCAFile=certifi.where())
db = client["aqi_db"]
collection = db["aqi_data"]

# ---------------- FETCH DATA ---------------- #
df = pd.DataFrame(list(collection.find({}, {"_id": 0})))
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")

# ---------------- FEATURE ENGINEERING ---------------- #
def create_features(df):
    df = df.copy()
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_month'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    for lag in [1, 2, 3, 6, 12, 24]:
        df[f'aqi_lag_{lag}'] = df['aqi'].shift(lag)
    for window in [3, 6, 12, 24]:
        df[f'aqi_rolling_mean_{window}'] = df['aqi'].rolling(window).mean()
        df[f'aqi_rolling_std_{window}'] = df['aqi'].rolling(window).std()
    df['aqi_diff_1'] = df['aqi'].diff(1)
    df['aqi_diff_3'] = df['aqi'].diff(3)
    if all(col in df.columns for col in ['pm25', 'pm10']):
        df['pm25_pm10_ratio'] = df['pm25'] / (df['pm10'] + 0.001)
    if all(col in df.columns for col in ['no2', 'o3']):
        df['nox_ratio'] = df['no2'] / (df['o3'] + 0.001)
    df = df.dropna()
    return df

df = create_features(df)

# ---------------- LOAD PREPROCESSORS & MODEL ---------------- #
scaler = joblib.load("scaler.joblib")
selector = joblib.load("selector.joblib")
feature_cols = joblib.load("feature_columns.joblib")
best_model = joblib.load("RandomForest_model_regularized.joblib")  # replace with your choice

# ---------------- STABLE RECURSIVE PREDICTION FUNCTION ---------------- #
def stable_recursive_prediction(model, df, feature_cols, scaler, selector, n_days=3, use_prob_sampling=True):
    # (Paste the same function from your original script here, unchanged)
    ...

# ---------------- RUN PREDICTION ---------------- #
use_probabilistic = True
predictions, confidences = stable_recursive_prediction(
    best_model, df, feature_cols, scaler, selector,
    n_days=3, use_prob_sampling=use_probabilistic
)

dates = [(datetime.now() + timedelta(days=i)).date() for i in range(1, 4)]
pred_df = pd.DataFrame({
    "date": dates,
    "predicted_aqi_class": predictions,
    "confidence": confidences,
    "model_used": "RandomForest",
    "prediction_mode": "probabilistic" if use_probabilistic else "deterministic"
})
pred_df.to_csv("recursive_3day_predictions.csv", index=False)
print("✅ Predictions saved!")
