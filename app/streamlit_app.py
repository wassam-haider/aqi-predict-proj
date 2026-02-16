# import os
# import mlflow
# import mlflow.sklearn
# import pandas as pd
# import numpy as np
# import streamlit as st
# from datetime import datetime, timedelta
# from dotenv import load_dotenv

# # ---------------- LOAD ENV ---------------- #
# load_dotenv()

# DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME")
# DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")
# DAGSHUB_REPO = os.getenv("DAGSHUB_REPO")

# os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
# os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

# tracking_uri = f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow"
# mlflow.set_tracking_uri(tracking_uri)

# # ---------------- LOAD MODEL ---------------- #
# model = mlflow.sklearn.load_model(
#     model_uri="models:/aqi_best_model@production"

# )

# # ---------------- STREAMLIT UI ---------------- #
# st.title("🌍 AQI 3-Day Prediction")

# st.write("Enter latest AQI and pollutant values")

# pm25 = st.number_input("PM2.5", value=50.0)
# pm10 = st.number_input("PM10", value=80.0)
# no2 = st.number_input("NO2", value=30.0)
# o3 = st.number_input("O3", value=20.0)
# current_aqi = st.number_input("Current AQI", value=100.0)

# if st.button("Predict Next 3 Days"):

#     predictions = []
#     current_time = datetime.now()

#     for i in range(1, 4):
#         future_time = current_time + timedelta(days=i)

#         input_data = pd.DataFrame([{
#             "pm25": pm25,
#             "pm10": pm10,
#             "no2": no2,
#             "o3": o3,
#             "aqi": current_aqi,
#             "hour": future_time.hour,
#             "day_of_week": future_time.weekday(),
#             "day_of_month": future_time.day,
#             "month": future_time.month
#         }])

#         pred = model.predict(input_data)[0]
#         predictions.append((future_time.date(), pred))

#     st.subheader("📊 Predictions")
#     for date, pred in predictions:
#         st.write(f"{date} → AQI Category: {pred}")
import os
import certifi
import mlflow
import joblib
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
from pymongo import MongoClient
from dotenv import load_dotenv

# ---------------- CONFIG ---------------- #
st.set_page_config(page_title="AQI Predictor", layout="wide")


load_dotenv()
DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")
DAGSHUB_REPO = os.getenv("DAGSHUB_REPO")

if DAGSHUB_USERNAME and DAGSHUB_TOKEN and DAGSHUB_REPO:
    os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN
    tracking_uri = f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow"
    mlflow.set_tracking_uri(tracking_uri)
    

MODEL_URI = "models:/aqi_best_model/2"


load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

# ---------------- LOAD MODEL FROM MLFLOW ---------------- #
@st.cache_resource
def load_model():
    model = mlflow.sklearn.load_model(MODEL_URI)
    return model

model = load_model()

# ---------------- LOAD PREPROCESSORS ---------------- #
@st.cache_resource
def load_artifacts():
    scaler = joblib.load("scaler.joblib")
    selector = joblib.load("selector.joblib")
    feature_cols = joblib.load("feature_columns.joblib")
    return scaler, selector, feature_cols

scaler, selector, feature_cols = load_artifacts()

# ---------------- FETCH DATA ---------------- #
@st.cache_data
def fetch_data():
    client = MongoClient(MONGO_URI, tls=True, tlsCAFile=certifi.where())
    db = client["aqi_db"]
    collection = db["aqi_data"]
    df = pd.DataFrame(list(collection.find({}, {"_id": 0})))
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    client.close()
    return df

df = fetch_data()

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

    if 'pm25' in df.columns and 'pm10' in df.columns:
        df['pm25_pm10_ratio'] = df['pm25'] / (df['pm10'] + 0.001)

    if 'no2' in df.columns and 'o3' in df.columns:
        df['nox_ratio'] = df['no2'] / (df['o3'] + 0.001)

    df = df.dropna()
    return df

df = create_features(df)

# ---------------- UI ---------------- #
st.title("🌍 AQI 3-Day Forecast")

if st.button("Predict Next 3 Days"):

    last_row = df.iloc[-1:].copy()
    predictions = []

    for i in range(1, 4):
        future_time = datetime.now() + timedelta(days=i)

        row = last_row.copy()
        row["timestamp"] = future_time

        row = create_features(pd.concat([df, row])).iloc[-1:]

        X_input = row[feature_cols]
        X_scaled = scaler.transform(X_input)
        X_selected = selector.transform(X_scaled)

        pred = model.predict(X_selected)[0]

        predictions.append({
            "date": future_time.date(),
            "predicted_aqi_class": int(pred)
        })

        row["aqi"] = pred
        df.loc[len(df)] = row.iloc[0]

    pred_df = pd.DataFrame(predictions)

    st.subheader("📊 Predictions")
    st.dataframe(pred_df)
