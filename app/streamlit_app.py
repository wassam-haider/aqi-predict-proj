import os
import certifi
import joblib
import mlflow
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import tempfile
from datetime import datetime, timedelta
from pymongo import MongoClient, errors
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="AQI Professional Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CONFIG & AUTH ---------------- #
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")
DAGSHUB_REPO = os.getenv("DAGSHUB_REPO")

# Configure MLflow
if DAGSHUB_USERNAME and DAGSHUB_TOKEN and DAGSHUB_REPO:
    os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN
    tracking_uri = f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow"
    mlflow.set_tracking_uri(tracking_uri)
else:
    st.error("MLflow credentials missing. Please check .env")

# ---------------- CONSTANTS ---------------- #
# Fallback hardcoded placeholders if download fails
HARDCODED_EDA = """
=== EDA REPORT (Placeholder) ===
Dataset Shape: (N, M)
Feature Summary:
Please run extract_stats.py locally or ensure MLflow artifacts are logged.
"""
HARDCODED_SHAP = """
=== SHAP FEATURE IMPORTANCE (Placeholder) ===
Feature    Importance
aqi_lag_1  0.50
hour       0.20
...
"""

CLASS_COLORS = {
    "Good": "green", "Fair": "lightgreen", "Moderate": "orange", 
    "Poor": "red", "Very Poor": "darkred"
}
CLASS_MAPPING = {0: "Good", 1: "Fair", 2: "Moderate", 3: "Poor", 4: "Very Poor"}

# ---------------- CACHED FUNCTIONS ---------------- #

@st.cache_resource
def get_model_and_reports():
    """Load model from MLflow and fetch EDA/SHAP reports from artifacts."""
    model_name = "aqi_best_model"
    model_version = "2" # Or fetch latest dynamically
    model_uri = f"models:/{model_name}/{model_version}"
    
    model = None
    eda_text = HARDCODED_EDA
    shap_text = HARDCODED_SHAP
    
    try:
        # Load Model
        model = mlflow.sklearn.load_model(model_uri)
        
        # Get Run ID to fetch artifacts
        client = MlflowClient()
        # Search for the version to get run_id
        # Note: mlflow.sklearn.load_model might not expose run_id easily on object
        # So we search versions
        versions = client.search_model_versions(f"name='{model_name}'")
        run_id = next((v.run_id for v in versions if v.version == model_version), None)
        
        if run_id:
            # List artifacts
            artifacts = client.list_artifacts(run_id)
            
            # Find reports (flexible matching)
            eda_file = next((x.path for x in artifacts if "eda_report.txt" in x.path), None)
            shap_file = next((x.path for x in artifacts if "shap_summary.txt" in x.path), None)
            
            # Download and Read
            with tempfile.TemporaryDirectory() as tmp_dir:
                if eda_file:
                    local_eda = client.download_artifacts(run_id, eda_file, dst_path=tmp_dir)
                    with open(local_eda, "r") as f: eda_text = f.read()
                    
                if shap_file:
                    local_shap = client.download_artifacts(run_id, shap_file, dst_path=tmp_dir)
                    with open(local_shap, "r") as f: shap_text = f.read()
                    
    except Exception as e:
        st.error(f"Error loading model/reports from MLflow: {e}")
        
    return model, eda_text, shap_text

@st.cache_resource
def load_artifacts_from_mlflow_or_local():
    """Load scaler, selector, feature_cols from MLflow or fallback to local."""
    # Similar logic: fetch from run_id of best model?
    # Or just use local model_files if deployed with repo.
    # Assuming repo deployment -> use local files for artifacts as they are static.
    # If purely cloud -> should fetch from MLflow too.
    # We'll stick to local for artifacts to minimize latency and complexity, assuming they are in repo.
    try:
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model_files"))
        
        # Check if folder exists
        if not os.path.exists(base_path):
             # Try fetching from MLflow if local missing (Advanced)
             return None, None, None, 0

        scaler = joblib.load(os.path.join(base_path, "scaler.joblib"))
        selector = joblib.load(os.path.join(base_path, "selector.joblib"))
        feature_cols = joblib.load(os.path.join(base_path, "feature_columns.joblib"))
        
        min_class_path = os.path.join(base_path, "min_class.joblib")
        if os.path.exists(min_class_path):
            min_class = joblib.load(min_class_path)
        else:
            min_class = 0
            
        return scaler, selector, feature_cols, min_class
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        return None, None, None, 0

@st.cache_data(ttl=600)
def fetch_data():
    if not MONGO_URI: return pd.DataFrame()
    try:
        client = MongoClient(MONGO_URI, tls=True, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=5000)
        data = list(client["aqi_db"]["aqi_data"].find({}, {"_id": 0}))
        df = pd.DataFrame(data)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp")
        return df
    except Exception as e:
        st.error(f"MongoDB Error: {e}")
        return pd.DataFrame()

def create_features(df):
    df = df.copy()
    if 'timestamp' in df.columns:
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
    
    if 'pm25' in df.columns and 'pm10' in df.columns:
        df['pm25_pm10_ratio'] = df['pm25'] / (df['pm10'] + 0.001)
    if 'no2' in df.columns and 'o3' in df.columns:
        df['nox_ratio'] = df['no2'] / (df['o3'] + 0.001)
        
    return df

# ---------------- UI SECTIONS ---------------- #

def render_overview(df):
    st.markdown("## 📊 Overview")
    if df.empty:
        st.info("No Data.")
        return
        
    last = df.iloc[-1]
    aqi = int(last['aqi'])
    # Map raw AQI or class
    label = CLASS_MAPPING.get(aqi if aqi < 5 else aqi-1, "Unknown") # Heuristic
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", len(df))
    c2.metric("Latest AQI", aqi)
    c3.metric("Class", label)
    c4.metric("Updated", last['timestamp'].strftime('%Y-%m-%d %H:%M'))
    
def render_historical(df):
    st.markdown("---")
    st.subheader("📈 Historical Trends")
    days = st.slider("Days", 7, 30, 14, key="hist_slider")
    subset = df[df['timestamp'] >= df['timestamp'].max() - timedelta(days=days)].copy()
    
    # Check if aqi is class (0-4) or raw.
    is_class = subset['aqi'].max() <= 10
    
    if is_class:
        subset['label'] = subset['aqi'].apply(lambda x: CLASS_MAPPING.get(x if x < 5 else x-1, "Unknown"))
        subset['color'] = subset['label'].apply(lambda x: CLASS_COLORS.get(x, "grey"))
    else:
        # Binning logic if needed, or just color gradient
        subset['color'] = "blue"
        
    fig = px.line(subset, x='timestamp', y='aqi', title=f"AQI History ({days} Days)")
    fig.add_trace(go.Scatter(
        x=subset['timestamp'], y=subset['aqi'], mode='markers',
        marker=dict(color=subset['color'], size=8), showlegend=False
    ))
    st.plotly_chart(fig, use_container_width=True)

def render_forecast(df, model, scaler, selector, feat_cols, min_class):
    st.markdown("---")
    st.subheader("🔮 3-Day Forecast")
    if st.button("Predict Future"):
        if not model or scaler is None or selector is None:
            st.error("Model or artifacts unavailable for prediction.")
            return
            
        history = df.copy()
        preds = []
        last_time = history['timestamp'].max()
        
        for i in range(1, 4):
            next_time = last_time + timedelta(days=i)
            # Use last known AQI to prevent nulls in diff/rolling
            last_val = history.iloc[-1]['aqi']
            
            # Placeholder row
            row = {"timestamp": next_time, "aqi": last_val}
            # Add to history
            temp_df = pd.concat([history, pd.DataFrame([row])], ignore_index=True).ffill()
            
            # Featurize
            feats = create_features(temp_df)
            target = feats.iloc[[-1]] # Last row
            
            # Ensure columns
            X = target[feat_cols]
            # Transform
            X_sc = scaler.transform(X)
            X_sel = selector.transform(X_sc)
            
            # Predict
            pred_inv = model.predict(X_sel)[0]
            pred_val = int(pred_inv + min_class)
            
            preds.append({"Date": next_time.date(), "AQI": pred_val, "Class": CLASS_MAPPING.get(pred_val, "Unknown")})
            
            # Update history with PREDICTION for next step recursion
            # Correct the placeholder
            history = pd.concat([history, pd.DataFrame([{"timestamp": next_time, "aqi": pred_val}])], ignore_index=True).ffill()
            
        st.table(pd.DataFrame(preds))
        
        # Plot
        fc_df = pd.DataFrame(preds).rename(columns={"Date": "timestamp", "AQI": "aqi"})
        fc_df["timestamp"] = pd.to_datetime(fc_df["timestamp"])
        combined = pd.concat([df.tail(10)[["timestamp", "aqi"]].assign(Type="History"), fc_df.assign(Type="Forecast")])
        fig = px.line(combined, x='timestamp', y='aqi', color='Type', line_dash='Type')
        st.plotly_chart(fig, use_container_width=True)

def render_reports(eda_text, shap_text):
    st.markdown("---")
    st.subheader("🔍 Model Explanations")
    
    t1, t2 = st.tabs(["EDA Report", "SHAP Importance"])
    
    with t1:
        st.text(eda_text)
        
    with t2:
        st.text(shap_text)

# ---------------- MAIN ---------------- #
def main():
    scaler, selector, feat_cols, min_class = load_artifacts_from_mlflow_or_local()
    model, eda_txt, shap_txt = get_model_and_reports()
    df = fetch_data()
    
    render_overview(df)
    
    if not df.empty:
        render_historical(df)
        render_forecast(df, model, scaler, selector, feat_cols, min_class)
        render_reports(eda_txt, shap_txt)

if __name__ == "__main__":
    main()
