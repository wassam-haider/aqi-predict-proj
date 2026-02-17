import os
import certifi
import joblib
import mlflow
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pymongo import MongoClient, errors
from dotenv import load_dotenv
from io import StringIO

import requests

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Karachi AQI Dashboard",
    page_icon="�🇰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CONFIG & AUTH ---------------- #
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# Karachi Coordinates
LAT = 24.8607
LON = 67.0011

# Color Scheme
CLASS_COLORS = {
    "Good": "green", "Fair": "lightgreen", "Moderate": "orange", 
    "Poor": "red", "Very Poor": "darkred"
}
CLASS_MAPPING = {
    0: "Good", 1: "Fair", 2: "Moderate", 3: "Poor", 4: "Very Poor", 
    5: "Very Poor" 
}

# ---------------- REPORT PARSERS ---------------- #
def parse_eda_report(text):
    sections = {}
    lines = text.split("\n")
    current_section = None
    buffer = []

    def process_buffer(section, buf):
        if not buf or not section: return
        
        if section == "class_dist":
            data = []
            for line in buf:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        # try picking the last valid int
                        val = int(parts[-1])
                        cls = " ".join(parts[:-1])
                        if cls.lower() not in ["aqi", "class"]:
                            data.append({"Class": cls, "Count": val})
                    except: pass
            if data: sections["class_dist"] = pd.DataFrame(data)

        elif section == "feature_summary":
            try:
                table_str = "\n".join(buf)
                # Filter out lines that look like headers if dragged in
                sections["feature_summary"] = pd.read_csv(StringIO(table_str), sep=r"\s+", engine='python')
            except: pass

        elif section == "correlation":
            data = []
            for line in buf:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        val = float(parts[-1])
                        feat = " ".join(parts[:-1])
                        if feat.lower() not in ["feature", "correlation"]:
                             data.append({"Feature": feat, "Correlation": val})
                    except: pass
            if data: sections["correlation"] = pd.DataFrame(data)

    for line in lines:
        stripped = line.strip()
        if not stripped: continue
        
        # State transitions
        if "Class Distribution" in line:
            process_buffer(current_section, buffer)
            current_section = "class_dist"
            buffer = []
        elif "Feature Summary" in line:
            process_buffer(current_section, buffer)
            current_section = "feature_summary"
            buffer = []
        elif "Top Correlated Features" in line:
            process_buffer(current_section, buffer)
            current_section = "correlation"
            buffer = []
        elif "===" in line:
            continue
        else:
            if current_section:
                buffer.append(line)
                
    # Flush last buffer
    process_buffer(current_section, buffer)
    
    return sections

def parse_shap_report(text):
    try:
        lines = text.split("\n")
        start_idx = -1
        for i, line in enumerate(lines):
            if "Feature" in line and "Importance" in line:
                start_idx = i + 1
                break
        if start_idx == -1: return pd.DataFrame()
        
        data = []
        for line in lines[start_idx:]:
            if not line.strip(): continue
            p = line.split()
            if len(p) >= 2: data.append({"Feature": p[0], "Importance": float(p[1])})
        return pd.DataFrame(data)
    except:
        return pd.DataFrame()

# ---------------- CACHED FUNCTIONS ---------------- #
@st.cache_resource
def get_reports(model_name):
    """Load and parse reports for selected model."""
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
    
    eda_data = {}
    shap_df = pd.DataFrame()
    
    eda_path = os.path.join(MODELS_DIR, f"{model_name}_eda_report.txt")
    shap_path = os.path.join(MODELS_DIR, f"{model_name}_shap_summary.txt")
    
    if os.path.exists(eda_path):
        with open(eda_path, "r") as f: eda_data = parse_eda_report(f.read())
            
    if os.path.exists(shap_path):
        with open(shap_path, "r") as f: shap_df = parse_shap_report(f.read())
        
    return eda_data, shap_df

@st.cache_resource
def load_artifacts():
    try:
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model_files"))
        
        # Load Best Model (RandomForest for now as default)
        model_path = os.path.join(base_path, "RandomForest_model.joblib")
        # Try local first
        if os.path.exists(model_path):
            model = joblib.load(model_path)
        else:
            # Fallback to MLflow if needed, or None
            try:
                model = mlflow.sklearn.load_model("models:/aqi_best_model/2")
            except:
                model = None

        scaler = joblib.load(os.path.join(base_path, "scaler.joblib"))
        selector = joblib.load(os.path.join(base_path, "selector.joblib"))
        feature_cols = joblib.load(os.path.join(base_path, "feature_columns.joblib"))
        
        min_class_path = os.path.join(base_path, "min_class.joblib")
        min_class = joblib.load(min_class_path) if os.path.exists(min_class_path) else 0
            
        return model, scaler, selector, feature_cols, min_class
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        return None, None, None, None, 0

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

@st.cache_data(ttl=900)
def fetch_live_aqi():
    """Fetch real-time AQI from OpenWeather API."""
    if not OPENWEATHER_API_KEY: return None
    try:
        url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={OPENWEATHER_API_KEY}"
        resp = requests.get(url, timeout=5).json()
        if "list" in resp and resp["list"]:
            data = resp["list"][0]
            return {
                "aqi": data["main"]["aqi"],
                "components": data["components"],
                "timestamp": datetime.utcnow()
            }
    except Exception as e:
        print(f"Live API Error: {e}")
    return None

# ---------------- UI RENDERERS ---------------- #
def render_overview(df):
    
    # Custom Header
    st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
            padding: 2rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        </style>
        <div class="main-header">
            <h1>🇵🇰 Karachi Air Quality Dashboard</h1>
            <p>Real-time Monitoring & Future Predictions</p>
        </div>
    """, unsafe_allow_html=True)
    
    live_data = fetch_live_aqi()
    
    # Determine what to show
    if live_data:
        current_aqi = live_data["aqi"]
        # Karachi is UTC+5
        last_updated = live_data["timestamp"] + timedelta(hours=5)
        source = "🟢 Live API (OpenWeather)"
        # Extra details from live
        pm25 = live_data["components"]["pm2_5"]
        pm10 = live_data["components"]["pm10"]
    elif not df.empty:
        last = df.iloc[-1]
        current_aqi = int(last['aqi'])
        # Taking DB timestamp as is (assuming it was collected in approximate local time or we blindly show it)
        last_updated = last['timestamp']
        source = "🟠 Database (Historical)"
        # Try to find PM2.5 if in columns
        pm25 = last.get("pm25", "N/A")
        pm10 = last.get("pm10", "N/A")
    else:
        st.error("No data available.")
        return

    label = CLASS_MAPPING.get(current_aqi if current_aqi < 5 else current_aqi-1, "Unknown")
    color = CLASS_COLORS.get(label, "grey")

    # Metrics Row
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown(f"### Current AQI")
        st.markdown(f"<h1 style='color: {color}; font-size: 3rem;'>{current_aqi}</h1>", unsafe_allow_html=True)
        st.caption(f"Status: **{label}**")
        
    with c2:
        st.markdown("### Pollutants")
        p25_val = f"{pm25:.1f}" if isinstance(pm25, (int, float)) else pm25
        p10_val = f"{pm10:.1f}" if isinstance(pm10, (int, float)) else pm10
        st.metric("PM2.5", p25_val, "µg/m³")
        st.metric("PM10", p10_val, "µg/m³")
        
    with c3:
        st.markdown("### Info")
        st.metric("Source", source)
        st.metric("Last Updated", last_updated.strftime('%H:%M %p, %d %b %Y'))

def render_historical(df):
    st.markdown("---")
    st.subheader("📈 Historical Trends")
    days = st.slider("Days", 7, 30, 14, key="hist_slider")
    subset = df[df['timestamp'] >= df['timestamp'].max() - timedelta(days=days)].copy()
    is_class = subset['aqi'].max() <= 10
    if is_class:
        subset['label'] = subset['aqi'].apply(lambda x: CLASS_MAPPING.get(x if x < 5 else x-1, "Unknown"))
        subset['color'] = subset['label'].apply(lambda x: CLASS_COLORS.get(x, "grey"))
    else:
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
            st.error("Model or artifacts unavailable.")
            return
        history = df.copy()
        preds = []
        last_time = history['timestamp'].max()
        for i in range(1, 4):
            next_time = last_time + timedelta(days=i)
            last_val = history.iloc[-1]['aqi']
            row = {"timestamp": next_time, "aqi": last_val}
            temp_df = pd.concat([history, pd.DataFrame([row])], ignore_index=True).ffill()
            feats = create_features(temp_df)
            target = feats.iloc[[-1]]
            X = target[feat_cols]
            X_sc = scaler.transform(X)
            X_sel = selector.transform(X_sc)
            pred_inv = model.predict(X_sel)[0]
            pred_val = int(pred_inv + min_class)
            preds.append({"Date": next_time.date(), "AQI": pred_val, "Class": CLASS_MAPPING.get(pred_val, "Unknown")})
            history = pd.concat([history, pd.DataFrame([{"timestamp": next_time, "aqi": pred_val}])], ignore_index=True).ffill()
        
        c1, c2 = st.columns([1, 2])
        with c1:
            st.table(pd.DataFrame(preds))
        with c2:
            fc_df = pd.DataFrame(preds).rename(columns={"Date": "timestamp", "AQI": "aqi"})
            fc_df["timestamp"] = pd.to_datetime(fc_df["timestamp"])
            combined = pd.concat([df.tail(10)[["timestamp", "aqi"]].assign(Type="History"), fc_df.assign(Type="Forecast")])
            fig = px.line(combined, x='timestamp', y='aqi', color='Type', line_dash='Type')
            st.plotly_chart(fig, use_container_width=True)

def render_reports():
    st.markdown("---")
    st.subheader("🔍 Model Explainability & EDA")
    
    # Hardcoded EDA Report Data
    eda_dataset_shape = (522, 37)
    
    class_distribution = pd.DataFrame({
        "AQI Class": [0, 1, 2, 3],
        "Label": ["Good", "Fair", "Moderate", "Poor"],
        "Count": [43, 181, 175, 123]
    })
    
    feature_summary = pd.DataFrame({
        "Statistic": ["count", "mean", "min", "25%", "50%", "75%", "max", "std"],
        "pm25": [522, 57.08, 3.95, 25.82, 50.89, 72.84, 260.59, 40.61],
        "pm10": [522, 110.66, 11.46, 59.72, 101.36, 143.17, 406.36, 67.60],
        "temperature": [522, 22.61, 10.97, 17.93, 22.57, 26.93, 36.99, 5.53],
        "humidity": [522, 42.62, 6.0, 24.0, 38.0, 58.0, 94.0, 22.53],
        "pressure": [522, 1015.92, 1007.0, 1014.0, 1016.0, 1018.0, 1024.0, 3.32],
        "aqi": [522, 3.72, 2.0, 3.0, 4.0, 4.0, 5.0, 0.92]
    })
    
    top_correlations = pd.DataFrame({
        "Feature": [
            "aqi (target)",
            "aqi_rolling_mean_3",
            "aqi_lag_1",
            "pm10",
            "aqi_rolling_mean_6",
            "pm25",
            "co",
            "aqi_lag_2",
            "aqi_lag_3",
            "so2"
        ],
        "Correlation with AQI": [1.0000, 0.9321, 0.8753, 0.8519, 0.8260, 0.8224, 0.7634, 0.7603, 0.6459, 0.6440]
    })
    
    model_opt = st.selectbox("Select Model", ["RandomForest", "GradientBoosting", "LogisticRegression"])
    eda_data, shap_df = get_reports(model_opt)
    
    t1, t2 = st.tabs(["📊 EDA Report", "🤖 SHAP Importance"])
    with t1:
        st.info(f"**Dataset Shape:** {eda_dataset_shape[0]} rows × {eda_dataset_shape[1]} columns", icon="📈")
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("📊 Class Distribution")
            st.dataframe(class_distribution, hide_index=True, use_container_width=True)
            st.caption("Balanced dataset across AQI classes with majority in Fair & Moderate categories")
        
        with c2:
            st.subheader("🔗 Top Correlated Features")
            st.dataframe(top_correlations, hide_index=True, use_container_width=True)
            st.caption("Lagged features & rolling means are strongest predictors of AQI")
        
        st.divider()
        st.subheader("📋 Feature Statistics Summary")
        st.dataframe(feature_summary, hide_index=True, use_container_width=True)
        st.caption("Statistical overview of key environmental features and AQI target variable")
        
        st.divider()
        st.subheader("🔍 Key EDA Insights")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Samples", "522")
        col2.metric("Features Used", "37")
        col3.metric("Target Classes", "4")
        
        insights = """
        - **Data Quality:** No missing values in target variable; complete feature set
        - **Class Imbalance:** Moderate imbalance with Good class (43) underrepresented
        - **Feature Importance:** PM2.5, PM10, and CO are strong AQI indicators
        - **Temporal Features:** Hour, day_of_week correlate with AQI patterns
        - **Lagged Features:** Previous AQI values (lag 1-24) have 0.64-0.88 correlation
        - **Rolling Statistics:** 3-hour and 6-hour rolling means excellent predictors (>0.93)
        """
        st.markdown(insights)

    with t2:
        if shap_df.empty:
            st.info(f"No SHAP summary found for {model_opt}.")
        else:
            shap_df = shap_df.sort_values("Importance", ascending=False)
            c1, c2 = st.columns([1, 2])
            with c1: st.dataframe(shap_df, hide_index=True)
            with c2:
                fig = px.bar(shap_df.head(10), x="Importance", y="Feature", orientation='h', title="Top 10 Features")
                fig.update_layout(yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig, use_container_width=True)

# ---------------- MAIN ---------------- #
def main():
    model, scaler, selector, feat_cols, min_class = load_artifacts()
    df = fetch_data()
    render_overview(df)
    if not df.empty:
        render_historical(df)
        render_forecast(df, model, scaler, selector, feat_cols, min_class)
        render_reports()

if __name__ == "__main__":
    main()
