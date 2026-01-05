import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
import warnings
import time
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters

warnings.filterwarnings("ignore")

# =====================================================
# MILESTONE 1, 2, & 3 CLASSES
# =====================================================

# =====================================================
# MILESTONE 1 : DATA UPLOADING AND PREPROCESSING
# =====================================================

class FitnessDataUploader:
    REQUIRED_COLUMNS = {
        "heart_rate": ["timestamp", "heart_rate"],
        "sleep": ["timestamp", "duration_minutes"],
        "steps": ["timestamp", "step_count"]
    }

    def upload(self):
        st.markdown("### 📁 Data Ingestion")
        files = st.file_uploader(
            "Upload CSV / JSON files",
            type=["csv", "json"],
            accept_multiple_files=True
        )

        data = {}
        if files:
            for file in files:
                df = self._read_file(file)
                data_type = self._detect_type(file.name)

                if data_type in self.REQUIRED_COLUMNS and self._validate(df, data_type):
                    data[data_type] = df
                    st.toast(f"Loaded {data_type}", icon="✅")
                else:
                    st.error(f"❌ Invalid structure in {file.name}")
        return data

    def _read_file(self, file):
        if file.name.endswith(".csv"):
            return pd.read_csv(file)
        else:
            return pd.DataFrame(json.load(file))

    def _detect_type(self, filename):
        name = filename.lower()
        if "heart" in name: return "heart_rate"
        if "sleep" in name: return "sleep"
        if "step" in name: return "steps"
        return "unknown"

    def _validate(self, df, data_type):
        df.columns = df.columns.str.lower()
        for col in self.REQUIRED_COLUMNS[data_type]:
            if col not in df.columns:
                return False
        return True
    
# =====================================================
# MILESTONE 2 : FEATURE EXTRACTION AND MODELLING
# =====================================================

class FitnessPreprocessor:
    def clean_and_align(self, df, data_type, freq):
        df = df.copy()
        df.columns = df.columns.str.lower()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df = df.set_index("timestamp").sort_index()
        numeric_cols = df.select_dtypes(include=np.number)
        if data_type == "heart_rate":
            df_resampled = numeric_cols.resample(freq).mean()
        else:
            df_resampled = numeric_cols.resample(freq).sum()
        df_resampled = df_resampled.interpolate().ffill().bfill()
        df_resampled.reset_index(inplace=True)
        return df_resampled

class FeatureForecaster:
    def extract_features(self, df):
        df = df.copy()
        df["id"] = 1
        numeric_cols = df.select_dtypes(include=np.number).columns
        features = {}
        for col in numeric_cols:
            if col == "id": continue
            temp = df[["id", "timestamp", col]].rename(columns={"timestamp": "time", col: "value"})
            temp["value"] = temp["value"].ffill().bfill()
            extracted = extract_features(
                temp, column_id="id", column_sort="time",
                default_fc_parameters=EfficientFCParameters(),
                disable_progressbar=True
            )
            features[col] = extracted
        return features

    def forecast(self, df):
        forecast_results = {}
        numeric_cols = df.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            last_mean = df[col].tail(5).mean()
            forecast_results[col] = [round(last_mean, 2)] * 5
        return forecast_results

# =====================================================
# MILESTONE 3 : ANOMALY DETECTION AND VISUALIZATION
# =====================================================

class AnomalyDetector:
    def detect(self, df):
        anomalies = {}
        numeric_cols = df.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            mean = df[col].mean()
            std = df[col].std()
            anomaly_points = df[(df[col] > mean + 2*std) | (df[col] < mean - 2*std)]
            anomalies[col] = anomaly_points
        return anomalies

# =====================================================
# MILESTONE 4 : DASHBOARD STYLING & CORE
# =====================================================

def apply_custom_style():
    st.markdown("""
        <style>
        .main { background-color: #f8f9fa; }
        
        /* Metric Box Fix for High Visibility */
        [data-testid="stMetric"] {
            background-color: #ffffff !important;
            padding: 20px !important;
            border-radius: 12px !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
            border: 1px solid #eeeeee !important;
        }
        [data-testid="stMetricLabel"] p {
            color: #333333 !important; 
            font-weight: 700 !important;
            font-size: 1rem !important;
        }
        [data-testid="stMetricValue"] div {
            color: #000000 !important;
            font-size: 2rem !important;
        }
        
        /* Professional Sidebar Buttons */
        .stButton>button { 
            width: 100%; 
            border-radius: 8px; 
            background-color: #0047AB; 
            color: white !important;
            font-weight: 600;
        }
        </style>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="FitPulse Health", layout="wide", page_icon="💓")
    apply_custom_style()

    # Required Heading
    st.title("💓 FitPulse Health: Anomaly Detection Using Fitness Devices")

    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = {}
    
    with st.sidebar:
        st.header("Control Panel")
        nav = st.radio("Step Selection", ["1. Upload & Process", "2. Interactive Dashboard", "3. Technical Logs"])
        st.markdown("---")
        freq = st.selectbox("Resample Frequency", ["1min", "5min", "15min", "1H"])
        run_btn = st.button("🚀 Run Full Pipeline")

    uploader = FitnessDataUploader()
    preprocessor = FitnessPreprocessor()
    forecaster = FeatureForecaster()
    detector = AnomalyDetector()

    # --- STEP 1: UPLOAD ---
    if nav == "1. Upload & Process":
        st.subheader("Data Upload Center")
        raw_data = uploader.upload()
        
        if run_btn:
            if not raw_data:
                st.warning("Please upload files first.")
            else:
                with st.spinner("Processing through Milestones 1-3..."):
                    temp_data = {}
                    for dtype, df in raw_data.items():
                        processed = preprocessor.clean_and_align(df, dtype, freq)
                        feats = forecaster.extract_features(processed)
                        fcasts = forecaster.forecast(processed)
                        anoms = detector.detect(processed)
                        
                        temp_data[dtype] = {
                            "df": processed,
                            "features": feats,
                            "forecast": fcasts,
                            "anomalies": anoms
                        }
                    st.session_state.processed_data = temp_data
                    st.success("Pipeline executed successfully!")

    # --- STEP 2: DASHBOARD ---
    elif nav == "2. Interactive Dashboard":
        if not st.session_state.processed_data:
            st.info("💡 Please process data in Step 1 first.")
        else:
            data_keys = list(st.session_state.processed_data.keys())
            selected_type = st.selectbox("Explore Activity Metric", data_keys)
            
            data_bundle = st.session_state.processed_data[selected_type]
            df = data_bundle["df"]
            metric_col = df.select_dtypes(include=np.number).columns[0]

            # High-Visibility Cards
            m1, m2, m3 = st.columns(3)
            current_val = df[metric_col].iloc[-1]
            avg_val = df[metric_col].mean()
            anom_count = len(data_bundle["anomalies"][metric_col])

            m1.metric(f"Latest {selected_type.title()}", f"{current_val:.1f}")
            m2.metric("Mean Average", f"{avg_val:.1f}")
            m3.metric("Anomalies Found", anom_count, delta=f"{anom_count} issues", delta_color="inverse")

            t1, t2 = st.tabs(["📈 Dynamic Visuals", "🔍 Anomaly Investigation"])

            with t1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df["timestamp"], y=df[metric_col], name="Historical", line=dict(color='#0047AB')))
                
                # Plot Forecast
                last_time = df["timestamp"].iloc[-1]
                future_times = pd.date_range(start=last_time, periods=6, freq=freq)[1:]
                fig.add_trace(go.Scatter(x=future_times, y=data_bundle["forecast"][metric_col], name="Forecast", line=dict(dash='dash', color='orange')))
                
                fig.update_layout(title=f"{selected_type.replace('_',' ').title()} Analysis", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)

            with t2:
                st.dataframe(data_bundle["anomalies"][metric_col], use_container_width=True)

    # --- STEP 3: TECHNICAL LOGS (IMPROVED FORMAT) ---
    elif nav == "3. Technical Logs":
        st.subheader("📑 Data Intelligence & Feature Sets")
        if not st.session_state.processed_data:
            st.info("Awaiting pipeline execution.")
        else:
            log_selection = st.selectbox("Select Log Stream", list(st.session_state.processed_data.keys()))
            content = st.session_state.processed_data[log_selection]
            
            col_meta, col_feat = st.columns([1, 2])
            with col_meta:
                st.markdown("#### 🛠️ Metadata Summary")
                st.json({
                    "Stream": log_selection,
                    "Total Rows": len(content["df"]),
                    "Frequency": freq,
                    "Memory": f"{content['df'].memory_usage().sum() / 1024:.2f} KB"
                })

            with col_feat:
                st.markdown("#### 🧬 Feature Set (Transposed)")
                raw_features = list(content["features"].values())[0]
                transposed_features = raw_features.T.reset_index()
                transposed_features.columns = ["Feature Name", "Value"]
                
                search = st.text_input("🔍 Search Features (e.g. 'mean', 'abs')", "")
                if search:
                    transposed_features = transposed_features[transposed_features["Feature Name"].str.contains(search, case=False)]
                
                st.dataframe(transposed_features, use_container_width=True, height=400, hide_index=True)

if __name__ == "__main__":
    main()
