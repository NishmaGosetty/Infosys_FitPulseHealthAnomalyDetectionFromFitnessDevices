import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import json
import warnings
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters

warnings.filterwarnings('ignore')

# ------------------ COMPONENT A: File Upload ------------------ #
class FitnessDataUploader:
    def __init__(self):
        self.supported_formats = ['.csv', '.json']
        self.required_columns = {
            'heart_rate': ['timestamp', 'heart_rate'],
            'sleep': ['timestamp', 'sleep_stage', 'duration_minutes'],
            'steps': ['timestamp', 'step_count']
        }

    def upload_files(self):
        st.header("📁 Upload Fitness Tracker Data")
        uploaded_files = st.file_uploader(
            "Choose CSV or JSON files",
            type=['csv', 'json'],
            accept_multiple_files=True
        )
        data_dict = {}
        if uploaded_files:
            for f in uploaded_files:
                try:
                    data_type = self._detect_data_type(f.name)
                    if f.name.endswith('.csv'):
                        df = pd.read_csv(f)
                    else:
                        df = self._load_json(f)
                    if self._validate(df, data_type):
                        data_dict[data_type] = df
                        st.success(f"✅ {data_type} data loaded ({len(df)} records)")
                    else:
                        st.error(f"❌ Invalid structure in {f.name}")
                except Exception as e:
                    st.error(f"❌ Error loading {f.name}: {e}")
        return data_dict

    def _detect_data_type(self, filename):
        fname = filename.lower()
        if 'heart' in fname or 'hr' in fname:
            return 'heart_rate'
        elif 'sleep' in fname:
            return 'sleep'
        elif 'step' in fname or 'activity' in fname:
            return 'steps'
        else:
            return 'unknown'

    def _load_json(self, f):
        data = json.load(f)
        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict) and 'data' in data:
            return pd.DataFrame(data['data'])
        else:
            return pd.DataFrame([data])

    def _validate(self, df, data_type):
        if data_type not in self.required_columns:
            return False
        df_cols = [c.lower() for c in df.columns]
        for col in self.required_columns[data_type]:
            if col.lower() not in df_cols:
                st.warning(f"Missing column: {col}")
                return False
        return True

# ------------------ COMPONENT B: Validation ------------------ #
class FitnessDataValidator:
    def __init__(self):
        self.rules = {
            'heart_rate': (30, 220),
            'step_count': (0, 100000),
            'duration_minutes': (0, 1440)
        }

    def clean_data(self, df, data_type):
        df = df.rename(columns=lambda x: x.lower().replace(' ', '_'))
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        for col, (low, high) in self.rules.items():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].clip(lower=low, upper=high)
        df = df.dropna(subset=['timestamp'])
        return df

# ------------------ COMPONENT C: Time Alignment ------------------ #
class TimeAligner:
    freq_map = {'1min':'1T', '5min':'5T', '15min':'15T', '30min':'30T', '1hour':'1H'}

    def align(self, df, data_type, target_freq='1min'):
        if 'timestamp' not in df.columns:
            return df
        df = df.set_index('timestamp').sort_index()
        freq_str = self.freq_map.get(target_freq, '1T')
        if data_type == 'heart_rate':
            df_resampled = df.resample(freq_str).mean()
        elif data_type == 'steps' or data_type=='duration_minutes':
            df_resampled = df.resample(freq_str).sum()
        elif data_type == 'sleep':
            df_resampled = df.resample(freq_str).agg(lambda x: x.mode().iloc[0] if len(x.mode())>0 else np.nan)
        df_resampled = df_resampled.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
        df_resampled = df_resampled.reset_index()
        return df_resampled

# ------------------ MILESTONE 2: Feature Extraction & Forecast ------------------ #
class FeatureForecaster:
    def extract_features(self, df, data_type):
        df_ff = df.copy()
        df_ff = df_ff.rename(columns=lambda x: x.lower().replace(' ', '_'))
        if 'timestamp' not in df_ff.columns:
            st.warning(f"No timestamp column for {data_type}")
            return None
        # Create ID column for TSFresh (required)
        df_ff['id'] = 1
        numeric_cols = df_ff.select_dtypes(include=np.number).columns.tolist()
        features_dict = {}
        for col in numeric_cols:
            if col in ['id']: 
                continue
            try:
                extracted = extract_features(
                    df_ff[['id','timestamp', col]].rename(columns={col:'value', 'timestamp':'time'}),
                    column_id='id', column_sort='time',
                    default_fc_parameters=EfficientFCParameters()
                )
                features_dict[col] = extracted
                st.success(f"✅ Features extracted for {col} in {data_type}")
            except Exception as e:
                st.error(f"❌ Error extracting features for {col}: {e}")
        return features_dict

    def forecast_data(self, df, data_type):
        # Placeholder simple forecast: next 5 values using rolling mean
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        forecast_dict = {}
        for col in numeric_cols:
            series = df[col].fillna(method='ffill')
            last_mean = series.rolling(5, min_periods=1).mean().iloc[-1]
            forecast = [last_mean]*5
            forecast_dict[col] = forecast
            st.info(f"Forecast (next 5 points) for {col} in {data_type}: {forecast}")
        return forecast_dict

# ------------------ COMPLETE PIPELINE ------------------ #
class FitnessPreprocessor:
    def __init__(self):
        self.uploader = FitnessDataUploader()
        self.validator = FitnessDataValidator()
        self.aligner = TimeAligner()
        self.processed_data = {}

    def run_pipeline(self, uploaded_files=None, target_freq='1min'):
        if uploaded_files is None:
            raw_data = self.uploader.upload_files()
        else:
            raw_data = uploaded_files

        if not raw_data:
            st.error("❌ No data uploaded!")
            return

        for dt, df in raw_data.items():
            cleaned = self.validator.clean_data(df, dt)
            aligned = self.aligner.align(cleaned, dt, target_freq)
            self.processed_data[dt] = aligned

        st.success("✅ Pipeline Completed!")
        return self.processed_data

    def preview_data(self):
        if not self.processed_data:
            st.warning("No processed data available!")
            return
        st.header("📊 Processed Data Preview")
        for dt, df in self.processed_data.items():
            st.subheader(f"{dt.title()} Data Sample")
            st.dataframe(df.head(20), use_container_width=True)
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if numeric_cols:
                col_left, col_right = st.columns(2)
                with col_left:
                    for num_col in numeric_cols:
                        st.metric(
                            label=f"{num_col.replace('_',' ').title()} Mean",
                            value=f"{df[num_col].mean():.1f}"
                        )
                with col_right:
                    for num_col in numeric_cols:
                        st.metric(
                            label=f"{num_col.replace('_',' ').title()} Max",
                            value=f"{df[num_col].max():.1f}"
                        )
            self.plot_data(df, numeric_cols, dt)

    def plot_data(self, df, numeric_cols, dt):
        if not numeric_cols:
            return
        col_select = st.selectbox(f"Select metric to plot for {dt}:", numeric_cols, key=dt)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df[col_select],
            mode='lines+markers', name=col_select
        ))
        st.plotly_chart(fig, use_container_width=True)

# ------------------ STREAMLIT APP ------------------ #
def main():
    st.set_page_config(page_title="Fitness Data Preprocessor", layout="wide", page_icon="🏋️‍♀️")
    st.title("🏋️‍♀️ Fitness Data Preprocessing & Forecasting Pipeline")
    st.markdown("**Milestone 2: Feature Extraction & Forecasting**")

    # Initialize
    preprocessor = FitnessPreprocessor()
    pipeline = FeatureForecaster()

    uploaded_files = preprocessor.uploader.upload_files()
    
    target_freq = st.sidebar.selectbox(
        "Target Frequency:",
        options=['1min', '5min', '15min', '30min', '1hour']
    )

    # Run pipeline
    if st.button("🚀 Run Pipeline"):
        processed = preprocessor.run_pipeline(uploaded_files, target_freq)
        if processed:
            preprocessor.preview_data()
            st.session_state['processed_data'] = processed

    # Extract Features & Forecast
    if 'processed_data' in st.session_state:
        processed_data = st.session_state['processed_data']
        if st.button("⚡ Extract Features & Forecast"):
            for dt, df in processed_data.items():
                if df is None or df.empty:
                    st.warning(f"No processed data available for {dt}")
                    continue
                st.subheader(f"🔹 {dt.title()} Feature Extraction & Forecast")
                pipeline.extract_features(df, dt)
                pipeline.forecast_data(df, dt)
            st.success("✅ Feature Extraction & Forecast Completed!")
    else:
        st.info("ℹ️ Run the pipeline first to process the data.")

if __name__ == "__main__":
    main()
 