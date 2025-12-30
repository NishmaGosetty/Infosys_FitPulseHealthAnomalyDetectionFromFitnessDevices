import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
import warnings
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters

warnings.filterwarnings("ignore")

# =====================================================
# MILESTONE 1 : DATA UPLOAD + VALIDATION + PREPROCESSING
# =====================================================

class FitnessDataUploader:
    REQUIRED_COLUMNS = {
        "heart_rate": ["timestamp", "heart_rate"],
        "sleep": ["timestamp", "duration_minutes"],
        "steps": ["timestamp", "step_count"]
    }

    def upload(self):
        st.header("📁 Upload Fitness Data")
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
                    st.success(f"✅ {data_type} data loaded ({len(df)} rows)")
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
        if "heart" in name:
            return "heart_rate"
        if "sleep" in name:
            return "sleep"
        if "step" in name:
            return "steps"
        return "unknown"

    def _validate(self, df, data_type):
        df.columns = df.columns.str.lower()
        for col in self.REQUIRED_COLUMNS[data_type]:
            if col not in df.columns:
                st.warning(f"Missing column: {col}")
                return False
        return True


class FitnessPreprocessor:
    def clean_and_align(self, df, data_type, freq):
        df = df.copy()
        df.columns = df.columns.str.lower()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df = df.set_index("timestamp").sort_index()

        # keep ONLY numeric columns for resampling
        numeric_cols = df.select_dtypes(include=np.number)

        if data_type == "heart_rate":
            df_resampled = numeric_cols.resample(freq).mean()
        else:
            df_resampled = numeric_cols.resample(freq).sum()

        df_resampled = df_resampled.interpolate().ffill().bfill()
        df_resampled.reset_index(inplace=True)
        return df_resampled


# =====================================================
# MILESTONE 2 : FEATURE EXTRACTION + FORECASTING
# =====================================================

class FeatureForecaster:
    def extract_features(self, df):
        df = df.copy()
        df["id"] = 1

        numeric_cols = df.select_dtypes(include=np.number).columns
        features = {}

        for col in numeric_cols:
            if col == "id":
                continue

            temp = df[["id", "timestamp", col]].rename(
                columns={"timestamp": "time", col: "value"}
            )

            temp["value"] = temp["value"].ffill().bfill()

            extracted = extract_features(
                temp,
                column_id="id",
                column_sort="time",
                default_fc_parameters=EfficientFCParameters(),
                disable_progressbar=True
            )

            features[col] = extracted
            st.success(f"✅ Features extracted for {col}")

        return features

    def forecast(self, df):
        forecast_results = {}
        numeric_cols = df.select_dtypes(include=np.number).columns

        for col in numeric_cols:
            last_mean = df[col].tail(5).mean()
            forecast_results[col] = [round(last_mean, 2)] * 5
            st.info(f"Forecast for {col}: {forecast_results[col]}")

        return forecast_results


# =====================================================
# MILESTONE 3 : ANOMALY DETECTION
# =====================================================

class AnomalyDetector:
    def detect(self, df):
        anomalies = {}
        numeric_cols = df.select_dtypes(include=np.number).columns

        for col in numeric_cols:
            mean = df[col].mean()
            std = df[col].std()
            threshold_high = mean + 2 * std
            threshold_low = mean - 2 * std

            anomaly_points = df[
                (df[col] > threshold_high) | (df[col] < threshold_low)
            ]

            anomalies[col] = anomaly_points
            st.warning(f"⚠️ {len(anomaly_points)} anomalies detected in {col}")

        return anomalies


# =====================================================
# STREAMLIT APP
# =====================================================

def main():
    st.set_page_config(page_title="Fitness Analytics Pipeline", layout="wide")
    st.title("🏋️ Fitness Data Analytics Pipeline")
    st.markdown("### Milestone 1 + 2 + 3 Integrated")

    uploader = FitnessDataUploader()
    preprocessor = FitnessPreprocessor()
    forecaster = FeatureForecaster()
    detector = AnomalyDetector()

    raw_data = uploader.upload()
    freq = st.sidebar.selectbox("Resample Frequency", ["1min", "5min", "15min"])

    if st.button("🚀 Run Pipeline"):
        processed_data = {}

        for dtype, df in raw_data.items():
            st.subheader(f"📊 {dtype.upper()} PROCESSING")
            processed = preprocessor.clean_and_align(df, dtype, freq)
            processed_data[dtype] = processed

            st.dataframe(processed.head())

            # Plot
            metric = processed.select_dtypes(include=np.number).columns[0]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=processed["timestamp"],
                y=processed[metric],
                mode="lines+markers",
                name=metric
            ))
            st.plotly_chart(fig, use_container_width=True)

            # Milestone 2
            st.subheader("🔹 Feature Extraction & Forecast")
            forecaster.extract_features(processed)
            forecaster.forecast(processed)

            # Milestone 3
            st.subheader("🔹 Anomaly Detection")
            detector.detect(processed)

        st.success("✅ All Milestones Executed Successfully!")


if __name__ == "__main__":
    main()
