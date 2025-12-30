# FitPulse – Health Anomaly Detection from Fitness Devices

FitPulse is an industry-oriented fitness data analytics pipeline designed to process time-series data from wearable fitness devices.  
The system performs data ingestion, preprocessing, feature extraction, trend forecasting, and anomaly detection using multiple techniques.

The project follows a modular, production-style architecture and is implemented as an interactive **Streamlit application** for real-time data analysis and visualization.

---

## Key Features

- CSV and JSON file ingestion
- Data cleaning, validation, and time alignment
- Time-series feature extraction using **TSFresh**
- Trend forecasting using rolling statistics
- Multi-method anomaly detection
- Interactive visualizations and performance metrics

---

## Milestones Covered

### Milestone 1 – Data Preprocessing
- File upload and schema validation  
- Timestamp normalization and alignment  
- Data cleaning and resampling  
- Interactive data preview  

### Milestone 2 – Feature Extraction & Forecasting
- Statistical feature extraction using TSFresh  
- Rolling-mean based trend forecasting  
- Metric summaries and trend visualizations  

### Milestone 3 – Anomaly Detection
- Threshold-based anomaly detection  
- Statistical anomaly detection using Z-score  
- Trend deviation-based detection  
- Reduction of false positives through multi-method validation  

---

## Why Multiple Detection Methods?

Using multiple anomaly detection techniques improves system reliability by:
- Capturing different types of anomaly patterns  
- Reducing false positives and false negatives  
- Aligning with real-world health monitoring and alerting systems  

---

## Supported Data Formats

| Data Type   | Required Columns |
|------------|------------------|
| Heart Rate | timestamp, heart_rate |
| Steps      | timestamp, step_count |
| Sleep      | timestamp, sleep_stage, duration_minutes |

---

## Tech Stack

- Python  
- Streamlit  
- Pandas, NumPy  
- Plotly  
- TSFresh  

---

## How to Run the Application

```bash
pip install -r requirements.txt
streamlit run health_anomaly_detection_pipeline.py
