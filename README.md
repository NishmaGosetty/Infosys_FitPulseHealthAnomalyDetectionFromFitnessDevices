# 🏃 FitPulse – Health Anomaly Detection System

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fitpulse-health.streamlit.app)

🚀 **Live Demo:** [https://fitpulse-health.streamlit.app](https://fitpulse-health.streamlit.app/)

## 🚀 Overview
FitPulse is an end-to-end machine learning pipeline for analyzing fitness device data and detecting health anomalies.

Built as an interactive Streamlit application, it enables users to upload time-series data, visualize trends, forecast behavior, and identify anomalies through a unified dashboard.

This project simulates a real-world health monitoring system used in wearable technology and preventive healthcare.

---

## 🧩 System Architecture
1. Data Ingestion (CSV/JSON)  
2. Data Preprocessing & Cleaning  
3. Feature Extraction (TSFresh)  
4. Trend Forecasting  
5. Multi-method Anomaly Detection  
6. Visualization via Streamlit Dashboard  

---

## 🧠 Key Features
- CSV and JSON file ingestion  
- Data cleaning, validation, and time alignment  
- Time-series feature extraction using **TSFresh**  
- Trend forecasting using rolling statistics  
- Multi-method anomaly detection  
- Interactive visualizations and performance metrics  
- Unified Streamlit dashboard with user interactivity  
- Accessible and consistent UI design for better user experience  

---

## 📌 Milestones Covered

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

### Milestone 4 – Interactive Dashboard & Visualization
- Development of a unified, interactive Streamlit dashboard  
- Real-time visualization of trends, forecasts, and detected anomalies  
- User-controlled filters and interactive components for data exploration  
- Consistent color scheme, layout, and typography for professional presentation  
- Accessibility-focused design ensuring clear text visibility and readable charts  
- Performance metrics and summary insights for quick decision-making  

---

## ❓ Why Multiple Detection Methods?
Using multiple anomaly detection techniques improves system reliability by:
- Capturing different types of anomaly patterns  
- Reducing false positives and false negatives  
- Aligning with real-world health monitoring and alerting systems  

---

## 📊 Supported Data Formats

| Data Type   | Required Columns |
|------------|------------------|
| Heart Rate | timestamp, heart_rate |
| Steps      | timestamp, step_count |
| Sleep      | timestamp, sleep_stage, duration_minutes |

---

## 🛠️ Tech Stack
- Python  
- Streamlit  
- Pandas, NumPy  
- Plotly  
- TSFresh  

---

## 📸 Demo
The following screenshots demonstrate the end-to-end workflow of the FitPulse system:

### 🔹 Dashboard & Data Ingestion
![Dashboard](ss1.png)

### 🔹 Historical Trends & Forecasting
![Forecasting](ss2.png)

### 🔹 Anomaly Detection Results
![Anomaly Detection](ss3.png)

### 🔹 Data Intelligence & Feature Insights
![Feature Insights](ss4.png)

---

## 🌍 Real-World Applications
- Wearable health monitoring systems  
- Early detection of abnormal heart rate patterns  
- Fitness analytics platforms  
- Preventive healthcare solutions  

---

## ⚙️ How to Run

1. Clone the repository

```
   git clone https://github.com/NishmaGosetty/fitpulse-health-anomaly-detection-ml
```

3. Navigate to the project folder
   
```
   cd fitpulse-health-anomaly-detection-ml
```  

5. Install dependencies

```
   pip install -r requirements.txt
```  

7. Run the application

```
   streamlit run health_anomaly_detection_pipeline.py
``` 

---

## 📌 Project Status
Completed core pipeline with dashboard. Future enhancements planned for real-time integration and deployment.

---

## 👩‍💻 Author
Nishma Gosetty
