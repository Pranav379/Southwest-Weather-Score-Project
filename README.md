# Southwest-Weather-Score-Project

A data science project focused on **quantifying weather-driven operational risk for Southwest Airlines**.  
Using 10 years of historical flight and meteorological data (2015–2025), this project develops a **0–100 Weather Impact Score** and machine learning models to predict flight delays and disruption probability.

---

## 📍 Project Purpose

Flight operations are highly sensitive to weather patterns. This project provides Southwest with:

- A standardized score indicating **weather impact severity**
- Early prediction of **departure delays and cancellations**
- Improved **operational planning and crew positioning**
- Reduced **customer disruption and operational cost**

---

## 🧾 Data Sources

| Provider | Details |
|---------|---------|
| Bureau of Transportation Statistics (BTS) | Historical flight performance data |
| Meteostat Python API | Daily/Hourly meteorological metrics |

Dataset scope limited to **Southwest-serving airports** to ensure regional relevance.

---

## 🔧 Tools & Technologies

| Category | Stack |
|---------|-------|
| Languages | Python |
| Data | Pandas, NumPy |
| Modeling | Scikit-Learn, LightGBM, Optuna, Keras, SMOTE |
| Visualization | Seaborn, Matplotlib |
| Environment | Google Colab |

---

## 🧹 Data Processing & Feature Engineering

Key preprocessing highlights:

- **Unit conversion** → °C → °F, km/h → mph, mm → in, hPa → inHg  
- **Cyclical transformations** → sine/cosine encodings for CRSDepTime & CRSArrTime  
- **Missing value mitigation** → columns dropped only when >95% null  
- **New disruption indicators** → `TotalDisruptionMinutes`, `IsSevereDisruption`, etc.
- **Rolling metrics** → 7-day averages for route, origin, and destination delay patterns
- **Operational congestion metrics** → `NumDepartures`, `CongestionRatio`
- **Target normalization strategies** → Log, Box-Cox, and Yeo-Johnson

---

## 🤖 Machine Learning Models

### 1. LightGBM — Regression (Primary Model)
Objective: Predict **log-transformed DepDelayMinutes**

| Metric (Test Set) | Score |
|------------------|-------|
| MAE | **12.20 minutes** |
| RMSE | 30.95 minutes |
| R² (original scale) | **-0.0433** |
| R² (log scale) | 0.1933 |

Top predictors: wind speed, average temperature, route distance

---

### 2. Random Forest — Classification
Objective: Predict **DepDel15 (delay ≥ 15 min)**  
- Class imbalance addressed using `class_weight="balanced"`  
- Hyperparameter optimization via Optuna (ROC-AUC)

| Metric | Score |
|--------|-------|
| Accuracy | 0.600 |
| Recall | 0.747 |
| Precision | 0.319 |
| ROC-AUC | 0.707 |

---

### 3. Neural Network (Keras) — Regression
- Three dense hidden layers: **128 → 64 → 32 (ReLU)**
- 200 epochs, batch size 64 + EarlyStopping

Model file stored at: `weather_prediction_model.h5`

---

### 4. Logistic Regression — Classification
Objective: Predict **weatherScore > 0** (weather-related disruption)

| Metric | Score |
|--------|-------|
| Accuracy | **0.8710** |
| Precision (disruption class) | 0.49 |
| Recall (disruption class) | 0.54 |

Data pipeline includes: MinMaxScaler → SMOTE → stratified sampling

---

## 🔁 Reproducibility

To fully replicate model outputs:

```bash
pip install -r requirements.txt
