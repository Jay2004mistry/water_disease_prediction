# 💧 AquaPredict — Water Pollution Disease Risk Estimator

> A supervised machine learning project that predicts disease burden from water quality and regional data, served via a Flask REST API with a modern dark-themed frontend.

---

## 🔍 Project Overview

This project uses **regression-based machine learning** to predict how many people per 100,000 will be affected by waterborne diseases based on water quality measurements and socio-economic indicators.

**Predicted Targets:**
- Diarrheal Cases (per 100,000 people)
- Cholera Cases (per 100,000 people)
- Typhoid Cases (per 100,000 people)
- Infant Mortality Rate (per 1,000 live births)

---

## 🗂️ Project Structure

```
aquapredict/
│
├── app.py                                  # Flask REST API
├── index.html                              # Frontend UI
├── Linear_reg_model_for_disease_water.pkl  # Saved best model
├── scaler.pkl                              # Saved StandardScaler
├── water_disease_ped.ipynb                 # ML training notebook
└── water_pollution_disease.csv             # Dataset
```

---

## 📊 Dataset

| Feature | Description |
|---|---|
| Country | 10 countries (Bangladesh, Brazil, China, Ethiopia, India, Indonesia, Mexico, Nigeria, Pakistan, USA) |
| Region | Central, East, North, South, West |
| Water Source Type | Lake, Pond, River, Spring, Tap, Well |
| Water Treatment Method | Boiling, Chlorination, Filtration, None |
| Contaminant Level (ppm) | Chemical contamination level |
| pH Level | Acidity of water |
| Turbidity (NTU) | Water clarity |
| Dissolved Oxygen (mg/L) | Oxygen content |
| Nitrate Level (mg/L) | Nitrate concentration |
| Lead Concentration (µg/L) | Lead pollution level |
| Bacteria Count (CFU/mL) | Microbial contamination |
| Access to Clean Water (%) | Population with clean water access |
| Sanitation Coverage (%) | Population with sanitation access |
| GDP per Capita (USD) | Economic indicator |
| Healthcare Access Index | Healthcare availability (0-100) |
| Urbanization Rate (%) | Urban population percentage |
| Rainfall (mm/year) | Annual rainfall |
| Temperature (°C) | Average temperature |
| Population Density (per km²) | People per square kilometer |

**Total: 3,000 rows × 24 columns**

---

## 🤖 ML Pipeline

```
Load Data → Preprocess → Encode → Split (80/20) → Scale → Train → Evaluate → Save
```

### Preprocessing
- Filled missing values in `Water Treatment Method` with mode
- Label encoded: Country, Region, Water Source Type, Water Treatment Method
- Applied `StandardScaler` on feature matrix

### Models Trained & Compared

| Model | Needs Scaling | Multi-Output |
|---|---|---|
| Linear Regression | ✅ Yes | ✅ Native |
| Decision Tree Regressor | ❌ No | ✅ Native |
| Random Forest Regressor | ❌ No | ✅ Native |
| SVR | ✅ Yes | ⚠️ Wrapped |
| KNN Regressor | ✅ Yes | ⚠️ Wrapped |

### Evaluation Metrics

| Metric | Meaning | Goal |
|---|---|---|
| R² Score | How well model fits overall | Closer to 1.0 |
| MAE | Average error in real units | Lower is better |
| RMSE | Penalizes large errors more | Lower is better |

### Result — Best Model

**Linear Regression** was selected as the best model based on highest average R² and lowest MAE across all 4 targets.

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install flask flask-cors scikit-learn numpy pandas
```

### 2. Start the Flask API

```bash
python app.py
```

API runs at: `http://127.0.0.1:5000`

### 3. Open the frontend

Double click `index.html` in your browser.

---

## 🔌 API Reference

### `GET /`
Health check — confirms API is running.

**Response:**
```json
{ "message": "API is running successfully!" }
```

---

### `POST /predict`
Accepts water quality features and returns disease predictions.

**Request Body:**
```json
{
  "features": [4, 2, 2024, 0, 5.2, 6.8, 12.5, 7.1,
               4.3, 18.0, 4500, 2, 45.0, 1800, 32.0,
               38.0, 29.0, 820, 28.5, 310]
}
```

> Feature order must match training column order: Country, Region, Year, Water Source Type, Contaminant, pH, Turbidity, Dissolved O₂, Nitrate, Lead, Bacteria, Treatment, Access Water, GDP, Healthcare, Urbanization, Sanitation, Rainfall, Temperature, Population Density

**Response:**
```json
{
  "Diarrheal": 265.56,
  "Cholera": 25.07,
  "Typhoid": 48.41,
  "Infant_Mortality": 44.89
}
```

---

## 🖥️ Frontend

- Dark themed UI built with pure HTML, CSS, JavaScript
- All categorical fields are **dropdowns with real names** (no encoded numbers shown to user)
- Automatic encoding happens in the browser before sending to API
- Shows loading spinner while waiting for prediction
- Displays 4 result cards with color-coded values

**Label Encoding Reference (for dropdowns):**

| Country | Code | Region | Code |
|---|---|---|---|
| Bangladesh | 0 | Central | 0 |
| Brazil | 1 | East | 1 |
| China | 2 | North | 2 |
| Ethiopia | 3 | South | 3 |
| India | 4 | West | 4 |
| Indonesia | 5 | | |
| Mexico | 6 | **Water Source** | **Code** |
| Nigeria | 7 | Lake | 0 |
| Pakistan | 8 | Pond | 1 |
| USA | 9 | River | 2 |
| | | Spring | 3 |
| **Treatment** | **Code** | Tap | 4 |
| Boiling | 0 | Well | 5 |
| Chlorination | 1 | | |
| Filtration | 2 | | |
| None | 3 | | |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3 |
| ML Library | scikit-learn |
| API | Flask + Flask-CORS |
| Data Processing | Pandas, NumPy |
| Model Persistence | Pickle |
| Frontend | HTML5, CSS3, Vanilla JS |
| Fonts | Syne, DM Sans (Google Fonts) |

---



## ⚠️ Notes

- This project uses a **synthetic dataset** — R² scores are near zero because the data was randomly generated without real-world relationships between features and targets.
- On a real water quality dataset, the same pipeline would produce meaningful predictions.
- Negative predictions are clipped to 0 using `np.clip()` in the API since disease cases cannot be negative.

---
for live api use Rendor.com
and for frontend use git hub

Live:-
https://jay2004mistry.github.io/water_disease_prediction/

