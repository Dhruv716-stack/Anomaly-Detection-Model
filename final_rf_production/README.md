# Anomaly Detection Production Model — Documentation

## Overview
This folder contains a fully production-ready anomaly detection pipeline based on a Random Forest model. It includes all code, model artifacts, preprocessing pipeline, imputation logic, datasets, and evaluation results needed for robust, reproducible, and portable deployment.

---

## Requirements
- **Python 3.8+** (recommended)
- **pip** (for installing dependencies)

### Python Packages
All required packages are listed in `requirements.txt`:
- pandas
- numpy
- scikit-learn

Install with:
```bash
pip install -r requirements.txt
```

---

## Environment Setup (Recommended)
1. **Create a virtual environment:**
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Input Requirements
The model expects input data as a Python dictionary (or JSON) with the following fields:
- user_id (str)
- session_id (str)
- device_type ("PC" or "Mobile")
- click_events (int)
- scroll_events (int)
- touch_events (int)
- keyboard_events (int)
- device_motion (float)
- time_on_page (int)
- screen_size (str, e.g., "360x640")
- browser_info (str, e.g., "Chrome")
- language (str, e.g., "english")
- timezone_offset (int)
- device_orientation (str)
- geolocation_city (str)
- transaction_amount (float)
- transaction_date (str, e.g., "2025-04-18 02:00:00")
- mouse_movement (int)

**Missing values are handled automatically** using mean/mode imputation (see below).

---

## How to Use the Model and Pipeline

### 1. **Single Prediction (from Python)**
```python
import pickle
import pandas as pd
import numpy as np

# Load artifacts
with open('rf_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)
with open('feature_cols.pkl', 'rb') as f:
    feature_cols = pickle.load(f)
with open('imputation_values.pkl', 'rb') as f:
    imputation_values = pickle.load(f)

def preprocess_input(input_dict):
    df = pd.DataFrame([input_dict])
    # Impute missing values
    for col, val in imputation_values.items():
        if col not in df.columns or pd.isnull(df.at[0, col]):
            df[col] = val
    # ... (add flag_obvious_anomalies, encode categoricals, scale, as in predict.py)
    # See predict.py for full logic
    return X_scaled, df

# Use the predict() function from predict.py for full pipeline
```

### 2. **Command Line Prediction**
Prepare your input as a JSON file (see `test_input_missing.json` for an example):
```bash
python predict.py input.json
```

### 3. **Integration in Other Backends**
- Load all artifacts once at server startup.
- For each request, preprocess input as above and call the model for prediction.
- Return the output (predicted_label, anomaly_score, risk_level, risk_reason) as your API response.

---

## Output
For each input, the model returns:
- `predicted_label`: 0 (normal) or 1 (anomaly)
- `anomaly_score`: probability of anomaly
- `risk_level`: Low, Medium, or High
- `risk_reason`: brief explanation for Medium/High risk

---

## Retraining or Updating the Model
To retrain or update the model with new data:
```bash
python export_rf_production_model.py
```
This will regenerate all artifacts using the latest training data and logic.

---

## Datasets and Results
- All datasets used for training and testing are included for reproducibility.
- The best evaluation results are in `realistic_model_results.csv`.

---

## Imputation Logic
- **Numerical features:** Imputed with mean (or 0 for counts)
- **Categorical features:** Imputed with mode (most frequent value)
- Imputation values are saved in `imputation_values.pkl` and used automatically.

---

## Notes for Production Use
- The folder is fully self-contained and portable.
- All preprocessing, imputation, and risk logic is included in `predict.py`.
- You can integrate the logic into any Python backend (Flask, FastAPI, Django, etc.)
- For batch scoring, simply loop over your data and call the prediction pipeline.

---

## Support
For questions, integration help, or further customization, see the main project README or contact the author. 