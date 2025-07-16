import pickle
import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import LabelEncoder

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

def flag_obvious_anomalies(df):
    flag = np.zeros(len(df), dtype=int)
    for i, row in df.iterrows():
        # Slightly less sensitive: higher threshold and fewer hours
        if (row['transaction_amount'] > 10000) and (pd.to_datetime(row['transaction_date']).hour in [1,2,3]):
            flag[i] = 1
        elif (row['click_events'] >= 30) or (row['mouse_movement'] >= 1500):
            flag[i] = 1
        elif (row['device_type'] == 'Mobile' and row['touch_events'] >= 20):
            flag[i] = 1
        elif (row['device_type'] == 'Mobile' and row['device_motion'] >= 3.0):
            flag[i] = 1
        elif row['time_on_page'] in [10, 600]:
            flag[i] = 1
        elif (row['keyboard_events'] == 0 and row['transaction_amount'] > 5000):
            flag[i] = 1
    df['obvious_anomaly_flag'] = flag
    return df

def assign_risk_level(score):
    if score < 0.33:
        return 'Low'
    elif score < 0.66:
        return 'Medium'
    else:
        return 'High'

def get_risk_reasons(row, risk_level):
    reasons = []
    # Slightly less sensitive: higher threshold and fewer hours
    if row.get('transaction_amount', 0) > 10000 and pd.to_datetime(row.get('transaction_date', '2000-01-01')).hour in [1,2,3]:
        reasons.append('Large transaction at odd hour')
    if row.get('click_events', 0) >= 30:
        reasons.append('Very high click events')
    if row.get('mouse_movement', 0) >= 1500:
        reasons.append('Extreme mouse movement')
    if row.get('device_type', -1) == 1 and row.get('touch_events', 0) >= 20:
        reasons.append('High touch events on Mobile')
    if row.get('device_type', -1) == 1 and row.get('device_motion', 0) >= 3.0:
        reasons.append('Device motion spike on Mobile')
    if row.get('time_on_page', 0) in [10, 600]:
        reasons.append('Unusual time on page')
    if row.get('keyboard_events', 1) == 0 and row.get('transaction_amount', 0) > 5000:
        reasons.append('Zero keyboard events with high transaction')
    if row.get('transaction_amount', 0) > 10000:
        reasons.append('Extremely large transaction')
    if row.get('click_events', 0) == 0:
        reasons.append('No click events')
    if row.get('touch_events', 0) == 0 and row.get('device_type', -1) == 1:
        reasons.append('No touch events on Mobile')
    if risk_level == 'High':
        return '; '.join(reasons[:3]) if reasons else 'Multiple strong anomaly signals'
    elif risk_level == 'Medium':
        return reasons[0] if reasons else 'Some features are moderately unusual'
    else:
        return ''

def preprocess_input(input_dict):
    df = pd.DataFrame([input_dict])
    # Impute missing values using imputation_values
    for col, val in imputation_values.items():
        if col not in df.columns or pd.isnull(df.at[0, col]):
            df[col] = val
    df = flag_obvious_anomalies(df)
    for col, le in label_encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col].astype(str))
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    X = df[feature_cols]
    X_scaled = scaler.transform(X)
    return X_scaled, df

def predict(input_dict):
    X_scaled, df_proc = preprocess_input(input_dict)
    proba = model.predict_proba(X_scaled)[:,1][0]
    pred = int(proba >= 0.5)
    risk = assign_risk_level(proba)
    reason = get_risk_reasons(df_proc.iloc[0], risk)
    return {
        'predicted_label': pred,
        'anomaly_score': proba,
        'risk_level': risk,
        'risk_reason': reason
    }

if __name__ == '__main__':
    import json
    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
        with open(sys.argv[1], 'r') as f:
            input_dict = json.load(f)
        result = predict(input_dict)
        print(json.dumps(result, indent=2))
    else:
        print('Usage: python predict.py input.json') 