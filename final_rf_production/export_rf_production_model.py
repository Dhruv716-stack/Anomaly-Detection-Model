import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

# Load training data
train_data = pd.read_csv('synthetic_behavior_dataset_multifeature_anomalies.csv', parse_dates=['transaction_date'])

# --- Preprocessing ---
# Add obvious anomaly flag (same logic as before)
def flag_obvious_anomalies(df):
    flag = np.zeros(len(df), dtype=int)
    for i, row in df.iterrows():
        # Slightly less sensitive: higher threshold and fewer hours
        if (row['transaction_amount'] > 10000) and (row['transaction_date'].hour in [1,2,3]):
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

train_data = flag_obvious_anomalies(train_data)

# Imputation values
imputation_values = {}
numeric_cols = train_data.select_dtypes(include=[np.number]).columns.tolist()
# For count-like features, 0 is logical; for others, use mean
count_like = ['click_events', 'scroll_events', 'touch_events', 'keyboard_events', 'mouse_movement']
for col in numeric_cols:
    if col in count_like:
        imputation_values[col] = 0
    else:
        imputation_values[col] = train_data[col].mean()
# For categorical features, use mode
categorical_cols = train_data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    imputation_values[col] = train_data[col].mode()[0]
# For 'transaction_date', use mode (most common date)
if 'transaction_date' in train_data.columns:
    imputation_values['transaction_date'] = train_data['transaction_date'].mode()[0]

# Impute missing values
for col, val in imputation_values.items():
    train_data[col] = train_data[col].fillna(val)

# Encode categoricals
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col])
    label_encoders[col] = le

# Prepare features/labels
feature_cols = [col for col in train_data.columns if col not in ['label', 'transaction_date']]
X = train_data[feature_cols]
y = train_data['label']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Random Forest
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42
)
rf.fit(X_scaled, y)

# Save artifacts
os.makedirs('.', exist_ok=True)
with open('rf_model.pkl', 'wb') as f:
    pickle.dump(rf, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
with open('feature_cols.pkl', 'wb') as f:
    pickle.dump(feature_cols, f)
with open('imputation_values.pkl', 'wb') as f:
    pickle.dump(imputation_values, f)

print('✅ Production Random Forest model, pipeline, and imputation values saved.') 