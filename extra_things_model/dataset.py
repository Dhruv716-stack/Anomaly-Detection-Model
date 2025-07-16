import pandas as pd
import numpy as np
import random

# Parameters
n_rows = 5000
anomaly_fraction = 0.05
n_anomalies = int(n_rows * anomaly_fraction)

# Seed for reproducibility
np.random.seed(42)
random.seed(42)

# Simulate identifiers
user_ids = [f"user_{i}" for i in np.random.randint(0, 500, n_rows)]
session_ids = [f"session_{i}" for i in range(n_rows)]

# Device types
device_types = np.random.choice(['PC', 'Mobile'], size=n_rows, p=[0.5, 0.5])

# Simulate session time (in seconds)
time_on_page = np.random.exponential(scale=90, size=n_rows).astype(int)
time_on_page = np.clip(time_on_page, 10, 600)

# Platform-specific behaviors
click_events = []
scroll_events = []
touch_events = []
keyboard_events = []
device_motion = []

for device in device_types:
    if device == 'PC':
        click_events.append(np.random.poisson(8))
        scroll_events.append(np.random.poisson(12))
        touch_events.append(0)
        keyboard_events.append(np.random.poisson(15))
        device_motion.append(0.0)
    else:
        click_events.append(np.random.poisson(5))
        scroll_events.append(np.random.poisson(6))
        touch_events.append(np.random.poisson(8))
        keyboard_events.append(np.random.poisson(5))
        device_motion.append(np.round(abs(np.random.normal(loc=1.2, scale=0.5)), 2))

# Create base DataFrame
df = pd.DataFrame({
    'user_id': user_ids,
    'session_id': session_ids,
    'device_type': device_types,
    'click_events': click_events,
    'scroll_events': scroll_events,
    'touch_events': touch_events,
    'keyboard_events': keyboard_events,
    'device_motion': device_motion,
    'time_on_page': time_on_page,
    'screen_size': [random.choice(['1920x1080', '1366x768', '1440x900', '360x640', '414x896']) for _ in range(n_rows)],
    'browser_info': np.random.choice(['Chrome', 'Firefox', 'Safari', 'Edge'], size=n_rows),
    'language': np.random.choice(['foreign', 'english', 'hindi'], size=n_rows),
    'timezone_offset': np.random.choice([-330, 0, 60, 330, 480], size=n_rows),
    'device_orientation': np.random.choice(['portrait', 'landscape', 'none'], size=n_rows),
    'geolocation_city': np.random.choice(['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Indore', 'Unknown'], size=n_rows),
    'transaction_amount': np.round(np.random.exponential(scale=1500, size=n_rows), 2),
    'transaction_date': pd.to_datetime(np.random.choice(pd.date_range('2024-01-01', '2025-07-01', freq='H'), size=n_rows))
})

# Add mouse_movement
mouse_movement = [
    np.random.poisson(lam=(time / 60) * 150) if device == 'PC' else 0
    for device, time in zip(df['device_type'], df['time_on_page'])
]
mouse_movement = np.clip(mouse_movement, 0, 2000)
df['mouse_movement'] = mouse_movement

# Initialize label
df['label'] = 0

# Select indices for anomalies
anomaly_indices = np.random.choice(n_rows, n_anomalies, replace=False)

# Inject multiple anomalous behaviors
for idx in anomaly_indices:
    anomaly_type = np.random.choice([
        'huge_transaction',
        'odd_hour',
        'weird_behavior',
        'combo'
    ])
    
    # Start by marking as anomaly
    df.loc[idx, 'label'] = 1
    
    # Apply multiple signals per anomaly type
    if anomaly_type == 'huge_transaction':
        # Huge transaction
        df.loc[idx, 'transaction_amount'] = np.random.uniform(10000, 20000)
        # Also inject unusual time_on_page (very short or very long)
        df.loc[idx, 'time_on_page'] = np.random.choice([10, 600])
        # Possibly odd hour
        if np.random.rand() < 0.5:
            date = df.loc[idx, 'transaction_date']
            df.loc[idx, 'transaction_date'] = date.replace(hour=np.random.choice([2,3,4]))
    
    elif anomaly_type == 'odd_hour':
        # Transaction at odd hour
        date = df.loc[idx, 'transaction_date']
        df.loc[idx, 'transaction_date'] = date.replace(hour=np.random.choice([2,3,4]))
        # Possibly higher transaction amount
        if np.random.rand() < 0.5:
            df.loc[idx, 'transaction_amount'] = np.random.uniform(5000, 12000)
        # Possibly zero clicks
        if np.random.rand() < 0.5:
            df.loc[idx, 'click_events'] = 0
    
    elif anomaly_type == 'weird_behavior':
        # Extremely high clicks and mouse movement
        df.loc[idx, 'click_events'] = np.random.randint(30, 50)
        df.loc[idx, 'mouse_movement'] = np.random.randint(1500, 2000)
        # Very high touch events if Mobile
        if df.loc[idx, 'device_type'] == 'Mobile':
            df.loc[idx, 'touch_events'] = np.random.randint(20, 40)
        # Possibly unusual device_motion
        if df.loc[idx, 'device_type'] == 'Mobile' and np.random.rand() < 0.7:
            df.loc[idx, 'device_motion'] = np.round(np.random.uniform(3.0,5.0),2)
    
    elif anomaly_type == 'combo':
        # Combine several suspicious signals
        df.loc[idx, 'transaction_amount'] = np.random.uniform(8000, 20000)
        df.loc[idx, 'transaction_date'] = df.loc[idx, 'transaction_date'].replace(hour=np.random.choice([1,2,3]))
        df.loc[idx, 'click_events'] = np.random.randint(20, 40)
        df.loc[idx, 'mouse_movement'] = np.random.randint(1000, 2000)
        # Possibly device motion spike
        if df.loc[idx, 'device_type'] == 'Mobile':
            df.loc[idx, 'device_motion'] = np.round(np.random.uniform(2.5,4.0),2)
        # Possibly zero keyboard events
        if np.random.rand() < 0.3:
            df.loc[idx, 'keyboard_events'] = 0

# Save to CSV
df.to_csv("synthetic_behavior_5%_dataset_multifeature_anomalies.csv", index=False)
print("✅ Dataset saved as synthetic_behavior_5%_dataset_multifeature_anomalies.csv")
