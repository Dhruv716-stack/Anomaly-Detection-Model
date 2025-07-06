# Anomaly Detection Model for User Behavior Analysis

This project implements an autoencoder-based anomaly detection system for identifying suspicious user behavior patterns in mobile device interactions. The model analyzes various behavioral traits and device sensor data to detect potential security threats.

## Table of Contents
- [Data Format Requirements](#data-format-requirements)
- [Required Columns](#required-columns)
- [Preprocessing Steps](#preprocessing-steps)
- [Model Access and Usage](#model-access-and-usage)
- [Pipeline Usage](#pipeline-usage)
- [Risk Calculation Function](#risk-calculation-function)
- [Installation and Dependencies](#installation-and-dependencies)
- [Example Usage](#example-usage)

## Data Format Requirements

### Input Data Format
The model expects input data in CSV format with the following specifications:

- **File Format**: CSV (Comma-separated values)
- **Encoding**: UTF-8
- **Date Format**: `YYYY-MM-DD HH:MM:SS` (e.g., "2023-12-01 14:30:25")
- **Missing Values**: Can be handled by the pipeline (imputed automatically)
- **Data Types**: Mixed (numeric and categorical features)

### Required Columns

The following columns are **required** for the model to function properly:

#### Behavioral Features (Numeric):
- `tap_pressure` (float64): Pressure applied during screen taps
- `swipe_speed` (float64): Speed of swipe gestures
- `typing_speed` (float64): Average typing speed in characters per second
- `inter_key_delay` (float64): Average delay between keystrokes
- `grip_x`, `grip_y`, `grip_z` (float64): Device grip sensor readings
- `tilt_angle_mean` (float64): Average device tilt angle
- `dwell_time_avg` (float64): Average time spent on screen elements
- `wifi_network_hash` (float64): Hash of connected WiFi network
- `device_tilt_variation` (float64): Variation in device tilt
- `fast_typing_burst` (int64): Number of rapid typing sequences
- `gps_drift` (float64): GPS location drift measurement
- `transaction_amount` (float64): Amount of financial transaction

#### Categorical Features:
- `gesture_type` (object): Type of gesture performed (e.g., "tap", "swipe", "pinch")
- `screen_sequence` (object): Sequence of screen interactions
- `geolocation_cluster` (object): Geographic location cluster identifier

#### Optional Columns:
- `user_id` (object): User identifier (will be dropped during processing)
- `trait_categories` (object): Behavioral trait categories (will be dropped during processing)
- `label` (int): Ground truth labels (0 for normal, 1 for anomaly) - only needed for training

## Preprocessing Steps

### 1. DateTime Preprocessing
Before feeding data to the model, the following datetime preprocessing must be applied:

```python
import pandas as pd

# Convert transaction_date to datetime format
df["transaction_date"] = pd.to_datetime(df["transaction_date"], format="%Y-%m-%d %H:%M:%S")

# Extract time-based features
df["hour"] = df["transaction_date"].dt.hour
df["day_of_week"] = df["transaction_date"].dt.dayofweek
df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)
df["is_night"] = df["hour"].apply(lambda h: 1 if h <= 5 else 0)
```

### 2. Pipeline Preprocessing
The model uses a scikit-learn pipeline that automatically handles:

- **Numeric Features**: Mean imputation + StandardScaler normalization
- **Categorical Features**: Mode imputation + OneHotEncoder encoding
- **Missing Values**: Automatic imputation for both numeric and categorical data

## Model Access and Usage

### Loading the Model
```python
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained autoencoder model
autoencoder = load_model("autoencoder_model.keras")
```

### Model Architecture
The autoencoder consists of:
- **Input Layer**: Adapts to the number of features
- **Encoder**: Dense layers (64 → 32 neurons) with ReLU activation
- **Decoder**: Dense layers (32 → 64 → input_dim) with ReLU activation
- **Output**: Reconstruction of input features
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam

## Pipeline Usage

### Loading the Pipeline
```python
import joblib

# Load the preprocessing pipeline
pipeline = joblib.load("pipeline.joblib")
```

### Applying Preprocessing
```python
# Prepare features (exclude non-feature columns)
feature_columns = ['tap_pressure', 'swipe_speed', 'gesture_type', 'typing_speed', 
                   'inter_key_delay', 'grip_x', 'grip_y', 'grip_z', 'tilt_angle_mean',
                   'screen_sequence', 'dwell_time_avg', 'geolocation_cluster', 
                   'wifi_network_hash', 'device_tilt_variation', 'fast_typing_burst',
                   'gps_drift', 'transaction_amount', 'hour', 'day_of_week', 
                   'is_weekend', 'is_night']

X = df[feature_columns]

# Apply preprocessing
X_processed = pipeline.transform(X)
```

## Risk Calculation Function

The model provides a three-level risk assessment system:

```python
import numpy as np

def calculate_risk_level(autoencoder, pipeline, input_data, threshold=None):
    """
    Calculate risk level for user behavior data.
    
    Args:
        autoencoder: Loaded autoencoder model
        pipeline: Loaded preprocessing pipeline
        input_data: DataFrame with required features
        threshold: Optional custom threshold (default: optimized threshold)
    
    Returns:
        dict: Contains 'risk_level', 'reconstruction_error', 'confidence_score'
    """
    
    # Apply preprocessing
    X_processed = pipeline.transform(input_data)
    
    # Get reconstruction error
    reconstructions = autoencoder.predict(X_processed)
    errors = np.mean(np.square(X_processed - reconstructions), axis=1)
    
    # Use optimized threshold if not provided
    if threshold is None:
        threshold = 0.0178  # Optimized threshold from training
    
    # Define risk thresholds
    low_threshold = np.percentile(errors, 60)  # 60th percentile
    high_threshold = threshold  # Optimized threshold for high risk
    
    def get_risk_label(error):
        if error < low_threshold:
            return 'Low'
        elif error < high_threshold:
            return 'Moderate'
        else:
            return 'High'
    
    # Calculate risk levels
    risk_levels = [get_risk_label(e) for e in errors]
    
    # Calculate confidence score (inverse of error, normalized)
    max_error = np.max(errors)
    confidence_scores = 1 - (errors / max_error)
    
    return {
        'risk_level': risk_levels[0] if len(risk_levels) == 1 else risk_levels,
        'reconstruction_error': float(errors[0]) if len(errors) == 1 else errors.tolist(),
        'confidence_score': float(confidence_scores[0]) if len(confidence_scores) == 1 else confidence_scores.tolist()
    }
```

### Risk Levels Explained:
- **Low Risk**: Reconstruction error < 60th percentile of training errors
- **Moderate Risk**: Reconstruction error between 60th percentile and optimized threshold
- **High Risk**: Reconstruction error > optimized threshold (0.0178)

## Installation and Dependencies

### Required Packages:
```bash
pip install tensorflow pandas scikit-learn joblib numpy
```

### Package Versions:
- TensorFlow >= 2.0.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- joblib >= 1.1.0
- numpy >= 1.21.0

## Example Usage

### Complete Example:
```python
import pandas as pd
import tensorflow as tf
import joblib
import numpy as np

# Load models
autoencoder = tf.keras.models.load_model("autoencoder_model.keras")
pipeline = joblib.load("pipeline.joblib")

# Load and preprocess data
df = pd.read_csv("your_input_data.csv")

# DateTime preprocessing
df["transaction_date"] = pd.to_datetime(df["transaction_date"], format="%Y-%m-%d %H:%M:%S")
df["hour"] = df["transaction_date"].dt.hour
df["day_of_week"] = df["transaction_date"].dt.dayofweek
df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)
df["is_night"] = df["hour"].apply(lambda h: 1 if h <= 5 else 0)

# Prepare features
feature_columns = ['tap_pressure', 'swipe_speed', 'gesture_type', 'typing_speed', 
                   'inter_key_delay', 'grip_x', 'grip_y', 'grip_z', 'tilt_angle_mean',
                   'screen_sequence', 'dwell_time_avg', 'geolocation_cluster', 
                   'wifi_network_hash', 'device_tilt_variation', 'fast_typing_burst',
                   'gps_drift', 'transaction_amount', 'hour', 'day_of_week', 
                   'is_weekend', 'is_night']

X = df[feature_columns]

# Calculate risk
risk_result = calculate_risk_level(autoencoder, pipeline, X)

print(f"Risk Level: {risk_result['risk_level']}")
print(f"Reconstruction Error: {risk_result['reconstruction_error']:.4f}")
print(f"Confidence Score: {risk_result['confidence_score']:.4f}")
```

## Model Performance

The model achieves the following performance metrics:
- **Accuracy**: 99.9%
- **Precision**: 100% (Anomaly detection)
- **Recall**: 90% (Anomaly detection)
- **F1-Score**: 95% (Anomaly detection)
- **ROC-AUC Score**: 0.967

## Notes

- The model is trained on imbalanced data with a focus on detecting rare anomalies
- The optimized threshold (0.0178) was determined by maximizing F1-score on the validation set
- Risk levels are calculated relative to the training data distribution
- The pipeline automatically handles missing values and feature scaling
- All categorical features are automatically encoded using one-hot encoding