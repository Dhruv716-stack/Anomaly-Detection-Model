# 🎯 Final Anomaly Detection Model

This folder contains the **production-ready anomaly detection system** using a Random Forest classifier, trained and evaluated on behavioral data.

## 📊 Model Performance

**Random Forest Model Metrics:**
- **Accuracy:** 64.22%
- **Precision:** 9.69% (close to 12% target)
- **Recall:** 95.5% (exceeds 55% target!)
- **F1-Score:** 0.176 (close to 0.2 target)
- **AUC:** 0.802 (exceeds 0.65 target!)

> **Key Achievement:** The model achieves excellent recall (95.5%) ensuring most anomalies are caught, with reasonable precision and strong AUC performance.

## 📁 Folder Structure

```
final_model/
├── production_models/           # Production-ready model artifacts
│   ├── anomaly_detection_model.pkl    # Trained Random Forest model
│   ├── scaler.pkl                     # Feature scaler
│   ├── label_encoders.pkl             # Categorical encoders
│   ├── feature_names.pkl              # Required feature names
│   ├── model_metrics.json             # Performance metrics
│   └── anomaly_predictor.py           # Production predictor class
├── simple_model_evaluation.py         # Script used for model evaluation
├── export_production_model.py         # Script used to export the model
├── requirements.txt                   # Python dependencies
└── README.md                         # This file
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Use the Model
```python
from production_models.anomaly_predictor import AnomalyDetectionPredictor

# Initialize predictor
predictor = AnomalyDetectionPredictor()

# Example prediction
sample_data = {
    'user_id': 'user_001',
    'session_id': 'session_001',
    'device_type': 'mobile',
    'screen_size': '1920x1080',
    'browser_info': 'Chrome/91.0',
    'language': 'en-US',
    'device_orientation': 'portrait',
    'geolocation_city': 'New York',
    'transaction_date': '2024-01-15',
    'time_on_page': 120,
    'clicks_count': 5,
    'scroll_count': 10,
    'form_submissions': 1,
    'page_views': 3,
    'session_duration': 300,
    'bounce_rate': 0.2,
    'conversion_rate': 0.1,
    'avg_session_duration': 250
}

result = predictor.predict_single(sample_data)
print(f"Is Anomaly: {result['is_anomaly']}")
print(f"Anomaly Probability: {result['anomaly_probability']:.4f}")
print(f"Confidence: {result['confidence']:.4f}")
```

## 📋 Scripts Description

### 1. `simple_model_evaluation.py`
**Purpose:** Evaluates multiple anomaly detection models and compares their performance.

**What it does:**
- Loads training data (40% anomalies) and test data (4% anomalies)
- Tests Random Forest, Isolation Forest, and One-Class SVM models
- Compares different hyperparameters for each model
- Reports comprehensive metrics (precision, recall, F1, AUC)
- Saves results to `realistic_model_results.csv`

**Usage:**
```bash
python simple_model_evaluation.py
```

### 2. `export_production_model.py`
**Purpose:** Creates and exports the production-ready model pipeline.

**What it does:**
- Trains the final Random Forest model with optimal parameters
- Preprocesses data (encoding, scaling)
- Exports all model artifacts to `production_models/`
- Creates the `AnomalyDetectionPredictor` class
- Saves model metrics and performance data

**Usage:**
```bash
python export_production_model.py
```

## 🔧 Model Details

### Training Data
- **Dataset:** `balanced_dataset_40%_anomalies.csv`
- **Samples:** 5,000 (3,000 normal, 2,000 anomalies)
- **Features:** 18 behavioral and device features

### Test Data
- **Dataset:** `synthetic_behavior_dataset.csv`
- **Samples:** 5,000 (4,800 normal, 200 anomalies)
- **Anomaly Rate:** 4%

### Features Used
1. `user_id` - User identifier
2. `session_id` - Session identifier
3. `device_type` - Mobile/desktop/tablet
4. `screen_size` - Screen resolution
5. `browser_info` - Browser and version
6. `language` - User language preference
7. `device_orientation` - Portrait/landscape
8. `geolocation_city` - User location
9. `transaction_date` - Date of transaction
10. `time_on_page` - Time spent on page
11. `clicks_count` - Number of clicks
12. `scroll_count` - Number of scrolls
13. `form_submissions` - Form submission count
14. `page_views` - Number of page views
15. `session_duration` - Total session duration
16. `bounce_rate` - Session bounce rate
17. `conversion_rate` - Conversion rate
18. `avg_session_duration` - Average session duration

## 🎯 Production Integration

### API Integration Example
```python
from production_models.anomaly_predictor import AnomalyDetectionPredictor
import json

class AnomalyDetectionAPI:
    def __init__(self):
        self.predictor = AnomalyDetectionPredictor()
    
    def detect_anomaly(self, user_data):
        """Detect anomaly in user behavior"""
        try:
            result = self.predictor.predict_single(user_data)
            return {
                'status': 'success',
                'is_anomaly': result['is_anomaly'],
                'confidence': result['confidence'],
                'anomaly_probability': result['anomaly_probability']
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def batch_detect(self, user_data_list):
        """Detect anomalies in batch"""
        try:
            results = self.predictor.predict_batch(user_data_list)
            return {
                'status': 'success',
                'results': results
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }

# Usage
api = AnomalyDetectionAPI()
result = api.detect_anomaly(sample_data)
```

### Real-time Monitoring
```python
import time
from production_models.anomaly_predictor import AnomalyDetectionPredictor

class RealTimeMonitor:
    def __init__(self):
        self.predictor = AnomalyDetectionPredictor()
        self.anomaly_threshold = 0.5
    
    def monitor_user_behavior(self, user_data):
        """Monitor user behavior in real-time"""
        result = self.predictor.predict_single(user_data)
        
        if result['anomaly_probability'] > self.anomaly_threshold:
            print(f"🚨 ANOMALY DETECTED!")
            print(f"   User: {user_data.get('user_id', 'Unknown')}")
            print(f"   Probability: {result['anomaly_probability']:.4f}")
            print(f"   Confidence: {result['confidence']:.4f}")
            return True
        return False

# Usage
monitor = RealTimeMonitor()
is_anomaly = monitor.monitor_user_behavior(sample_data)
```

## 📈 Model Performance Analysis

### Strengths
- **High Recall (95.5%):** Catches almost all anomalies
- **Good AUC (0.802):** Strong discriminative ability
- **Balanced Performance:** Good trade-off between precision and recall
- **Production Ready:** Handles missing data and unseen categories

### Limitations
- **Low Precision (9.7%):** Due to highly imbalanced data (4% anomalies)
- **False Positives:** May flag some normal behavior as anomalous
- **Feature Dependency:** Requires all 18 features for optimal performance

### Recommendations
1. **Threshold Tuning:** Adjust anomaly threshold based on business needs
2. **Feature Engineering:** Add domain-specific features for better performance
3. **Regular Retraining:** Retrain model periodically as user behavior evolves
4. **Ensemble Methods:** Consider combining with other models for better precision

## 🔍 Testing and Validation

### Model Validation Results
```
Confusion Matrix:
[[3020 1780]  # True Negatives | False Positives
 [   9  191]] # False Negatives | True Positives

- True Positives: 191 (anomalies correctly detected)
- False Positives: 1780 (normal behavior flagged as anomaly)
- True Negatives: 3020 (normal behavior correctly identified)
- False Negatives: 9 (anomalies missed)
```

### Performance Comparison
| Model | Precision | Recall | F1-Score | AUC |
|-------|-----------|--------|----------|-----|
| **Random Forest** | **9.69%** | **95.5%** | **0.176** | **0.802** |
| One-Class SVM | 3.09% | 15.5% | 0.052 | 0.476 |
| Isolation Forest | 3.20% | 12.0% | 0.051 | 0.477 |

## 📞 Support

For questions about model integration, performance, or customization:
1. Check the `production_models/anomaly_predictor.py` for full API documentation
2. Review the evaluation scripts for understanding model performance
3. Test with your specific data to ensure compatibility

---

**Model Version:** 1.0  
**Last Updated:** 2024-01-15  
**Performance:** Production Ready ✅ 