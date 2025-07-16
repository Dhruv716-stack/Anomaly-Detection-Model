import pandas as pd
import numpy as np
import pickle
import json
from typing import Dict, List, Union, Tuple
import warnings
warnings.filterwarnings('ignore')

class AnomalyDetectionPredictor:
    """
    Production-ready anomaly detection predictor
    """
    
    def __init__(self, model_dir: str = 'production_models'):
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.label_encoders = None
        self.feature_names = None
        self.metrics = None
        self.is_loaded = False
        self.load_model()
    
    def load_model(self):
        try:
            with open(f'{self.model_dir}/anomaly_detection_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            with open(f'{self.model_dir}/scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            with open(f'{self.model_dir}/label_encoders.pkl', 'rb') as f:
                self.label_encoders = pickle.load(f)
            with open(f'{self.model_dir}/feature_names.pkl', 'rb') as f:
                self.feature_names = pickle.load(f)
            with open(f'{self.model_dir}/model_metrics.json', 'r') as f:
                self.metrics = json.load(f)
            self.is_loaded = True
            print("Model loaded successfully!")
            print(f"Model Performance: F1={self.metrics['f1']:.4f}, AUC={self.metrics['auc']:.4f}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.is_loaded = False
    
    def preprocess_data(self, data: pd.DataFrame) -> np.ndarray:
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        processed_data = data.copy()
        for col, encoder in self.label_encoders.items():
            if col in processed_data.columns:
                unique_values = processed_data[col].unique()
                for val in unique_values:
                    if val not in encoder.classes_:
                        processed_data[col] = processed_data[col].replace(val, encoder.classes_[0])
                processed_data[col] = encoder.transform(processed_data[col])
        missing_features = set(self.feature_names) - set(processed_data.columns)
        if missing_features:
            print(f"Missing features: {missing_features}. Filling with zeros.")
            for feature in missing_features:
                processed_data[feature] = 0
        processed_data = processed_data[self.feature_names]
        scaled_data = self.scaler.transform(processed_data)
        return scaled_data
    
    def predict(self, data: Union[pd.DataFrame, Dict, List[Dict]]) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        elif isinstance(data, list):
            data = pd.DataFrame(data)
        elif not isinstance(data, pd.DataFrame):
            raise ValueError("Input must be DataFrame, dict, or list of dicts")
        processed_data = self.preprocess_data(data)
        predictions = self.model.predict(processed_data)
        probabilities = self.model.predict_proba(processed_data)[:, 1]
        return predictions, probabilities

    def get_risk_level(self, anomaly_score: float) -> str:
        if anomaly_score < 0.25:
            return "low"
        elif anomaly_score < 0.5:
            return "medium"
        else:
            return "high"

    def predict_single(self, data: Dict) -> Dict:
        _, probabilities = self.predict(data)
        risk_level = self.get_risk_level(probabilities[0])
        return {
            'anomaly_score': float(probabilities[0]),
            'risk_level': risk_level
        }

    def predict_batch(self, data: List[Dict]) -> List[Dict]:
        _, probabilities = self.predict(data)
        results = []
        for i in range(len(probabilities)):
            risk_level = self.get_risk_level(probabilities[i])
            results.append({
                'anomaly_score': float(probabilities[i]),
                'risk_level': risk_level
            })
        return results

    def get_model_info(self) -> Dict:
        return {
            'model_type': 'Random Forest',
            'performance_metrics': self.metrics,
            'feature_count': len(self.feature_names),
            'features': self.feature_names,
            'is_loaded': self.is_loaded
        }

if __name__ == "__main__":
    predictor = AnomalyDetectionPredictor()
    sample_data = {
        'user_id': 'unknown_user_999',
        'session_id': 'session_999',
        'device_type': 'unknown_device',
        'click_events': 0,
        'scroll_events': 0,
        'touch_events': 0,
        'keyboard_events': 9999,
        'device_motion': 9999,
        'time_on_page': 99999,
        'screen_size': '9999x9999',
        'browser_info': 'Unknown/0.0',
        'language': 'xx-YY',
        'timezone_offset': 999,
        'device_orientation': 'upside_down',
        'geolocation_city': 'Atlantis',
        'transaction_amount': 999999,
        'transaction_date': '2099-12-31',
        'mouse_movement': 99999
    }
    result = predictor.predict_single(sample_data)
    print("\nSingle Prediction Result:")
    for k, v in result.items():
        print(f"   {k}: {v}")
