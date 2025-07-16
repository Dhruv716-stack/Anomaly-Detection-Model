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
        """
        Initialize the predictor with trained model artifacts
        
        Args:
            model_dir: Directory containing model artifacts
        """
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.label_encoders = None
        self.feature_names = None
        self.metrics = None
        self.is_loaded = False
        
        self.load_model()
    
    def load_model(self):
        """Load all model artifacts"""
        try:
            # Load model
            with open(f'{self.model_dir}/anomaly_detection_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            
            # Load scaler
            with open(f'{self.model_dir}/scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            
            # Load label encoders
            with open(f'{self.model_dir}/label_encoders.pkl', 'rb') as f:
                self.label_encoders = pickle.load(f)
            
            # Load feature names
            with open(f'{self.model_dir}/feature_names.pkl', 'rb') as f:
                self.feature_names = pickle.load(f)
            
            # Load metrics
            with open(f'{self.model_dir}/model_metrics.json', 'r') as f:
                self.metrics = json.load(f)
            
            self.is_loaded = True
            print("Model loaded successfully!")
            print(f"Model Performance: F1={self.metrics['f1']:.4f}, AUC={self.metrics['auc']:.4f}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            self.is_loaded = False
    
    def preprocess_data(self, data: pd.DataFrame) -> np.ndarray:
        """
        Preprocess input data for prediction
        
        Args:
            data: Input DataFrame with features
            
        Returns:
            Preprocessed numpy array
        """
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Create a copy to avoid modifying original data
        processed_data = data.copy()
        
        # Encode categorical variables
        for col, encoder in self.label_encoders.items():
            if col in processed_data.columns:
                # Handle unseen categories
                unique_values = processed_data[col].unique()
                for val in unique_values:
                    if val not in encoder.classes_:
                        # Replace with most common category
                        processed_data[col] = processed_data[col].replace(val, encoder.classes_[0])
                
                processed_data[col] = encoder.transform(processed_data[col])
        
        # Ensure all required features are present
        missing_features = set(self.feature_names) - set(processed_data.columns)
        if missing_features:
            print(f"Missing features: {missing_features}. Filling with zeros.")
            for feature in missing_features:
                processed_data[feature] = 0
        
        # Select only required features in correct order
        processed_data = processed_data[self.feature_names]
        
        # Scale features
        scaled_data = self.scaler.transform(processed_data)
        
        return scaled_data
    
    def predict(self, data: Union[pd.DataFrame, Dict, List[Dict]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomalies in the data
        
        Args:
            data: Input data (DataFrame, dict, or list of dicts)
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Convert input to DataFrame
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        elif isinstance(data, list):
            data = pd.DataFrame(data)
        elif not isinstance(data, pd.DataFrame):
            raise ValueError("Input must be DataFrame, dict, or list of dicts")
        
        # Preprocess data
        processed_data = self.preprocess_data(data)
        
        # Make predictions
        predictions = self.model.predict(processed_data)
        probabilities = self.model.predict_proba(processed_data)[:, 1]  # Probability of anomaly
        
        return predictions, probabilities
    
    def predict_single(self, data: Dict) -> Dict:
        """
        Predict anomaly for a single data point
        
        Args:
            data: Single data point as dictionary
            
        Returns:
            Dictionary with prediction results
        """
        predictions, probabilities = self.predict(data)
        
        return {
            'is_anomaly': bool(predictions[0]),
            'anomaly_probability': float(probabilities[0]),
            'confidence': float(max(probabilities[0], 1 - probabilities[0])),
            'prediction': int(predictions[0])
        }
    
    def predict_batch(self, data: List[Dict]) -> List[Dict]:
        """
        Predict anomalies for multiple data points
        
        Args:
            data: List of data points
            
        Returns:
            List of prediction results
        """
        predictions, probabilities = self.predict(data)
        
        results = []
        for i in range(len(predictions)):
            results.append({
                'is_anomaly': bool(predictions[i]),
                'anomaly_probability': float(probabilities[i]),
                'confidence': float(max(probabilities[i], 1 - probabilities[i])),
                'prediction': int(predictions[i])
            })
        
        return results
    
    def get_model_info(self) -> Dict:
        """Get model information and performance metrics"""
        return {
            'model_type': 'Random Forest',
            'performance_metrics': self.metrics,
            'feature_count': len(self.feature_names),
            'features': self.feature_names,
            'is_loaded': self.is_loaded
        }

# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = AnomalyDetectionPredictor()
    
    # Example single prediction
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
    print(f"Single Prediction Result:")
    print(f"   Is Anomaly: {result['is_anomaly']}")
    print(f"   Anomaly Probability: {result['anomaly_probability']:.4f}")
    print(f"   Confidence: {result['confidence']:.4f}")

