#!/usr/bin/env python3
"""
Export Production-Ready Anomaly Detection Model
Exports the best performing Random Forest model and preprocessing pipeline
"""

import pandas as pd
import numpy as np
import joblib
import pickle
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load and prepare datasets"""
    print("📂 Loading datasets...")
    
    # Load training data (40% anomalies)
    train_data = pd.read_csv("balanced_dataset_40%_anomalies.csv")
    print(f"✅ Training data: {len(train_data)} samples")
    
    # Load test data (4% anomalies)
    test_data = pd.read_csv("synthetic_behavior_dataset.csv")
    print(f"✅ Test data: {len(test_data)} samples")
    
    return train_data, test_data

def create_production_pipeline(train_data, test_data):
    """Create and train the production pipeline"""
    print("\n🔧 Creating Production Pipeline...")
    
    # Handle categorical columns
    categorical_cols = train_data.select_dtypes(include=['object']).columns
    print(f"📝 Categorical columns: {list(categorical_cols)}")
    
    # Create label encoders for each categorical column
    label_encoders = {}
    for col in categorical_cols:
        if col in train_data.columns and col in test_data.columns:
            le = LabelEncoder()
            # Combine unique values from both datasets
            all_values = pd.concat([train_data[col], test_data[col]]).unique()
            le.fit(all_values)
            label_encoders[col] = le
            
            # Transform the data
            train_data[col] = le.transform(train_data[col])
            test_data[col] = le.transform(test_data[col])
    
    # Separate features and labels
    X_train = train_data.drop('label', axis=1)
    y_train = train_data['label']
    X_test = test_data.drop('label', axis=1)
    y_test = test_data['label']
    
    # Create and fit scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model with chosen class weight
    print("Training Random Forest Model...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight={0: 1, 1: 2},  # Chosen class weight
        random_state=42
    )
    rf_model.fit(X_train_scaled, y_train)
    
    # Evaluate final model
    y_pred = rf_model.predict(X_test_scaled)
    y_scores = rf_model.predict_proba(X_test_scaled)[:, 1]
    
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_scores)
    accuracy = (y_pred == y_test).mean()

    # Evaluate with chosen threshold
    chosen_threshold = 0.6
    y_pred_custom = (y_scores >= chosen_threshold).astype(int)
    precision_custom = precision_score(y_test, y_pred_custom)
    recall_custom = recall_score(y_test, y_pred_custom)
    f1_custom = f1_score(y_test, y_pred_custom)
    print(f"\nCustom Threshold ({chosen_threshold}) Metrics:")
    print(f"   Precision: {precision_custom:.4f}")
    print(f"   Recall:    {recall_custom:.4f}")
    print(f"   F1-Score:  {f1_custom:.4f}")
    print(f"\nFinal Production Model Metrics:")
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    print(f"   AUC:       {auc:.4f}")
    
    return {
        'model': rf_model,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'feature_names': list(X_train.columns),
        'metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
    }

def export_production_artifacts(pipeline_components):
    """Export all production artifacts"""
    print("\n💾 Exporting Production Artifacts...")
    
    # Create models directory
    import os
    os.makedirs('production_models', exist_ok=True)
    
    # Export model
    model_path = 'production_models/anomaly_detection_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline_components['model'], f)
    print(f"✅ Model exported to: {model_path}")
    
    # Export scaler
    scaler_path = 'production_models/scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(pipeline_components['scaler'], f)
    print(f"✅ Scaler exported to: {scaler_path}")
    
    # Export label encoders
    encoders_path = 'production_models/label_encoders.pkl'
    with open(encoders_path, 'wb') as f:
        pickle.dump(pipeline_components['label_encoders'], f)
    print(f"✅ Label encoders exported to: {encoders_path}")
    
    # Export feature names
    features_path = 'production_models/feature_names.pkl'
    with open(features_path, 'wb') as f:
        pickle.dump(pipeline_components['feature_names'], f)
    print(f"✅ Feature names exported to: {features_path}")
    
    # Export metrics
    metrics_path = 'production_models/model_metrics.json'
    import json
    with open(metrics_path, 'w') as f:
        json.dump(pipeline_components['metrics'], f, indent=2)
    print(f"✅ Model metrics exported to: {metrics_path}")
    
    # Create production predictor class
    create_production_predictor()
    
    return {
        'model_path': model_path,
        'scaler_path': scaler_path,
        'encoders_path': encoders_path,
        'features_path': features_path,
        'metrics_path': metrics_path
    }

def create_production_predictor():
    """Create a production-ready predictor class"""
    predictor_code = '''import pandas as pd
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
            print("✅ Model loaded successfully!")
            print(f"📊 Model Performance: F1={self.metrics['f1']:.4f}, AUC={self.metrics['auc']:.4f}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
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
            print(f"⚠️ Missing features: {missing_features}. Filling with zeros.")
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
    print(f"\n🔍 Single Prediction Result:")
    print(f"   Is Anomaly: {result['is_anomaly']}")
    print(f"   Anomaly Probability: {result['anomaly_probability']:.4f}")
    print(f"   Confidence: {result['confidence']:.4f}")
'''
    
    with open('production_models/anomaly_predictor.py', 'w') as f:
        f.write(predictor_code)
    
    print("✅ Production predictor class created: production_models/anomaly_predictor.py")

def main():
    """Main function"""
    print("🚀 Exporting Production-Ready Anomaly Detection Model")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    train_data, test_data = load_and_prepare_data()
    
    # Create production pipeline
    pipeline_components = create_production_pipeline(train_data, test_data)
    
    # Export artifacts
    export_paths = export_production_artifacts(pipeline_components)
    
    print(f"\n🎉 Production Model Export Complete!")
    print(f"📁 All artifacts saved in: production_models/")
    print(f"📊 Model Performance: F1={pipeline_components['metrics']['f1']:.4f}, AUC={pipeline_components['metrics']['auc']:.4f}")
    print(f"💡 Use 'production_models/anomaly_predictor.py' for easy integration")
    
    print(f"\n✅ Export completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main() 