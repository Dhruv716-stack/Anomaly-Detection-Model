import numpy as np
import pandas as pd
import joblib
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, 
    precision_score, recall_score, roc_auc_score, accuracy_score,
    roc_curve, precision_recall_curve
)

# Deep Learning Libraries
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

class AdvancedFeatureGenerator(BaseEstimator, TransformerMixin):
    """
    Advanced feature generator for behavioral anomaly detection
    """
    def __init__(self):
        self.user_stats = {}
        self.feature_names = []
        
    def fit(self, X, y=None):
        # Calculate user-level statistics for normalization
        if 'user_id' in X.columns:
            for user_id in X['user_id'].unique():
                user_data = X[X['user_id'] == user_id]
                self.user_stats[user_id] = {
                    'mean_interaction_score': user_data['interaction_score'].mean() if 'interaction_score' in user_data.columns else 0,
                    'std_interaction_score': user_data['interaction_score'].std() if 'interaction_score' in user_data.columns else 1,
                    'mean_time_on_page': user_data['time_on_page'].mean(),
                    'std_time_on_page': user_data['time_on_page'].std(),
                    'mean_transaction_amount': user_data['transaction_amount'].mean() if 'transaction_amount' in user_data.columns else 0,
                    'std_transaction_amount': user_data['transaction_amount'].std() if 'transaction_amount' in user_data.columns else 1
                }
        return self
    
    def transform(self, X):
        df = X.copy()
        
        # Ensure transaction_date is datetime
        if 'transaction_date' in df.columns:
            df['transaction_date'] = pd.to_datetime(df['transaction_date'])
            
            # Advanced datetime features
            df['transaction_hour'] = df['transaction_date'].dt.hour
            df['transaction_day'] = df['transaction_date'].dt.dayofweek
            df['transaction_month'] = df['transaction_date'].dt.month
            df['transaction_quarter'] = df['transaction_date'].dt.quarter
            df['is_weekend'] = df['transaction_day'].isin([5, 6]).astype(int)
            df['is_business_hours'] = ((df['transaction_hour'] >= 9) & (df['transaction_hour'] <= 17)).astype(int)
            df['is_night'] = ((df['transaction_hour'] >= 22) | (df['transaction_hour'] <= 6)).astype(int)
            
            # Cyclical encoding for time features
            df['hour_sin'] = np.sin(2 * np.pi * df['transaction_hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['transaction_hour'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['transaction_day'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['transaction_day'] / 7)
            df['month_sin'] = np.sin(2 * np.pi * df['transaction_month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['transaction_month'] / 12)

        # Handle screen size parsing
        if 'screen_size' in df.columns:
            try:
                df[['screen_width', 'screen_height']] = df['screen_size'].str.split('x', expand=True).astype(int)
                df['screen_area'] = df['screen_width'] * df['screen_height']
                df['screen_ratio'] = df['screen_width'] / (df['screen_height'] + 1)
                df['screen_density'] = df['screen_area'] / (df['screen_width'] + df['screen_height'])
            except:
                df['screen_width'] = 1920
                df['screen_height'] = 1080
                df['screen_area'] = 2073600
                df['screen_ratio'] = 16/9
                df['screen_density'] = 2073600/3000
            # Drop the original string column
            df = df.drop(columns=['screen_size'])

        # Avoid divide-by-zero in time calculations
        df['time_on_page'] = df['time_on_page'].replace(0, 1)
        
        # Rate-based features
        df['clicks_per_sec'] = df['click_events'] / df['time_on_page']
        df['scrolls_per_sec'] = df['scroll_events'] / df['time_on_page']
        df['touches_per_sec'] = df['touch_events'] / df['time_on_page']
        df['keyboard_per_sec'] = df['keyboard_events'] / df['time_on_page']
        df['mouse_per_sec'] = df['mouse_movement'] / df['time_on_page']
        
        # Advanced interaction features
        df['total_events'] = (df['click_events'] + df['scroll_events'] + 
                             df['touch_events'] + df['keyboard_events'])
        df['events_per_second'] = df['total_events'] / df['time_on_page']
        
        # Interaction ratios and diversity
        df['click_to_scroll_ratio'] = df['click_events'] / (df['scroll_events'] + 1)
        df['touch_to_click_ratio'] = df['touch_events'] / (df['click_events'] + 1)
        df['keyboard_to_click_ratio'] = df['keyboard_events'] / (df['click_events'] + 1)
        df['input_to_navigation_ratio'] = (df['keyboard_events'] + df['touch_events']) / (df['click_events'] + df['scroll_events'] + 1)
        
        # Behavioral consistency features
        interaction_cols = ['click_events', 'scroll_events', 'touch_events', 'keyboard_events']
        df['interaction_diversity'] = df[interaction_cols].std(axis=1)
        df['behavioral_consistency'] = 1 / (df['interaction_diversity'] + 1)
        
        # Efficiency metrics
        df['session_efficiency'] = df['total_events'] / (df['time_on_page'] + 1)
        df['click_efficiency'] = df['click_events'] / (df['time_on_page'] + 1)
        df['mouse_per_click'] = df['mouse_movement'] / (df['click_events'] + 1)
        
        # Device motion features
        df['motion_per_event'] = df['device_motion'] / (df['total_events'] + 1)
        df['motion_per_second'] = df['device_motion'] / (df['time_on_page'] + 1)
        
        # Transaction features
        if 'transaction_amount' in df.columns:
            df['transaction_per_event'] = df['transaction_amount'] / (df['total_events'] + 1)
            df['transaction_per_second'] = df['transaction_amount'] / (df['time_on_page'] + 1)
            df['transaction_efficiency'] = df['transaction_amount'] / (df['time_on_page'] + 1)
        
        # Composite behavioral scores
        df['interaction_score'] = (df['click_events'] + df['scroll_events'] + 
                                  df['touch_events'] + df['keyboard_events']) / df['time_on_page']
        
        df['session_complexity'] = df['total_events'] * df['interaction_diversity']
        df['time_behavior_score'] = df['transaction_hour'] * df['events_per_second']
        
        # Device-specific features
        df['is_mobile'] = (df['device_type'] == 'Mobile').astype(int)
        df['is_pc'] = (df['device_type'] == 'PC').astype(int)
        df['device_behavior_score'] = (df['is_mobile'] * df['touch_events'] + 
                                      df['is_pc'] * df['mouse_movement']) / df['time_on_page']
        
        # User deviation features
        if 'user_id' in df.columns:
            df['user_deviation_score'] = 0
            for user_id in df['user_id'].unique():
                user_mask = df['user_id'] == user_id
                if user_id in self.user_stats:
                    user_avg = self.user_stats[user_id]['mean_interaction_score']
                    df.loc[user_mask, 'user_deviation_score'] = np.abs(
                        df.loc[user_mask, 'interaction_score'] - user_avg
                    )
        
        # Advanced statistical features
        df['event_velocity'] = df['total_events'] / (df['time_on_page'] ** 0.5)
        df['interaction_momentum'] = df['total_events'] * df['time_on_page']
        
        # Risk indicators
        df['high_frequency_risk'] = (df['events_per_second'] > df['events_per_second'].quantile(0.95)).astype(int)
        df['low_frequency_risk'] = (df['events_per_second'] < df['events_per_second'].quantile(0.05)).astype(int)
        df['time_anomaly_risk'] = (df['time_on_page'] > df['time_on_page'].quantile(0.95)).astype(int)
        
        # Store feature names for later use
        self.feature_names = [col for col in df.columns if col not in ['user_id', 'session_id', 'transaction_date', 'label']]
        
        return df

class AnomalyDetectionPipeline:
    """
    Comprehensive anomaly detection pipeline with multiple algorithms
    """
    def __init__(self, algorithm='autoencoder', contamination=0.04):
        self.algorithm = algorithm
        self.contamination = contamination
        self.feature_generator = AdvancedFeatureGenerator()
        self.preprocessor = None
        self.model = None
        self.threshold = None
        self.feature_names = []
        self.is_fitted = False
        
    def build_preprocessor(self, categorical_features, numerical_features):
        """Build the preprocessing pipeline"""
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', KNNImputer(n_neighbors=5)),
                    ('scaler', RobustScaler())
                ]), numerical_features),
                
                ('cat', Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                ]), categorical_features)
            ],
            remainder='drop'
        )
        
    def build_autoencoder(self, input_dim, encoding_dim=32):
        """Build a deep autoencoder for anomaly detection"""
        # Input layer
        input_layer = Input(shape=(input_dim,))
        
        # Encoder
        encoded = Dense(128, activation='relu')(input_layer)
        encoded = BatchNormalization()(encoded)
        encoded = Dropout(0.3)(encoded)
        
        encoded = Dense(64, activation='relu')(encoded)
        encoded = BatchNormalization()(encoded)
        encoded = Dropout(0.3)(encoded)
        
        bottleneck = Dense(encoding_dim, activation='relu', name='bottleneck')(encoded)
        
        # Decoder
        decoded = Dense(64, activation='relu')(bottleneck)
        decoded = BatchNormalization()(decoded)
        decoded = Dropout(0.3)(decoded)
        
        decoded = Dense(128, activation='relu')(decoded)
        decoded = BatchNormalization()(decoded)
        decoded = Dropout(0.3)(decoded)
        
        output_layer = Dense(input_dim, activation='linear')(decoded)
        
        # Create and compile model
        autoencoder = Model(inputs=input_layer, outputs=output_layer)
        autoencoder.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return autoencoder
    
    def fit(self, X, y=None):
        """Fit the anomaly detection pipeline"""
        print("🔄 Starting pipeline fitting...")
        
        # Step 1: Feature generation
        print("📊 Generating advanced features...")
        X_processed = self.feature_generator.fit_transform(X)
        self.feature_names = self.feature_generator.feature_names
        
        # Step 2: Define feature categories
        categorical_features = [
            'device_type', 'browser_info', 'language', 'device_orientation',
            'geolocation_city', 'is_weekend', 'is_business_hours', 'is_night',
            'is_mobile', 'is_pc', 'high_frequency_risk', 'low_frequency_risk',
            'time_anomaly_risk'
        ]
        
        numerical_features = [col for col in self.feature_names if col not in categorical_features]
        
        # Step 3: Build preprocessor
        print("🔧 Building preprocessing pipeline...")
        self.build_preprocessor(categorical_features, numerical_features)
        
        # Step 4: Fit preprocessor
        X_transformed = self.preprocessor.fit_transform(X_processed)
        
        # Step 5: Train model based on algorithm
        if self.algorithm == 'autoencoder':
            print("🧠 Training Autoencoder...")
            self.model = self.build_autoencoder(X_transformed.shape[1])
            
            # Train only on normal data if labels are available
            if y is not None:
                normal_mask = y == 0
                X_normal = X_transformed[normal_mask]
            else:
                X_normal = X_transformed
            
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss'),
                ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
            ]
            
            history = self.model.fit(
                X_normal, X_normal,
                epochs=100,
                batch_size=32,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=1
            )
            
            # Calculate reconstruction error threshold
            reconstructed = self.model.predict(X_transformed)
            reconstruction_errors = np.mean(np.square(X_transformed - reconstructed), axis=1)
            self.threshold = np.percentile(reconstruction_errors, (1 - self.contamination) * 100)
            
        elif self.algorithm == 'isolation_forest':
            print("🌲 Training Isolation Forest...")
            self.model = IsolationForest(
                n_estimators=200,
                contamination=self.contamination,
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(X_transformed)
            
        elif self.algorithm == 'one_class_svm':
            print("🔍 Training One-Class SVM...")
            self.model = OneClassSVM(
                kernel='rbf',
                nu=self.contamination,
                gamma='scale'
            )
            self.model.fit(X_transformed)
            
        self.is_fitted = True
        print("✅ Pipeline fitting completed!")
        return self
    
    def predict(self, X):
        """Predict anomalies"""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before prediction")
        
        # Transform data
        X_processed = self.feature_generator.transform(X)
        X_transformed = self.preprocessor.transform(X_processed)
        
        # Make predictions
        if self.algorithm == 'autoencoder':
            reconstructed = self.model.predict(X_transformed)
            reconstruction_errors = np.mean(np.square(X_transformed - reconstructed), axis=1)
            predictions = (reconstruction_errors > self.threshold).astype(int)
            return predictions, reconstruction_errors
            
        elif self.algorithm == 'isolation_forest':
            predictions = self.model.predict(X_transformed)
            # Convert -1 (anomaly) to 1, 1 (normal) to 0
            predictions = np.where(predictions == -1, 1, 0)
            return predictions, None
            
        elif self.algorithm == 'one_class_svm':
            predictions = self.model.predict(X_transformed)
            # Convert -1 (anomaly) to 1, 1 (normal) to 0
            predictions = np.where(predictions == -1, 1, 0)
            return predictions, None
    
    def evaluate(self, X, y_true):
        """Evaluate the model performance"""
        predictions, scores = self.predict(X)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, predictions)
        precision = precision_score(y_true, predictions, zero_division=0)
        recall = recall_score(y_true, predictions, zero_division=0)
        f1 = f1_score(y_true, predictions, zero_division=0)
        
        # Calculate AUC if scores are available
        auc_score = None
        if scores is not None:
            auc_score = roc_auc_score(y_true, scores)
        
        # Print results
        print("\n" + "="*50)
        print("📊 MODEL EVALUATION RESULTS")
        print("="*50)
        print(f"Algorithm: {self.algorithm.upper()}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        if auc_score:
            print(f"AUC Score: {auc_score:.4f}")
        
        # Classification report
        print("\n📋 Classification Report:")
        print(classification_report(y_true, predictions, digits=4))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, predictions)
        print("\n🔍 Confusion Matrix:")
        print(cm)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Normal', 'Anomaly'],
                    yticklabels=['Normal', 'Anomaly'])
        plt.title(f'Confusion Matrix - {self.algorithm.upper()}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
        
        # Plot ROC curve if scores available
        if scores is not None:
            fpr, tpr, _ = roc_curve(y_true, scores)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {self.algorithm.upper()}')
            plt.legend()
            plt.grid(True)
            plt.show()
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_score': auc_score,
            'predictions': predictions,
            'scores': scores
        }
    
    def save_model(self, filepath):
        """Save the complete pipeline"""
        pipeline_data = {
            'algorithm': self.algorithm,
            'contamination': self.contamination,
            'feature_generator': self.feature_generator,
            'preprocessor': self.preprocessor,
            'model': self.model,
            'threshold': self.threshold,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(pipeline_data, filepath)
        print(f"✅ Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath):
        """Load a saved pipeline"""
        pipeline_data = joblib.load(filepath)
        
        pipeline = cls(
            algorithm=pipeline_data['algorithm'],
            contamination=pipeline_data['contamination']
        )
        
        pipeline.feature_generator = pipeline_data['feature_generator']
        pipeline.preprocessor = pipeline_data['preprocessor']
        pipeline.model = pipeline_data['model']
        pipeline.threshold = pipeline_data['threshold']
        pipeline.feature_names = pipeline_data['feature_names']
        pipeline.is_fitted = pipeline_data['is_fitted']
        
        return pipeline

def main():
    """Main execution function"""
    print("🚀 Starting Anomaly Detection Pipeline")
    print("="*50)
    
    # Load datasets
    print("📂 Loading datasets...")
    train_data = pd.read_csv('balanced_dataset_40%_anomalies.csv')
    test_data = pd.read_csv('imbalanced_user_behaviour.csv')
    
    print(f"Training data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    # Prepare training data
    X_train = train_data.drop(['label'], axis=1, errors='ignore')
    y_train = train_data['label'] if 'label' in train_data.columns else None
    
    # Prepare test data
    X_test = test_data.drop(['label'], axis=1, errors='ignore')
    y_test = test_data['label'] if 'label' in test_data.columns else None
    
    # Train multiple models
    algorithms = ['autoencoder', 'isolation_forest', 'one_class_svm']
    results = {}
    
    for algorithm in algorithms:
        print(f"\n🎯 Training {algorithm.upper()} model...")
        print("-" * 30)
        
        # Create and train pipeline
        pipeline = AnomalyDetectionPipeline(algorithm=algorithm, contamination=0.04)
        pipeline.fit(X_train, y_train)
        
        # Evaluate on test data
        if y_test is not None:
            result = pipeline.evaluate(X_test, y_test)
            results[algorithm] = result
        
        # Save model
        pipeline.save_model(f'models/{algorithm}_pipeline.joblib')
    
    # Compare results
    if results:
        print("\n🏆 MODEL COMPARISON")
        print("="*50)
        comparison_df = pd.DataFrame({
            'Algorithm': list(results.keys()),
            'Accuracy': [results[alg]['accuracy'] for alg in results.keys()],
            'Precision': [results[alg]['precision'] for alg in results.keys()],
            'Recall': [results[alg]['recall'] for alg in results.keys()],
            'F1-Score': [results[alg]['f1_score'] for alg in results.keys()],
            'AUC-Score': [results[alg]['auc_score'] for alg in results.keys()]
        })
        
        print(comparison_df.to_string(index=False))
        
        # Find best model
        best_model = comparison_df.loc[comparison_df['F1-Score'].idxmax(), 'Algorithm']
        print(f"\n🥇 Best Model: {best_model} (Highest F1-Score)")
    
    print("\n✅ Pipeline execution completed!")

if __name__ == "__main__":
    main() 