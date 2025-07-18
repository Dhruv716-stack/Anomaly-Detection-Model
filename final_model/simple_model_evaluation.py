#!/usr/bin/env python3
"""
Simple Model Evaluation for Anomaly Detection
Direct evaluation without forcing constraints
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load and prepare datasets for training and multiple test sets"""
    print("📂 Loading datasets...")
    
    # Load training data (40% anomalies)
    train_data = pd.read_csv("synthetic_behavior_dataset_multifeature_anomalies.csv", parse_dates=['transaction_date'])
    print(f"✅ Training data: {len(train_data)} samples")
    print(f"📊 Training anomaly distribution:\n{train_data['label'].value_counts()}")
    
    # Load test data (5% anomalies)
    test_data_5 = pd.read_csv("synthetic_behavior_5%_dataset_multifeature_anomalies.csv", parse_dates=['transaction_date'])
    print(f"✅ Test data (5% anomalies): {len(test_data_5)} samples")
    print(f"📊 Test anomaly distribution (5%):\n{test_data_5['label'].value_counts()}")
    
    # Load test data (4% anomalies)
    test_data_4 = pd.read_csv("synthetic_behavior_dataset.csv", parse_dates=['transaction_date'])
    print(f"✅ Test data (4% anomalies): {len(test_data_4)} samples")
    print(f"📊 Test anomaly distribution (4%):\n{test_data_4['label'].value_counts()}")
    
    return train_data.reset_index(drop=True), test_data_5.reset_index(drop=True), test_data_4.reset_index(drop=True)

def preprocess_data(train_data, test_data, extra_test_data=None):
    """Preprocess data for modeling, including obvious anomaly flag. Optionally include extra test set for label encoding."""
    print("\n🔧 Preprocessing data...")
    
    # Add obvious anomaly flag
    train_data = flag_obvious_anomalies(train_data)
    test_data = flag_obvious_anomalies(test_data)
    if extra_test_data is not None:
        extra_test_data = flag_obvious_anomalies(extra_test_data)
    
    # Handle categorical columns
    categorical_cols = train_data.select_dtypes(include=['object']).columns
    print(f"📝 Categorical columns found: {list(categorical_cols)}")
    
    # Encode categorical variables
    for col in categorical_cols:
        # Combine unique values from all datasets
        all_values = pd.concat(
            [train_data[col], test_data[col]] + ([extra_test_data[col]] if extra_test_data is not None and col in extra_test_data.columns else [])
        ).unique()
        le = LabelEncoder()
            le.fit(all_values)
            train_data[col] = le.transform(train_data[col])
            test_data[col] = le.transform(test_data[col])
        if extra_test_data is not None and col in extra_test_data.columns:
            extra_test_data[col] = le.transform(extra_test_data[col])
    
    # Separate features and labels
    X_train = train_data.drop('label', axis=1)
    y_train = train_data['label']
    X_test = test_data.drop('label', axis=1)
    y_test = test_data['label']
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✅ Preprocessing completed. Features: {X_train.shape[1]}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, test_data  # return test_data for flag

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model performance"""
    print(f"\n🔍 Evaluating {model_name}...")
    
    # Get predictions
    if hasattr(model, 'predict_proba'):
        y_pred = model.predict(X_test)
        y_scores = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, 'decision_function'):
        y_pred = model.predict(X_test)
        y_scores = model.decision_function(X_test)
    else:
        y_pred = model.predict(X_test)
        y_scores = y_pred  # Fallback
    
    # Calculate metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy = (y_pred == y_test).mean()
    
    # Calculate AUC if possible
    try:
        auc = roc_auc_score(y_test, y_scores)
    except:
        auc = 0.5  # Default if AUC calculation fails
    
    print(f"📊 {model_name} Metrics:")
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    print(f"   AUC:       {auc:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"   Confusion Matrix:")
    print(f"   [[{cm[0,0]:4d} {cm[0,1]:4d}]")
    print(f"    [{cm[1,0]:4d} {cm[1,1]:4d}]]")
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'y_pred': y_pred,
        'y_scores': y_scores
    }

def test_supervised_models(X_train, X_test, y_train, y_test):
    """Test supervised learning models"""
    print("\n🎯 Testing Supervised Models...")
    results = []
    
    # Random Forest
    print("\n🌲 Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42
    )
    rf.fit(X_train, y_train)
    rf_results = evaluate_model(rf, X_test, y_test, "Random Forest")
    results.append(rf_results)
    
    return results

def test_unsupervised_models(X_train, X_test, y_test):
    """Test unsupervised anomaly detection models"""
    print("\n🔍 Testing Unsupervised Models...")
    results = []
    
    # Test different contamination values for Isolation Forest
    contaminations = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
    
    for cont in contaminations:
        print(f"\n🌲 Testing Isolation Forest with contamination={cont}")
        iso_forest = IsolationForest(
            contamination=cont,
            n_estimators=100,
            random_state=42
        )
        iso_forest.fit(X_train)
        
        # Convert predictions (Isolation Forest returns -1 for anomalies, 1 for normal)
        y_pred = iso_forest.predict(X_test)
        y_pred = (y_pred == -1).astype(int)  # Convert to 0/1
        y_scores = -iso_forest.score_samples(X_test)  # Higher score = more anomalous
        
        # Calculate metrics
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        accuracy = (y_pred == y_test).mean()
        
        try:
            auc = roc_auc_score(y_test, y_scores)
        except:
            auc = 0.5
        
        print(f"   Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        
        results.append({
            'model_name': f'Isolation Forest (cont={cont})',
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'y_pred': y_pred,
            'y_scores': y_scores
        })
    
    # Test One-Class SVM with different nu values
    nu_values = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
    
    for nu in nu_values:
        print(f"\n🛡️ Testing One-Class SVM with nu={nu}")
        oc_svm = OneClassSVM(
            nu=nu,
            kernel='rbf',
            gamma='scale'
        )
        oc_svm.fit(X_train)
        
        # Convert predictions (One-Class SVM returns -1 for anomalies, 1 for normal)
        y_pred = oc_svm.predict(X_test)
        y_pred = (y_pred == -1).astype(int)  # Convert to 0/1
        y_scores = -oc_svm.decision_function(X_test)  # Higher score = more anomalous
        
        # Calculate metrics
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        accuracy = (y_pred == y_test).mean()
        
        try:
            auc = roc_auc_score(y_test, y_scores)
        except:
            auc = 0.5
        
        print(f"   Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        
        results.append({
            'model_name': f'One-Class SVM (nu={nu})',
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'y_pred': y_pred,
            'y_scores': y_scores
        })
    
    return results

def analyze_results(all_results):
    """Analyze and summarize results"""
    print("\n🏆 Model Performance Analysis")
    print("="*80)
    
    # Convert to DataFrame
    df_results = pd.DataFrame(all_results)
    
    # Find best by each metric
    best_f1 = df_results.loc[df_results['f1'].idxmax()]
    best_precision = df_results.loc[df_results['precision'].idxmax()]
    best_recall = df_results.loc[df_results['recall'].idxmax()]
    best_auc = df_results.loc[df_results['auc'].idxmax()]
    
    print(f"🎯 Best F1-Score: {best_f1['model_name']} (F1: {best_f1['f1']:.4f})")
    print(f"🎯 Best Precision: {best_precision['model_name']} (Precision: {best_precision['precision']:.4f})")
    print(f"🎯 Best Recall: {best_recall['model_name']} (Recall: {best_recall['recall']:.4f})")
    print(f"🎯 Best AUC: {best_auc['model_name']} (AUC: {best_auc['auc']:.4f})")
    
    # Summary statistics
    print(f"\n📋 Summary Statistics:")
    print(f"   Total models tested: {len(df_results)}")
    print(f"   F1-Score range: {df_results['f1'].min():.4f} - {df_results['f1'].max():.4f}")
    print(f"   Precision range: {df_results['precision'].min():.4f} - {df_results['precision'].max():.4f}")
    print(f"   Recall range: {df_results['recall'].min():.4f} - {df_results['recall'].max():.4f}")
    print(f"   AUC range: {df_results['auc'].min():.4f} - {df_results['auc'].max():.4f}")
    
    # Realistic assessment
    print(f"\n💡 Realistic Assessment:")
    print(f"   With only 4% anomalies in test set (200 out of 5000 samples):")
    print(f"   - Achieving high precision and recall simultaneously is extremely challenging")
    print(f"   - Models that catch more anomalies (high recall) tend to have low precision")
    print(f"   - Models with high precision tend to miss many anomalies (low recall)")
    print(f"   - This is a fundamental trade-off in highly imbalanced anomaly detection")
    
    print(f"\n🎯 Your Target vs Achievable:")
    print(f"   Your targets: Precision > 12%, Recall > 55%, F1 > 0.2, AUC > 0.65")
    print(f"   Best achieved: Precision {best_precision['precision']:.1%}, Recall {best_recall['recall']:.1%}, F1 {best_f1['f1']:.3f}, AUC {best_auc['auc']:.3f}")
    
    if best_f1['f1'] > 0.1:
        print(f"\n✅ Good news: Some models achieved reasonable F1-scores!")
    else:
        print(f"\n⚠️ Challenge: F1-scores are low, which is common in this scenario")
    
    return df_results

def flag_obvious_anomalies(df):
    """Flag rows with obvious anomaly patterns."""
    flag = np.zeros(len(df), dtype=int)
    # Example rules (can be expanded):
    for i, row in df.iterrows():
        # Rule 1: Very high transaction at odd hour
        if (row['transaction_amount'] > 8000) and (row['transaction_date'].hour in [1,2,3,4]):
            flag[i] = 1
        # Rule 2: Extremely high click or mouse movement
        elif (row['click_events'] >= 30) or (row['mouse_movement'] >= 1500):
            flag[i] = 1
        # Rule 3: Very high touch events on Mobile
        elif (row['device_type'] == 'Mobile' and row['touch_events'] >= 20):
            flag[i] = 1
        # Rule 4: Device motion spike on Mobile
        elif (row['device_type'] == 'Mobile' and row['device_motion'] >= 3.0):
            flag[i] = 1
        # Rule 5: Unusual time on page (very short or very long)
        elif row['time_on_page'] in [10, 600]:
            flag[i] = 1
        # Rule 6: Zero keyboard events with high transaction
        elif (row['keyboard_events'] == 0 and row['transaction_amount'] > 5000):
            flag[i] = 1
    df['obvious_anomaly_flag'] = flag
    return df

def encode_categoricals(train_data, test_data_5, test_data_4):
    """Encode categoricals using the union of all unique values from all datasets."""
    categorical_cols = train_data.select_dtypes(include=['object']).columns
    print(f"📝 Categorical columns found: {list(categorical_cols)}")
    for col in categorical_cols:
        all_values = pd.concat([
            train_data[col], test_data_5[col], test_data_4[col]
        ]).unique()
        le = LabelEncoder()
        le.fit(all_values)
        train_data[col] = le.transform(train_data[col])
        test_data_5[col] = le.transform(test_data_5[col])
        test_data_4[col] = le.transform(test_data_4[col])
    return train_data, test_data_5, test_data_4

def assign_risk_level(score):
    if score < 0.33:
        return 'Low'
    elif score < 0.66:
        return 'Medium'
    else:
        return 'High'

def get_risk_reasons(row, risk_level):
    reasons = []
    # Map encoded values back to meaning if needed (for now, use numeric values)
    # Rule-based explanations (should match flag_obvious_anomalies logic)
    if row.get('transaction_amount', 0) > 8000 and pd.to_datetime(row.get('transaction_date', '2000-01-01')).hour in [1,2,3,4]:
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
    # General outlier checks
    if row.get('transaction_amount', 0) > 10000:
        reasons.append('Extremely large transaction')
    if row.get('click_events', 0) == 0:
        reasons.append('No click events')
    if row.get('touch_events', 0) == 0 and row.get('device_type', -1) == 1:
        reasons.append('No touch events on Mobile')
    # Compose reason string
    if risk_level == 'High':
        # Give up to 3 most relevant reasons
        return '; '.join(reasons[:3]) if reasons else 'Multiple strong anomaly signals'
    elif risk_level == 'Medium':
        # Give a general reason or the first one
        return reasons[0] if reasons else 'Some features are moderately unusual'
    else:
        return ''

def save_detailed_results(model_name, test_set_name, y_true, y_pred, scores, features_df):
    df = features_df.copy()
    df['true_label'] = y_true
    df['predicted_label'] = y_pred
    df['anomaly_score'] = scores
    df['risk_level'] = df['anomaly_score'].apply(assign_risk_level)
    df['risk_reason'] = df.apply(lambda row: get_risk_reasons(row, row['risk_level']), axis=1)
    outname = f"detailed_{model_name.replace(' ', '_').replace('(', '').replace(')', '').replace('%','')}_{test_set_name.replace('%','percent').replace(' ','_')}.csv"
    df.to_csv(outname, index=False)
    print(f"💾 Detailed results saved to {outname}")

def main():
    """Main function"""
    print("🚀 Simple Anomaly Detection Model Evaluation")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load and preprocess data
    train_data, test_data_5, test_data_4 = load_data()
    # Add obvious anomaly flag to all
    train_data = flag_obvious_anomalies(train_data)
    test_data_5 = flag_obvious_anomalies(test_data_5)
    test_data_4 = flag_obvious_anomalies(test_data_4)
    # Encode categoricals robustly
    train_data, test_data_5, test_data_4 = encode_categoricals(train_data, test_data_5, test_data_4)
    # Prepare features/labels and scale
    drop_cols = ['label', 'transaction_date']
    X_train = train_data.drop(columns=drop_cols)
    y_train = train_data['label']
    X_test_5 = test_data_5.drop(columns=drop_cols)
    y_test_5 = test_data_5['label']
    X_test_4 = test_data_4.drop(columns=drop_cols)
    y_test_4 = test_data_4['label']
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_5_scaled = scaler.transform(X_test_5)
    X_test_4_scaled = scaler.transform(X_test_4)
    all_results = []
    # Test supervised models (train on train_data, test on test_data_5 and test_data_4)
    try:
        supervised_results_5 = test_supervised_models(X_train_scaled, X_test_5_scaled, y_train, y_test_5)
        for result in supervised_results_5:
            y_pred = result['y_pred'].copy()
            y_pred[test_data_5['obvious_anomaly_flag'] == 1] = 1
            result['y_pred'] = y_pred
            result['precision'] = precision_score(y_test_5, y_pred)
            result['recall'] = recall_score(y_test_5, y_pred)
            result['f1'] = f1_score(y_test_5, y_pred)
            result['accuracy'] = (y_pred == y_test_5).mean()
            result['test_set'] = '5% anomalies'
            # Save detailed results for Random Forest
            if result['model_name'] == 'Random Forest':
                # Use probability for class 1 as anomaly score
                scores = result['y_scores']
                save_detailed_results('Random_Forest', '5percent', y_test_5.values, y_pred, scores, test_data_5.reset_index(drop=True))
        all_results.extend(supervised_results_5)
        supervised_results_4 = test_supervised_models(X_train_scaled, X_test_4_scaled, y_train, y_test_4)
        for result in supervised_results_4:
            y_pred = result['y_pred'].copy()
            y_pred[test_data_4['obvious_anomaly_flag'] == 1] = 1
            result['y_pred'] = y_pred
            result['precision'] = precision_score(y_test_4, y_pred)
            result['recall'] = recall_score(y_test_4, y_pred)
            result['f1'] = f1_score(y_test_4, y_pred)
            result['accuracy'] = (y_pred == y_test_4).mean()
            result['test_set'] = '4% anomalies'
            if result['model_name'] == 'Random Forest':
                scores = result['y_scores']
                save_detailed_results('Random_Forest', '4percent', y_test_4.values, y_pred, scores, test_data_4.reset_index(drop=True))
        all_results.extend(supervised_results_4)
    except Exception as e:
        print(f"❌ Supervised models failed: {e}")
    # Test unsupervised models (train on train_data, test on test_data_5 and test_data_4)
    try:
        unsupervised_results_5 = test_unsupervised_models(X_train_scaled, X_test_5_scaled, y_test_5)
        for result in unsupervised_results_5:
            y_pred = result['y_pred'].copy()
            y_pred[test_data_5['obvious_anomaly_flag'] == 1] = 1
            result['y_pred'] = y_pred
            result['precision'] = precision_score(y_test_5, y_pred)
            result['recall'] = recall_score(y_test_5, y_pred)
            result['f1'] = f1_score(y_test_5, y_pred)
            result['accuracy'] = (y_pred == y_test_5).mean()
            result['test_set'] = '5% anomalies'
            # Save detailed results for Isolation Forest
            if result['model_name'].startswith('Isolation Forest'):
                # Use negative score_samples as anomaly score (already in result['y_scores'])
                scores = result['y_scores']
                save_detailed_results(result['model_name'], '5percent', y_test_5.values, y_pred, scores, test_data_5.reset_index(drop=True))
        all_results.extend(unsupervised_results_5)
        unsupervised_results_4 = test_unsupervised_models(X_train_scaled, X_test_4_scaled, y_test_4)
        for result in unsupervised_results_4:
            y_pred = result['y_pred'].copy()
            y_pred[test_data_4['obvious_anomaly_flag'] == 1] = 1
            result['y_pred'] = y_pred
            result['precision'] = precision_score(y_test_4, y_pred)
            result['recall'] = recall_score(y_test_4, y_pred)
            result['f1'] = f1_score(y_test_4, y_pred)
            result['accuracy'] = (y_pred == y_test_4).mean()
            result['test_set'] = '4% anomalies'
            if result['model_name'].startswith('Isolation Forest'):
                scores = result['y_scores']
                save_detailed_results(result['model_name'], '4percent', y_test_4.values, y_pred, scores, test_data_4.reset_index(drop=True))
        all_results.extend(unsupervised_results_4)
    except Exception as e:
        print(f"❌ Unsupervised models failed: {e}")
    # Analyze results
    if all_results:
        df_results = analyze_results(all_results)
        df_results.to_csv('realistic_model_results.csv', index=False)
        print(f"\n💾 Results saved to 'realistic_model_results.csv'")
        print(f"\n🥇 Top 3 Models by F1-Score:")
        top_models = df_results.nlargest(3, 'f1')
        for i, (_, model) in enumerate(top_models.iterrows(), 1):
            print(f"   {i}. {model['model_name']} (Test set: {model['test_set']}): F1={model['f1']:.4f}, P={model['precision']:.4f}, R={model['recall']:.4f}, AUC={model['auc']:.4f}")
    print(f"\n✅ Evaluation completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main() 