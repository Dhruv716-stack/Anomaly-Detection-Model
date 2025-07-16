#!/usr/bin/env python3
"""
Test script for the exported anomaly detection model
Verifies the model works correctly and demonstrates usage
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from production_models.anomaly_predictor import AnomalyDetectionPredictor
import json

def test_model_loading():
    """Test if the model loads correctly"""
    print("🔧 Testing Model Loading...")
    try:
        predictor = AnomalyDetectionPredictor()
        if predictor.is_loaded:
            print("✅ Model loaded successfully!")
            print(f"📊 Model Performance: F1={predictor.metrics['f1']:.4f}, AUC={predictor.metrics['auc']:.4f}")
            return predictor
        else:
            print("❌ Model failed to load")
            return None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def test_single_prediction(predictor):
    """Test single prediction"""
    print("\n🔍 Testing Single Prediction...")
    
    # Sample data for testing
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
    
    try:
        result = predictor.predict_single(sample_data)
        print("✅ Single prediction successful!")
        print(f"   Is Anomaly: {result['is_anomaly']}")
        print(f"   Anomaly Probability: {result['anomaly_probability']:.4f}")
        print(f"   Confidence: {result['confidence']:.4f}")
        print(f"   Prediction: {result['prediction']}")
        return True
    except Exception as e:
        print(f"❌ Single prediction failed: {e}")
        return False

def test_batch_prediction(predictor):
    """Test batch prediction"""
    print("\n📦 Testing Batch Prediction...")
    
    # Multiple sample data points
    batch_data = [
        {
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
        },
        {
            'user_id': 'user_002',
            'session_id': 'session_002',
            'device_type': 'desktop',
            'screen_size': '2560x1440',
            'browser_info': 'Firefox/89.0',
            'language': 'en-GB',
            'device_orientation': 'landscape',
            'geolocation_city': 'London',
            'transaction_date': '2024-01-15',
            'time_on_page': 300,
            'clicks_count': 15,
            'scroll_count': 25,
            'form_submissions': 2,
            'page_views': 8,
            'session_duration': 600,
            'bounce_rate': 0.1,
            'conversion_rate': 0.3,
            'avg_session_duration': 450
        }
    ]
    
    try:
        results = predictor.predict_batch(batch_data)
        print("✅ Batch prediction successful!")
        for i, result in enumerate(results):
            print(f"   Sample {i+1}: Anomaly={result['is_anomaly']}, Prob={result['anomaly_probability']:.4f}")
        return True
    except Exception as e:
        print(f"❌ Batch prediction failed: {e}")
        return False

def test_model_info(predictor):
    """Test model information retrieval"""
    print("\n📋 Testing Model Information...")
    
    try:
        info = predictor.get_model_info()
        print("✅ Model info retrieved successfully!")
        print(f"   Model Type: {info['model_type']}")
        print(f"   Feature Count: {info['feature_count']}")
        print(f"   Is Loaded: {info['is_loaded']}")
        print(f"   Performance Metrics: {json.dumps(info['performance_metrics'], indent=2)}")
        return True
    except Exception as e:
        print(f"❌ Model info retrieval failed: {e}")
        return False

def test_edge_cases(predictor):
    """Test edge cases and error handling"""
    print("\n⚠️ Testing Edge Cases...")
    
    # Test with missing features
    incomplete_data = {
        'user_id': 'user_003',
        'device_type': 'mobile',
        'time_on_page': 100,
        'clicks_count': 3
        # Missing other required features
    }
    
    try:
        result = predictor.predict_single(incomplete_data)
        print("✅ Edge case (missing features) handled successfully!")
        print(f"   Result: Anomaly={result['is_anomaly']}, Prob={result['anomaly_probability']:.4f}")
        return True
    except Exception as e:
        print(f"❌ Edge case handling failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Testing Exported Anomaly Detection Model")
    print("="*50)
    
    # Test model loading
    predictor = test_model_loading()
    if not predictor:
        print("❌ Cannot proceed with tests - model failed to load")
        return
    
    # Run all tests
    tests = [
        ("Single Prediction", lambda: test_single_prediction(predictor)),
        ("Batch Prediction", lambda: test_batch_prediction(predictor)),
        ("Model Information", lambda: test_model_info(predictor)),
        ("Edge Cases", lambda: test_edge_cases(predictor))
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Model is ready for production use.")
    else:
        print("⚠️ Some tests failed. Please check the model setup.")
    
    # Display final model performance
    print(f"\n📈 Final Model Performance:")
    print(f"   Accuracy:  {predictor.metrics['accuracy']:.4f}")
    print(f"   Precision: {predictor.metrics['precision']:.4f}")
    print(f"   Recall:    {predictor.metrics['recall']:.4f}")
    print(f"   F1-Score:  {predictor.metrics['f1']:.4f}")
    print(f"   AUC:       {predictor.metrics['auc']:.4f}")

if __name__ == "__main__":
    main() 