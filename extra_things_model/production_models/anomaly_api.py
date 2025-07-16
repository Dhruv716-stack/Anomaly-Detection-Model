from flask import Flask, request, jsonify
from anomaly_predictor import AnomalyDetectionPredictor

app = Flask(__name__)
predictor = AnomalyDetectionPredictor()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if isinstance(data, dict):
        result = predictor.predict_single(data)
        return jsonify(result)
    elif isinstance(data, list):
        results = predictor.predict_batch(data)
        return jsonify(results)
    else:
        return jsonify({'error': 'Input must be a dict or list of dicts'}), 400

@app.route('/')
def home():
    return 'Anomaly Detection API is running.'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 