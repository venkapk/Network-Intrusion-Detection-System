from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import pickle
import pandas as pd

# Initialize Flask app with static folder pointing to UI
app = Flask(__name__, static_folder='../ui')
CORS(app)

# Routes to serve the React frontend
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_ui(path):
    # Serve the specific file if it exists, otherwise serve index.html
    # This handles React routing
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

# Set up paths for models and data
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

# Load all models upfront - This is memory heavy but has faster response
# Three categories of models:
# - binary: differentiates between normal traffic and attack
# - multiclass_3: classifies traffic into three categories
# - multiclass_4: classifies traffic into four categories
models = {
    'binary': {
        'logistic_regression': pickle.load(open(os.path.join(MODEL_DIR, 'binary', 'logistic.pkl'), 'rb')),
        'random_forest': pickle.load(open(os.path.join(MODEL_DIR, 'binary', 'random_forest.pkl'), 'rb')),
        'xgboost': pickle.load(open(os.path.join(MODEL_DIR, 'binary', 'xgboost.pkl'), 'rb'))
    },
    'multiclass_3': {
        'logistic_regression': pickle.load(open(os.path.join(MODEL_DIR, 'multiclass_3', 'logistic.pkl'), 'rb')),
        'random_forest': pickle.load(open(os.path.join(MODEL_DIR, 'multiclass_3', 'random_forest.pkl'), 'rb')),
        'xgboost': pickle.load(open(os.path.join(MODEL_DIR, 'multiclass_3', 'xgboost.pkl'), 'rb'))
    },
    'multiclass_4': {
        'logistic_regression': pickle.load(open(os.path.join(MODEL_DIR, 'multiclass_4', 'logistic.pkl'), 'rb')),
        'random_forest': pickle.load(open(os.path.join(MODEL_DIR, 'multiclass_4', 'random_forest.pkl'), 'rb')),
        'xgboost': pickle.load(open(os.path.join(MODEL_DIR, 'multiclass_4', 'xgboost.pkl'), 'rb'))
    }
}

# Load label encoders for translating numeric predictions to labels
label_encoders = {
    'binary': {
        'inverse_transform': lambda x: ['BENIGN' if val == 0 else 'ATTACK' for val in x]
    },
    'multiclass_3': pickle.load(open(os.path.join(DATA_DIR, 'preprocessed', 'tuesday_working_label_encoder.pkl'), 'rb')),
    'multiclass_4': pickle.load(open(os.path.join(DATA_DIR, 'preprocessed', 'thursday_morning_label_encoder.pkl'), 'rb'))
}

# Load scalers for normalizing input features
scalers = {
    'binary': pickle.load(open(os.path.join(DATA_DIR, 'preprocessed', 'friday_afternoon_scaler.pkl'), 'rb')),
    'multiclass_3': pickle.load(open(os.path.join(DATA_DIR, 'preprocessed', 'tuesday_working_scaler.pkl'), 'rb')),
    'multiclass_4': pickle.load(open(os.path.join(DATA_DIR, 'preprocessed', 'thursday_morning_scaler.pkl'), 'rb'))
}

# Load PCA models for dimensionality reduction
pca_models = {
    'binary': pickle.load(open(os.path.join(DATA_DIR, 'preprocessed', 'friday_afternoon_pca.pkl'), 'rb')),
    'multiclass_3': pickle.load(open(os.path.join(DATA_DIR, 'preprocessed', 'tuesday_working_pca.pkl'), 'rb')),
    'multiclass_4': pickle.load(open(os.path.join(DATA_DIR, 'preprocessed', 'thursday_morning_pca.pkl'), 'rb'))
}

# Get the feature names expected by each scaler
scaler_feature_names = {
    'binary': scalers['binary'].feature_names_in_.tolist() if hasattr(scalers['binary'], 'feature_names_in_') else [],
    'multiclass_3': scalers['multiclass_3'].feature_names_in_.tolist() if hasattr(scalers['multiclass_3'], 'feature_names_in_') else [],
    'multiclass_4': scalers['multiclass_4'].feature_names_in_.tolist() if hasattr(scalers['multiclass_4'], 'feature_names_in_') else []
}

# Print expected feature names for debugging during startup
print("Expected feature names for binary:", scaler_feature_names['binary'])
print("Expected feature names for multiclass_3:", scaler_feature_names['multiclass_3'])
print("Expected feature names for multiclass_4:", scaler_feature_names['multiclass_4'])

# Define the minimum feature sets required by each model type
feature_sets = {
    'binary': [
        'Destination Port', 'Total Length of Fwd Packets', 'Bwd Packet Length Max',
        'Bwd Packet Length Min', 'Bwd IAT Total', 'Min Packet Length', 'URG Flag Count',
        'Down/Up Ratio', 'min_seg_size_forward'
    ],
    'multiclass_3': [
        'Min Packet Length', 'PSH Flag Count', 'Init_Win_bytes_forward', 'act_data_pkt_fwd'
    ],
    'multiclass_4': [
        'PSH Flag Count', 'Down/Up Ratio', 'Init_Win_bytes_forward', 'Init_Win_bytes_backward'
    ]
}

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint: takes network flow features and returns classification
    
    Expects JSON with model_type, model_name, and features dictionary
    Returns prediction label, numeric value, and probability breakdown
    """
    data = request.json
    
    # Extract request parameters with defaults
    model_type = data.get('model_type', 'multiclass_4')
    model_name = data.get('model_name', 'random_forest')
    input_features = data.get('features', {})
    
    # Input validation
    required_features = feature_sets[model_type]
    missing_features = [f for f in required_features if f not in input_features]
    
    if missing_features:
        # Return a helpful error if features are missing
        return jsonify({
            'error': 'Missing required features',
            'missing_features': missing_features,
            'required_features': required_features
        }), 400
    
    # Extract only the required features in the correct order
    features = {feature: input_features[feature] for feature in required_features}
    
    # Convert dictionary to DataFrame for scikit-learn compatibility 
    df = pd.DataFrame([features])
    
    # Apply the same preprocessing pipeline used during training:
    # 1. Scale features to normalize their ranges
    scaled_features = scalers[model_type].transform(df)
    
    # 2. Apply PCA to transform to principal components
    pca_features = pca_models[model_type].transform(scaled_features)
    
    # Make prediction using the selected model
    prediction_numeric = models[model_type][model_name].predict(pca_features)[0]
    
    # Get class probabilities
    probabilities = models[model_type][model_name].predict_proba(pca_features)[0]
    
    # Format response based on model type
    if model_type == 'binary':
        prediction_label = 'BENIGN' if prediction_numeric == 0 else 'ATTACK'
        prob_dict = {
            'BENIGN': float(probabilities[0]),
            'ATTACK': float(probabilities[1])
        }
    else:
        # For multiclass, use the label encoder to get original class names
        prediction_label = label_encoders[model_type].inverse_transform([prediction_numeric])[0]
        prob_dict = {label_encoders[model_type].inverse_transform([i])[0]: float(prob) 
                    for i, prob in enumerate(probabilities)}
    
    # Return prediction results as JSON
    return jsonify({
        'prediction': prediction_label,
        'prediction_numeric': int(prediction_numeric),
        'probabilities': prob_dict
    })

@app.route('/api/models', methods=['GET'])
def get_models():
    """Return available model types and algorithms"""
    return jsonify({
        'model_types': list(models.keys()),
        'model_names': list(models['binary'].keys())
    })

@app.route('/api/features', methods=['GET'])
def get_features():
    """
    Get required features for a specified model type
    Also provides descriptions of what each feature means
    """
    model_type = request.args.get('model_type', 'multiclass_4')
    
    # Descriptions of network traffic features
    feature_descriptions = {
        'Destination Port': 'Target port of the connection',
        'Total Length of Fwd Packets': 'Total packet length in the forward direction',
        'Bwd Packet Length Max': 'Maximum packet length in the backward direction',
        'Bwd Packet Length Min': 'Minimum packet length in the backward direction',
        'Bwd IAT Total': 'Total inter-arrival time of packets in the backward direction',
        'Bwd IAT Mean': 'Mean inter-arrival time of packets in the backward direction',
        'Bwd IAT Std': 'Standard deviation of inter-arrival time of packets in the backward direction',
        'Fwd PSH Flags': 'Number of PSH flags in the forward direction',
        'Min Packet Length': 'Minimum length of packets in the flow',
        'PSH Flag Count': 'Number of times the PSH flag was set in the connection',
        'URG Flag Count': 'Number of times the URG flag was set in the connection',
        'Down/Up Ratio': 'Ratio of download to upload size', 
        'Init_Win_bytes_forward': 'The total number of bytes sent in initial window in the forward direction',
        'Init_Win_bytes_backward': 'The total number of bytes sent in initial window in the backward direction',
        'min_seg_size_forward': 'Minimum segment size observed in the forward direction',
        'act_data_pkt_fwd': 'Number of packets with at least 1 byte of TCP data payload in the forward direction'
    }
    
    return jsonify({
        'features': feature_sets[model_type],
        'description': {feature: feature_descriptions.get(feature, 'No description available') 
                      for feature in feature_sets[model_type]}
    })

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """
    Handle file uploads (PCAP or CSV)
    TODO: Implement actual PCAP parsing with scapy or similar
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    # This is just a placeholder - needs implementation
    # Would need to:
    # 1. Save the file
    # 2. Process it (extract features)
    # 3. Run prediction
    # 4. Return results
    
    # For now, just return success
    return jsonify({'message': 'File uploaded successfully'})

if __name__ == '__main__':
    app.run(debug=True)