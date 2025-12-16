# app.py
from flask import Flask, request, jsonify
import joblib
import json
import pandas as pd
import numpy as np
from flask_cors import CORS 
from xgboost import XGBRegressor # Needed to load the XGBoost model type

# --- 1. Load Model and Features ---
# Paths are relative to app.py inside MOV_PROJECT
MODEL_PATH = 'deployment_assets/movie_predictor_model.pkl'
FEATURES_PATH = 'deployment_assets/feature_list.json'

try:
    # Load the XGBoost model and feature list
    MODEL = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH, 'r') as f:
        FEATURE_LIST = json.load(f)['features']
    print("[INIT] Model and features loaded successfully.")
except Exception as e:
    print(f"[FATAL ERROR] Failed to load model files: {e}")
    exit()

# --- 2. Initialize Flask App ---
app = Flask(__name__)
CORS(app) # Enables Cross-Origin Resource Sharing for the Wix site

# --- 3. Prediction Function ---
def format_and_predict(raw_input_data: dict) -> float:
    """Converts raw JSON input into the 20-feature array and runs prediction."""
    
    # Create template DataFrame (1 row, 20 columns)
    X_predict = pd.DataFrame(0.0, index=[0], columns=FEATURE_LIST)
    
    # Map the raw input (from the web) to the template
    for feature, value in raw_input_data.items():
        if feature in X_predict.columns and value is not None:
            # Safely convert input value to float
            X_predict.loc[0, feature] = float(value)
        
    # Predict and convert from log scale
    predicted_revenue_log = MODEL.predict(X_predict)
    predicted_revenue_actual = np.expm1(predicted_revenue_log)
    
    return predicted_revenue_actual[0]

# --- 4. API Endpoint ---
@app.route('/predict', methods=['POST'])
def predict_endpoint():
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400

        result = format_and_predict(data)

        return jsonify({
            'status': 'success',
            'predicted_revenue': f'{result:,.2f}',
            'note': 'Model optimized for general trends, may under-predict extreme outliers.'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- 5. Run the Server ---
if __name__ == '__main__':
    # You must have Flask and Flask-CORS installed: pip install Flask Flask-CORS
    app.run(debug=True, host='0.0.0.0', port=5000)
    # Final dummy line to force push