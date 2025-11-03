"""
Heart Disease Prediction Web Application
Optimized for Render deployment
"""

import os
import pandas as pd
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

class HeartDiseasePredictionAPI:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.is_loaded = False
        
    def load_model_components(self):
        """Load all necessary model components"""
        try:
            print("🔄 Loading model components...")
            
            # Load model
            model_path = os.path.join('models', 'heart_disease_model.pkl')
            self.model = joblib.load(model_path)
            print("✅ Model loaded successfully")
            
            # Load scaler
            scaler_path = os.path.join('models', 'scaler.pkl')
            self.scaler = joblib.load(scaler_path)
            print("✅ Scaler loaded successfully")
            
            # Define feature names in correct order
            self.feature_names = [
                'Age', 'Sex', 'RestingBP', 'Cholesterol', 'FastingBS',
                'MaxHR', 'ExerciseAngina', 'Oldpeak',
                'ChestPain_ATA', 'ChestPain_ASY', 'ChestPain_NAP', 'ChestPain_TA',
                'RestingECG_LVH', 'RestingECG_Normal', 'RestingECG_ST',
                'ST_Slope_Down', 'ST_Slope_Flat', 'ST_Slope_Up'
            ]
            print(f"✅ Feature names loaded: {len(self.feature_names)} features")
            
            self.is_loaded = True
            return True
            
        except Exception as e:
            print(f"❌ Error loading model components: {str(e)}")
            return False
    
    def preprocess_patient_data(self, patient_data):
        """Convert raw patient data to model-ready format"""
        try:
            # Create base dataframe
            processed_data = pd.DataFrame({
                'Age': [float(patient_data['age'])],
                'Sex': [1 if patient_data['sex'] == 'M' else 0],
                'RestingBP': [float(patient_data['restingBP'])],
                'Cholesterol': [float(patient_data['cholesterol'])],
                'FastingBS': [int(patient_data['fastingBS'])],
                'MaxHR': [float(patient_data['maxHR'])],
                'ExerciseAngina': [1 if patient_data['exerciseAngina'] == 'Y' else 0],
                'Oldpeak': [float(patient_data['oldpeak'])]
            })
            
            # One-hot encode ChestPainType
            for cp_type in ['ATA', 'ASY', 'NAP', 'TA']:
                processed_data[f'ChestPain_{cp_type}'] = [1 if patient_data['chestPain'] == cp_type else 0]
            
            # One-hot encode RestingECG
            for ecg_type in ['LVH', 'Normal', 'ST']:
                processed_data[f'RestingECG_{ecg_type}'] = [1 if patient_data['restingECG'] == ecg_type else 0]
            
            # One-hot encode ST_Slope
            for slope_type in ['Down', 'Flat', 'Up']:
                processed_data[f'ST_Slope_{slope_type}'] = [1 if patient_data['stSlope'] == slope_type else 0]
            
            # Ensure column order matches training
            processed_data = processed_data[self.feature_names]
            
            # Scale numerical features
            numerical_columns = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
            processed_data[numerical_columns] = self.scaler.transform(processed_data[numerical_columns])
            
            return processed_data
            
        except Exception as e:
            print(f"❌ Error preprocessing data: {str(e)}")
            raise ValueError(f"Data preprocessing failed: {str(e)}")
    
    def make_prediction(self, patient_data):
        """Make heart disease prediction"""
        if not self.is_loaded:
            raise ValueError("Model components not loaded")
        
        try:
            # Preprocess the data
            processed_data = self.preprocess_patient_data(patient_data)
            
            # Convert to numpy array to avoid feature name warnings
            X = processed_data.values
            
            # Make prediction
            prediction = self.model.predict(X)[0]
            probability = self.model.predict_proba(X)[0][1]
            
            # Determine risk level
            if probability < 0.50:
                risk_level = "Low Risk"
                recommendation = "Maintain healthy lifestyle. Regular checkups recommended."
            elif probability < 0.65:
                risk_level = "Moderate Risk"
                recommendation = "Consider lifestyle changes. Consult with healthcare provider."
            elif probability < 0.80:
                risk_level = "High Risk"
                recommendation = "Seek medical attention soon. Lifestyle changes strongly recommended."
            else:
                risk_level = "Very High Risk"
                recommendation = "Seek immediate medical attention. Urgent consultation required."
            
            return {
                'prediction': int(prediction),
                'probability': round(probability * 100, 2),
                'risk_level': risk_level,
                'recommendation': recommendation
            }
            
        except Exception as e:
            print(f"❌ Error making prediction: {str(e)}")
            raise ValueError(f"Prediction failed: {str(e)}")

# Initialize prediction API
prediction_api = HeartDiseasePredictionAPI()

@app.route('/')
def home():
    """Serve the main web interface"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        patient_data = request.get_json()
        
        if not patient_data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        # Load model if not already loaded
        if not prediction_api.is_loaded:
            if not prediction_api.load_model_components():
                return jsonify({'success': False, 'error': 'Model loading failed'}), 500
        
        # Make prediction
        result = prediction_api.make_prediction(patient_data)
        result['success'] = True
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': prediction_api.is_loaded})

if __name__ == '__main__':
    # Get port from environment (Render sets this)
    port = int(os.environ.get('PORT', 5000))
    
    # Load model on startup
    print("🚀 Starting Heart Disease Prediction System...")
    if prediction_api.load_model_components():
        print(f"🌐 Server starting on port {port}")
        app.run(host='0.0.0.0', port=port, debug=False)
    else:
        print("❌ Failed to load model")
