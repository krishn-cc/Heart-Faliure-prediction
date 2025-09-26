"""
Flask Backend for Heart Disease Prediction Web Interface
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)
CORS(app)

class HeartDiseasePredictionAPI:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.load_model()
        
    def load_model(self):
        """Load the trained model and scaler."""
        try:
            # Load model and scaler
            model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "heart_disease_model.pkl")
            scaler_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "scaler.pkl")
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            
            # Define feature names in the correct order
            self.feature_names = [
                'Age', 'Sex', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR',
                'ExerciseAngina', 'Oldpeak', 'ChestPain_ASY', 'ChestPain_ATA',
                'ChestPain_NAP', 'ChestPain_TA', 'RestingECG_LVH', 'RestingECG_Normal',
                'RestingECG_ST', 'ST_Slope_Down', 'ST_Slope_Flat', 'ST_Slope_Up'
            ]
            
            print("✅ Model loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def preprocess_input(self, patient_data):
        """Preprocess patient data for prediction."""
        try:
            # Create a DataFrame with the input data
            df = pd.DataFrame([patient_data])
            
            # Convert sex to binary
            df['Sex'] = df['Sex'].map({'M': 1, 'F': 0})
            df['ExerciseAngina'] = df['ExerciseAngina'].map({'Y': 1, 'N': 0})
            
            # One-hot encode ChestPainType
            chest_pain_types = ['ASY', 'ATA', 'NAP', 'TA']
            for cp_type in chest_pain_types:
                df[f'ChestPain_{cp_type}'] = (df['ChestPainType'] == cp_type).astype(int)
            df.drop('ChestPainType', axis=1, inplace=True)
            
            # One-hot encode RestingECG
            ecg_types = ['LVH', 'Normal', 'ST']
            for ecg_type in ecg_types:
                df[f'RestingECG_{ecg_type}'] = (df['RestingECG'] == ecg_type).astype(int)
            df.drop('RestingECG', axis=1, inplace=True)
            
            # One-hot encode ST_Slope
            slope_types = ['Down', 'Flat', 'Up']
            for slope_type in slope_types:
                df[f'ST_Slope_{slope_type}'] = (df['ST_Slope'] == slope_type).astype(int)
            df.drop('ST_Slope', axis=1, inplace=True)
            
            # Ensure all features are present and in correct order
            for feature in self.feature_names:
                if feature not in df.columns:
                    df[feature] = 0
            
            # Select features in the correct order
            df = df[self.feature_names]
            
            # Scale numerical features
            numerical_features = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
            df[numerical_features] = self.scaler.transform(df[numerical_features])
            
            return df.values[0]
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            raise e
    
    def predict(self, patient_data):
        """Make prediction for a patient."""
        try:
            # Preprocess the data
            processed_data = self.preprocess_input(patient_data)
            
            # Make prediction
            prediction = self.model.predict([processed_data])[0]
            probability = self.model.predict_proba([processed_data])[0]
            
            # Determine risk level
            prob_disease = probability[1]
            if prob_disease < 0.50:
                risk_level = "Low Risk"
            elif prob_disease < 0.65:
                risk_level = "Moderate Risk"
            elif prob_disease < 0.80:
                risk_level = "High Risk"
            else:
                risk_level = "Very High Risk"
            
            return {
                'prediction': int(prediction),
                'probability_no_disease': float(probability[0]),
                'probability_heart_disease': float(probability[1]),
                'risk_level': risk_level,
                'success': True
            }
        except Exception as e:
            return {
                'error': str(e),
                'success': False
            }

# Initialize the API
predictor = HeartDiseasePredictionAPI()

@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Convert string numbers to appropriate types
        patient_data = {
            'Age': int(data['age']),
            'Sex': data['sex'],
            'ChestPainType': data['chestPain'],
            'RestingBP': int(data['restingBP']),
            'Cholesterol': int(data['cholesterol']),
            'FastingBS': int(data['fastingBS']),
            'RestingECG': data['restingECG'],
            'MaxHR': int(data['maxHR']),
            'ExerciseAngina': data['exerciseAngina'],
            'Oldpeak': float(data['oldpeak']),
            'ST_Slope': data['stSlope']
        }
        
        # Make prediction
        result = predictor.predict(patient_data)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'error': f'Prediction failed: {str(e)}',
            'success': False
        }), 400

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor.model is not None
    })

if __name__ == '__main__':
    print("🚀 Starting Heart Disease Prediction Web Server...")
    print("📱 Interface will be available at: http://localhost:5000")
    print("🔗 API endpoint: http://localhost:5000/predict")
    
    app.run(debug=True, host='0.0.0.0', port=5000)