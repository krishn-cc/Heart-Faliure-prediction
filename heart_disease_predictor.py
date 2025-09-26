"""
Heart Disease Prediction Interface

This script provides a simple interface for the heart disease prediction model.
It includes functionality for training the model and making predictions on new data.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from data_preprocessing import HeartDiseasePreprocessor
from model_training import HeartDiseaseModelTrainer
import os

class HeartDiseasePredictionInterface:
    def __init__(self):
        """Initialize the prediction interface."""
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.scaler = None
        
    def train_and_setup_model(self):
        """Train the model and set up the prediction interface."""
        print("HEART DISEASE PREDICTION MODEL")
        print("=" * 50)
        print("Setting up the model...")
        
        # Check if model already exists
        if os.path.exists('heart_disease_model.pkl'):
            print("Found existing model. Loading...")
            self.load_model()
        else:
            print("No existing model found. Training new model...")
            self.train_new_model()
            
        print("Model setup completed!")
        
    def train_new_model(self):
        """Train a new model from scratch."""
        # Run preprocessing and training
        trainer = HeartDiseaseModelTrainer()
        self.model, metrics = trainer.complete_training_pipeline()
        
        # Set up preprocessor for future predictions
        self.preprocessor = HeartDiseasePreprocessor("heart.csv")
        self.preprocessor.preprocess_complete_pipeline()
        
        # Store feature names and scaler
        self.feature_names = trainer.X_train.columns.tolist()
        self.scaler = self.preprocessor.scaler
        
        # Save scaler
        joblib.dump(self.scaler, 'scaler.pkl')
        
        # Save feature names
        with open('feature_names.txt', 'w') as f:
            for feature in self.feature_names:
                f.write(f"{feature}\n")
                
    def load_model(self):
        """Load existing model and related components."""
        self.model = joblib.load('heart_disease_model.pkl')
        self.scaler = joblib.load('scaler.pkl') 
        
        # Load feature names
        with open('feature_names.txt', 'r') as f:
            self.feature_names = [line.strip() for line in f.readlines()]
            
    def preprocess_input(self, patient_data):
        """
        Preprocess a single patient's data for prediction.
        
        Args:
            patient_data (dict): Dictionary containing patient information
            
        Returns:
            numpy.ndarray: Preprocessed data ready for prediction
        """
        # Create a dataframe from input
        df = pd.DataFrame([patient_data])
        
        # Apply the same preprocessing steps as training data
        
        # Handle categorical variables
        df['Sex'] = df['Sex'].map({'M': 1, 'F': 0})
        df['ExerciseAngina'] = df['ExerciseAngina'].map({'Y': 1, 'N': 0})
        
        # One-hot encode ChestPainType
        chest_pain_types = ['ATA', 'ASY', 'NAP', 'TA']
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
        
        # Ensure all feature columns are present and in correct order
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0
                
        # Reorder columns to match training data
        df = df[self.feature_names]
        
        # Scale numerical features
        numerical_features = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
        df[numerical_features] = self.scaler.transform(df[numerical_features])
        
        return df.values
        
    def predict_single_patient(self, patient_data):
        """
        Make a prediction for a single patient.
        
        Args:
            patient_data (dict): Dictionary containing patient information
            
        Returns:
            dict: Prediction results
        """
        # Preprocess the input
        processed_data = self.preprocess_input(patient_data)
        
        # Make prediction
        prediction = self.model.predict(processed_data)[0]
        probability = self.model.predict_proba(processed_data)[0]
        
        return {
            'prediction': int(prediction),
            'probability_no_disease': float(probability[0]),
            'probability_heart_disease': float(probability[1]),
            'risk_level': self.get_risk_level(probability[1])
        }
        
    def get_risk_level(self, probability):
        """
        Determine risk level based on probability.
        
        Args:
            probability (float): Probability of heart disease
            
        Returns:
            str: Risk level description
        """
        if probability < 0.3:
            return "Low Risk"
        elif probability < 0.6:
            return "Moderate Risk"
        elif probability < 0.8:
            return "High Risk"
        else:
            return "Very High Risk"
            
    def interactive_prediction(self):
        """Interactive interface for making predictions."""
        print("\n" + "="*50)
        print("INTERACTIVE HEART DISEASE PREDICTION")
        print("="*50)
        
        while True:
            print("\nEnter patient information:")
            
            try:
                # Get user input
                patient_data = {}
                
                patient_data['Age'] = int(input("Age: "))
                patient_data['Sex'] = input("Sex (M/F): ").upper()
                
                print("Chest Pain Types: ATA (Atypical Angina), ASY (Asymptomatic), NAP (Non-Anginal Pain), TA (Typical Angina)")
                patient_data['ChestPainType'] = input("Chest Pain Type: ").upper()
                
                patient_data['RestingBP'] = int(input("Resting Blood Pressure (mm Hg): "))
                patient_data['Cholesterol'] = int(input("Cholesterol (mm/dl): "))
                patient_data['FastingBS'] = int(input("Fasting Blood Sugar > 120 mg/dl (1=Yes, 0=No): "))
                
                print("Resting ECG: Normal, ST, LVH")
                patient_data['RestingECG'] = input("Resting ECG: ")
                
                patient_data['MaxHR'] = int(input("Maximum Heart Rate: "))
                patient_data['ExerciseAngina'] = input("Exercise Induced Angina (Y/N): ").upper()
                patient_data['Oldpeak'] = float(input("ST Depression (Oldpeak): "))
                
                print("ST Slope: Up, Flat, Down")
                patient_data['ST_Slope'] = input("ST Slope: ")
                
                # Make prediction
                result = self.predict_single_patient(patient_data)
                
                # Display results
                print("\n" + "-"*40)
                print("PREDICTION RESULTS")
                print("-"*40)
                print(f"Prediction: {'Heart Disease' if result['prediction'] == 1 else 'No Heart Disease'}")
                print(f"Probability of Heart Disease: {result['probability_heart_disease']:.2%}")
                print(f"Risk Level: {result['risk_level']}")
                print("-"*40)
                
                # Ask if user wants to continue
                continue_pred = input("\nMake another prediction? (y/n): ").lower()
                if continue_pred != 'y':
                    break
                    
            except (ValueError, KeyError) as e:
                print(f"Error: {e}")
                print("Please check your input and try again.")
                continue
                
    def batch_prediction_demo(self):
        """Demonstrate batch predictions with sample data."""
        print("\n" + "="*50)
        print("BATCH PREDICTION DEMO")
        print("="*50)
        
        # Sample patients for demonstration
        sample_patients = [
            {
                'Age': 45, 'Sex': 'M', 'ChestPainType': 'ATA', 'RestingBP': 140,
                'Cholesterol': 289, 'FastingBS': 0, 'RestingECG': 'Normal',
                'MaxHR': 172, 'ExerciseAngina': 'N', 'Oldpeak': 0, 'ST_Slope': 'Up'
            },
            {
                'Age': 60, 'Sex': 'M', 'ChestPainType': 'ASY', 'RestingBP': 150,
                'Cholesterol': 300, 'FastingBS': 1, 'RestingECG': 'Normal',
                'MaxHR': 120, 'ExerciseAngina': 'Y', 'Oldpeak': 2.0, 'ST_Slope': 'Flat'
            },
            {
                'Age': 35, 'Sex': 'F', 'ChestPainType': 'NAP', 'RestingBP': 120,
                'Cholesterol': 200, 'FastingBS': 0, 'RestingECG': 'Normal',
                'MaxHR': 180, 'ExerciseAngina': 'N', 'Oldpeak': 0, 'ST_Slope': 'Up'
            }
        ]
        
        print("Predicting for sample patients...")
        
        for i, patient in enumerate(sample_patients, 1):
            print(f"\nPatient {i}:")
            print(f"Age: {patient['Age']}, Sex: {patient['Sex']}, Chest Pain: {patient['ChestPainType']}")
            
            result = self.predict_single_patient(patient)
            
            print(f"Prediction: {'Heart Disease' if result['prediction'] == 1 else 'No Heart Disease'}")
            print(f"Probability: {result['probability_heart_disease']:.2%}")
            print(f"Risk Level: {result['risk_level']}")
            
    def display_model_info(self):
        """Display information about the trained model."""
        print("\n" + "="*50)
        print("MODEL INFORMATION")
        print("="*50)
        
        print(f"Model Type: Logistic Regression")
        print(f"Number of Features: {len(self.feature_names)}")
        print(f"Features: {', '.join(self.feature_names[:5])}...")  # Show first 5
        
        if hasattr(self.model, 'coef_'):
            print(f"Model Coefficients Available: Yes")
            
        print(f"Model File: heart_disease_model.pkl")
        print(f"Scaler File: scaler.pkl")
        
    def main_menu(self):
        """Main menu interface."""
        while True:
            print("\n" + "="*50)
            print("HEART DISEASE PREDICTION SYSTEM")
            print("="*50)
            print("1. Train/Setup Model")
            print("2. Make Single Prediction")
            print("3. Batch Prediction Demo")
            print("4. View Model Information")
            print("5. Exit")
            print("-"*50)
            
            choice = input("Select an option (1-5): ")
            
            if choice == '1':
                self.train_and_setup_model()
            elif choice == '2':
                if self.model is None:
                    print("Please train/setup the model first (option 1)")
                else:
                    self.interactive_prediction()
            elif choice == '3':
                if self.model is None:
                    print("Please train/setup the model first (option 1)")
                else:
                    self.batch_prediction_demo()
            elif choice == '4':
                if self.model is None:
                    print("Please train/setup the model first (option 1)")
                else:
                    self.display_model_info()
            elif choice == '5':
                print("Thank you for using the Heart Disease Prediction System!")
                break
            else:
                print("Invalid option. Please select 1-5.")


if __name__ == "__main__":
    # Create and run the interface
    interface = HeartDiseasePredictionInterface()
    interface.main_menu()