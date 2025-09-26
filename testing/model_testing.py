"""
Heart Disease Testing Script - Organized Version
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class HeartDiseaseModelTester:
    def __init__(self):
        """Initialize the model tester."""
        self.model = None
        self.scaler = None
        self.X_test = None
        self.y_test = None
        
    def load_model_and_data(self):
        """Load trained model and test data."""
        print("Loading model and test data...")
        
        try:
            # Load model and scaler
            self.model = joblib.load("../models/heart_disease_model.pkl")
            self.scaler = joblib.load("../models/scaler.pkl")
            
            # Load test data
            self.X_test = pd.read_csv("../data/X_test.csv")
            self.y_test = pd.read_csv("../data/y_test.csv").iloc[:, 0]
            
            print("✅ Model and data loaded successfully!")
            return True
        except FileNotFoundError as e:
            print(f"❌ Error loading files: {e}")
            return False
            
    def test_model_performance(self):
        """Test model performance on test set."""
        print("\n🧪 Testing Model Performance...")
        
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        
        print(f"Test Accuracy: {accuracy:.4f} ({accuracy:.2%})")
        print("\nDetailed Classification Report:")
        print(classification_report(self.y_test, y_pred, 
                                  target_names=['No Disease', 'Heart Disease']))
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        return {
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'accuracy': accuracy
        }
        
    def test_single_prediction(self, patient_data):
        """Test prediction for a single patient."""
        # This would require the same preprocessing pipeline
        # For now, we'll return a placeholder
        return {
            'prediction': 1,
            'probability': 0.75,
            'risk_level': 'High Risk'
        }
        
    def run_all_tests(self):
        """Run all testing procedures."""
        print("HEART DISEASE MODEL TESTING")
        print("=" * 40)
        
        if not self.load_model_and_data():
            return None
            
        results = self.test_model_performance()
        
        print("\n🎉 TESTING COMPLETED!")
        return results

if __name__ == "__main__":
    tester = HeartDiseaseModelTester()
    results = tester.run_all_tests()