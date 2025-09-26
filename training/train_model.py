"""
Heart Disease Model Training Script - Organized Version
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class HeartDiseaseModelTrainer:
    def __init__(self):
        """Initialize the model trainer."""
        self.model = LogisticRegression(random_state=42, max_iter=1000, solver='liblinear')
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self):
        """Load preprocessed data."""
        print("Loading preprocessed data...")
        
        try:
            self.X_train = pd.read_csv("../data/X_train.csv")
            self.X_test = pd.read_csv("../data/X_test.csv")
            self.y_train = pd.read_csv("../data/y_train.csv").iloc[:, 0]
            self.y_test = pd.read_csv("../data/y_test.csv").iloc[:, 0]
            print("✅ Data loaded successfully!")
        except FileNotFoundError:
            print("❌ Preprocessed data not found. Please run preprocessing first.")
            return False
        return True
        
    def train_model(self):
        """Train the model."""
        print("\n🤖 Training Logistic Regression Model...")
        
        self.model.fit(self.X_train, self.y_train)
        
        # Save the model
        joblib.dump(self.model, "../models/heart_disease_model.pkl")
        
        print("✅ Model training completed and saved!")
        
    def evaluate_model(self):
        """Evaluate the model."""
        print("\n📊 Evaluating Model Performance...")
        
        y_pred = self.model.predict(self.X_test)
        
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        
        print(f"Accuracy:  {accuracy:.4f} ({accuracy:.2%})")
        print(f"Precision: {precision:.4f} ({precision:.2%})")
        print(f"Recall:    {recall:.4f} ({recall:.2%})")
        print(f"F1-Score:  {f1:.4f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
    def complete_training_pipeline(self):
        """Complete training pipeline."""
        print("HEART DISEASE MODEL TRAINING")
        print("=" * 40)
        
        if not self.load_data():
            return None, None
            
        self.train_model()
        metrics = self.evaluate_model()
        
        print("\n🎉 TRAINING PIPELINE COMPLETED!")
        return self.model, metrics

if __name__ == "__main__":
    trainer = HeartDiseaseModelTrainer()
    model, metrics = trainer.complete_training_pipeline()