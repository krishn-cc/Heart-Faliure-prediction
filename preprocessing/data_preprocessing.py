"""
Heart Disease Dataset Preprocessing Script - Organized Version

This script handles all data preprocessing steps for the heart disease prediction model.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class HeartDiseasePreprocessor:
    def __init__(self, file_path="../data/heart.csv"):
        """Initialize the preprocessor with the dataset file path."""
        self.file_path = file_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load the dataset from CSV file."""
        print("Step 1: Loading Data")
        print("-" * 30)
        
        self.data = pd.read_csv(self.file_path)
        print(f"Dataset loaded successfully!")
        print(f"Shape: {self.data.shape}")
        print(f"Columns: {list(self.data.columns)}")
        print("\nFirst 5 rows:")
        print(self.data.head())
        
        return self.data
    
    def explore_data(self):
        """Perform initial data exploration."""
        print("\n" + "="*50)
        print("Step 2: Data Exploration")
        print("="*50)
        
        # Basic information
        print("\nDataset Info:")
        print(f"Number of samples: {len(self.data)}")
        print(f"Number of features: {len(self.data.columns) - 1}")
        
        # Check for missing values
        print("\nMissing Values:")
        missing_values = self.data.isnull().sum()
        print(missing_values)
        
        # Check for invalid values
        print("\nInvalid Values Check:")
        print(f"RestingBP = 0: {(self.data['RestingBP'] == 0).sum()} records")
        print(f"Cholesterol = 0: {(self.data['Cholesterol'] == 0).sum()} records")
        
        # Target variable distribution
        print("\nTarget Variable Distribution:")
        target_dist = self.data['HeartDisease'].value_counts()
        print(target_dist)
        print(f"Heart Disease Rate: {target_dist[1] / len(self.data) * 100:.2f}%")
        
    def handle_missing_values(self):
        """Handle missing and invalid values."""
        print("\n" + "="*50)
        print("Step 3: Handling Missing/Invalid Values")
        print("="*50)
        
        # Handle Cholesterol = 0
        cholesterol_median = self.data[self.data['Cholesterol'] > 0]['Cholesterol'].median()
        invalid_chol = (self.data['Cholesterol'] == 0).sum()
        self.data.loc[self.data['Cholesterol'] == 0, 'Cholesterol'] = cholesterol_median
        print(f"Replaced {invalid_chol} invalid Cholesterol values with median: {cholesterol_median}")
        
        # Handle RestingBP = 0
        bp_median = self.data[self.data['RestingBP'] > 0]['RestingBP'].median()
        invalid_bp = (self.data['RestingBP'] == 0).sum()
        if invalid_bp > 0:
            self.data.loc[self.data['RestingBP'] == 0, 'RestingBP'] = bp_median
            print(f"Replaced {invalid_bp} invalid RestingBP values with median: {bp_median}")
        else:
            print("No invalid RestingBP values found")
            
    def encode_categorical_features(self):
        """Encode categorical features."""
        print("\n" + "="*50)
        print("Step 4: Categorical Feature Encoding")
        print("="*50)
        
        # Binary encoding
        self.data['Sex'] = self.data['Sex'].map({'M': 1, 'F': 0})
        self.data['ExerciseAngina'] = self.data['ExerciseAngina'].map({'Y': 1, 'N': 0})
        
        # One-hot encoding
        chest_pain_dummies = pd.get_dummies(self.data['ChestPainType'], prefix='ChestPain')
        self.data = pd.concat([self.data, chest_pain_dummies], axis=1)
        self.data.drop('ChestPainType', axis=1, inplace=True)
        
        ecg_dummies = pd.get_dummies(self.data['RestingECG'], prefix='RestingECG')
        self.data = pd.concat([self.data, ecg_dummies], axis=1)
        self.data.drop('RestingECG', axis=1, inplace=True)
        
        slope_dummies = pd.get_dummies(self.data['ST_Slope'], prefix='ST_Slope')
        self.data = pd.concat([self.data, slope_dummies], axis=1)
        self.data.drop('ST_Slope', axis=1, inplace=True)
        
        print("Categorical encoding completed!")
        
    def split_data(self, test_size=0.2, random_state=42):
        """Split data into training and testing sets."""
        print("\n" + "="*50)
        print("Step 5: Data Splitting")
        print("="*50)
        
        X = self.data.drop('HeartDisease', axis=1)
        y = self.data['HeartDisease']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set: {len(self.X_train)} samples")
        print(f"Testing set: {len(self.X_test)} samples")
        
    def scale_features(self):
        """Apply feature scaling."""
        print("\n" + "="*50)
        print("Step 6: Feature Scaling")
        print("="*50)
        
        numerical_features = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
        
        self.X_train[numerical_features] = self.scaler.fit_transform(self.X_train[numerical_features])
        self.X_test[numerical_features] = self.scaler.transform(self.X_test[numerical_features])
        
        print("Feature scaling completed!")
        
    def save_preprocessed_data(self):
        """Save preprocessed data."""
        self.X_train.to_csv("../data/X_train.csv", index=False)
        self.X_test.to_csv("../data/X_test.csv", index=False)
        self.y_train.to_csv("../data/y_train.csv", index=False)
        self.y_test.to_csv("../data/y_test.csv", index=False)
        
        # Save scaler
        import joblib
        joblib.dump(self.scaler, "../models/scaler.pkl")
            
    def preprocess_complete_pipeline(self):
        """Run the complete preprocessing pipeline."""
        print("HEART DISEASE DATA PREPROCESSING")
        print("=" * 50)
        
        self.load_data()
        self.explore_data()
        self.handle_missing_values()
        self.encode_categorical_features()
        self.split_data()
        self.scale_features()
        self.save_preprocessed_data()
        
        print("\n✅ PREPROCESSING COMPLETED!")
        return self.X_train, self.X_test, self.y_train, self.y_test

if __name__ == "__main__":
    preprocessor = HeartDiseasePreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.preprocess_complete_pipeline()