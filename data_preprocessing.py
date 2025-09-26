"""
Heart Disease Dataset Preprocessing Script

This script handles all data preprocessing steps for the heart disease prediction model.
It includes data loading, cleaning, encoding, scaling, and splitting operations.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

class HeartDiseasePreprocessor:
    def __init__(self, file_path):
        """
        Initialize the preprocessor with the dataset file path.
        
        Args:
            file_path (str): Path to the heart disease CSV file
        """
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
        
        # Check for invalid values (0 in BP and Cholesterol)
        print("\nInvalid Values Check:")
        print(f"RestingBP = 0: {(self.data['RestingBP'] == 0).sum()} records")
        print(f"Cholesterol = 0: {(self.data['Cholesterol'] == 0).sum()} records")
        
        # Target variable distribution
        print("\nTarget Variable Distribution:")
        target_dist = self.data['HeartDisease'].value_counts()
        print(target_dist)
        print(f"Heart Disease Rate: {target_dist[1] / len(self.data) * 100:.2f}%")
        
        # Statistical summary
        print("\nStatistical Summary:")
        print(self.data.describe())
        
    def handle_missing_values(self):
        """Handle missing and invalid values."""
        print("\n" + "="*50)
        print("Step 3: Handling Missing/Invalid Values")
        print("="*50)
        
        # Handle Cholesterol = 0 (invalid values)
        cholesterol_median = self.data[self.data['Cholesterol'] > 0]['Cholesterol'].median()
        invalid_chol = (self.data['Cholesterol'] == 0).sum()
        self.data.loc[self.data['Cholesterol'] == 0, 'Cholesterol'] = cholesterol_median
        print(f"Replaced {invalid_chol} invalid Cholesterol values with median: {cholesterol_median}")
        
        # Handle RestingBP = 0 (invalid values)
        bp_median = self.data[self.data['RestingBP'] > 0]['RestingBP'].median()
        invalid_bp = (self.data['RestingBP'] == 0).sum()
        if invalid_bp > 0:
            self.data.loc[self.data['RestingBP'] == 0, 'RestingBP'] = bp_median
            print(f"Replaced {invalid_bp} invalid RestingBP values with median: {bp_median}")
        else:
            print("No invalid RestingBP values found")
            
        print("Missing value handling completed!")
        
    def encode_categorical_features(self):
        """Encode categorical features."""
        print("\n" + "="*50)
        print("Step 4: Categorical Feature Encoding")
        print("="*50)
        
        # Binary encoding for Sex
        self.data['Sex'] = self.data['Sex'].map({'M': 1, 'F': 0})
        print("Sex encoded: M=1, F=0")
        
        # Binary encoding for ExerciseAngina
        self.data['ExerciseAngina'] = self.data['ExerciseAngina'].map({'Y': 1, 'N': 0})
        print("ExerciseAngina encoded: Y=1, N=0")
        
        # One-hot encoding for ChestPainType
        chest_pain_dummies = pd.get_dummies(self.data['ChestPainType'], prefix='ChestPain')
        self.data = pd.concat([self.data, chest_pain_dummies], axis=1)
        self.data.drop('ChestPainType', axis=1, inplace=True)
        print(f"ChestPainType one-hot encoded: {list(chest_pain_dummies.columns)}")
        
        # One-hot encoding for RestingECG
        ecg_dummies = pd.get_dummies(self.data['RestingECG'], prefix='RestingECG')
        self.data = pd.concat([self.data, ecg_dummies], axis=1)
        self.data.drop('RestingECG', axis=1, inplace=True)
        print(f"RestingECG one-hot encoded: {list(ecg_dummies.columns)}")
        
        # One-hot encoding for ST_Slope
        slope_dummies = pd.get_dummies(self.data['ST_Slope'], prefix='ST_Slope')
        self.data = pd.concat([self.data, slope_dummies], axis=1)
        self.data.drop('ST_Slope', axis=1, inplace=True)
        print(f"ST_Slope one-hot encoded: {list(slope_dummies.columns)}")
        
        print(f"Final dataset shape after encoding: {self.data.shape}")
        print(f"All features: {list(self.data.columns)}")
        
    def split_data(self, test_size=0.2, random_state=42):
        """Split data into training and testing sets."""
        print("\n" + "="*50)
        print("Step 5: Data Splitting")
        print("="*50)
        
        # Separate features and target
        X = self.data.drop('HeartDisease', axis=1)
        y = self.data['HeartDisease']
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set size: {len(self.X_train)} samples")
        print(f"Testing set size: {len(self.X_test)} samples")
        print(f"Training set heart disease rate: {self.y_train.mean():.2%}")
        print(f"Testing set heart disease rate: {self.y_test.mean():.2%}")
        
    def scale_features(self):
        """Apply feature scaling to numerical features."""
        print("\n" + "="*50)
        print("Step 6: Feature Scaling")
        print("="*50)
        
        # Identify numerical features (excluding binary encoded features)
        numerical_features = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
        print(f"Scaling numerical features: {numerical_features}")
        
        # Fit scaler on training data and transform both training and testing data
        self.X_train[numerical_features] = self.scaler.fit_transform(self.X_train[numerical_features])
        self.X_test[numerical_features] = self.scaler.transform(self.X_test[numerical_features])
        
        print("Feature scaling completed!")
        print("Training set feature ranges after scaling:")
        for feature in numerical_features:
            print(f"  {feature}: [{self.X_train[feature].min():.2f}, {self.X_train[feature].max():.2f}]")
            
    def get_preprocessing_summary(self):
        """Generate a summary of preprocessing steps."""
        print("\n" + "="*70)
        print("PREPROCESSING SUMMARY")
        print("="*70)
        
        summary = {
            "Original dataset shape": f"{self.data.shape[0]} samples, {self.data.shape[1]-1} features",
            "Final dataset shape": f"{len(self.X_train) + len(self.X_test)} samples, {self.X_train.shape[1]} features",
            "Training samples": len(self.X_train),
            "Testing samples": len(self.X_test),
            "Feature scaling": "Applied to numerical features",
            "Categorical encoding": "One-hot encoding for multi-class, binary for binary features",
            "Missing values": "Handled by median imputation"
        }
        
        for key, value in summary.items():
            print(f"{key}: {value}")
            
    def visualize_data(self):
        """Create visualizations of the preprocessed data."""
        print("\n" + "="*50)
        print("Step 7: Data Visualization")
        print("="*50)
        
        plt.figure(figsize=(15, 10))
        
        # Target distribution
        plt.subplot(2, 3, 1)
        self.data['HeartDisease'].value_counts().plot(kind='bar')
        plt.title('Heart Disease Distribution')
        plt.ylabel('Count')
        
        # Age distribution by heart disease
        plt.subplot(2, 3, 2)
        plt.boxplot([self.data[self.data['HeartDisease']==0]['Age'], 
                     self.data[self.data['HeartDisease']==1]['Age']], 
                    labels=['No Disease', 'Disease'])
        plt.title('Age Distribution by Heart Disease')
        plt.ylabel('Age')
        
        # Cholesterol distribution by heart disease
        plt.subplot(2, 3, 3)
        plt.boxplot([self.data[self.data['HeartDisease']==0]['Cholesterol'], 
                     self.data[self.data['HeartDisease']==1]['Cholesterol']], 
                    labels=['No Disease', 'Disease'])
        plt.title('Cholesterol Distribution by Heart Disease')
        plt.ylabel('Cholesterol')
        
        # MaxHR distribution by heart disease
        plt.subplot(2, 3, 4)
        plt.boxplot([self.data[self.data['HeartDisease']==0]['MaxHR'], 
                     self.data[self.data['HeartDisease']==1]['MaxHR']], 
                    labels=['No Disease', 'Disease'])
        plt.title('Max Heart Rate Distribution by Heart Disease')
        plt.ylabel('Max Heart Rate')
        
        # Sex distribution by heart disease
        plt.subplot(2, 3, 5)
        sex_crosstab = pd.crosstab(self.data['Sex'], self.data['HeartDisease'])
        sex_crosstab.plot(kind='bar', ax=plt.gca())
        plt.title('Sex Distribution by Heart Disease')
        plt.ylabel('Count')
        plt.xticks([0, 1], ['Female', 'Male'], rotation=0)
        plt.legend(['No Disease', 'Disease'])
        
        # Correlation heatmap (top features)
        plt.subplot(2, 3, 6)
        correlation_features = ['Age', 'Sex', 'RestingBP', 'Cholesterol', 'MaxHR', 'HeartDisease']
        correlation_matrix = self.data[correlation_features].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        
        plt.tight_layout()
        plt.savefig('preprocessing_visualizations.png', dpi=300, bbox_inches='tight')
        # plt.show()  # Commented out to avoid GUI issues
        plt.close()  # Close the figure to free memory
        
        print("Visualizations saved as 'preprocessing_visualizations.png'")
        
    def preprocess_complete_pipeline(self):
        """Run the complete preprocessing pipeline."""
        print("HEART DISEASE DATA PREPROCESSING PIPELINE")
        print("=" * 60)
        
        # Run all preprocessing steps
        self.load_data()
        self.explore_data()
        self.handle_missing_values()
        self.encode_categorical_features()
        self.split_data()
        self.scale_features()
        self.get_preprocessing_summary()
        self.visualize_data()
        
        print("\n" + "="*60)
        print("PREPROCESSING COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return self.X_train, self.X_test, self.y_train, self.y_test


if __name__ == "__main__":
    # Run the preprocessing pipeline
    preprocessor = HeartDiseasePreprocessor("heart.csv")
    X_train, X_test, y_train, y_test = preprocessor.preprocess_complete_pipeline()
    
    # Save preprocessed data
    X_train.to_csv("X_train.csv", index=False)
    X_test.to_csv("X_test.csv", index=False)
    y_train.to_csv("y_train.csv", index=False)
    y_test.to_csv("y_test.csv", index=False)
    
    print("\nPreprocessed data saved as:")
    print("- X_train.csv")
    print("- X_test.csv") 
    print("- y_train.csv")
    print("- y_test.csv")