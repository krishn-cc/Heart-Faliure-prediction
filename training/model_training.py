"""
Heart Disease Prediction Model Training Script

This script trains a logistic regression model on the preprocessed heart disease dataset
and evaluates its performance using various metrics.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from data_preprocessing import HeartDiseasePreprocessor

class HeartDiseaseModelTrainer:
    def __init__(self):
        """Initialize the model trainer."""
        self.model = LogisticRegression(random_state=42, max_iter=1000, solver='liblinear')
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.y_pred_proba = None
        
    def load_preprocessed_data(self, X_train_path=None, X_test_path=None, 
                              y_train_path=None, y_test_path=None):
        """
        Load preprocessed data from CSV files or run preprocessing pipeline.
        
        Args:
            X_train_path (str): Path to training features CSV
            X_test_path (str): Path to testing features CSV  
            y_train_path (str): Path to training target CSV
            y_test_path (str): Path to testing target CSV
        """
        print("Loading Preprocessed Data")
        print("-" * 30)
        
        try:
            if all([X_train_path, X_test_path, y_train_path, y_test_path]):
                # Load from CSV files
                self.X_train = pd.read_csv(X_train_path)
                self.X_test = pd.read_csv(X_test_path)
                self.y_train = pd.read_csv(y_train_path).iloc[:, 0]  # First column
                self.y_test = pd.read_csv(y_test_path).iloc[:, 0]    # First column
                print("Data loaded from CSV files successfully!")
            else:
                raise FileNotFoundError("CSV files not found")
                
        except FileNotFoundError:
            print("Preprocessed CSV files not found. Running preprocessing pipeline...")
            # Run preprocessing pipeline
            preprocessor = HeartDiseasePreprocessor("heart.csv")
            self.X_train, self.X_test, self.y_train, self.y_test = preprocessor.preprocess_complete_pipeline()
            
        print(f"Training data shape: {self.X_train.shape}")
        print(f"Testing data shape: {self.X_test.shape}")
        print(f"Training target shape: {self.y_train.shape}")
        print(f"Testing target shape: {self.y_test.shape}")
        
    def train_model(self):
        """Train the logistic regression model."""
        print("\n" + "="*50)
        print("MODEL TRAINING")
        print("="*50)
        
        print("Training Logistic Regression Model...")
        print(f"Model parameters: {self.model.get_params()}")
        
        # Train the model
        self.model.fit(self.X_train, self.y_train)
        
        print("Model training completed!")
        
        # Display feature importance (coefficients)
        feature_names = self.X_train.columns
        coefficients = self.model.coef_[0]
        
        # Create feature importance dataframe
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coefficients,
            'Abs_Coefficient': np.abs(coefficients)
        }).sort_values('Abs_Coefficient', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
        
        return self.model
        
    def make_predictions(self):
        """Make predictions on the test set."""
        print("\n" + "="*50)
        print("MAKING PREDICTIONS")
        print("="*50)
        
        # Make predictions
        self.y_pred = self.model.predict(self.X_test)
        self.y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]  # Probability of positive class
        
        print("Predictions completed!")
        print(f"Predictions shape: {self.y_pred.shape}")
        print(f"Probability predictions shape: {self.y_pred_proba.shape}")
        
        # Show sample predictions
        print("\nSample Predictions:")
        prediction_sample = pd.DataFrame({
            'Actual': self.y_test.iloc[:10].values,
            'Predicted': self.y_pred[:10],
            'Probability': self.y_pred_proba[:10]
        })
        print(prediction_sample)
        
    def evaluate_model(self):
        """Evaluate the model performance."""
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, self.y_pred)
        precision = precision_score(self.y_test, self.y_pred)
        recall = recall_score(self.y_test, self.y_pred)
        f1 = f1_score(self.y_test, self.y_pred)
        
        # Display metrics
        print("Classification Metrics:")
        print(f"Accuracy:  {accuracy:.4f} ({accuracy:.2%})")
        print(f"Precision: {precision:.4f} ({precision:.2%})")
        print(f"Recall:    {recall:.4f} ({recall:.2%})")
        print(f"F1-Score:  {f1:.4f}")
        
        # Detailed classification report
        print("\nDetailed Classification Report:")
        print(classification_report(self.y_test, self.y_pred, 
                                  target_names=['No Disease', 'Heart Disease']))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, self.y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm
        }
        
    def visualize_results(self):
        """Create visualizations of model results."""
        print("\n" + "="*50)
        print("CREATING VISUALIZATIONS")
        print("="*50)
        
        plt.figure(figsize=(15, 10))
        
        # Confusion Matrix Heatmap
        plt.subplot(2, 3, 1)
        cm = confusion_matrix(self.y_test, self.y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Disease', 'Heart Disease'],
                   yticklabels=['No Disease', 'Heart Disease'])
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')  
        plt.xlabel('Predicted')
        
        # Feature Importance
        plt.subplot(2, 3, 2)
        feature_names = self.X_train.columns
        coefficients = self.model.coef_[0]
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coefficients
        }).sort_values('Coefficient', key=abs, ascending=False).head(10)
        
        plt.barh(range(len(feature_importance)), feature_importance['Coefficient'])
        plt.yticks(range(len(feature_importance)), feature_importance['Feature'])
        plt.title('Top 10 Feature Importance (Coefficients)')
        plt.xlabel('Coefficient Value')
        
        # Prediction Probability Distribution
        plt.subplot(2, 3, 3)
        plt.hist(self.y_pred_proba[self.y_test == 0], alpha=0.5, label='No Disease', bins=20)
        plt.hist(self.y_pred_proba[self.y_test == 1], alpha=0.5, label='Heart Disease', bins=20)
        plt.xlabel('Predicted Probability')
        plt.ylabel('Frequency')
        plt.title('Prediction Probability Distribution')
        plt.legend()
        
        # Model Performance Metrics
        plt.subplot(2, 3, 4)
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = [
            accuracy_score(self.y_test, self.y_pred),
            precision_score(self.y_test, self.y_pred),
            recall_score(self.y_test, self.y_pred),
            f1_score(self.y_test, self.y_pred)
        ]
        plt.bar(metrics, values)
        plt.title('Model Performance Metrics')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        for i, v in enumerate(values):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
            
        # ROC-like visualization (Prediction vs Actual)
        plt.subplot(2, 3, 5)
        plt.scatter(range(len(self.y_test)), self.y_test, alpha=0.5, label='Actual', s=20)
        plt.scatter(range(len(self.y_pred)), self.y_pred, alpha=0.5, label='Predicted', s=20)
        plt.xlabel('Sample Index')
        plt.ylabel('Class')
        plt.title('Actual vs Predicted Values')
        plt.legend()
        plt.yticks([0, 1], ['No Disease', 'Heart Disease'])
        
        # Residuals (for probability predictions)
        plt.subplot(2, 3, 6)
        residuals = self.y_test - self.y_pred_proba
        plt.scatter(self.y_pred_proba, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Residuals (Actual - Predicted)')
        plt.title('Residual Plot')
        
        plt.tight_layout()
        plt.savefig('model_evaluation_results.png', dpi=300, bbox_inches='tight')
        # plt.show()  # Commented out to avoid GUI issues
        plt.close()  # Close the figure to free memory
        
        print("Visualizations saved as 'model_evaluation_results.png'")
        
    def save_model(self, model_path='heart_disease_model.pkl'):
        """Save the trained model to disk."""
        print(f"\nSaving model to {model_path}...")
        joblib.dump(self.model, model_path)
        print("Model saved successfully!")
        
    def load_model(self, model_path='heart_disease_model.pkl'):
        """Load a trained model from disk."""
        print(f"Loading model from {model_path}...")
        self.model = joblib.load(model_path)
        print("Model loaded successfully!")
        
    def complete_training_pipeline(self):
        """Run the complete model training pipeline."""
        print("HEART DISEASE MODEL TRAINING PIPELINE")
        print("=" * 60)
        
        # Load data
        self.load_preprocessed_data()
        
        # Train model
        self.train_model()
        
        # Make predictions
        self.make_predictions()
        
        # Evaluate model
        metrics = self.evaluate_model()
        
        # Create visualizations
        self.visualize_results()
        
        # Save model
        self.save_model()
        
        print("\n" + "="*60)
        print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return self.model, metrics


if __name__ == "__main__":
    # Run the complete training pipeline
    trainer = HeartDiseaseModelTrainer()
    model, metrics = trainer.complete_training_pipeline()
    
    print(f"\nFinal Model Performance:")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Precision: {metrics['precision']:.2%}")
    print(f"Recall: {metrics['recall']:.2%}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")