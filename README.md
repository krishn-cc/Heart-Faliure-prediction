# ❤️ Heart Disease Prediction System

A complete machine learning web application that predicts heart disease risk using a trained Logistic Regression model. Features a modern web interface with real-time predictions and medical-themed design.

## 🚀 Quick Start



## 📊 Dataset Description

The dataset contains **918 patient records** with 11 medical features:

### Input Features:
- **Age**: Patient age (20-100 years)
- **Sex**: Gender (Male/Female)
- **ChestPainType**: Type of chest pain (TA/ATA/NAP/ASY)
- **RestingBP**: Resting blood pressure (80-250 mm Hg)
- **Cholesterol**: Serum cholesterol (100-600 mg/dl)
- **FastingBS**: Fasting blood sugar >120 mg/dl (Yes/No)
- **RestingECG**: Resting ECG results (Normal/ST/LVH)
- **MaxHR**: Maximum heart rate (60-220 bpm)
- **ExerciseAngina**: Exercise-induced chest pain (Yes/No)
- **Oldpeak**: ST depression (-3 to 7)
- **ST_Slope**: ST segment slope (Up/Flat/Down)

### Target Variable:
- **HeartDisease**: 0 = No heart disease, 1 = Heart disease present

## 📁 Project Structure

```
AIML pbl/
├── heart.csv                           # Original dataset (918 patients)
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
│
├── data/                              # Processed datasets
│   ├── X_train.csv                   # Training features (734 patients)
│   ├── X_test.csv                    # Testing features (184 patients)
│   ├── y_train.csv                   # Training labels
│   └── y_test.csv                    # Testing labels
│
├── preprocessing/                     # Data preprocessing
│   └── data_preprocessing.py         # Complete preprocessing pipeline
│
├── training/                         # Model training
│   └── train_model.py               # Model training script
│
├── testing/                          # Model evaluation
│   └── model_testing.py             # Performance testing
│
├── models/                           # Trained models
│   ├── heart_disease_model.pkl      # Trained Logistic Regression (89.13% accuracy)
│   ├── scaler.pkl                   # Feature scaler
│   └── feature_names.pkl            # Feature names and order
│
└── web_interface/                    # Web application
    ├── app.py                       # Flask backend server
    ├── requirements.txt             # Web app dependencies
    └── templates/
        └── index.html               # Modern web interface
```



## 🔧 Data Preprocessing Pipeline

Our preprocessing pipeline handles the complete data transformation:

### 1. Missing Value Treatment
- **Cholesterol**: Fixed 172 impossible zero values → replaced with median (237 mg/dl)
- **RestingBP**: Fixed 1 impossible zero value → replaced with median (130 mm Hg)

### 2. Categorical Encoding
- **Binary Features**: Sex (M→1, F→0), ExerciseAngina (Y→1, N→0)
- **One-Hot Encoding**: 
  - ChestPainType → 4 binary columns (ATA, ASY, NAP, TA)
  - RestingECG → 3 binary columns (Normal, ST, LVH)
  - ST_Slope → 3 binary columns (Up, Flat, Down)

### 3. Feature Scaling
- **StandardScaler** applied to numerical features: Age, RestingBP, Cholesterol, MaxHR, Oldpeak
- **Result**: All features scaled to mean=0, std=1 for optimal model performance

### 4. Data Splitting
- **Training Set**: 734 patients (80%)
- **Testing Set**: 184 patients (20%)
- **Stratified sampling** maintains class distribution

## 🤖 Model Performance

### Trained Model: Logistic Regression
- **Algorithm**: Logistic Regression with liblinear solver
- **Training Data**: 734 patients
- **Test Data**: 184 patients

### Actual Results:
- ✅ **Accuracy**: 89.13% (164/184 correct predictions)
- ✅ **Precision**: 88.68% (89% of positive predictions correct)
- ✅ **Recall**: 92.16% (92% of actual heart disease cases found)
- ✅ **F1-Score**: 90.38% (balanced precision and recall)

### Risk Classification Thresholds:
- **Low Risk**: < 50% probability
- **Moderate Risk**: 50-65% probability
- **High Risk**: 65-80% probability
- **Very High Risk**: > 80% probability

## 🌐 Web Interface Features

### Modern Medical-Themed Design:
- **Red & White Theme**: Professional medical appearance
- **Responsive Design**: Works on desktop and mobile
- **Real-time Validation**: Instant feedback on input values
- **Animated Results**: Smooth probability bar animations

### Functionality:
- **11 Medical Inputs**: All required patient parameters
- **Instant Predictions**: Results in <100ms
- **Risk Assessment**: Color-coded risk levels
- **Medical Recommendations**: Actionable advice based on risk level
- **Input Validation**: Ensures medical value ranges


This will check:
- ✅ Python package imports
- ✅ Required file existence
- ✅ Model loading capability
- ✅ Web interface functionality

## 📝 Notes

- **Model Accuracy**: 89.13% on test data (184 patients)
- **Training Data**: 734 patients from 918 total records
- **Response Time**: Predictions typically complete in <100ms
- **Browser Support**: Works on all modern browsers
- **Mobile Friendly**: Responsive design for mobile devices

