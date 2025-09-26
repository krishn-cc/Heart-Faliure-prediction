#!/usr/bin/env python3
"""
System Test Script for Heart Disease Prediction System

This script verifies that all essential components are working correctly.
"""

import os
import sys

def test_imports():
    """Test if all required imports work"""
    print("🔍 Testing Python imports...")
    
    try:
        import pandas as pd
        print("✅ pandas imported successfully")
        
        import numpy as np
        print("✅ numpy imported successfully")
        
        import sklearn
        print("✅ scikit-learn imported successfully")
        
        import joblib
        print("✅ joblib imported successfully")
        
        import flask
        print("✅ flask imported successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_files():
    """Test if all required files exist"""
    print("\n📁 Testing file existence...")
    
    required_files = [
        "heart.csv",
        "web_interface/app.py",
        "web_interface/templates/index.html",
        "models/heart_disease_model.pkl",
        "models/scaler.pkl",
        "preprocessing/data_preprocessing.py",
        "training/train_model.py"
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - NOT FOUND")
            all_exist = False
    
    return all_exist

def test_model_loading():
    """Test if the trained model can be loaded"""
    print("\n🤖 Testing model loading...")
    
    try:
        import joblib
        
        # Test model loading
        model = joblib.load("models/heart_disease_model.pkl")
        print("✅ Trained model loaded successfully")
        
        # Test scaler loading
        scaler = joblib.load("models/scaler.pkl")
        print("✅ Feature scaler loaded successfully")
        
        return True
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        return False

def test_web_interface():
    """Test if the web interface can be imported"""
    print("\n🌐 Testing web interface...")
    
    try:
        sys.path.append('web_interface')
        import app
        print("✅ Flask web interface imported successfully")
        return True
    except Exception as e:
        print(f"❌ Web interface error: {e}")
        return False

def main():
    """Run all system tests"""
    print("🩺 HEART DISEASE PREDICTION SYSTEM - SYSTEM TEST")
    print("=" * 60)
    
    tests = [
        ("Python Imports", test_imports),
        ("Required Files", test_files),
        ("Model Loading", test_model_loading),
        ("Web Interface", test_web_interface)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
    
    print(f"\n📊 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! System is ready to run.")
        print("\n🚀 To start the web application:")
        print("   cd web_interface")
        print("   python app.py")
        print("   Then open: http://localhost:5000")
    else:
        print("⚠️  Some tests failed. Please check the requirements.")
        
    return passed == total

if __name__ == "__main__":
    main()