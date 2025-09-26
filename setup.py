#!/usr/bin/env python3
"""
Quick setup script for GitHub deployment
Run this after cloning the repository
"""

import os
import subprocess
import sys

def install_requirements():
    """Install required Python packages"""
    print("📦 Installing required packages...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install packages. Please install manually:")
        print("pip install flask flask-cors pandas numpy scikit-learn joblib")
        return False

def verify_files():
    """Verify all required files exist"""
    print("\n📁 Verifying project files...")
    
    required_files = [
        "heart.csv",
        "web_interface/app.py", 
        "web_interface/templates/index.html",
        "models/heart_disease_model.pkl",
        "models/scaler.pkl"
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            missing_files.append(file_path)
    
    return len(missing_files) == 0

def main():
    """Main setup function"""
    print("🩺 HEART DISEASE PREDICTION SYSTEM - SETUP")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("heart.csv"):
        print("❌ Please run this script from the project root directory")
        return False
    
    # Install requirements
    if not install_requirements():
        return False
    
    # Verify files
    if not verify_files():
        print("\n❌ Some required files are missing!")
        return False
    
    print("\n🎉 Setup complete! Ready to run:")
    print("   cd web_interface")
    print("   python app.py")
    print("   Then open: http://localhost:5000")
    
    return True

if __name__ == "__main__":
    main()