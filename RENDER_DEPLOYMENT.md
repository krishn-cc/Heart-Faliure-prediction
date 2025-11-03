# Deployment Guide for Render

## ✅ Your app is ready for Render deployment!

### 📁 Project Structure (Render-Ready)
```
Heart-Faliure-prediction/
├── app.py                    # ✅ Main Flask app in root
├── templates/
│   └── index.html           # ✅ Frontend template
├── models/
│   ├── heart_disease_model.pkl   # ✅ ML model
│   ├── scaler.pkl           # ✅ Feature scaler
│   └── feature_names.pkl    # Model features
├── requirements.txt          # ✅ Production dependencies
├── .gitignore               # ✅ Git ignore file
└── README.md                # Documentation
```

## 🚀 Deploy to Render (Step-by-Step)

### Step 1: Push to GitHub
```bash
git add .
git commit -m "Ready for Render deployment"
git push origin main
```

### Step 2: Create Render Web Service
1. Go to [https://render.com](https://render.com)
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repository: `Heart-Faliure-prediction`

### Step 3: Configure Render Settings
Fill in these settings:

- **Name**: `heart-disease-prediction`
- **Region**: Choose nearest (e.g., Oregon, Frankfurt)
- **Branch**: `main`
- **Root Directory**: (leave blank)
- **Environment**: `Python 3`
- **Build Command**: 
  ```
  pip install -r requirements.txt
  ```
- **Start Command**: 
  ```
  gunicorn app:app
  ```
- **Instance Type**: `Free`

### Step 4: Environment Variables (Optional)
No environment variables needed for basic deployment.

### Step 5: Deploy!
Click **"Create Web Service"**

Render will:
1. Clone your repository
2. Install dependencies
3. Start your application
4. Provide you with a URL like: `https://heart-disease-prediction-XXXX.onrender.com`

## 🎯 Your App Will Be Live At:
```
https://your-app-name.onrender.com
```

## ✅ What's Included:
- ✅ Flask web server with gunicorn
- ✅ Trained ML model (89.13% accuracy)
- ✅ Modern red/white medical theme
- ✅ Real-time predictions
- ✅ Mobile-responsive design
- ✅ Health check endpoint at `/health`

## ⚠️ Important Notes:
- **Free tier**: App sleeps after 15 minutes of inactivity
- **First request**: May take 30-60 seconds to wake up
- **Cold start**: Normal for free tier
- **Memory**: 512MB RAM (sufficient for this app)

## 🔧 Troubleshooting:

### If Build Fails:
1. Check logs in Render dashboard
2. Ensure all files are committed to GitHub
3. Verify `models/` folder contains `.pkl` files

### If App Crashes:
1. Check Render logs
2. Verify Python version compatibility
3. Test locally first: `python app.py`

## 📊 Test Your Deployment:
Once deployed, test with sample data:
- Age: 45
- Sex: Male
- Chest Pain: ASY
- BP: 140
- Cholesterol: 280
etc.

## 🎉 You're Ready!
Your heart disease prediction system is production-ready for Render!
