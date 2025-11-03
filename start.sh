#!/bin/bash
# Render startup script

echo "🚀 Starting Heart Disease Prediction System on Render..."
echo "📦 Installing dependencies..."

# Start the application with gunicorn
exec gunicorn app:app \
    --bind 0.0.0.0:$PORT \
    --workers 2 \
    --timeout 120 \
    --access-logfile - \
    --error-logfile -
