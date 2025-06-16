#!/bin/sh

# Print environment information
echo "Starting application..."
echo "Current directory: $(pwd)"
echo "Directory contents: $(ls -la)"
echo "Python version: $(python --version)"
echo "Environment variables:"
env | sort

# Get port from environment variable or use default
PORT=${PORT:-8000}
echo "Using port: $PORT"

# Verify model file
echo "Verifying model file..."
python -c "import os, joblib; \
    print('Model file exists:', os.path.exists('difficulty_model.pkl')); \
    print('Model file size:', os.path.getsize('difficulty_model.pkl')); \
    model = joblib.load('difficulty_model.pkl'); \
    print('Model loaded successfully')"

# Start the application with debug logging
echo "Starting uvicorn server..."
exec uvicorn backup:app --host 0.0.0.0 --port $PORT --log-level debug --reload-dir /app 