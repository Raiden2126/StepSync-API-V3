#!/bin/bash

# Print environment information for debugging
echo "Current directory: $(pwd)"
echo "Directory contents:"
ls -la

echo "Python version:"
python --version

echo "Environment variables:"
env | sort

# Set default port if not provided
PORT=8000
echo "Using port: $PORT"

# Verify model file
echo "Verifying model file..."
if [ -f "difficulty_model.pkl" ]; then
    echo "Model file exists, size: $(stat -f%z difficulty_model.pkl || stat -c%s difficulty_model.pkl) bytes"
    python -c "import joblib; model = joblib.load('difficulty_model.pkl'); print('Model loaded successfully')"
else
    echo "ERROR: Model file not found!"
    exit 1
fi

# Start the application with explicit port number
echo "Starting application on port $PORT..."
exec uvicorn backup:app --host 0.0.0.0 --port "$PORT" --log-level debug --reload-dir /app 