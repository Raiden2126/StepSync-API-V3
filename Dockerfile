# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the model file first to ensure it exists
COPY difficulty_model.pkl .

# Copy the rest of the application
COPY . .

# Verify model file exists and is readable
RUN python -c "import joblib; model = joblib.load('difficulty_model.pkl'); print('Model loaded successfully:', model.keys() if isinstance(model, dict) else 'Model loaded')"

# Expose the port
EXPOSE ${PORT}

# Command to run the application with debug logging
CMD exec uvicorn backup:app --host 0.0.0.0 --port ${PORT} --log-level debug --reload
