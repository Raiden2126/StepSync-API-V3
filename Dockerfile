# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
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

# Make the start script executable
RUN chmod +x start.sh

# Verify model file exists and is readable
RUN python -c "import joblib; model = joblib.load('difficulty_model.pkl'); print('Model loaded successfully:', model.keys() if isinstance(model, dict) else 'Model loaded')"

# Expose the port (this is just documentation, actual port is set at runtime)
EXPOSE 8000

# Use the shell script to start the application
CMD ["./start.sh"]
