# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    LOG_LEVEL=debug

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create a directory for logs
RUN mkdir -p /app/logs

# Copy the model file first to ensure it exists
COPY difficulty_model.pkl .

# Verify model file exists
RUN ls -l difficulty_model.pkl || echo "Model file not found"

# Copy the rest of the application
COPY . .

# Make the start script executable
RUN chmod +x start.sh

# Create a simple Python script to verify the model
RUN echo 'import os, sys, joblib; \
    try: \
        print("Current directory:", os.getcwd()); \
        print("Files in directory:", os.listdir(".")); \
        if not os.path.exists("difficulty_model.pkl"): \
            print("Model file not found!"); \
            sys.exit(1); \
        print("Model file size:", os.path.getsize("difficulty_model.pkl")); \
        try: \
            model = joblib.load("difficulty_model.pkl"); \
            print("Model loaded successfully"); \
            print("Model type:", type(model)); \
            if isinstance(model, dict): \
                print("Model keys:", list(model.keys())); \
                print("Model components:", {k: str(type(v)) for k, v in model.items()}); \
        except Exception as e: \
            print("Error loading model:", str(e)); \
            sys.exit(1); \
    except Exception as e: \
        print("Error during verification:", str(e)); \
        sys.exit(1);' > verify_model.py

# Run the verification script
RUN python verify_model.py

# Expose the port (this is just documentation, actual port is set at runtime)
EXPOSE 8000

# Use the shell script to start the application
CMD ["./start.sh"]
