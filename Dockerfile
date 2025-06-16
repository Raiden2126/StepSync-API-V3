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

# Copy the verification script first
COPY verify_model.py .

# Copy the model file
COPY difficulty_model.pkl .

# Verify model file exists
RUN ls -l difficulty_model.pkl || echo "Model file not found"

# Copy the rest of the application
COPY . .

# Run the verification script
RUN python verify_model.py

# Expose the port (this is just documentation, actual port is set at runtime)
EXPOSE 8000

# Use the specified uvicorn command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
