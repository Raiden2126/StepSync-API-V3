#!/bin/sh

# Get port from environment variable or use default
PORT=${PORT:-8000}

# Start the application
exec uvicorn backup:app --host 0.0.0.0 --port $PORT --log-level debug 