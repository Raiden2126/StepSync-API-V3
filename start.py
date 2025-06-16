import os
import sys
import logging
import uvicorn
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_model():
    """Verify the model file exists and can be loaded."""
    try:
        model_path = "difficulty_model.pkl"
        logger.info(f"Current directory: {os.getcwd()}")
        logger.info(f"Directory contents: {os.listdir('.')}")
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at: {model_path}")
            return False
            
        file_size = os.path.getsize(model_path)
        logger.info(f"Model file size: {file_size} bytes")
        
        model = joblib.load(model_path)
        logger.info("Model loaded successfully")
        logger.info(f"Model type: {type(model)}")
        if isinstance(model, dict):
            logger.info(f"Model keys: {list(model.keys())}")
        return True
        
    except Exception as e:
        logger.error(f"Error verifying model: {str(e)}")
        return False

def main():
    """Start the FastAPI application with proper port configuration."""
    # Get port from environment variable or use default
    try:
        port = int(os.getenv('PORT', '8000'))
        logger.info(f"Using port: {port}")
    except ValueError as e:
        logger.error(f"Invalid PORT value: {e}")
        port = 8000
        logger.info(f"Using default port: {port}")
    
    # Verify model before starting server
    if not verify_model():
        logger.error("Model verification failed. Exiting...")
        sys.exit(1)
    
    # Start the server
    logger.info(f"Starting server on port {port}...")
    uvicorn.run(
        "backup:app",
        host="0.0.0.0",
        port=port,
        log_level="debug",
        reload_dir="/app"
    )

if __name__ == "__main__":
    main() 