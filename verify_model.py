import os
import sys
import joblib
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def verify_model():
    """Verify that the model file exists and can be loaded."""
    try:
        logger.info(f"Current directory: {os.getcwd()}")
        logger.info(f"Files in directory: {os.listdir('.')}")
        
        model_path = "difficulty_model.pkl"
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at: {model_path}")
            return False
            
        logger.info(f"Model file size: {os.path.getsize(model_path)} bytes")
        
        try:
            model = joblib.load(model_path)
            logger.info("Model loaded successfully")
            logger.info(f"Model type: {type(model)}")
            
            if isinstance(model, dict):
                logger.info("Model keys: " + ", ".join(model.keys()))
                for key, value in model.items():
                    logger.info(f"{key}: {type(value)}")
                    if key == 'health_score_stats':
                        logger.info(f"Stats: {value}")
                    elif key in ['easy_threshold', 'medium_threshold']:
                        logger.info(f"Value: {value}")
                
                # Verify required components
                required_keys = ['easy_threshold', 'medium_threshold', 'health_score_stats']
                missing_keys = [key for key in required_keys if key not in model]
                if missing_keys:
                    logger.error(f"Model missing required components: {missing_keys}")
                    return False
            else:
                logger.error(f"Model is not a dictionary as expected. Got type: {type(model)}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}", exc_info=True)
            return False
            
    except Exception as e:
        logger.error(f"Error during verification: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    logger.info("Starting model verification...")
    if not verify_model():
        logger.error("Model verification failed!")
        sys.exit(1)
    logger.info("Model verification completed successfully!") 