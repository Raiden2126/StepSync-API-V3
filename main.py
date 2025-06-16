import logging
import numpy as np
import pandas as pd
import joblib
import os
import uvicorn
from typing import Optional, Dict, Any, Union
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator, field_validator, ValidationInfo
from fastapi.exceptions import RequestValidationError

# ----------------- Logging Setup -----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ----------------- FastAPI App -------------------
app = FastAPI(
    title="StepSync API",
    description="API for predicting workout intensity based on user metrics using Health Score Model",
    version="3.0.0"
)

# Enhanced CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your app's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# ----------------- Request Schema ----------------
class UserInput(BaseModel):
    # More flexible field names with aliases for JavaScript compatibility
    age: float = Field(..., description="User's age in years", alias="Age")
    bmi: float = Field(..., description="User's BMI", alias="BMI")
    workout_frequency: float = Field(..., ge=0, le=7, description="Workout frequency in days per week", alias="Workout_Frequency")

    # Add validators for reasonable ranges without strict limits
    @field_validator('age', 'bmi')
    @classmethod
    def validate_positive(cls, v: float, info: ValidationInfo) -> float:
        if v <= 0:
            raise ValueError(f"{info.field_name} must be greater than 0")
        return v

    model_config = {
        # Allow both camelCase and snake_case
        'populate_by_name': True,
        # Reject extra fields
        'extra': 'forbid',
        # Example: {"age": 25} or {"Age": 25} both work
        'json_schema_extra': {
            "example": {
                "age": 25,
                "bmi": 22.5,
                "workout_frequency": 3
            }
        }
    }

class PredictionResponse(BaseModel):
    difficulty_level: str = Field(..., alias="difficultyLevel")
    confidence_score: float = Field(..., alias="confidenceScore")
    recommendation: str
    health_score: float = Field(..., alias="healthScore")
    debug_info: Optional[Dict[str, Any]] = Field(None, alias="debugInfo")

    model_config = {
        # This ensures responses use the alias names (camelCase)
        'populate_by_name': True,
        'by_alias': True,
        # Convert snake_case to camelCase in response
        'json_encoders': {
            float: lambda v: round(v, 3)  # Round floats to 3 decimal places
        },
        # Example response
        'json_schema_extra': {
            "example": {
                "difficultyLevel": "Medium",
                "confidenceScore": 0.85,
                "recommendation": "You can handle moderate intensity workouts...",
                "healthScore": 0.65,
                "debugInfo": {
                    "inputData": {"age": 25, "bmi": 22.5, "workoutFrequency": 3},
                    "healthScore": 0.65,
                    "thresholds": {"easyThreshold": 0.57, "mediumThreshold": 0.73}
                }
            }
        }
    }

# ----------------- Model Handler -----------------
class StepSyncModel:
    def __init__(self):
        self.model_components: Optional[Dict[str, Any]] = None
        self.feature_names = ["age", "bmi", "workout_frequency"]
        self._load_model_and_assets()

    def _load_model_and_assets(self) -> None:
        """Load the model components and thresholds."""
        try:
            # Load model components
            model_path = "difficulty_model.pkl"
            logger.info(f"Attempting to load model from: {model_path}")
            
            if not os.path.exists(model_path):
                error_msg = f"Model file not found at path: {model_path}"
                logger.error(error_msg)
                logger.error(f"Current working directory: {os.getcwd()}")
                logger.error(f"Directory contents: {os.listdir('.')}")
                raise FileNotFoundError(error_msg)
            
            try:
                self.model_components = joblib.load(model_path)
                logger.info(f"Model components loaded successfully: {self.model_components.keys() if isinstance(self.model_components, dict) else 'Model loaded'}")
                
                # Verify required components
                required_keys = ['easy_threshold', 'medium_threshold', 'health_score_stats']
                if isinstance(self.model_components, dict):
                    missing_keys = [key for key in required_keys if key not in self.model_components]
                    if missing_keys:
                        error_msg = f"Model missing required components: {missing_keys}"
                        logger.error(error_msg)
                        raise ValueError(error_msg)
                
            except Exception as e:
                error_msg = f"Error loading model file: {str(e)}"
                logger.error(error_msg)
                raise
            
        except Exception as e:
            error_msg = f"Failed to load model assets: {str(e)}"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)

    def _validate_input(self, input_data: UserInput) -> None:
        """Validate input data ranges."""
        # Only validate workout frequency as it's based on days of the week
        if not (0 <= input_data.workout_frequency <= 7):
            raise HTTPException(
                status_code=400, 
                detail="Workout frequency must be between 0 and 7 days"
            )

    def _calculate_health_score(self, input_data: UserInput) -> float:
        """Calculate health score based on input metrics."""
        try:
            age = input_data.age
            bmi = input_data.bmi
            workout_freq = input_data.workout_frequency
            
            # Age score: More flexible scoring that doesn't penalize extreme ages as harshly
            age_score = 1.0 / (1.0 + abs(age - 25) / 50)  # Smoother curve for age scoring
            
            # BMI score: More flexible scoring that considers a wider range of healthy BMIs
            if 18.5 <= bmi <= 24.9:  # Standard healthy BMI range
                bmi_score = 1.0
            else:
                # Smoother curve for BMI scoring
                bmi_score = 1.0 / (1.0 + abs(bmi - 21.7) / 20)  # 21.7 is the midpoint of healthy range
            
            # Workout score: Linear scale up to 5 days, then plateaus
            workout_score = min(workout_freq / 5.0, 1.0)
            
            # Calculate final health score with equal weights
            health_score = (age_score + bmi_score + workout_score) / 3.0
            
            # Store score components for debug info
            self._last_score_components = {
                "age_score": age_score,
                "bmi_score": bmi_score,
                "workout_score": workout_score
            }
            
            return health_score
        except Exception as e:
            logger.error(f"Error calculating health score: {str(e)}", exc_info=True)
            raise ValueError(f"Failed to calculate health score: {str(e)}")

    def _interpret_prediction(self, health_score: float) -> tuple[str, str]:
        """Convert health score to difficulty level and recommendation."""
        easy_threshold = self.model_components['easy_threshold']
        medium_threshold = self.model_components['medium_threshold']
        
        if health_score <= easy_threshold:
            return "Easy", "Start with light exercises and gradually increase intensity. Focus on building endurance and proper form."
        elif health_score <= medium_threshold:
            return "Medium", "You can handle moderate intensity workouts. Mix cardio and strength training with progressive overload."
        else:
            return "Hard", "You're ready for high-intensity workouts. Challenge yourself with complex exercises and advanced training techniques."

    def predict(self, input_data: UserInput) -> PredictionResponse:
        """Make a prediction based on input data."""
        try:
            # Validate input
            self._validate_input(input_data)
            
            # Calculate health score
            health_score = self._calculate_health_score(input_data)
            
            # Get thresholds from model
            easy_threshold = self.model_components['easy_threshold']
            medium_threshold = self.model_components['medium_threshold']
            
            # Determine difficulty level and recommendation
            if health_score < easy_threshold:
                difficulty = "Easy"
                recommendation = (
                    "Based on your current metrics, you should start with low-intensity workouts. "
                    "Focus on building a consistent routine and gradually increasing intensity."
                )
            elif health_score < medium_threshold:
                difficulty = "Medium"
                recommendation = (
                    "You can handle moderate intensity workouts. "
                    "Mix cardio and strength training while maintaining proper form and recovery."
                )
            else:
                difficulty = "Hard"
                recommendation = (
                    "You're ready for high-intensity workouts. "
                    "Challenge yourself with advanced exercises while maintaining proper form and recovery."
                )
            
            # Calculate confidence score based on how far the health score is from thresholds
            if difficulty == "Easy":
                confidence = 1 - (health_score / easy_threshold)
            elif difficulty == "Medium":
                confidence = 1 - abs(health_score - (easy_threshold + medium_threshold) / 2) / ((medium_threshold - easy_threshold) / 2)
            else:  # Hard
                confidence = (health_score - medium_threshold) / (1 - medium_threshold)
            
            # Ensure confidence is between 0 and 1
            confidence = max(0, min(1, confidence))
            
            # Get score components from last calculation
            score_components = getattr(self, '_last_score_components', {})
            
            return PredictionResponse(
                difficulty_level=difficulty,
                confidence_score=confidence,
                recommendation=recommendation,
                health_score=health_score,
                debug_info={
                    "input_data": {
                        "age": input_data.age,
                        "bmi": input_data.bmi,
                        "workout_frequency": input_data.workout_frequency
                    },
                    "health_score": health_score,
                    "thresholds": {
                        "easy_threshold": easy_threshold,
                        "medium_threshold": medium_threshold
                    },
                    "score_components": score_components
                }
            )
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}", exc_info=True)
            raise ValueError(f"Error making prediction: {str(e)}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the loaded model."""
        return {
            "model_type": "Health Score Model",
            "feature_names": self.feature_names,
            "thresholds": {
                "easy_threshold": self.model_components['easy_threshold'],
                "medium_threshold": self.model_components['medium_threshold']
            },
            "health_score_stats": self.model_components['health_score_stats']
        }

# Instantiate model handler
model_handler = StepSyncModel()

# ----------------- Error Handlers -----------------
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors with more user-friendly messages."""
    # Log the raw request body for debugging
    try:
        body = await request.json()
        logger.error(f"Invalid request body: {body}")
    except:
        logger.error("Could not parse request body")
    
    errors = []
    for error in exc.errors():
        field = error["loc"][-1]
        msg = error["msg"]
        if "type" in error and error["type"] == "type_error.number":
            errors.append(f"{field} must be a number")
        elif "type" in error and error["type"] == "type_error.float":
            errors.append(f"{field} must be a number")
        elif "type" in error and error["type"] == "value_error.missing":
            errors.append(f"Missing required field: {field}")
        else:
            errors.append(f"{field}: {msg}")
    
    return JSONResponse(
        status_code=422,
        content={
            "status": "error",
            "code": 422,
            "message": "Validation error",
            "details": errors,
            "help": "Please ensure all fields are numbers and workout_frequency is between 0 and 7. Field names can be: age/Age, bmi/BMI, workout_frequency/Workout_Frequency"
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle all other exceptions with a consistent format."""
    logger.error(f"Unexpected error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "code": 500,
            "message": "Internal server error",
            "details": str(exc) if os.getenv("LOG_LEVEL", "").lower() == "debug" else "An unexpected error occurred"
        }
    )

# ----------------- API Endpoints -----------------
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "status": "healthy",
        "message": "StepSync Health Score API",
        "version": "3.0.0",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "model_info": "/model-info"
        },
        "documentation": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model_handler.model_components is not None,
        "model_info": model_handler.get_model_info()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(user_input: UserInput):
    """Make a workout difficulty prediction based on user metrics."""
    try:
        # Log the incoming request for debugging
        logger.info(f"Received prediction request: {user_input.model_dump()}")
        
        # Make prediction directly with the input
        try:
            return model_handler.predict(user_input)
        except Exception as e:
            logger.error(f"Model prediction error: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "Prediction failed",
                    "error": str(e) if os.getenv("LOG_LEVEL", "").lower() == "debug" else "An error occurred during prediction"
                }
            )
    except HTTPException:
        # Re-raise HTTP exceptions as they're already properly formatted
        raise
    except Exception as e:
        # Log unexpected errors
        logger.error(f"Unexpected error in predict endpoint: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Internal server error",
                "error": str(e) if os.getenv("LOG_LEVEL", "").lower() == "debug" else "An unexpected error occurred"
            }
        )

@app.get("/model-info")
async def get_model_info():
    """Get detailed information about the loaded model."""
    return model_handler.get_model_info()

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up StepSync Health Score API...")
    logger.info("API startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down StepSync Health Score API...")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)