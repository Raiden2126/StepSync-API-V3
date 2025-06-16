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
from pydantic import BaseModel, Field, validator
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
    age: float = Field(..., ge=18, le=80, description="User's age in years", alias="Age")
    bmi: float = Field(..., ge=15, le=40, description="User's calculated BMI", alias="Calc_BMI")
    workout_frequency: float = Field(..., ge=0, le=7, description="Workout frequency in days per week", alias="Workout_Frequency")

    # Allow string inputs and convert them to float
    @validator('age', 'bmi', 'workout_frequency', pre=True)
    def convert_to_float(cls, v):
        if isinstance(v, str):
            try:
                return float(v)
            except ValueError:
                raise ValueError(f"Could not convert {v} to a number")
        return v

    class Config:
        # Allow both camelCase and snake_case
        allow_population_by_field_name = True
        # Example: {"age": 25} or {"Age": 25} both work
        schema_extra = {
            "example": {
                "age": 25,
                "bmi": 22.5,
                "workout_frequency": 3
            }
        }

class PredictionResponse(BaseModel):
    difficulty_level: str
    confidence_score: float
    recommendation: str
    health_score: float
    debug_info: Optional[Dict[str, Any]] = None

    class Config:
        # Convert snake_case to camelCase in response
        json_encoders = {
            float: lambda v: round(v, 3)  # Round floats to 3 decimal places
        }
        # Example response
        schema_extra = {
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

# ----------------- Model Handler -----------------
class StepSyncModel:
    def __init__(self):
        self.model_components: Optional[Dict[str, Any]] = None
        self.feature_names = ["Age", "Calc_BMI", "Workout_Frequency"]
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
        if not (18 <= input_data.age <= 80):
            raise HTTPException(status_code=400, detail="Age must be between 18 and 80")
        if not (15 <= input_data.bmi <= 40):
            raise HTTPException(status_code=400, detail="BMI must be between 15 and 40")
        if not (0 <= input_data.workout_frequency <= 7):
            raise HTTPException(status_code=400, detail="Workout frequency must be between 0 and 7 days")

    def _calculate_health_score(self, input_data: UserInput) -> float:
        """Calculate health score based on input metrics."""
        age = input_data.age
        bmi = input_data.bmi
        workout_freq = input_data.workout_frequency
        
        # Age score: Peak at 25, decline as you move away
        age_score = max(0, 1 - abs(age - 25) / 25)  # Normalized 0-1
        
        # BMI score: Peak between 18.5-24.5 (healthy range)
        if 18.5 <= bmi <= 24.5:
            bmi_score = 1.0
        else:
            # Distance from healthy range
            if bmi < 18.5:
                bmi_score = max(0, 1 - (18.5 - bmi) / 10)
            else:  # bmi > 24.5
                bmi_score = max(0, 1 - (bmi - 24.5) / 15)
        
        # Workout frequency score: Higher is better (1-5 scale)
        workout_score = min(workout_freq / 5.0, 1.0)  # Normalize to 0-1
        
        # Combine scores
        total_score = (age_score + bmi_score + workout_score) / 3
        
        return total_score

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
        """Make prediction using the health score model."""
        try:
            # Validate input
            self._validate_input(input_data)
            
            # Calculate health score
            health_score = self._calculate_health_score(input_data)
            logger.info(f"Health score calculated: {health_score}")
            
            # Get difficulty level and recommendation
            difficulty_level, recommendation = self._interpret_prediction(health_score)
            
            # Calculate confidence score based on distance from thresholds
            easy_threshold = self.model_components['easy_threshold']
            medium_threshold = self.model_components['medium_threshold']
            
            if health_score <= easy_threshold:
                confidence = 1 - (health_score / easy_threshold)
            elif health_score <= medium_threshold:
                confidence = 1 - abs(health_score - (easy_threshold + medium_threshold) / 2) / (medium_threshold - easy_threshold)
            else:
                confidence = (health_score - medium_threshold) / (1 - medium_threshold)
            
            confidence_score = float(max(0, min(1, confidence)))
            
            # Prepare debug info
            debug_info = {
                "input_data": input_data.dict(),
                "health_score": float(health_score),
                "thresholds": {
                    "easy_threshold": easy_threshold,
                    "medium_threshold": medium_threshold
                },
                "score_components": {
                    "age_score": max(0, 1 - abs(input_data.age - 25) / 25),
                    "bmi_score": 1.0 if 18.5 <= input_data.bmi <= 24.5 else max(0, 1 - abs(input_data.bmi - 21.5) / 15),
                    "workout_score": min(input_data.workout_frequency / 5.0, 1.0)
                }
            }
            
            return PredictionResponse(
                difficulty_level=difficulty_level,
                confidence_score=confidence_score,
                recommendation=recommendation,
                health_score=float(health_score),
                debug_info=debug_info
            )
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

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
    errors = []
    for error in exc.errors():
        field = error["loc"][-1]
        msg = error["msg"]
        if "type" in error and error["type"] == "type_error.number":
            errors.append(f"{field} must be a number")
        elif "type" in error and error["type"] == "type_error.float":
            errors.append(f"{field} must be a number")
        else:
            errors.append(f"{field}: {msg}")
    
    return JSONResponse(
        status_code=422,
        content={
            "status": "error",
            "code": 422,
            "message": "Validation error",
            "details": errors,
            "help": "Please check that all fields are numbers within the valid ranges: age (18-80), bmi (15-40), workout_frequency (0-7)"
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
        # Convert to the format expected by the model
        model_input = UserInput(
            Age=user_input.age,
            Calc_BMI=user_input.bmi,
            Workout_Frequency=user_input.workout_frequency
        )
        return model_handler.predict(model_input)
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Prediction failed",
                "error": str(e) if os.getenv("LOG_LEVEL", "").lower() == "debug" else "An error occurred during prediction"
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