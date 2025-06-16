import logging
import numpy as np
import pandas as pd
import joblib
import os
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- Request Schema ----------------
class UserInput(BaseModel):
    Age: float = Field(..., ge=18, le=80, description="User's age in years")
    Calc_BMI: float = Field(..., ge=15, le=40, description="User's calculated BMI")
    Workout_Frequency: float = Field(..., ge=0, le=7, description="Workout frequency in days per week")

class PredictionResponse(BaseModel):
    difficulty_level: str
    confidence_score: float
    recommendation: str
    health_score: float
    debug_info: Optional[Dict[str, Any]] = None

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
        if not (18 <= input_data.Age <= 80):
            raise HTTPException(status_code=400, detail="Age must be between 18 and 80")
        if not (15 <= input_data.Calc_BMI <= 40):
            raise HTTPException(status_code=400, detail="BMI must be between 15 and 40")
        if not (0 <= input_data.Workout_Frequency <= 7):
            raise HTTPException(status_code=400, detail="Workout frequency must be between 0 and 7 days")

    def _calculate_health_score(self, input_data: UserInput) -> float:
        """Calculate health score based on input metrics."""
        age = input_data.Age
        bmi = input_data.Calc_BMI
        workout_freq = input_data.Workout_Frequency
        
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
                    "age_score": max(0, 1 - abs(input_data.Age - 25) / 25),
                    "bmi_score": 1.0 if 18.5 <= input_data.Calc_BMI <= 24.5 else max(0, 1 - abs(input_data.Calc_BMI - 21.5) / 15),
                    "workout_score": min(input_data.Workout_Frequency / 5.0, 1.0)
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

# ----------------- API Endpoints -----------------
@app.get("/")
async def root():
    return {
        "status": "healthy",
        "message": "StepSync Health Score API",
        "version": "3.0.0",
        "model_type": "Health Score Model"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model_handler.model_components is not None,
        "model_info": model_handler.get_model_info()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(user_input: UserInput):
    """Make a workout difficulty prediction based on user metrics."""
    return model_handler.predict(user_input)

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
