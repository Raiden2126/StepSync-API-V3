# StepSync Health Score API v3.0.0

A FastAPI-based service that predicts workout difficulty levels based on user health metrics. The API uses a sophisticated health score algorithm to calculate appropriate workout intensity levels.

## Features

- **Flexible Input Format**: Supports multiple input formats (camelCase, snake_case, PascalCase) with automatic field name conversion
- **Comprehensive Validation**: Detailed input validation with clear error messages and type conversion
- **Advanced Health Score Algorithm**: Sophisticated scoring system considering age, BMI, and workout frequency with smooth curves for better accuracy
- **Detailed Response**: Includes difficulty level, confidence score, personalized recommendations, and debug information
- **Robust Error Handling**: Clear error messages, proper HTTP status codes, and detailed validation feedback
- **Enhanced CORS Support**: Ready for web and mobile applications with configurable origins
- **Comprehensive Testing**: Built-in test suite with extensive test cases for API validation
- **Detailed Logging**: Configurable logging levels for debugging and monitoring

## API Endpoints

### 1. Root
```http
GET /
```
Returns basic API information and available endpoints.

**Response Example:**
```json
{
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
```

### 2. Health Check
```http
GET /health
```
Returns the API's health status and model information.

**Response Example:**
```json
{
    "status": "healthy",
    "model_loaded": true,
    "model_info": {
        "model_type": "Health Score Model",
        "feature_names": ["Age", "BMI", "Workout_Frequency"],
        "thresholds": {
            "easy_threshold": 0.57,
            "medium_threshold": 0.73
        },
        "health_score_stats": {
            // Model statistics and metrics
        }
    }
}
```

### 3. Prediction
```http
POST /predict
```
Predicts workout difficulty based on user metrics.

**Request Body:**
```json
{
    "age": 25,              // Required: Must be positive
    "bmi": 22.5,            // Required: Must be positive
    "workout_frequency": 3  // Required: 0-7 days per week
}
```

**Note:** Field names can be in any of these formats:
- camelCase: `age`, `bmi`, `workoutFrequency`
- snake_case: `age`, `bmi`, `workout_frequency`
- PascalCase: `Age`, `BMI`, `Workout_Frequency`

**Response Example:**
```json
{
    "difficultyLevel": "Medium",
    "confidenceScore": 0.85,
    "recommendation": "You can handle moderate intensity workouts...",
    "healthScore": 0.65,
    "debugInfo": {
        "inputData": {
            "age": 25,
            "bmi": 22.5,
            "workoutFrequency": 3
        },
        "healthScore": 0.65,
        "thresholds": {
            "easyThreshold": 0.57,
            "mediumThreshold": 0.73
        },
        "scoreComponents": {
            "ageScore": 0.75,
            "bmiScore": 0.85,
            "workoutScore": 0.60
        }
    }
}
```

### 4. Model Information
```http
GET /model-info
```
Returns detailed information about the model and its configuration.

## Error Handling

The API uses standard HTTP status codes and provides detailed error messages:

- `400 Bad Request`: Invalid input data
- `422 Unprocessable Entity`: Validation errors
- `500 Internal Server Error`: Server-side errors

**Error Response Example:**
```json
{
    "status": "error",
    "code": 422,
    "message": "Validation error",
    "details": [
        "age must be a number",
        "bmi must be positive"
    ],
    "help": "Please ensure all fields are numbers and workout_frequency is between 0 and 7. Field names can be: age/Age, bmi/BMI, workout_frequency/Workout_Frequency"
}
```

## Usage Examples

### Python
```python
import requests

def make_prediction(age: float, bmi: float, workout_frequency: float):
    response = requests.post(
        "https://your-api-url/predict",
        json={
            "age": age,
            "bmi": bmi,
            "workout_frequency": workout_frequency
        }
    )
    return response.json()
```

### JavaScript/React Native
```javascript
const makePrediction = async (userData) => {
  try {
    const response = await fetch('https://your-api-url/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: JSON.stringify({
        age: userData.age,
        bmi: userData.bmi,
        workout_frequency: userData.workoutFrequency
      }),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.message || 'API request failed');
    }

    return data;
  } catch (error) {
    console.error('Prediction error:', error);
    throw error;
  }
};
```

## Testing

The API includes a comprehensive test suite with extensive test cases. To run the tests:

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the test suite:
```bash
# Test against local server
python test.py

# Test against deployed server
python test.py https://your-api-url
```

The test suite includes:
- Health check validation
- Model information verification
- Prediction testing with various input formats (camelCase, snake_case, PascalCase)
- Input validation testing (invalid types, missing fields, out-of-range values)
- Error handling verification
- Edge case testing (extreme ages, BMIs, workout frequencies)

## Deployment

The API is configured for deployment on Railway with the following files:
- `Dockerfile`: Container configuration using Python 3.9 slim image
- `requirements.txt`: Python dependencies with specific versions
- `Procfile`: Process configuration for Railway
- `nixpacks.toml`: Build configuration

### Environment Variables
- `LOG_LEVEL`: Set to "debug" for detailed logging (default: info)
- `PORT`: Server port (default: 8080)
- `PYTHONPATH`: Set to /app in container
- `PYTHONDONTWRITEBYTECODE`: Disabled bytecode writing
- `PYTHONUNBUFFERED`: Unbuffered Python output

### Dependencies
Key dependencies include:
- FastAPI 0.104.1
- Uvicorn 0.24.0
- NumPy 1.26.2
- Pandas 2.1.3
- Joblib 1.3.2
- Pydantic 2.5.2

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.