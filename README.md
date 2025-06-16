# StepSync Health Score API

A FastAPI-based service that predicts workout difficulty levels based on user health metrics. The API uses a machine learning model to calculate a health score and determine appropriate workout intensity levels.

## Features

- **Flexible Input Format**: Supports multiple input formats (camelCase, snake_case, PascalCase)
- **Comprehensive Validation**: Detailed input validation with clear error messages
- **Health Score Calculation**: Sophisticated algorithm considering age, BMI, and workout frequency
- **Detailed Response**: Includes difficulty level, confidence score, and personalized recommendations
- **Robust Error Handling**: Clear error messages and proper HTTP status codes
- **CORS Support**: Ready for web and mobile applications
- **Comprehensive Testing**: Built-in test suite for API validation

## API Endpoints

### 1. Health Check
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
            "easy_threshold": 0.575,
            "medium_threshold": 0.731
        }
    }
}
```

### 2. Prediction
```http
POST /predict
```
Predicts workout difficulty based on user metrics.

**Request Body:**
```json
{
    "age": 25,              // Required: 18-80
    "bmi": 22.5,            // Required: 15-40
    "workout_frequency": 3  // Required: 0-7
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
            "easyThreshold": 0.575,
            "mediumThreshold": 0.731
        }
    }
}
```

### 3. Model Information
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
        "bmi must be between 15 and 40"
    ],
    "help": "Please check that all fields are numbers within the valid ranges: age (18-80), bmi (15-40), workout_frequency (0-7)"
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

The API includes a comprehensive test suite. To run the tests:

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
- Prediction testing with various input formats
- Input validation testing
- Error handling verification

## Deployment

The API is configured for deployment on Railway with the following files:
- `Dockerfile`: Container configuration
- `requirements.txt`: Python dependencies
- `Procfile`: Process configuration
- `nixpacks.toml`: Build configuration

### Environment Variables
- `LOG_LEVEL`: Set to "debug" for detailed logging
- `PORT`: Server port (default: 8080)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.