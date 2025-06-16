# StepSync Health Score API

A FastAPI-based service that predicts workout difficulty levels based on user health metrics. The API uses a health score model to determine appropriate workout intensities and provides personalized recommendations.

## Features

- Health score calculation based on age, BMI, and workout frequency
- Difficulty level prediction (Easy, Medium, Hard)
- Personalized workout recommendations
- Confidence scoring for predictions
- Comprehensive API health monitoring
- Detailed model information endpoints

## API Endpoints

- `GET /`: Root endpoint with basic API information
- `GET /health`: Health check endpoint with model status
- `POST /predict`: Make workout difficulty predictions
- `GET /model-info`: Get detailed model information

## Prerequisites

- Python 3.11 or higher
- pip (Python package manager)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd stepsync-api
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure you have the model file (`difficulty_model.pkl`) in the root directory.

## Running the Application

### Local Development

1. Start the server:
```bash
uvicorn backup:app --reload
```

2. The API will be available at `http://localhost:8000`

### Using Docker

1. Build the Docker image:
```bash
docker build -t stepsync-api .
```

2. Run the container:
```bash
docker run -p 8000:8000 stepsync-api
```

### Testing

Run the test suite:
```bash
python test.py
```

## Deployment

The application can be deployed using various platforms:

### Heroku
- Uses `Procfile` for process management
- Automatically detects Python runtime from `runtime.txt`

### Railway
- Uses `nixpacks.toml` for deployment configuration
- Automatically detects and builds the application

### Docker
- Use the provided `Dockerfile` for containerized deployment
- Build and run using Docker commands

## API Documentation

Once the server is running, you can access:
- Interactive API documentation: `http://localhost:8000/docs`
- Alternative API documentation: `http://localhost:8000/redoc`

## Input Parameters

The prediction endpoint (`/predict`) accepts the following parameters:

- `Age`: User's age in years (18-80)
- `Calc_BMI`: User's calculated BMI (15-40)
- `Workout_Frequency`: Workout frequency in days per week (0-7)

## Response Format

The prediction endpoint returns:

```json
{
    "difficulty_level": "string",
    "confidence_score": float,
    "recommendation": "string",
    "health_score": float,
    "debug_info": {
        "input_data": object,
        "health_score": float,
        "thresholds": object,
        "score_components": object
    }
}
```

## License

[Your License Here]

## Contributing

[Your Contributing Guidelines Here]