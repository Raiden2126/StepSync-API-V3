import requests
import json
import logging
import sys
from typing import Dict, Any, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class APITester:
    def __init__(self, base_url: str):
        """Initialize API tester with base URL."""
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })

    def _make_request(self, method: str, endpoint: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make HTTP request and handle response."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        try:
            response = self.session.request(method, url, json=data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            if hasattr(e.response, 'json'):
                try:
                    error_data = e.response.json()
                    logger.error(f"Error details: {json.dumps(error_data, indent=2)}")
                except:
                    logger.error(f"Response text: {e.response.text}")
            raise

    def test_health(self) -> bool:
        """Test health check endpoint."""
        try:
            response = self._make_request('GET', '/health')
            logger.info("Health check response:")
            logger.info(json.dumps(response, indent=2))
            return response.get('status') == 'healthy'
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False

    def test_model_info(self) -> bool:
        """Test model info endpoint."""
        try:
            response = self._make_request('GET', '/model-info')
            logger.info("Model info response:")
            logger.info(json.dumps(response, indent=2))
            return 'model_type' in response
        except Exception as e:
            logger.error(f"Model info check failed: {str(e)}")
            return False

    def test_prediction(self, test_cases: list) -> bool:
        """Test prediction endpoint with various test cases."""
        all_passed = True
        for i, test_case in enumerate(test_cases, 1):
            logger.info(f"\nTesting case {i}:")
            logger.info(f"Input: {json.dumps(test_case, indent=2)}")
            
            try:
                # Test both camelCase and snake_case formats
                response = self._make_request('POST', '/predict', test_case)
                logger.info("Prediction response:")
                logger.info(json.dumps(response, indent=2))
                
                # Verify response structure
                required_fields = {
                    'difficultyLevel', 'confidenceScore', 
                    'recommendation', 'healthScore'
                }
                if not all(field in response for field in required_fields):
                    logger.error(f"Missing required fields in response: {required_fields - set(response.keys())}")
                    all_passed = False
                    continue

                # Verify value ranges
                if not (0 <= response['confidenceScore'] <= 1):
                    logger.error(f"Invalid confidence score: {response['confidenceScore']}")
                    all_passed = False
                if not (0 <= response['healthScore'] <= 1):
                    logger.error(f"Invalid health score: {response['healthScore']}")
                    all_passed = False
                if response['difficultyLevel'] not in ['Easy', 'Medium', 'Hard']:
                    logger.error(f"Invalid difficulty level: {response['difficultyLevel']}")
                    all_passed = False

            except Exception as e:
                logger.error(f"Prediction test failed: {str(e)}")
                all_passed = False

        return all_passed

    def test_validation(self, invalid_cases: list) -> bool:
        """Test input validation with invalid cases."""
        all_passed = True
        for i, test_case in enumerate(invalid_cases, 1):
            logger.info(f"\nTesting invalid case {i}:")
            logger.info(f"Input: {json.dumps(test_case, indent=2)}")
            
            try:
                self._make_request('POST', '/predict', test_case)
                logger.error("Expected validation error but request succeeded")
                all_passed = False
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 422:
                    logger.info("Validation error (expected):")
                    try:
                        error_data = e.response.json()
                        logger.info(json.dumps(error_data, indent=2))
                    except:
                        logger.info(f"Response text: {e.response.text}")
                else:
                    logger.error(f"Unexpected error status: {e.response.status_code}")
                    all_passed = False

        return all_passed

def main():
    """Run API tests."""
    # Get base URL from command line or use default
    base_url = "https://web-production-348f7.up.railway.app"
    tester = APITester(base_url)

    # Test cases
    valid_test_cases = [
        # Test camelCase format with various ages and BMIs
        {
            "age": 15,  # Young age
            "bmi": 18.0,  # Low BMI
            "workout_frequency": 3
        },
        # Test snake_case format
        {
            "age": 85,  # Older age
            "bmi": 35.0,  # Higher BMI
            "workoutFrequency": 4
        },
        # Test PascalCase format (with aliases)
        {
            "Age": 45,
            "BMI": 28.0,
            "Workout_Frequency": 5
        },
        # Test edge cases for workout frequency
        {
            "age": 30,
            "bmi": 25.0,
            "workout_frequency": 0  # No workouts
        },
        {
            "age": 35,
            "bmi": 30.0,
            "workout_frequency": 7  # Daily workouts
        },
        # Test extreme cases
        {
            "age": 100,  # Very old age
            "bmi": 45.0,  # Very high BMI
            "workout_frequency": 2
        },
        {
            "age": 5,  # Very young age
            "bmi": 12.0,  # Very low BMI
            "workout_frequency": 1
        }
    ]

    invalid_test_cases = [
        # Invalid workout frequency
        {"age": 25, "bmi": 22.5, "workout_frequency": -1},
        {"age": 25, "bmi": 22.5, "workout_frequency": 8},
        # Invalid types
        {"age": "25", "bmi": "22.5", "workout_frequency": "3"},
        {"age": None, "bmi": 22.5, "workout_frequency": 3},
        # Missing fields
        {"age": 25, "bmi": 22.5},
        {"age": 25, "workout_frequency": 3},
        {"bmi": 22.5, "workout_frequency": 3},
        # Zero or negative values
        {"age": 0, "bmi": 22.5, "workout_frequency": 3},
        {"age": -1, "bmi": 22.5, "workout_frequency": 3},
        {"age": 25, "bmi": 0, "workout_frequency": 3},
        {"age": 25, "bmi": -1, "workout_frequency": 3},
        # Extra fields
        {"age": 25, "bmi": 22.5, "workout_frequency": 3, "extra": "field"}
    ]

    # Run tests
    logger.info(f"Starting API tests against {base_url}")
    logger.info("=" * 50)

    # Test health endpoint
    logger.info("\nTesting health endpoint...")
    health_ok = tester.test_health()
    logger.info(f"Health check {'passed' if health_ok else 'failed'}")

    # Test model info endpoint
    logger.info("\nTesting model info endpoint...")
    model_info_ok = tester.test_model_info()
    logger.info(f"Model info check {'passed' if model_info_ok else 'failed'}")

    # Test valid predictions
    logger.info("\nTesting valid predictions...")
    predictions_ok = tester.test_prediction(valid_test_cases)
    logger.info(f"Prediction tests {'passed' if predictions_ok else 'failed'}")

    # Test validation
    logger.info("\nTesting input validation...")
    validation_ok = tester.test_validation(invalid_test_cases)
    logger.info(f"Validation tests {'passed' if validation_ok else 'failed'}")

    # Summary
    logger.info("\nTest Summary:")
    logger.info("=" * 50)
    logger.info(f"Health Check: {'✓' if health_ok else '✗'}")
    logger.info(f"Model Info: {'✓' if model_info_ok else '✗'}")
    logger.info(f"Predictions: {'✓' if predictions_ok else '✗'}")
    logger.info(f"Validation: {'✓' if validation_ok else '✗'}")
    
    all_passed = all([health_ok, model_info_ok, predictions_ok, validation_ok])
    logger.info(f"\nOverall: {'All tests passed! ✓' if all_passed else 'Some tests failed! ✗'}")

if __name__ == "__main__":
    main()