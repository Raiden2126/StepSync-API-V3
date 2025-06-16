import requests
import json
from typing import Dict, Any
from datetime import datetime

# API endpoints
BASE_URL = "https://web-production-348f7.up.railway.app"
PREDICT_URL = f"{BASE_URL}/predict"
MODEL_INFO_URL = f"{BASE_URL}/model-info"
HEALTH_URL = f"{BASE_URL}/health"

def check_server_health() -> None:
    """Check the overall health of the server and its components."""
    print("\nChecking Server Health")
    print("=" * 50)
    
    try:
        response = requests.get(HEALTH_URL)
        response.raise_for_status()
        health_info = response.json()
        
        print("\nServer Status:")
        print("-" * 30)
        print(f"Status: {health_info.get('status', 'Unknown')}")
        print(f"Model Loaded: {health_info.get('model_loaded', False)}")
        
        if health_info.get('model_info'):
            print("\nDetailed Model Info:")
            print("-" * 30)
            model_info = health_info['model_info']
            print(f"Model Type: {model_info.get('model_type', 'Unknown')}")
            print(f"Feature Names: {model_info.get('feature_names', [])}")
            
            if 'thresholds' in model_info:
                print("\nDifficulty Thresholds:")
                print(f"Easy Threshold: {model_info['thresholds'].get('easy_threshold', 'Unknown')}")
                print(f"Medium Threshold: {model_info['thresholds'].get('medium_threshold', 'Unknown')}")
            
            if 'health_score_stats' in model_info:
                print("\nHealth Score Statistics:")
                stats = model_info['health_score_stats']
                print(f"Mean: {stats.get('mean', 'Unknown'):.3f}")
                print(f"Std: {stats.get('std', 'Unknown'):.3f}")
                print(f"Min: {stats.get('min', 'Unknown'):.3f}")
                print(f"Max: {stats.get('max', 'Unknown'):.3f}")
        
        if not health_info.get('model_loaded'):
            print("\n⚠️ WARNING: Model failed to load!")
        else:
            print("\n✓ Model loaded successfully")
            
    except requests.exceptions.RequestException as e:
        print(f"Error checking server health: {str(e)}")
        if hasattr(e.response, 'text'):
            print(f"Server response: {e.response.text}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")

def check_model_status() -> None:
    """Check if the model and thresholds are properly loaded."""
    print("\nChecking Model Status")
    print("=" * 50)
    
    try:
        # Check model info
        response = requests.get(MODEL_INFO_URL)
        response.raise_for_status()
        model_info = response.json()
        
        print("\nModel Information:")
        print("-" * 30)
        print(f"Model Type: {model_info.get('model_type', 'Unknown')}")
        print(f"Feature Names: {model_info.get('feature_names', [])}")
        
        if 'thresholds' in model_info:
            print("\nDifficulty Thresholds:")
            print("-" * 30)
            thresholds = model_info['thresholds']
            print(f"Easy Threshold: {thresholds.get('easy_threshold', 'Not found')}")
            print(f"Medium Threshold: {thresholds.get('medium_threshold', 'Not found')}")
            
        if 'health_score_stats' in model_info:
            print("\nHealth Score Statistics:")
            print("-" * 30)
            stats = model_info['health_score_stats']
            print(f"Mean: {stats.get('mean', 'Unknown'):.3f}")
            print(f"Std: {stats.get('std', 'Unknown'):.3f}")
            print(f"Min: {stats.get('min', 'Unknown'):.3f}")
            print(f"Max: {stats.get('max', 'Unknown'):.3f}")
            
    except requests.exceptions.RequestException as e:
        print(f"Error checking model status: {str(e)}")
        if hasattr(e.response, 'text'):
            print(f"Server response: {e.response.text}")
    except json.JSONDecodeError as e:
        print(f"Error decoding response: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")

def format_response(response: Dict[str, Any]) -> str:
    """Format the API response in a readable way."""
    return f"""
Workout Difficulty Assessment
----------------------------
Difficulty Level: {response['difficulty_level']}
Health Score: {response['health_score']:.3f}
Confidence Score: {response['confidence_score']:.2%}
Recommendation: {response['recommendation']}

Score Components:
----------------
Age Score: {response['debug_info']['score_components']['age_score']:.3f}
BMI Score: {response['debug_info']['score_components']['bmi_score']:.3f}
Workout Score: {response['debug_info']['score_components']['workout_score']:.3f}
"""

def test_prediction(data: Dict[str, float], test_name: str) -> None:
    """Make a prediction request and print the results."""
    print(f"\nTest Case: {test_name}")
    print("-" * 50)
    print(f"Input Data: {json.dumps(data, indent=2)}")
    
    try:
        response = requests.post(PREDICT_URL, json=data)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        result = response.json()
        print(format_response(result))
        
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {str(e)}")
    except json.JSONDecodeError as e:
        print(f"Error decoding response: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")

def main():
    """Run multiple test cases with different input combinations."""
    print(f"Starting API Tests at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # First check server health
    check_server_health()
    
    # Then check model status
    check_model_status()
    
    print("\nRunning Prediction Tests")
    print("=" * 50)
    
    # Test Case 1: Young, healthy, active person (should be Hard)
    test_prediction({
        "Age": 25.0,
        "Calc_BMI": 22.0,
        "Workout_Frequency": 4.0
    }, "Young, Healthy, Active Person")
    
    # Test Case 2: Middle-aged, slightly overweight, moderate activity (should be Medium)
    test_prediction({
        "Age": 35.0,
        "Calc_BMI": 25.0,
        "Workout_Frequency": 2.0
    }, "Middle-aged, Slightly Overweight, Moderate Activity")
    
    # Test Case 3: Older, overweight, low activity (should be Easy)
    test_prediction({
        "Age": 45.0,
        "Calc_BMI": 28.0,
        "Workout_Frequency": 1.0
    }, "Older, Overweight, Low Activity")
    
    # Test Case 4: Very young, underweight, high activity
    test_prediction({
        "Age": 18.0,
        "Calc_BMI": 18.5,
        "Workout_Frequency": 7.0
    }, "Very Young, Underweight, High Activity")
    
    # Test Case 5: Senior, obese, no activity
    test_prediction({
        "Age": 65.0,
        "Calc_BMI": 35.0,
        "Workout_Frequency": 0.0
    }, "Senior, Obese, No Activity")
    
    # Test Case 6: Young adult, healthy BMI, moderate activity
    test_prediction({
        "Age": 30.0,
        "Calc_BMI": 23.0,
        "Workout_Frequency": 3.0
    }, "Young Adult, Healthy BMI, Moderate Activity")
    
    # Test Case 7: Middle-aged, overweight, regular activity
    test_prediction({
        "Age": 40.0,
        "Calc_BMI": 26.0,
        "Workout_Frequency": 2.0
    }, "Middle-aged, Overweight, Regular Activity")
    
    print("\n" + "=" * 80)
    print(f"API Tests completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
