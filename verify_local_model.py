import os
import joblib
import sys

def verify_model():
    try:
        print("Current directory:", os.getcwd())
        print("Files in directory:", os.listdir("."))
        
        model_path = "difficulty_model.pkl"
        if not os.path.exists(model_path):
            print("Error: Model file not found!")
            return False
            
        print("Model file size:", os.path.getsize(model_path), "bytes")
        
        try:
            model = joblib.load(model_path)
            print("\nModel loaded successfully!")
            print("Model type:", type(model))
            
            if isinstance(model, dict):
                print("\nModel contents:")
                for key, value in model.items():
                    print(f"- {key}: {type(value)}")
                    if key == 'health_score_stats':
                        print(f"  Stats: {value}")
                    elif key in ['easy_threshold', 'medium_threshold']:
                        print(f"  Value: {value}")
            return True
            
        except Exception as e:
            print("Error loading model:", str(e))
            return False
            
    except Exception as e:
        print("Error during verification:", str(e))
        return False

if __name__ == "__main__":
    if not verify_model():
        sys.exit(1) 