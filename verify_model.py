import os
import sys
import joblib

def verify_model():
    try:
        print("Current directory:", os.getcwd())
        print("Files in directory:", os.listdir("."))
        
        model_path = "difficulty_model.pkl"
        if not os.path.exists(model_path):
            print("Model file not found!")
            return False
            
        print("Model file size:", os.path.getsize(model_path), "bytes")
        
        try:
            model = joblib.load(model_path)
            print("Model loaded successfully")
            print("Model type:", type(model))
            
            if isinstance(model, dict):
                print("Model keys:", list(model.keys()))
                print("Model components:", {k: str(type(v)) for k, v in model.items()})
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