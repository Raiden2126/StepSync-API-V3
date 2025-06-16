import os
import sys
import uvicorn

def get_port():
    try:
        # Try to get PORT from environment
        port_str = os.environ.get("PORT")
        print(f"Raw PORT value from environment: {port_str}", file=sys.stderr)
        
        if port_str is None:
            print("PORT not found in environment, using default 8000", file=sys.stderr)
            return 8000
            
        # Try to convert to integer
        port = int(port_str)
        print(f"Using port: {port}", file=sys.stderr)
        return port
    except ValueError as e:
        print(f"Error converting PORT to integer: {e}", file=sys.stderr)
        print(f"Invalid PORT value: {port_str}", file=sys.stderr)
        print("Falling back to default port 8000", file=sys.stderr)
        return 8000
    except Exception as e:
        print(f"Unexpected error getting port: {e}", file=sys.stderr)
        print("Falling back to default port 8000", file=sys.stderr)
        return 8000

if __name__ == "__main__":
    port = get_port()
    print(f"Starting server on port {port}", file=sys.stderr)
    uvicorn.run("backup:app", host="0.0.0.0", port=port, log_level="info") 