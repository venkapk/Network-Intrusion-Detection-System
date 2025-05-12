import argparse
from src.training.models import train_models
from api.app import app

def main():
    """
    Main entry point for the Network Intrusion Detection System.
    Handles command-line args and starts either training, the API server, or both.
    """
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Network Intrusion Detection System')
    parser.add_argument('--train', action='store_true', help='Train models before starting the application')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the application on')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the application on')
    args = parser.parse_args()
    
    # Train models if requested (useful for initial setup or retraining)
    if args.train:
        print("Training models...")
        train_models()
    
    # Start the Flask application with the specified host/port
    # Default is all interfaces (0.0.0.0) on port 5000
    print(f"Starting application on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)

if __name__ == "__main__":
    main()