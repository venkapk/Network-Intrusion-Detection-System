# Network Intrusion Detection System

A machine learning-based platform for detecting and classifying network intrusions using real-world traffic data.

## Features

- **Multi-model classification system** with support for binary and multi-class attack detection
- **Flask REST API backend** with endpoints for predictions, model information, and data upload
- **React frontend** with interactive visualizations for network traffic analysis
- **ML pipeline** with feature scaling and PCA

## Tech Stack

### Backend
- **Python**: Core programming language
- **Flask**: Web framework for API development
- **scikit-learn**: Machine learning library for classification algorithms
- **XGBoost**: Gradient boosting framework
- **NumPy/Pandas**: Data manipulation libraries

### Frontend
- **React**: Frontend library for building the user interface
- **JavaScript/ES6**: Programming language for frontend logic
- **HTML5/CSS3**: Markup and styling
- **Bootstrap**: UI component library

### Machine Learning
- **Random Forest**: Ensemble learning method for classification
- **XGBoost**: Advanced gradient boosting implementation
- **Logistic Regression**: Linear model for binary and multiclass classification
- **PCA**: Dimensionality reduction technique

## Project Structure

```
network-intrusion-detection-system/
├── api/                          # Backend Flask application
│   ├── app.py                    # Main Flask app
│   └── requirements.txt          # Python dependencies
├── models/                       # ML model storage
│   ├── binary/                   # Binary classification models
│   ├── multiclass_3/             # 3-class models
│   └── multiclass_4/             # 4-class models
├── data/                         # Data storage
│   ├── raw/                      # Raw datasets
│   └── preprocessed/             # Processed data and preprocessing components
├── src/                          # Source code
│   ├── preprocessing/            # Data preprocessing modules
│   ├── training/                 # Model training scripts
│   │   └── models.py             # Model creation code
│   └── utils/                    # Utility functions
├── ui/                           # Frontend React application
│   ├── index.html                # Main HTML file
│   ├── css/                      # Stylesheets
│   └── js/                       # JavaScript files
├── notebooks/                    # Jupyter notebooks for EDA and model development
├── tests/                        # Test scripts
├── main.py                       # Entry point script
├── README.md                     # Project documentation
├── requirements.txt              # Top-level Python dependencies
└── LICENSE                       # MIT License
```

## Installation & Setup

### Prerequisites
- Python 3.8+
- Node.js 14+
- Required Python packages (see requirements.txt)

### Backend Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/network-intrusion-detection-system.git
cd network-intrusion-detection-system

# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt

# Start the Flask API server
python main.py --host 127.0.0.1 --port 5000

## Usage

1. Access the web interface at http://localhost:5000
2. Upload network traffic data or manually input features
3. Select a model type '(binary/multiclass)' and algorithm
4. View prediction results and analysis

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/predict` | POST | Make predictions using trained models |
| `/api/models` | GET | Get information about available models |
| `/api/features` | GET | Get feature information for a model type |
| `/api/upload` | POST | Upload network traffic data for analysis |

## Model Performance

### Binary Classification
- **Accuracy**: 97%
- **Precision**: 0.96-1.00
- **Recall**: 0.92-1.00
- **F1-Score**: 0.96-1.00

### Multi-class Classification (3 classes)
- **Accuracy**: 98%
- **Weighted Avg Precision**: 0.99
- **Weighted Avg Recall**: 0.98
- **Weighted Avg F1-Score**: 0.98

### Multi-class Classification (4 classes)
- **Accuracy**: 99%
- **Weighted Avg Precision**: 1.00
- **Weighted Avg Recall**: 0.99
- **Weighted Avg F1-Score**: 0.99

## Feature Importance

The models identified these key features for detecting network intrusions:
- PSH Flag Count
- Down/Up Ratio
- Initial Window Bytes (Forward/Backward)
- Minimum Segment Size (Forward)

## Challenges & Solutions

- **Imbalanced Dataset**: Implemented specialized handling for highly skewed class distributions
- **Feature Selection**: Identified optimal feature subsets for different intrusion types
- **Real-time Analysis**: Optimized inference pipeline for rapid processing
- **Model Interpretability**: Added feature importance visualization for explainability

## Future Improvements

- [ ] Add support for real-time network traffic monitoring
- [ ] Implement anomaly detection for zero-day attack identification
- [ ] Develop more sophisticated visualization options
- [ ] Integrate with SIEM systems for enterprise security

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Network Intrusion Detection Dataset providers
- Flask and React communities for excellent documentation
- scikit-learn and XGBoost contributors

## Contact

Your Name - your.email@example.com