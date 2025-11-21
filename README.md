# Spam Detection System

## Overview
This project implements a machine learning pipeline to detect spam messages. It supports multiple classifiers (Naive Bayes, SVM, Logistic Regression) with TF-IDF vectorization, configurable via `config.yaml`. The system is designed to be modular, scalable, and easy to maintain.

## Project Structure
```
├── config/             # Configuration files
│   └── config.yaml     # Main config (paths, params)
├── data/               # Data directory
│   ├── raw/            # Original dataset
│   └── processed/      # Processed data (if any)
├── logs/               # Application logs
├── models/             # Saved models and vectorizers
├── src/                # Source code
│   ├── data_loader.py  # Data ingestion
│   ├── preprocessor.py # Text cleaning
│   ├── model_trainer.py# Model training pipeline
│   └── evaluator.py    # Metrics and evaluation
├── tests/              # Notebooks and test scripts
├── main.py             # Main execution script
└── requirements.txt    # Dependencies
```

## Setup

1. **Clone the repository** (if applicable)
2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Model
To train the model, simply run:
```bash
python main.py
```
This will:
- Load the data from `data/raw/spam.csv`
- Preprocess the text
- Train the configured classifier (e.g., Logistic Regression, Naive Bayes)
- Save the model to `models/spam_classifier.joblib`
- Log the progress to `logs/app.log`

### Testing
You can test the trained model with custom messages using:
```bash
python tests/quick_test.py
```

## Results
The current model achieves approximately **96% accuracy** on the test set.

## License
MIT
