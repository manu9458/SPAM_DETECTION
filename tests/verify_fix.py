import joblib
from src.preprocessor import TextPreprocessor

def test_model():
    model_path = "models/spam_classifier.joblib"
    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    preprocessor = TextPreprocessor()
    
    text = "Get a free iPhone 15 by taking part in our survey. Limited time only!"
    clean_text = preprocessor.clean_text(text)
    
    prediction = model.predict([clean_text])[0]
    proba = model.predict_proba([clean_text])[0]
    
    print(f"Text: {text}")
    print(f"Cleaned Text: {clean_text}")
    print(f"Prediction: {prediction}")
    print(f"Probabilities: {proba}")

if __name__ == "__main__":
    test_model()
