import joblib
import os
import sys

# Add project root to path so we can import from src if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def quick_test():
    model_path = 'models/spam_classifier.joblib'
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please run main.py first.")
        return

    print(f"Loading model from {model_path}...")
    pipeline = joblib.load(model_path)
    
    test_messages = [
        "Hey, are we still on for lunch today?",
        "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt WORD to 80086 to claim No: 80086",
        "Mom called, she wants you to pick up some milk.",
        "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005."
    ]
    
    print("\nRunning predictions on test messages:\n")
    
    predictions = pipeline.predict(test_messages)
    probabilities = pipeline.predict_proba(test_messages)
    
    for msg, pred, prob in zip(test_messages, predictions, probabilities):
        spam_prob = prob[1]
        print(f"Message: '{msg}'")
        print(f"Prediction: {pred} (Spam probability: {spam_prob:.4f})")
        print("-" * 50)

if __name__ == "__main__":
    quick_test()
