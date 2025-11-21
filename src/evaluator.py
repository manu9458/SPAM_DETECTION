from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import logging

class ModelEvaluator:
    """
    Class to evaluate the trained model.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def evaluate(self, model, X_test, y_test):
        """
        Predicts on test data and prints evaluation metrics.
        """
        self.logger.info("Evaluating model...")
        
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        print("\n=== Model Evaluation Results ===")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:\n")
        print(report)
        print("\nConfusion Matrix:\n")
        print(conf_matrix)
        print("================================\n")
        
        self.logger.info(f"Evaluation complete. Accuracy: {accuracy:.4f}")
        
        return {
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': conf_matrix
        }
