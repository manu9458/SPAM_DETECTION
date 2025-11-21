from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import logging
import os

class SpamModelTrainer:
    """
    Class to train and save the spam detection model.
    """
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.pipeline = None

    def build_pipeline(self):
        """
        Constructs the sklearn pipeline.
        """
        model_type = self.config['model']['type']
        tfidf_config = self.config['model']['tfidf']

        # Initialize Vectorizer
        vectorizer = TfidfVectorizer(
            max_features=tfidf_config['max_features'],
            stop_words=tfidf_config['stop_words'],
            ngram_range=tuple(tfidf_config.get('ngram_range', [1, 1]))
        )

        # Initialize Classifier
        if model_type == 'naive_bayes':
            classifier = MultinomialNB()
        elif model_type == 'svm':
            classifier = SVC(probability=True, random_state=self.config['data']['random_state'])
        elif model_type == 'logistic_regression':
            classifier = LogisticRegression(random_state=self.config['data']['random_state'])
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        self.pipeline = Pipeline([
            ('tfidf', vectorizer),
            ('clf', classifier)
        ])
        
        self.logger.info(f"Pipeline built with {model_type} classifier.")

    def train(self, X_train, y_train):
        """
        Trains the model pipeline.
        """
        if self.pipeline is None:
            self.build_pipeline()
        
        self.logger.info("Starting model training...")
        self.pipeline.fit(X_train, y_train)
        self.logger.info("Model training completed.")

    def save_model(self):
        """
        Saves the trained pipeline and vectorizer to disk.
        """
        model_save_path = self.config['paths']['model_save_path']
        vectorizer_save_path = self.config['paths'].get('vectorizer_save_path')
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        
        joblib.dump(self.pipeline, model_save_path)
        self.logger.info(f"Model saved to {model_save_path}")

        # Save vectorizer separately if path is provided
        if vectorizer_save_path and self.pipeline:
            os.makedirs(os.path.dirname(vectorizer_save_path), exist_ok=True)
            vectorizer = self.pipeline.named_steps['tfidf']
            joblib.dump(vectorizer, vectorizer_save_path)
            self.logger.info(f"Vectorizer saved to {vectorizer_save_path}")
