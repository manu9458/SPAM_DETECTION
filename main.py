import yaml
import logging
import sys
import os
from src.data_loader import DataLoader
from src.preprocessor import TextPreprocessor
from src.model_trainer import SpamModelTrainer
from src.evaluator import ModelEvaluator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/app.log')
    ]
)
logger = logging.getLogger(__name__)

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    try:
        # 1. Load Configuration
        config_path = 'config/config.yaml'
        if not os.path.exists(config_path):
            logger.error(f"Config file not found at {config_path}")
            return
            
        config = load_config(config_path)
        logger.info("Configuration loaded.")

        # Ensure logs directory exists
        os.makedirs('logs', exist_ok=True)

        # 2. Data Loading
        loader = DataLoader(config)
        df = loader.load_data()
        
        # 3. Preprocessing
        # Note: We are doing basic cleaning here, but TF-IDF in the pipeline also handles tokenization.
        # It's good practice to clean noise before feeding to vectorizer.
        preprocessor = TextPreprocessor()
        df[config['data']['text_column']] = preprocessor.preprocess_batch(df[config['data']['text_column']])
        
        # Split data
        X_train, X_test, y_train, y_test = loader.get_train_test_split(df)

        # 4. Model Training
        trainer = SpamModelTrainer(config)
        trainer.train(X_train, y_train)
        trainer.save_model()

        # 5. Evaluation
        evaluator = ModelEvaluator()
        # We use the trained pipeline directly for evaluation
        evaluator.evaluate(trainer.pipeline, X_test, y_test)

    except Exception as e:
        logger.exception("An error occurred during the execution of the pipeline.")

if __name__ == "__main__":
    main()
