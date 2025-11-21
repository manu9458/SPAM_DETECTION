import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import os

class DataLoader:
    """
    Class to handle data loading and splitting.
    """
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def load_data(self):
        """
        Loads data from the specified path in config.
        """
        data_path = self.config['data']['raw_data_path']
        try:
            # Try reading with different encodings as spam datasets often have weird characters
            try:
                df = pd.read_csv(data_path, encoding='utf-8')
            except UnicodeDecodeError:
                self.logger.warning("UTF-8 encoding failed, trying latin-1")
                df = pd.read_csv(data_path, encoding='latin-1')
            
            self.logger.info(f"Data loaded successfully from {data_path}. Shape: {df.shape}")
            
            # Check for augmented data
            augmented_path = os.path.join(os.path.dirname(data_path), 'augmented_spam.csv')
            if os.path.exists(augmented_path):
                try:
                    df_aug = pd.read_csv(augmented_path)
                    # Ensure columns match (v1, v2)
                    if 'v1' in df_aug.columns and 'v2' in df_aug.columns:
                        df_aug = df_aug[['v1', 'v2']]
                        df = pd.concat([df, df_aug], ignore_index=True)
                        self.logger.info(f"Augmented data merged. New Shape: {df.shape}")
                except Exception as e:
                    self.logger.warning(f"Failed to load augmented data: {e}")

            return df
        except FileNotFoundError:
            self.logger.error(f"File not found at {data_path}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise

    def get_train_test_split(self, df):
        """
        Splits the dataframe into train and test sets.
        """
        text_col = self.config['data']['text_column']
        label_col = self.config['data']['label_column']
        test_size = self.config['data']['test_size']
        random_state = self.config['data']['random_state']

        X = df[text_col]
        y = df[label_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        self.logger.info(f"Data split into train ({len(X_train)}) and test ({len(X_test)}) sets.")
        return X_train, X_test, y_train, y_test
