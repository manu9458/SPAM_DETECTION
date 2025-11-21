import re
import string

class TextPreprocessor:
    """
    Class for cleaning and preprocessing text data.
    """
    def __init__(self):
        pass

    def clean_text(self, text):
        """
        Basic text cleaning: lowercase, remove punctuation, etc.
        """
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Replace currency symbols with text
        text = re.sub(r'[$€£¥]', ' currency_token ', text)
        
        # Replace exclamation marks with text (count matters, but simple replacement helps)
        text = re.sub(r'!', ' exclamation_token ', text)
        
        # Remove punctuation (now safe to remove rest)
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def preprocess_batch(self, texts):
        """
        Apply cleaning to a list/series of texts.
        """
        return texts.apply(self.clean_text)
