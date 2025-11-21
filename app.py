import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import os
import logging
from src.preprocessor import TextPreprocessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Spam Detection API")

# Load model and preprocessor
MODEL_PATH = "models/spam_classifier.joblib"
try:
    model = joblib.load(MODEL_PATH)
    logger.info(f"Model loaded from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise RuntimeError("Model could not be loaded")

preprocessor = TextPreprocessor()

class TextRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    text: str
    prediction: str
    probability: float

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: TextRequest):
    try:
        # Preprocess
        clean_text = preprocessor.clean_text(request.text)
        
        # Predict
        # The pipeline expects an iterable of strings
        prediction = model.predict([clean_text])[0]
        proba = model.predict_proba([clean_text])[0]
        
        # Get probability of the predicted class
        # classes_ are usually ['ham', 'spam'] sorted alphabetically? 
        # Let's check model.classes_ if possible, but usually it's safe to assume mapping
        # For binary classification, proba is [prob_class_0, prob_class_1]
        
        predicted_class = prediction
        confidence = max(proba)
        
        return PredictionResponse(
            text=request.text,
            prediction=predicted_class,
            probability=float(confidence)
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
