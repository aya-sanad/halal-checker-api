from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import os
import logging
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from deep_translator import GoogleTranslator  
import langid
from functools import lru_cache

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize Flask app
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # Support multilingual output

# Load Model & Tokenizer
MODEL_PATH = "halal_model.h5"
TOKENIZER_PATH = "tokenizer.json"
MAX_LENGTH = 50

# Warm-up model loading on the first request
logging.info("📥 Loading trained LSTM model...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    logging.info("✅ Model loaded successfully!")
except Exception as e:
    logging.error(f"❌ Model loading failed: {e}")
    raise RuntimeError(f"❌ Model loading failed: {e}")

# Load Tokenizer
if os.path.exists(TOKENIZER_PATH):
    with open(TOKENIZER_PATH, "r", encoding="utf-8") as f:
        tokenizer = tokenizer_from_json(f.read())
    logging.info("✅ Tokenizer loaded successfully!")
else:
    raise FileNotFoundError("❌ Tokenizer file not found!")

# Home Route
@app.route("/")
def home():
    return jsonify({"message": "Welcome to the Halal Ingredient Classification API!"}), 200

# Translation Function with Caching (Improved using lru_cache)
@lru_cache(maxsize=100)
def translate_text(text, src_lang, dest_lang="en"):
    """Translates text with caching to optimize performance."""
    if src_lang == dest_lang or not text.strip():
        return text  
    try:
        translated = GoogleTranslator(source=src_lang, target=dest_lang).translate(text)
        return translated
    except Exception as e:
        logging.warning(f"⚠️ Translation failed: {e}")
        return text  

# Enhanced Language Detection with fallback
def detect_language(text):
    """Detects language with a fallback for misclassified English words."""
    detected_lang = langid.classify(text)[0]
    
    # Correct common misclassification: 'beef gelatin' → 'nl' (Dutch)
    if detected_lang == "nl" and "gelatin" in text.lower():
        return "en"
    
    # Add additional checks for known common misclassifications
    return detected_lang

# Prediction Route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        ingredients = data.get("ingredients", [])
        if not ingredients:
            return jsonify({"error": "❌ No ingredients provided!"}), 400

        detected_languages = {}
        translated_ingredients = []

        # Detect Language & Translate
        for ingredient in ingredients:
            detected_lang = detect_language(ingredient)
            detected_languages[ingredient] = detected_lang
            translated_text = translate_text(ingredient, detected_lang, "en").lower()
            translated_ingredients.append(translated_text)
            logging.info(f"🌍 Ingredient: {ingredient} | Lang: {detected_lang} | Translated: {translated_text}")

        # Prepare inputs for model
        processed_texts = pad_sequences(tokenizer.texts_to_sequences(translated_ingredients), maxlen=MAX_LENGTH, padding="post", truncating="post")

        # Make predictions
        predictions = model.predict(processed_texts).flatten()
        classifications = []

        for i, ing in enumerate(ingredients):
            # Rule-based override for beef gelatin
            if "beef gelatin" in translated_ingredients[i]:
                classification = "doubtful"
            else:
                score = predictions[i]
                if score >= 0.7:
                    classification = "halal"
                elif score <= 0.3:
                    classification = "haram"
                else:
                    classification = "doubtful"

            classifications.append(classification)
            logging.info(f"📊 {ing} → Score: {predictions[i]:.4f} → Classified: {classification}")

        # Prepare response
        ingredient_predictions = [{"ingredient": ingredients[i], "classification": classifications[i]} for i in range(len(ingredients))]

        # Determine overall classification
        if "haram" in classifications:
            overall_classification = "haram"
        elif "doubtful" in classifications:
            overall_classification = "doubtful"
        else:
            overall_classification = "halal"


        # Return the response with ingredient details and overall classification
        response = {
            "detected_languages": detected_languages,
            "ingredients": ingredient_predictions,
            "overall_classification": overall_classification
        }

        return jsonify(response), 200

    except Exception as e:
        logging.error(f"⚠️ Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Run Flask Server
if __name__ == "__main__":
    from waitress import serve  
    logging.info("🚀 Starting Flask server at: http://127.0.0.1:5000/")
    serve(app, host="0.0.0.0", port=5000, threads=4)
