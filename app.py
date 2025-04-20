from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import os
import logging
from tensorflow.keras.preprocessing.sequence import pad_sequences
from deep_translator import GoogleTranslator
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# Configure Logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize Flask app
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # Support multilingual output

# Load Model & Tokenizer
MODEL_PATH = "halal_model.h5"
TOKENIZER_PATH = "tokenizer.json"
MAX_LENGTH = 50

logging.info("📥 Loading trained LSTM model...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    model.predict(np.zeros((1, MAX_LENGTH)))  # Warm-up model
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
    return jsonify(
        {"message":
         "Welcome to the Halal Ingredient Classification API!"}), 200


# Translation Function with Caching
translation_cache = {}


def translate_text(text, src_lang="auto", dest_lang="en"):
    """Translates text with caching to optimize performance."""
    if src_lang == dest_lang or not text.strip():
        return text
    cache_key = f"{text}-{src_lang}-{dest_lang}"
    if cache_key in translation_cache:
        return translation_cache[cache_key]
    try:
        translated = GoogleTranslator(source=src_lang,
                                      target=dest_lang).translate(text)
        translation_cache[cache_key] = translated
        return translated
    except Exception as e:
        logging.warning(f"⚠️ Translation failed: {e}")
        return text


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

        for ingredient in ingredients:
            # Detect language skipped; using auto-detection by GoogleTranslator
            detected_lang = "auto"
            detected_languages[ingredient] = detected_lang
            translated_text = translate_text(ingredient, detected_lang,
                                             "en").lower()
            translated_ingredients.append(translated_text)
            logging.info(
                f"🌍 Ingredient: {ingredient} | Translated: {translated_text}")

        processed_texts = pad_sequences(
            tokenizer.texts_to_sequences(translated_ingredients),
            maxlen=MAX_LENGTH,
            padding="post",
            truncating="post")
        predictions = model.predict(processed_texts).flatten()
        classifications = []

        for i, ing in enumerate(ingredients):
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
            logging.info(
                f"📊 {ing} → Score: {predictions[i]:.4f} → Classified: {classification}"
            )

        ingredient_predictions = [{
            "ingredient": ingredients[i],
            "classification": classifications[i]
        } for i in range(len(ingredients))]

        if "haram" in classifications:
            overall_classification = "haram"
        elif "doubtful" in classifications:
            overall_classification = "doubtful"
        else:
            overall_classification = "halal"

        return jsonify({
            "detected_languages": detected_languages,
            "ingredients": ingredient_predictions,
            "overall_classification": overall_classification
        }), 200

    except Exception as e:
        logging.error(f"⚠️ Error: {str(e)}")
        return jsonify({"error": str(e)}), 500


# Run Server (Replit style)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=81)
