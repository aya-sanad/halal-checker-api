from flask import Flask, request, jsonify
import os
import logging
import json
from deep_translator import GoogleTranslator
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import tflite_runtime.interpreter as tflite

# Configure Logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize Flask app
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# Constants
MODEL_PATH = "halal_model.tflite"
TOKENIZER_PATH = "tokenizer.json"
MAX_LENGTH = 50

# Load Tokenizer
if os.path.exists(TOKENIZER_PATH):
    with open(TOKENIZER_PATH, "r", encoding="utf-8") as f:
        tokenizer = tokenizer_from_json(f.read())
    logging.info("✅ Tokenizer loaded successfully!")
else:
    raise FileNotFoundError("❌ Tokenizer file not found!")

# Load TFLite Model
logging.info("📥 Loading TFLite model...")
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
logging.info("✅ TFLite model loaded!")

# Translation Cache
translation_cache = {}

def translate_text(text, src_lang="auto", dest_lang="en"):
    if src_lang == dest_lang or not text.strip():
        return text
    cache_key = f"{text}-{src_lang}-{dest_lang}"
    if cache_key in translation_cache:
        return translation_cache[cache_key]
    try:
        translated = GoogleTranslator(source=src_lang, target=dest_lang).translate(text)
        translation_cache[cache_key] = translated
        return translated
    except Exception as e:
        logging.warning(f"⚠️ Translation failed: {e}")
        return text

# Home Route
@app.route("/")
def home():
    return jsonify({"message": "Welcome to the Halal Ingredient Classification API!"}), 200

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
            detected_lang = "auto"
            detected_languages[ingredient] = detected_lang
            translated_text = translate_text(ingredient, detected_lang, "en").lower()
            translated_ingredients.append(translated_text)
            logging.info(f"🌍 Ingredient: {ingredient} | Translated: {translated_text}")

        sequences = tokenizer.texts_to_sequences(translated_ingredients)
        padded = pad_sequences(sequences, maxlen=MAX_LENGTH, padding="post", truncating="post")

        predictions = []
        for seq in padded:
            input_data = [[float(x) for x in seq]]  # No numpy!
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            predictions.append(output_data[0][0])

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
            logging.info(f"📊 {ing} → Score: {predictions[i]:.4f} → Classified: {classification}")

        ingredient_predictions = [{
            "ingredient": ingredients[i],
            "classification": classifications[i]
        } for i in range(len(ingredients))]

        overall = "halal"
        if "haram" in classifications:
            overall = "haram"
        elif "doubtful" in classifications:
            overall = "doubtful"

        return jsonify({
            "detected_languages": detected_languages,
            "ingredients": ingredient_predictions,
            "overall_classification": overall
        }), 200

    except Exception as e:
        logging.error(f"⚠️ Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Run app on Replit
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=81)
