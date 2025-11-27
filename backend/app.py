"""
Flask API for Toxicity Classification and Gemini Commentary Proxy
Uses pretrained toxicity.h5 model (6-label Jigsaw model)
The Gemini API key is securely loaded from a .env file.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM
import warnings
import os
import json
import re
import requests # NEW: Import requests for making external API calls
from dotenv import load_dotenv # NEW: Import load_dotenv for reading the .env file

# Load environment variables from .env file
# This must happen early to load GEMINI_API_KEY
load_dotenv() 

# Suppress TF warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")

# Flask app
app = Flask(__name__)
CORS(app)

# Config
MODEL_PATH = "toxicity.h5"
VOCAB_PATH = "vocab.json"   # exported from Colab via vectorizer.get_vocabulary()
MAX_SEQUENCE_LENGTH = 1800  # must match training

# NEW: Gemini API Configuration loaded from environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent"


model = None
token_to_id = None

# Categories (order must match model output!)
TOXICITY_LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

########################
# Loading utilities
########################

def load_toxicity_model():
    """Load toxicity.h5 with LSTM custom object."""
    global model
    try:
        print(f"Loading model from {MODEL_PATH}...")
        custom_objects = {"LSTM": LSTM}
        try:
            model = load_model(MODEL_PATH, custom_objects=custom_objects)
        except Exception:
            model = load_model(MODEL_PATH, compile=False, custom_objects=custom_objects)
            model.compile(
                optimizer="adam",
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )
        print("✅ Model loaded successfully!")
        return True
    except FileNotFoundError:
        print(f"❌ ERROR: {MODEL_PATH} not found")
        return False
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False


def load_vocab():
    """Load vocabulary list exported from TextVectorization."""
    global token_to_id
    try:
        print(f"Loading vocabulary from {VOCAB_PATH}...")
        with open(VOCAB_PATH, "r", encoding="utf-8") as f:
            vocab = json.load(f)  # list of tokens, index = id
        token_to_id = {tok: idx for idx, tok in enumerate(vocab)}
        print(f"✅ Vocabulary loaded! Size = {len(token_to_id)}")
        return True
    except FileNotFoundError:
        print(f"❌ ERROR: {VOCAB_PATH} not found")
        return False
    except Exception as e:
        print(f"❌ Error loading vocabulary: {e}")
        return False

########################
# Preprocessing
########################

_tokenizer_re = re.compile(r"[a-z0-9']+")

def simple_tokenize(text: str):
    """Lowercase + simple word tokenizer similar to TextVectorization."""
    text = text.lower()
    return _tokenizer_re.findall(text)


def preprocess_text(text, max_len=MAX_SEQUENCE_LENGTH):
    """
    Convert text -> integer sequence using exported vocab and pad to max_len.
    """
    if token_to_id is None:
        raise RuntimeError("Vocabulary not loaded")

    tokens = simple_tokenize(text)
    seq = []

    # TextVectorization: index 0 is usually padding, 1 is OOV
    pad_id = token_to_id.get("", 0)      # often "" token
    oov_id = token_to_id.get("[UNK]", 1)  # or whatever your vocab uses; falls back to 1

    for tok in tokens:
        seq.append(token_to_id.get(tok, oov_id))

    if len(seq) > max_len:
        seq = seq[:max_len]
    else:
        seq += [pad_id] * (max_len - len(seq))

    return np.array([seq], dtype="int32")

########################
# NEW: LLM Commentary Logic
########################

def get_gemini_commentary(comment: str) -> str:
    """
    Calls the Gemini API using the secure key from the environment.
    """
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        print("⚠️ WARNING: GEMINI_API_KEY is not configured or is the placeholder.")
        return "BEEP BOOP! Failed to generate quirky analysis. API Key not configured on the server."

    system_prompt = "You are a sassy, quirky, and slightly cynical AI analyst. Provide a short, funny, 3-4 line analysis of the comment based on its content. Do not repeat the comment. Use a fun, cartoonish tone."
    user_query = f"Analyze this comment: \"{comment}\""

    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
    }

    try:
        # Construct the API URL with the key
        url = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}"
        
        # Use requests library to make the API call
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=15 # Set a timeout
        )
        response.raise_for_status() # Raise exception for bad status codes (4xx or 5xx)
        
        result = response.json()
        
        # Extract the generated text
        text = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text")
        
        if text:
            return text
        else:
            return "The AI agent is taking a coffee break. (No text generated by LLM)"

    except requests.exceptions.RequestException as e:
        print(f"❌ Gemini API Call Error: {e}")
        return "BEEP BOOP! Failed to generate quirky analysis due to network or API error on server side."
    except Exception as e:
        print(f"❌ Unexpected Error during Gemini processing: {e}")
        return "BEEP BOOP! Failed to generate quirky analysis due to unexpected server error."


########################
# Routes
########################

@app.route("/api/classify", methods=["POST"])
def classify():
    """Classify comment, generate LLM commentary, and return all scores."""
    try:
        data = request.get_json()
        if not data or "comment" not in data:
            return jsonify({"error": "Missing 'comment'"}), 400

        comment = data["comment"].strip()
        if not comment:
            return jsonify({"error": "Comment cannot be empty"}), 400

        if model is None or token_to_id is None:
            return jsonify({"error": 'Model or vocabulary not loaded'}), 500

        # 1. Toxicity Prediction
        x = preprocess_text(comment)
        preds = model.predict(x, verbose=0)[0]  # shape (6,)
        scores = preds.tolist()

        overall = float(np.mean(scores))
        is_toxic = overall > 0.5
        
        # 2. LLM Commentary (NEW STEP)
        commentary = get_gemini_commentary(comment)

        # 3. Return Combined Result, including the new analysis_text field
        return jsonify(
            {
                "success": True,
                "comment": comment,
                "toxic_score": round(overall, 4),
                "is_toxic": is_toxic,
                "percentage": round(overall * 100, 2),
                "message": "Toxic! ⚠️" if is_toxic else "Safe ✅",
                "detailed_scores": [round(float(s), 4) for s in scores],
                "categories": TOXICITY_LABELS,
                "analysis_text": commentary, # The key the frontend expects
            }
        ), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def health():
    status = "healthy"
    message = "All systems operational."
    
    if model is None or token_to_id is None:
        status = "unhealthy"
        message = "TensorFlow model or vocabulary not loaded."
    
    # Check if LLM API key is present
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        status = "degraded"
        message += " WARNING: Gemini API Key not found or is placeholder. LLM Commentary may fail."

    return jsonify({"status": status, "message": message, "model": MODEL_PATH}), 200


@app.route("/", methods=["GET"])
def index():
    return jsonify({"name": "Toxicity Classifier API", "version": "2.0 (with LLM Proxy)"}), 200

########################
# Main
########################

if __name__ == "__main__":
    print("=" * 70)
    print("🚀 Toxicity Classifier API Starting...")
    print(f"Gemini API Key Loaded: {'Yes' if GEMINI_API_KEY and GEMINI_API_KEY != 'YOUR_GEMINI_API_KEY_HERE' else 'No/Warning'}")
    print("=" * 70)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Model file: {MODEL_PATH}")
    print(f"Vocab file: {VOCAB_PATH}")

    if load_toxicity_model() and load_vocab():
        print("✅ Ready to classify!")
        print("🌐 API running at http://localhost:5000")
        app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
    else:
        print("❌ Failed to start (model/vocab load error)")