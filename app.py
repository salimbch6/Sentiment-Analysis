from flask import Flask, render_template, request, jsonify
from joblib import load
from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Optional: load your local sentiment model (for emotion detection)
MODEL_PATH = Path("models/sentiment.joblib")
model = load(MODEL_PATH) if MODEL_PATH.exists() else None

conversation_history = []  # store chat context locally


# -------- Sentiment (optional) --------
def predict_sentiment(text):
    """Use local model to detect sentiment."""
    if not model:
        return None
    label = model.predict([text])[0]
    proba = model.predict_proba([text])[0].max()
    return label, round(float(proba), 2)


# -------- Main Routes --------
@app.route("/", methods=["GET"])
def index():
    """Display main chat page with Chatbase embed."""
    return render_template("chat.html")


@app.route("/sentiment", methods=["POST"])
def sentiment_api():
    """Optional endpoint: test your sentiment model from frontend."""
    data = request.get_json() or {}
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "missing text"}), 400
    label, conf = predict_sentiment(text)
    return jsonify({"sentiment": label, "confidence": conf})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
