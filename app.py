# app.py

from flask import Flask, request, jsonify
import torch
import pickle
import re
from model import SentimentRNN

MAX_LEN = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)

# Load vocab
with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

# Load model
model = SentimentRNN(len(vocab)).to(device)
model.load_state_dict(torch.load("sentiment_model.pth", map_location=device))
model.eval()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

def encode_text(text):
    tokens = text.split()
    encoded = [vocab.get(t, vocab['<UNK>']) for t in tokens]
    return encoded[:MAX_LEN] + [0]*(MAX_LEN - len(encoded))

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    review = data.get("review")

    if not review:
        return jsonify({"error": "No review provided"}), 400

    review = clean_text(review)
    encoded = encode_text(review)
    tensor = torch.tensor([encoded]).to(device)

    with torch.no_grad():
        output = model(tensor).item()

    sentiment = "Positive" if output > 0.5 else "Negative"

    return jsonify({
        "review": review,
        "sentiment": sentiment,
        "confidence": float(output)
    })

if __name__ == "__main__":
    app.run(debug=True)