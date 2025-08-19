# app.py
from flask import Flask, request, jsonify
import pickle, json, numpy as np, os

MODEL_PATH = "thyroid_model.pkl"
FEATURES_PATH = "feature_order.json"

# Ensure model exists (if not, give a helpful error)
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("thyroid_model.pkl not found. Build step must run train_model.py.")

model = pickle.load(open(MODEL_PATH, "rb"))
feature_order = json.load(open(FEATURES_PATH))

app = Flask(__name__)

@app.route("/")
def home():
    return "✅ Thyroid Prediction API is running on Render."

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    # Build feature row in the exact order used for training
    try:
        row = [float(data[col]) for col in feature_order]
    except KeyError as e:
        return jsonify({
            "error": f"Missing required feature: {e.args[0]}",
            "required_features": feature_order
        }), 400

    X = np.array([row], dtype=float)
    y = model.predict(X)
    return jsonify({"prediction": int(y[0])})
