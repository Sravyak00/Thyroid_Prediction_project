from flask import Flask, request, jsonify, render_template
import pickle, json, numpy as np, os

MODEL_PATH = "thyroid_model.pkl"
FEATURES_PATH = "feature_order.json"

# --- Load model and feature order ---
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("thyroid_model.pkl not found. Make sure to train or upload the model file.")

if not os.path.exists(FEATURES_PATH):
    raise FileNotFoundError("feature_order.json not found. Run training script to generate it.")

model = pickle.load(open(MODEL_PATH, "rb"))
feature_order = json.load(open(FEATURES_PATH))

app = Flask(__name__)

@app.route("/")
def home():
    return "✅ Thyroid Prediction API is running on Render. Visit <a href='/form'>/form</a> to try the web form."

# ---- API endpoint (JSON input) ----
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    try:
        row = [float(data[f]) for f in feature_order]  # enforce feature order
    except KeyError as e:
        return jsonify({
            "error": f"Missing field: {e.args[0]}",
            "required_features": feature_order
        }), 400
    X = np.array([row], dtype=float)
    y = model.predict(X)
    return jsonify({"prediction": int(y[0])})

# ---- Browser form endpoint ----
@app.route("/form", methods=["GET", "POST"])
def form():
    if request.method == "POST":
        try:
            row = [float(request.form[f]) for f in feature_order]
            X = np.array([row], dtype=float)
            y = model.predict(X)
            return render_template("form.html", features=feature_order, result=int(y[0]))
        except Exception as e:
            return render_template("form.html", features=feature_order, error=str(e))
    return render_template("form.html", features=feature_order)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
