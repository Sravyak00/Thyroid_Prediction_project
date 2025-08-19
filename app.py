from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load trained model (make sure your model file is in repo)
model = pickle.load(open("thyroid_model.pkl", "rb"))

app = Flask(__name__)

@app.route("/")
def home():
    return "Thyroid Prediction App is running on Render!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    features = np.array([list(data.values())])
    prediction = model.predict(features)
    return jsonify({"prediction": int(prediction[0])})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
