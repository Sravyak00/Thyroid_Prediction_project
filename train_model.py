# train_model.py
import pandas as pd, numpy as np, json, pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

CSV_PATH = "thyroid_cancer_risk_data.csv"     # make sure this file is in repo root
MODEL_PATH = "thyroid_model.pkl"
FEATURES_PATH = "feature_order.json"

# --- Load data
df = pd.read_csv(CSV_PATH)

# --- Heuristic: try common target column names
possible_targets = ["target","label","Outcome","outcome","Class","class","Diagnosis","diagnosis","Risk","risk"]
target = next((c for c in possible_targets if c in df.columns), None)
if target is None:
    raise ValueError(f"Could not find a target column. Add one named any of: {possible_targets}")

# --- Use numeric features only (excluding target)
X = df.select_dtypes(include=[np.number]).drop(columns=[target], errors="ignore")
y = df[target]

if X.empty:
    raise ValueError("No numeric features found. Ensure your CSV has numeric columns for the model.")

# --- Simple pipeline
pipe = Pipeline([
    ("scaler", StandardScaler(with_mean=False)),  # robust for sparse-like inputs
    ("clf", LogisticRegression(max_iter=1000))
])

pipe.fit(X, y)

# --- Save model and the feature order so the API can build vectors correctly
pickle.dump(pipe, open(MODEL_PATH, "wb"))
with open(FEATURES_PATH, "w") as f:
    json.dump(list(X.columns), f)

print(f"Saved {MODEL_PATH} and {FEATURES_PATH}. Features: {list(X.columns)}")
