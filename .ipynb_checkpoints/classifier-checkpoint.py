import sys
import pandas as pd
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton,
    QFormLayout, QLineEdit, QMessageBox, QFileDialog
)
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

class ClassifierApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🔍 Thyroid Cancer Diagnosis Prediction")
        self.setMinimumSize(700, 700)
        self.setStyleSheet("background-color: #121212; color: white;")

        self.df = None
        self.columns = []
        self.label_encoders = {}
        self.model = None
        self.scaler = None
        self.target_encoder = None
        self.inputs = {}
        self.form_layout = None

        self.init_ui()

    def init_ui(self):
        self.layout = QVBoxLayout()

        title = QLabel("🔍 Predict Thyroid Cancer Diagnosis")
        title.setFont(QFont("Segoe UI", 20, QFont.Bold))
        title.setStyleSheet("color: #00c2ff; margin-bottom: 20px;")
        self.layout.addWidget(title)

        upload_btn = QPushButton("📂 Upload CSV")
        upload_btn.setFont(QFont("Segoe UI", 14))
        upload_btn.setStyleSheet("background-color: #1f77b4; color: white;")
        upload_btn.clicked.connect(self.upload_csv)
        self.layout.addWidget(upload_btn)

        self.setLayout(self.layout)

    def upload_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV Files (*.csv)")
        if path:
            self.df = pd.read_csv(path)
            self.df.columns = [col.strip() for col in self.df.columns]  # Clean column names

            df = self.df.copy()
            df = df.drop(columns=["Patient_ID", "Thyroid_Cancer_Risk"], errors='ignore')

            self.label_encoders = {}
            for col in df.select_dtypes(include=['object', 'category']).columns:
                if col != "Diagnosis":
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self.label_encoders[col] = le

            target_encoder = LabelEncoder()
            df["Diagnosis"] = target_encoder.fit_transform(df["Diagnosis"])
            self.target_encoder = target_encoder

            X = df.drop("Diagnosis", axis=1)
            y = df["Diagnosis"]
            self.columns = X.columns.tolist()

            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

            self.model = LogisticRegression()
            self.model.fit(X_train, y_train)

            self.build_form()

    def build_form(self):
        if self.form_layout:
            # Remove old form if exists
            for i in reversed(range(self.form_layout.count())):
                self.form_layout.itemAt(i).widget().setParent(None)

        self.form_layout = QFormLayout()
        self.inputs = {}

        for feature in self.columns:
            input_field = QLineEdit()
            self.inputs[feature] = input_field
            self.form_layout.addRow(QLabel(f"{feature.replace('_', ' ')}:"), input_field)

        self.layout.addLayout(self.form_layout)

        predict_btn = QPushButton("✅❌ Predict")
        predict_btn.clicked.connect(self.handle_prediction)
        predict_btn.setStyleSheet("padding: 10px; background-color: #00c2ff; color: white; border-radius: 10px;")
        self.layout.addWidget(predict_btn)

    def handle_prediction(self):
        try:
            user_input = {k: v.text() for k, v in self.inputs.items()}
            for k in user_input:
                if k not in self.label_encoders:
                    user_input[k] = float(user_input[k])
            result, confidence = self.predict(user_input)
            QMessageBox.information(self, "Prediction Result", f"Diagnosis Prediction: {result} ({confidence:.2f}% confidence)")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to predict: {str(e)}")

    def predict(self, input_data):
        df_input = pd.DataFrame([input_data])
        for col, encoder in self.label_encoders.items():
            df_input[col] = encoder.transform([df_input[col][0]])
        X_input = df_input[self.columns]
        X_scaled = self.scaler.transform(X_input)
        probs = self.model.predict_proba(X_scaled)[0]
        prediction = self.model.predict(X_scaled)[0]
        label = self.target_encoder.inverse_transform([prediction])[0]
        confidence = probs[prediction] * 100
        return label, confidence

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ClassifierApp()
    window.show()
    sys.exit(app.exec())
