import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PySide6.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel,
    QFileDialog, QScrollArea, QFrame
)
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

class VisualizationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.df = None
        self.setWindowTitle("📊 Thyroid Risk Visualizations")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("background-color: #121212; color: white;")
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        title = QLabel("📊 Thyroid Risk Prediction System Visualizations")
        title.setFont(QFont("Segoe UI", 20, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #00e6e6; margin: 10px;")
        layout.addWidget(title)

        upload_btn = QPushButton("📂 Upload Dataset")
        upload_btn.setFont(QFont("Segoe UI", 14))
        upload_btn.setStyleSheet("background-color: #444; padding: 10px; border-radius: 10px;")
        upload_btn.clicked.connect(self.load_data)
        layout.addWidget(upload_btn, alignment=Qt.AlignCenter)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        layout.addWidget(self.scroll_area)

        self.setLayout(layout)

    def load_data(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV Files (*.csv)")
        if path:
            self.df = pd.read_csv(path)
            self.df.columns = self.df.columns.str.strip()
            self.show_buttons()

    def show_buttons(self):
        container = QFrame()
        layout = QHBoxLayout(container)
        left = QVBoxLayout()
        right = QVBoxLayout()

        visualizations = [
            ("📊 Risk by Age", self.plot_age_distribution),
            ("⚖️ Gender-wise Risk", self.plot_gender_risk),
            ("🌍 Country-wise Count", self.plot_country_distribution),
            ("📈 TSH vs T3", self.plot_hormone_scatter),
            ("📉 Correlation Heatmap", self.plot_correlation_heatmap),
            ("🧮 Risk vs Obesity/Diabetes", self.plot_obesity_diabetes),
            ("🧪 Hormone Distributions", self.plot_hormone_distribution),
            ("🧬 Family History vs Risk", self.plot_family_history_risk),
            ("☢️ Radiation Risk", self.plot_radiation_risk),
            ("📊 Feature Importance", self.plot_feature_importance)
        ]

        for i, (label, func) in enumerate(visualizations):
            btn = QPushButton(label)
            btn.setFont(QFont("Segoe UI", 12))
            btn.setMinimumHeight(50)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #222;
                    color: white;
                    border-radius: 8px;
                }
                QPushButton:hover {
                    background-color: #333;
                }
            """)
            btn.clicked.connect(func)
            if i % 2 == 0:
                left.addWidget(btn)
            else:
                right.addWidget(btn)

        layout.addLayout(left)
        layout.addLayout(right)
        self.scroll_area.setWidget(container)

    # -------------------- Visualization Methods --------------------

    def plot_age_distribution(self):
        sns.histplot(self.df['Age'], kde=True, color='skyblue')
        plt.title("📊 Age Distribution")
        plt.xlabel("Age")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()

    def plot_gender_risk(self):
        if 'Thyroid_Cancer_Risk' in self.df.columns:
            df = self.df.copy()
            df['Risk_Label'] = LabelEncoder().fit_transform(df['Thyroid_Cancer_Risk'])
            sns.barplot(x='Gender', y='Risk_Label', data=df)
            plt.title("⚖️ Gender vs Risk")
            plt.ylabel("Encoded Risk")
            plt.tight_layout()
            plt.show()

    def plot_country_distribution(self):
        sns.countplot(y='Country', data=self.df, order=self.df['Country'].value_counts().index)
        plt.title("🌍 Patients per Country")
        plt.tight_layout()
        plt.show()

    def plot_hormone_scatter(self):
        x = next((c for c in self.df.columns if 'TSH' in c), None)
        y = next((c for c in self.df.columns if 'T3' in c), None)
        hue = 'Thyroid_Cancer_Risk'
        if x and y and hue in self.df.columns:
            sns.scatterplot(x=x, y=y, hue=hue, data=self.df)
            plt.title(f"📈 {x} vs {y}")
            plt.tight_layout()
            plt.show()

    def plot_correlation_heatmap(self):
        num_df = self.df.select_dtypes(include=['float64', 'int64'])
        sns.heatmap(num_df.corr(), annot=True, cmap='coolwarm')
        plt.title("📉 Correlation Heatmap")
        plt.tight_layout()
        plt.show()

    def plot_obesity_diabetes(self):
        df = self.df.copy()
        grouped = df.groupby(['Obesity', 'Diabetes'])['Thyroid_Cancer_Risk'].count().unstack()
        grouped.plot(kind='bar', stacked=True)
        plt.title("🧮 Obesity + Diabetes vs Risk")
        plt.ylabel("Patient Count")
        plt.tight_layout()
        plt.show()

    def plot_hormone_distribution(self):
        for col in ['TSH', 'T3', 'T4']:
            cname = next((c for c in self.df.columns if col in c), None)
            if cname:
                sns.histplot(self.df[cname], kde=True)
                plt.title(f"🧪 {cname} Distribution")
                plt.tight_layout()
                plt.show()

    def plot_family_history_risk(self):
        sns.countplot(x='Family_History', hue='Thyroid_Cancer_Risk', data=self.df)
        plt.title("🧬 Family History vs Risk")
        plt.tight_layout()
        plt.show()

    def plot_radiation_risk(self):
        sns.countplot(x='Radiation_Exposure', hue='Thyroid_Cancer_Risk', data=self.df)
        plt.title("☢️ Radiation vs Risk")
        plt.tight_layout()
        plt.show()

    def plot_feature_importance(self):
        df = self.df.dropna()
        enc = LabelEncoder()
        for col in df.select_dtypes(include='object'):
            df[col] = enc.fit_transform(df[col])
        X = df.drop('Thyroid_Cancer_Risk', axis=1)
        y = enc.fit_transform(df['Thyroid_Cancer_Risk'])
        model = RandomForestClassifier()
        model.fit(X, y)
        feat_imp = pd.Series(model.feature_importances_, index=X.columns)
        feat_imp.nlargest(10).plot(kind='barh', color='teal')
        plt.title("📊 Feature Importance (Random Forest)")
        plt.tight_layout()
        plt.show()

# Run the app
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = VisualizationApp()
    win.show()
    sys.exit(app.exec())
