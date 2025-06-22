"""
This is a boilerplate pipeline 'modeling'
generated using Kedro 0.19.14
"""

from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# === Trening modelu ===
def train_model(data: pd.DataFrame) -> TabularPredictor:
    label = "Depression"
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    predictor = TabularPredictor(label=label, path="data/06_models/").fit(
        train_data,
        presets="best_quality",
        time_limit=600
    )

    test_data.to_csv("data/05_model_input/test_data.csv", index=False)
    return predictor

# === Zapis wykresu: confusion matrix ===
def save_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Predykcja")
    ax.set_ylabel("Rzeczywista klasa")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig("data/08_reporting/confusion_matrix.png")

# === Zapis wykresu: feature importance ===
def save_feature_importance(predictor, test_data):
    importance = predictor.feature_importance(test_data)
    fig, ax = plt.subplots(figsize=(8, 5))
    importance["importance"].head(10).sort_values().plot(kind='barh', ax=ax)
    ax.set_title("Najważniejsze cechy (Feature Importance)")
    fig.tight_layout()
    fig.savefig("data/08_reporting/feature_importance.png")

# === Ewaluacja modelu ===
def evaluate_model(_: pd.DataFrame) -> str:
    predictor = TabularPredictor.load("data/06_models/")
    test_data = pd.read_csv("data/05_model_input/test_data.csv")

    y_true = test_data["Depression"]
    y_pred = predictor.predict(test_data.drop(columns=["Depression"]))

    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred))

    print("\n--- Confusion Matrix ---")
    print(confusion_matrix(y_true, y_pred))

    print("\n--- Leaderboard ---")
    leaderboard = predictor.leaderboard(test_data, silent=True)
    print(leaderboard)

    print("\n--- Feature importance ---")
    importance = predictor.feature_importance(test_data)
    print(importance.head(10))

    # Zapisz obrazy
    save_confusion_matrix(y_true, y_pred)
    save_feature_importance(predictor, test_data)

    return "ok"
