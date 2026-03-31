import pandas as pd
import joblib
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import log_loss, accuracy_score

DATA_PATH = "data/processed/dataset.csv"
MODEL_PATH = "models/logistic_regression.pkl"


def train():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["win", "game_id"])
    y = df["win"]

    model = LogisticRegression(max_iter=1000)

    cv_results = cross_validate(
        model,
        X,
        y,
        cv=5,
        scoring=["accuracy", "neg_log_loss"],
        return_train_score=True
    )

    print("=== Cross-Validation Results (5-Fold) ===")
    print(f"Accuracy: {cv_results['test_accuracy'].mean():.4f} (+/- {cv_results['test_accuracy'].std():.4f})")
    print(f"Log Loss: {-cv_results['test_neg_log_loss'].mean():.4f} (+/- {cv_results['test_neg_log_loss'].std():.4f})")
    print()

    # Train final model on full dataset and save
    model.fit(X, y)

    # Print feature coefficients
    print("=== Feature Coefficients ===")
    for feature, coef in sorted(zip(X.columns, model.coef_[0]), key=lambda x: abs(x[1]), reverse=True):
        print(f"{feature}: {coef:.4f}")

    joblib.dump(model, MODEL_PATH)
    print("\nModel saved!")


if __name__ == "__main__":
    train()
