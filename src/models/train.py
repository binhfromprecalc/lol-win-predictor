import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score

DATA_PATH = "data/processed/dataset.csv"
MODEL_PATH = "models/logistic_regression.pkl"


def train():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["win", "game_id"])
    y = df["win"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    preds = model.predict_proba(X_test)[:, 1]

    print("Log Loss:", log_loss(y_test, preds))
    print("Accuracy:", accuracy_score(y_test, preds > 0.5))

    joblib.dump(model, MODEL_PATH)
    print("Model saved!")


if __name__ == "__main__":
    train()
