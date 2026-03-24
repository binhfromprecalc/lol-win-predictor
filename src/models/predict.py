import joblib
import pandas as pd

MODEL_PATH = "models/logistic_regression.pkl"


def predict(features: dict):
    model = joblib.load(MODEL_PATH)

    df = pd.DataFrame([features])
    prob = model.predict_proba(df)[0][1]

    return prob


if __name__ == "__main__":
    sample = {
        "gold_diff": 2000,
        "minute": 15
    }

    print("Win Probability:", predict(sample))
