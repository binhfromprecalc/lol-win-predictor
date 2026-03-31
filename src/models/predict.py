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
        "minute": 15,
        "kills_diff": 10,
        "towers_diff": 2,
        "dragons_diff": 1,
        # "grubs_diff": 0,
        # "herald_diff": 0,
        # "baron_diff": 0,
        "gold_diff_per_min": 200,
    }

    print("Win Probability:", predict(sample))
