import pandas as pd
import joblib
import matplotlib.pyplot as plt

MODEL_PATH = "models/logistic_regression.pkl"


def plot_game_probability(csv_path, game_id):
    model = joblib.load(MODEL_PATH)

    df = pd.read_csv(csv_path)

    df = df[df["game_id"] == game_id]

    X = df.drop(columns=["win", "game_id"])
    probs = model.predict_proba(X)[:, 1]

    plt.figure()
    plt.plot(df["minute"], probs)
    plt.xlabel("Minute")
    plt.ylabel("Win Probability (Blue Team)")
    plt.title(f"Win Probability Over Time ({game_id})")
    plt.show()

if __name__ == "__main__":
    plot_game_probability("data/processed/dataset.csv", "NA1_5500415228")

