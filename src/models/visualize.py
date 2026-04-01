import pandas as pd
import matplotlib.pyplot as plt

try:
    from src.models.model_io import MODEL_PATH, load_model_bundle, prepare_features
except ModuleNotFoundError:
    from model_io import MODEL_PATH, load_model_bundle, prepare_features


def plot_game_probability(csv_path, game_id):
    bundle = load_model_bundle(MODEL_PATH)
    model = bundle["model"]

    df = pd.read_csv(csv_path)

    df = df[df["game_id"] == game_id]

    X = prepare_features(df.drop(columns=["win", "game_id"]), bundle.get("feature_names"))
    probs = model.predict_proba(X)[:, 1]

    plt.figure()
    plt.plot(df["minute"], probs)
    plt.xlabel("Minute")
    plt.ylabel("Win Probability (Blue Team)")
    plt.title(f"Win Probability Over Time ({game_id})")
    plt.show()

if __name__ == "__main__":
    plot_game_probability("data/processed/dataset.csv", "NA1_5497255828")

