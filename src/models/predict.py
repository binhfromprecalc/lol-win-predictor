try:
    from src.models.model_io import MODEL_PATH, load_model_bundle, prepare_features
except ModuleNotFoundError:
    from model_io import MODEL_PATH, load_model_bundle, prepare_features


def predict(features: dict):
    bundle = load_model_bundle(MODEL_PATH)
    model = bundle["model"]
    df = prepare_features(features, bundle.get("feature_names"))
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
