import joblib
import pandas as pd

MODEL_PATH = "models/xgboost_model.pkl"


def load_model_bundle(model_path: str = MODEL_PATH):
    bundle = joblib.load(model_path)

    if isinstance(bundle, dict) and "model" in bundle:
        return bundle

    feature_names = getattr(bundle, "feature_names_in_", None)
    if feature_names is not None:
        feature_names = list(feature_names)

    return {
        "model": bundle,
        "feature_names": feature_names,
    }


def prepare_features(features, feature_names: list[str] | None):
    if isinstance(features, pd.DataFrame):
        df = features.copy()
    else:
        df = pd.DataFrame([features])

    if not feature_names:
        return df

    missing_features = [feature for feature in feature_names if feature not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features for prediction: {missing_features}")

    return df.loc[:, feature_names]
