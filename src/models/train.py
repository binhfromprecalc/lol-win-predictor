import pandas as pd
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import log_loss, accuracy_score

try:
    from src.models.model_io import MODEL_PATH
except ModuleNotFoundError:
    from model_io import MODEL_PATH

try:
    from xgboost import XGBClassifier
except ImportError as exc:
    raise ImportError(
        "xgboost is required to train this model. Install it with `pip install xgboost`."
    ) from exc

DATA_PATH = "data/processed/dataset.csv"
RANDOM_STATE = 42
DEFAULT_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "tree_method": "hist",
    "n_estimators": 120,
    "max_depth": 2,
    "learning_rate": 0.08,
    "subsample": 0.8,
    "colsample_bytree": 0.9,
    "min_child_weight": 1,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "gamma": 0.2,
    "max_delta_step": 1,
}


def build_model(params: dict | None = None):
    model_params = {**DEFAULT_PARAMS, **(params or {})}
    return XGBClassifier(
        **model_params,
    )


def train():
    df = pd.read_csv(DATA_PATH)

    groups = df["game_id"]
    X = df.drop(columns=["win", "game_id"])
    y = df["win"]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]
    train_groups = groups.iloc[train_idx]
    test_groups = groups.iloc[test_idx]
    model = build_model()

    print("=== Grouped Split Summary ===")
    print(f"Train games: {train_groups.nunique()} | Test games: {test_groups.nunique()}")
    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")
    print()

    print("=== Fixed XGBoost Params ===")
    for key, value in DEFAULT_PARAMS.items():
        print(f"{key}: {value}")
    print()

    model.fit(X_train, y_train)
    test_probs = model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= 0.5).astype(int)

    print("=== Holdout Test Results (grouped by game_id) ===")
    print(f"Accuracy: {accuracy_score(y_test, test_preds):.4f}")
    print(f"Log Loss: {log_loss(y_test, test_probs):.4f}")
    print()

    final_model = build_model()
    # Refit on the full dataset after grouped evaluation so the saved model uses all available games.
    final_model.fit(X, y)

    model_bundle = {
        "model": final_model,
        "feature_names": list(X.columns),
        "params": DEFAULT_PARAMS,
    }

    print("=== Feature Importances ===")
    for feature, importance in sorted(
        zip(X.columns, final_model.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    ):
        print(f"{feature}: {importance:.4f}")

    joblib.dump(model_bundle, MODEL_PATH)
    print("\nModel saved!")


if __name__ == "__main__":
    train()
