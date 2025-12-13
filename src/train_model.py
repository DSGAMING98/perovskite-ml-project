import os
import pickle

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# If you already have get_model_path in utils, we reuse it.
# Otherwise, we fallback to a local implementation.
try:
    from utils import get_model_path
except ImportError:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR, exist_ok=True)

    def get_model_path(filename="model.pkl"):
        return os.path.join(MODELS_DIR, filename)



# CONFIG


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

DATASET_NAME = "perovskite_bandgap_60000rows.csv"
DATASET_PATH = os.path.join(DATA_DIR, DATASET_NAME)

FEATURE_COLS = ["Electronegativity", "IonicRadius", "AtomicNumber"]
TARGET_COL = "BandGap"

N_ESTIMATORS = 250      # tunable: fewer trees = smaller model
MAX_DEPTH = None        # or set e.g. 20 to shrink further
RANDOM_STATE = 42
TEST_SIZE = 0.2


def ensure_dirs():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR, exist_ok=True)


def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(6, 4))
    plt.title("Feature importance")
    plt.bar(range(len(feature_names)), importances[indices])
    plt.xticks(range(len(feature_names)), np.array(feature_names)[indices], rotation=20)
    plt.tight_layout()

    out_path = os.path.join(RESULTS_DIR, "feature_importance.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved feature importance plot → {out_path}")


def plot_pred_vs_actual(y_true, y_pred):
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, s=6, alpha=0.6)
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1)
    plt.xlabel("Actual band gap (eV)")
    plt.ylabel("Predicted band gap (eV)")
    plt.title("Predicted vs actual band gaps")
    plt.tight_layout()

    out_path = os.path.join(RESULTS_DIR, "pred_vs_actual.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved predicted vs actual plot → {out_path}")


def plot_pred_vs_actual_band_like(y_true, y_pred, max_points=200):
    # Sort by true band gap and take subset for a smooth "band" look
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    idx = np.argsort(y_true)
    y_true_sorted = y_true[idx]
    y_pred_sorted = y_pred[idx]

    if max_points is not None and len(y_true_sorted) > max_points:
        step = len(y_true_sorted) // max_points
        y_true_sorted = y_true_sorted[::step]
        y_pred_sorted = y_pred_sorted[::step]

    x_axis = np.arange(len(y_true_sorted))

    plt.figure(figsize=(7, 4))
    plt.plot(x_axis, y_true_sorted, label="Actual", linewidth=1.5)
    plt.plot(x_axis, y_pred_sorted, label="Predicted", linewidth=1.5, alpha=0.85)
    plt.xlabel("Sample index (sorted by actual band gap)")
    plt.ylabel("Band gap (eV)")
    plt.title("Predicted vs actual band gaps (band-style curve)")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(RESULTS_DIR, "pred_vs_actual_band_style.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved band-style predicted vs actual plot → {out_path}")


def main():
    print("=== Perovskite Band Gap Model Training (60k dataset) ===")
    ensure_dirs()


    # 1. Load dataset

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}. "
            f"Make sure perovskite_bandgap_60000rows.csv is in /data."
        )

    print(f"[INFO] Loading dataset from: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)

    missing_cols = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing columns: {missing_cols}")

    print(f"[INFO] Dataset shape: {df.shape}")
    print(df.head())


    # 2. Split features / target

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    print(f"[INFO] Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")


    # 3. Train model

    print(
        f"[INFO] Training RandomForestRegressor("
        f"n_estimators={N_ESTIMATORS}, max_depth={MAX_DEPTH})"
    )
    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    model.fit(X_train, y_train)


    # 4. Evaluate

    print("[INFO] Evaluating on test set...")
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    print(f"[METRICS] MAE  = {mae:.4f} eV")
    print(f"[METRICS] RMSE = {rmse:.4f} eV")
    print(f"[METRICS] R²   = {r2:.4f}")


    # 5. Save model

    model_path = get_model_path("model.pkl")
    print(f"[INFO] Saving model to: {model_path}")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)


    # 6. Plots → results/

    print("[INFO] Creating plots in /results ...")
    plot_feature_importance(model, FEATURE_COLS)
    plot_pred_vs_actual(y_test, y_pred)
    plot_pred_vs_actual_band_like(y_test, y_pred, max_points=200)

    print("=== Training complete. Model + results updated. ===")


if __name__ == "__main__":
    main()
