import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# PATH HELPERS
# =========================

def get_base_path() -> str:
    """
    Returns the root project directory:
    perovskite-ml-project/
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_data_path(filename: str) -> str:
    """
    Full path for a file inside /data/.
    Creates /data/ if it does not exist.
    """
    data_dir = os.path.join(get_base_path(), "data")
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, filename)


def get_model_path(filename: str = "model.pkl") -> str:
    """
    Full path for a model file inside /models/.
    Creates /models/ if it does not exist.
    """
    models_dir = os.path.join(get_base_path(), "models")
    os.makedirs(models_dir, exist_ok=True)
    return os.path.join(models_dir, filename)


def get_results_path(filename: str) -> str:
    """
    Full path for saving plots/results into /results/.
    Creates /results/ if it does not exist.
    """
    results_dir = os.path.join(get_base_path(), "results")
    os.makedirs(results_dir, exist_ok=True)
    return os.path.join(results_dir, filename)


# =========================
# DATA LOADING HELPERS
# =========================

def load_dataset(name: str) -> pd.DataFrame:
    """
    Load a single CSV dataset from the /data/ folder.
    """
    path = get_data_path(name)
    return pd.read_csv(path)


def load_batches(
    pattern: str = "perovskite_bandgap_batch_{i}_100k.csv",
    num_batches: int = 5,
) -> pd.DataFrame:
    """
    Load and concatenate multiple batch CSVs from /data/.

    Default expects:
      perovskite_bandgap_batch_1_100k.csv
      ...
      perovskite_bandgap_batch_5_100k.csv
    """
    dfs = []
    for i in range(1, num_batches + 1):
        fname = pattern.format(i=i)
        path = get_data_path(fname)
        df = pd.read_csv(path)
        dfs.append(df)
    full_df = pd.concat(dfs, ignore_index=True)
    return full_df


# =========================
# MINI PERIODIC TABLE
# =========================

# Z  = atomic number
# EN = electronegativity (Pauling)
# IR = ionic radius (Å, very rough)
periodic_table = {
    "H":  {"Z": 1,  "EN": 2.20, "IR": 0.25},
    "He": {"Z": 2,  "EN": 0.00, "IR": 0.31},
    "Li": {"Z": 3,  "EN": 0.98, "IR": 0.76},
    "Be": {"Z": 4,  "EN": 1.57, "IR": 0.31},
    "B":  {"Z": 5,  "EN": 2.04, "IR": 0.27},
    "C":  {"Z": 6,  "EN": 2.55, "IR": 0.16},
    "N":  {"Z": 7,  "EN": 3.04, "IR": 0.13},
    "O":  {"Z": 8,  "EN": 3.44, "IR": 0.14},
    "F":  {"Z": 9,  "EN": 3.98, "IR": 0.14},
    "Na": {"Z": 11, "EN": 0.93, "IR": 1.02},
    "Mg": {"Z": 12, "EN": 1.31, "IR": 0.72},
    "Al": {"Z": 13, "EN": 1.61, "IR": 0.53},
    "Si": {"Z": 14, "EN": 1.90, "IR": 0.40},
    "P":  {"Z": 15, "EN": 2.19, "IR": 0.35},
    "S":  {"Z": 16, "EN": 2.58, "IR": 0.30},
    "Cl": {"Z": 17, "EN": 3.16, "IR": 0.27},
    "K":  {"Z": 19, "EN": 0.82, "IR": 1.38},
    "Ca": {"Z": 20, "EN": 1.00, "IR": 1.00},
    "Ti": {"Z": 22, "EN": 1.54, "IR": 0.61},
    "Br": {"Z": 35, "EN": 2.96, "IR": 1.96},
    "I":  {"Z": 53, "EN": 2.66, "IR": 2.20},
    "Cs": {"Z": 55, "EN": 0.79, "IR": 1.67},
    "Ba": {"Z": 56, "EN": 0.89, "IR": 1.35},
    "Pb": {"Z": 82, "EN": 2.33, "IR": 1.33},
    # extend if needed
}


# =========================
# FORMULA → FEATURES
# =========================

def parse_formula(formula: str) -> dict:
    """
    Convert a formula like 'CsPbI3' into averaged features:

        - Electronegativity
        - IonicRadius
        - AtomicNumber

    Simple inorganic formula parsing (no parentheses).
    """
    if not formula or not isinstance(formula, str):
        raise ValueError("Formula must be a non-empty string.")

    formula = formula.strip()

    # (Element, count) pairs: Cs, Pb, I3 etc.
    elements = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
    if not elements:
        raise ValueError(f"Could not parse formula: {formula}")

    EN_list, IR_list, Z_list = [], [], []

    for elem, count_str in elements:
        count = int(count_str) if count_str else 1

        if elem not in periodic_table:
            raise ValueError(f"Unknown / unsupported element in formula: {elem}")

        data = periodic_table[elem]
        EN_list += [data["EN"]] * count
        IR_list += [data["IR"]] * count
        Z_list  += [data["Z"]]  * count

    features = {
        "Electronegativity": sum(EN_list) / len(EN_list),
        "IonicRadius":       sum(IR_list) / len(IR_list),
        "AtomicNumber":      sum(Z_list)  / len(Z_list),
    }
    return features


def formula_to_dataframe(formula: str) -> pd.DataFrame:
    """
    Helper: formula string → 1-row DataFrame of features.
    """
    feats = parse_formula(formula)
    return pd.DataFrame([feats])


def is_abx3(formula: str) -> bool:
    """
    Check if a formula roughly matches an ABX3-type perovskite:
    - exactly 3 distinct elements
    - last element has stoichiometry 3 (e.g. CsPbI3, BaTiO3)
    """
    if not formula or not isinstance(formula, str):
        return False

    formula = formula.strip()
    elements = re.findall(r"([A-Z][a-z]?)(\d*)", formula)

    # must be exactly 3 elements
    if len(elements) != 3:
        return False

    counts = []
    for elem, count_str in elements:
        if elem not in periodic_table:
            return False
        count = int(count_str) if count_str else 1
        if count <= 0:
            return False
        counts.append(count)

    # A, B can be whatever (1,2...), but X must be 3
    if counts[2] != 3:
        return False

    return True


# =========================
# PLOTTING HELPERS
# =========================

def plot_feature_importance(model, feature_names):
    """
    Feature importance bar plot → /results/feature_importance.png
    """
    importance = model.feature_importances_

    plt.figure(figsize=(6, 4))
    plt.bar(feature_names, importance)
    plt.title("Feature Importance")
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.tight_layout()

    out_path = get_results_path("feature_importance.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"[SAVED] Feature importance plot → {out_path}")


def plot_pred_vs_actual(actual, predicted):
    """
    Standard scatter plot: Predicted vs Actual band gaps.
    Saved as /results/pred_vs_actual.png
    """
    actual = np.array(actual)
    predicted = np.array(predicted)

    plt.figure(figsize=(5, 4))
    plt.scatter(actual, predicted, alpha=0.7)
    plt.xlabel("Actual Band Gap (eV)")
    plt.ylabel("Predicted Band Gap (eV)")
    plt.title("Predicted vs Actual Band Gaps")
    plt.tight_layout()

    out_path = get_results_path("pred_vs_actual.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"[SAVED] Predicted vs Actual scatter → {out_path}")


def plot_pred_vs_actual_band_like(actual, predicted, max_points: int = 200):
    """
    'Band-style' comparison:
    - Sort samples by true band gap
    - Plot true (black) vs predicted (red) as lines, similar to band-structure style.

    Saved as /results/pred_vs_actual_band_style.png
    """
    actual = np.array(actual)
    predicted = np.array(predicted)

    # sort by true band gap
    idx = np.argsort(actual)
    actual_sorted = actual[idx]
    pred_sorted = predicted[idx]

    # optional truncation for cleaner figure
    if max_points is not None and max_points < len(actual_sorted):
        actual_sorted = actual_sorted[:max_points]
        pred_sorted = pred_sorted[:max_points]

    x = np.arange(len(actual_sorted))

    plt.figure(figsize=(8, 3))
    plt.plot(x, actual_sorted, "k-", label="True band gap")       # black
    plt.plot(x, pred_sorted, "r--", label="Predicted band gap")   # red dashed
    plt.xlabel("Test samples (sorted by true band gap)")
    plt.ylabel("Band gap (eV)")
    plt.title("Predicted (red) vs True (black) Band Gaps")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()

    out_path = get_results_path("pred_vs_actual_band_style.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"[SAVED] Band-style Predicted vs True plot → {out_path}")
