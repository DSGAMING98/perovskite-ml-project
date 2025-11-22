import os
import re
import pandas as pd
import matplotlib.pyplot as plt


# PATH HELPERS


def get_base_path():
    """
    Returns the root project directory:
    perovskite-ml-project/
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_data_path(filename: str) -> str:
    """
    Full path for a file inside /data/
    """
    return os.path.join(get_base_path(), "data", filename)


def get_model_path(filename: str = "model.pkl") -> str:
    """
    Full path for a model file inside /models/
    """
    return os.path.join(get_base_path(), "models", filename)


def get_results_path(filename: str) -> str:
    """
    Full path for saving plots/results into /results/
    Creates /results/ if it does not exist.
    """
    results_dir = os.path.join(get_base_path(), "results")
    os.makedirs(results_dir, exist_ok=True)
    return os.path.join(results_dir, filename)



# DATA LOADING


def load_dataset(name: str = "perovskite_bandgap_500rows.csv") -> pd.DataFrame:
    """
    Load a CSV dataset from the /data/ folder.
    """
    path = get_data_path(name)
    df = pd.read_csv(path)
    return df


# PERIODIC TABLE (MINI) FOR FEATURE BUILDING
# Z  = atomic number
# EN = electronegativity (Pauling)
# IR = ionic radius (very rough, in Å)


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
    # You can extend this dict with more elements if needed
}


# FORMULA → FEATURES


def parse_formula(formula: str) -> dict:
    """
    Convert a chemical formula like 'CsPbI3' into
    averaged features compatible with the ML model:
        Electronegativity, IonicRadius, AtomicNumber

    Assumes formula is a simple inorganic type without parentheses.
    """
    if not formula or not isinstance(formula, str):
        raise ValueError("Formula must be a non-empty string.")

    # Matches stuff like Cs, Pb, I3, Ba1, etc.
    elements = re.findall(r'([A-Z][a-z]?)(\d*)', formula.strip())

    if not elements:
        raise ValueError(f"Could not parse formula: {formula}")

    EN_list = []
    IR_list = []
    Z_list = []

    for (elem, count_str) in elements:
        count = int(count_str) if count_str else 1

        if elem not in periodic_table:
            raise ValueError(f"Unknown / unsupported element in formula: {elem}")

        data = periodic_table[elem]
        EN_list += [data["EN"]] * count
        IR_list += [data["IR"]] * count
        Z_list  += [data["Z"]]  * count

    # Average the values over all atoms in the formula
    features = {
        "Electronegativity": sum(EN_list) / len(EN_list),
        "IonicRadius":       sum(IR_list) / len(IR_list),
        "AtomicNumber":      sum(Z_list)  / len(Z_list),
    }

    return features

# PLOTTING HELPERS


def plot_feature_importance(model, feature_names):
    """
    Creates and saves a feature importance bar plot to /results/.
    """
    importance = model.feature_importances_

    plt.figure(figsize=(6, 4))
    plt.bar(feature_names, importance, color='purple')
    plt.title("Feature Importance")
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.tight_layout()

    out_path = get_results_path("feature_importance.png")
    plt.savefig(out_path)
    plt.close()

    print(f"[SAVED] Feature importance plot → {out_path}")


def plot_pred_vs_actual(actual, predicted):
    """
    Creates and saves a Predicted vs Actual scatter plot to /results/.
    """
    plt.figure(figsize=(6, 4))
    plt.scatter(actual, predicted, color='blue', alpha=0.7)
    plt.xlabel("Actual Band Gap")
    plt.ylabel("Predicted Band Gap")
    plt.title("Predicted vs Actual Band Gap")
    plt.tight_layout()

    out_path = get_results_path("pred_vs_actual.png")
    plt.savefig(out_path)
    plt.close()

    print(f"[SAVED] Predicted vs Actual plot → {out_path}")
