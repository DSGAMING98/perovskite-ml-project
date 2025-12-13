import os
import sys
import pickle

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# =========================
# PATH SETUP
# =========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

# Import helpers from utils (you already have these or similar)
from utils import (
    formula_to_dataframe,
    is_abx3,
    get_model_path,
)

# =========================
# MODEL LOADING
# =========================

@st.cache_resource(show_spinner=True)
def load_model():
    model_path = get_model_path("model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at: {model_path}\n\n"
            "Run `python src/train_model.py` first so the model and plots are generated."
        )
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


model = load_model()


# =========================
# HELPER FUNCTIONS
# =========================

def classify_bandgap(bg_value: float) -> str:
    """Text interpretation for band-gap value."""
    if bg_value < 1.3:
        return "Very low band gap – strong absorber (good for near-IR / deep visible)."
    elif bg_value < 1.9:
        return "Optimal solar-absorber window (good candidate for high-efficiency cells)."
    elif bg_value < 3.0:
        return "Moderate band gap – still useful (visible / UV mixed response)."
    else:
        return "Wide band gap – more insulating / UV-focused behaviour."


def solar_flag(bg_value: float) -> str:
    """Rough suitability tag for solar active layer."""
    if 1.1 <= bg_value <= 1.8:
        return "✅ Likely suitable as a solar absorber."
    elif 0.8 <= bg_value < 1.1 or 1.8 < bg_value <= 2.1:
        return "🟡 Borderline / niche use – could work in tandem or special stack."
    else:
        return "🔴 Unfavourable as a primary absorber (but might be useful elsewhere in the device)."


def descriptor_plot(row: pd.Series):
    """Bar chart of EN, IR, Z for a single composition."""
    labels = ["Electronegativity", "Ionic radius (Å)", "Atomic number"]
    values = [
        float(row["Electronegativity"]),
        float(row["IonicRadius"]),
        float(row["AtomicNumber"]),
    ]

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(labels, values)
    ax.set_ylabel("Value")
    ax.set_title("Elemental descriptor profile")
    plt.xticks(rotation=20)
    plt.tight_layout()
    return fig


def get_insight_images():
    feat_imp = os.path.join(RESULTS_DIR, "feature_importance.png")
    pred_band = os.path.join(RESULTS_DIR, "pred_vs_actual_band_style.png")
    pred_plain = os.path.join(RESULTS_DIR, "pred_vs_actual.png")

    if not os.path.exists(feat_imp):
        feat_imp = None
    if not os.path.exists(pred_band):
        pred_band = pred_plain if os.path.exists(pred_plain) else None

    return feat_imp, pred_band


def batch_predict_formulas(formula_text: str):
    """
    Take multi-line / comma-separated formulas,
    filter valid ABX3 ones, and return prediction table.
    """
    raw_items = [f.strip() for f in formula_text.replace(",", "\n").splitlines()]
    formulas = [f for f in raw_items if f]  # drop empty

    records = []
    skipped = []

    for f in formulas:
        if not is_abx3(f):
            skipped.append(f)
            continue
        try:
            df_feat = formula_to_dataframe(f)
            pred = float(model.predict(df_feat)[0])
            records.append(
                {
                    "Formula": f,
                    "Electronegativity": df_feat["Electronegativity"].iloc[0],
                    "IonicRadius": df_feat["IonicRadius"].iloc[0],
                    "AtomicNumber": df_feat["AtomicNumber"].iloc[0],
                    "BandGap_eV": round(pred, 3),
                    "Solar_Assessment": solar_flag(pred),
                }
            )
        except Exception:
            skipped.append(f)

    result_df = pd.DataFrame(records) if records else pd.DataFrame()

    return result_df, skipped


# =========================
# STREAMLIT LAYOUT
# =========================

st.set_page_config(
    page_title="Perovskite Band Gap Studio",
    layout="wide",
)

# ----- SIDEBAR -----
with st.sidebar:
    st.title("Perovskite ML Lab")
    st.markdown(
        """
This tool uses a trained Random Forest model on a synthetic
**60,000-sample ABX₃ dataset** to estimate band gaps from very
simple descriptors.

Use it to:
- Explore new formulas
- Compare candidates
- Guess solar suitability
        """
    )
    data_path = os.path.join(DATA_DIR, "perovskite_bandgap_60000rows.csv")
    if os.path.exists(data_path):
        try:
            df_preview = pd.read_csv(data_path, nrows=5)
            st.markdown("**Dataset preview (first 5 rows)**")
            st.dataframe(df_preview, use_container_width=True)
        except Exception:
            st.caption("Dataset found but could not preview (CSV too large or unreadable).")
    else:
        st.caption("No dataset CSV detected in /data (optional for app runtime).")


st.title("🔬 Perovskite Band Gap Studio")
st.write(
    "Machine-learning powered **ABX₃ perovskite band-gap predictor** "
    "using elemental descriptors."
)

tab_formula, tab_desc, tab_batch, tab_insights = st.tabs(
    [
        "⚗️ Single formula (ABX₃)",
        "🧮 Descriptor mode",
        "📚 Batch formula mode",
        "📊 Model insights",
    ]
)

# -----------------------------------
# TAB 1: SINGLE FORMULA
# -----------------------------------
with tab_formula:
    st.subheader("Single ABX₃ formula → band-gap prediction")

    st.markdown(
        """
Enter a perovskite-like **ABX₃** composition such as `CsPbI3`, `BaTiO3`, `CsPbBr3`.

The tool will:
1. Validate that it looks like ABX₃  
2. Parse the formula and compute **averaged descriptors**  
3. Predict the band gap (eV) using the trained model  
4. Give an interpretation & solar suitability tag
        """
    )

    col_input, col_presets = st.columns([2, 1])

    with col_input:
        formula = st.text_input("Formula (ABX₃ only)", placeholder="e.g. CsPbI3")

    with col_presets:
        st.caption("Quick presets")
        c1, c2, c3 = st.columns(3)
        if c1.button("CsPbI3"):
            formula = "CsPbI3"
        if c2.button("BaTiO3"):
            formula = "BaTiO3"
        if c3.button("CsPbBr3"):
            formula = "CsPbBr3"
        st.info(f"Current formula: `{formula or '—'}`")

    if st.button("🔮 Predict band gap", key="predict_single"):
        if not formula or not formula.strip():
            st.error("Please enter a formula.")
        else:
            clean_formula = formula.strip()

            if not is_abx3(clean_formula):
                st.error(
                    "This tool only supports **ABX₃ perovskites**.\n\n"
                    "Examples: `CsPbI3`, `BaTiO3`, `CsPbBr3`.\n\n"
                    "Rules:\n"
                    "- Exactly 3 distinct elements (A, B, X)\n"
                    "- The last element X must have subscript 3"
                )
            else:
                try:
                    feat_df = formula_to_dataframe(clean_formula)
                    pred_val = float(model.predict(feat_df)[0])

                    col_left, col_right = st.columns([1, 1.2])

                    with col_left:
                        st.metric(
                            label=f"Predicted band gap for {clean_formula}",
                            value=f"{pred_val:.3f} eV",
                        )
                        st.success(classify_bandgap(pred_val))
                        st.info(solar_flag(pred_val))

                    with col_right:
                        st.markdown("**Extracted descriptors**")
                        st.dataframe(feat_df, use_container_width=True)

                    fig = descriptor_plot(feat_df.iloc[0])
                    st.pyplot(fig)

                except Exception as e:
                    st.error(f"Error while processing formula: {e}")
                    st.info(
                        "Check the element symbols. You can extend the periodic table "
                        "inside `src/utils.py` if needed."
                    )

# -----------------------------------
# TAB 2: DESCRIPTOR MODE
# -----------------------------------
with tab_desc:
    st.subheader("Direct descriptor input (advanced)")

    st.markdown(
        """
If you already know or estimate descriptor values for a composition,
you can skip the formula and talk to the model directly.

This is useful for:
- Exploring hypothetical compositions
- Scanning the descriptor space
- Sensitivity / trend analysis
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        en_val = st.slider(
            "Average electronegativity",
            min_value=0.7,
            max_value=4.0,
            value=2.5,
            step=0.05,
        )
    with col2:
        ir_val = st.slider(
            "Average ionic radius (Å)",
            min_value=0.5,
            max_value=2.5,
            value=1.3,
            step=0.05,
        )
    with col3:
        z_val = st.slider(
            "Average atomic number",
            min_value=1,
            max_value=83,
            value=40,
            step=1,
        )

    if st.button("🔮 Predict from descriptors", key="predict_desc"):
        df_in = pd.DataFrame(
            [
                {
                    "Electronegativity": float(en_val),
                    "IonicRadius": float(ir_val),
                    "AtomicNumber": float(z_val),
                }
            ]
        )
        try:
            pred_val = float(model.predict(df_in)[0])

            c_left, c_right = st.columns([1, 1.2])

            with c_left:
                st.metric("Predicted band gap", f"{pred_val:.3f} eV")
                st.success(classify_bandgap(pred_val))
                st.info(solar_flag(pred_val))

            with c_right:
                st.markdown("**Descriptor input**")
                st.dataframe(df_in, use_container_width=True)

            fig = descriptor_plot(df_in.iloc[0])
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error during prediction: {e}")

# -----------------------------------
# TAB 3: BATCH FORMULA MODE
# -----------------------------------
with tab_batch:
    st.subheader("Batch prediction for multiple ABX₃ formulas")

    st.markdown(
        """
Paste **multiple formulas** separated by commas or new lines.  
The app will:
- Filter only valid ABX₃ formulas  
- Predict band gaps for each  
- Add a quick solar-suitability label
        """
    )

    default_example = "CsPbI3, BaTiO3, CsPbBr3, NaCl, Cs2PbI4"
    text_block = st.text_area(
        "Formulas list",
        value=default_example,
        height=120,
        help="Separate formulas using commas or new lines.",
    )

    if st.button("📚 Run batch prediction", key="predict_batch"):
        results_df, skipped = batch_predict_formulas(text_block)

        if results_df.empty:
            st.error("No valid ABX₃ formulas found in the input.")
        else:
            st.markdown("### Batch results")
            st.dataframe(results_df, use_container_width=True)

        if skipped:
            st.warning(
                "These entries were skipped (not valid ABX₃ or parsing failed): "
                + ", ".join(skipped)
            )

# -----------------------------------
# TAB 4: MODEL INSIGHTS
# -----------------------------------
with tab_insights:
    st.subheader("Model insights & training visuals")

    feat_imp_path, pred_vs_path = get_insight_images()

    col_img1, col_img2 = st.columns(2)

    with col_img1:
        st.markdown("**Feature importance (model perspective)**")
        if feat_imp_path:
            st.image(feat_imp_path, use_container_width=True)
        else:
            st.warning("feature_importance.png not found in /results/.")

    with col_img2:
        st.markdown("**Predicted vs actual band gaps**")
        if pred_vs_path:
            st.image(pred_vs_path, use_container_width=True)
        else:
            st.warning(
                "pred_vs_actual_band_style.png or pred_vs_actual.png not found in /results/."
            )

    st.markdown("---")
    st.markdown(
        f"""
### Model summary

- Model type: `{type(model).__name__}`  
- Training data: synthetic **60,000-sample** ABX₃ perovskite-like dataset  
- Input features: `Electronegativity`, `IonicRadius`, `AtomicNumber`  

Use this as a **screening tool**, not as a replacement for:
- Full DFT workflows  
- Experimental validation  
        """
    )
