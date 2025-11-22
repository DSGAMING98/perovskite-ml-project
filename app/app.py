import streamlit as st
import pickle
import pandas as pd
import os
import sys

# Add src/ to Python path so we can import utils
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from utils import parse_formula  # from src/utils.py

# Load trained model
model_path = os.path.join(BASE_DIR, "models", "model.pkl")
model = pickle.load(open(model_path, "rb"))

st.title("Perovskite Band Gap Predictor (Can use formulas)")

st.write(
    "Enter a **perovskite-like formula** (e.g., `CsPbI3`, `BaTiO3`) "
    "and the model will predict the band gap based on elemental features."
)

formula = st.text_input("Formula", placeholder="e.g. CsPbI3")

if st.button("Predict Band Gap"):
    if not formula.strip():
        st.error("Please enter a valid formula.")
    else:
        try:
            # Convert formula → features using utils.parse_formula
            features = parse_formula(formula.strip())
            feat_df = pd.DataFrame([features])

            # Show the interpreted features (for explanation)
            st.subheader("Interpreted Features")
            st.table(feat_df)

            # Predict band gap
            prediction = model.predict(feat_df)[0]
            st.success(f"Predicted Band Gap for {formula.strip()}: {round(prediction, 3)} eV")

        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info(
                "Make sure you only use supported elements like Cs, Pb, I, Br, Ba, Ti, etc. "
                "You can extend the periodic_table in utils.py for more elements."
            )
