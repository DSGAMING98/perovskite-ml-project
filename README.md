# 🔬 Perovskite Band Gap Studio  
### Streamlit-Based ABX₃ Band Gap Predictor

This project implements a **machine-learning framework** to rapidly estimate the **band gaps of perovskite-like ABX₃ compounds** using simple elemental descriptors. The aim is to support **fast virtual screening** of candidate materials before expensive experiments or full DFT workflows.

The final prototype is a **Streamlit web app** with:

- Formula-based prediction (`ABX₃` mode)  
- Descriptor-based prediction (advanced mode)  
- Batch prediction for multiple formulas  
- Model insight visualizations (feature importance + predicted vs actual plots)

---

## 🚀 Key Features

### 1. ABX₃ Formula Mode
- Input: a single perovskite-like **ABX₃ formula** (e.g. `CsPbI3`, `BaTiO3`, `CsPbBr3`)
- The app:
  - Validates ABX₃ structure (A, B, X with subscript 3)
  - Parses the formula into elemental components
  - Computes averaged:
    - Electronegativity  
    - Ionic radius  
    - Atomic number  
  - Predicts the band gap (eV) using the trained model
  - Provides:
    - A qualitative interpretation of the band gap
    - A rough “solar suitability” tag (ideal / borderline / unfavourable)

---

### 2. Descriptor Mode (Advanced)
- Input: directly specify descriptor values:
  - Average electronegativity  
  - Average ionic radius (Å)  
  - Average atomic number  
- Useful for:
  - Exploring hypothetical compositions
  - Sensitivity analysis over descriptor space
  - Understanding model trends without committing to a specific formula

---

### 3. Batch Formula Mode
- Input: multiple formulas separated by **commas** or **new lines**
- The app:
  - Filters valid ABX₃ formulas
  - Predicts band gaps for each valid entry
  - Assigns solar suitability tags
  - Displays a table with:
    - Formula  
    - Descriptors  
    - Predicted band gap (eV)  
    - Qualitative assessment

---

### 4. Model Insights
- Visualizations (stored under `results/`):
  - `feature_importance.png`  
    - Ranking of descriptor importance for the model  
  - `pred_vs_actual_band_style.png` (or `pred_vs_actual.png`)  
    - Predicted vs actual band gaps, shown as:
      - Scatter plot, and/or  
      - Band-style sorted curve

These plots are generated automatically when training the model.

---

## 🧠 Methodology

### Dataset

- Synthetic dataset with **60,000 samples**, stored as:
  - `data/perovskite_bandgap_60000rows.csv`
- Columns:
  - `Electronegativity`
  - `IonicRadius`
  - `AtomicNumber`
  - `BandGap`
- Band gaps are generated using a **physically inspired non-linear function** with added noise, ensuring:
  - Anti-correlation with electronegativity and atomic number  
  - Positive correlation with ionic radius  
  - Values clipped to a realistic range (e.g. ~0.5–4.0 eV)

The dataset is **synthetic**, designed to approximate plausible perovskite-like trends for demonstration and prototyping.

### Model

- Algorithm: **RandomForestRegressor** (scikit-learn)
- Input features:
  - `Electronegativity`
  - `IonicRadius`
  - `AtomicNumber`
- Target:
  - `BandGap` (eV)
- Train/test split (default): 80% train / 20% test
- Metrics:
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
  - R² score (coefficient of determination)

Hyperparameters (e.g. `n_estimators`, `max_depth`) can be tuned in `src/train_model.py` to balance:
- Model size (on disk)
- Accuracy
- Hosting / deployment constraints

---

## 📁 Project Structure

```text
perovskite-ml-project/
│
├── app/
│   └── app.py                     # Streamlit UI (main application)
│
├── src/
│   ├── train_model.py             # Training script (uses 60k dataset)
│   └── utils.py                   # Formula parsing, ABX₃ validation, helpers
│
├── data/
│   └── perovskite_bandgap_60000rows.csv   # Synthetic descriptor + band gap data
│
├── models/
│   └── model.pkl                  # Trained RandomForest model
│
├── results/
│   ├── feature_importance.png
│   ├── pred_vs_actual.png
│   └── pred_vs_actual_band_style.png
│
├── website/                       # (Optional) Static front-end site, if used
│   ├── index.html
│   └── style.css
│
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
⚙️ Installation & Setup
1. Create and activate a virtual environment (recommended)
cd perovskite-ml-project

# Example using venv:
python -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate

2. Install dependencies
pip install -r requirements.txt


requirements.txt should include at least:

streamlit
pandas
numpy
scikit-learn
matplotlib

🏋️‍♀️ Training / Updating the Model

If you modify the dataset or want to retrain:

Make sure data/perovskite_bandgap_60000rows.csv exists.

Run:

python src/train_model.py


This will:

Load the 60k dataset

Train RandomForestRegressor

Evaluate on a held-out test split (prints MAE, RMSE, R²)

Save the model to:

models/model.pkl

Regenerate plots in:

results/feature_importance.png

results/pred_vs_actual.png

results/pred_vs_actual_band_style.png

The Streamlit app will automatically use the latest models/model.pkl.

▶️ Running the Streamlit App

From the project root:

streamlit run app/app.py


Streamlit will launch the app in your browser, typically at:

http://localhost:8501


You can then:

Use Single Formula Mode for an individual ABX₃ composition

Use Descriptor Mode to explore feature space directly

Use Batch Mode to screen multiple formulas at once

Explore Model Insights to understand what the ML model has learned

📌 Notes & Limitations

The current dataset is synthetic, intended for:

Method demonstration

Prototyping

UI/UX and workflow validation

For real scientific deployment, the model should be retrained on:

High-quality DFT-calculated or experimentally reported perovskite band gaps

Extended descriptor sets (e.g. tolerance factor, octahedral factor, etc.)

The solar suitability tags are heuristic and should be treated as rough screening guidance, not final device design advice.

🏆 Credits

Developed as part of a research-oriented project on:

“Machine Learning–Driven Screening of Perovskite Materials for Band Gap Optimization.”
BY:
Prajwal C Pradhan
Malavika Vinod
Mumukka Sanjana Reddy
Namgay D Wangchuk
Guided By:
Dr. Yesheanth Kumar

The codebase, dataset, and app structure are intended as a foundation that can be extended with:

Real materials datasets

Additional stability / formation energy predictors

Multi-objective screening workflows for perovskite solar cells.
