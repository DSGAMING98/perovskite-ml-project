📘 Perovskite Band Gap Prediction (ML Researchthon Project)

This project builds a Machine Learning framework to quickly screen perovskite materials by estimating their band gaps using elemental features like:

Electronegativity (EN)

Ionic Radius (IR)

Atomic Number (Z)

The goal is to help researchers identify high-efficiency perovskite solar cell candidates faster than traditional lab experiments.

🚀 Features
✔ Predict band gap directly from chemical formula

Example inputs:

CsPbI3
BaTiO3
CsPbBr3


The app automatically:

Parses the formula

Extracts EN, IR, Z for each element

Computes averaged features

Predicts the material’s band gap using a trained ML model

✔ Trains on 5000+ synthetic perovskite-like samples

You can scale up to 10k–15k rows if needed.

✔ Full ML workflow

Dataset creation

Training (RandomForest)

Feature importance visualization

Predicted vs actual plots

Streamlit UI

📁 Project Structure
perovskite-ml-project/
│
├── data/
│   └── perovskite_bandgap_5000rows.csv
│
├── models/
│   └── model.pkl
│
├── results/
│   ├── feature_importance.png
│   └── pred_vs_actual.png
│
├── notebooks/
│   └── exploratory_analysis.ipynb   (optional but recommended)
│
├── src/
│   ├── utils.py
│   └── train_model.py
│
└── app/
    └── app.py

⚙️ How to Train the Model

Place your dataset in /data/

Run:

cd src
python train_model.py


This will:

Train RandomForest on 5000 rows

Save a new model.pkl

Update plots inside /results/

Print MAE + R² score

🌐 Running the Streamlit App

From project root:

streamlit run app/app.py


Then open the browser and enter formulas like:

CsPbI3
BaTiO3
CsSnBr3


The app will show:

Parsed features

Predicted band gap

Error messages for unsupported elements

🔬 How Formula Prediction Works

Example:
Input = CsPbI3

The parser extracts elements:

Cs

Pb

I (×3)

For each element, it looks up:

Electronegativity (EN)

Ionic Radius (IR)

Atomic Number (Z)

Computes the average:

Electronegativity_avg
IonicRadius_avg
AtomicNumber_avg


Feeds into the ML model → predicts band gap.

📊 Visual Outputs (Auto Generated)

The training script creates:

feature_importance.png

pred_vs_actual.png

These appear inside results/ every time you run:

python train_model.py

🧠 Model Used

RandomForestRegressor

n_estimators = 800

n_jobs = -1

Works well with tabular scientific data

Stable results with large datasets

🔍 Future Improvements

Add support for organic perovskites (MA, FA)

Add XGBoost version

Add SHAP explainability

Add larger elemental database

Add CSV batch prediction mode

📝 Author

Developed as part of Rezonix Researchthon Project
BY GROUP ATOMIC ALLIANCE
Machine Learning contribution:
Dataset creation, model training, formula parsing, Streamlit UI.