import pickle
import pandas as pd
import os

model_path = os.path.join("..", "models", "model.pkl")
model = pickle.load(open(model_path, "rb"))

# Example input
data = pd.DataFrame([{
    "Electronegativity": 3.0,
    "IonicRadius": 1.2,
    "AtomicNumber": 50
}])

pred = model.predict(data)[0]
print("Predicted Band Gap:", round(pred, 3), "eV")
