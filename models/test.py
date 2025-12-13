from huggingface_hub import hf_hub_download
import pickle

model_path = hf_hub_download(
    repo_id="Prajwal0906/perovskite-bandgap_predictor",
    filename="model.pkl"
)

print("Model downloaded at:", model_path)

with open(model_path, "rb") as f:
    model = pickle.load(f)

print("Model loaded successfully.")
