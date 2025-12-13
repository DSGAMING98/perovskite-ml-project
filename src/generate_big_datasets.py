import os
import numpy as np
import pandas as pd


def get_base_path():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_data_path(filename: str) -> str:
    data_dir = os.path.join(get_base_path(), "data")
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, filename)


def generate_batch(batch_id: int, n_rows: int = 100_000, seed: int = 42):
    """
    Generate one batch of synthetic perovskite-like data with realistic band gap formula.
    """
    np.random.seed(seed + batch_id)  # different randomness per batch

    # Elemental descriptors
    EN = np.random.uniform(0.7, 4.0, n_rows)      # Electronegativity
    IR = np.random.uniform(0.5, 2.5, n_rows)      # Ionic radius (Å, rough)
    Z  = np.random.randint(1, 84, n_rows)         # Atomic number

    # Physically-inspired non-linear band gap model
    # BandGap ≈ f(EN, IR, Z) + small noise
    bandgap = (
        1.1
        + 0.55 * EN
        - 0.42 * IR
        + 0.018 * np.sqrt(Z)
        - 0.028 * EN * IR
        + 0.0038 * (IR ** 2)
        + np.random.normal(0, 0.12, n_rows)   # Gaussian noise
    )

    bandgap = np.clip(bandgap, 0.0, 4.0)  # typical perovskite band gap window

    df = pd.DataFrame({
        "Electronegativity": np.round(EN, 3),
        "IonicRadius": np.round(IR, 3),
        "AtomicNumber": Z,
        "BandGap": np.round(bandgap, 3)
    })

    out_name = f"perovskite_bandgap_batch_{batch_id}_100k.csv"
    out_path = get_data_path(out_name)
    df.to_csv(out_path, index=False)

    print(f"[BATCH {batch_id}] Saved → {out_path} (rows = {len(df)})")


def main():
    total_batches = 1          # 5 × 100k = 500,000 rows total
    rows_per_batch = 100_000

    print("Generating BIG perovskite dataset in batches...")
    for batch_id in range(1, total_batches + 1):
        generate_batch(batch_id, n_rows=rows_per_batch, seed=42)

    print("All batches generated successfully!")


if __name__ == "__main__":
    main()
