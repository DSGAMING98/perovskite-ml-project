import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from utils import (
    load_dataset,
    get_model_path,
    plot_feature_importance,
    plot_pred_vs_actual,
)


def main():

    # LOAD DATASET (5000 ROWS)

    df = load_dataset("perovskite_bandgap_5000rows.csv")

    # Features + target
    X = df.drop("BandGap", axis=1)
    y = df["BandGap"]


    # TRAIN–TEST SPLIT

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # MODEL DEFINITION

    model = RandomForestRegressor(
        n_estimators=800,      # a bit more trees since we have more data
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1              # use all CPU cores
    )


    # TRAIN MODEL

    model.fit(X_train, y_train)


    # PREDICTIONS + METRICS

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)


    print("  PEROVSKITE BAND GAP ML TRAINING  ")
    print("===================================")
    print(f"Dataset size       : {len(df)} rows")
    print(f"Train size         : {len(X_train)}")
    print(f"Test size          : {len(X_test)}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R² Score           : {r2:.4f}")


    # SAVE TRAINED MODEL

    model_path = get_model_path("model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"[SAVED] Model saved at: {model_path}")

    # SAVE PLOTS

    plot_feature_importance(model, X.columns)
    plot_pred_vs_actual(y_test, y_pred)

    print("[DONE] All results exported to /results/")


if __name__ == "__main__":
    main()
