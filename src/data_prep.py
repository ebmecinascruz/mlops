import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split(path="data/ObesityDataSet_raw_and_data_sinthetic.csv", seed = 42):
    import os
    full_path = os.path.join(os.path.dirname(__file__), "..", path)
    df = pd.read_csv(full_path)
    y = df["NObeyesdad"]
    X = df.drop(columns=["NObeyesdad"])

    # One-hot encode before splitting
    X_encoded = pd.get_dummies(
        X,
        columns=[
            "Gender",
            "family_history_with_overweight",
            "FAVC",
            "CAEC",
            "SMOKE",
            "SCC",
            "CALC",
            "MTRANS",
        ],
    )

    # Split out 20% for scoring, return the rest for training
    X_remaining, X_holdout, y_remaining, y_holdout = train_test_split(
        X_encoded, y, test_size=0.2, random_state=seed
    )

    holdout_df = X_holdout.copy()
    holdout_df["label"] = y_holdout
    holdout_df.to_csv("data/holdout.csv", index=False)
    print("💾 Saved holdout set to data/holdout.csv")

    return X_remaining, y_remaining