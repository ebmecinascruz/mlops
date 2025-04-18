import pandas as pd

def load_and_preprocess_data(path="data/ObesityDataSet_raw_and_data_sinthetic.csv"):
    import os
    full_path = os.path.join(os.path.dirname(__file__), "..", path)
    df = pd.read_csv(full_path)
    return df

def encode_data(df):
    df_encoded = pd.get_dummies(
        df.drop(columns=["NObeyesdad"]),
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
    return df_encoded