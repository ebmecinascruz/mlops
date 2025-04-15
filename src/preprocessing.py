import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/ObesityDataSet_raw_and_data_sinthetic.csv")

target_col = "NObeyesdad"
y = df[target_col]
X = df.drop(columns=[target_col])

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

numerical_cols = X_encoded.select_dtypes(include=["int64", "float64"]).columns
scaler = StandardScaler()
X_encoded[numerical_cols] = scaler.fit_transform(X_encoded[numerical_cols])

df_processed = pd.concat([X_encoded, y], axis=1)

df_processed.to_csv("data/obesity_preprocessed.csv", index=False)