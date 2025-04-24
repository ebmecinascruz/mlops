from metaflow import FlowSpec, step, Parameter, conda_base, resources, retry, timeout
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, classification_report
import fsspec
import joblib

@conda_base(python="3.9")
@resources(memory=2000, cpu=2)
@retry(times=2)
@timeout(seconds=600)
class ScoringFlowGCP(FlowSpec):

    model_name = Parameter(
        "model_name",
        help="Name of the registered model in MLflow Model Registry (e.g., 'Best_RF_Model')",
        default="Best_RF_Model"
    )

    model_stage = Parameter(
        "model_stage",
        help="Stage of the model to load (e.g., 'Production', 'Staging', or 'None' to load latest)",
        default="None"
    )

    holdout_path = Parameter(
        "holdout_path",
        help="Path to the holdout CSV file for scoring (use GCS path)",
        default="lab7_bucket_eli/holdout.csv"  # ✅ updated path
    )

    @step
    def start(self):
        mlflow.set_tracking_uri("https://service-run-916309862802.us-west2.run.app")
        mlflow.set_experiment("metaflow-experiment")
        print("🔹 Loading holdout set for scoring...")

        fs = fsspec.filesystem("gcs")
        with fs.open(self.holdout_path, "r") as f:
            df = pd.read_csv(f)

        self.y_true = df["label"]
        self.X_holdout = df.drop(columns=["label"])
        self.next(self.load_model)

    @step
    def load_model(self):
        if self.model_stage.lower() == "none":
            model_uri = f"models:/{self.model_name}/latest"
        else:
            model_uri = f"models:/{self.model_name}/{self.model_stage}"

        self.model = mlflow.sklearn.load_model(model_uri)
        print("✅ Model loaded successfully.")
        self.next(self.predict)

    @step
    def predict(self):
        print("🔹 Aligning holdout features to match training features...")

        fs = fsspec.filesystem("gcs")
        with fs.open("lab7_bucket_eli/selected_feature_columns.pkl", "rb") as f:
            expected_columns = joblib.load(f)

        for col in expected_columns:
            if col not in self.X_holdout.columns:
                self.X_holdout[col] = 0

        self.X_holdout = self.X_holdout[expected_columns]

        print("🔹 Making predictions...")
        self.y_pred = self.model.predict(self.X_holdout)
        self.next(self.evaluate)

    @step
    def evaluate(self):
        acc = accuracy_score(self.y_true, self.y_pred)
        report = classification_report(self.y_true, self.y_pred, zero_division=0)

        print(f"\n✅ Accuracy on holdout: {acc:.4f}")
        print("\n📋 Classification Report:\n")
        print(report)

        self.accuracy = acc
        self.next(self.end)

    @step
    def end(self):
        print("🏁 Scoring Flow Complete.")
        print(f"Final Accuracy: {self.accuracy:.4f}")


if __name__ == "__main__":
    ScoringFlowGCP()