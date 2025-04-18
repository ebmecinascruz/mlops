from metaflow import FlowSpec, step, Parameter
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, classification_report

mlflow.set_tracking_uri("https://service-run-394279149427.us-west2.run.app")


class ScoringFlow(FlowSpec):

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
        help="Path to the holdout CSV file for scoring",
        default="data/holdout.csv"
    )
    @step
    def start(self):
        print("🔹 Loading holdout set for scoring...")
        df = pd.read_csv(self.holdout_path)
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

        import joblib

        # Load expected column names from training
        expected_columns = joblib.load("data/selected_feature_columns.pkl")

        # Add missing columns
        for col in expected_columns:
            if col not in self.X_holdout.columns:
                self.X_holdout[col] = 0

        # Drop any extra columns
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
    ScoringFlow()