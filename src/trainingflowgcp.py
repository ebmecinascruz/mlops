from metaflow import FlowSpec, step, Parameter, conda_base, kubernetes, resources, timeout, retry
from data_prep import load_and_split
from train_models import (
    train_logistic_regression,
    train_random_forest,
    train_decision_tree
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV

@conda_base(python='3.9', libraries={
    "pandas": "1.5.3",
    "scikit-learn": "1.2.2",
    "mlflow": "2.15.1",
    "joblib": "1.2.0",
    "gcsfs": "2023.6.0",
    "fsspec": "2023.6.0"
})
class ObesityMLFlowGCP(FlowSpec):

    seed = Parameter(
        "seed",
        help="Random seed for reproducibility",
        default=42,
        type=int
    )

    @kubernetes(memory=2000, cpu=1)
    @resources(memory=2000, cpu=1)
    @step
    def start(self):
        print("🔹 Loading and preprocessing data...")
        self.X, self.y = load_and_split(seed=self.seed)
        self.next(self.select_features)

    @kubernetes(memory=2000, cpu=1)
    @resources(memory=2000, cpu=1)
    @step
    def select_features(self):
        print("🔹 Selecting features using Random Forest + RFECV...")
        rf = RandomForestClassifier(n_estimators=140, max_features=7, oob_score=True, random_state=self.seed)
        rfecv = RFECV(estimator=rf, step=1, cv=StratifiedKFold(5), scoring="accuracy")
        rfecv.fit(self.X, self.y)

        self.selected_features = self.X.columns[rfecv.get_support()].tolist()
        self.X = self.X[self.selected_features]

        import joblib
        import fsspec

        fs = fsspec.filesystem("gcs")
        with fs.open("lab7_bucket_eli/selected_feature_columns.pkl", "wb") as f:
            joblib.dump(self.X.columns.tolist(), f)

        print(f"✅ Selected features: {self.selected_features}")
        self.next(self.logistic, self.rf, self.tree)

    def _mlflow_setup(self):
        import mlflow
        mlflow.set_tracking_uri("https://service-run-916309862802.us-west2.run.app")
        if not mlflow.get_experiment_by_name("metaflow-experiment"):
            mlflow.create_experiment("metaflow-experiment")
        mlflow.set_experiment("metaflow-experiment")

    @kubernetes(memory=2000, cpu=1)
    @resources(memory=2000, cpu=1)
    @step
    def logistic(self):
        self._mlflow_setup()
        print("🔹 Training Logistic Regression...")
        self.log_results = train_logistic_regression(self.X, self.y, self.seed)
        self.next(self.join_models)

    @kubernetes(memory=2000, cpu=1)
    @resources(memory=2000, cpu=1)
    @step
    def rf(self):
        self._mlflow_setup()
        print("🔹 Training Random Forest...")
        self.rf_results = train_random_forest(self.X, self.y, self.seed)
        self.next(self.join_models)

    @kubernetes(memory=2000, cpu=1)
    @resources(memory=2000, cpu=1)
    @step
    def tree(self):
        self._mlflow_setup()
        print("🔹 Training Decision Tree...")
        self.tree_results = train_decision_tree(self.X, self.y, self.seed)
        self.next(self.join_models)

    @step
    def join_models(self, inputs):
        print("🔹 Joining model results...")
        self.results = {
            "logistic": inputs[0].log_results,
            "rf": inputs[1].rf_results,
            "tree": inputs[2].tree_results,
        }
        self.next(self.end)

    @step
    def end(self):
        print("✅ Flow complete! Results summary:")
        for model, metrics in self.results.items():
            print(f"\n🔸 {model.capitalize()} results:")
            for key, value in metrics.items():
                print(f"   {key}: {value}")


if __name__ == "__main__":
    ObesityMLFlowGCP()