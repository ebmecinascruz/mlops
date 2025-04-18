from metaflow import FlowSpec, step, Parameter
from data_prep import load_and_preprocess_data, encode_data
from train_models import (
    train_logistic_regression,
    train_random_forest,
    train_decision_tree
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV


class ObesityMLFlow(FlowSpec):

    seed = Parameter(
        "seed",
        help="Random seed for reproducibility",
        default=42,
        type=int
    )

    @step
    def start(self):
        print("🔹 Loading and preprocessing data...")
        df = load_and_preprocess_data()
        self.df_encoded = encode_data(df)
        self.target = df["NObeyesdad"]
        self.next(self.select_features)

    @step
    def select_features(self):
        print("🔹 Selecting features using Random Forest + RFECV...")
        rf = RandomForestClassifier(n_estimators=140, max_features=7, oob_score=True, random_state=self.seed)
        rfecv = RFECV(estimator=rf, step=1, cv=StratifiedKFold(5), scoring="accuracy")
        rfecv.fit(self.df_encoded, self.target)

        self.selected_features = self.df_encoded.columns[rfecv.get_support()].tolist()
        self.X = self.df_encoded[self.selected_features]
        self.y = self.target

        print(f"✅ Selected features: {self.selected_features}")
        self.next(self.logistic, self.rf, self.tree)

    @step
    def logistic(self):
        print("🔹 Training Logistic Regression...")
        self.log_results = train_logistic_regression(self.X, self.y, self.seed)
        self.next(self.join_models)

    @step
    def rf(self):
        print("🔹 Training Random Forest...")
        self.rf_results = train_random_forest(self.X, self.y, self.seed)
        self.next(self.join_models)

    @step
    def tree(self):
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
    ObesityMLFlow()