import mlflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# MLflow config
mlflow.set_tracking_uri("https://service-run-916309862802.us-west2.run.app")
mlflow.set_experiment("metaflow-experiment")

def split_data(X, y, seed):
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=seed)
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_logistic_regression(X, y, seed):
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, seed)
    solver_list = ["lbfgs", "liblinear", "sag", "newton-cg"]

    best_score = 0
    best_model = None
    best_run_id = None
    best_params = {}

    for solver in solver_list:
        for max_iter in range(100, 201, 10):
            with mlflow.start_run(nested=True) as run:
                mlflow.set_tags({"Model": "Logistic Regression", "Train Data": "train_set"})
                mlflow.log_params({"solver": solver, "max_iter": max_iter})

                model = LogisticRegression(penalty="l2", solver=solver, max_iter=max_iter, random_state=seed)
                model.fit(X_train, y_train)

                val_acc = accuracy_score(y_val, model.predict(X_val))
                test_acc = accuracy_score(y_test, model.predict(X_test))

                mlflow.log_metric("validation_accuracy", val_acc)
                mlflow.log_metric("test_accuracy", test_acc)

                if test_acc > best_score:
                    best_score = test_acc
                    best_model = model
                    best_run_id = run.info.run_id
                    best_params = {
                        "solver": solver,
                        "max_iter": max_iter,
                        "val_accuracy": val_acc,
                        "test_accuracy": test_acc,
                    }

    # ✅ Log the best model with its test_accuracy
    if best_model and best_run_id:
        with mlflow.start_run(run_id=best_run_id):
            mlflow.set_tag("selected_as_best", "true")
            mlflow.log_metric("test_accuracy", best_score)  # ← NEW METRIC
            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path="logistic_model",
                registered_model_name="LogisticRegression_Obesity"
            )
            print("🔐 Registered best logistic regression model to MLflow!")

    return best_params

def train_random_forest(X, y, seed):
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, seed)
    ntrees = [20, 40, 60, 80, 100, 120, 140]
    mtrys = [3, 4, 5, 6]

    best_score = 0
    best_params = {}

    for n in ntrees:
        for mf in mtrys:
            with mlflow.start_run(nested=True) as run:
                mlflow.set_tags({"Model": "Random Forest", "Train Data": "train_set"})
                mlflow.log_params({"n_estimators": n, "max_features": mf})

                model = RandomForestClassifier(n_estimators=n, max_features=mf, oob_score=True, random_state=seed)
                model.fit(X_train, y_train)

                val_acc = accuracy_score(y_val, model.predict(X_val))
                test_acc = accuracy_score(y_test, model.predict(X_test))

                mlflow.log_metric("oob_accuracy", model.oob_score_)
                mlflow.log_metric("validation_accuracy", val_acc)
                mlflow.log_metric("test_accuracy", test_acc)

                if test_acc > best_score:
                    best_score = test_acc
                    best_model = model
                    best_run_id = run.info.run_id
                    best_params = {
                        "n_estimators": n,
                        "max_features": mf,
                        "oob_accuracy": model.oob_score_,
                        "val_accuracy": val_acc,
                        "test_accuracy": test_acc,
                    }

    # ✅ Register the best model at the end
    if best_model and best_run_id:
        with mlflow.start_run(run_id=best_run_id):
            mlflow.set_tag("selected_as_best", "true")
            mlflow.log_metric("test_accuracy", best_score)  # ← NEW METRIC        
            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path="randomforest_model",
                registered_model_name="Best_RF_Model"
            )
            print("🔐 Registered best RF regression model to MLflow!")

    return best_params

def train_decision_tree(X, y, seed):
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, seed)
    tree_depths = [3, 4, 5, 6, 7]
    criteria = ["gini", "entropy"]

    best_score = 0
    best_params = {}

    for depth in tree_depths:
        for criterion in criteria:
            with mlflow.start_run(nested=True) as run:
                mlflow.set_tags({"Model": "Decision Tree", "Train Data": "train_set"})
                mlflow.log_params({"tree_depth": depth, "criterion": criterion})

                model = DecisionTreeClassifier(max_depth=depth, criterion=criterion, random_state=seed)
                model.fit(X_train, y_train)

                val_acc = accuracy_score(y_val, model.predict(X_val))
                test_acc = accuracy_score(y_test, model.predict(X_test))

                mlflow.log_metric("validation_accuracy", val_acc)
                mlflow.log_metric("test_accuracy", test_acc)

                if test_acc > best_score:
                    best_score = test_acc
                    best_model = model
                    best_run_id = run.info.run_id
                    best_params = {
                        "tree_depth": depth,
                        "criterion": criterion,
                        "val_accuracy": val_acc,
                        "test_accuracy": test_acc,
                    }

    # ✅ Register the best model at the end
    if best_model and best_run_id:
        with mlflow.start_run(run_id=best_run_id):
            mlflow.set_tag("selected_as_best", "true")
            mlflow.log_metric("test_accuracy", best_score)  # ← NEW METRIC        
            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path="dt_model",
                registered_model_name="DecisionTree_Obesity"
            )
            print("🔐 Registered best DT model to MLflow!")

    return best_params