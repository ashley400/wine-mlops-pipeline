import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold

from src.data import load_data


TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "Wine-Cultivar-Classification"
REGISTERED_MODEL_NAME = "WineClassifier"
RANDOM_STATE = 42


def evaluate_model(model, X_train, y_train):
    """Evaluate a model using 5-fold stratified cross-validation."""
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    f1_scores = []
    accuracy_scores = []
    log_loss_scores = []

    for train_idx, validation_idx in cv.split(X_train, y_train):
        X_fold_train = X_train.iloc[train_idx]
        X_fold_valid = X_train.iloc[validation_idx]
        y_fold_train = y_train.iloc[train_idx]
        y_fold_valid = y_train.iloc[validation_idx]

        model.fit(X_fold_train, y_fold_train)

        predictions = model.predict(X_fold_valid)
        probabilities = model.predict_proba(X_fold_valid)

        f1_scores.append(
            f1_score(
                y_fold_valid,
                predictions,
                average="macro",
            )
        )

        accuracy_scores.append(
            accuracy_score(
                y_fold_valid,
                predictions,
            )
        )

        log_loss_scores.append(
            log_loss(
                y_fold_valid,
                probabilities,
                labels=[0, 1, 2],
            )
        )

    return {
        "validation_macro_f1": sum(f1_scores) / len(f1_scores),
        "validation_accuracy": sum(accuracy_scores) / len(accuracy_scores),
        "validation_log_loss": sum(log_loss_scores) / len(log_loss_scores),
    }


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, _, y_train, _ = load_data()

    models = {
        "RF-1": {
            "model": RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            "params": {
                "model_family": "RandomForest",
                "n_estimators": 100,
                "max_depth": "None",
                "min_samples_split": 2,
                "n_jobs": 1,
            },
        },
        "RF-2": {
            "model": RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_split=2,
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            "params": {
                "model_family": "RandomForest",
                "n_estimators": 200,
                "max_depth": 8,
                "min_samples_split": 2,
                "n_jobs": 1,
            },
        },
        "RF-3": {
            "model": RandomForestClassifier(
                n_estimators=150,
                max_depth=5,
                min_samples_split=2,
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            "params": {
                "model_family": "RandomForest",
                "n_estimators": 150,
                "max_depth": 5,
                "min_samples_split": 2,
                "n_jobs": 1,
            },
        },
        "GB-1": {
            "model": GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=3,
                random_state=RANDOM_STATE,
            ),
            "params": {
                "model_family": "GradientBoosting",
                "n_estimators": 100,
                "learning_rate": 0.05,
                "max_depth": 3,
            },
        },
        "GB-2": {
            "model": GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.10,
                max_depth=2,
                random_state=RANDOM_STATE,
            ),
            "params": {
                "model_family": "GradientBoosting",
                "n_estimators": 100,
                "learning_rate": 0.10,
                "max_depth": 2,
            },
        },
        "GB-3": {
            "model": GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=2,
                random_state=RANDOM_STATE,
            ),
            "params": {
                "model_family": "GradientBoosting",
                "n_estimators": 150,
                "learning_rate": 0.05,
                "max_depth": 2,
            },
        },
    }

    results = []

    for run_name, config in models.items():
        model = config["model"]

        with mlflow.start_run(run_name=run_name) as run:
            metrics = evaluate_model(
                model,
                X_train,
                y_train,
            )

            mlflow.log_params(config["params"])
            mlflow.log_metrics(metrics)
            mlflow.set_tag("run_name", run_name)

            results.append(
                {
                    "run_name": run_name,
                    "run_id": run.info.run_id,
                    **config["params"],
                    **metrics,
                }
            )

            print(
                f"{run_name}: "
                f"F1={metrics['validation_macro_f1']:.4f}, "
                f"Accuracy={metrics['validation_accuracy']:.4f}, "
                f"LogLoss={metrics['validation_log_loss']:.4f}"
            )

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="validation_macro_f1",
        ascending=False,
    )

    print("\nHyperparameter Results:")
    print(results_df.to_string(index=False))

    best_run_id = results_df.iloc[0]["run_id"]
    best_run_name = results_df.iloc[0]["run_name"]

    best_config = models[best_run_name]
    best_model = best_config["model"]

    best_model.fit(X_train, y_train)

    signature = infer_signature(
        X_train,
        best_model.predict(X_train),
    )

    input_example = X_train.head(3)

    with mlflow.start_run(
        run_name=f"{best_run_name}-final-model"
    ):
        mlflow.log_params(best_config["params"])

        mlflow.sklearn.log_model(
            best_model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        mlflow.set_tag("source_run_id", best_run_id)
        mlflow.set_tag("source_run_name", best_run_name)

        print(
            f"\nRegistered model: {REGISTERED_MODEL_NAME}"
        )
        print(f"Source best run: {best_run_id}")
        print(f"Source best model: {best_run_name}")

    results_df.to_csv(
        "results/hyperparameter_results.csv",
        index=False,
    )

    client = MlflowClient(
        tracking_uri=TRACKING_URI
    )

    latest_versions = client.search_model_versions(
        f"name='{REGISTERED_MODEL_NAME}'"
    )

    latest_version = max(
        latest_versions,
        key=lambda version: int(version.version),
    )

    client.set_registered_model_alias(
        REGISTERED_MODEL_NAME,
        "champion",
        latest_version.version,
    )

    print(
        f"Champion alias set to version "
        f"{latest_version.version}"
    )


if __name__ == "__main__":
    main()
