import time

import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, f1_score, log_loss

from src.data import load_data


TRACKING_URI = "sqlite:///mlflow.db"
MODEL_URI = "models:/WineClassifier@champion"


def main():
    mlflow.set_tracking_uri(TRACKING_URI)

    _, X_test, _, y_test = load_data()

    model = mlflow.sklearn.load_model(MODEL_URI)

    # Warm up before measuring prediction latency.
    model.predict(X_test)

    start = time.perf_counter()
    predictions = model.predict(X_test)
    elapsed_ms = (time.perf_counter() - start) * 1000

    # Probability calculation is measured separately for Log Loss.
    probabilities = model.predict_proba(X_test)

    test_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
    )

    test_accuracy = accuracy_score(
        y_test,
        predictions,
    )

    test_log_loss = log_loss(
        y_test,
        probabilities,
        labels=[0, 1, 2],
    )

    f1_pass = test_f1 >= 0.88
    latency_pass = elapsed_ms <= 30.0
    classes_pass = set(predictions).issubset({0, 1, 2})

    print(f"Test Macro F1: {test_f1:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Log Loss: {test_log_loss:.4f}")
    print(f"Batch latency: {elapsed_ms:.4f} ms")
    print(f"Prediction classes: {sorted(set(predictions))}")

    print("\nQuality Gate:")
    print(f"F1 >= 0.88: {f1_pass}")
    print(f"Latency <= 30 ms: {latency_pass}")
    print(
        "Classes in {0,1,2}: "
        f"{classes_pass}"
    )

    if not (f1_pass and latency_pass and classes_pass):
        raise RuntimeError("Quality gate failed.")

    print("\nQuality Gate: PASSED")


if __name__ == "__main__":
    main()
