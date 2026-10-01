import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from src.data import load_data


F1_THRESHOLD = 0.88
LATENCY_THRESHOLD_MS = 30.0
ALLOWED_CLASSES = {0, 1, 2}


def test_validation_macro_f1_gate():
    X_train, _, y_train, _ = load_data()

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=1,
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    scores = []

    for train_idx, validation_idx in cv.split(X_train, y_train):
        X_fold_train = X_train.iloc[train_idx]
        X_fold_valid = X_train.iloc[validation_idx]
        y_fold_train = y_train.iloc[train_idx]
        y_fold_valid = y_train.iloc[validation_idx]

        model.fit(X_fold_train, y_fold_train)

        predictions = model.predict(X_fold_valid)

        scores.append(
            f1_score(
                y_fold_valid,
                predictions,
                average="macro",
            )
        )

    validation_f1 = sum(scores) / len(scores)

    print(f"\nQuality Gate F1: {validation_f1:.4f}")

    assert validation_f1 >= F1_THRESHOLD


def test_batch_inference_latency_gate():
    X_train, X_test, y_train, _ = load_data()

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=1,
    )

    model.fit(X_train, y_train)

    # Warm up before measuring.
    model.predict(X_test)

    timings = []

    for _ in range(5):
        start = time.perf_counter()
        predictions = model.predict(X_test)
        elapsed_ms = (time.perf_counter() - start) * 1000
        timings.append(elapsed_ms)

    timings.sort()
    median_latency = timings[len(timings) // 2]

    print(f"\nBatch inference timings: {timings}")
    print(
        f"Median batch inference latency: "
        f"{median_latency:.4f} ms"
    )

    assert median_latency <= LATENCY_THRESHOLD_MS
    assert len(predictions) == len(X_test)


def test_prediction_schema_gate():
    X_train, X_test, y_train, _ = load_data()

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=1,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    assert set(predictions).issubset(ALLOWED_CLASSES)
