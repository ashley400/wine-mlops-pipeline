import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_data():
    """Load and validate the Wine dataset."""
    wine = load_wine()

    X = pd.DataFrame(wine.data, columns=wine.feature_names)
    y = pd.Series(wine.target, name="target")

    if X.shape[1] != 13:
        raise ValueError("Wine dataset must contain exactly 13 features.")

    if X.isnull().any().any():
        raise ValueError("Feature data contains null values.")

    if y.isnull().any():
        raise ValueError("Target data contains null values.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Classes: {sorted(y_train.unique().tolist())}")
