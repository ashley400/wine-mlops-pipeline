from src.data import load_data


def test_dataset_shape_and_split():
    X_train, X_test, y_train, y_test = load_data()

    assert X_train.shape[1] == 13
    assert X_test.shape[1] == 13

    assert len(X_train) == 142
    assert len(X_test) == 36

    assert len(y_train) == 142
    assert len(y_test) == 36


def test_dataset_has_no_nulls():
    X_train, X_test, y_train, y_test = load_data()

    assert not X_train.isnull().any().any()
    assert not X_test.isnull().any().any()
    assert not y_train.isnull().any()
    assert not y_test.isnull().any()


def test_target_classes():
    _, _, y_train, y_test = load_data()

    assert set(y_train.unique()) == {0, 1, 2}
    assert set(y_test.unique()) == {0, 1, 2}
