
## MLflow Tracking

The pipeline uses MLflow to track six candidate model configurations with validation Macro F1, Accuracy, and Log Loss metrics. The selected model is registered as `WineClassifier` and exposed through the `champion` alias.
