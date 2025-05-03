import joblib
from pathlib import Path
from loguru import logger

import pandas as pd
import typer

from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from spmlops.config import PROCESSED_DATA_DIR, MODELS_DIR

app = typer.Typer()


@app.command(name="train-anomaly-detection")
def train_anomaly_detection(
    X_train: Path = PROCESSED_DATA_DIR / "X_train.csv",
    X_test: Path = PROCESSED_DATA_DIR / "X_test.csv",
    model_path: Path = MODELS_DIR / "anomaly_detection_model.pkl",
):
    """
    Train an Isolation Forest model for anomaly detection on the provided features.

    Args:
        X_train (Path): Path to the CSV file containing the training features.
        X_test (Path): Path to the CSV file containing the testing features.
        model_path (Path): Path to save the trained model.
    """
    try:
        if not X_train.exists() or not X_test.exists():
            logger.error("Training or testing file not found.")
            raise FileNotFoundError

        # Load the training data
        logger.info("Loading training data...")
        X_train_df = pd.read_csv(X_train)
        logger.info("Training data loaded successfully.")

        # Train the Isolation Forest model
        model = IsolationForest(contamination=0.001, random_state=42)
        logger.info("Training the Isolation Forest model...")
        model.fit(X_train_df)
        logger.info("Model training complete.")

        # Evaluate the model
        logger.info("Evaluating the model...")
        X_test_df = pd.read_csv(X_test)
        y_pred = model.predict(X_test_df)
        pred_counts = pd.Series(y_pred).value_counts().to_dict()
        logger.info(f"Anomaly counts: {pred_counts}")

        # Save predictions (optional)
        output_pred_path = PROCESSED_DATA_DIR / "anomaly_predictions.csv"
        pd.DataFrame({"prediction": y_pred}).to_csv(output_pred_path, index=False)
        logger.info(f"Predictions saved to {output_pred_path}")

        # Save the trained model
        logger.info(f"Saving the model to {model_path}...")
        joblib.dump(model, model_path)
        logger.info("Model saved successfully.")

    except Exception as e:
        logger.error(f"An error occurred during training: {e}")


if __name__ == "__main__":
    app()
