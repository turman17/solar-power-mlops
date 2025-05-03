from math import log
from pathlib import Path
import joblib

from loguru import logger
import typer
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from mlflow.models.signature import infer_signature
from tqdm import tqdm

import mlflow
import mlflow.sklearn

from spmlops.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

mlflow.set_tracking_uri("http://localhost:5002")


@app.command()
def train(
    X_train: Path = PROCESSED_DATA_DIR / "X_train.csv",
    X_test: Path = PROCESSED_DATA_DIR / "X_test.csv",
    y_train: Path = PROCESSED_DATA_DIR / "y_train.csv",
    y_test: Path = PROCESSED_DATA_DIR / "y_test.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
):
    """
    Train a Gradient Boosting Regressor model on the provided features and labels.

    Args:
        features_path (Path): Path to the CSV file containing the features.
        labels_path (Path): Path to the CSV file containing the labels.
        model_path (Path): Path to save the trained model.
    """
    try:
        # Load the training data
        logger.info("Loading training / testing data...")
        X_train = pd.read_csv(X_train)
        y_train = pd.read_csv(y_train)
        X_test = pd.read_csv(X_test)
        y_test = pd.read_csv(y_test)
        logger.info("Training/testing data loaded successfully.")

        with mlflow.start_run():
            from sklearn.model_selection import GridSearchCV

            param_grid = [
                {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1},
                {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.05},
                # ...
            ]

            for params in tqdm(param_grid, desc="Training models"):
                model = MultiOutputRegressor(GradientBoostingRegressor(**params))
            model.fit(X_train, y_train)
            logger.info("Model training complete.")

            logger.info("Evaluating the model...")
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            logger.info(f"Model evaluation complete. MSE: {mse}, R2: {r2}, MAE: {mae}")

            # Log with MLflow
            mlflow.log_param("model_type", "GradientBoostingRegressor")
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mae", mae)

            X_test_float = X_test.astype("float64")
            import numpy as np

            y_train_df = pd.read_csv(y_train) if isinstance(y_train, (str, Path)) else y_train
            signature = infer_signature(X_test_float, model.predict(X_test_float))
            input_example = X_test_float.iloc[:5]

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=signature,
                input_example=input_example,
            )

            logger.info(f"Saving the model to {model_path}...")
            joblib.dump(model, model_path)

        logger.success("Modeling training complete.")
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except pd.errors.EmptyDataError as e:
        logger.error(f"Empty data file: {e}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing data file: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise


if __name__ == "__main__":
    app()
