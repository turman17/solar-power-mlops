from math import log
from pathlib import Path
import joblib
import re

from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


from spmlops.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


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
        y_train = pd.read_csv(y_train).squeeze()
        X_test = pd.read_csv(X_test)
        y_test = pd.read_csv(y_test).squeeze()
        logger.info("Training/testing data loaded successfully.")
        
        model = GradientBoostingRegressor()
        logger.info("Training the model...")
        model.fit(X_train, y_train)
        logger.info("Model training complete.")
        
        logger.info("Evaluating the model...")
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        logger.info(f"Model evaluation complete. MSE: {mse}, R2: {r2}, MAE: {mae}")
        
    
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
