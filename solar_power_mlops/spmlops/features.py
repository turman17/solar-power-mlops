from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from spmlops.config import PROCESSED_DATA_DIR
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd

app = typer.Typer()

@app.command()
def split_data(
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    try:
        X = pd.read_csv(PROCESSED_DATA_DIR / "X.csv")
        y = pd.read_csv(PROCESSED_DATA_DIR / "y.csv").squeeze()

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
        )

        X_train.to_csv(PROCESSED_DATA_DIR / "X_train.csv", index=False)
        X_test.to_csv(PROCESSED_DATA_DIR / "X_test.csv", index=False)
        y_train.to_csv(PROCESSED_DATA_DIR / "y_train.csv", index=False)
        y_test.to_csv(PROCESSED_DATA_DIR / "y_test.csv", index=False)

        logger.success("Data split successfully.")
    except FileNotFoundError:
        logger.error("File not found. Please check the file paths.")
        raise
    except pd.errors.EmptyDataError:
        logger.error("File is empty. Please check the file.")
        raise
    except pd.errors.ParserError:
        logger.error("Error parsing file. Please check the file format.")
        raise
        

def load_data(input_path: Path) -> pd.DataFrame:
    """Load the dataset from the given path."""
    logger.info(f"Loading data from {input_path}...")
    data = pd.read_csv(input_path)
    logger.info("Data loaded successfully.")
    return data


def generate_features(data: pd.DataFrame) -> pd.DataFrame:
    """Generate features from the dataset."""
    logger.info("Generating features...")
    data["DATE_TIME"] = pd.to_datetime(data["DATE_TIME"])
    data["MINUTE"] = data["DATE_TIME"].dt.minute
    data["HOUR"] = data["DATE_TIME"].dt.hour
    data["DAY"] = data["DATE_TIME"].dt.day
    data["MONTH"] = data["DATE_TIME"].dt.month

    data.drop(columns=["DATE_TIME"], inplace=True)

    data["SOURCE_KEY"] = data["SOURCE_KEY_x"].astype(str) + "_" + data["SOURCE_KEY_y"].astype(str)
    data.drop(columns=["SOURCE_KEY_x", "SOURCE_KEY_y"], inplace=True)

    data["PLANT_ID"] = data["PLANT_ID_x"].astype(str)
    data.drop(columns=["PLANT_ID_x", "PLANT_ID_y"], inplace=True)

    le = LabelEncoder()
    data["SOURCE_KEY"] = le.fit_transform(data["SOURCE_KEY"])

    front_cols = ["PLANT_ID", "SOURCE_KEY"]
    other_cols = [col for col in data.columns if col not in front_cols]
    data = data[front_cols + other_cols]

    logger.info("Features generated successfully.")
    return data


def select_features(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Select relevant features from the dataset."""
    logger.info("Selecting relevant features...")

    y = data[["DC_POWER", "AC_POWER"]]
    X = data.drop(columns=["DC_POWER", "AC_POWER"])

    logger.info("Features selected successfully.")
    return X, y


@app.command()
def build_features(input_path: Path = PROCESSED_DATA_DIR / "Processed_Plant_1_Generation_Data.csv"):
    try:
        data = load_data(input_path)
        data = generate_features(data)
        X, y = select_features(data)

        logger.info("Data processing completed successfully.")
        logger.info(f"X shape: {X.shape}, y shape: {y.shape}")

        X.to_csv(PROCESSED_DATA_DIR / "X.csv", index=False)
        y.to_csv(PROCESSED_DATA_DIR / "y.csv", index=False)
        logger.success("Saved X and y to data/processed/")
    except FileNotFoundError:
        logger.error(f"File not found: {input_path}")
        raise
    except pd.errors.EmptyDataError:
        logger.error(f"File is empty: {input_path}")
        raise
    except pd.errors.ParserError:
        logger.error(f"Error parsing file: {input_path}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise


if __name__ == "__main__":
    app()
