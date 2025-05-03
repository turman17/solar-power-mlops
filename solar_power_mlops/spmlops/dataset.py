from heapq import merge
from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from spmlops.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
import pandas as pd

app = typer.Typer()


@app.command()
def prepare_dataset(
    input_plant_path: Path = RAW_DATA_DIR / "Plant_1_Generation_Data.csv",
    input_weather_path: Path = RAW_DATA_DIR / "Plant_1_Weather_Sensor_Data.csv",
    output_path: Path = PROCESSED_DATA_DIR / "Processed_Plant_1_Generation_Data.csv",
):
    """Process the datasets and save the cleaned data.

    Args:
        input_plant_path (Path, optional): _description_. Defaults to RAW_DATA_DIR/"Plant_1_Generation_Data.csv".
        input_weather_path (Path, optional): _description_. Defaults to RAW_DATA_DIR/"Plant_1_Weather_Sensor_Data.csv".
        output_path (Path, optional): _description_. Defaults to PROCESSED_DATA_DIR/"Processed_Plant_1_Generation_Data.csv".
        
    Raises:
        FileNotFoundError: If the input files do not exist.
        Exception: If any other error occurs during processing.
    """

    try: 
        logger.info("Processing datasets...")
        plant1_data = pd.read_csv(input_plant_path)
        logger.info(f"Loaded {len(plant1_data)} records from {input_plant_path}.")

        weather_data = pd.read_csv(input_weather_path)
        logger.info(f"Loaded {len(weather_data)} records from {input_weather_path}.")

        plant1_data["DATE_TIME"] = pd.to_datetime(plant1_data["DATE_TIME"])
        weather_data["DATE_TIME"] = pd.to_datetime(weather_data["DATE_TIME"])
        merged_data = pd.merge(plant1_data, weather_data, on="DATE_TIME", how="inner")
        logger.info(f"Merged datasets, resulting in {len(merged_data)} records.")

        merged_data.dropna(inplace=True)
        logger.info(f"Cleaned data, resulting in {len(merged_data)} records after dropping NaN values.")

        merged_data.to_csv(output_path, index=False)
        logger.info(f"Processed data saved to {output_path}.")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")

    except Exception as e:
        logger.error(f"An error occurred: {e}")

    # -----------------------------------------


if __name__ == "__main__":
    app()
