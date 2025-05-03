import pandas as pd
import joblib
import argparse
from loguru import logger

def predict_from_model(features_path, model_path, predictions_path):
    logger.info("Loading features...")
    X_test = pd.read_csv(features_path)

    logger.info("Loading model...")
    model = joblib.load(model_path)

    logger.info("Generating predictions...")
    predictions = model.predict(X_test)
    pd.DataFrame(predictions).to_csv(predictions_path, index=False)
    logger.success(f"Predictions saved to {predictions_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict using a trained model")
    parser.add_argument("--features-path", type=str, required=True, help="Path to test features CSV")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model file")
    parser.add_argument("--predictions-path", type=str, required=True, help="Path to save predictions CSV")

    args = parser.parse_args()
    predict_from_model(args.features_path, args.model_path, args.predictions_path)
