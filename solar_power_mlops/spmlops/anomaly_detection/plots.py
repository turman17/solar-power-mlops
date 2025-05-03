import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger
import seaborn as sns

import typer

app = typer.Typer()
@app.command(name="plot-anomaly-predictions")
def plot_anomaly_predictions(predictions_path: Path, output_path: Path):
    """
    Generate a bar plot of anomaly prediction counts.

    Args:
        predictions_path (Path): Path to the CSV file containing anomaly predictions.
        output_path (Path): Path to save the generated plot image.
    """
    try:
        logger.info(f"Loading predictions from {predictions_path}...")
        df = pd.read_csv(predictions_path)

        if 'prediction' not in df.columns:
            logger.error("The 'prediction' column is missing in the input CSV.")
            return

        counts = df['prediction'].value_counts().sort_index()
        logger.info(f"Prediction value counts: {counts.to_dict()}")

        plt.figure(figsize=(6, 4))
        counts.plot(kind='bar', color=['red', 'green'])
        plt.title("Anomaly Detection Results")
        plt.xlabel("Prediction")
        plt.ylabel("Count")
        plt.xticks(ticks=[0, 1], labels=["Anomaly (-1)", "Normal (1)"], rotation=0)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logger.success(f"Anomaly plot saved to {output_path}")
    except Exception as e:
        logger.error(f"Error generating anomaly plot: {e}")


# New function: plot_anomaly_diagnostics
@app.command(name="plot-anomaly-diagnostics")
def plot_anomaly_diagnostics(predictions_path: Path, output_path: Path):
    """
    Generate a composite plot with multiple visualizations of anomaly detection results.
    """
    try:
        logger.info(f"Loading predictions from {predictions_path}...")
        df = pd.read_csv(predictions_path)

        if 'prediction' not in df.columns:
            logger.error("The 'prediction' column is missing in the input CSV.")
            return

        counts = df['prediction'].value_counts().sort_index()
        logger.info(f"Prediction value counts: {counts.to_dict()}")

        fig, axs = plt.subplots(2, 2, figsize=(12, 8))

        # Bar plot
        axs[0, 0].bar(['Anomaly (-1)', 'Normal (1)'], [counts.get(-1, 0), counts.get(1, 0)], color=['red', 'green'])
        axs[0, 0].set_title("Bar Plot of Predictions")
        axs[0, 0].set_ylabel("Count")

        # Pie chart
        axs[0, 1].pie([counts.get(-1, 0), counts.get(1, 0)],
                      labels=['Anomaly (-1)', 'Normal (1)'],
                      colors=['red', 'green'],
                      autopct='%1.1f%%',
                      startangle=90)
        axs[0, 1].set_title("Class Distribution Pie Chart")

        # Feature histogram (first numeric feature)
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        feature_col = next((col for col in numeric_cols if col != 'prediction'), None)
        if feature_col:
            df[df['prediction'] == -1][feature_col].hist(bins=30, ax=axs[1, 0], alpha=0.7, label='Anomaly', color='red')
            df[df['prediction'] == 1][feature_col].hist(bins=30, ax=axs[1, 0], alpha=0.7, label='Normal', color='green')
            axs[1, 0].set_title(f"Histogram of {feature_col}")
            axs[1, 0].set_xlabel(feature_col)
            axs[1, 0].legend()
        else:
            axs[1, 0].text(0.5, 0.5, 'No numeric feature found.', horizontalalignment='center', verticalalignment='center')

        # Index scatter plot
        axs[1, 1].scatter(df.index, df['prediction'], alpha=0.6, c=df['prediction'].map({-1: 'red', 1: 'green'}))
        axs[1, 1].set_title("Prediction by Index")
        axs[1, 1].set_xlabel("Index")
        axs[1, 1].set_ylabel("Prediction")

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logger.success(f"Anomaly diagnostic plot saved to {output_path}")
    except Exception as e:
        logger.error(f"Error generating anomaly diagnostics: {e}")


@app.command(name="plot-anomaly-trigger-points")
def plot_anomaly_trigger_points(predictions_path: Path, features_path: Path, output_path: Path):
    """
    Generate a scatter plot of two selected features with anomalies highlighted.
    """
    try:
        logger.info(f"Loading predictions from {predictions_path} and features from {features_path}...")
        predictions_df = pd.read_csv(predictions_path)
        features_df = pd.read_csv(features_path)

        if "prediction" not in predictions_df.columns:
            raise ValueError("Missing 'prediction' column in predictions file.")

        combined_df = features_df.copy()
        combined_df["prediction"] = predictions_df["prediction"]

        # Select 2 common features; adjust as needed
        x_feature = "IRRADIATION"
        y_feature = "AC_POWER"

        if x_feature not in combined_df.columns or y_feature not in combined_df.columns:
            raise ValueError(f"Features '{x_feature}' or '{y_feature}' not found in data.")

        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=combined_df,
            x=x_feature,
            y=y_feature,
            hue="prediction",
            palette={1: "green", -1: "red"},
            alpha=0.6,
        )
        plt.title(f"Anomalies in {x_feature} vs {y_feature}")
        plt.legend(title="Prediction")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logger.success(f"Anomaly scatter plot saved to {output_path}")
    except Exception as e:
        logger.error(f"Error generating scatter plot of anomalies: {e}")


if __name__ == "__main__":
    app()
