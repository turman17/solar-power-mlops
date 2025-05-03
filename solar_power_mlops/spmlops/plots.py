import matplotlib.pyplot as plt
import pandas as pd
import joblib
import argparse
import mlflow


def plot_data(
    features_path, y_test_path, model_path, predictions_path, output_path, log_to_mlflow=False
):
    # Load data
    X_test = pd.read_csv(features_path)
    y_test = pd.read_csv(y_test_path)
    predictions = pd.read_csv(predictions_path)

    # Load model
    model = joblib.load(model_path)

    residuals = y_test.values.flatten() - predictions.values.flatten()

    # Plot setup
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Scatter Plot
    axs[0, 0].scatter(y_test, predictions, alpha=0.5, color="orange", label="Predicted vs Actual")
    axs[0, 0].plot(
        [y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "b--", label="Ideal"
    )
    axs[0, 0].set_title("Scatter: Actual vs Predicted")
    axs[0, 0].set_xlabel("Actual")
    axs[0, 0].set_ylabel("Predicted")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # 2. Residuals Plot
    axs[0, 1].scatter(range(len(residuals)), residuals, alpha=0.5)
    axs[0, 1].axhline(0, color="red", linestyle="--")
    axs[0, 1].set_title("Residuals Plot")
    axs[0, 1].set_xlabel("Sample Index")
    axs[0, 1].set_ylabel("Residual (Actual - Predicted)")
    axs[0, 1].grid(True)

    # 3. Residual Histogram
    axs[1, 0].hist(residuals, bins=50, color="skyblue", edgecolor="black")
    axs[1, 0].set_title("Histogram of Residuals")
    axs[1, 0].set_xlabel("Residual")
    axs[1, 0].set_ylabel("Frequency")
    axs[1, 0].grid(True)

    # 4. Time Series Plot (first 300 samples) for DC_POWER and AC_POWER
    axs[1, 1].plot(y_test.iloc[:300, 1].values, label="Actual AC", linewidth=2)
    axs[1, 1].plot(predictions.iloc[:300, 1].values, label="Predicted AC", linewidth=2, alpha=0.7)
    axs[1, 1].plot(y_test.iloc[:300, 0].values, label="Actual DC", linewidth=2)
    axs[1, 1].plot(predictions.iloc[:300, 0].values, label="Predicted DC", linewidth=2, alpha=0.7)
    axs[1, 1].set_title("Time Series (First 300 samples)")
    axs[1, 1].set_xlabel("Sample Index")
    axs[1, 1].set_ylabel("Power Generation")
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot power generation predictions")
    parser.add_argument(
        "--features-path", type=str, required=True, help="Path to features CSV file"
    )
    parser.add_argument(
        "--y-test-path", type=str, required=True, help="Path to true output CSV file"
    )
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model file")
    parser.add_argument(
        "--predictions-path", type=str, required=True, help="Path to predictions CSV file"
    )
    parser.add_argument(
        "--output-path", type=str, required=True, help="Path to save the plot image"
    )

    args = parser.parse_args()

    plot_data(
        args.features_path,
        args.y_test_path,
        args.model_path,
        args.predictions_path,
        args.output_path,
    )
