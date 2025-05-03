from typer import Typer
from spmlops.dataset import prepare_dataset
from spmlops.features import build_features, split_data
from spmlops.modeling.train import train
from spmlops.plots import plot_data
from spmlops.anomaly_detection import train_anomaly_detection, plot_anomaly_predictions, plot_anomaly_diagnostics, plot_anomaly_trigger_points

app = Typer()

app.command()(prepare_dataset)
app.command()(build_features)
app.command()(split_data)
app.command()(train)
app.command()(plot_data)

# anomaly detection commands
app.command()(train_anomaly_detection)
app.command()(plot_anomaly_predictions)
app.command()(plot_anomaly_diagnostics)
app.command()(plot_anomaly_trigger_points)

if __name__ == "__main__":
    app()
