from typer import Typer
from spmlops.dataset import prepare_dataset
from spmlops.features import build_features, split_data
from spmlops.modeling.train import train
from spmlops.plots import plot_data

app = Typer()
app.command()(prepare_dataset)
app.command()(build_features)
app.command()(split_data)
app.command()(train)
app.command()(plot_data)

if __name__ == "__main__":
    app()
