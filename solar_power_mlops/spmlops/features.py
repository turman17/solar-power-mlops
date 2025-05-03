from matplotlib import pyplot as plt

def plot_data(y_test, y_pred):
    # Plot actual and predicted as lines over sample index
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.values, label="Actual", color="blue", linewidth=1.5)
    plt.plot(y_pred, label="Predicted", color="orange", linewidth=1.5)
    plt.xlabel("Sample Index")
    plt.ylabel("DC Power")
    plt.title("Actual vs Predicted DC Power")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
