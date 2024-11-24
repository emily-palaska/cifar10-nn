import json
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})

def load_json(file_path):
    """Load JSON data from a file."""
    with open(file_path, 'r') as file:
        return json.load(file)


def extract_losses(data):
    """Extract 'loss' values from the 'epochs' array in JSON data."""
    return [epoch["loss"] for epoch in data.get("epochs", []) if "loss" in epoch]


def plot_losses(files, labels, save_path="../plots/loss_plot.png"):
    """Plot and save the loss values from two JSON files."""
    # Load JSON files and extract loss values
    data = []
    losses = []
    for file in files:
        data.append(load_json(file))
        losses.append(extract_losses(data[-1]))

    # Plot losses
    plt.figure(figsize=(10, 6))
    for i, loss in enumerate(losses):
        plt.plot(loss, label=f'{labels[i]}')
    plt.title('Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.savefig(save_path)
    plt.show()
    print(f"Plot saved to {save_path}")


# Example usage
file1 = "../results/results_fc_3layers_lr0.01_he.json"

files = [file1]
labels = ['FC - 3 layers, LR 0.01, Xavier']
plot_losses(files, labels)
