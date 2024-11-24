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


def plot_losses(files, labels, save_path="loss_plot.png"):
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
file1 = "../results/results_vgg16_lr0.0001_20ep.json"
file2 = "../results/results_cnn_16,16_bs1000_lr0.001_xavier.json"
files = [file2, file1]
labels = ['CNN - Exp1 - 100 epochs', 'CNN - Exp2 - 30 epochs']
plot_losses(files, labels)
