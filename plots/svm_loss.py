import json, random
import matplotlib.pyplot as plt

# Load the JSON data
with open('../results/svm/1v1_lr0.0001_l0.01_n10.json', 'r') as file:
    data = json.load(file)

# Extract all classifiers
fit_data = data["fit"]
all_classifiers = list(fit_data.keys())

# Randomly select 10 classifiers
random_classifiers = random.sample(all_classifiers, 10)

# Extract the loss data for the selected classifiers
losses = {classifier: fit_data[classifier]["loss"] for classifier in random_classifiers}

# Plot the losses
plt.figure(figsize=(10, 6))
for classifier, loss in losses.items():
    plt.plot(loss, label=classifier)

# Add labels, legend, and title
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curves for 10 Classifiers")
plt.legend(title="Classifiers")
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.savefig('svm/svm_loss.png')
