import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import random

random.seed(42)
np.random.seed(42)
torch.random.manual_seed(42)
# Define the CNN model (same as used during training)
class Cnn8(torch.nn.Module):
    def __init__(self, in_size=(8, 8, 1), targets=2, filter_size=64, padding=1, kernel_conv=3, hiddenlayer=512):
        super().__init__()
        self.filter_size = filter_size
        self.kernel_conv = kernel_conv
        self.padding = padding
        self.hiddenlayer = hiddenlayer

        self.input_layer = torch.nn.Conv2d(
            in_channels=in_size[-1], out_channels=self.filter_size, kernel_size=self.kernel_conv, padding=self.padding)
        self.relu = torch.nn.ReLU()

        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer2 = torch.nn.Conv2d(in_channels=self.filter_size, out_channels=self.filter_size * 2,
                                      kernel_size=self.kernel_conv, padding=self.padding)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer3 = torch.nn.Conv2d(in_channels=self.filter_size * 2, out_channels=self.filter_size * 4,
                                      kernel_size=self.kernel_conv, padding=self.padding)
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = torch.nn.Linear(in_features=self.filter_size * 4 * (in_size[0] // 8) * (in_size[1] // 8),
                                   out_features=self.hiddenlayer)
        self.fc2 = torch.nn.Linear(in_features=self.hiddenlayer, out_features=targets)

    def forward(self, x: torch.Tensor):
        x = self.input_layer(x)
        x = self.relu(x)
        x = self.maxpool1(x)

        x = self.layer2(x)
        x = self.relu(x)
        x = self.maxpool2(x)

        x = self.layer3(x)
        x = self.relu(x)
        x = self.maxpool3(x)

        x = x.view(x.size(0), -1)  # Flatten

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Custom Dataset class for loading CFMs and labels
class CFMSDataset(torch.utils.data.Dataset):
    def __init__(self, cfms_path, labels_path, transform=None):
        with open(cfms_path, "rb") as file_:
            self.cfms = pkl.load(file_)
        with open(labels_path, "rb") as file_:
            self.labels = pkl.load(file_)
        self.transform = transform

    def __getitem__(self, index):
        cfms = self.cfms[index].astype(np.float32)
        if self.transform:
            cfms = self.transform(cfms)
        label = torch.tensor(self.labels[index], dtype=torch.float32)
        return cfms, label

    def __len__(self):
        return self.cfms.shape[0]


# Function to load the model and make predictions
def load_model_and_predict_test(model_path, testloader, device="cpu"):
    # Load the model
    model = Cnn8()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    actual_values = []
    predicted_values = []

    with torch.no_grad():
        for cfms, targets in testloader:
            cfms, targets = cfms.to(device), targets.to(device)
            outputs = model(cfms)

            actual_values.extend(targets.cpu().numpy())
            predicted_values.extend(outputs.cpu().numpy())

    return np.array(actual_values), np.array(predicted_values)


# Function to plot histograms for each actual value category on the same graph
def plot_histograms_on_same_graph(actual_values, predicted_values, value_name="Valence", model_name="CNN"):
    unique_values = np.unique(actual_values)

    # Define colors for the histograms
    colors = ['blue', 'green', 'red', 'purple', 'orange']

    plt.figure(figsize=(10, 6))

    # Plot a histogram for each actual value
    for i, actual_value in enumerate(unique_values):
        indices = np.where(actual_values == actual_value)
        predictions_for_value = predicted_values[indices]

        plt.hist(predictions_for_value, bins=20, alpha=0.5, color=colors[i],
                 edgecolor='black', label=f'Actual {value_name} = {actual_value}')

    plt.title(f"Histograms of Predicted {value_name} for Different Actual {value_name} Values")
    plt.suptitle(f"{model_name}")
    plt.xlabel(f"Predicted {value_name}")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.show()


# Main function to load, predict, plot (only for test data)
def main():
    # File paths
    model_name = "88_64_0.01_300epochs_bestparams.pt"
    model_path = f"./models_og/{model_name}"  # Path to saved model
    cfms_path = "./data_og/raw/cfms.pkl"
    labels_path = "./data_og/raw/labels.pkl"

    # Load the dataset and split it into train, validation, and test
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((8, 8))])
    dataset = CFMSDataset(cfms_path=cfms_path, labels_path=labels_path, transform=transform)

    # Split the dataset into train, validation, and test
    train_size = int(0.7 * len(dataset))  # 70% for training
    val_size = int(0.1 * len(dataset))  # 10% for validation
    test_size = len(dataset) - train_size - val_size  # 20% for test

    trainset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Predict on test data and plot
    actual_values, predicted_values = load_model_and_predict_test(model_path, testloader)

    # Split actual and predicted values into valence and arousal
    actual_valence = actual_values[:, 0]
    predicted_valence = predicted_values[:, 0]

    actual_arousal = actual_values[:, 1]
    predicted_arousal = predicted_values[:, 1]

    # Plot histograms on the same graph for valence and arousal
    plot_histograms_on_same_graph(actual_valence, predicted_valence, value_name="Valence", model_name=model_name)
    plot_histograms_on_same_graph(actual_arousal, predicted_arousal, value_name="Arousal", model_name=model_name)


if __name__ == "__main__":
    main()

# gyros