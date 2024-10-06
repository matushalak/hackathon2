import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import time

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
    model.load_state_dict(torch.load(model_path, weights_only=True))
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
def main(model_name):
    # File paths
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

    return actual_values, predicted_values

def plot_single_cfm_matrix(cfm_directory="./our_brain_cfms", file_name=None):
    """
    Load and plot a single CFM matrix from a .npy file.

    Args:
        cfm_directory (str): Directory where the .npy files are stored.
        file_name (str): Name of the .npy file to load (without extension).

    """
    # Ensure the file exists
    file_path = os.path.join(cfm_directory, f"{file_name}.npy")

    if not os.path.exists(file_path):
        print(f"File {file_name}.npy not found in {cfm_directory}")
        return

    # Load the CFM matrix
    cfm_matrix = np.load(file_path)

    # make it from 1,8,8 to 8,8,1
    cfm_matrix = np.moveaxis(cfm_matrix, 0, -1)

    # Plot the CFM matrix
    plt.figure(figsize=(6, 6))
    plt.imshow(cfm_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(label='CFM Values')
    plt.title(f"CFM Matrix from {file_name}.npy")

    plt.show()

def predict_and_plot_single_cfm(cfm_directory="./our_brain_cfms", file_name=None, model_path=None, device="mps"):
    """
    Load and plot a single CFM matrix and predict its valence and arousal using the pre-trained CNN model.

    Args:
        cfm_directory (str): Directory where the .npy files are stored.
        file_name (str): Name of the .npy file to load (without extension).
        model_path (str): Path to the pre-trained model.
        device (str): Device to run the prediction on ("cpu" or "cuda").
    """

    # Ensure the file exists
    file_path = os.path.join(cfm_directory, f"{file_name}.npy")

    model_path = f"./models_og/{model_path}"  # Path to saved model
    if not os.path.exists(file_path):
        print(f"File {file_name}.npy not found in {cfm_directory}")
        return

    # Load the CFM matrix
    cfm_matrix = np.load(file_path)

    # Reshape cfm_matrix from (1, 8, 8) to (8, 8) to plot it with imshow
    cfm_matrix = np.squeeze(cfm_matrix)

    # Load the model
    model = Cnn8()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # Preprocess the CFM matrix (convert to tensor and add necessary dimensions)
    cfm_tensor = torch.tensor(cfm_matrix, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(
        device)  # Shape (1, 1, 8, 8)

    # Perform the prediction
    with torch.no_grad():
        prediction = model(cfm_tensor).cpu().numpy()[0]  # Get the prediction and convert to numpy

    valence_pred = prediction[0]
    arousal_pred = prediction[1]

    # Plot the CFM matrix and predicted values
    plt.figure(figsize=(8, 6))

    # Plot the CFM matrix
    plt.subplot(1, 2, 1)
    plt.imshow(cfm_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(label='CFM Values')
    plt.title(f"CFM Matrix from {file_name}.npy")


    # Display the predictions
    plt.subplot(1, 2, 2)
    plt.text(0.5, 0.5, f"Predicted Valence: {valence_pred:.2f}\nPredicted Arousal: {arousal_pred:.2f}",
             horizontalalignment='center', verticalalignment='center', fontsize=14,
             bbox=dict(facecolor='white', alpha=0.8))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')

    plt.suptitle("CFM Matrix and Predicted Valence/Arousal", fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_histograms_from_labels(labels):
    # Split actual values for valence and arousal
    actual_valence = labels[:, 0]
    actual_arousal = labels[:, 1]

    # Plot histograms for valence
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)  # Subplot 1 for Valence
    plt.hist(actual_valence, bins=5, edgecolor='black', alpha=0.7, color='blue')
    plt.title("Histogram of Valence")
    plt.xlabel("Valence")
    plt.ylabel("Frequency")
    plt.grid(True)

    # Plot histograms for arousal
    plt.subplot(1, 2, 2)  # Subplot 2 for Arousal
    plt.hist(actual_arousal, bins=5, edgecolor='black', alpha=0.7, color='green')
    plt.title("Histogram of Arousal")
    plt.xlabel("Arousal")
    plt.ylabel("Frequency")
    plt.grid(True)

    # Display the plots
    plt.tight_layout()
    plt.show()


# import labels and cfms
def load_labels_and_cfms():
    # Load the labels and CFMs
    labels_path = "./data_og/raw/labels.pkl"
    cfms_path = "./data_og/raw/cfms.pkl"

    with open(labels_path, "rb") as file_:
        labels = pkl.load(file_)

    with open(cfms_path, "rb") as file_:
        cfms = pkl.load(file_)

    return labels, cfms


def compute_median_split_metrics(true_values, predicted_values, label_name):
    # Calculate the median for true values and predicted values
    median_true = np.median(true_values)
    median_pred = np.median(predicted_values)

    # Convert to binary (>= median is high (1), < median is low (0))
    true_binary = (true_values >= median_true).astype(int)
    pred_binary = (predicted_values >= median_pred).astype(int)

    # Calculate accuracy
    accuracy = np.mean(true_binary == pred_binary)

    # Calculate sensitivity and specificity
    TP = np.sum((true_binary == 1) & (pred_binary == 1))
    TN = np.sum((true_binary == 0) & (pred_binary == 0))
    FP = np.sum((true_binary == 0) & (pred_binary == 1))
    FN = np.sum((true_binary == 1) & (pred_binary == 0))

    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

    print(f"{label_name} Metrics:")
    print(f"  Accuracy: {accuracy * 100:.2f}%")
    print(f"  Sensitivity: {sensitivity * 100:.2f}%")
    print(f"  Specificity: {specificity * 100:.2f}%")
    print(f"  Median of True Values: {median_true}")
    print(f"  Median of Predicted Values: {median_pred}")

    return {"accuracy": accuracy, "sensitivity": sensitivity, "specificity": specificity}


def calculate_metrics(true_labels, predictions):
    # Split the labels and predictions into valence and arousal components
    true_valence = np.array([label[0] for label in true_labels])
    true_arousal = np.array([label[1] for label in true_labels])
    pred_valence = np.array([pred[0] for pred in predictions])
    pred_arousal = np.array([pred[1] for pred in predictions])

    # Calculate accuracy, sensitivity, and specificity for valence
    valence_metrics = compute_median_split_metrics(true_valence, pred_valence, 'Valence')

    # Calculate accuracy, sensitivity, and specificity for arousal
    arousal_metrics = compute_median_split_metrics(true_arousal, pred_arousal, 'Arousal')

    return valence_metrics, arousal_metrics

def predict_single_cfm(cfm_matrix, model_path, device="cpu"):
    """
    Load a CNN model and predict the valence and arousal values from a given CFM matrix.

    Args:
        cfm_matrix (numpy.ndarray): The CFM matrix (must be of shape (1, 8, 8) or (8, 8)).
        model_path (str): Path to the pre-trained model.
        device (str): Device to run the prediction on ("cpu" or "cuda").

    Returns:
        tuple: Predicted valence and arousal values.
    """
    # Ensure cfm_matrix has the correct shape (1, 8, 8) -> convert to (1, 1, 8, 8)
    if cfm_matrix.shape == (8, 8):
        cfm_matrix = np.expand_dims(cfm_matrix, axis=0)
    elif cfm_matrix.shape != (1, 8, 8):
        raise ValueError(f"Invalid CFM shape: {cfm_matrix.shape}, expected (1, 8, 8) or (8, 8)")

    # Load the model
    model = Cnn8()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # Convert the CFM matrix to a tensor and add a batch dimension
    cfm_tensor = torch.tensor(cfm_matrix, dtype=torch.float32).unsqueeze(0).to(device)  # Shape (1, 1, 8, 8)

    # Perform the prediction
    with torch.no_grad():
        prediction = model(cfm_tensor).cpu().numpy()[0]  # Get the prediction and convert to numpy

    valence_pred = prediction[0]
    arousal_pred = prediction[1]

    return valence_pred, arousal_pred


if __name__ == "__main__":
    # Load the labels and CFMs
    labels, cfms = load_labels_and_cfms()

    plot_histograms_from_labels(labels)

    model_name = "8x864_0.001_200epochs_512hiddenlayers.pt"
    actual, predicted = main(model_name)
    valence_metrics, arousal_metrics = calculate_metrics(actual, predicted)


    # User specifies the file name (without .npy extension)
    file_name = input("Enter the name of the CFM .npy file (without extension): ")


    # start timer
    start = time.time()
    # Call the function to load and plot the matrix
    plot_single_cfm_matrix(file_name=file_name)
    predict_and_plot_single_cfm(file_name=file_name,
                                model_path=model_name
                                )

    # end timer
    end = time.time()
    print(f"Time taken: {end - start:.2f} seconds")
# gyros