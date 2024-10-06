import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from tqdm import tqdm
import pickle as pkl
import numpy as np
import random
import wandb
from math import sqrt
import fire
import os

random.seed(42)
np.random.seed(42)
torch.random.manual_seed(42)


class Cnn(nn.Module):  # We are defining to pytorch that we are building a model
    def __init__(
        self,
        in_size: tuple = (32, 32, 1),
        targets: int = 2,
        filter_size: int = 64,
        padding: int = 1,
        kernel_conv: int = 3,
        hiddenlayer: int = 512,
    ):
        super().__init__()  # This initializes all the torch things
        self.filter_size = filter_size
        self.kernel_conv = kernel_conv
        self.padding = padding
        self.hiddenlayer = hiddenlayer

        self.input_layer = nn.Conv2d(  # 32x32 pixels
            in_channels=in_size[-1],
            out_channels=self.filter_size,
            kernel_size=self.kernel_conv,
            padding=self.padding,
        )
        self.relu = (  # Activation function
            nn.ReLU()  # We can change this to leaky Relu if we need something a bit more fancy
        )

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 16x16 pixels
        self.layer2 = nn.Conv2d(
            in_channels=self.filter_size,
            out_channels=self.filter_size * 2,
            kernel_size=self.kernel_conv,
            padding=self.padding,
        )
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 8x8 pixels
        self.layer3 = nn.Conv2d(
            in_channels=self.filter_size * 2,
            out_channels=self.filter_size * 4,
            kernel_size=self.kernel_conv,
            padding=self.padding,
        )
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 4x4 pixels

        self.fc1 = nn.Linear(
            in_features=self.filter_size * 4 * (in_size[0] // 8) * (in_size[1] // 8),
            out_features=self.hiddenlayer,
        )  # Fully connected layer 1, aka flattened layer 1
        # self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(
            in_features=self.hiddenlayer, out_features=targets
        )  # Fully connected layer 2, aka just the output layer

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

        x = x.view(x.size(0), -1)  # This is how you flatten

        x = self.fc1(x)
        x = self.relu(x)
        # x = self.dropout(x)
        x = self.fc2(x)
        # x = torch.clamp(x, min=1, max=5)
        return x

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


class CFMS(Dataset):

    def __init__(
        self,
        cfms_path="./data_og/raw/cfms.pkl",
        labels_path="./data_og/raw/labels.pkl",
        transform=None,
    ):
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


def r2_score(y_true, y_pred):
    ss_res = torch.sum((y_true - y_pred) ** 2)  # Residual sum of squares
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)  # Total sum of squares
    r2 = 1 - ss_res / ss_tot
    return r2




def main(batch_size=64, device="mps", lr=1e-3, epochs=200, in_size=(8, 8, 1)):
    torch.set_num_threads(8)
    name_model = f"8x8{batch_size}_{lr}_{epochs}epochs_512hiddenlayers"
    wandb.init(
        project="Hackaton",
        name=name_model,
        config=locals(),
    )  # Setup the tracker, locals() refers to the args in the main()

    # Define the transformations for the dataset
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Convert images to tensor
            transforms.Resize(
                (in_size[0], in_size[1])
            ),  # Make the CFMS a bit larger (maybe check this as a hparam)
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), May reenable to improve results
        ]  # Normalize the images
    )

    # Load the training set
    dataset = CFMS(transform=transform)
    # Define the sizes for train, validation, and test sets
    train_size = int(0.7 * len(dataset))  # 80% for training
    val_size = len(dataset) - train_size  # 10% for validation
    # test_size = len(dataset) - train_size - val_size  # 10% for test

    # Split the dataset into train, val, and test
    trainset, val_dataset = random_split(dataset, [train_size, val_size])
    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
    )
    valloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    # testloader = DataLoader(
    #     test_dataset,
    #     batch_size=batch_size,
    #     shuffle=False,
    # )

    model = Cnn8()
    model.to(device)

    wandb.watch(model)

    lossfn = nn.MSELoss()  # Use MSE bc we are doing regression
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_train = 0.0
        total_r2_train = 0.0
        totals = 0
        with tqdm(total=len(trainloader.dataset) // batch_size) as pbar:
            for i, (cfms, targets) in enumerate(trainloader):
                cfms = cfms.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()  # Important to do first so it cleans the gradient of the last batch/steps

                outputs = model(cfms)
                loss = lossfn(outputs, targets)
                loss.backward()
                optimizer.step()

                pbar.set_postfix({"loss": loss.item()})
                pbar.update()
                r2_train = r2_score(targets, outputs)

                total_train += loss.item()
                total_r2_train += r2_train.item()
                totals += i

                wandb.log(
                    {
                        "losses/train_loss": loss.item(),
                        "losses/train_loss_RMSE": sqrt(loss.item()),
                        "train r2": r2_train.item(),
                    }
                )
        wandb.log(
            {
                "losses/total_train": total_train / totals,
                "losses/total_train_loss_RMSE": sqrt(total_train / totals),
                "losses/total_r2_train": total_r2_train / totals,
            }
        )

        model.eval()
        with torch.no_grad():  # Enforces not using gradients
            with tqdm(total=len(valloader.dataset) // batch_size) as pbar:
                validation_loss = 0
                validation_r2 = 0
                totals = 0
                for i, (cfms, targets) in enumerate(valloader):
                    cfms = cfms.to(device)
                    targets = targets.to(device)

                    outputs = model(cfms)

                    loss = lossfn(outputs, targets)

                    validation_loss += loss.item()
                    totals += i

                    pbar.set_postfix({"loss": loss.item()})
                    pbar.update()

                    r2_val = r2_score(targets, outputs)
                    validation_r2 += r2_val.item()

                avg_val_loss = validation_loss / totals
                avg_val_r2 = validation_r2 / totals
                wandb.log(
                    {
                        "losses/validation_loss": avg_val_loss,
                        "losses/validation_RMSE": sqrt(avg_val_loss),
                        "r2/validation_r2": avg_val_r2,
                    }
                )
                print("Validation_loss:", validation_loss / totals)

    # Do Test
    # with torch.no_grad():
    #     with tqdm(total=len(testloader.dataset) // batch_size) as pbar:
    #         test_loss = 0
    #         totals = 0
    #         for i, (cfms, targets) in enumerate(testloader):
    #             cfms = cfms.to(device)
    #             targets = targets.to(device)

    #             outputs = model(cfms)

    #             loss = lossfn(outputs, targets)

    #             test_loss += loss.item()
    #             totals += cfms.size(0)

    #             pbar.set_postfix({"loss": loss.item()})
    #             pbar.update()

    #         wandb.log({"losses/test_loss": test_loss / totals})
    #         wandb.log({"losses/test_RMSE": sqrt(test_loss / totals)})
    #         print("Validation_loss:", test_loss / totals)
    # Evaluation loop goes here when you have defined the loader etc.
    # The model.eval() is important so the weights dont update
    # The with nograd is also important for that

    # Save the model

    save_to = "./models_og"
    os.makedirs(save_to, exist_ok=True)
    torch.save(model.state_dict(), f"{save_to}/{name_model}.pt")



if __name__ == "__main__":
    fire.Fire(main())
