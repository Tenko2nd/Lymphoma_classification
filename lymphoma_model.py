"""\033[38;5;224m    This python script is used to train a model based on a dataset required as an input.
    It can take on multiple parameters : 
        - batch_size (default 32)
        - workers (default 4)
        - learning_rate (default 0.001)
        - early_stop (default 5)
        - name (default '')
    It will save the output model on the Model folder as 'mod_{name}_{date}.pth' and the learning curve of the model.
    The model as an early stopping and only take the most efficient model in loss.
    \033[38;5;213m
"""

import argparse
from lymphoma_dataset import LymphomaDataset, Rescale, Crop, ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import models
import torch
import pickle
import torch.nn as nn
import torch.optim as optim
from sklearn import preprocessing
import math
from tqdm import tqdm
from datetime import datetime


def main():

    date = datetime.now().strftime("%m%d-%H%M")

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "-d",
        "--dataset_path",
        help="The path of the dataset used for training",
        required=True,
    )

    parser.add_argument(
        "-bs",
        "--batch_size",
        default=32,
        type=int,
        help="The batch size for the model",
        required=False,
    )

    parser.add_argument(
        "-w",
        "--workers",
        default=4,
        type=int,
        help="The number of workers for the model",
        required=False,
    )

    parser.add_argument(
        "-lr",
        "--learning_rate",
        default=0.001,
        type=float,
        help="The learning rate for the model",
        required=False,
    )

    parser.add_argument(
        "-es",
        "--early_stop",
        default=5,
        type=int,
        help="The early stopping value for the model",
        required=False,
    )

    parser.add_argument(
        "-name",
        "--name",
        default="anon",
        help="The name of the categories the model is training on\033[0m",
        required=False,
    )

    name = parser.parse_args().name

    # Parameters
    BATCH_SIZE = parser.parse_args().batch_size
    WORKERS = parser.parse_args().workers
    DATASET_PATH = parser.parse_args().dataset_path
    LE_PATH = "Lymphoma_labelEncoder.pkl"
    LEARNING_RATE = parser.parse_args().learning_rate
    SAVE_MODEL_PATH = f"Model/mod_{name}_{date}.pth"
    NUM_EPOCHS = 1000
    EARLY_STOP = parser.parse_args().early_stop
    MINIMUM_EPOCH = 10

    with open(DATASET_PATH, "rb") as f:
        DATASET = LymphomaDataset
        DATASET = pickle.load(f)

    train_data, val_data, test_data = [DATASET[x] for x in ["train", "val", "test"]]

    lenDataSet = {
        "train": len(train_data),
        "val": len(val_data),
        "test": len(test_data),
    }

    with open(LE_PATH, "rb") as f:
        le = preprocessing.LabelEncoder()
        le.classes_ = pickle.load(f)

    loaders = {
        "train": DataLoader(
            train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS
        ),
        "val": DataLoader(
            val_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS
        ),
    }

    model = models.resnet18(weights="DEFAULT")

    # Freeze all layers except the final classification layer
    for name, param in model.named_parameters():
        param.requires_grad = "fc" in name

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Move the model to the GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    es = 0
    # Training loop
    trainLossList, valLossList, trainAccList, valAccList = [], [], [], []
    for i in range(NUM_EPOCHS):
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for idx, (inputs, labels, _) in tqdm(
                enumerate(loaders[phase]),
                total=(math.ceil(lenDataSet[phase] / BATCH_SIZE)),
            ):
                inputs = inputs.float().to(device)
                labels = torch.tensor(le.transform(labels)).to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels.long())

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / lenDataSet[phase]
            epoch_acc = running_corrects.double().cpu().numpy() / lenDataSet[phase]

            if phase == "train":
                trainAccList.append(epoch_acc)
                trainLossList.append(epoch_loss)
            else:
                if epoch_loss < min(valLossList) if valLossList else float("inf"):
                    print(
                        "\033[92m "
                        + "Best model "
                        + "\033[95m\033[3m"
                        + f"(Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f})\033[0m"
                    )
                    # Save the model
                    torch.save(model.state_dict(), SAVE_MODEL_PATH)
                    es = 0
                else:
                    if es >= EARLY_STOP:
                        break
                    elif MINIMUM_EPOCH < idx:
                        es += 1
                valAccList.append(epoch_acc)
                valLossList.append(epoch_loss)

            print(f"{phase, i+1} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        if es >= EARLY_STOP:
            break

    print("Training complete!")

    plt.subplot(2, 1, 1)
    plt.plot(trainLossList, "g", valLossList, "r")
    plt.legend(("train", "val"))
    plt.title("Loss")
    plt.subplot(2, 1, 2)
    plt.plot(trainAccList, "g", valAccList, "r")
    plt.title("Accuracy")
    plt.legend(("train", "val"))
    plt.savefig(f"{SAVE_MODEL_PATH.split('.')[0]}_res.png")


if __name__ == "__main__":
    main()
