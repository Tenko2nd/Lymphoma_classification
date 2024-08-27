"""\033[38;5;224m    This python script is used to train a model based on a dataset required as an input.
    It can take on multiple parameters : 
        - batch_size (default 32)
        - workers (default 4)
        - learning_rate (default 0.001)
        - early_stop (default 5)
        - name (default '')
    It will save the output model on the Model folder as 'mod_{name}_{date}.pth' and the learning curve of the model.
    The model as an early stopping and only take the most efficient model in loss.\033[38;5;213m
"""

from datetime import datetime
import argparse
import math
import pickle
import os
import warnings

from sklearn import preprocessing
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve, auc
from torch.utils.data import DataLoader
from torchvision import models
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import lymphoma_dataset_class as L
import constant as C


warnings.simplefilter("ignore", FutureWarning)


def parser_init():
    """initialize the parser and returns it"""
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
        "--pb_disable",
        dest="disable",
        action="store_true",
        help="If you want to disable the progress bar (default : True)",
    )

    parser.add_argument(
        "-name",
        "--name",
        default="anon",
        help="The name of the categories the model is training on\033[0m",
        required=False,
    )

    return parser


def train(p: argparse.ArgumentParser, dataset, le, save_path):
    """Create a model with the base resnet18 and modification of the fc layers.
        Automatically save the best model based on the validation loss

    Args:
        p (argparse.ArgumentParser): the parser initialized before
        dataset (Dict of torch.utils.data.Dataset): The lymphoma dataset used to train the model
        le (sklearn.preprocessing.LabelEncoder): the label encoder used for the classification of the dataset
        save_path (str): the path where the model will be saved

    Returns:
        List: the lists of loss and accuracy over epoch for train and validation
    """
    # Parameters
    BATCH_SIZE = p.parse_args().batch_size
    EARLY_STOP = p.parse_args().early_stop
    LEARNING_RATE = p.parse_args().learning_rate
    MINIMUM_EPOCH = 10
    NUM_EPOCHS = 500
    WORKERS = p.parse_args().workers

    # Load train and validation dataset
    train_data, val_data = [dataset[x] for x in ["train", "val"]]
    lenDataSet = {"train": len(train_data), "val": len(val_data)}
    loaders = {
        "train": DataLoader(
            train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS
        ),
        "val": DataLoader(
            val_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS
        ),
    }

    # create model
    model = models.resnet18(weights="DEFAULT")  # change weights
    # Freeze all layers except the final classification layer
    for name, param in model.named_parameters():
        param.requires_grad = "fc" in name  # try all layers
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Move the model to the GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Training loop
    es = 0
    trainLossList, valLossList, trainAccList, valAccList = [], [], [], []
    for i in range(NUM_EPOCHS):
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Loop for each batch in the loaders
            for idx, (inputs, labels, _) in tqdm(
                enumerate(loaders[phase]),
                total=(math.ceil(lenDataSet[phase] / BATCH_SIZE)),
                disable=p.parse_args().disable,
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
                # save the model if it beats previous performance
                if epoch_loss < min(valLossList) if valLossList else float("inf"):
                    print(
                        "\033[92mBest model\033[95m\033[3m"
                        + f"(Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f})\033[0m"
                    )
                    torch.save(model.state_dict(), save_path)
                    es = 0
                # Stop if it didn't beat it enought time in a row
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

    return trainLossList, valLossList, trainAccList, valAccList

def saveLearnCurves(tLoss, vLoss, tAcc, vAcc, save_path):
    """save the learning curves figure

    Args:
        tLoss (List): training loss list of the model
        vLoss (List): validation loss list of the model
        tAcc (List): training accuracy list of the model
        vAcc (List): validation accuracy list of the model
        save_path (str): path to save the image
    """
    plt.subplot(2, 1, 1)
    plt.plot(tLoss, "g", vLoss, "r")
    plt.legend(("train", "val"))
    plt.title("Loss")
    plt.subplot(2, 1, 2)
    plt.plot(tAcc, "g", vAcc, "r")
    plt.title("Accuracy")
    plt.legend(("train", "val"))
    plt.savefig(f"{save_path}_res.png")

def test(testDS, model_path, le):
    """Test the performance of the model on unseen data from the dataset

    Args:
        testDS (torch.utils.data.Dataset): the test dataset containing unseen data
        model_path (str): the path of the model to train
        le (sklearn.preprocessing.LabelEncoder): the label encoder used for the classification of the dataset

    Returns:
        List: The lists of targets and predictions of the model
    """
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 1000)  # Adjust to match the original model's output units
    model.load_state_dict(torch.load(model_path))
    model.eval()

    y_test = le.transform(testDS.targets)
    y_pred = []
    from tqdm import tqdm
    for img, _, _ in tqdm(testDS):
        # Perform inference
        with torch.no_grad():
            output = model(img.float().unsqueeze(0))
        y_pred.append(F.softmax(output, dim=1)[:, 1])

    return y_test, y_pred

def saveAUC(y_test, y_pred, save_path):
    """Save the AUC scores for the model

    Args:
        y_test (List): list of targets
        y_pred (List): list of predictions of the model
        save_path (str): path to save the images
    """
    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    pr_auc =  auc(recall, precision)
    plt.figure()
    plt.plot(fpr, tpr, marker='.', label=f"Test dataset (AUC:{roc_auc:.3f})")
    plt.plot([0,1], [0,1], marker='.', linestyle='--', label="No Skill")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate' )
    plt.legend()
    plt.savefig(f"{save_path}_auc_roc.png")
    # Precision Recall
    plt.figure()
    plt.plot(recall, precision, marker='.', label=f"Test dataset (AUC:{pr_auc:.3f})")
    plt.plot([1,0], [0.5,0.5], marker='.', linestyle='--', label="No Skill")
    plt.ylabel('Precision')
    plt.xlabel('Recall' )
    plt.legend()
    plt.savefig(f"{save_path}_auc_pr.png")


if __name__ == "__main__":
    parser = parser_init()

    dataset_path = parser.parse_args().dataset_path
    df = pd.read_csv(dataset_path)

    # print repartition LCM / LZM for each folder
    for x in ["train", "val"] + (["test"] if "test" in df["folder"].values else []): 
        print(f"{x} : {df.loc[df["folder"] == x].groupby(["categorie"])["categorie"].count().to_dict()}")

    dataset = {
        x: L.LymphomaDataset(
            pd_file=df.loc[df["folder"] == x].reset_index(drop=True),
            transform=transforms.Compose(
                [
                    L.Rescale(360),
                    L.Crop(360),
                    L.ToTensor(),
                    transforms.Normalize(C.lymphomas_mean, C.lymphomas_std),
                ]
            ),
        )
        for x in ["train", "val"] + (["test"] if "test" in df["folder"].values else [])
    }
    
    le_path = f"{os.getcwd()}/Lymphoma_labelEncoder.pkl"
    with open(le_path, "rb") as f:
        le = preprocessing.LabelEncoder()
        le.classes_ = pickle.load(f)

    date = datetime.now().strftime("%m%d-%H%M")
    name = parser.parse_args().name
    save_model_path = f"{os.getcwd()}/Model/mod_{name}_{date}/mod_{name}_{date}.pth"
    
    # create folder for model, result and AUC to save in
    os.mkdir(f"{os.getcwd()}/Model/mod_{name}_{date}")

    tLoss, vLoss, tAcc, vAcc = train(parser, dataset, le, save_model_path)
    saveLearnCurves(tLoss, vLoss, tAcc, vAcc, save_model_path.split('.')[0])

    if "test" in dataset:
        y_test, y_pred = test(dataset["test"], save_model_path, le)
        saveAUC(y_test, y_pred)
