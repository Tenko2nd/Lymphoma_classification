from time import perf_counter as timer

from sklearn.metrics import roc_auc_score
import pandas as pd
import sklearn.preprocessing
from torch.utils.data import DataLoader
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from transformers import AutoModel
from MyModel_Class import MyModel


from IftimDevLib.IDL.pipelines.evaluation.classification import EarlyStopping


def loaders(dataset: torch.utils.data.Dataset, batch_size: int = 64, workers: int = 4):
    # Load train and validation dataset
    loaders = {
        x: DataLoader(
            dataset[x],
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            drop_last=True,
        )
        for x in ["train", "val"] + (["test"] if "test" in dataset.keys() else [])
    }
    return loaders


def create_model(
    learning_rate: float = 0.0001, decay: float = 0.0, precomputed: bool = False
):
    # create model
    model = MyModel(precomputed=precomputed)
    """
    models.efficientnet_b4(
        weights=models.EfficientNet_B4_Weights.DEFAULT
    )
    """
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)
    return criterion, optimizer, model


def train_step(
    criterion: nn.Module,
    device: torch.device,
    disable_tqdm: bool,
    label_encoder: sklearn.preprocessing.LabelEncoder,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_dataloader: DataLoader,
):
    # Cleaning remaining data in GPU
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Putting model in train mode
    model.train()

    # Setup train loss and train accuracy values
    tr_loss = 0
    # Create an empty DataFrame
    df = pd.DataFrame(columns=["predictions", "targets", "patients"])
    for inputs, labels, tabular in tqdm(
        train_dataloader,
        total=len(train_dataloader),
        disable=disable_tqdm,
    ):
        patients = tabular["patient"]

        targets = label_encoder.transform(labels)
        labels = torch.from_numpy(label_encoder.transform(labels))

        images = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        tr_loss = torch.add(tr_loss, loss.item())
        loss.backward()
        optimizer.step()

        preds = F.softmax(outputs, dim=1)[:, 1].cpu().detach().numpy()
        new_df = pd.DataFrame(
            {
                "predictions": preds,
                "targets": targets,
                "patients": patients,
            }
        )
        # Append the new DataFrame to the main DataFrame
        df = pd.concat([df, new_df])

    df = df.groupby("patients")[["predictions", "targets"]].mean().reset_index()

    tr_score = roc_auc_score(
        df["targets"].tolist(), df["predictions"].tolist(), multi_class="ovo"
    )

    del df

    return float(tr_loss / len(train_dataloader)), tr_score


def val_step(
    criterion: nn.Module,
    device: torch.device,
    disable_tqdm: bool,
    label_encoder: sklearn.preprocessing.LabelEncoder,
    model: nn.Module,
    val_dataloader: DataLoader,
):
    # Cleaning remaining data in GPU
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Putting model in eval mode
    model.eval()

    val_loss = 0
    # Create an empty DataFrame
    df = pd.DataFrame(columns=["predictions", "targets", "patients"])

    for inputs, labels, tabular in tqdm(
        val_dataloader,
        disable=disable_tqdm,
    ):
        patients = tabular["patient"]

        targets = label_encoder.transform(labels)
        labels = torch.from_numpy(label_encoder.transform(labels))

        images = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Turning on inference context manager
        with torch.inference_mode():
            # Forward pass + calculate loss
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss = torch.add(val_loss, loss.item())

        preds = F.softmax(outputs, dim=1)[:, 1].cpu().detach().numpy()
        new_df = pd.DataFrame(
            {
                "predictions": preds,
                "targets": targets,
                "patients": patients,
            }
        )

        # Append the new DataFrame to the main DataFrame
        df = pd.concat([df, new_df])

    df = df.groupby("patients")[["predictions", "targets"]].mean().reset_index()

    val_score = roc_auc_score(
        df["targets"].tolist(), df["predictions"].tolist(), multi_class="ovo"
    )

    del df

    return float(val_loss / len(val_dataloader)), val_score


def train(
    disable_tqdm: bool,
    early_stopping: EarlyStopping,
    epochs: int,
    label_encoder: sklearn.preprocessing.LabelEncoder,
    learning_rate: float,
    decay: float,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    precomputed: bool,
):
    # Creating empty results dictionary
    results = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    criterion, optimizer, model = create_model(learning_rate, decay, precomputed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    start_time = timer()
    for epoch in range(epochs):
        epoch_start = timer()
        train_loss, train_auc = train_step(
            criterion=criterion,
            device=device,
            disable_tqdm=disable_tqdm,
            label_encoder=label_encoder,
            model=model,
            optimizer=optimizer,
            train_dataloader=train_dataloader,
        )
        val_loss, val_auc = val_step(
            criterion=criterion,
            device=device,
            disable_tqdm=disable_tqdm,
            label_encoder=label_encoder,
            model=model,
            val_dataloader=val_dataloader,
        )

        # Ending the timer and print out the time per epoch
        epoch_end = timer()
        epoch_time = epoch_end - epoch_start

        # Printing out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train : Loss: {train_loss:.4f}, ROC AUC: {train_auc:.4f} | "
            f"validation : Loss {val_loss:.4f}, ROC AUC: {val_auc:.4f}"
            f"\nElapsed time for epoch {epoch+1}: {epoch_time // 60} minutes {epoch_time % 60:.4f} seconds"
        )

        # Updating results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_auc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_auc)

        # Comparing the loss per epoch
        if early_stopping:
            early_stopping(val_loss, model)

            if early_stopping.early_stop:
                print("Early stopping")
                break

        # Ending the timer and print out how long it took
    end_time = timer()
    elapsed_time = end_time - start_time

    # Printing the total training time
    print(
        f"[INFO] Total training time: {elapsed_time // 60} minutes, {elapsed_time % 60:.4f} seconds'"
    )

    # Return the filled results at the end of the epochs
    return results


def saveLearnCurves(tLoss, vLoss, tAcc, vAcc, save_path):
    """save the learning curves figure

    Args:
        tLoss (List): training loss list of the model
        vLoss (List): validation loss list of the model
        tAcc (List): training accuracy list of the model
        vAcc (List): validation accuracy list of the model
        save_path (str): path to save the image
    """
    maxLoss = max(max(tLoss), max(vLoss))

    with plt.style.context("ggplot"):
        plt.figure()
        axl = plt.subplot(2, 1, 1)
        axl.plot(tLoss, "g", vLoss, "r")
        # Mark the minimum point
        axl.scatter(vLoss.index(min(vLoss)), min(vLoss), color="red", marker="o", s=50)
        axl.set_ylim([0, maxLoss + 0.1])
        axl.legend(("train", "val"))
        axl.set_title(f"Loss (min Loss: {min(vLoss):.4f})")
        axv = plt.subplot(2, 1, 2)
        axv.plot(tAcc, "g", vAcc, "r")
        axv.scatter(vAcc.index(max(vAcc)), max(vAcc), color="red", marker="o", s=50)
        axv.set_title(f"AUROC (max val: {max(vAcc):.4f})", y=-0.01)
        axv.legend(("train", "val"))
        plt.savefig(f"{save_path}_res.png")
        plt.close()
