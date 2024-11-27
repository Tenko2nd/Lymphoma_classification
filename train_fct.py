from time import perf_counter as timer

from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from MyModel_Class import MyModel

from IftimDevLib.IDL.pipelines.evaluation.classification import EarlyStopping


def loaders(
    dataset: dict[str, Dataset], batch_size: int, workers: int
) -> dict[str, DataLoader]:
    """Create the loaders train/val/test based on the dict of datasets given as input

    Args:
        dataset (dict[str, Dataset]): A dictionnary containing the three different datasets train/val/test
        batch_size (int): The number of element to be passed at once for each iteration
        workers (int): The number of workers you want you computer to use

    Returns:
        dict[str, DataLoader]: A dictionnary containing the three different loaders train/val/test
    """
    loaders = {
        x: DataLoader(
            dataset[x],
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            drop_last=True,
        )
        for x in ["train", "val", "test"]
    }
    return loaders


def create_model(labels, learning_rate: float, decay: float, precomputed: bool):
    # create model
    model = MyModel(precomputed=precomputed)
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(labels), y=labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=decay)
    return criterion, optimizer, model


def train_step(
    criterion: nn.Module,
    device: torch.device,
    disable_tqdm: bool,
    label_encoder: LabelEncoder,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_dataloader: DataLoader,
):
    # Cleaning remaining data in GPU
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Putting model in train mode
    model.train()
    log_softmax = nn.LogSoftmax(dim=1)
    # Setup train loss and train accuracy values
    tr_loss = 0
    # Create an empty DataFrame
    for inputs, labels, _ in tqdm(
        train_dataloader,
        total=len(train_dataloader),
        disable=disable_tqdm,
    ):
        labels = torch.from_numpy(label_encoder.transform(labels))

        images = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(log_softmax(outputs), labels)
        tr_loss = torch.add(tr_loss, loss.item())
        loss.backward()
        optimizer.step()

    return float(tr_loss / len(train_dataloader))


def val_step(
    criterion: nn.Module,
    device: torch.device,
    disable_tqdm: bool,
    label_encoder: LabelEncoder,
    model: nn.Module,
    val_dataloader: DataLoader,
):
    # Cleaning remaining data in GPU
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Putting model in eval mode
    model.eval()

    val_loss = 0
    log_softmax = nn.LogSoftmax(dim=1)

    for inputs, labels, _ in tqdm(
        val_dataloader,
        disable=disable_tqdm,
    ):
        labels = torch.from_numpy(label_encoder.transform(labels))

        images = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Turning on inference context manager
        with torch.inference_mode():
            # Forward pass + calculate loss
            outputs = model(images)
            loss = criterion(log_softmax(outputs), labels)
            val_loss = torch.add(val_loss, loss.item())

    return float(val_loss / len(val_dataloader))


def train(
    disable_tqdm: bool,
    early_stopping: EarlyStopping,
    epochs: int,
    label_encoder: LabelEncoder,
    learning_rate: float,
    decay: float,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    precomputed: bool,
    targets: list,
):
    # Creating empty results dictionary
    results = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    criterion, optimizer, model = create_model(
        targets, learning_rate, decay, precomputed
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion.to(device)

    start_time = timer()
    for epoch in range(epochs):
        epoch_start = timer()
        train_loss = train_step(
            criterion=criterion,
            device=device,
            disable_tqdm=disable_tqdm,
            label_encoder=label_encoder,
            model=model,
            optimizer=optimizer,
            train_dataloader=train_dataloader,
        )
        val_loss = val_step(
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
            f"train : Loss: {train_loss:.4f}| "
            f"validation : Loss {val_loss:.4f}"
            f"\nElapsed time for epoch {epoch+1}: {epoch_time // 60} minutes {epoch_time % 60:.4f} seconds"
        )

        # Updating results dictionary
        results["train_loss"].append(train_loss)
        results["val_loss"].append(val_loss)

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


def saveLearnCurves(tLoss, vLoss, save_path):
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
        plt.plot(tLoss, "g", vLoss, "r")
        # Mark the minimum point
        plt.scatter(vLoss.index(min(vLoss)), min(vLoss), color="red", marker="o", s=50)
        plt.ylim([0, maxLoss + 0.1])
        plt.legend(("train", "val"))
        plt.title(f"Loss (min Loss: {min(vLoss):.4f})")
        plt.savefig(f"{save_path}_res.png")
        plt.close()
