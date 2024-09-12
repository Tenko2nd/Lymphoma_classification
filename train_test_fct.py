from time import perf_counter as timer
from collections import defaultdict

from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import roc_curve, roc_auc_score
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


from IftimDevLib.IDL.pipelines.evaluation.classification import EarlyStopping


def loaders(dataset: torch.utils.data.Dataset, batch_size: int = 64, workers: int = 4):
    # Load train and validation dataset
    loaders = {
        x: DataLoader(
            dataset[x], batch_size=batch_size, shuffle=True, num_workers=workers
        )
        for x in ["train", "val"] + (["test"] if "test" in dataset.keys() else [])
    }
    return loaders


def create_model(learning_rate: float = 0.0001):
    # create model
    model = models.efficientnet_b3(
        weights=models.EfficientNet_B3_Weights.DEFAULT
    )  # change weights
    # Freeze all layers except the final classification layer
    for name, param in model.named_parameters():
        param.requires_grad = "fc" in name  # try all layers
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
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
    tr_preds = None
    tr_labels = None

    for inputs, labels, _ in tqdm(
        train_dataloader,
        total=len(train_dataloader),
        disable=disable_tqdm,
    ):
        labels = torch.from_numpy(label_encoder.transform(labels))
        if tr_labels is None:
            tr_labels = labels.squeeze(-1)
        else:
            tr_labels = torch.cat((tr_labels, labels.squeeze(-1)), dim=0)

        images = inputs.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.float)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels.squeeze(-1).to(device, dtype=torch.long))
        preds = F.softmax(outputs, dim=1)[:, 1].cpu()
        tr_loss += loss.item()

        loss.backward()
        optimizer.step()

        if tr_preds is None:
            tr_preds = preds
        else:
            tr_preds = torch.cat((tr_preds, preds), dim=0)

    tr_score = roc_auc_score(
        tr_labels.detach().numpy(), tr_preds.detach().numpy(), multi_class="ovo"
    )

    return tr_loss / len(train_dataloader), tr_score


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
    val_preds = None
    val_labels = None

    for inputs, labels, _ in tqdm(
        val_dataloader,
        total=len(val_dataloader),
        disable=disable_tqdm,
    ):
        labels = torch.from_numpy(label_encoder.transform(labels))
        if val_labels is None:
            val_labels = labels.squeeze(-1)
        else:
            val_labels = torch.cat((val_labels, labels.squeeze(-1)), dim=0)

        images = inputs.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.float)

        # Turning on inference context manager
        with torch.inference_mode():
            # Forward pass + calculate loss
            outputs = model(images)
            loss = criterion(outputs, labels.squeeze(-1).to(device, dtype=torch.long))
            preds = F.softmax(outputs, dim=1)[:, 1].cpu()
            val_loss += loss.item()

            if val_preds is None:
                val_preds = preds
            else:
                val_preds = torch.cat((val_preds, preds), dim=0)

        val_score = roc_auc_score(
            val_labels.detach().numpy(), val_preds.detach().numpy(), multi_class="ovo"
        )

    return val_loss / len(val_dataloader), val_score


def train(
    disable_tqdm: bool,
    early_stopping: EarlyStopping,
    epochs: int,
    label_encoder: sklearn.preprocessing.LabelEncoder,
    learning_rate: float,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
):
    # Creating empty results dictionary
    results = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    criterion, optimizer, model = create_model(learning_rate)
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
    maxLoss = max(tLoss)
    minLoss = min(tLoss)
    zoom = (np.mean(tLoss) - minLoss) * 2 + minLoss

    # define the custom scale function
    def custom_scale(x):
        return np.where(x >= zoom, x, zoom + (x - zoom) / 0.1)

    # define the inverse custom scale function
    def custom_scale_inv(x):
        return np.where(x >= zoom, x, zoom + (x - zoom) * 0.1)

    with plt.style.context("ggplot"):
        axl = plt.subplot(2, 1, 1)
        axl.plot(tLoss, "g", vLoss, "r")
        # apply the custom scale to the axis
        axl.set_yscale("function", functions=(custom_scale, custom_scale_inv))
        # set the y-axis limits to ensure the custom scale is applied correctly
        axl.set_ylim([minLoss - minLoss / 10, maxLoss + maxLoss / 10])
        axl.legend(("train", "val"))
        axl.set_title(f"Loss (min val: {min(vLoss):.4f})")
        axv = plt.subplot(2, 1, 2)
        axv.plot(tAcc, "g", vAcc, "r")
        axv.set_title(f"AUROC (max val: {max(vAcc):.4f})", y=-0.01)
        axv.legend(("train", "val"))
        plt.savefig(f"{save_path}_res.png")


def test(loader, model_path, le, disable_tqdm):
    """Test the performance of the model on unseen data from the dataset

    Args:
        testDS (torch.utils.data.Dataset): the test dataset containing unseen data
        model_path (str): the path of the model to train
        le (sklearn.preprocessing.LabelEncoder): the label encoder used for the classification of the dataset

    Returns:
        List: The lists of targets and predictions of the model
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = models.efficientnet_b3()
    model.load_state_dict(
        torch.load(model_path, weights_only=True, map_location=torch.device(device))
    )
    model.eval()
    model = model.to(device)

    # Initialize a defaultdict with a list as default value
    patient_preds = defaultdict(list)
    patient_targ = {}
    with torch.inference_mode():
        for images, target, tabular in tqdm(loader, disable=disable_tqdm):
            patients = tabular["patient"]
            images = images.float()

            target = le.transform(target)

            images = images.to(device, dtype=torch.float)

            outputs = model(images)
            preds = F.softmax(outputs, dim=1)[:, 1].cpu()
            for patient, targ in zip(patients, target):
                if patient not in patient_targ:
                    patient_targ[patient] = targ

            # Iterate over the patients and values
            for patient, pred in zip(patients, preds):
                # Append the value to the list of the corresponding patient
                patient_preds[patient].append(pred)

        patient_preds_med = {
            patient: np.median(values) for patient, values in patient_preds.items()
        }
        patient_preds_moy = {
            patient: np.mean(values) for patient, values in patient_preds.items()
        }

        patient_merged_dic = {}
        for key in set(list(patient_preds_med.keys())):
            patient_merged_dic[key] = {
                "med": patient_preds_med.get(key, 0),
                "moy": patient_preds_moy.get(key, 0),
                "targ": patient_targ.get(key, 0),
            }

    return patient_merged_dic


def saveAUC(patient_merged_dic, save_path):
    """Save the AUC scores for the model

    Args:
        y_targ (List): list of targets
        y_pred (List): list of predictions of the model
        save_path (str): path to save the images
    """
    y = {"med": [], "moy": [], "targ": []}
    for _, value in patient_merged_dic.items():
        y["med"].append(value["med"])
        y["moy"].append(value["moy"])
        y["targ"].append(value["targ"])

    with plt.style.context("ggplot"):
        for thresh in ["med", "moy"]:
            # ROC
            fpr, tpr, _ = roc_curve(y["targ"], y[thresh])
            roc_auc = roc_auc_score(y["targ"], y[thresh])
            plt.figure()
            plt.plot(fpr, tpr, label=f"Test dataset (AUC:{roc_auc:.3f})")
            plt.plot([0, 1], [0, 1], linestyle="--", label="No Skill")
            plt.ylabel("True Positive Rate")
            plt.xlabel("False Positive Rate")
            plt.legend()
            plt.savefig(f"{save_path}_{thresh}_auc_roc.png")
            # Precision Recall
            precision, recall, _ = precision_recall_curve(y["targ"], y[thresh])
            pr_auc = auc(recall, precision)
            plt.figure()
            plt.plot(recall, precision, label=f"Test dataset (AUC:{pr_auc:.3f})")
            plt.plot([1, 0], [0.5, 0.5], linestyle="--", label="No Skill")
            plt.ylabel("Precision")
            plt.xlabel("Recall")
            plt.legend()
            plt.savefig(f"{save_path}_{thresh}_auc_pr.png")
