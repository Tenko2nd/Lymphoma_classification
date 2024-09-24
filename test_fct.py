import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import torch
from tqdm import tqdm
from torchvision import models
import torch.nn.functional as F

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


def test(loader, model_path, le, disable_tqdm):
    """Test the performance of the model on unseen data from the dataset

    Args:
        model_path (str): the path of the model to train
        le (sklearn.preprocessing.LabelEncoder): the label encoder used for the classification of the dataset

    Returns:
        List: The lists of targets and predictions of the model
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = models.efficientnet_b4()
    model.load_state_dict(
        torch.load(model_path, weights_only=True, map_location=torch.device(device))
    )
    model.eval()
    model = model.to(device)

    # Create an empty DataFrame
    df = pd.DataFrame(columns=["predictions", "targets", "patients"])

    for inputs, labels, tabular in tqdm(loader, disable=disable_tqdm):
        patients = tabular["patient"]
        references = tabular["reference"]

        labels = torch.from_numpy(le.transform(labels))

        images = inputs.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.float)

        # Turning on inference context manager
        with torch.inference_mode():
            # Forward pass + calculate loss
            outputs = model(images)
            preds = F.softmax(outputs, dim=1)[:, 1].cpu()

        # Create a new DataFrame for the current iteration
        new_df = pd.DataFrame(
            {
                "predictions": preds.detach().numpy(),
                "targets": labels.squeeze(-1).cpu().detach().numpy(),
                "patients": patients,
                "references": references,
            }
        )

        # Append the new DataFrame to the main DataFrame
        df = pd.concat([df, new_df])

    return df


def agrThenAss(foldersDf, save_path):
    """Agregation then assembling

    Args:
        foldersDf (List): a list of all the pandas dataframes resulting from the test of the model (the list is for the K models created in the Kfold_CV)

    Returns:
        pandas.DataFrame: contains the predictions (based on mean or median threshold) of a patient for if it is LCM or LZM along with their target
    """
    # Agregation
    for i, df in enumerate(foldersDf):
        foldersDf[i] = df.drop(columns=["references"])
        foldersDf[i] = (
            df.groupby("patients")
            .agg({"predictions": ["mean", "median"], "targets": "mean"})
            .reset_index()
        )
        foldersDf[i].columns = ["patients", "pred_mean", "pred_median", "targets"]

    # Assembling
    df_concat = pd.concat(foldersDf)
    patient_stats = (
        df_concat.groupby("patients")
        .agg(
            {
                "pred_mean": ["mean", "median"],
                "pred_median": ["mean", "median"],
                "targets": "mean",
            }
        )
        .reset_index()
    )
    patient_stats.columns = [
        "patients",
        "agr_mean_ass_mean",
        "agr_mean_ass_med",
        "agr_med_ass_mean",
        "agr_med_ass_med",
        "targets",
    ]

    saveResult(patient_stats, f"{save_path}_agr_ass")

    return patient_stats


def assThenArg(foldersDf, save_path):
    """Assembling then agregation

    Args:
        foldersDf (List): a list of all the pandas dataframes resulting from the test of the model (the list is for the K models created in the Kfold_CV)

    Returns:
        pandas.DataFrame: contains the predictions (based on mean or median threshold) of a patient for if it is LCM or LZM along with their target
    """
    # Assembling
    df_concat = pd.concat(foldersDf)
    patient_stats = (
        df_concat.groupby("references")
        .agg(
            {
                "predictions": ["mean", "median"],
                "targets": "median",
                "patients": "first",
            }
        )
        .reset_index()
    )
    patient_stats.columns = [
        "references",
        "pred_mean",
        "pred_median",
        "targets",
        "patients",
    ]

    # Agregation
    patient_stats = patient_stats.drop(columns=["references"])
    patient_stats = (
        patient_stats.groupby("patients")
        .agg(
            {
                "pred_mean": ["mean", "median"],
                "pred_median": ["mean", "median"],
                "targets": "mean",
            }
        )
        .reset_index()
    )
    patient_stats.columns = [
        "patients",
        "ass_mean_agr_mean",
        "ass_mean_agr_med",
        "ass_med_agr_mean",
        "ass_med_agr_med",
        "targets",
    ]

    saveResult(patient_stats, f"{save_path}_ass_agr")

    return patient_stats


def saveResult(patient_stats: pd.DataFrame, save_path):
    targets = patient_stats["targets"].tolist()
    patient_pred = patient_stats.drop(columns=["patients", "targets"])
    with plt.style.context("ggplot"):
        # ROC
        plt.figure()
        plt.plot([0, 1], [0, 1], color="#ccca68", linestyle="--", label="No Skill")
        for col, values in patient_pred.items():
            fpr, tpr, _ = roc_curve(targets, values.tolist())
            roc_auc = roc_auc_score(targets, values.tolist())
            plt.plot(fpr, tpr, label=f"{col} (AUC:{roc_auc:.3f})")
        plt.ylabel("True Positive Rate")
        plt.xlabel("False Positive Rate")
        plt.legend()
        plt.savefig(f"{save_path}_auroc.png")
