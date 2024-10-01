import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import torch
from tqdm import tqdm
from torchvision import models
import numpy as np
from sklearn import metrics
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
    
    best = saveResult(patient_stats, f"{save_path}_agr_ass")
    return best


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

    best = saveResult(patient_stats, f"{save_path}_ass_agr")
    return best

def saveResult(patient_stats: pd.DataFrame, save_path):
    best = {"score" : 0}
    targets = patient_stats["targets"].tolist()
    patient_pred = patient_stats.drop(columns=["patients", "targets"])
    with plt.style.context("ggplot"):
        # ROC
        plt.figure()
        plt.plot([0, 1], [0, 1], color="#ccca68", linestyle="--", label="No Skill")
        for col, values in patient_pred.items():
            fpr, tpr, thresh = roc_curve(targets, values.tolist())
            roc_auc = roc_auc_score(targets, values.tolist())
            plt.plot(fpr, tpr, label=f"{col} (AUC:{roc_auc:.3f})")
            if best["score"] < roc_auc:
                optimal_thresh = sorted(list(zip(np.abs(tpr - fpr), thresh)), key=lambda i: i[0], reverse=True)[0][1]
                roc_predictions = [1 if i >= optimal_thresh else 0 for i in values.tolist()]
                best = {"score" : roc_auc, "name" :col, "threshold": optimal_thresh, "targets": targets, "predictions": roc_predictions}
        plt.ylabel("True Positive Rate")
        plt.xlabel("False Positive Rate")
        plt.legend()
        plt.savefig(f"{save_path}_auroc.png")
    return best

def confusion_matrix(best_method, save_path, le):
    confusion_matrix = metrics.confusion_matrix(best_method["targets"], best_method["predictions"])
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = le.inverse_transform([0, 1]))
    cm_display.plot()
    plt.suptitle(f"method : {best_method["name"]}, threshold: {best_method["threshold"]}")
    plt.savefig(f"{save_path}_confusion_matrix.png")
