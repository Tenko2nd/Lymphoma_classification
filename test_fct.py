import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision import models
import numpy as np
from sklearn import metrics
import torch.nn.functional as F
from scipy.stats import mannwhitneyu
from transformers import AutoModel
from MyModel_Class import MyModel

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


def test(loader, model_path, le, precomputed, disable_tqdm):
    """Test the performance of the model on unseen data from the dataset

    Args:
        model_path (str): the path of the model to train
        le (sklearn.preprocessing.LabelEncoder): the label encoder used for the classification of the dataset

    Returns:
        List: The lists of targets and predictions of the model
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = MyModel(precomputed=precomputed)
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

def assemble_n_aggregate(foldersDf, save_path):
    """Assembling by median then agregation by mean
    """
    # Assembling
    df_concat = pd.concat(foldersDf)
    patient_stats = (
        df_concat.groupby("references")
        .agg(
            {
                "predictions": ["median"],
                "targets": "median",
                "patients": "first",
            }
        )
        .reset_index()
    )
    patient_stats.columns = [
        "references",
        "predictions",
        "targets",
        "patients",
    ]

    # Agregation
    patient_stats = patient_stats.drop(columns=["references"])
    patient_stats = (
        patient_stats.groupby("patients")[["predictions", "targets"]].mean().reset_index()
    )
    
    return saveResult(patient_stats, save_path)

def p_value(preds, targets): 
    x = [p for p, t in zip(preds, targets) if t == 0]     
    y = [p for p, t in zip(preds, targets) if t == 1]
    _, p = mannwhitneyu(y, x)    
    return p

def saveResult(patient_stats: pd.DataFrame, save_path):
    targets = patient_stats["targets"].tolist()
    predictions = patient_stats["predictions"].tolist()
    with plt.style.context("ggplot"):
        # ROC
        plt.figure()
        plt.plot([0, 1], [0, 1], color="#ccca68", linestyle="--", label="No Skill")
        fpr, tpr, thresh = roc_curve(targets, predictions)
        roc_auc = roc_auc_score(targets, predictions)
        plt.plot(fpr, tpr, label=f"ass_med_agr_mean (AUC:{roc_auc:.3f})")
        optimal_thresh = sorted(list(zip(np.abs(tpr - fpr), thresh)), key=lambda i: i[0], reverse=True)[0][1]
        roc_predictions = [1 if i >= optimal_thresh else 0 for i in predictions]
        matrix_info = {"auc" : roc_auc, "name" :"ss_med_agr_mean", "threshold": optimal_thresh, "targets": targets, "predictions": roc_predictions}
        plt.ylabel("True Positive Rate")
        plt.xlabel("False Positive Rate")
        plt.legend()
        plt.savefig(f"{save_path}_auroc.png")
    return matrix_info

def confusion_matrix(matrix_info, save_path, le):
    confusion_matrix = metrics.confusion_matrix(matrix_info["targets"], matrix_info["predictions"])
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = le.inverse_transform([0, 1]))
    cm_display.plot()
    plt.suptitle(f"p_value: {matrix_info["p_value"]:.5f}, threshold: {matrix_info["threshold"]:.4f}")
    plt.savefig(f"{save_path}_confusion_matrix.png")
