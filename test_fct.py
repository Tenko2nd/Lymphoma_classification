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
import constant as C

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
    df = pd.DataFrame()

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
            score_df = pd.DataFrame(
                [
                    F.softmax(outputs, dim=1)[:, i].cpu().detach().numpy()
                    for i in range(C.NUM_CLASSES)
                ]
            ).T.add_prefix("score_")
        # Create a new DataFrame for the current iteration
        new_df = pd.DataFrame(
            {
                "targets": labels.squeeze(-1).cpu().detach().numpy(),
                "patients": patients,
                "references": references,
            }
        )
        new_df = pd.concat([new_df, score_df], axis=1)
        # Append the new DataFrame to the main DataFrame
        df = pd.concat([df, new_df])

    return df


def assemble_n_aggregate(foldersDf, save_path, le):
    """Assembling by mean then agregation by median"""
    # Assembling
    df_concat = pd.concat(foldersDf)
    patient_stats = (
        df_concat.groupby("references")
        .agg(
            {
                **{
                    col: "mean" for col in df_concat.columns if col.startswith("score_")
                },
                "targets": "median",
                "patients": "first",
            }
        )
        .reset_index()
    )

    # Agregation
    patient_stats = patient_stats.drop(columns=["references"])
    patient_stats = (
        patient_stats.groupby("patients")[
            patient_stats.columns[
                patient_stats.columns.str.startswith("score_")
            ].tolist()
            + ["targets"]
        ]
        .median()
        .reset_index()
    )
    patient_stats["prediction"] = np.argmax(
        patient_stats[["score_0", "score_1", "score_2"]].values, axis=1
    )

    return saveResult(patient_stats, save_path, le)


def p_value(preds, targets):
    x = [p for p, t in zip(preds, targets) if t == 0]
    y = [p for p, t in zip(preds, targets) if t == 1]
    _, p = mannwhitneyu(y, x)
    return p


def saveResult(patient_stats: pd.DataFrame, save_path, le):
    print(patient_stats)
    targets = patient_stats["targets"].to_numpy()
    prediction = patient_stats["prediction"].to_numpy()
    output = []

    # Compute the ROC curve and AUROC for each class
    for i in range(C.NUM_CLASSES):
        predictions = patient_stats[f"score_{i}"].to_numpy()
        auroc = roc_auc_score(targets == i, predictions)
        output.append(f"Class {le.inverse_transform([i])}: AUROC = {auroc:.3f}")
        print(output[-1])

    cm = metrics.confusion_matrix(targets, prediction)
    cm_display = metrics.ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=le.inverse_transform(list(range(C.NUM_CLASSES))),
    )
    cm_display.plot()
    plt.suptitle(
        f"balanced accuracy : {metrics.balanced_accuracy_score(targets, prediction):.3f}, accuracy : {metrics.accuracy_score(targets, prediction):.3f}"
    )
    plt.savefig(f"{save_path}_confusion_matrix.png")

    return output
