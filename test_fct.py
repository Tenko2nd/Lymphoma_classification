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
import os

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
        cell_types = tabular["type"]

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
                "target": labels.squeeze(-1).cpu().detach().numpy(),
                "patient": patients,
                "reference": references,
                "cell_type": cell_types,
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
        df_concat.groupby("reference")
        .agg(
            {
                **{
                    col: "mean" for col in df_concat.columns if col.startswith("score_")
                },
                "target": "median",
                "patient": "first",
                "cell_type": "first",
            }
        )
        .reset_index()
    )

    before_agg = patient_stats

    # Agregation
    patient_stats = patient_stats.drop(columns=["reference"])
    patient_stats = (
        patient_stats.groupby("patient")[
            patient_stats.columns[
                patient_stats.columns.str.startswith("score_")
            ].tolist()
            + ["target"]
        ]
        .median()
        .reset_index()
    )
    patient_stats["prediction"] = np.argmax(
        patient_stats[["score_0", "score_1", "score_2"]].values, axis=1
    )

    output = saveResult(patient_stats, save_path, le)
    patient_stats, before_agg = best_worst(patient_stats, save_path, before_agg)
    return patient_stats, before_agg, output


def p_value(preds, targets):
    x = [p for p, t in zip(preds, targets) if t == 0]
    y = [p for p, t in zip(preds, targets) if t == 1]
    _, p = mannwhitneyu(y, x)
    return p


def saveResult(patient_stats: pd.DataFrame, save_path, le):
    target = patient_stats["target"].to_numpy()
    prediction = patient_stats["prediction"].to_numpy()
    output = []

    for i in range(C.NUM_CLASSES):
        probas = patient_stats[f"score_{i}"].to_numpy()

        with plt.style.context("ggplot"):
            plt.figure()
            plt.plot([0, 1], [0, 1], color="#ccca68", linestyle="--", label="No Skill")
            fpr, tpr, _ = roc_curve(target == i, probas)
            roc_auc = roc_auc_score(target == i, probas)
            plt.plot(fpr, tpr, label=f"AUC:{roc_auc:.3f}")
            plt.ylabel("True Positive Rate")
            plt.xlabel("False Positive Rate")
            p = p_value(probas, target.astype(int))
            output.append(
                f"Class {le.inverse_transform([i])}: AUROC = {roc_auc:.3f}, p_value = {p:.4f}"
            )
            print(output[-1])
            plt.suptitle(f"p value : {p}")
            plt.legend()
            plt.savefig(f"{save_path}auroc_{le.inverse_transform([i])[0]}.png")

    confusion_matrix = metrics.confusion_matrix(target, prediction)
    cm_display = metrics.ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix,
        display_labels=le.inverse_transform(list(range(C.NUM_CLASSES))),
    )
    cm_display.plot()
    plt.suptitle(
        f"balanced accuracy : {metrics.balanced_accuracy_score(target, prediction):.3f}, accuracy : {metrics.accuracy_score(target, prediction):.3f}"
    )
    plt.savefig(f"{save_path}confusion_matrix.png")
    plt.close()

    return output


def best_worst(patient_stats, save_path, before_agg):
    os.makedirs(f"{save_path}best_worst", exist_ok=True)
    patient_stats["diff"] = np.where(
        patient_stats["target"] == 0,
        np.abs(1 - patient_stats["score_0"]),
        np.where(
            patient_stats["target"] == 1,
            np.abs(1 - patient_stats["score_1"]),
            np.abs(1 - patient_stats["score_2"]),
        ),
    )
    before_agg["diff"] = np.where(
        before_agg["target"] == 0,
        np.abs(1 - before_agg["score_0"]),
        np.where(
            before_agg["target"] == 1,
            np.abs(1 - before_agg["score_1"]),
            np.abs(1 - before_agg["score_2"]),
        ),
    )
    return patient_stats, before_agg
