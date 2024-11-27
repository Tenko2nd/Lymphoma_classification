import argparse
import pandas as pd
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import roc_curve, roc_auc_score
import pickle
import torch.nn.functional as F
import os
from scipy.stats import mannwhitneyu
import shutil

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import lymphoma_dataset_class as L
import constant as C
from MyModel_Class import MyModel


def parser_init():
    """initialize the parser and returns it"""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "-path",
        "--root_path",
        help="The CSV with the patients stats",
        required=True,
    )

    parser.add_argument(
        "-d",
        "--dataset_path",
        help="The path of the dataset used for training",
        required=True,
    )

    return parser


# def saveResult(patient_stats: pd.DataFrame, patients_by_fold, save_path, le):

#     with plt.style.context("ggplot"):

#         # Compute the ROC curve and AUROC for each class
#         for i in range(C.NUM_CLASSES):
#             plt.figure()
#             plt.plot([0, 1], [0, 1], color="#ccca68", linestyle="--", label="No Skill")
#             for fold, patients in patients_by_fold.items():
#                 print(fold, i)
#                 # Filter the score data to keep only the patients in the current fold
#                 fold_score_df = patient_stats[patient_stats["patient"].isin(patients)]
#                 target = fold_score_df["target"].to_numpy()
#                 probas = fold_score_df[f"score_{i}"].to_numpy()

#                 fpr, tpr, _ = roc_curve(target == i, probas)
#                 roc_auc = roc_auc_score(target == i, probas)
#                 plt.ylabel("True Positive Rate")
#                 plt.xlabel("False Positive Rate")
#                 p = p_value(probas, target.astype(int) == i)
#                 plt.plot(fpr, tpr, label=f"{fold}, AUC={roc_auc*100:.1f}%, P = {p:.3f}")
#                 print(
#                     f"Class {le.inverse_transform([i])}: AUROC = {roc_auc:.3f}, p_value = {p:.3f}"
#                 )
#                 plt.legend()
#             plt.title(f"Class : {le.inverse_transform([i])[0]}")
#             plt.savefig(
#                 f"{save_path}FinalModel_AUROC_distinct_{le.inverse_transform([i])[0]}.png"
#             )


def saveResult(patient_stats: pd.DataFrame, patients_by_fold, save_path, le):

    with plt.style.context("ggplot"):

        # Compute the ROC curve and AUROC for each class
        for i in range(C.NUM_CLASSES):
            plt.figure()
            plt.plot([0, 1], [0, 1], color="#ccca68", linestyle="--", label="No Skill")
            target = patient_stats["target"].to_numpy()
            probas = patient_stats[f"score_{i}"].to_numpy()

            fpr, tpr, _ = roc_curve(target == i, probas)
            roc_auc = roc_auc_score(target == i, probas)
            plt.ylabel("True Positive Rate")
            plt.xlabel("False Positive Rate")
            p = p_value(probas, target.astype(int) == i)
            plt.plot(fpr, tpr, label=f"AUC={roc_auc*100:.1f}%, P = {p:.2e}")
            print(
                f"Class {le.inverse_transform([i])}: AUROC = {roc_auc:.3f}, p_value = {p:.2e}"
            )
            plt.legend(loc="lower right")
            plt.title(f"Class : {le.inverse_transform([i])[0]}")
            plt.savefig(
                f"{save_path}FinalModel_AUROC_merged_{le.inverse_transform([i])[0]}.png"
            )


def result_by_cell_type(images_stats: pd.DataFrame, save_path, le):
    if not os.path.exists(f"{save_path}/cells_result/"):
        os.makedirs(f"{save_path}/cells_result/")
    # Create an empty DataFrame
    result_df = pd.DataFrame()

    # Get unique cell types from the column
    cell_types = images_stats["cell_type"].unique()
    with plt.style.context("ggplot"):
        # Iterate through the unique cell types
        for cell_type in cell_types:
            plt.figure(figsize=(15, 5))
            cell_df = images_stats[image_stats["cell_type"] == cell_type]
            AUCs = {}
            PVals = {}
            # Compute the ROC curve and AUROC for each class
            for i in range(C.NUM_CLASSES):
                plt.subplot(1, 3, i + 1)
                target = cell_df["target"].to_numpy()
                probas = cell_df[f"score_{i}"].to_numpy()
                try:
                    roc_auc = roc_auc_score(target == i, probas)
                    fpr, tpr, _ = roc_curve(target == i, probas)
                    p = p_value(probas, target.astype(int) == i)
                    plt.plot(
                        [0, 1],
                        [0, 1],
                        color="#ccca68",
                        linestyle="--",
                        label="No Skill",
                    )
                    plt.plot(fpr, tpr, label=f"AUC={roc_auc*100:.1f}%\nP = {p:.4f}")
                    plt.legend(loc="lower right")
                    plt.title(le.inverse_transform([i])[0])
                    AUCs[f"AUC_{le.inverse_transform([i])[0]}"] = roc_auc
                    PVals[f"P_{le.inverse_transform([i])[0]}"] = p
                except ValueError:
                    print("les cellules : ", cell_type, "ne peuvent avoir d'AUC")
                    AUCs = {}
                    PVals = {}
                    break
            if AUCs != {}:
                plt.suptitle(f"Cell type : {cell_type}")
                plt.savefig(f"{save_path}/cells_result/{cell_type}.png")
                additionnal_info = {
                    "cell_type": cell_type,
                    "number_of_images": len(cell_df),
                }
                new_df = pd.DataFrame([{**additionnal_info, **AUCs, **PVals}])
                result_df = pd.concat([result_df, new_df])
            plt.close()
    print(result_df)
    result_df.to_csv(
        f"{save_path}/cells_result/cell_result.csv",
        index=False,
        header=True,
    )


def p_value(preds, targets):
    x = [p for p, t in zip(preds, targets) if t == 0]
    y = [p for p, t in zip(preds, targets) if t == 1]
    _, p = mannwhitneyu(y, x)
    return p


if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        "expandable_segments:True"  # helps prevent out-of-memory errors and makes more efficient use of the available GPU memory
    )
    parser = parser_init()

    dataset_path = parser.parse_args().dataset_path
    df = pd.read_csv(dataset_path)
    patients_by_fold = df.groupby("folder")["patient"].unique().apply(list)

    root = parser.parse_args().root_path

    patient_stats = pd.read_csv(f"{root}/all_patients.csv")
    image_stats = pd.read_csv(f"{root}/all_images.csv")

    le_path = f"{os.getcwd()}/Lymphoma_labelEncoder.pkl"
    with open(le_path, "rb") as f:
        le = preprocessing.LabelEncoder()
        le.classes_ = pickle.load(f)

    # saveResult(
    #     patient_stats,
    #     patients_by_fold,
    #     f"{root}/",
    #     le,
    # )

    result_by_cell_type(
        image_stats,
        f"{root}/",
        le,
    )
