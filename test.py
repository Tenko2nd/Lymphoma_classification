import argparse
import pandas as pd
from torchvision import transforms
from tqdm import tqdm
import torch
from torchvision import models
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import roc_curve, roc_auc_score
import pickle
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader
from sklearn import metrics
import numpy as np
import glob
from scipy.stats import mannwhitneyu

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import lymphoma_dataset_class as L
import constant as C


def parser_init():
    """initialize the parser and returns it"""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "-path",
        "--model_root_path",
        help="The path of the model",
        required=True,
    )

    parser.add_argument(
        "-d",
        "--dataset_path",
        help="The path of the dataset used for training",
        required=True,
    )

    return parser

def test(loader, model_path, le):
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
    model = models.efficientnet_b4()
    model.load_state_dict(
        torch.load(model_path, weights_only=True, map_location=torch.device(device))
    )
    model.eval()
    model = model.to(device)

    # Create an empty DataFrame
    df = pd.DataFrame(columns=['predictions', 'targets', 'patients', 'references'])

    for inputs, labels, tabular in tqdm(
        loader
    ):
        patients = tabular["patient"]
        references = tabular['reference']

        labels = torch.from_numpy(le.transform(labels))

        images = inputs.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.float)

        # Turning on inference context manager
        with torch.inference_mode():
            # Forward pass + calculate loss
            outputs = model(images)
            preds = F.softmax(outputs, dim=1)[:, 1].cpu()

                
        # Create a new DataFrame for the current iteration
        new_df = pd.DataFrame({'predictions': preds.detach().numpy(), 'targets': labels.squeeze(-1).cpu().detach().numpy(), 'patients': patients, "references" : references})

        # Append the new DataFrame to the main DataFrame
        df = pd.concat([df, new_df])

    return  df

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

    best = saveResult(patient_stats, f"{save_path}")
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

    best = saveResult(patient_stats, f"{save_path}")

    return best

def saveResult(patient_stats : pd.DataFrame, save_path):
    best = {"score" : 0}
    targets = patient_stats['targets'].tolist()
    patient_pred = patient_stats.drop(columns=['patients', 'targets'])
    with plt.style.context("ggplot"):
        #ROC
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
        # plt.savefig(f"{save_path}_auroc.png")
    return best

def confusion_matrix(best_method, save_path, le):
    confusion_matrix = metrics.confusion_matrix(best_method["targets"], best_method["predictions"])
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = le.inverse_transform([0, 1]))
    cm_display.plot()
    plt.suptitle(f"p_value: {best_method["p_value"]:.5f}, threshold: {best_method["threshold"]:.4f}")
    plt.savefig(save_path)

def p_value(preds, targets): 
    x = [p for p, t in zip(preds, targets) if t == 0]     
    y = [p for p, t in zip(preds, targets) if t == 1]
    _, p = mannwhitneyu(y, x)    
    return p

if __name__ == "__main__":
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'  # helps prevent out-of-memory errors and makes more efficient use of the available GPU memory
    parser = parser_init()

    dataset_path = parser.parse_args().dataset_path
    df = pd.read_csv(dataset_path)
    
    modelRootPath = parser.parse_args().model_root_path

    num = modelRootPath.split('_')[-1]
    test_df  = df[df['folder'] == f'folder_{num}']
    print(f"test : {test_df.groupby(["categorie"])["categorie"].count().to_dict()}")
    


    dataset = L.LymphomaDataset(
            pd_file=test_df.reset_index(drop=True),
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(size=C.IMG_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.CenterCrop(C.IMG_SIZE),
                ]
            ),
        )
    
    loader =  DataLoader(
            dataset, batch_size=16, shuffle=False, num_workers=15
        )
    
    le_path = f"{os.getcwd()}/Lymphoma_labelEncoder.pkl"
    with open(le_path, "rb") as f:
        le = preprocessing.LabelEncoder()
        le.classes_ = pickle.load(f)   

    

    modelPaths = glob.glob(f"{modelRootPath}/**/*.pt", recursive=True)

    foldersDf = []
    for folder, path in enumerate(modelPaths):
        print("test val : ", folder)
        resDF = test(loader, path, le)
        foldersDf.append(resDF)

    foldArgAss = foldersDf
    foldAssArg = foldersDf

    best_method1 = assThenArg(foldAssArg, f"{modelRootPath}/ass_agr")
    best_method2 = agrThenAss(foldArgAss, f"{modelRootPath}/agr_ass")
    if best_method1["score"] < best_method2["score"]:
        best_method = best_method1
    else:
        best_method = best_method2

    best_method['p_value'] = p_value(best_method['targets'], best_method['predictions'])
    confusion_matrix(best_method, f"{modelRootPath}/confusion_matrix.png", le)

