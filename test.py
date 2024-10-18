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
from transformers import AutoModel

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
    model = AutoModel.from_pretrained("owkin/phikon-v2")
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
            preds = F.softmax(outputs.last_hidden_state[:, 0, :], dim=1)[:, 1].cpu()

                
        # Create a new DataFrame for the current iteration
        new_df = pd.DataFrame({'predictions': preds.detach().numpy(), 'targets': labels.squeeze(-1).cpu().detach().numpy(), 'patients': patients, "references" : references})

        # Append the new DataFrame to the main DataFrame
        df = pd.concat([df, new_df])

    return  df

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
    sortedPatientResults(patient_stats, save_path)
    return saveResult(patient_stats, save_path)

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
        plt.savefig(f"{save_path}auroc.png")
    return matrix_info

def confusion_matrix(matrix_info, save_path, le):
    confusion_matrix = metrics.confusion_matrix(matrix_info["targets"], matrix_info["predictions"])
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = le.inverse_transform([0, 1]))
    cm_display.plot()
    plt.suptitle(f"p_value: {matrix_info["p_value"]:.5f}, threshold: {matrix_info["threshold"]:.4f}")
    plt.savefig(f"{save_path}confusion_matrix.png")

def p_value(preds, targets): 
    x = [p for p, t in zip(preds, targets) if t == 0]     
    y = [p for p, t in zip(preds, targets) if t == 1]
    _, p = mannwhitneyu(y, x)    
    return p

def sortedPatientResults(patient_stats: pd.DataFrame, save_path):
    patient_stats["difference"] = np.abs(patient_stats["targets"] - patient_stats["predictions"])
    df_sorted = patient_stats.sort_values(by='difference')
    df_sorted.to_csv(f'{save_path}sorted_data.csv', index=False)


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

    save = f"{modelRootPath}/"
    matrix_info = assemble_n_aggregate(foldersDf, save)
    matrix_info['p_value'] = p_value(matrix_info['targets'], matrix_info['predictions'])
    confusion_matrix(matrix_info, save, le)


