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
from torch.utils.data import DataLoader
from sklearn import metrics
import numpy as np
import glob
from scipy.stats import mannwhitneyu

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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
    model = MyModel(precomputed=True)
    model.load_state_dict(
        torch.load(model_path, weights_only=True, map_location=torch.device(device))
    )
    model.eval()
    model = model.to(device)

    # Create an empty DataFrame
    df = pd.DataFrame()

    for inputs, labels, tabular in tqdm(loader):
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

def assemble_n_aggregate(foldsDf, save_path, le):
    """Assembling by mean then agregation by median"""
    # Assembling
    df_concat = pd.concat(foldsDf)
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

    saveResult(patient_stats, save_path, le)
    return best_worst(patient_stats, save_path, before_agg)

def best_worst(patient_stats, save_path, before_agg):
    os.makedirs(f"{save_path}best_worst", exist_ok=True)
    patient_stats['diff'] = np.where(patient_stats['target'] == 0, 
                                np.abs(1 - patient_stats['score_0']), 
                                np.where(patient_stats['target'] == 1, 
                                        np.abs(1 - patient_stats['score_1']), 
                                        np.abs(1 - patient_stats['score_2'])))
    before_agg['diff'] = np.where(before_agg['target'] == 0, 
                                np.abs(1 - before_agg['score_0']), 
                                np.where(before_agg['target'] == 1, 
                                        np.abs(1 - before_agg['score_1']), 
                                        np.abs(1 - before_agg['score_2'])))
    image_best = before_agg.sort_values('diff').groupby('target').head(20).reset_index(drop=True)
    image_worst = before_agg.sort_values('diff', ascending=False).groupby('target').head(20).reset_index(drop=True)
    image_best.to_csv(f"{save_path}best_worst/best.csv",index=True,header=True,)
    image_worst.to_csv(f"{save_path}best_worst/worst.csv",index=True,header=True,)
    return patient_stats, before_agg


    

def saveResult(patient_stats: pd.DataFrame, save_path, le):
    targets = patient_stats["target"].to_numpy()
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
    DS_FOLDER = parser.parse_args().model_root_path
    
    modelRootPaths = [os.path.abspath(os.path.join(DS_FOLDER, f)) 
                    for f in os.listdir(DS_FOLDER) 
                    if os.path.isdir(os.path.join(DS_FOLDER, f))]

    # Create empty DataFrames
    all_patient_df = pd.DataFrame()
    all_images_df = pd.DataFrame()

    for modelRootPath in modelRootPaths:

        num = modelRootPath.split('_')[-1]
        test_df  = df[df['folder'] == f'folder_{num}']
        print(f"test : {test_df.groupby(["categorie"])["categorie"].count().to_dict()}")
        


        dataset = L.LymphomaDataset(
                pd_file=test_df.reset_index(drop=True),
                precomputed=True
            )
        
        loader =  DataLoader(
                dataset, batch_size=16, shuffle=False, num_workers=15
            )
        
        le_path = f"{os.getcwd()}/Lymphoma_labelEncoder.pkl"
        with open(le_path, "rb") as f:
            le = preprocessing.LabelEncoder()
            le.classes_ = pickle.load(f)   

        

        modelPaths = glob.glob(f"{modelRootPath}/**/*.pt", recursive=True)
        foldsDf = []
        for fold, path in enumerate(modelPaths):
            print("test val : ", fold)
            resDF = test(loader, path, le)
            foldsDf.append(resDF)

        save = f"{modelRootPath}/"
        patient_stats, before_agg = assemble_n_aggregate(foldsDf, save, le)
        all_patient_df = pd.concat([all_patient_df, patient_stats])
        all_images_df = pd.concat([all_images_df, before_agg])
    
    all_images_df = all_images_df.sort_values('diff').reset_index(drop=True)
    all_images_df.to_csv(f"{DS_FOLDER}all_images.csv",index=False,header=True,)
    all_patient_df = all_patient_df.sort_values('diff').reset_index(drop=True)
    all_patient_df.to_csv(f"{DS_FOLDER}all_patients.csv",index=False,header=True,)

    targets = all_patient_df["target"].to_numpy()
    predictions = all_patient_df["prediction"].to_numpy()

    for i in range(C.NUM_CLASSES):
        probas = all_patient_df[f"score_{i}"].to_numpy()
    
        with plt.style.context("ggplot"):
            plt.figure()
            plt.plot([0, 1], [0, 1], color="#ccca68", linestyle="--", label="No Skill")
            fpr, tpr, tresh = roc_curve(targets == i, probas)
            roc_auc = roc_auc_score(targets == i, probas)
            plt.plot(fpr, tpr, label=f"AUC:{roc_auc:.3f}")
            plt.ylabel("True Positive Rate")
            plt.xlabel("False Positive Rate")
            p = p_value(probas, targets.astype(int) == i)
            plt.suptitle(f"p value : {p}")
            plt.legend()
            plt.savefig(f"{DS_FOLDER}auroc_{le.inverse_transform([i])[0]}.png")

    confusion_matrix = metrics.confusion_matrix(targets, predictions)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = le.inverse_transform(list(range(C.NUM_CLASSES))))
    cm_display.plot()
    plt.suptitle(
        f"balanced accuracy : {metrics.balanced_accuracy_score(targets, predictions):.3f}, accuracy : {metrics.accuracy_score(targets, predictions):.3f}"
    )
    plt.savefig(f"{DS_FOLDER}confusion_matrix.png")
    plt.close()
