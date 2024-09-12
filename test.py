import argparse
from collections import defaultdict
import pandas as pd
from torchvision import transforms
from tqdm import tqdm
import torch
from torchvision import models
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve, auc
import pickle
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader
import numpy as np

import lymphoma_dataset_class as L
import constant as C


def parser_init():
    """initialize the parser and returns it"""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "-path",
        "--model_path",
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
    model = models.efficientnet_b3()
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device(device)))
    model.eval()
    model = model.to(device)

    # Initialize a defaultdict with a list as default value
    patient_preds = defaultdict(list)
    patient_targ = {}
    with torch.inference_mode():
        for images, target, tabular in tqdm(loader):
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


if __name__ == "__main__":
    parser = parser_init()

    dataset_path = parser.parse_args().dataset_path
    df = pd.read_csv(dataset_path)

    test_df  = df[df['folder'] == 'test']
    print(f"test : {test_df.groupby(["categorie"])["categorie"].count().to_dict()}")
    


    dataset = L.LymphomaDataset(
            pd_file=test_df.reset_index(drop=True),
            transform=transforms.Compose(
                [
                    L.Rescale(300),
                    L.Crop(300),
                    L.ToTensor(),
                    transforms.Normalize(C.lymphomas_mean, C.lymphomas_std),
                ]
            ),
        )
    
    loader =  DataLoader(
            dataset, batch_size=16, shuffle=False, num_workers=0
        )
    
    le_path = f"{os.getcwd()}/Lymphoma_labelEncoder.pkl"
    with open(le_path, "rb") as f:
        le = preprocessing.LabelEncoder()
        le.classes_ = pickle.load(f)    
    model_path = parser.parse_args().model_path

    patient_merged_dic = test(loader, model_path, le)
    saveAUC(patient_merged_dic, model_path.split('.')[0])
