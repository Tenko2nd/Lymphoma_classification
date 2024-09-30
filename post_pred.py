import argparse
import pandas as pd
from torchvision import transforms
from tqdm import tqdm
import torch
from torchvision import models
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
import pickle
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader
import glob

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
        help="The path of the models",
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

    for inputs, labels, tabular in tqdm(loader, disable=True):
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

def results(foldersDf):
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
            df.groupby("patients")[["predictions", "targets"]].mean().reset_index()
        )

    # Assembling
    df_concat = pd.concat(foldersDf)
    patient_stats = (
        df_concat.groupby("patients")[["predictions", "targets"]].median().reset_index()
    )

    targets = patient_stats["targets"].tolist()
    predictions = patient_stats["predictions"].tolist()
    print("patient len : ", len(predictions))
    roc_auc = roc_auc_score(targets, predictions)

    return roc_auc

if __name__ == "__main__":
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'  # helps prevent out-of-memory errors and makes more efficient use of the available GPU memory
    parser = parser_init()

    dataset_path = parser.parse_args().dataset_path
    df = pd.read_csv(dataset_path)

    
    le_path = f"{os.getcwd()}/Lymphoma_labelEncoder.pkl"
    with open(le_path, "rb") as f:
        le = preprocessing.LabelEncoder()
        le.classes_ = pickle.load(f)   

    modelRootPath = parser.parse_args().model_root_path

    modelPaths = glob.glob(f"{modelRootPath}/**/*.pt", recursive=True)

    test_df  = df[df['folder'] == 'test']
    print(f"test : {test_df.groupby(["categorie"])["categorie"].count().to_dict()}")

    resultat = {}
    for subtype, group in test_df.groupby('type'):
        if group["categorie"].nunique() == 1:
            continue
        print(subtype, "taille :" ,len(group))
        dataset = L.LymphomaDataset(
                pd_file=group.reset_index(drop=True),
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
        
        foldersDf = []
        for folder, path in enumerate(modelPaths):
            # print("test val : ", folder)
            resDF = test(loader, path, le)
            foldersDf.append(resDF)

        resultat[subtype] = results(foldersDf)
        print(resultat[subtype])

    print(f"For the model {modelRootPath} : ")
    for k , v in resultat.items():
        print(f"subtype : {k}, AUROC : {v}")
