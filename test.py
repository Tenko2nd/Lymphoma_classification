import argparse
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

    y_pred = None
    y_targ = None
    with torch.inference_mode():
        for images, target, _ in tqdm(loader):
            images = images.float()

            target = torch.from_numpy(le.transform(target))
            if y_targ is None:
                y_targ = target.squeeze(-1)
            else:
                y_targ = torch.cat((y_targ, target.squeeze(-1)), dim=0)

            images = images.to(device, dtype=torch.float)
            target = target.to(device, dtype=torch.float)

            outputs = model(images)
            preds = F.softmax(outputs, dim=1)[:, 1].cpu()
            
            if y_pred is None:
                y_pred = preds
            else:
                y_pred = torch.cat((y_pred, preds), dim=0)

    return y_targ, y_pred

def saveAUC(y_targ, y_pred, save_path):
    """Save the AUC scores for the model

    Args:
        y_targ (List): list of targets
        y_pred (List): list of predictions of the model
        save_path (str): path to save the images
    """
    with plt.style.context("ggplot"):
        # ROC
        fpr, tpr, _ = roc_curve(y_targ, y_pred)
        roc_auc = roc_auc_score(y_targ, y_pred)
        plt.figure()
        plt.plot(fpr, tpr, label=f"Test dataset (AUC:{roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], linestyle="--", label="No Skill")
        plt.ylabel("True Positive Rate")
        plt.xlabel("False Positive Rate")
        plt.legend()
        plt.savefig(f"{save_path}_auc_roc.png")
        # Precision Recall
        precision, recall, _ = precision_recall_curve(y_targ, y_pred)
        pr_auc = auc(recall, precision)
        plt.figure()
        plt.plot(recall, precision, label=f"Test dataset (AUC:{pr_auc:.3f})")
        plt.plot([1, 0], [0.5, 0.5], linestyle="--", label="No Skill")
        plt.ylabel("Precision")
        plt.xlabel("Recall")
        plt.legend()
        plt.savefig(f"{save_path}_auc_pr.png")


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

    y_targ, y_pred = test(loader, model_path, le)
    saveAUC(y_targ, y_pred, model_path.split('.')[0])
