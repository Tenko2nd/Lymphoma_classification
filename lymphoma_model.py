"""\033[38;5;224m    This python script is used to train a model based on a dataset required as an input.
    It can take on multiple parameters : 
        - batch_size (default 32)
        - workers (default 4)
        - learning_rate (default 0.001)
        - early_stop (default 5)
        - name (default '')
    It will save the output model on the Model folder as 'mod_{name}_{date}.pth' and the learning curve of the model.
    The model as an early stopping and only take the most efficient model in loss.\033[38;5;213m
"""

from datetime import datetime
import argparse
import pickle
import os
import warnings

from sklearn import preprocessing
from torchvision import transforms
import pandas as pd

import lymphoma_dataset_class as L
import constant as C
import train_test_fct as tt


from IftimDevLib.IDL.pipelines.evaluation.classification import EarlyStopping

warnings.simplefilter("ignore", FutureWarning)


def parser_init():
    """initialize the parser and returns it"""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "-d",
        "--dataset_path",
        help="The path of the dataset used for training",
        required=True,
    )

    parser.add_argument(
        "-bs",
        "--batch_size",
        default=32,
        type=int,
        help="The batch size for the model",
        required=False,
    )

    parser.add_argument(
        "-w",
        "--workers",
        default=4,
        type=int,
        help="The number of workers for the model",
        required=False,
    )

    parser.add_argument(
        "-lr",
        "--learning_rate",
        default=0.001,
        type=float,
        help="The learning rate for the model",
        required=False,
    )

    parser.add_argument(
        "-es",
        "--early_stop",
        default=5,
        type=int,
        help="The patience value of early stopping for the model",
        required=False,
    )

    parser.add_argument(
        "--pb_disable",
        dest="disable_tqdm",
        action="store_true",
        help="If you want to disable the progress bar (default : enable)",
    )

    parser.set_defaults(disable=False)

    parser.add_argument(
        "-name",
        "--name",
        default="anon",
        help="The name of the categories the model is training on\033[0m",
        required=False,
    )

    return parser


if __name__ == "__main__":
    p = parser_init()
    
    dataset_path = p.parse_args().dataset_path
    df = pd.read_csv(dataset_path)

    dataset_df = {}
    # print repartition LCM / LZM for each folder
    train_val_rows = df[df['folder'].str.startswith('train')]
    dataset_df["train"] = train_val_rows[train_val_rows['folder'] != 'train_1']
    dataset_df["val"] = train_val_rows[train_val_rows['folder'] == 'train_1']
    print(f"train : {dataset_df["train"].groupby(["categorie"])["categorie"].count().to_dict()}")
    print(f"validation : {dataset_df["val"].groupby(["categorie"])["categorie"].count().to_dict()}")
    if "test" in df["folder"].values:
        dataset_df["test"] = df[df['folder'] == 'test']
        print(f"test : {dataset_df["test"].groupby(["categorie"])["categorie"].count().to_dict()}")
    


    dataset = {
        x: L.LymphomaDataset(
            pd_file=dataset_df[x].reset_index(drop=True),
            transform=transforms.Compose(
                [
                    L.Rescale(300),
                    L.Crop(300),
                    L.ToTensor(),
                    transforms.Normalize(C.lymphomas_mean, C.lymphomas_std),
                ]
            ),
        )
        for x in ["train", "val"] + (["test"] if "test" in dataset_df.keys() else [])
    }
    lenDataset = {x: len(dataset[x]) for x in dataset.keys()}
    
    le_path = f"{os.getcwd()}/Lymphoma_labelEncoder.pkl"
    with open(le_path, "rb") as f:
        le = preprocessing.LabelEncoder()
        le.classes_ = pickle.load(f)

    date = datetime.now().strftime("%m%d-%H%M")
    name = p.parse_args().name
    save_model_path = f"{os.getcwd()}/Model/mod_{name}_{date}/mod_{name}_{date}.pt"
    
    # create folder for model, result and AUC to save in
    os.mkdir(f"{os.getcwd()}/Model/mod_{name}_{date}")

    early_stopping = EarlyStopping(
        patience=p.parse_args().early_stop,
        verbose=True,
        path=save_model_path,
        use_kfold=False,
    )


    loaders = tt.loaders(
        dataset=dataset, 
        batch_size=p.parse_args().batch_size, 
        workers=p.parse_args().workers)
    
    results = tt.train(
        train_dataloader=loaders["train"],
        val_dataloader=loaders["val"],
        learning_rate=p.parse_args().learning_rate,
        epochs=200,
        label_encoder=le,
        early_stopping=early_stopping,
        disable_tqdm = p.parse_args().disable_tqdm)
    
    tt.saveLearnCurves(
        tLoss=results["train_loss"], 
        vLoss=results["val_loss"],
        tAcc=results["train_acc"], 
        vAcc=results["val_acc"], 
        save_path=save_model_path.split('.')[0])

    if "test" in dataset:
        patient_merged_dic = tt.test(loaders["test"], save_model_path, le, p.parse_args().disable_tqdm)
        tt.saveAUC(patient_merged_dic, save_model_path.split('.')[0])
