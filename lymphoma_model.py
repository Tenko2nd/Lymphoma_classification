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
import torch
import pandas as pd

import lymphoma_dataset_class as L
import constant as C
import train_fct as train
import test_fct as test


from IftimDevLib.IDL.pipelines.evaluation.classification import EarlyStopping
from IftimDevLib.IDL.components.datasets.data_utils import set_seed

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
        "-wd",
        "--decay",
        default=0.0,
        type=float,
        help="The weight decay for the model",
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
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'  # helps prevent out-of-memory errors and makes more efficient use of the available GPU memory
    set_seed(221) # random seed
    p = parser_init()
    
    le_path = f"{os.getcwd()}/Lymphoma_labelEncoder.pkl"
    with open(le_path, "rb") as f:
        le = preprocessing.LabelEncoder()
        le.classes_ = pickle.load(f)

    date = datetime.now().strftime("%m%d-%H%M")
    name = p.parse_args().name
    
    # create folder for model, result and AUC to save in
    os.mkdir(f"{os.getcwd()}/Model/{name}_{date}")

    dataset_path = p.parse_args().dataset_path
    df = pd.read_csv(dataset_path)

    for test_fold, _ in df.groupby('folder'):

        dataset_df = {}
        dataset_df["test"] = df[df['folder'] == test_fold]
        train_val_df= df[df['folder'] != test_fold]

        print(f"test = {test_fold}")
        
        os.mkdir(f"{os.getcwd()}/Model/{name}_{date}/{name}_{date}_{test_fold}")

        modelPaths = {}

        for folder, _ in train_val_df.groupby('folder'):

            save_model_path = f"{os.getcwd()}/Model/{name}_{date}/{name}_{date}_{test_fold}/{name}_{date}_{folder}.pt"
            modelPaths[folder] = save_model_path

            print(f"Val = {folder}")
            dataset_df["train"] = train_val_df[train_val_df['folder'] != folder]
            dataset_df["val"] = train_val_df[train_val_df['folder'] == folder]
            print(f"train : {dataset_df["train"].groupby(["categorie"])["categorie"].count().to_dict()}")
            print(f"validation : {dataset_df["val"].groupby(["categorie"])["categorie"].count().to_dict()}")
            
            dataset = {
                x: L.LymphomaDataset(
                    pd_file=dataset_df[x].reset_index(drop=True),
                    transform=transforms.Compose(
                        [
                            transforms.ToTensor(),
                            transforms.Resize(size = C.IMG_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
                            transforms.CenterCrop(C.IMG_SIZE),
                            transforms.RandomApply(torch.nn.ModuleList([
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomVerticalFlip(p=0.5),
                                transforms.RandomRotation(degrees=45),]), p=0 if x == "train" else 0)
                        ]
                    ),
                )
                for x in ["train", "val"] + (["test"] if "test" in dataset_df.keys() else [])
            }
            lenDataset = {x: len(dataset[x]) for x in dataset.keys()}

            early_stopping = EarlyStopping(
                patience=p.parse_args().early_stop,
                verbose=True,
                path=save_model_path,
            )

            loaders = train.loaders(
                dataset=dataset, 
                batch_size=p.parse_args().batch_size, 
                workers=p.parse_args().workers)
            
            results = train.train(
                train_dataloader=loaders["train"],
                val_dataloader=loaders["val"],
                learning_rate=p.parse_args().learning_rate,
                decay=p.parse_args().decay,
                epochs=200,
                label_encoder=le,
                early_stopping=early_stopping,
                disable_tqdm = p.parse_args().disable_tqdm)
            
            train.saveLearnCurves(
                tLoss=results["train_loss"], 
                vLoss=results["val_loss"],
                tAcc=results["train_acc"], 
                vAcc=results["val_acc"], 
                save_path=save_model_path.split('.')[0])

        if "test" in dataset:
            print(f"test : {dataset_df["test"].groupby(["categorie"])["categorie"].count().to_dict()}")
            foldersDf = []
            for folder, path in modelPaths.items():
                print("test val : ", folder)
                resDF = test.test(loaders["test"], path, le, p.parse_args().disable_tqdm)
                foldersDf.append(resDF)

            save = f"{os.getcwd()}/Model/{name}_{date}/{name}_{date}_{test_fold}/out_{name}_{date}_{test_fold}"
            matrix_info = test.assemble_n_aggregate(foldersDf, save)
            matrix_info['p_value'] = test.p_value(matrix_info['targets'], matrix_info['predictions'])
            test.confusion_matrix(matrix_info, save, le)
