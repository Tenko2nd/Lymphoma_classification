"""\033[38;5;224m    This python script is used to train a model based on a dataset required as an input.
    It can take on multiple parameters : 
        - batch_size (default 16)
        - workers (default 4)
        - learning_rate (default 0.0001)
        - weight_decay (default 0.0)
        - early_stop (default 5)
        - name (default 'anon')
    It will save the output model on the Model folder as a new folder '{name}_{date}' all models and results will be saved in it.
    The model as an early stopping and only take the most efficient model in loss.\033[38;5;213m
"""

from argparse import ArgumentParser, RawTextHelpFormatter
from datetime import datetime
from pickle import load
from warnings import simplefilter
import os

from sklearn.preprocessing import LabelEncoder
from torchvision import transforms
import pandas as pd
import torch

import constant as C
import lymphoma_dataset_class as L
import test_fct as test
import train_fct as train


from IftimDevLib.IDL.components.datasets.data_utils import set_seed
from IftimDevLib.IDL.pipelines.evaluation.classification import EarlyStopping



def init_environment():
    """Initialize environment variable and seeds"""
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'  # helps prevent out-of-memory errors and makes more efficient use of the available GPU memory
    set_seed(221) # random seed
    simplefilter("ignore", FutureWarning)

def parser_init():
    """initialize the parser and returns it"""
    parser = ArgumentParser(
        description=__doc__, formatter_class=RawTextHelpFormatter
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
        default=16,
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
        default=0.0001,
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
        "--patience",
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

    parser.add_argument(
        "--precomputed",
        dest="precomputed",
        action="store_true",
        help="If you have already precomputed the embedings",
    )

    parser.add_argument(
        "-name",
        "--name",
        default="anon",
        help="The name of the categories the model is training on\033[0m",
        required=False,
    )

    return parser

def load_label_encoder() -> LabelEncoder:
    """Load the label encoder"""
    le_path = f"{os.getcwd()}/Lymphoma_labelEncoder.pkl"
    with open(le_path, "rb") as f:
        label_encoder = LabelEncoder()
        label_encoder.classes_ = load(f)
    return label_encoder

def create_customs_datasets(dataset_df: dict, precomputed : bool) -> dict:
    """Create the customs dataset based on the dictionnary of Dataframes

    Args:
        dataset_df (dict): a dictionnary of dataframes containing the train, val, and test dataframes
        precomputed (bool): If the images has already been precomputed

    Returns:
        dict: a dictionnary of customs datasets, containing the train, val, and test datasets
    """
    # create the datasets in a dict
    datasets = {
        x: L.LymphomaDataset(
            pd_file=dataset_df[x].reset_index(drop=True),
            precomputed=precomputed,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(size = C.IMG_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.CenterCrop(C.IMG_SIZE),
                    transforms.RandomApply(torch.nn.ModuleList([
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomVerticalFlip(p=0.5),
                        transforms.RandomRotation(degrees=45),]), p=0 if x == "train" else 0) #change p value for data augmentation. Don't work with precomputed
                ]
            ),
        )
        for x in ["train", "val", "test"]
    }
    return datasets

def main():
    if not os.path.exists(f"{os.getcwd()}/Model/"):
        os.makedirs(f"{os.getcwd()}/Model/")

    parser = parser_init()
    # initialize variable
    batch_size=parser.parse_args().batch_size
    dataset_path = parser.parse_args().dataset_path
    date = datetime.now().strftime("%m%d-%H%M")
    decay=parser.parse_args().decay
    df = pd.read_csv(dataset_path)
    disable_tqdm = parser.parse_args().disable_tqdm
    label_encoder = load_label_encoder()
    learning_rate=parser.parse_args().learning_rate
    name = parser.parse_args().name
    patience=parser.parse_args().patience
    precomputed = parser.parse_args().precomputed
    workers=parser.parse_args().workers

    # Create empty DataFrames
    all_patient_df = pd.DataFrame()
    all_images_df = pd.DataFrame()
    
    # create fold for model, result and AUC to save in
    root = f"{os.getcwd()}/Model/{name}_{date}"
    os.mkdir(root)

    # loop throught each fold, it will become the test fold, the others are the train/val
    for test_fold, _ in df.groupby('fold'):

        dataset_df = {}
        # Divide the fold train-val / test
        dataset_df["test"] = df[df['fold'] == test_fold]
        train_val_df= df[df['fold'] != test_fold]

        print(f"test = {test_fold}")

        fold_path = f"{root}/{name}_{date}_{test_fold}"
        os.mkdir(fold_path)

        foldsDf = []
        # Loop through the train/val folds, it will become the val fold, the others the train folds
        for fold, _ in train_val_df.groupby('fold'):

            save_model_path = f"{fold_path}/{name}_{date}_{fold}.pt"

            print(f"Val = {fold}")

            # Divide the fold train / val
            dataset_df["train"] = train_val_df[train_val_df['fold'] != fold]
            dataset_df["val"] = train_val_df[train_val_df['fold'] == fold]
            # Print repartition of each categories for train and val folds
            print(f"train : {dataset_df["train"].groupby(["categorie"])["categorie"].count().to_dict()}")
            print(f"validation : {dataset_df["val"].groupby(["categorie"])["categorie"].count().to_dict()}")
            
            datasets = create_customs_datasets(dataset_df=dataset_df, precomputed=precomputed)

            early_stopping = EarlyStopping(
                patience=patience,
                verbose=True,
                path=save_model_path)

            loaders = train.loaders(
                dataset=datasets, 
                batch_size=batch_size, 
                workers=workers)
            
            results = train.train(
                train_dataloader=loaders["train"],
                val_dataloader=loaders["val"],
                precomputed=precomputed,
                learning_rate=learning_rate,
                decay=decay,
                epochs=200,
                label_encoder=label_encoder,
                early_stopping=early_stopping,
                disable_tqdm = disable_tqdm,
                targets=datasets["train"].targets)
            
            train.saveLearnCurves(
                tLoss=results["train_loss"], 
                vLoss=results["val_loss"],
                save_path=save_model_path.split('.')[0])
            
            # Test the performance of the model for this fold
            print("test for ", fold)
            resDF = test.test(
                loader=loaders["test"], 
                model_path=save_model_path, 
                le=label_encoder, 
                precomputed=precomputed, 
                disable_tqdm=disable_tqdm)
            foldsDf.append(resDF)

        # Print repartition of each categories for test fold
        print(f"test : {dataset_df["test"].groupby(["categorie"])["categorie"].count().to_dict()}")

        # Test the performances of the models for this fold of the k-fold
        save = f"{fold_path}/out_{name}_{date}_{test_fold}_"
        patient_stats, before_agg, output = test.assemble_n_aggregate(foldersDf=foldsDf, save_path=save, le=label_encoder)
        # Write the results in a txt file for faster accessibility
        with open(f'{root}/resume_{name}.txt', 'a+') as resume:
            resume.write(f"test folds : {test_fold}\n")
            resume.write("\n".join(output))
            resume.write("\n")

        # Add the performance data of this fold to the previous folds
        all_patient_df = pd.concat([all_patient_df, patient_stats])
        all_images_df = pd.concat([all_images_df, before_agg])

    # After all all fold has been the test fold on wich a k-fold has been applied, arrange the global results
    # Sort the data by lower difference between result and expectation
    all_images_df = all_images_df.sort_values('diff').reset_index(drop=True)
    all_patient_df = all_patient_df.sort_values('diff').reset_index(drop=True)
    # Save the data to a CSV file
    all_images_df.to_csv(f"{root}/all_images.csv",index=False,header=True,)
    all_patient_df.to_csv(f"{root}/all_patients.csv",index=False,header=True,)

    # Print the global results in the txt file
    output = test.saveResult(patient_stats=all_patient_df, save_path=f"{root}/", le=label_encoder)
    with open(f'{root}/resume_{name}.txt', 'a+') as resume:
            resume.write(f"[INFO] Global results: \n")
            resume.write("\n".join(output))


if __name__ == "__main__":
    init_environment()
    main()

    
