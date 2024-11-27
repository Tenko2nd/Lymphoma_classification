"""
This python code is used solely to precompute the embedding, and to save them as numpy files on your computer.
For it to work, you first need to have the file data.csv, generated in the 'lymphoma_csv_data.py' programm.
"""

from argparse import ArgumentParser, RawTextHelpFormatter
import os

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from transformers import AutoModel
from tqdm import tqdm
import torch
import numpy as np

import constant as C
import lymphoma_dataset_class as L


def parser_init() -> ArgumentParser:
    """initialize the parser and returns it"""
    parser = ArgumentParser(description=__doc__, formatter_class=RawTextHelpFormatter)

    parser.add_argument(
        "-d",
        "--data_CSV_path",
        help="The path of the CSV file created in the script 'lympphoma_csv_data.py'",
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
        "--pb_disable",
        dest="disable_tqdm",
        action="store_true",
        help="If you want to disable the progress bar (default : enable)",
    )

    return parser


def create_loader(dataset: Dataset, batch_size: int, workers: int) -> DataLoader:
    """Create the loaders train/val/test based on the dict of datasets given as input

    Args:
        dataset (dict[str, Dataset]): A dictionnary containing the three different datasets train/val/test
        batch_size (int): The number of element to be passed at once for each iteration
        workers (int): The number of workers you want you computer to use

    Returns:
        dict[str, DataLoader]: A dictionnary containing the three different loaders train/val/test
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        drop_last=False,
    )
    return loader


def create_customs_datasets(dataset_df: pd.DataFrame) -> Dataset:
    """Create the customs dataset based on the dictionnary of Dataframes

    Args:
        dataset_df (dict): a dictionnary of dataframes containing the train, val, and test dataframes
        precomputed (bool): If the images has already been precomputed

    Returns:
        dict: a dictionnary of customs datasets, containing the train, val, and test datasets
    """
    # create the datasets in a dict
    dataset = L.LymphomaDataset(
        pd_file=dataset_df.reset_index(drop=True),
        precomputed=False,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    size=C.IMG_SIZE, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.CenterCrop(C.IMG_SIZE),
            ]
        ),
    )
    return dataset


def main():
    parser = parser_init()
    model_name = "owkin/phikon-v2"

    data_CSV_path = parser.parse_args().data_CSV_path
    batch_size = parser.parse_args().batch_size
    workers = parser.parse_args().workers
    disable_tqdm = parser.parse_args().disable_tqdm

    data_path = os.path.dirname(data_CSV_path)
    if not os.path.exists(f"{data_path}/{model_name}/"):
        os.makedirs(f"{data_path}/{model_name}/LZM", exist_ok=True)
        os.makedirs(f"{data_path}/{model_name}/LCM", exist_ok=True)
        os.makedirs(f"{data_path}/{model_name}/SGTEM", exist_ok=True)

    data = pd.read_csv(data_CSV_path)
    data["fold"] = ""
    dataset = create_customs_datasets(data)
    loader = create_loader(dataset=dataset, batch_size=batch_size, workers=workers)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    encoder = AutoModel.from_pretrained(model_name)
    encoder.requires_grad_(False)  # Freeze the encoder layers
    encoder.to(device)
    # Putting encoder in eval mode
    encoder.eval()
    for inputs, labels, tabular in tqdm(
        loader,
        disable=disable_tqdm,
    ):

        images = inputs.to(device, non_blocking=True)

        # Turning on inference context manager
        with torch.inference_mode():
            # Forward pass
            outputs = encoder(images)
            pooled_outputs = outputs.pooler_output

        for i, reference in enumerate(tabular["reference"]):
            category = labels[i]
            patient = tabular["patient"][i]
            embedding = pooled_outputs[i].cpu()
            if not os.path.exists(f"{data_path}/{model_name}/{category}/{patient}/"):
                os.makedirs(
                    f"{data_path}/{model_name}/{category}/{patient}/", exist_ok=True
                )
            np.save(
                f"{data_path}/{model_name}/{category}/{patient}/{reference}.npy",
                embedding,
            )


if __name__ == "__main__":
    main()
