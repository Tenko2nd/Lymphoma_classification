"""\033[38;5;224m    This python code is used to create CSV datasets of lymphoma pictures with the data stored in the csv thanks to the code "lymphoma_csv_data.py".
    it automatically create a csv with a name based on the inputs given\033[38;5;213m
"""

from argparse import ArgumentParser, RawTextHelpFormatter
import os

import numpy as np
from pandas import read_csv, DataFrame


def map_images_2_fold(full_pd_df : DataFrame, extern_val : bool = True, k : int = 5, two_classes : bool = False) -> DataFrame:
    """ Create a csv file with a new column fold for later repartition.
        Input a pandas dataframe for it to be worked on (The one created in "lymphoma_csv_data.py").
        You can choose to have an external validation (one patient can only be in one fold train/val/test) or internal.
        The file is saved at the location /Dataset/{name}.csv of the code.

    Args:
        full_pd_df (DataFrame): The dataset you want to categorize train/val/test
        extern_val (bool, optional): if you want external validation or not. Defaults to True.
        k (int, optional): The number of fold you want in your dataset. Defaults to 5.
        two_classes (bool, optional): if you don't want the control patients. Defaults to False.

    Returns:
        DataFrame: The dataframe where all patient is assigned to a unique fold
    """

    if two_classes:
        full_pd_df = full_pd_df[full_pd_df["categorie"] != "SGTEM"]

    # if internal validation
    if not extern_val:
        # Create the fold column using vectorized operations
        full_pd_df["fold"] = "fold_" + ((full_pd_df.index % k) + 1).astype(str)

    else:
        # Get unique patients and shuffle them
        unique_patients = full_pd_df["patient"].unique()
        np.random.shuffle(unique_patients)

        # Create a mapping from patient into k fold
        patient_to_fold = {}
        for i, patient in enumerate(unique_patients):
            patient_to_fold[patient] = f"fold_{(i % k) + 1}"

        # Assign the fold to each row in the DataFrame
        full_pd_df["fold"] = full_pd_df["patient"].map(patient_to_fold)

    return full_pd_df


def parser_init() -> ArgumentParser:
    parser = ArgumentParser(
        description=__doc__, formatter_class=RawTextHelpFormatter
    )

    parser.add_argument(
        "-csv",
        "--csv_path",
        help="The path of the input csv with your data",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--int",
        dest="extern_val",
        action='store_false',
        help="If you want internal validation (one patient in train, val and test all at once)"
    )

    parser.add_argument(
        "-k",
        "--number_fold",
        default=5,
        type=int,
        help="The number of fold you want for later k_fold. Defaults to 5.",
        required=False,
    )

    parser.add_argument(
        "--two",
        dest="two_classes",
        action='store_true',
        help="If you only want two classes (no control patients)"
    )

    parser.add_argument(
        "-n",
        "--number_dataset",
        default=1,
        type=int,
        help="If external validation, the number of different mapping between patient and fold. Default to 1.\033[0m",
        required=False,
    )

    parser.set_defaults(extern_val=True)
    parser.set_defaults(two_classes=False)


    return parser


def main() -> None:
    parser = parser_init()

    # Read CSV and create <Dataframe>
    CSV_PATH = parser.parse_args().csv_path
    fullDF = read_csv(CSV_PATH)

    extern_val = parser.parse_args().extern_val
    two_classes = parser.parse_args().two_classes
    k = parser.parse_args().number_fold
    n = parser.parse_args().number_dataset
    if n<0:
        raise Exception(f"\033[38;5;160m[ERROR] 'n' cannot be negative but received : {n} \033[0m")
    
    if k<3:
        raise Exception(f"\033[38;5;160m[ERROR] 'k' must be at least 3 for train/val/test repartition. Received: {k} \033[0m")
    
    if not os.path.exists(f"{os.getcwd()}/Dataset/"):
        os.makedirs(f"{os.getcwd()}/Dataset/")
    
    for i in range(n):
        full_pd_df = map_images_2_fold(full_pd_df=fullDF, extern_val=extern_val, k=k, two_classes=two_classes)

        # Save CSV file
        full_pd_df.to_csv(
            f"{os.getcwd()}/Dataset/DS_Lymph_{"2" if two_classes else "3"}class_{"ext" if extern_val else "int"}_k{k}{f"_{i+1}" if n != 1 else ""}.csv",
            index=False,
            header=True,
        )


if __name__ == "__main__":
    main()
