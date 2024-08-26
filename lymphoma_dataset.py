"""\033[38;5;224m    This python code is used to create datasets of lymphoma pictures with the data stored in the csv thanks to the code "lymphoma_csv_data.py".
    it automatically create a csv with a name based on the inputs given    
    \033[38;5;213m
"""

import argparse
import pandas as pd
import numpy as np
import os


def df_train_val_test(full_pd_df, st = "", fract_sample=1, extern_val=True):
    """Create a csv file at the location /Dataset/{name}.csv of the code.
        Input a pandas dataframe for it to be worked on. you can choose to have a smaller dataset, using fract_sample.
        And to have an external validation (one patient can only be in one folder train/val/test) or internal

    Args:
        full_pd_df (pandas.core.frame.DataFrame): The dataset you want to categorize train/val/test
        st (str): the name of the subtype for saving
        fract_sample (float, optional): the fraction of sample you want from your bigger dataset (between 0 and 1) . Defaults to 1.0.
        extern_val (bool, optional): if you want external validation or not. Defaults to True.

    Raises:
        Exception: if fract_sample is not between 0 and 1
    """
    if fract_sample > 1 or fract_sample < 0:
        raise Exception(
            "Problem with the value of fract_sample, it must be between 0 and 1"
        )
    # if internal validation 80% train / 20% val
    if not extern_val:
        full_pd_df = full_pd_df.sample(frac=fract_sample)
        full_pd_df["folder"] = np.where(full_pd_df.index % 5 < 4, "train", "val")

    # if external validation 60% train / 20% val / 20% test
    else:
        # choose random patient (mix them in the process)
        sample_pat = np.random.choice(
            full_pd_df["patient"].unique(),
            int(full_pd_df["patient"].nunique() * fract_sample),
            replace=False,
        )
        train_size = int(len(sample_pat) * 0.6)
        test_size = int(len(sample_pat) * 0.2)
        train_patients, val_patients, test_patients = (
            sample_pat[:train_size],
            sample_pat[train_size:-test_size],
            sample_pat[-test_size:],
        )
        # Mapping patient in folders train/val/test
        patient_folder_map = {p: "train" for p in train_patients}
        patient_folder_map.update({p: "val" for p in val_patients})
        patient_folder_map.update({p: "test" for p in test_patients})
        full_pd_df["folder"] = full_pd_df["patient"].map(patient_folder_map)

    # drop data with no folder attributed and save in csv file
    full_pd_df = full_pd_df.dropna(subset=["folder"])
    full_pd_df.to_csv(
        f"{os.path.dirname(os.path.realpath(__file__))}/Dataset/DS_Lymph_{st if st is not None else ""}{f"{int(fract_sample*100)}p_" if fract_sample!=1 else ""}{"ext" if extern_val else "int"}.csv",
        index=False,
        header=True,
    )


def parser_init():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "-csv",
        "--csv_path",
        help="The path of the input csv with your data",
        type=str,
        required=True,
    )

    parser.add_argument(
        "-st",
        "--sub_type",
        help="If you want your dataset to be only about one cell type (eg. 'LY', 'MO', ...)",
        default=None,
        type=str,
        required=False,
    )

    parser.add_argument(
        "-fs",
        "--frac_sample",
        help="If your dataset is to big, the fraction of data you would like (default : 1)",
        default=1,
        type=float,
        required=False,
    )

    parser.add_argument(
        "--int",
        dest="extern_val",
        action='store_false',
        help="If you want you patient can be in multiple categories train/val/test (default : False)\033[0m"
    )

    parser.set_defaults(extern_val=True)


    return parser


def create_csv(p: argparse.ArgumentParser):
    CSV_PATH = p.parse_args().csv_path
    fullDF = pd.read_csv(CSV_PATH)

    st = p.parse_args().sub_type
    if st:
        fullDF = fullDF[fullDF["type"] == st].reset_index(drop=True)
        st += "_"

    frac_sample = p.parse_args().frac_sample
    extern_val = p.parse_args().extern_val
    print(extern_val)
    df_train_val_test(fullDF, st, frac_sample, extern_val)


if __name__ == "__main__":
    parser = parser_init()
    create_csv(parser)
