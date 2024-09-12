"""\033[38;5;224m    This python code is used to create datasets of lymphoma pictures with the data stored in the csv thanks to the code "lymphoma_csv_data.py".
    it automatically create a csv with a name based on the inputs given\033[38;5;213m
"""

import argparse
import os

import numpy as np
import pandas as pd


def df_train_val_test(full_pd_df, st, fract_sample, extern_val):
    """ Create a csv file with a new column folder for later repartition. It as multiple train and no validation for k-fold CV.
        Input a pandas dataframe for it to be worked on. you can choose to have a smaller dataset, using fract_sample.
        And to have an external validation (one patient can only be in one folder train/val/test) or internal.
        The file is saved at the location /Dataset/{name}.csv of the code.

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
    # if internal validation 100% train-val
    if not extern_val:
        full_pd_df = full_pd_df.sample(frac=fract_sample)
        full_pd_df["folder"] = np.where(full_pd_df.index % 5 == 0, "train_1",
                                np.where(full_pd_df.index % 5 == 1, "train_2",
                                np.where(full_pd_df.index % 5 == 2, "train_3",
                                np.where(full_pd_df.index % 5 == 3, "train_4", "train_5"))))

    # if external validation 80% train-val / 20% test
    else:
        # choose random patient (mix them in the process)
        sample_pat = np.random.choice(
            full_pd_df["patient"].unique(),
            int(full_pd_df["patient"].nunique() * fract_sample),
            replace=False,
        )
        split_pat = np.array_split(sample_pat, 5)
        # Mapping patient in 5 folders train_(1,2,3,4)/test
        patient_folder_map = {p: f"train_{i+1}" if i < len(split_pat) - 1 else "test" for i, folder in enumerate(split_pat) for p in folder}
        full_pd_df["folder"] = full_pd_df["patient"].map(patient_folder_map)

    # drop data with no folder attributed and save in csv file
    full_pd_df = full_pd_df.dropna(subset=["folder"])
    full_pd_df.to_csv(
        f"{os.getcwd()}/Dataset/DS_Lymph_{st if st is not None else ""}{f"{int(fract_sample*100)}p_" if fract_sample!=1 else ""}{"ext" if extern_val else "int"}.csv",
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
        help="If you want your dataset to be only about one or multiple cell type (eg. ['LY'] or ['MO', 'SNE'])",
        default=None,
        nargs='+',
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
    print(st)
    if st:
        fullDF = fullDF[fullDF["type"].isin(st)].reset_index(drop=True)
        st = '_'.join(st)+'_'
        # else:
        #     raise Exception(f"\033[38;5;208m//!\\\\ the subtype {st} is not in the dataset!\033[0m")

    frac_sample = p.parse_args().frac_sample
    extern_val = p.parse_args().extern_val
    df_train_val_test(fullDF, st, frac_sample, extern_val)


if __name__ == "__main__":
    parser = parser_init()
    create_csv(parser)
