"""\033[38;5;224m    This python code is used to create a <Label Encoder> based on the categories of the dataset created, thanks to the code "lymphoma_dataset.py".
    It take the different classes defined in the <classes> intern variable of the dataset and fit them to the label encoder
    \033[38;5;208m//!\\\\ The classes must be references in a variable named 'classes' inside of the __init__ function of the dataset.\033[38;5;224m
    After that the different classes are serialized in a <Pickle> file with the name "Lymphoma_labelEncoder.pkl".
    \033[38;5;110m//!\\\\ To deserialize the label encoder and use it you must follow this steps :
        from sklearn import preprocessing                   
        with open('Lymphoma_labelEncoder.pkl', 'rb') as f:  
            le = preprocessing.LabelEncoder()               
            le.classes_ = pickle.load(f)\033[38;5;213m
"""

import argparse

from sklearn import preprocessing
from torchvision import transforms
import pandas as pd
import pickle

import constant as C
import lymphoma_dataset_class as L


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "-d",
        "--dataset_path",
        help="The path of the dataset used for label encoding\033[0m",
        required=True,
    )

    df = pd.read_csv(parser.parse_args().dataset_path)
    dataset = L.LymphomaDataset(
        pd_file=df.loc[df["folder"] == "train"].reset_index(drop=True),
        transform=transforms.Compose(
            [
                L.Rescale(360),
                L.Crop(360),
                L.ToTensor(),
                transforms.Normalize(C.lymphomas_mean, C.lymphomas_std),
            ]
        ),
    )

    le = preprocessing.LabelEncoder()
    le.fit(dataset.classes)

    with open("Lymphoma_labelEncoder.pkl", "wb") as f:
        pickle.dump(le.classes_, f)


if __name__ == "__main__":
    main()
