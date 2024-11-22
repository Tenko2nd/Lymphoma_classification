"""\033[38;5;224m    This python code is used to create a <Label Encoder> based on the categories of the dataset created, thanks to the code "lymphoma_dataset.py".
    It take the different classes defined in the <classes> intern variable of the dataset and fit them to the label encoder
    \033[38;5;208m//!\\\\ The classes must be references in a variable named 'classes' inside of the __init__ function of the dataset.\033[38;5;224m
    After, the different classes are serialized in a <Pickle> file with the name "Lymphoma_labelEncoder.pkl".
    \033[38;5;110m//!\\\\ Here is an example for loading the <Label Encoder> :
        from sklearn import preprocessing                   
        with open('Lymphoma_labelEncoder.pkl', 'rb') as f:  
            le = preprocessing.LabelEncoder()               
            le.classes_ = pickle.load(f)
        print(le.classes_)\033[38;5;213m
"""

from argparse import ArgumentParser, RawTextHelpFormatter

from sklearn.preprocessing import LabelEncoder
import pandas as pd
import pickle

import lymphoma_dataset_class as L


def parser_init() -> ArgumentParser:
    """Initialize the parser and returns it"""

    parser = ArgumentParser(description=__doc__, formatter_class=RawTextHelpFormatter)

    parser.add_argument(
        "-d",
        "--dataset_path",
        help="The path of the dataset used for label encoding\033[0m",
        required=True,
    )

    return parser


def create_label_encoder(parser: ArgumentParser) -> None:
    """Create the pickle file containing the classes given from the dataset to the label encoder"""

    # read the dataset CSV and create a pandas dataframe
    dataframe = pd.read_csv(parser.parse_args().dataset_path)

    # Create custom dataset based on the dataframe
    dataset = L.LymphomaDataset(
        pd_file=dataframe.reset_index(drop=True),
    )

    # Initialize label encoder and fit the classes of the dataset
    le = LabelEncoder()
    le.fit(dataset.classes)

    # Put the classes of the label encoder in a pickle file
    with open("Lymphoma_labelEncoder.pkl", "wb") as f:
        pickle.dump(le.classes_, f)


def main() -> None:
    parser = parser_init()
    create_label_encoder(parser=parser)


if __name__ == "__main__":
    main()
