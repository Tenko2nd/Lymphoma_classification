"""\033[38;5;224mThis python code is used to generate a csv file of all the images paths along with the category they belong to.
Note that for it to work you need a data architecture like :\033[38;5;110m
        DS_FOLDER/
        ├── Categorie1/
        │   ├── patient1/           \033[38;5;224m<-- The sub folder are not necessary but you can have them if you need\033[38;5;110m
        │   │   ├── picture1.jpg
        │   │   ├── picture2.jpg
        │   │   └── ~~~.jpg
        │   ├── patient2/
        │   └── ~~~/
        ├── Categorie2/
        │   ├── picture1.jpg
        │   ├── picture2.jpg
        │   └── ~~~.jpg
        └── data.csv                \033[38;5;224m<-- No need to have it at the beginning, it will be either create or overwrite

If you want it to work with other extention than jpg, just change it when it loop to get the images (ctrl+f 'jpg')
The data in the csv are stored as: 'absolute_image_path', 'categorie', 'type', 'patient', 'reference_of_image'
\033[38;5;208m//!\\\\ It is better for the DS_FOLDER path to be absolute !\033[38;5;213m
"""

from argparse import ArgumentParser, RawTextHelpFormatter
from glob import glob
import csv
import os
import pathlib


def parser_init() -> ArgumentParser:
    """Initialize the parser and returns it

    Returns:
        ArgumentParser: the parser to be returned
    """

    parser = ArgumentParser(description=__doc__, formatter_class=RawTextHelpFormatter)

    parser.add_argument(
        "-d",
        "--ds_path",
        help="The path of the DS_Folder\033[0m",
        required=True,
    )

    return parser


def get_image_paths(DS_FOLDER: str, categories: list) -> dict:
    """Get every image path in the dataset folder

    Args:
        DS_FOLDER (str): the dataset folder path
        categories (list): the list of categories contained in the dataset

    Returns:
        dict: dictionnary with every image path for each category
    """
    image_paths = {}
    # Loop throught all categories
    for c in categories:
        # Get a list of all images
        image_paths[c] = glob(f"{DS_FOLDER}/{c}/**/*.jpg", recursive=True)
        print(f"taille categorie {c} = {len(image_paths[c])}")
    return image_paths


def Create_CSV(DS_FOLDER: str, categories: list, image_paths: dict) -> None:
    """Create a CSV 'data.csv' with all the data from every images in our dataset

    Args:
        DS_FOLDER (str): the dataset folder path
        categories (list): the list of categories contained in the dataset
        image_paths (dict): the dictionnary with every image path for each category
    """
    # Open csv file to write in it
    with open(f"{DS_FOLDER}/data.csv", "w", newline="") as csvfile:
        dataWriter = csv.writer(
            csvfile, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        # Write first line as legends
        dataWriter.writerow(
            [
                "img_path",
                "categorie",
                "type",
                "patient",
                "reference",
            ]
        )
        # Loop for each category
        for c in categories:
            # Loop for each image in the category
            for path in image_paths[c]:
                # Get the cell type (the file is named 'LY_1234.jpg' for example)
                cell_type = os.path.basename(path).split("_")[0]
                # Get the parent directory of the file and extract the patient reference
                parent_dir = os.path.dirname(path)
                patient = os.path.basename(parent_dir)
                # Get the reference of the image (like : 'LY_1234')
                reference = pathlib.Path(path).stem
                # Write the info of the image in the CSV
                dataWriter.writerow([path, c, cell_type, patient, reference])


def main() -> None:
    parser = parser_init()
    DS_FOLDER = parser.parse_args().ds_path
    # Get the categories in dataset folder
    categories = [f for f in os.listdir(DS_FOLDER) if os.path.isdir(f"{DS_FOLDER}/{f}")]
    if not categories:
        raise Exception("\033[38;5;160m[ERROR] No categories detected\033[0m")
    image_paths = get_image_paths(DS_FOLDER=DS_FOLDER, categories=categories)
    Create_CSV(DS_FOLDER=DS_FOLDER, categories=categories, image_paths=image_paths)


if __name__ == "__main__":
    main()
