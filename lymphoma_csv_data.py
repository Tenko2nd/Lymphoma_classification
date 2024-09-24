"""\033[38;5;224mThis python code is used to generate a csv file of all the images paths along with the categorie and the tabular metadata they belong to.
Note that for it to work you need an architecture like :\033[38;5;110m
        DS_FOLDER/
        |-- Categorie1/
        |   |-- patient1/           \033[38;5;224m<-- The sub folder are not necessary but can can have them if you need\033[38;5;110m
        |   |   |-- picture1.jpg
        |   |   |-- picture2.jpg
        |   |   L-- ~~~.jpg
        |   |-- patient2/
        |   L-- ~~~/
        |-- Categorie2/
        |   |-- picture1.jpg
        |   |-- picture2.jpg
        |   L-- ~~~.jpg
        |-- tabularMetadata.xlsx
        L-- data.csv                \033[38;5;224m<-- No need to have it at the beginning, it will be either create or overwrite

If you want it to work with other extention than jpg, just change it when it loop to get the images (ctrl+f 'jpg')
The data in the csv are stored as: 'absolute_image_path', 'categorie', 'type', 'patient', tabularMetadata
\033[38;5;208m//!\\\\ It is better for the DS_FOLDER path to be absolute !\033[38;5;213m
"""

import argparse
import csv
import glob
import os
import pathlib
import pandas as pd
from unidecode import unidecode

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawTextHelpFormatter
)

parser.add_argument(
    "-d",
    "--ds_path",
    help="The path of the DS_Folder\033[0m",
    required=True,
)

DS_FOLDER = parser.parse_args().ds_path

# Categories in dataset
categories = [f for f in os.listdir(DS_FOLDER) if os.path.isdir(f"{DS_FOLDER}{f}")]

image_paths = {}
# Loop throught all categories
for c in categories:
    # Get a list of all images
    image_paths[c] = glob.glob(f"{DS_FOLDER}{c}/**/*.jpg", recursive=True)
    print(f"taille categorie {c} = {len(image_paths[c])}")


# # Get the tabular metadata
# tabularPath = glob.glob(f"{DS_FOLDER}*.xlsx", recursive=False)

# xls = pd.ExcelFile(tabularPath[0])
# lcmDf, lzmDf = pd.read_excel(xls, "LCM_send"), pd.read_excel(xls, "LZM_send")
# lcmDf, lzmDf = lcmDf[lcmDf["ID"].notnull()], lzmDf[lzmDf["ID"].notnull()]


# # All the collumns usefull for our dataset
# tabCol = [
#     "CD5",
#     "CD10",
#     "CD20",
#     "CD23",
#     "CD38",
#     "CD43",
#     "CD79b",
#     "CD180",
#     "FMC7",
#     "intensité",
# ]

# Write data in csv file
with open(f"{DS_FOLDER}data.csv", "w", newline="") as csvfile:
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
        ]  # + [unidecode(x) for x in tabCol]
    )
    for c in ["LCM", "LZM"]:
        # df = lzmDf if (c == "LZM") else lcmDf
        for path in image_paths[c]:
            typ = os.path.basename(path).split("_")[0]
            # Get the parent directory of the file
            parent_dir = os.path.dirname(path)
            patient = os.path.basename(parent_dir)
            reference = pathlib.Path(path).stem
            # # Normalize, remove accents
            # tabular = [
            #     unidecode(x) if type(x) == str else x
            #     for x in df.loc[df["ID"] == patient, tabCol].values.flatten().tolist()
            # ]
            dataWriter.writerow([path, c, typ, patient, reference])  # + tabular
