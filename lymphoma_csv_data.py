"""
    This python code is used to generate a csv file of all the images paths along with the categorie and the tabular metadata they belong to.\n
    Note that for it to work you need an architecture like :\n
            DS_FOLDER/                  \n
            ├── Categorie1/             \n
            │   ├── patient1/           <-- The sub folder are not necessary but can can have them if you need\n
            │   │   ├── picture1.jpg    \n
            │   │   ├── picture2.jpg    \n
            │   │   └── ~~~.jpg         \n
            │   ├── patient2/           \n
            │   └── ~~~/                \n
            ├── Categorie2/             \n
            │   ├── picture1.jpg        \n
            │   ├── picture2.jpg        \n
            │   └── ~~~.jpg             \n
            ├── tabularMetadata.xlsx    \n
            └── data.csv                <-- No need to have it at the beginning, it will be either create or overwrite\n
\n
    If you want it to work with other extention than jpg, just change it when it loop to get the images (ctrl+f 'jpg')\n
    The data in the csv are stored as: 'absolute_image_path', 'categorie', 'type', 'patient', tabularMetadata\n
    //!\\ It is better for the DS_FOLDER path to be absolute !\n

"""

import csv
import glob
import os
import pandas as pd
from unidecode import unidecode

# Root images folder path
DS_FOLDER = r"C:\Users\mc29047i\Documents\Data2train"

# Categories in dataset
categories = [f for f in os.listdir(DS_FOLDER) if os.path.isdir(f"{DS_FOLDER}\\{f}")]

image_paths = {}
# Loop throught all categories
for c in categories:
    # Get a list of all images
    image_paths[c] = glob.glob(f"{DS_FOLDER}\\{c}/**/*.jpg", recursive=True)
    print(f"taille categorie {c} = {len(image_paths[c])}")


# Get the tabular metadata
tabularPath = glob.glob(f"{DS_FOLDER}/*.xlsx", recursive=False)

xls = pd.ExcelFile(tabularPath[0])
lcmDf, lzmDf = pd.read_excel(xls, "LCM_send"), pd.read_excel(xls, "LZM_send")
lcmDf, lzmDf = lcmDf[lcmDf["ID"].notnull()], lzmDf[lzmDf["ID"].notnull()]


# All the collumns usefull for our dataset
tabCol = [
    "CD5",
    "CD10",
    "CD20",
    "CD23",
    "CD38",
    "CD43",
    "CD79b",
    "CD180",
    "FMC7",
    "intensité",
]

# Write data in csv file
with open(f"{DS_FOLDER}\\data.csv", "w", newline="") as csvfile:
    dataWriter = csv.writer(
        csvfile, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL
    )
    # Write first line as legends
    dataWriter.writerow(
        ["img_path", "categorie", "type", "patient"] + [unidecode(x) for x in tabCol]
    )
    for c in categories:
        # Ignore the SGTEM pictures (because I don't know what it is)
        if c != "SGTEM":
            df = lzmDf if (c == "LZM") else lcmDf
            for path in image_paths[c]:
                typ = os.path.basename(path).split("_")[0]
                patient = path.split("\\")[-2]
                # Normalize, remove accents
                tabular = [
                    unidecode(x) if type(x) == str else x
                    for x in df.loc[df["ID"] == patient, tabCol]
                    .values.flatten()
                    .tolist()
                ]
                dataWriter.writerow([path, c, typ, patient] + tabular)
