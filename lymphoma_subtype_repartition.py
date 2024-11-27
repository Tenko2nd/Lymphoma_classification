"""
    This python code is used only to calculate wich cells type can be used to prepare a dataset based on the parameters minCatSize and minDataSize.\n
    The parameter minDataSize is here to filter out the cells type that are less pistures than the parameter requires, the minCatSize is the same but for the categories repartition.\n
    The cells type the respect both conditions will be displayed green, else brown if it only doesn't respect the categories parameters or red for the rest.\n
"""

import pandas as pd

CSV_PATH = r"C:\Users\mc29047i\Lymphoma_classification\DS_Lymph_3class_k5.csv"
fullDF = pd.read_csv(CSV_PATH)

for fold in fullDF["folder"].unique():
    print(fold)
    df = fullDF[fullDF["folder"] == fold]
    dftype = {
        x: df[df["type"] == x].reset_index(drop=True) for x in df["type"].unique()
    }
    print("Type\t\tSize\tLCM/LZM repartition")
    for x in dftype:
        if "+" not in x:
            color = "\033[0m"  # no color
            lenTOT, lenLCM, lenLZM, lenSGTEM = (
                len(dftype[x]),
                len(dftype[x][dftype[x]["categorie"] == "LCM"]),
                len(dftype[x][dftype[x]["categorie"] == "LZM"]),
                len(dftype[x][dftype[x]["categorie"] == "SGTEM"]),
            )
            print(
                f"{x}\t\t{lenTOT}\tLCM : {(lenLCM/lenTOT)*100:.1f}%\tLZM : {(lenLZM/lenTOT)*100:.1f}%\tSGTEM : {(lenSGTEM/lenTOT)*100:.1f}%"
            )
