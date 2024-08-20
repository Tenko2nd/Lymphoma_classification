"""
    This python code is used only to calculate wich cells type can be used to prepare a dataset based on the parameters minCatSize and minDataSize.\n
    The parameter minDataSize is here to filter out the cells type that are less pistures than the parameter requires, the minCatSize is the same but for the categories repartition.\n
    The cells type the respect both conditions will be displayed green, else brown if it only doesn't respect the categories parameters or red for the rest.\n
"""

import pandas as pd

CSV_PATH = r"C:\Users\mc29047i\Documents\Data2train\data.csv"
fullDF = pd.read_csv(CSV_PATH)

dftype = {
    x: fullDF[fullDF["type"] == x].reset_index(drop=True)
    for x in fullDF["type"].unique()
}

minDataSize = 1000  # The minimum number of pictures you tolerate in your dataset
minCatSize = 500  # The minimum number of picture you tolerate for the categories

print("Type\tSize\tLCM/LZM repartition")
for x in dftype:
    color = "\033[0m"  # no color
    lenTOT, lenLCM, lenLZM = (
        len(dftype[x]),
        len(dftype[x][dftype[x]["categorie"] == "LCM"]),
        len(dftype[x][dftype[x]["categorie"] == "LZM"]),
    )
    if lenTOT < minDataSize:
        color = "\033[38;5;124m"  # red color
    elif lenLCM < minCatSize or lenLZM < minCatSize:
        color = "\033[38;5;130m"  # brow color
    else:
        color = "\033[38;5;154m"  # green color
    print(
        f"{color}{x}\t{lenTOT}\tLCM : {(lenLCM/lenTOT)*100:.1f}%\tLZM : {(lenLZM/lenTOT)*100:.1f}%\033[0m"
    )
