"""\033[38;5;224m    This is where the pytorch dataset class is initialized.
    It's using pytorch dataset class and some transformation (Rescale, Crop, ToTensor & Normalize) thanks to torchvision.
    the data are stored as variable 'image', 'categorie' and 'tabular with :
            - image in the form of torch.tensor (3, 360, 360)
            - categorie is either 'LCM' ('Lymphome à cellule du manteau') ou 'LZM' ('Lymphome de la zone marginale')
            - tabular are all the metadata associated with each images\033[0m
"""

from skimage import io, transform
from torch.utils.data import Dataset
import torch
import numpy as np


class LymphomaDataset(Dataset):
    """MZL & MCL Dataset"""

    def __init__(self, pd_file, precomputed=False, transform=None):
        self.df = pd_file
        self.precomputed = precomputed
        self.transform = transform
        self.classes = self.df["categorie"].unique().tolist()
        self.tabular = self.df.drop(columns=["img_path", "categorie", "folder"])
        self.targets = self.df["categorie"].tolist()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            idx = idx.tolist()
        if self.precomputed:
            image = np.load(self.df.iloc[index]["img_path"].replace(".jpg", ".npy"))
        else:
            image = io.imread(self.df.iloc[index]["img_path"]) / 255.0
            if self.transform:
                image = self.transform(image)
        categorie = self.df.iloc[index]["categorie"]
        tabular = self.tabular.loc[index].to_dict()
        return image, categorie, tabular
