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


class LymphomaDataset(Dataset):
    """MZL & MCL Dataset"""

    def __init__(self, pd_file, transform=None):
        self.df = pd_file
        self.transform = transform
        self.classes = self.df["categorie"].unique().tolist()
        self.tabular = self.df.drop(columns=["img_path", "categorie", "folder"])
        self.targets = self.df["categorie"].tolist()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            idx = idx.tolist()
        image = io.imread(self.df.iloc[index]["img_path"]) / 255.0
        categorie = self.df.iloc[index]["categorie"]
        tabular = self.tabular.loc[index].to_dict()
        if self.transform:
            image = self.transform(image)
        return image, categorie, tabular
