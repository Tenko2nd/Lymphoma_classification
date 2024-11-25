"""
    This is where the pytorch dataset class is initialized.
    It's using pytorch dataset class and transformation (optional) thanks to torchvision.
    It take as input a pandas dataframe with all the data from a csv dataset.
    Optional, precomputed, used only if we use the precomuted version of an image instead (if the image has been saved as an numpy file in exit of the encoder)
    the data are stored as variable 'image', 'classes' and 'tabular with :
            - image in the form of torch.tensor (3, 360, 360)
            - categorie is either 'LCM' ('Lymphome à cellule du manteau'), 'LZM' ('Lymphome de la zone marginale') or SGTEM (control patients)
            - tabular are all the metadata associated with each images (here it is the 'patient', 'image_reference' and 'cell_type')
"""

from numpy import load
from skimage.io import imread
from torch import is_tensor
from torch.utils.data import Dataset


class LymphomaDataset(Dataset):
    """MZL & MCL Dataset"""

    def __init__(self, pd_file, precomputed=False, transform=None):
        self.df = pd_file
        self.precomputed = precomputed
        self.transform = transform
        self.classes = self.df["categorie"].unique().tolist()
        self.tabular = self.df.drop(columns=["img_path", "categorie", "fold"])
        self.targets = self.df["categorie"].tolist()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index):
        if is_tensor(index):
            idx = idx.tolist()
        if self.precomputed:
            image = load(self.df.iloc[index]["img_path"].replace(".jpg", ".npy"))
        else:
            image = imread(self.df.iloc[index]["img_path"]) / 255.0
            if self.transform:
                image = self.transform(image)
        categorie = self.df.iloc[index]["categorie"]
        tabular = self.tabular.loc[index].to_dict()
        return image, categorie, tabular
