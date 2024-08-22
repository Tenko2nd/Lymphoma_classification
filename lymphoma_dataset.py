"""
    This python code is used to create datasets of lymphoma pictures with the data stored in the csv thanks to the code "lymphoma_csv_data.py".\n
    It's using pytorch dataset class and some transformation (Rescale, Crop, ToTensor & Normalize) thanks to torchvision.\n
    the data are stored as variable 'image', 'categorie' and 'tabular with :\n
            _ image in the form of torch.tensor (3, 360, 360)\n
            _ categorie is either 'LCM' ('Lymphome à cellule du manteau') ou 'LZM' ('Lymphome de la zone marginale')\n
            _ tabular are all the metadata associated with each images\n
\n
    The datasets created are stored in pickle files for you to use them later.\n
    //!\\ You need to import the class for you to use the pickled datasets !!!\n

"""

import torch
import pandas as pd
from skimage import io, transform
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import pickle

CSV_PATH = r"C:\Users\mc29047i\Documents\Data2train\data.csv"
fullDF = pd.read_csv(CSV_PATH)


class LymphomaDataset(Dataset):
    """MZL & MCL Dataset"""

    def __init__(self, pd_file, transform=None):
        self.df = pd_file
        self.transform = transform
        self.classes = self.df["categorie"].unique().tolist()
        self.tabular = self.df.drop(columns=["img_path", "categorie"])

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


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        return transform.resize(image, (new_h, new_w))


class Crop(object):
    """Crop the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image):

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = int((h - new_h) / 2)
        left = int((w - new_w) / 2)

        image = image[top : top + new_h, left : left + new_w]

        return image


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        return torch.from_numpy(image.transpose((2, 0, 1)))


lymphomas_mean = torch.Tensor([0.8417, 0.7080, 0.7004])
lymphomas_std = torch.Tensor([0.1695, 0.1980, 0.0855])


def df_train_val_test(full_pd_df, fract_sample=1, extern_val=True):
    """Create two pandas dataframes (train and validation) based on a bigger given in parameter.
        It make sure there is the same number of data in each categories.
        You can tune the percentile of data you want from your bigger dataframe, the percentile of train/validation and if you need external validation.

    Args:
        full_pd_df (pandas.core.frame.DataFrame): Dataframe with all internal informations
        fract_sample (int, optional): If you only want a small part of a big dataset. Defaults to 1.
        fract_train (float, optional): Percentile of data you want in your train set. Defaults to 0.8.
        extern_val (bool, optional): If you want your validation to be external or not (internal). Defaults to True.

    Returns:
        dic of Dataframe: Dataframes train and val placed in a dic (accessed with ["train"] or ["val"])
    """
    if fract_sample > 1 or fract_sample < 0:
        raise Exception(
            "Problem with the value of fract_sample, it must be between 0 and 1"
        )
    # if internal validation
    if not extern_val:
        dfTot = full_pd_df.sample(frac=fract_sample)
        small_dfs = {"train": dfTot[dfTot.index % 5 < 3]}
        small_dfs["val"] = dfTot[dfTot.index % 5 == 3]
        small_dfs["test"] = dfTot[dfTot.index % 5 == 4]

    # if external validation
    else:
        sample_pat = np.random.choice(
            full_pd_df["patient"].unique(),
            int(full_pd_df["patient"].nunique() * fract_sample),
            replace=False,
        )
        train_size = int(len(sample_pat) * 0.6)
        test_size = int(len(sample_pat) * 0.2)
        train_patients, val_patients, test_patients = (
            sample_pat[:train_size],
            sample_pat[train_size:-test_size],
            sample_pat[-test_size:],
        )
        # Create the training and validation dataframes
        small_dfs = {
            "train": full_pd_df.query("patient in @train_patients"),
            "val": full_pd_df.query("patient in @val_patients"),
            "test": full_pd_df.query("patient in @test_patients"),
        }
    return small_dfs


# Dataset with all images
DSLymph = LymphomaDataset(
    pd_file=fullDF,
    transform=transforms.Compose(
        [
            Rescale(360),
            Crop(360),
            ToTensor(),
            transforms.Normalize(lymphomas_mean, lymphomas_std),
        ]
    ),
)
with open("Dataset/DS_Lymph.pkl", "wb") as f:
    pickle.dump(DSLymph, f, -1)

# Dataset with only around 30% images and external validation
pd_small_ext = df_train_val_test(fullDF, fract_sample=0.3, extern_val=True)
DSLymph_small_ext = {
    x: LymphomaDataset(
        pd_file=pd_small_ext[x].reset_index(drop=True),
        transform=transforms.Compose(
            [
                Rescale(360),
                Crop(360),
                ToTensor(),
                transforms.Normalize(lymphomas_mean, lymphomas_std),
            ]
        ),
    )
    for x in ["train", "val", "test"]
}
with open("Dataset/DS_Lymph_small_ext.pkl", "wb") as f:
    pickle.dump(DSLymph_small_ext, f, -1)

# Dataset with only around 30% images and internal validation
pd_small_int = df_train_val_test(fullDF, fract_sample=0.3, extern_val=False)
DSLymph_small_int = {
    x: LymphomaDataset(
        pd_file=pd_small_int[x].reset_index(drop=True),
        transform=transforms.Compose(
            [
                Rescale(360),
                Crop(360),
                ToTensor(),
                transforms.Normalize(lymphomas_mean, lymphomas_std),
            ]
        ),
    )
    for x in ["train", "val", "test"]
}
with open("Dataset/DS_Lymph_small_int.pkl", "wb") as f:
    pickle.dump(DSLymph_small_int, f, -1)


# Dataset with only 'LY' subtype and external validation
LYdf = fullDF[fullDF["type"] == "LY"].reset_index(drop=True)
pd_LY = df_train_val_test(LYdf, fract_sample=1, extern_val=True)
DSLymph_LY_ext = {
    x: LymphomaDataset(
        pd_file=pd_LY[x].reset_index(drop=True),
        transform=transforms.Compose(
            [
                Rescale(360),
                Crop(360),
                ToTensor(),
                transforms.Normalize(lymphomas_mean, lymphomas_std),
            ]
        ),
    )
    for x in ["train", "val", "test"]
}
with open("Dataset/DS_Lymph_LY_ext.pkl", "wb") as f:
    pickle.dump(DSLymph_LY_ext, f, -1)

# Dataset with only 'SNE' subtype and external validation
SNEdf = fullDF[fullDF["type"] == "SNE"].reset_index(drop=True)
pd_SNE = df_train_val_test(SNEdf, fract_sample=1, extern_val=True)
DSLymph_SNE_ext = {
    x: LymphomaDataset(
        pd_file=pd_SNE[x].reset_index(drop=True),
        transform=transforms.Compose(
            [
                Rescale(360),
                Crop(360),
                ToTensor(),
                transforms.Normalize(lymphomas_mean, lymphomas_std),
            ]
        ),
    )
    for x in ["train", "val", "test"]
}
with open("Dataset/DS_Lymph_SNE_ext.pkl", "wb") as f:
    pickle.dump(DSLymph_SNE_ext, f, -1)
