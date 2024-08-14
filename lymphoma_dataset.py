"""
    This python code is used to create a dataset of lymphoma pictures with the data stored in the csv thanks to the code "lymphoma_csv_data.py".\n
    It's using pytorch dataset class and some transformation (Rescale, Crop, ToTensor & Normalize) thanks to torchvision.\n
    the data are stored as dictionnary {'image': img, "categorie": cat} with :\n
            _ image in the form of torch.tensor (3, 360, 360)\n
            _ categorie is either 'LCM' ('Lymphome à cellule du manteau') ou 'LZM' ('Lymphome de la zone marginale')
\n
    You can import the dataset 'LymphomaDS_resize360' where you want to use it (eg: from lymphoma_dataset import LymphomaDS_resize360)\n
    //!\\ Make sure to be in the same folder !!!\n

"""

import torch
import pandas as pd
from skimage import io, transform
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.utils import shuffle

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


# Dataset with all images
# sourcery skip: identity-comprehension
LymphomaDS_resize360 = LymphomaDataset(
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


# Dataset with only around 20% images and same number LCM and LZM and unique patients in train and val
fraction_of_sample = 0.2

# Group by 'categorie' and then by 'patient', and sample a fraction of the patients
dfLCM = fullDF[fullDF["categorie"] == "LCM"]
dfLZM = fullDF[fullDF["categorie"] == "LZM"]
patients_LCM, patients_LZM = dfLCM["patient"].unique(), dfLZM["patient"].unique()
sampled_patients = shuffle(
    (patients_LCM).tolist()[: int(len(patients_LCM) * fraction_of_sample)]
    + (patients_LZM).tolist()[: int(len(patients_LCM) * fraction_of_sample)]
)

# Split the sampled patients into training and validation sets
split_train = 0.8
train_patients = sampled_patients[: int(len(sampled_patients) * split_train)]
val_patients = sampled_patients[int(len(sampled_patients) * split_train) :]

# Create the training and validation dataframes
small_dfs = {
    "train": fullDF[fullDF["patient"].isin(train_patients)],
    "val": fullDF[fullDF["patient"].isin(val_patients)],
}

LymphomaDS_resize360_small = {
    x: LymphomaDataset(
        pd_file=small_dfs[x].reset_index(drop=True),
        transform=transforms.Compose(
            [
                Rescale(360),
                Crop(360),
                ToTensor(),
                transforms.Normalize(lymphomas_mean, lymphomas_std),
            ]
        ),
    )
    for x in ["train", "val"]
}
