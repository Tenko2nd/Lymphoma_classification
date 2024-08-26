"""\033[38;5;224m    This python code is used only to calculate the values we need to make our normalization more efficient on our dataset.
    This will print out the mean and standard deviation of your images based on the CSV you provided to your dataset.
    \033[38;5;208m//!\\\\ Use the same CSV and change the variable names based on yours!\033[0m
"""

from PIL import Image
import pandas as pd
import torch
import torchvision.transforms as transforms
import warnings

warnings.simplefilter("ignore", UserWarning)

# Read the CSV and store it in a pandas dataframe
CSV_PATH = r"C:\Users\mc29047i\Documents\Data2train\data.csv"
df = pd.read_csv(CSV_PATH)

# Get only a list of the paths of your images
img_path = df.loc[:, "img_path"]

# Initialize a list of mean and std for each images
allMean, allStd = [], []
transform = transforms.Compose([transforms.ToTensor()])

# Calculate mean and std of each images and append it to the list
for path in img_path:
    img = transform(Image.open(path))
    mean, std = img.mean([1, 2]), img.std([1, 2])
    allMean.append(mean)
    allStd.append(std)

# Calculate the mean of all means and stds and print them
mean_mean = torch.stack(allMean, dim=0).mean(dim=0)
mean_std = torch.stack(allStd, dim=0).mean(dim=0)
print(
    f"The means values of the images on your dataset are:\n\tMean : {mean_mean}\n\tStd : {mean_std}"
)
