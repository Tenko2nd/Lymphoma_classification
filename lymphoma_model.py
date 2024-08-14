from lymphoma_dataset import LymphomaDS_resize360_small
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import models
import torch
import pickle
import torch.nn as nn
import torch.optim as optim
from sklearn import preprocessing
import math


# Parameters
BATCH_SIZE = 32
WORKERS = 3
RATIO_TRAIN_VAL_TEST = [0.8, 0.2, 0]
DATASET = LymphomaDS_resize360_small
LE_PATH = "Lymphoma_labelEncoder.pkl"
LEARNING_RATE = 0.001
SAVE_MODEL_PATH = "mod_res50_50ep_3k_ccub.pth"
NUM_EPOCHS = 50

train_data, test_data = [LymphomaDS_resize360_small[x] for x in ["train", "val"]]

lenDataSet = {"train": len(train_data), "val": len(test_data)}

with open(LE_PATH, "rb") as f:
    le = preprocessing.LabelEncoder()
    le.classes_ = pickle.load(f)


loaders = {
    "train": DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS
    ),
    "val": DataLoader(
        test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS
    ),
}

model = models.resnet50(pretrained=True)

# Freeze all layers except the final classification layer
for name, param in model.named_parameters():
    param.requires_grad = "fc" in name


# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Move the model to the GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)


from custom_funcions import printProgressBar

if __name__ == "__main__":
    # Training loop
    trainLossList, valLossList, trainAccList, valAccList = [], [], [], []
    for i in range(NUM_EPOCHS):
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for idx, (inputs, labels, _) in enumerate(loaders[phase]):
                inputs = inputs.float().to(device)
                labels = torch.tensor(le.transform(labels)).to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels.long())

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                printProgressBar(
                    idx + 1,
                    math.ceil(lenDataSet[phase] / BATCH_SIZE),
                    f"{phase, i+1} Progress",
                    "Complete",
                    length=50,
                )

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / (
                len(train_data) if phase == "train" else len(test_data)
            )
            epoch_acc = running_corrects.double() / (
                len(train_data) if phase == "train" else len(test_data)
            )

            if phase == "train":
                trainAccList.append(epoch_acc)
                trainLossList.append(epoch_loss)
            else:
                if epoch_loss < min(valLossList) if valLossList else float("inf"):
                    print(
                        "\033[92m "
                        + "Best model "
                        + "\033[95m\033[3m"
                        + f"(acc: {epoch_acc}, loss: {epoch_loss})\033[0m"
                    )
                    # Save the model
                    torch.save(model.state_dict(), SAVE_MODEL_PATH)
                valAccList.append(epoch_acc)
                valLossList.append(epoch_loss)

            print(f"{phase, i+1} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    print("Training complete!")

    plt.subplot(2, 1, 1)
    plt.plot(trainLossList, "g", valLossList, "r")
    plt.legend(("train", "val"))
    plt.title("Loss")
    plt.subplot(2, 1, 2)
    plt.plot(trainAccList, "g", valAccList, "r")
    plt.title("Accuracy")
    plt.legend(("train", "val"))
    plt.savefig(f"{SAVE_MODEL_PATH}_res_train.png")
    plt.show()
