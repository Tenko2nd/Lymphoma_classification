from lymphoma_dataset import LymphomaDS_resize360
from torch.utils.data import DataLoader, random_split
from sklearn import preprocessing
import pickle
from torchvision import models
import torch
import torch.nn as nn
import torch.optim as optim


# Parameters
BATCH_SIZE = 32
RATIO_TRAIN_VAL_TEST = [0.8, 0.2, 0]
DATASET = LymphomaDS_resize360
LE_PATH = "Lymphoma_labelEncoder.pkl"
LEARNING_RATE = 0.001
SAVE_MODEL_PATH = "Lymphoma_classification_model.pth"

train_data, test_data = random_split(
    DATASET,
    [
        int(len(DATASET) * RATIO_TRAIN_VAL_TEST[0]),
        int(len(DATASET) * RATIO_TRAIN_VAL_TEST[1]) + len(DATASET) % 2,
    ],  # In case of odd number in DATASET
)

lenDataSet = {"train": len(train_data), "val": len(test_data)}

with open(LE_PATH, "rb") as f:
    le = preprocessing.LabelEncoder()
    le.classes_ = pickle.load(f)


loaders = {
    "train": DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0),
    "val": DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0),
}

model = models.resnet18(pretrained=True)

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

# Training loop
NUM_EPOCHS = 10
for i in range(NUM_EPOCHS):
    for phase in ["train", "val"]:
        if phase == "train":
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0

        for idx, (inputs, labels) in enumerate(loaders[phase]):
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
                int(lenDataSet[phase] / BATCH_SIZE) + 1,
                f"{phase} Progress",
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

        print(f"{phase, i+1} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

print("Training complete!")

# Save the model
torch.save(model.state_dict(), "Lymphoma_classification_model.pth")
