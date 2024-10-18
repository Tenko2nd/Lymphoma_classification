import torch
import torch.nn as nn
from transformers import AutoModel


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.encoder = AutoModel.from_pretrained("owkin/phikon-v2")
        self.encoder.requires_grad_(False)  # Freeze the encoder layers
        self.fc1 = nn.Linear(1024, 256)  # hidden layer 1
        self.fc2 = nn.Linear(256, 128)  # hidden layer 2
        self.fc3 = nn.Linear(128, 64)  # hidden layer 3
        self.fc4 = nn.Linear(64, 32)  # hidden layer 4
        self.fc5 = nn.Linear(32, 2)  # output layer

    def forward(self, x):
        output = self.encoder(x)
        pooled_output = output.pooler_output
        x = torch.relu(self.fc1(pooled_output))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x
