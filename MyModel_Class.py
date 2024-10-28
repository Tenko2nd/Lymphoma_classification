import torch
import torch.nn as nn
from transformers import AutoModel


class MyModel(nn.Module):
    def __init__(self, precomputed=False):
        self.precomputed = precomputed
        super(MyModel, self).__init__()
        if not precomputed:
            self.encoder = AutoModel.from_pretrained("owkin/phikon-v2")
            self.encoder.requires_grad_(False)  # Freeze the encoder layers
        self.fc1 = nn.Linear(1024, 128)  # hidden layer 1
        self.dropout = nn.Dropout(p=0.5)  # dropout layer
        self.fc2 = nn.Linear(128, 2)  # hidden layer 2

    def forward(self, x):
        if not self.precomputed:
            output = self.encoder(x)
            pooled_output = output.pooler_output
        else:
            pooled_output = x
        x = torch.relu(self.fc1(pooled_output))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
