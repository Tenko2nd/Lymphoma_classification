import torch
import torch.nn as nn
from transformers import AutoModel

import constant as C


class MyModel(nn.Module):
    def __init__(self, precomputed=False):
        self.precomputed = precomputed
        super(MyModel, self).__init__()
        if not precomputed:
            self.encoder = AutoModel.from_pretrained("owkin/phikon-v2")
            self.encoder.requires_grad_(False)  # Freeze the encoder layers
        self.dropout = nn.Dropout(p=0.3)  # dropout layer
        self.fc1 = nn.Linear(1024, 256)  # hidden layer 1
        self.fc2 = nn.Linear(256, 64)  # hidden layer 2
        self.fc3 = nn.Linear(64, C.NUM_CLASSES)  # final layer

    def forward(self, x):
        if not self.precomputed:
            output = self.encoder(x)
            pooled_output = output.pooler_output
        else:
            pooled_output = x
        x = torch.relu(self.fc1(pooled_output))
        if self.training:
            x = self.dropout(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x