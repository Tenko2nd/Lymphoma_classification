"""
    This is where the Model I use is initialized.
    It's using pytorch.nn.Module class and transformers.AutoModel for HuggingFace pretrained model.
    Optional, precomputed, used only if we use the precomuted version of an image instead (if the image has been saved as an numpy file at the exit of the encoder)
    It consist of:
        - The pretrained model phikon-v2
        - A MLP of 1024 → 128 → 32 → C.NUM_CLASSES
"""

from torch import Tensor
from transformers import AutoModel
import torch.nn as nn

import constant as C


class MyModel(nn.Module):
    def __init__(self, precomputed=False) -> None:
        self.precomputed = precomputed
        super(MyModel, self).__init__()
        if not precomputed:
            self.encoder = AutoModel.from_pretrained("owkin/phikon-v2")
            self.encoder.requires_grad_(False)  # Freeze the encoder layers
        self.dropout = nn.Dropout(p=0.3)  # dropout layer
        self.fc1 = nn.Linear(1024, 128)  # hidden layer 1
        self.fc2 = nn.Linear(128, 32)  # hidden layer 2
        self.fc3 = nn.Linear(32, C.NUM_CLASSES)  # final layer

    def forward(self, x) -> Tensor:
        if not self.precomputed:
            output = self.encoder(x)
            pooled_output = output.pooler_output
        else:
            pooled_output = x
        x = self.fc1(pooled_output)
        if self.training:
            x = self.dropout(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
