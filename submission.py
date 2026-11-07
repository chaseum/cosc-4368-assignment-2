
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


class DogCNN(nn.Module):
    def __init__(self):
        super().__init__()

        def block(in_ch, out_ch):
            return [
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ]

        self.net = nn.Sequential(
            *block(3, 32),
            *block(32, 64),
            *block(64, 128),
            *block(128, 256),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.net(x)


def load_model(path="model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DogCNN()

    path = Path(path)
    if not path.exists() and not path.is_absolute():
        path = Path(__file__).resolve().parent / path

    weights = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(weights)

    model.to(device)
    model.eval()

    return model


def predict(model, images):
    model.eval()
    device = next(model.parameters()).device

    predictions = []

    with torch.no_grad():
        for i in range(0, len(images), 128):
            batch = images[i:i + 128]

            batch = np.ascontiguousarray(batch)
            batch = torch.from_numpy(batch).float() / 255.0
            batch = batch.permute(0, 3, 1, 2).to(device)

            outputs = model(batch)
            predicted = torch.argmax(outputs, dim=1)

            predictions.extend(predicted.cpu().numpy())

    return np.array(predictions, dtype=np.int64)
