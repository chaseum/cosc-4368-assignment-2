import numpy as np
import torch
from torch import nn


class CustomDogClassifier(nn.Module):
    def __init__(self, dropout: float = 0.3):
        super().__init__()
        layers = []
        in_channels = 3
        for out_channels in (32, 64, 128, 256):
            for _ in range(2):
                layers.extend([
                    nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ])
                in_channels = out_channels
            layers.append(nn.MaxPool2d(2))
        layers.extend([
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 10),
        ])
        self.network = nn.Sequential(*layers)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)


def load_model(path="model.pth"):
    """Load a training checkpoint on the available device and return eval model."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["architecture"] != "deep":
        raise ValueError("This model file loads only the selected deep CNN checkpoint")
    model = CustomDogClassifier(checkpoint.get("dropout", 0.3))
    model.load_state_dict(checkpoint["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device).eval()


@torch.inference_mode()
def predict(model, images):
    """Convert raw NHWC uint8 RGB images into class IDs in classes.txt order."""
    if not isinstance(images, np.ndarray) or images.dtype != np.uint8:
        raise TypeError("images must be a uint8 numpy array")
    if images.ndim != 4 or images.shape[1:] != (128, 128, 3):
        raise ValueError("images must have shape (N, 128, 128, 3)")
    model.eval()
    device = next(model.parameters()).device
    predictions = []
    for start in range(0, len(images), 32):
        batch = torch.from_numpy(np.ascontiguousarray(images[start:start + 32]))
        batch = batch.permute(0, 3, 1, 2).to(device=device, dtype=torch.float32) / 255.0
        original_logits = model(batch)
        flipped_logits = model(torch.flip(batch, dims=[3]))
        predictions.append((original_logits + flipped_logits).argmax(dim=1).cpu().numpy())
    return np.concatenate(predictions) if predictions else np.empty(0, dtype=np.int64)
