import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ARCHITECTURES = ("baseline", "deep", "deep_spatial", "deep_pointwise", "deep_2233", "wide_2233", "wide_22333",
                 "deep_mish", "deep_blur", "deep_blur_mish")
CHANNELS = {"baseline": (32, 64, 128), "wide_2233": (64, 128, 256, 512), "wide_22333": (64, 128, 256, 512, 512)}
CONVOLUTIONS = {"deep_2233": (2, 2, 3, 3), "wide_2233": (2, 2, 3, 3), "wide_22333": (2, 2, 3, 3, 3)}


class MaxBlurPool(nn.Module):
    """Max pooling that blurs with a fixed, untrained kernel before downsampling by 2."""

    def __init__(self):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2, stride=1)
        binomial = torch.tensor([1.0, 2.0, 1.0])
        self.register_buffer("kernel", (binomial[None] * binomial[:, None])[None, None] / 16, persistent=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = F.pad(self.maxpool(images), (1, 1, 1, 1), mode="reflect")
        return F.conv2d(images, self.kernel.expand(images.shape[1], -1, -1, -1), stride=2, groups=images.shape[1])


class CustomDogClassifier(nn.Module):
    def __init__(self, architecture: str = "baseline", dropout: float = 0.3, normalization=None):
        super().__init__()
        if architecture not in ARCHITECTURES:
            raise ValueError(f"architecture must be one of {ARCHITECTURES}")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be at least 0 and less than 1")
        if normalization is not None:
            mean = np.asarray(normalization["mean"], dtype=np.float64)
            std = np.asarray(normalization["std"], dtype=np.float64)
            if (mean.shape != (3,) or std.shape != (3,) or
                    not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0)):
                raise ValueError("Normalization requires three finite means and three positive finite std values")
        self.normalization = normalization
        self.architecture = architecture
        channels = CHANNELS.get(architecture, (32, 64, 128, 256))
        convolutions = CONVOLUTIONS.get(architecture, (2,) * len(channels))
        layers = []
        in_channels = 3
        for block_index, out_channels in enumerate(channels):
            for _ in range(convolutions[block_index]):
                layers.extend([
                    nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.Mish(inplace=True) if "mish" in architecture else nn.ReLU(inplace=True),
                ])
                in_channels = out_channels
            layers.append(MaxBlurPool() if "blur" in architecture else nn.MaxPool2d(2))
        if architecture == "deep_pointwise":
            layers.extend([
                nn.Conv2d(256, 256, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            ])
        pooled_size = 2 if architecture == "deep_spatial" else 1
        # Fixed pooling avoids nondeterministic adaptive 2x2 pooling backward on CUDA.
        pooling = nn.AvgPool2d(4) if pooled_size == 2 else nn.AdaptiveAvgPool2d(1)
        layers.extend([
            pooling,
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1] * pooled_size ** 2, 10),
        ])
        self.network = nn.Sequential(*layers)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)


def load_model(path="model.pth"):
    """Load a training checkpoint on CPU and return the matching model in eval mode."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = CustomDogClassifier(checkpoint["architecture"], checkpoint.get("dropout", 0.3),
                                checkpoint.get("normalization"))
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval()


@torch.inference_mode()
def predict(model, images):
    """Convert raw NHWC uint8 RGB images into class IDs in classes.txt order."""
    if not isinstance(images, np.ndarray) or images.dtype != np.uint8:
        raise TypeError("images must be a uint8 numpy array")
    if images.ndim != 4 or images.shape[1:] != (128, 128, 3):
        raise ValueError("images must have shape (N, 128, 128, 3)")
    model.eval()
    device = next(model.parameters()).device
    normalization = getattr(model, "normalization", None)
    if normalization is not None:
        mean = torch.tensor(normalization["mean"], dtype=torch.float32, device=device).view(1, 3, 1, 1)
        std = torch.tensor(normalization["std"], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    predictions = []
    for start in range(0, len(images), 32):
        batch = torch.from_numpy(np.ascontiguousarray(images[start:start + 32]))
        batch = batch.permute(0, 3, 1, 2).to(device=device, dtype=torch.float32) / 255.0
        if normalization is not None:
            batch = (batch - mean) / std
        # Average the image and its mirror; one model, no extra data.
        logits = model(batch) + model(batch.flip(dims=(3,)))
        predictions.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(predictions) if predictions else np.empty(0, dtype=np.int64)
