from pathlib import Path
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import ConcatDataset, DataLoader
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
AUGMENTATIONS = ("original", "crop", "crop_erase", "strong")
TRAIN_MEAN_RGB = (128, 118, 102)  # Rotation corners blend in instead of turning black.

def read_class_names(classes_file: Path) -> list[str]:
    names_by_id = {}

    for line in classes_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        class_id, class_name = line.split(maxsplit=1)
        names_by_id[int(class_id)] = class_name

    return [names_by_id[index] for index in range(len(names_by_id))]

def compute_rgb_statistics(train_folder: Path):
    dataset = ImageFolder(train_folder)
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_square_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0
    for image, _ in dataset:
        pixels = np.asarray(image, dtype=np.float64) / 255.0
        if pixels.shape != (128, 128, 3):
            raise ValueError("Training images must be 128 x 128 RGB")
        channel_sum += pixels.sum(axis=(0, 1))
        channel_square_sum += np.square(pixels).sum(axis=(0, 1))
        pixel_count += pixels.shape[0] * pixels.shape[1]
    mean = channel_sum / pixel_count
    variance = np.maximum(channel_square_sum / pixel_count - np.square(mean), 0.0)
    std = np.sqrt(variance)
    if np.any(std <= 0):
        raise ValueError("Each training RGB channel must have nonzero standard deviation")
    return {"mean": mean.tolist(), "std": std.tolist()}


def create_train_transform(augmentation: str = "original", normalization=None):
    if augmentation not in AUGMENTATIONS:
        raise ValueError(f"augmentation must be one of {AUGMENTATIONS}")
    if augmentation == "strong":
        operations = [
            transforms.RandomRotation(10, transforms.InterpolationMode.BILINEAR, fill=TRAIN_MEAN_RGB),
            transforms.RandomResizedCrop(128, scale=(0.5, 1.0), ratio=(0.75, 4 / 3), antialias=True),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.ToTensor(),
        ]
        if normalization is not None:
            operations.append(transforms.Normalize(normalization["mean"], normalization["std"]))
        operations.append(transforms.RandomErasing(p=0.25, value="random"))
        return transforms.Compose(operations)
    operations = []
    if augmentation != "original":
        operations.append(transforms.RandomResizedCrop(
            128, scale=(0.82, 1.0), ratio=(0.90, 1.10), antialias=True,
        ))
    operations.extend([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=8),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
    ])
    if augmentation == "crop_erase":
        operations.append(transforms.RandomErasing(
            p=0.12, scale=(0.01, 0.05), ratio=(0.7, 1.4), value="random",
        ))
    if normalization is not None:
        operations.append(transforms.Normalize(normalization["mean"], normalization["std"]))
    return transforms.Compose(operations)


def create_data_pipeline(batch_size: int = 16, augmentation: str = "original", normalization=None,
                         num_workers: int = 0, include_val: bool = False):
    """Build the data loaders and raw validation arrays."""
    classes_file = PROJECT_ROOT / "imagewoof_train" / "classes.txt"
    class_names = read_class_names(classes_file)
    assignment_class_to_idx = {
         class_name: index for index, class_name in enumerate(class_names)
    }
    
    train_transform = create_train_transform(augmentation, normalization)
    val_transform = transforms.ToTensor()
    if normalization is not None:
        val_transform = transforms.Compose([
            val_transform, transforms.Normalize(normalization["mean"], normalization["std"]),
        ])

    train_folder = PROJECT_ROOT / "imagewoof_train" / "train"
    train_dataset = ImageFolder(train_folder, transform=train_transform)
    folder_idx_to_assignment_idx = {
        folder_idx: assignment_class_to_idx[class_name]
        for class_name, folder_idx in train_dataset.class_to_idx.items()
    }
    train_dataset.target_transform = folder_idx_to_assignment_idx.get
    
    val_folder = PROJECT_ROOT / "imagewoof_val" / "val"
    val_dataset = ImageFolder(val_folder, transform=val_transform)
    
    if val_dataset.class_to_idx != train_dataset.class_to_idx:
        raise ValueError("Training and validation class folders do not match")
    
    val_dataset.target_transform = folder_idx_to_assignment_idx.get
    
    raw_val_dataset = ImageFolder(val_folder)
    raw_val_dataset.target_transform = folder_idx_to_assignment_idx.get

    raw_val_images = []
    raw_val_labels = []

    for image, label in raw_val_dataset:
        raw_val_images.append(np.asarray(image, dtype=np.uint8))
        raw_val_labels.append(label)

    val_images = np.stack(raw_val_images)
    val_labels = np.asarray(raw_val_labels, dtype=np.int64)

    if include_val:
        train_dataset = ConcatDataset([train_dataset, ImageFolder(
            val_folder, transform=train_transform, target_transform=folder_idx_to_assignment_idx.get,
        )])

    train_loader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = num_workers,
        persistent_workers = num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = 0,
    )

    return train_loader, val_loader, val_images, val_labels


if __name__ == "__main__":
    train_loader, val_loader, raw_val_images, raw_val_labels = create_data_pipeline()

    print("Raw validation array:", raw_val_images.shape, raw_val_images.dtype)
    print("Raw validation labels:", raw_val_labels.shape, raw_val_labels.dtype, raw_val_labels.min(), raw_val_labels.max())

    train_images, train_labels = next(iter(train_loader))
    val_images, val_labels = next(iter(val_loader))

    print("Training batch:", train_images.shape, train_labels.shape)
    print("Training label range:", train_labels.min().item(), train_labels.max().item())
    print("Validation batch:", val_images.shape, val_labels.shape)
    print("Validation label range:", val_labels.min().item(), val_labels.max().item())

    validation_class_ids = set()
    for _, labels in val_loader:
        validation_class_ids.update(labels.tolist())

    print("Validation class IDs:", sorted(validation_class_ids))
