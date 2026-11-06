from pathlib import Path
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent

def read_class_names(classes_file: Path) -> list[str]:
    names_by_id = {}

    for line in classes_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        class_id, class_name = line.split(maxsplit=1)
        names_by_id[int(class_id)] = class_name

    return [names_by_id[index] for index in range(len(names_by_id))]

def create_data_pipeline(batch_size: int = 16):
    """Build the data loaders and raw validation arrays."""
    classes_file = PROJECT_ROOT / "imagewoof_train" / "classes.txt"
    class_names = read_class_names(classes_file)
    assignment_class_to_idx = {
         class_name: index for index, class_name in enumerate(class_names)
    }
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=8),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1,),
        transforms.ToTensor(),
    ])
    val_transform = transforms.ToTensor()

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

    train_loader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 0,
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