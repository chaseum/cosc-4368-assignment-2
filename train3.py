import argparse
import copy
import csv
import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn, update_bn
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import transforms

from data_pipeline import create_data_pipeline
from model import CustomDogClassifier, load_model


# ----------------------------------------------------------------------
# Config: everything that defines a run lives here so it is reproducible
# and gets saved into the checkpoint.
# ----------------------------------------------------------------------
CONFIG = {
    "seed": 42,
    "epochs": 60,
    "batch_size": 32,
    "max_lr": 2e-3,
    "weight_decay": 5e-2,
    "dropout": 0.3,
    "label_smoothing": 0.1,
    "pct_start": 0.2,          # fraction of training spent warming up LR
    "grad_clip": 2.0,
    "ema_decay": 0.999,
    "cutmix_prob": 0.5,        # chance a batch gets CutMix
    "cutmix_off_frac": 0.03,   # last 3% of epochs train on clean images (set 0.0 to never turn off)
    "foxhound_weight": 1.5,    # loss weight for class 3 (English foxhound); 1.0 = off
    "patience": 15,            # early stopping on val accuracy
    "num_workers": 4,          # set to 0 if workers cause problems on Windows
}


# ----------------------------------------------------------------------
# CutMix (batch-level augmentation, applied on the GPU)
# ----------------------------------------------------------------------
def cutmix(images, labels, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(images.size(0), device=images.device)
    H, W = images.shape[2:]
    r = np.sqrt(1 - lam)
    ch, cw = int(H * r), int(W * r)
    cy, cx = np.random.randint(H), np.random.randint(W)
    y1, y2 = max(cy - ch // 2, 0), min(cy + ch // 2, H)
    x1, x2 = max(cx - cw // 2, 0), min(cx + cw // 2, W)
    images[:, :, y1:y2, x1:x2] = images[idx, :, y1:y2, x1:x2]
    # recompute lambda from the actual (clipped) box area
    lam = 1 - (y2 - y1) * (x2 - x1) / (H * W)
    return images, labels, labels[idx], lam


# ----------------------------------------------------------------------
# Training / validation
# ----------------------------------------------------------------------
def train_one_epoch(model, ema, train_loader, criterion, optimizer, scheduler,
                    scaler, device, use_cutmix, cfg):

    model.train()

    use_amp = device.type == "cuda"
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device, non_blocking=True).contiguous(
            memory_format=torch.channels_last
        )
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        mixed = use_cutmix and np.random.rand() < cfg["cutmix_prob"]
        if mixed:
            images, y_a, y_b, lam = cutmix(images, labels)

        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=use_amp):
            outputs = model(images)
            if mixed:
                loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
            else:
                loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        scaler.step(optimizer)
        scaler.update()

        # OneCycleLR steps once per batch
        scheduler.step()
        ema.update_parameters(model)

        total_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()   # approximate on CutMix batches
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, val_loader, criterion, device):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in val_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def save_checkpoint(model, path, cfg):
    torch.save({
        "architecture": "deep",
        "dropout": cfg["dropout"],
        "seed": cfg["seed"],
        "config": cfg,
        "state_dict": model.state_dict(),
    }, path)


def train(model, ema, train_loader, val_loader, criterion, optimizer, scheduler,
          scaler, device, cfg, checkpoint_path, full_mode=False):

    epochs = cfg["epochs"]
    val_criterion = nn.CrossEntropyLoss()   # plain CE so val loss is comparable

    best_val_accuracy = 0.0
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    cutmix_stop_epoch = int(epochs * (1 - cfg["cutmix_off_frac"]))
    history = []

    for epoch in range(epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        use_cutmix = epoch < cutmix_stop_epoch

        train_loss, train_accuracy = train_one_epoch(
            model, ema, train_loader, criterion, optimizer, scheduler,
            scaler, device, use_cutmix, cfg
        )

        if full_mode:
            # No clean validation set when val is part of training
            history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "learning_rate": current_lr,
            })
            print(f"Epoch {epoch + 1}/{epochs} | lr {current_lr:.6f} | "
                  f"train loss {train_loss:.4f} | train acc {train_accuracy:.2%}")
            continue

        val_loss, val_accuracy = validate(model, val_loader, val_criterion, device)
        update_bn(train_loader, ema, device=device)
        ema_val_loss, ema_val_accuracy = validate(ema.module, val_loader, val_criterion, device)

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "ema_val_loss": ema_val_loss,
            "ema_val_accuracy": ema_val_accuracy,
            "learning_rate": current_lr,
        })

        print(f"\nEpoch {epoch + 1}/{epochs}  (CutMix {'on' if use_cutmix else 'off'})")
        print(f"Learning Rate: {current_lr:.6f}")
        print(f"Training Loss: {train_loss:.4f} | Training Accuracy: {train_accuracy:.2%}")
        print(f"Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_accuracy:.2%}")
        print(f"EMA Val Loss: {ema_val_loss:.4f} | EMA Val Accuracy: {ema_val_accuracy:.2%}")

        # Pick the better of raw weights vs EMA weights this epoch
        if ema_val_accuracy >= val_accuracy:
            cand_model, cand_acc, cand_loss = ema.module, ema_val_accuracy, ema_val_loss
        else:
            cand_model, cand_acc, cand_loss = model, val_accuracy, val_loss

        improved = (cand_acc > best_val_accuracy or
                    (cand_acc == best_val_accuracy and cand_loss < best_val_loss))

        if improved:
            best_val_accuracy = cand_acc
            best_val_loss = cand_loss
            epochs_without_improvement = 0
            save_checkpoint(cand_model, checkpoint_path, cfg)
            print("New best model saved!")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg["patience"]:
                print(f"Early stopping: no improvement for {cfg['patience']} epochs.")
                break

    # Always keep the final EMA weights too
    last_path = checkpoint_path.replace(".pth", "_last.pth")
    save_checkpoint(ema.module, last_path, cfg)

    if full_mode:
        # No validation signal: the final EMA weights are the submission candidate
        save_checkpoint(ema.module, checkpoint_path, cfg)
        print(f"\nFull-data run finished. Weights saved to {checkpoint_path}")
    else:
        print(f"\nBest Validation Accuracy: {best_val_accuracy:.2%}")

    return history


# ----------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------
@torch.no_grad()
def per_class_report(model, loader, device, n=10):
    model.eval()
    cm = torch.zeros(n, n, dtype=torch.long)
    for x, y in loader:
        p = model(x.to(device)).argmax(1).cpu()
        for t, q in zip(y, p):
            cm[t, q] += 1
    acc = (cm.diag().float() / cm.sum(1).clamp(min=1)).tolist()
    print("\nPer-class accuracy (class id order from classes.txt):")
    for i, a in enumerate(acc):
        print(f"  class {i}: {a:.2%}")
    print("Confusion matrix (rows = true, cols = predicted):")
    print(cm)
    return cm


# ----------------------------------------------------------------------
# Data tweaks done entirely inside train.py (data_pipeline.py is untouched)
# ----------------------------------------------------------------------
def build_train_transform():
    """Stronger augmentation: random resized crop + random erasing."""
    return transforms.Compose([
        transforms.RandomResizedCrop(128, scale=(0.5, 1.0), ratio=(0.8, 1.25)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=8),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])


def make_loader(dataset, batch_size, shuffle, num_workers, seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=g,
    )


@torch.no_grad()
def validate_flip_tta(model, val_loader, device):
    """Val accuracy if predict() averaged logits of each image and its flip."""
    model.eval()
    correct, total = 0, 0
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images) + model(torch.flip(images, dims=[3]))
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return correct / total


# ----------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=CONFIG["seed"])
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    parser.add_argument("--name", type=str, default="scratch_onecycle_cutmix_ema")
    parser.add_argument("--full", action="store_true",
                        help="train on train+val with the validated recipe (no validation)")
    args = parser.parse_args()

    cfg = dict(CONFIG)
    cfg["seed"] = args.seed
    cfg["epochs"] = args.epochs
    experiment_name = f"{args.name}_seed{cfg['seed']}" + ("_full" if args.full else "")

    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("Config:", cfg)

    train_loader, val_loader, _, _ = create_data_pipeline(batch_size=cfg["batch_size"])

    # Swap in stronger augmentation on the training dataset only.
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    train_dataset.transform = build_train_transform()

    if args.full:
        # Combine train + val. The val half gets no augmentation because
        # the val dataset uses plain ToTensor().
        train_dataset = ConcatDataset([train_dataset, val_dataset])

    nw = cfg["num_workers"]
    train_loader = make_loader(train_dataset, cfg["batch_size"], True, nw, cfg["seed"])
    val_loader = make_loader(val_dataset, cfg["batch_size"], False, nw, cfg["seed"])

    # ---- model: FRESH init, not fine-tuning an old checkpoint ----
    model = CustomDogClassifier(dropout=cfg["dropout"]).to(device)
    model = model.to(memory_format=torch.channels_last)

    # EMA copy of the weights (use_buffers=True so BatchNorm stats are averaged too)
    ema = AveragedModel(
        model,
        multi_avg_fn=get_ema_multi_avg_fn(cfg["ema_decay"]),
        use_buffers=False,
    )

    # label smoothing on the training loss only
    class_weights = torch.ones(10)
    class_weights[3] = cfg["foxhound_weight"]
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=cfg["label_smoothing"],
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["max_lr"],
        weight_decay=cfg["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg["max_lr"],
        epochs=cfg["epochs"],
        steps_per_epoch=len(train_loader),
        pct_start=cfg["pct_start"],
    )

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    checkpoint_path = f"{experiment_name}.pth"

    history = train(
        model, ema, train_loader, val_loader, criterion, optimizer, scheduler,
        scaler, device, cfg, checkpoint_path, full_mode=args.full,
    )

    with open(f"{experiment_name}.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    # ---- sanity check: reload the saved file through load_model, like the harness does ----
    reloaded = load_model(checkpoint_path).to(device)
    if not args.full:
        _, reloaded_acc = validate(reloaded, val_loader, nn.CrossEntropyLoss(), device)
        print(f"Reloaded checkpoint val accuracy: {reloaded_acc:.2%}")
        tta_acc = validate_flip_tta(reloaded, val_loader, device)
        print(f"Val accuracy with flip TTA (needs predict() change in model.py): {tta_acc:.2%}")
        per_class_report(reloaded, val_loader, device)


if __name__ == "__main__":
    main()