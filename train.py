import argparse
import csv
import json
import os
from pathlib import Path
import random
import time

# CUDA matrix operations need this setting for deterministic execution.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from data_pipeline import AUGMENTATIONS, PROJECT_ROOT, compute_rgb_statistics, create_data_pipeline, read_class_names
from model import ARCHITECTURES, CustomDogClassifier, load_model, predict


def main():
    args = parse_arguments()
    seed_everything(args.seed)
    device = torch.device(args.device)
    print(f"Device: {torch.cuda.get_device_name() if device.type == 'cuda' else 'CPU'}", flush=True)
    class_names = read_class_names(PROJECT_ROOT / "imagewoof_train/classes.txt")
    if args.evaluate or args.init_from:
        model = load_model(args.evaluate or args.init_from).to(device)
        args.normalization = model.normalization
        args.architecture, args.dropout = model.architecture, model.network[-2].p
    elif args.normalize:
        print("Computing RGB statistics from unaugmented training images only...", flush=True)
        args.normalization = compute_rgb_statistics(PROJECT_ROOT / "imagewoof_train/train")
        print(f"RGB normalization: {args.normalization}", flush=True)
    else:
        args.normalization = None
    train_loader, val_loader, raw_images, raw_labels = create_data_pipeline(
        args.batch_size, args.augmentation, args.normalization, args.workers, args.train_on_all,
    )
    if args.evaluate:
        predictions = predict(model, raw_images)
        confusion = torch.bincount(
            torch.from_numpy(raw_labels * 10 + predictions), minlength=100,
        ).reshape(10, 10)
        print(f"Validation accuracy through predict: {(predictions == raw_labels).mean():.4%}")
        print(json.dumps(class_metrics(confusion, class_names), indent=2))
        return
    if args.overfit and args.overfit > len(train_loader.dataset):
        raise ValueError("--overfit exceeds the training dataset size")
    args.output.mkdir(parents=True, exist_ok=False)
    if not args.init_from:
        model = CustomDogClassifier(args.architecture, args.dropout, args.normalization).to(device)
    if args.smoke_test or args.overfit:
        count = args.overfit or 32
        indices = torch.randperm(len(train_loader.dataset))[:count].tolist()
        if args.overfit:
            train_loader.dataset.transform = val_loader.dataset.transform
            for layer in model.modules():
                if isinstance(layer, nn.Dropout):
                    layer.p = 0.0
        subset = Subset(train_loader.dataset, indices)
        train_loader = DataLoader(subset, batch_size=args.batch_size, shuffle=True)
        validation = subset if args.overfit else Subset(val_loader.dataset, range(32))
        val_loader = DataLoader(validation, batch_size=args.batch_size)
        print("DEBUG RUN: metrics are not full held-out validation results", flush=True)
    if args.smoke_test:
        args.epochs = 1
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(torch_version=str(torch.__version__), numpy_version=np.__version__,
                  device_name=torch.cuda.get_device_name() if device.type == "cuda" else "CPU",
                  parameters=sum(parameter.numel() for parameter in model.parameters()),
                  class_names=class_names)
    (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    train_model(model, train_loader, val_loader, args, class_names)



def parse_arguments():
    parser = argparse.ArgumentParser(description="Train or evaluate an Imagewoof CNN")
    parser.add_argument("--architecture", choices=ARCHITECTURES, default="baseline")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--augmentation", choices=AUGMENTATIONS, default="original")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--normalize", action="store_true", help="Standardize RGB using training-set mean/std")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8, help="Training data-loading processes")
    parser.add_argument("--train-on-all", action="store_true",
                        help="Final fit on train+val; saves the last epoch because no held-out data remains")
    parser.add_argument("--output", type=Path, default=Path("runs/baseline"))
    debug = parser.add_mutually_exclusive_group()
    debug.add_argument("--smoke-test", action="store_true", help="One epoch on 32 train/val images")
    debug.add_argument("--overfit", type=int, metavar="N", help="Fit N training images without augmentation/dropout")
    parser.add_argument("--evaluate", type=Path, help="Evaluate a checkpoint through predict on all validation images")
    parser.add_argument("--init-from", type=Path,
                        help="Fine-tune this checkpoint; its architecture, dropout and normalization are reused")
    args = parser.parse_args()
    if args.evaluate and args.init_from:
        parser.error("--evaluate cannot be combined with --init-from")
    if min(args.epochs, args.batch_size, args.patience) < 1 or args.lr <= 0 or args.weight_decay < 0:
        parser.error("epochs, batch-size, patience and lr must be positive; weight-decay cannot be negative")
    if args.overfit is not None and args.overfit < 1:
        parser.error("--overfit must be positive")
    if args.workers < 0:
        parser.error("--workers cannot be negative")
    if args.train_on_all and args.overfit:
        parser.error("--train-on-all cannot be combined with --overfit")
    if not 0 <= args.dropout < 1:
        parser.error("--dropout must be at least 0 and less than 1")
    if not 0 <= args.label_smoothing < 1:
        parser.error("--label-smoothing must be at least 0 and less than 1")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this Python. Install CUDA PyTorch or explicitly use --device cpu")
    return args



def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)



def train_model(model, train_loader, val_loader, args, class_names):
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    training_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    validation_criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    if args.train_on_all:
        print("Validation images are being trained on; logged validation accuracy is NOT held out.", flush=True)
    best_accuracy = -1.0
    stale_epochs = 0
    fields = ["epoch", "lr", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "seconds"]
    with (args.output / "history.csv").open("w", newline="", encoding="utf-8") as log:
        writer = csv.DictWriter(log, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            started = time.perf_counter()
            lr = optimizer.param_groups[0]["lr"]
            train_loss, train_accuracy, _ = run_epoch(model, train_loader, device, training_criterion, optimizer, scaler)
            val_loss, val_accuracy, confusion = run_epoch(model, val_loader, device, validation_criterion)
            row = {
                "epoch": epoch,
                "lr": lr,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "seconds": time.perf_counter() - started,
            }
            writer.writerow(row)
            log.flush()
            print(f"Epoch {epoch:03d}/{args.epochs}: train {train_loss:.4f}/{train_accuracy:.2%} "
                  f"val {val_loss:.4f}/{val_accuracy:.2%} lr={lr:.2g} {row['seconds']:.1f}s", flush=True)
            if val_accuracy > best_accuracy or args.train_on_all:
                best_accuracy = val_accuracy
                stale_epochs = 0
                temporary = args.output / "model.tmp"
                torch.save({"architecture": args.architecture, "dropout": args.dropout,
                            "normalization": args.normalization,
                            "augmentation": args.augmentation, "label_smoothing": args.label_smoothing,
                            "state_dict": model.state_dict(),
                            "seed": args.seed, "epoch": epoch, "val_accuracy": val_accuracy,
                            "class_names": class_names}, temporary)
                temporary.replace(args.output / "model.pth")
                metrics = {**row, "debug_run": bool(args.smoke_test or args.overfit),
                           "classes": class_metrics(confusion, class_names),
                           "confusion_matrix": confusion.tolist()}
                (args.output / "best_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            else:
                stale_epochs += 1
            scheduler.step()
            if stale_epochs >= args.patience:
                print(f"Early stopping after {args.patience} epochs without improved validation accuracy")
                break
    if args.train_on_all:
        print(f"Saved final-epoch weights: {args.output / 'model.pth'}")
    else:
        print(f"Best accuracy: {best_accuracy:.2%}; checkpoint: {args.output / 'model.pth'}")



def run_epoch(model, loader, device, criterion, optimizer=None, scaler=None):
    model.train(optimizer is not None)
    # Validation stays float32 so its accuracy matches predict.
    mixed_precision = scaler is not None and scaler.is_enabled()
    total_loss = 0.0
    confusion = torch.zeros(10, 10, dtype=torch.int64)
    with torch.set_grad_enabled(optimizer is not None):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=mixed_precision):
                logits = model(images)
                loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss; stop and check inputs and learning rate")
            if optimizer is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * len(labels)
            indices = (labels * 10 + logits.argmax(dim=1)).detach().cpu()
            confusion += torch.bincount(indices, minlength=100).reshape(10, 10)
    count = confusion.sum().item()
    return total_loss / count, confusion.diag().sum().item() / count, confusion



def class_metrics(confusion, class_names):
    metrics = {}
    for index, name in enumerate(class_names):
        support = confusion[index].sum().item()
        metrics[name] = {
            "support": support,
            "recall": confusion[index, index].item() / support if support else None,
        }
    return metrics



if __name__ == "__main__":
    main()
