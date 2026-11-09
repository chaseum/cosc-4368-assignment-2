"""Queue of OneCycle + CutMix/Mixup + EMA experiments.

python train_recipe.py --seed 42 dexter cutmix_end
python train_recipe.py --seed 42 --train-on-all cutmix_end
Each run writes runs/sweep/<name>_seed<seed>/{model.pth,last.pth,history.csv}; finished runs are skipped.
model.pth is the best-validation EMA epoch, last.pth the final EMA. Summaries go to runs/sweep/results.csv.
"""
import argparse
import csv
from itertools import islice
import math
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn, update_bn

from data_pipeline import create_data_pipeline
from model import CustomDogClassifier, load_model, predict
from train import run_epoch, seed_everything

BASE = dict(architecture="deep", augmentation="original", epochs=60, batch_size=32, lr=2e-3, weight_decay=5e-2,
            dropout=0.3, label_smoothing=0.1, pct_start=0.2, grad_clip=2.0, ema_decay=0.999,
            cutmix=0.5, mixup=0.0, off_frac=0.1, optimizer="adamw", schedule="onecycle")
END = dict(off_frac=0.0)
EXPERIMENTS = {
    "dexter": {},
    "cutmix_end": END,
    "cutmix_always": {**END, "cutmix": 1.0},
    "cutmix_mixup": {**END, "cutmix": 0.5, "mixup": 0.5},
    "mixup_only": {**END, "cutmix": 0.0, "mixup": 0.5},
    "no_mixing": {**END, "cutmix": 0.0},
    "mish": {**END, "architecture": "deep_mish"},
    "blur": {**END, "architecture": "deep_blur"},
    "randaug": {**END, "augmentation": "randaug"},
    "radam_flatcos": {**END, "optimizer": "radam", "schedule": "flat_cosine", "lr": 1e-3},
    "wide": {**END, "architecture": "wide_2233"},
    "wide_randaug": {**END, "architecture": "wide_2233", "augmentation": "randaug"},
    "wider": {**END, "architecture": "wide_22333"},
}


def mix_batch(images, labels, cutmix, mixup):
    """Return mixed images, both label sets and the weight of the original labels."""
    draw = np.random.rand()
    if draw >= cutmix + mixup:
        return images, labels, labels, 1.0
    other = torch.randperm(len(images), device=images.device)
    if draw < cutmix:
        height, width = images.shape[2:]
        side = math.sqrt(1 - np.random.beta(1.0, 1.0))
        box_h, box_w = int(height * side), int(width * side)
        y, x = np.random.randint(height), np.random.randint(width)
        y1, y2, x1, x2 = max(y - box_h // 2, 0), min(y + box_h // 2, height), max(x - box_w // 2, 0), min(x + box_w // 2, width)
        images[:, :, y1:y2, x1:x2] = images[other, :, y1:y2, x1:x2]
        return images, labels, labels[other], 1 - (y2 - y1) * (x2 - x1) / (height * width)
    weight = float(np.random.beta(0.8, 0.8))
    return weight * images + (1 - weight) * images[other], labels, labels[other], weight


def make_training_tools(model, cfg, steps_per_epoch):
    optimizer = (torch.optim.RAdam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"],
                                   decoupled_weight_decay=True) if cfg["optimizer"] == "radam"
                 else torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]))
    total = cfg["epochs"] * steps_per_epoch
    if cfg["schedule"] == "flat_cosine":
        flat = int(0.75 * total)
        return optimizer, torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: 1.0 if step < flat else 0.5 * (1 + math.cos(math.pi * (step - flat) / (total - flat))))
    return optimizer, torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg["lr"], total_steps=total, pct_start=cfg["pct_start"])


def save(model, path, cfg, seed, epoch, accuracy):
    torch.save({"architecture": cfg["architecture"], "dropout": cfg["dropout"], "normalization": None,
                "state_dict": model.state_dict(), "seed": seed, "epoch": epoch, "val_accuracy": accuracy,
                "config": cfg}, path)


def run_experiment(name, seed, train_on_all, workers):
    cfg = {**BASE, **EXPERIMENTS[name]}
    output = Path("runs/sweep") / f"{name}_seed{seed}{'_all' if train_on_all else ''}"
    if (output / "last.pth").exists():
        print(f"Skipping finished run {output}", flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    device = torch.device("cuda")
    train_loader, val_loader, raw_images, raw_labels = create_data_pipeline(
        cfg["batch_size"], cfg["augmentation"], None, workers, train_on_all)
    model = CustomDogClassifier(cfg["architecture"], cfg["dropout"]).to(device)
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(cfg["ema_decay"]))
    optimizer, scheduler = make_training_tools(model, cfg, len(train_loader))
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    val_criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda")
    mixing_stops = int(cfg["epochs"] * (1 - cfg["off_frac"]))
    best, best_epoch, started = -1.0, 0, time.perf_counter()
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as log:
        writer = csv.writer(log)
        writer.writerow(["epoch", "lr", "train_loss", "raw_val", "ema_val", "seconds"])
        for epoch in range(1, cfg["epochs"] + 1):
            epoch_started = time.perf_counter()
            lr, total_loss, seen = optimizer.param_groups[0]["lr"], 0.0, 0
            model.train()
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                if epoch <= mixing_stops:
                    images, first, second, weight = mix_batch(images, labels, cfg["cutmix"], cfg["mixup"])
                else:
                    first, second, weight = labels, labels, 1.0
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda"):
                    logits = model(images)
                    loss = weight * criterion(logits, first) + (1 - weight) * criterion(logits, second)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite loss")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                ema.update_parameters(model)
                total_loss += loss.item() * len(labels)
                seen += len(labels)
            if train_on_all and epoch < cfg["epochs"]:
                print(f"{name} {epoch}/{cfg['epochs']} loss {total_loss / seen:.4f}", flush=True)
                continue
            _, raw_accuracy, _ = run_epoch(model, val_loader, device, val_criterion)
            # ponytail: 2,048 images estimate BatchNorm well and save a full extra data pass.
            update_bn(islice(train_loader, 64), ema, device=device)
            _, ema_accuracy, _ = run_epoch(ema.module, val_loader, device, val_criterion)
            seconds = time.perf_counter() - epoch_started
            writer.writerow([epoch, lr, total_loss / seen, raw_accuracy, ema_accuracy, seconds])
            log.flush()
            print(f"{name} {epoch}/{cfg['epochs']} loss {total_loss / seen:.4f} raw {raw_accuracy:.2%} "
                  f"ema {ema_accuracy:.2%} {seconds:.0f}s", flush=True)
            if ema_accuracy > best and not train_on_all:
                best, best_epoch = ema_accuracy, epoch
                save(ema.module, output / "model.pth", cfg, seed, epoch, ema_accuracy)
    save(ema.module, output / "last.tmp", cfg, seed, cfg["epochs"], ema_accuracy)
    (output / "last.tmp").replace(output / "last.pth")
    if train_on_all:
        print(f"{name}: final train+val weights in {output / 'last.pth'} (no held-out score)", flush=True)
        return
    flip = {path: (predict(load_model(output / path).to(device), raw_images) == raw_labels).mean()
            for path in ("model.pth", "last.pth")}
    row = {"name": name, "seed": seed, "best_ema": best, "best_epoch": best_epoch, "last_ema": ema_accuracy,
           "best_flip": flip["model.pth"], "last_flip": flip["last.pth"],
           "minutes": (time.perf_counter() - started) / 60}
    results = Path("runs/sweep/results.csv")
    is_new = not results.exists()
    with results.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=row)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
    print(f"RESULT {row}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="+", choices=EXPERIMENTS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--train-on-all", action="store_true")
    args = parser.parse_args()
    for name in args.names:
        try:
            run_experiment(name, args.seed, args.train_on_all, args.workers)
        except Exception:  # A failed run must not stop the rest of an unattended queue.
            print(f"FAILED {name}", flush=True)
            traceback.print_exc()


if __name__ == "__main__":
    main()
