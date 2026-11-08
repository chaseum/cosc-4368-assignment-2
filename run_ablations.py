import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parent

# Fixed manual ablations: compare crop + smoothing against crop alone.
EXPERIMENTS = (
    ("deep_crop", "deep", "crop", 0.0),
    ("deep_spatial_control", "deep_spatial", "original", 0.0),
    ("deep_crop_smooth", "deep", "crop", 0.05),
)


def main():
    parser = argparse.ArgumentParser(description="Run three predefined CNN ablations sequentially")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.patience) < 1:
        parser.error("epochs, batch-size and patience must be positive")
    output = args.output or PROJECT_ROOT / "runs" / f"ablations-{datetime.now():%Y%m%d-%H%M%S-%f}"
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for filename in ("model.py", "data_pipeline.py", "train.py", "run_ablations.py"):
        shutil.copy2(PROJECT_ROOT / filename, output / filename)
    print(f"Results: {output}", flush=True)
    print("Three fixed trials; no adaptive tuning or automatic model selection.", flush=True)
    references = [] if args.smoke_test else [
        PROJECT_ROOT / "runs/baseline", PROJECT_ROOT / "runs/deep",
    ]
    runs = []
    for name, architecture, augmentation, label_smoothing in EXPERIMENTS:
        run_folder = output / name
        command = [
            sys.executable, "-u", str(PROJECT_ROOT / "train.py"),
            "--architecture", architecture, "--dropout", "0.3",
            "--augmentation", augmentation, "--label-smoothing", str(label_smoothing),
            "--device", args.device, "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size), "--patience", str(args.patience),
            "--seed", str(args.seed), "--output", str(run_folder),
        ]
        if args.smoke_test:
            command.append("--smoke-test")
        print(f"\nStarting {name} ({architecture}, {augmentation}, smoothing={label_smoothing})", flush=True)
        status = "interrupted"
        try:
            result = subprocess.run(command, cwd=PROJECT_ROOT)
            status = "completed" if result.returncode == 0 else "failed"
        finally:
            runs.append((run_folder, status))
            write_comparison(output, runs, references)
        if result.returncode:
            raise SystemExit(f"{name} failed (exit {result.returncode}); stopped. Results remain in {output}")
    print(f"\nComparison: {output / 'comparison.md'}", flush=True)


def write_comparison(output, runs, references):
    rows = []
    for folder, status in [(folder, "historical") for folder in references] + runs:
        metrics_path = folder / "best_metrics.json"
        if status == "historical" and not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
        config_path = folder / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        history_path = folder / "history.csv"
        history = []
        if history_path.exists():
            with history_path.open(newline="", encoding="utf-8") as log:
                history = list(csv.DictReader(log))
        if status == "historical" and metrics.get("debug_run"):
            continue
        accuracy = metrics.get("val_accuracy")
        rows.append({
            "run": folder.name,
            "status": status,
            "architecture": config.get("architecture", ""),
            "dropout": config.get("dropout", "unrecorded"),
            "augmentation": config.get("augmentation", "unrecorded"),
            "label_smoothing": config.get("label_smoothing", "unrecorded"),
            "val_accuracy_pct": round(accuracy * 100, 4) if accuracy is not None else "",
            "best_epoch": metrics.get("epoch", ""),
            "val_loss": metrics.get("val_loss", ""),
            "parameters": config.get("parameters", ""),
            "epochs_completed": len(history),
            "minutes": round(sum(float(epoch["seconds"]) for epoch in history) / 60, 2),
            "debug_run": metrics.get("debug_run", False),
            "checkpoint": (folder / "model.pth").as_posix() if (folder / "model.pth").exists() else "",
        })
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as log:
        writer = csv.DictWriter(log, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Fixed CNN ablation results", "",
        "Each new trial starts from random weights. Compare crop + smoothing against crop alone.",
        "Historical runs are context, not a newly replicated control. Missing settings are unrecorded.",
        "Smoke results are debug checks, not full validation accuracy. Runs appear in execution order.", "",
        "| Run | Status | Augmentation | Smoothing | Best validation % | Epoch | Minutes | Debug |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(f"| {row['run']} | {row['status']} | {row['augmentation']} | "
                     f"{row['label_smoothing']} | {row['val_accuracy_pct']} | "
                     f"{row['best_epoch']} | {row['minutes']} | {row['debug_run']} |")
    lines.extend(["", "Existing run folders are never overwritten; no model is automatically promoted.",
                  "Compare per-class metrics before choosing a submission."])
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[6:]), flush=True)


if __name__ == "__main__":
    main()
