import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_ablations


class AblationTests(unittest.TestCase):
    def test_launcher_runs_fixed_trials_and_writes_comparison(self):
        def finish_trial(command, cwd):
            folder = Path(command[command.index("--output") + 1])
            folder.mkdir()
            config = {
                "architecture": command[command.index("--architecture") + 1],
                "dropout": float(command[command.index("--dropout") + 1]),
                "augmentation": command[command.index("--augmentation") + 1],
                "label_smoothing": float(command[command.index("--label-smoothing") + 1]),
            }
            (folder / "config.json").write_text(json.dumps(config))
            (folder / "best_metrics.json").write_text(json.dumps({
                "epoch": 1, "val_accuracy": 0.8, "val_loss": 0.7, "debug_run": True,
            }))
            (folder / "history.csv").write_text("epoch,seconds\n1,30\n")
            (folder / "model.pth").write_bytes(b"test fixture")
            return subprocess.CompletedProcess(command, 0)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "suite"
            arguments = ["run_ablations.py", "--output", str(output), "--smoke-test", "--device", "cpu"]
            with patch.object(sys, "argv", arguments), \
                    patch("run_ablations.subprocess.run", side_effect=finish_trial) as launch:
                run_ablations.main()
            self.assertEqual(launch.call_count, 3)
            with (output / "comparison.csv").open(newline="") as comparison:
                rows = list(csv.DictReader(comparison))
            self.assertEqual([row["run"] for row in rows], [trial[0] for trial in run_ablations.EXPERIMENTS])
            self.assertEqual([row["dropout"] for row in rows], ["0.3", "0.3", "0.3"])
            self.assertEqual([row["augmentation"] for row in rows], ["crop", "original", "crop"])
            self.assertEqual([row["label_smoothing"] for row in rows], ["0.0", "0.0", "0.05"])
            self.assertTrue(all(row["val_accuracy_pct"] == "80.0" for row in rows))
            self.assertTrue(all(row["minutes"] == "0.5" for row in rows))
            self.assertTrue((output / "comparison.md").exists())
            self.assertEqual((output / "model.py").read_bytes(),
                             (run_ablations.PROJECT_ROOT / "model.py").read_bytes())
            with patch.object(sys, "argv", arguments), self.assertRaises(FileExistsError):
                run_ablations.main()

    def test_failure_stops_remaining_trials_and_records_status(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "suite"
            arguments = ["run_ablations.py", "--output", str(output), "--smoke-test"]
            with patch.object(sys, "argv", arguments), \
                    patch("run_ablations.subprocess.run", return_value=subprocess.CompletedProcess([], 1)) as launch, \
                    self.assertRaises(SystemExit):
                run_ablations.main()
            self.assertEqual(launch.call_count, 1)
            with (output / "comparison.csv").open(newline="") as comparison:
                rows = list(csv.DictReader(comparison))
            self.assertEqual(rows[0]["status"], "failed")
            self.assertEqual(rows[0]["val_accuracy_pct"], "")


if __name__ == "__main__":
    unittest.main()
