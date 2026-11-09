import argparse
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import CustomDogClassifier
from train import class_metrics, main, run_epoch, train_model
from train_recipe import mix_batch


class TrainingTests(unittest.TestCase):
    def test_mix_batch_labels_match_the_pixels_kept(self):
        images = torch.arange(4.0).view(4, 1, 1, 1).expand(-1, 3, 64, 64).clone()
        labels = torch.arange(4)
        for cutmix, mixup in ((1.0, 0.0), (0.0, 1.0)):
            for _ in range(20):
                mixed, first, second, weight = mix_batch(images.clone(), labels, cutmix, mixup)
                expected = weight * first.float() + (1 - weight) * second.float()
                torch.testing.assert_close(mixed.mean(dim=(1, 2, 3)), expected, atol=1e-5, rtol=0)

    def test_new_architectures_keep_output_shape(self):
        for architecture in ("deep_mish", "deep_blur", "deep_blur_mish"):
            model = CustomDogClassifier(architecture).eval()
            self.assertEqual(model(torch.zeros(2, 3, 128, 128)).shape, (2, 10))
            self.assertFalse(any("kernel" in key for key in model.state_dict()))

    def test_smoothing_applies_only_to_training_loss(self):
        def finish_epoch(model, loader, device, criterion, optimizer=None, scaler=None):
            if optimizer is not None:
                optimizer.step()
            return 0.8, 0.5, torch.eye(10, dtype=torch.int64)

        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(
                output=Path(directory), lr=0.001, weight_decay=0.0001, epochs=1,
                patience=15, architecture="baseline", dropout=0.3, augmentation="crop",
                label_smoothing=0.05, seed=42, smoke_test=True, overfit=None,
                normalization={"mean": [0.2, 0.3, 0.4], "std": [0.5, 0.4, 0.3]}, train_on_all=False,
            )
            with patch("train.run_epoch", side_effect=finish_epoch) as epoch, patch("builtins.print"):
                train_model(nn.Linear(3, 10), [], [], args, [str(index) for index in range(10)])
            self.assertEqual(epoch.call_args_list[0].args[3].label_smoothing, 0.05)
            self.assertEqual(epoch.call_args_list[1].args[3].label_smoothing, 0.0)
            checkpoint = torch.load(args.output / "model.pth", weights_only=True)
            self.assertEqual(checkpoint["normalization"], args.normalization)

    def test_train_on_all_saves_final_epoch_even_when_val_accuracy_falls(self):
        accuracies = iter([0.9, 0.9, 0.2, 0.2])

        def finish_epoch(model, loader, device, criterion, optimizer=None, scaler=None):
            if optimizer is not None:
                optimizer.step()
            return 0.8, next(accuracies), torch.eye(10, dtype=torch.int64)

        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(
                output=Path(directory), lr=0.001, weight_decay=0.0001, epochs=2,
                patience=1, architecture="baseline", dropout=0.3, augmentation="strong",
                label_smoothing=0.1, seed=42, smoke_test=False, overfit=None,
                normalization=None, train_on_all=True,
            )
            with patch("train.run_epoch", side_effect=finish_epoch), patch("builtins.print"):
                train_model(nn.Linear(3, 10), [], [], args, [str(index) for index in range(10)])
            self.assertEqual(torch.load(args.output / "model.pth", weights_only=True)["epoch"], 2)

    def test_init_from_continues_the_saved_model_and_settings(self):
        statistics = {"mean": [0.2, 0.3, 0.4], "std": [0.5, 0.4, 0.3]}
        saved = CustomDogClassifier("baseline", 0.4, statistics)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "start.pth"
            torch.save({"architecture": "baseline", "dropout": 0.4, "normalization": statistics,
                        "state_dict": saved.state_dict()}, path)
            argv = ["train.py", "--device", "cpu", "--architecture", "deep", "--init-from", str(path),
                    "--output", str(Path(directory) / "run")]
            with patch("sys.argv", argv), patch("train.seed_everything"), patch("builtins.print"), \
                    patch("train.create_data_pipeline", return_value=([], [], None, None)), \
                    patch("train.train_model") as training:
                main()
        model, _, _, args, _ = training.call_args.args
        self.assertEqual((args.architecture, args.dropout, args.normalization), ("baseline", 0.4, statistics))
        for name, value in saved.state_dict().items():
            torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)

    def test_class_recall_uses_actual_class_support(self):
        confusion = torch.tensor([[2, 1, 0], [0, 1, 1], [0, 0, 0]])
        metrics = class_metrics(confusion, ["beagle", "dingo", "samoyed"])
        self.assertEqual(metrics["beagle"], {"support": 3, "recall": 2 / 3})
        self.assertEqual(metrics["dingo"], {"support": 2, "recall": 0.5})
        self.assertEqual(metrics["samoyed"], {"support": 0, "recall": None})

    def test_validation_reports_metrics_without_changing_weights(self):
        model = nn.Linear(3, 10)
        images = torch.tensor([[1., 2., 3.], [3., 2., 1.], [0., 0., 0.]])
        labels = torch.tensor([0, 3, 9])
        loader = DataLoader(TensorDataset(images, labels), batch_size=2)
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            logits = model(images)
            expected_loss = criterion(logits, labels).item()
            expected_accuracy = (logits.argmax(1) == labels).float().mean().item()
        weights = model.weight.detach().clone()

        loss, accuracy, confusion = run_epoch(model, loader, torch.device("cpu"), criterion)

        self.assertAlmostEqual(loss, expected_loss, places=6)
        self.assertAlmostEqual(accuracy, expected_accuracy, places=6)
        self.assertEqual(confusion.sum().item(), len(labels))
        self.assertFalse(model.training)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        torch.testing.assert_close(model.weight, weights, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
