import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch
from torch import nn

from model import ARCHITECTURES, CustomDogClassifier, load_model, predict


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def test_architectures_use_only_allowed_layers_in_sequence(self):
        allowed_layers = (nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d, nn.AvgPool2d,
                          nn.AdaptiveAvgPool2d, nn.Flatten, nn.Dropout, nn.Linear)
        for architecture, parameters, sizes in [
            ("baseline", 288746, [64, 32, 16]),
            ("deep", 1175786, [64, 32, 16, 8]),
            ("deep_spatial", 1183466, [64, 32, 16, 8]),
            ("deep_pointwise", 1241834, [64, 32, 16, 8]),
            ("deep_2233", 1913834, [64, 32, 16, 8]),
            ("wide_2233", 7643082,[64, 32, 16, 8]),
            ("wide_22333", 14724042, [64, 32, 16, 8, 4]),
        ]:
            with self.subTest(architecture=architecture):
                model = CustomDogClassifier(architecture).eval()
                self.assertIsInstance(model.network, nn.Sequential)
                self.assertTrue(all(type(layer) in allowed_layers for layer in model.network))
                self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), parameters)
                graph = torch.fx.symbolic_trace(model).graph
                nodes = list(graph.nodes)
                for previous, node in zip(nodes, nodes[1:-1]):
                    self.assertEqual(node.op, "call_module")
                    self.assertEqual(node.args, (previous,))
                self.assertEqual(nodes[-1].args, (nodes[-2],))

                features = torch.zeros(1, 3, 128, 128)
                pooled_sizes = []
                with torch.no_grad():
                    for layer in model.network:
                        features = layer(features)
                        if isinstance(layer, nn.MaxPool2d):
                            pooled_sizes.append(features.shape[-1])
                self.assertEqual(pooled_sizes, sizes)
                self.assertEqual(features.shape, (1, 10))

    def test_deep_2233_adds_convolutions_only_in_late_blocks(self):
        model = CustomDogClassifier("deep_2233")
        counts = []
        convolutions = 0
        for layer in model.network:
            if isinstance(layer, nn.Conv2d):
                convolutions += 1
            elif isinstance(layer, nn.MaxPool2d):
                counts.append(convolutions)
                convolutions = 0
        self.assertEqual(counts, [2, 2, 3, 3])

    def test_architectures_have_finite_gradients(self):
        for architecture in ARCHITECTURES:
            with self.subTest(architecture=architecture):
                model = CustomDogClassifier(architecture)
                logits = model(torch.zeros(2, 3, 128, 128))
                nn.CrossEntropyLoss()(logits, torch.tensor([0, 9])).backward()
                self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                                    for parameter in model.parameters()))

    def test_checkpoint_preserves_logits_and_dropout(self):
        inputs = torch.zeros(1, 3, 128, 128)
        for architecture in ARCHITECTURES:
            with self.subTest(architecture=architecture), tempfile.TemporaryDirectory() as directory:
                model = CustomDogClassifier(architecture, dropout=0.4).eval()
                path = Path(directory) / "model.pth"
                torch.save({"architecture": architecture, "dropout": 0.4,
                            "state_dict": model.state_dict()}, path)
                loaded = load_model(path)
                self.assertFalse(loaded.training)
                self.assertEqual(loaded.network[-2].p, 0.4)
                with torch.no_grad():
                    torch.testing.assert_close(loaded(inputs), model(inputs), rtol=0, atol=0)

    def test_legacy_checkpoint_without_dropout_still_loads(self):
        model = CustomDogClassifier("deep").eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pth"
            torch.save({"architecture": "deep", "state_dict": model.state_dict()}, path)
            loaded = load_model(path)
            self.assertIsNone(loaded.normalization)
            with torch.no_grad():
                inputs = torch.zeros(1, 3, 128, 128)
                torch.testing.assert_close(loaded(inputs), model(inputs), rtol=0, atol=0)

    def test_checkpoint_restores_normalization_for_raw_image_prediction(self):
        statistics = {"mean": [0.2, 0.3, 0.4], "std": [0.5, 0.4, 0.3]}
        model = CustomDogClassifier(normalization=statistics).eval()
        images = np.random.default_rng(42).integers(0, 256, (2, 128, 128, 3), dtype=np.uint8)
        expected = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255
        expected = (expected - torch.tensor(statistics["mean"]).view(1, 3, 1, 1)) / \
            torch.tensor(statistics["std"]).view(1, 3, 1, 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pth"
            torch.save({"architecture": "baseline", "normalization": statistics,
                        "state_dict": model.state_dict()}, path)
            loaded = load_model(path)
            self.assertEqual(loaded.normalization, statistics)
            captured = []
            hook = loaded.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0].clone()))
            predictions = predict(loaded, images)
            hook.remove()
            torch.testing.assert_close(captured[0], expected, rtol=0, atol=0)
            with torch.no_grad():
                logits = loaded(expected) + loaded(expected.flip(dims=(3,)))
                np.testing.assert_array_equal(predictions, logits.argmax(1).numpy())

    def test_predict_averages_the_image_and_its_mirror(self):
        images = np.zeros((1, 128, 128, 3), dtype=np.uint8)
        images[0, :, :64] = 255  # Bright left half; the mirror is bright on the right.

        class SideVote(nn.Module):
            def forward(self, batch):
                # Left-bright scores class 0 a little; right-bright scores class 1 a lot.
                left_bright = batch[:, :, :, :64].mean((1, 2, 3)) > 0.5
                return torch.stack([left_bright * 1.0, ~left_bright * 2.0], dim=1)

        model = nn.Sequential(SideVote(), nn.Identity())
        model.register_parameter("anchor", nn.Parameter(torch.zeros(1)))
        self.assertEqual(predict(model, images).tolist(), [1])

    def test_predict_matches_tensor_inference_across_batch_boundary(self):
        images = np.random.default_rng(42).integers(0, 256, (3, 128, 128, 3), dtype=np.uint8)
        images = np.tile(images, (11, 1, 1, 1))
        inputs = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255
        model = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 10))
        with torch.no_grad():
            expected = model(inputs).argmax(1).numpy()
        predictions = predict(model, images)
        np.testing.assert_array_equal(predictions, expected)
        self.assertEqual(predictions.dtype, np.int64)
        self.assertFalse(model.training)
        self.assertEqual(predict(model, images[:1]).shape, (1,))
        self.assertEqual(predict(model, images[:0]).dtype, np.int64)
        self.assertEqual(predict(model, images[:0]).shape, (0,))

    def test_invalid_inputs(self):
        model = CustomDogClassifier()
        with self.assertRaises(ValueError):
            CustomDogClassifier("resnet")
        for dropout in (-0.1, 1, float("nan")):
            with self.subTest(dropout=dropout), self.assertRaises(ValueError):
                CustomDogClassifier(dropout=dropout)
        for statistics in (
            {"mean": [0, 0, 0], "std": [0, 1, 1]},
            {"mean": [0], "std": [1, 1, 1]},
            {"mean": [float("nan"), 0, 0], "std": [1, 1, 1]},
        ):
            with self.subTest(normalization=statistics), self.assertRaises(ValueError):
                CustomDogClassifier(normalization=statistics)
        with self.assertRaises(TypeError):
            predict(model, np.zeros((1, 128, 128, 3), dtype=np.float32))
        with self.assertRaises(ValueError):
            predict(model, np.zeros((1, 3, 128, 128), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
