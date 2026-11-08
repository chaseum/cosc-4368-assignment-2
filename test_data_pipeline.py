import unittest
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from unittest.mock import patch

from data_pipeline import AUGMENTATIONS, compute_rgb_statistics, create_data_pipeline, create_train_transform


class AugmentationTests(unittest.TestCase):
    def setUp(self):
        pixels = np.random.default_rng(42).integers(0, 256, (128, 128, 3), dtype=np.uint8)
        self.image = Image.fromarray(pixels)

    def test_statistics_use_all_pixels_from_only_the_selected_training_folder(self):
        images = [np.full((128, 128, 3), color, dtype=np.uint8)
                  for color in ([0, 32, 64], [255, 160, 192])]
        with tempfile.TemporaryDirectory() as directory:
            training = Path(directory) / "train" / "beagle"
            validation = Path(directory) / "val" / "beagle"
            training.mkdir(parents=True)
            validation.mkdir(parents=True)
            for index, pixels in enumerate(images):
                Image.fromarray(pixels).save(training / f"{index}.png")
            Image.fromarray(np.zeros_like(images[0])).save(validation / "unused.png")
            statistics = compute_rgb_statistics(training.parent)
        pixels = np.stack(images).astype(np.float64) / 255.0
        np.testing.assert_allclose(statistics["mean"], pixels.mean(axis=(0, 1, 2)), atol=1e-12)
        np.testing.assert_allclose(statistics["std"], pixels.std(axis=(0, 1, 2)), atol=1e-12)

    def test_normalization_is_applied_after_training_augmentation(self):
        statistics = {"mean": [0.2, 0.3, 0.4], "std": [0.5, 0.4, 0.3]}
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            pixels = create_train_transform()(self.image)
            torch.manual_seed(42)
            normalized = create_train_transform(normalization=statistics)(self.image)
        expected = transforms.Normalize(statistics["mean"], statistics["std"])(pixels)
        torch.testing.assert_close(normalized, expected, rtol=0, atol=0)

    def test_original_matches_previous_preprocessing(self):
        previous = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=8),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
        ])
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            expected = previous(self.image)
            torch.manual_seed(42)
            actual = create_train_transform()(self.image)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_augmentation_modes_preserve_shape_and_repeat_with_seed(self):
        for augmentation in AUGMENTATIONS:
            with self.subTest(augmentation=augmentation), torch.random.fork_rng(devices=[]):
                transform = create_train_transform(augmentation)
                torch.manual_seed(42)
                first = transform(self.image)
                torch.manual_seed(42)
                second = transform(self.image)
                self.assertEqual(first.shape, (3, 128, 128))
                self.assertEqual(first.dtype, torch.float32)
                self.assertTrue(torch.isfinite(first).all())
                torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_erasing_changes_the_crop_when_applied(self):
        crop = create_train_transform("crop")
        crop_erase = create_train_transform("crop_erase")
        crop_erase.transforms[-1].p = 1.0
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            original = crop(self.image)
            torch.manual_seed(42)
            erased = crop_erase(self.image)
        self.assertFalse(torch.equal(original, erased))

    def test_strong_erases_after_normalization_with_smooth_rotation(self):
        statistics = {"mean": [0.2, 0.3, 0.4], "std": [0.5, 0.4, 0.3]}
        operations = create_train_transform("strong", statistics).transforms
        self.assertIsInstance(operations[-2], transforms.Normalize)
        self.assertIsInstance(operations[-1], transforms.RandomErasing)
        self.assertEqual(operations[0].interpolation, transforms.InterpolationMode.BILINEAR)

    def test_include_val_adds_labelled_validation_images_to_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for folder, count in (("imagewoof_train/train", 3), ("imagewoof_val/val", 2)):
                for name in ("beagle", "dingo"):
                    (root / folder / name).mkdir(parents=True)
                    for index in range(count):
                        Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8)).save(
                            root / folder / name / f"{index}.png")
            (root / "imagewoof_train/classes.txt").write_text("0 dingo\n1 beagle\n", encoding="utf-8")
            with patch("data_pipeline.PROJECT_ROOT", root):
                split = create_data_pipeline(2)[0].dataset
                combined = create_data_pipeline(2, include_val=True)[0].dataset
            self.assertEqual((len(split), len(combined)), (6, 10))
            self.assertEqual(sorted(combined[index][1] for index in range(10)), [0] * 5 + [1] * 5)

    def test_unknown_augmentation_is_rejected(self):
        with self.assertRaises(ValueError):
            create_train_transform("unknown")


if __name__ == "__main__":
    unittest.main()
