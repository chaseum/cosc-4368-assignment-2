
from pathlib import Path

import numpy as np
import torch

from data_pipeline import create_data_pipeline
from submission import DogCNN, load_model, predict


def main():
    print("Testing model...")

    if not Path("model.pth").exists():
        print("No model.pth yet, saving random weights for testing")
        torch.save(DogCNN().state_dict(), "model.pth")

    model = load_model()
    assert not model.training
    print("Model loaded successfully on", next(model.parameters()).device)

    for n in [1, 7, 131]:
        images = np.random.randint(
            0, 256, (n, 128, 128, 3), dtype=np.uint8
        )

        predictions = predict(model, images)

        assert predictions.shape == (n,)
        assert np.issubdtype(predictions.dtype, np.integer)
        assert np.all(predictions >= 0)
        assert np.all(predictions < 10)

        print("Passed test with", n, "images")

    images = np.random.randint(
        0, 256, (10, 128, 128, 3), dtype=np.uint8
    )

    first = predict(model, images)
    second = predict(model, images)

    assert np.array_equal(first, second)
    print("Predictions are consistent")

    _, _, val_images, val_labels = create_data_pipeline()
    accuracy = (predict(model, val_images) == val_labels).mean()
    print(f"Validation accuracy: {accuracy:.2%}")

    print("All basic tests passed!")


if __name__ == "__main__":
    main()
