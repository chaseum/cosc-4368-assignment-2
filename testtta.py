import numpy as np
import torch

@torch.inference_mode()
def predict_tta(model, images):
    model.eval()
    device = next(model.parameters()).device
    predictions = []

    for start in range(0, len(images), 32):
        batch = torch.from_numpy(
            np.ascontiguousarray(images[start:start + 32])
        )

        batch = batch.permute(0, 3, 1, 2)
        batch = batch.to(device=device, dtype=torch.float32) / 255.0

        original_logits = model(batch)
        flipped_logits = model(torch.flip(batch, dims=[3]))

        combined_logits = original_logits + flipped_logits

        predictions.append(
            combined_logits.argmax(dim=1).cpu().numpy()
        )

    return (
        np.concatenate(predictions)
        if predictions
        else np.empty(0, dtype=np.int64)
    )


import numpy as np
from model import load_model
from data_pipeline import create_data_pipeline
#from tta_predict import predict_tta

_, _, images, labels = create_data_pipeline(batch_size=32)

model = load_model("advanced_60epochs_seed42.pth")

predictions = predict_tta(model, images)

print(f"TTA Accuracy: {np.mean(predictions == labels):.2%}")