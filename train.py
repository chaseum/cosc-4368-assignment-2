import torch
import torch.nn as nn
import random
import numpy as np
import csv

from data_pipeline import create_data_pipeline
from temp_model import create_temp_model


def train_one_epoch(model, train_loader, criterion, optimizer, device):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    # Loop through batches of dog images
    for images, labels in train_loader:

        # move images and labels to GPU
        images = images.to(device)
        labels = labels.to(device)

        # Clear gradients from previous batch
        optimizer.zero_grad()

        # forward pass: CNN predicts breeds
        outputs = model(images)

        # calc loss
        loss = criterion(outputs, labels)

        # calc gradients (backprop)
        loss.backward()

        # update weights
        optimizer.step()

        # record loss for this batch
        total_loss += loss.item() * images.size(0)

        # find predicted breed for each image
        predictions = outputs.argmax(dim=1)

        # count correct predictions
        correct += (predictions == labels).sum().item()

        # count images processed
        total += labels.size(0)

    # calc averages across all images
    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def train(model, train_loader, val_loader, criterion,
          optimizer, scheduler, device, epochs=5, checkpoint_path="temporary_model.pth"):

    best_val_accuracy = 0.0
    history = []

    for epoch in range(epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_accuracy = validate(
            model, val_loader, criterion, device
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "learning_rate": current_lr
        })

        print(f"Learning Rate: {current_lr:.6f}")
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Training Accuracy: {train_accuracy:.2%}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.2%}")

        # Save model if validation accuracy improved
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), checkpoint_path)
            print("New best model saved!")

        scheduler.step()
        

    print(f"\nBest Validation Accuracy: {best_val_accuracy:.2%}")
    return history



def validate(model, val_loader, criterion, device):

    # model to evaluation mode
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    # disable gradient calculations
    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            # forward pass
            outputs = model(images)

            # calc validation loss
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            # get predicted class
            predictions = outputs.argmax(dim=1)

            # count correct predictions
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




def main():
    set_seed(42)
    learning_rate = 0.001
    weight_decay = 0.01
    epochs = 5
    experiment_name = "temp_adamw_lr001_step3_seed42"

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    #  Tyler's data
    train_loader, val_loader, _, _ = create_data_pipeline(
        batch_size=16
    )

    # create temporary CNN
    model = create_temp_model()

    # CNN to GPU
    model = model.to(device)

    #choose a loss function
    criterion = nn.CrossEntropyLoss()

    # coose an optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.5
    )

    # train for x epochs
    history = train(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        epochs=epochs,
        checkpoint_path=f"{experiment_name}.pth"
    )

    with open(f"{experiment_name}.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)



if __name__ == "__main__":
    main()
