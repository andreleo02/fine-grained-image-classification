import yaml
import argparse
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import SwinTransformer, swin_t, Swin_T_Weights
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import Flowers102
from pathlib import Path

def load_model() -> SwinTransformer:
    # weights = Swin_T_Weights.DEFAULT
    return swin_t(weights = 0)

def optimal_model(model: SwinTransformer,
                 dataset_name: str,
                 num_epochs: int = 10,
                 learning_rate: float = 0.001,
                 num_classes: int = 0,
                 batch_size: int = 32,
                 patience: int = 3)-> SwinTransformer:
    model.head = nn.Linear(model.head.in_features, num_classes)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
    ])

    dataset_path = "../../data/" + dataset_name
    train_dataset = Flowers102(root = dataset_path, split = "train", transform = transform, download = True)
    test_dataset = Flowers102(root = dataset_path, split = "test", transform = transform, download = True)
    val_dataset = Flowers102(root = dataset_path, split = "val", transform = transform, download = True)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(val_dataset)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 3, gamma = 0.1)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    for epoch in range(num_epochs):
        
        train_loss, train_acc = train_model(model = model, train_loader = train_loader, optimizer = optimizer, criterion = criterion)
        scheduler.step()
        val_loss, val_acc = validate_model(model = model, val_loader = val_loader, criterion = criterion)
        print(f"Epoch {epoch}, Train loss: {train_loss}, Validation loss: {val_loss}")

        wandb.log({
            # log training stats
            "train/loss":train_loss,
            "train/accuracy":train_acc,
            # log validation stats
            "val/loss":val_loss,
            "val/accuracy":val_acc
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f"Early stop at epoch {epoch}.")
                break

    return model

def train_model(model: SwinTransformer, train_loader: DataLoader, optimizer, criterion):
    train_loss = 0
    train_acc = 0
    for inputs, labels in train_loader:
        model.train()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        train_acc += calc_accuracy(labels, outputs)
    training_loss = train_loss / len(train_loader)
    training_accuracy = train_acc / len(train_loader)
    return training_loss, training_accuracy

def validate_model(model: SwinTransformer, val_loader: DataLoader, criterion):
    model.eval()
    val_loss = 0
    val_accuracy = 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_accuracy += calc_accuracy(labels, outputs)
    validation_loss = val_loss / len(val_loader)
    validation_accuracy = val_accuracy / len(val_loader)
    return validation_loss, validation_accuracy

def calc_accuracy(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

if __name__ == "__main__":

    wandb.login()

    torch.manual_seed(1234)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser("")
    parser.add_argument("--config", required = True, type = str, help = "Path to the configuration file")
    parser.add_argument("--run_name", required = False, type = str, help = "Name of the run")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    learning_rate = config["training"]["lr"]
    num_epochs = config["training"]["num_epochs"]
    run_name = config["run_name"]
    batch_size = config["data"]["batch_size"]
    patience = config["training"]["patience"]

    # WandB initialization
    wandb.init(
        # Set the project where this run will be logged
        project="Competition",
        # Set a run name
        # (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=f"{run_name}",
        # Track hyperparameters and run metadata
        config={
        "learning_rate": learning_rate,
        "dataset": "Flowers102",
        "epochs": num_epochs,
        "batch_size":batch_size,
        })

    checkpoint_path = Path("./checkpoint")
    checkpoint_path = checkpoint_path / run_name
    checkpoint_path.mkdir(exist_ok=True, parents=True)

    model = load_model()
    model = optimal_model(model = model,
                         dataset_name = "Flowers102", 
                         num_epochs = num_epochs,
                         learning_rate = learning_rate,
                         num_classes = 102,
                         batch_size = batch_size,
                         patience = patience)

