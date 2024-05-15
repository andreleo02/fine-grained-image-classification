import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import SwinTransformer, swin_t, Swin_T_Weights
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import Flowers102

def load_model() -> SwinTransformer:
    # weights = Swin_T_Weights.DEFAULT
    return swin_t(weights = 0)

def train_model(model: SwinTransformer,
                dataset_name: str,
                num_epochs: int = 10,
                learning_rate: float = 0.001,
                num_classes: int = 0,
                batch_size: int = 32)-> SwinTransformer:
    num_classes = 200
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
        for inputs, labels in train_loader:
            model.train()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        scheduler.step()

        val_loss = validate_model(model = model, val_loader = val_loader, criterion = criterion)
        print(f"Epoch {epoch}, Validation loss: {loss.item()}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == 2:
                print(f"Early stop at epoch {epoch}.")
                break

    return model

def validate_model(model: SwinTransformer, val_loader: DataLoader, criterion):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    return val_loss / len(val_loader)

model = load_model()
model = train_model(model = model,
                    dataset_name = "Flowers102", 
                    num_epochs = 10,
                    learning_rate = 0.001,
                    num_classes = 102,
                    batch_size = 32)
model.eval()

