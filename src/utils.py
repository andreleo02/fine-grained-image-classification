import os, requests, zipfile, tarfile, yaml
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
from torch import DeviceObjType
from torchvision import transforms
from torch.utils.data import DataLoader


def load_model(function, weights = None):
    return function(weights = weights)

def optimal_model(model,
                  train_loader: DataLoader,
                  val_loader: DataLoader,
                  num_epochs: int,
                  learning_rate: float,
                  num_classes: int,
                  patience: int,
                  device: DeviceObjType,
                  net: str,
                  checkpoint_path):
    model.head = nn.Linear(model.head.in_features, num_classes)

    criterion = nn.CrossEntropyLoss().to(device = device)
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 3, gamma = 0.1)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    for epoch in range(num_epochs):
        train_loss, train_acc = train_model(model = model,
                                            train_loader = train_loader,
                                            optimizer = optimizer,
                                            criterion = criterion,
                                            device = device)
        scheduler.step()
        val_loss, val_acc = validate_model(model = model,
                                           val_loader = val_loader,
                                           criterion = criterion,
                                           device = device)
        print(f"Epoch {epoch}, Train loss: {train_loss}, Validation loss: {val_loss}")
        print(f"Epoch {epoch}, Train accuracy: {train_acc}, Validation accuracy: {val_acc}")

        wandb.log({
            "train/loss":train_loss,
            "train/accuracy":train_acc,
            "val/loss":val_loss,
            "val/accuracy":val_acc
        })

        torch.save(net.state_dict(), checkpoint_path / f'epoch-{epoch}.pth')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(net.state_dict(), checkpoint_path / f'best.pth')
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f"Validation Accuracy did not imporove for {patience} epochs. Killing the training...")
                break

    return model

def train_model(model, train_loader: DataLoader, optimizer, criterion, device):
    model.train()
    model.to(device)
    train_loss = 0
    train_accuracy = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        train_loss += loss.item()
        train_accuracy += calc_accuracy(labels, outputs.argmax(dim = 1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss /= len(train_loader)
    train_accuracy /= len(train_loader)
    return train_loss, train_accuracy

def validate_model(model, val_loader: DataLoader, criterion, device):
    model.eval()
    model.to(device)
    val_loss = 0
    val_accuracy = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_accuracy += calc_accuracy(labels, outputs.argmax(dim = 1))
    val_loss /= len(val_loader)
    val_accuracy /= len(val_loader)
    return val_loss, val_accuracy

def test_model(model, test_loader: DataLoader, device):
    model.eval()
    model.to(device)
    test_accuracy = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            test_accuracy += calc_accuracy(labels, outputs.argmax(dim = 1))
    test_accuracy /= len(test_loader)

    print(f"Test accuracy: {test_accuracy}")

def main(args, model_function, dataset_function, dataset_name, num_classes):
    wandb.login()

    torch.manual_seed(1234)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    net = config["net"]
    learning_rate = config["training"]["lr"]
    num_epochs = config["training"]["num_epochs"]
    patience = config["training"]["patience"]
    batch_size = config["data"]["batch_size"]

    wandb.init(
        project = "Competition",
        name = f"{args.run_name}",
        config = {
        "learning_rate": learning_rate,
        "architecture": net,
        "dataset": dataset_name,
        "epochs": num_epochs,
        "batch_size":batch_size,
        })

    checkpoint_path = Path("./checkpoint")
    checkpoint_path = checkpoint_path / args.run_name
    checkpoint_path.mkdir(exist_ok = True, parents = True)

    model = load_model(model_function)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
    ])
    train_loader, val_loader = get_data(dataset_function = dataset_function,
                                                     dataset_name = dataset_name,
                                                     batch_size = batch_size,
                                                     transform = transform)
    
    model = optimal_model(model = model,
                          train_loader = train_loader,
                          val_loader = val_loader,
                          num_epochs = num_epochs,
                          learning_rate = learning_rate,
                          num_classes = num_classes,
                          patience = patience,
                          device = device,
                          net = net,
                          checkpoint_path = checkpoint_path)
    # test_model(model, test_loader, device = device)

def get_data(dataset_function, dataset_name, batch_size, transform):
    dataset_path = "../../data/" + dataset_name
    train_dataset = dataset_function(root = dataset_path, split = "train", transform = transform, download = True)
    val_dataset = dataset_function(root = dataset_path, split = "val", transform = transform, download = True)
    # test_dataset = dataset_function(root = dataset_path, split = "test", transform = transform, download = True)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)
    # test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

    return train_loader, val_loader

def calc_accuracy(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

def download_dataset_zip(url: str, output_dir: str = "dataset") -> None:
    response = requests.get(url)
    final_output_dir: str = "src/data/" + output_dir
    
    if response.status_code == 200:
        with open("tmp.zip", "wb") as f:
            f.write(response.content)
        try:
            with zipfile.ZipFile("tmp.zip", "r") as zip_ref:
                zip_ref.extractall(final_output_dir)
            print(f"Download and extracted from {url} completed successfully!")
            pass
        except zipfile.BadZipFile as e: 
            print(f"Exception while extracting files from zip. Zip downloaded from {url}. Exception {e}")
            pass
        try:
            os.remove("tmp.zip")
        except Exception as e:
            print(f"Error occurred: {e}")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")

def download_dataset_tgz(url: str, output_dir: str = "dataset") -> None:
    response = requests.get(url)
    final_output_dir: str = "src/data/" + output_dir
    
    if response.status_code == 200:
        with open("tmp.tgz", "wb") as f:
            f.write(response.content)
        try:
            with tarfile.open("tmp.tgz", "r:gz") as tgz_ref:
                tgz_ref.extractall(final_output_dir)
            print(f"Download and extracted from {url} completed successfully!")
            pass
        except tarfile.TarError as e: 
            print(f"Exception while extracting files from tgz. Tar downloaded from {url}. Exception {e}")
            pass
        try:
            os.remove("tmp.tgz")
        except Exception as e:
            print(f"Error occurred: {e}")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")
