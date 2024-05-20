import os, requests, zipfile, tarfile
import wandb
import torch
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from pathlib import Path
from torch import DeviceObjType
from torch.utils.data import DataLoader


def load_model(function, weights = None):
    return function(weights = weights)

def freeze_layers(model, num_blocks_to_freeze):
    layers_frozen = 0
    for _, child in model.features.named_children():
        if layers_frozen < num_blocks_to_freeze:
            for param in child.parameters():
                param.requires_grad = False
            layers_frozen += 1
        else:
            break   

def optimal_model(model,
                  train_loader: DataLoader,
                  val_loader: DataLoader,
                  num_epochs: int,
                  patience: int,
                  device: DeviceObjType,
                  checkpoint_path,
                  optimizer,
                  criterion,
                  scheduler):

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        print(f"\nRUNNING EPOCH {epoch} ...")
        # train_loss, train_acc = train_model(model = model,
        #                                     train_loader = train_loader,
        #                                     optimizer = optimizer,
        #                                     criterion = criterion,
        #                                     scheduler = scheduler,
        #                                     device = device)
        print("Training phase ...")
        model.train()
        model.to(device)
        train_loss = 0
        train_acc = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            train_acc += calc_accuracy(labels, torch.argmax(input = outputs, dim = 1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        # val_loss, val_acc = validate_model(model = model,
        #                                    val_loader = val_loader,
        #                                    criterion = criterion,
        #                                    device = device)
        print("Validation phase ...")
        model.eval()
        model.to(device)
        val_loss = 0
        val_acc = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                batch_acc = calc_accuracy(labels, torch.argmax(input = outputs, dim = 1))
                val_acc += batch_acc

        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        print(f"\nTraining loss: {train_loss:.3}, Training accuracy: {train_acc:.3}")
        print(f"Validation loss: {val_loss:.3}, Validation accuracy: {val_acc:.3}")

        wandb.log({
            "train/loss":train_loss,
            "train/accuracy":train_acc,
            "val/loss":val_loss,
            "val/accuracy":val_acc
        })

        torch.save(model.state_dict(), checkpoint_path / f'epoch-{epoch}.pth')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path / f'best.pth')
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f"Validation loss did not imporove for {patience} epochs. Killing the training...")
                break

def train_model(model, train_loader: DataLoader, optimizer, criterion, scheduler, device):
    model.train()
    model.to(device)
    train_loss = 0
    train_accuracy = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        train_loss += loss.item()
        train_accuracy += calc_accuracy(labels, torch.argmax(input = outputs, dim = 1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()
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
            batch_acc = calc_accuracy(labels, torch.argmax(input = outputs, dim = 1))
            val_accuracy += batch_acc

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

def main(args, model, dataset_function, dataset_name, download_dataset,  criterion, optimizer, scheduler, device, config):
    wandb.login()

    torch.manual_seed(1234)

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

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(size = (224, 224), antialias = True),
        transforms.RandomHorizontalFlip(p = 0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
    ])
    
    train_loader, val_loader = get_data(dataset_function = dataset_function,
                                        download_dataset = download_dataset,
                                        batch_size = batch_size,
                                        train_transforms = train_transforms,
                                        val_transforms = val_transforms)
    

    optimal_model(model = model,
                  train_loader = train_loader,
                  val_loader = val_loader,
                  num_epochs = num_epochs,
                  patience = patience,
                  device = device,
                  checkpoint_path = checkpoint_path,
                  criterion = criterion,
                  optimizer = optimizer,
                  scheduler = scheduler)
    
    # test_model(model, test_loader, device = device)

def get_data(dataset_function, download_dataset, batch_size, train_transforms, val_transforms):
    dataset_path = "../../data/"
    if dataset_function is None:
        download_dataset_tgz(url = download_dataset, output_dir = dataset_path)
        # split in train test and transform
    else:
        #train_dataset = dataset_function(root = dataset_path, split = "train", transform = train_transforms, download = True)
        #val_dataset = dataset_function(root = dataset_path, split = "val", transform = val_transforms, download = True)
        train_dataset = dataset_function(root = dataset_path, train = True, transform = train_transforms, download = True)
        val_dataset = dataset_function(root = dataset_path, train = False, transform = val_transforms, download = True)
    #test_dataset = dataset_function(root = dataset_path, split = "test", transform = transform, download = True)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, num_workers = 4, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, num_workers = 4, shuffle = False)
    # test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

    return train_loader, val_loader

def calc_accuracy(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    print(f"Correct predictions: {correct} / {len(y_pred)}")
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
    print(f"Downloading dataset from {url} ...")
    response = requests.get(url)
    
    if response.status_code == 200:
        print("\nDownload completed. Extracting files ...")
        with open("tmp.tgz", "wb") as f:
            f.write(response.content)
        try:
            with tarfile.open("tmp.tgz", "r:gz") as tgz_ref:
                tgz_ref.extractall(output_dir)
            print(f"Download and extracted from {url} completed successfully! Extracted files in {output_dir}")
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
