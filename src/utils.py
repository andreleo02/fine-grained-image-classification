import os, requests, zipfile, tarfile, shutil
import wandb
import torch
import time
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
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
                  test_loader: DataLoader,
                  num_epochs: int,
                  patience: int,
                  device: DeviceObjType,
                  checkpoint_path,
                  optimizer,
                  criterion,
                  scheduler):
    start_time = time.time()

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        print(f"\nRUNNING EPOCH {epoch} ...")
        train_loss, train_acc = train_model(model = model,
                                            train_loader = train_loader,
                                            optimizer = optimizer,
                                            criterion = criterion,
                                            scheduler = scheduler,
                                            device = device)
        val_loss, val_acc = validate_model(model = model,
                                           val_loader = val_loader,
                                           criterion = criterion,
                                           device = device)
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
    end_time = time.time()
    model.load_state_dict(torch.load(checkpoint_path / f'best.pth'))
    test_accuracy = test_model(model = model, test_loader = test_loader, device = device)
    wandb.log({
        "training_time": f"{(end_time - start_time):.3} seconds",
        "test_accuracy": test_accuracy
    })


def train_model(model, train_loader: DataLoader, optimizer, criterion, scheduler, device):
    print("Training phase ...")
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
    print("Validation phase ...")
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

    return test_accuracy


def main(args, model, dataset_function, num_classes, dataset_name, criterion, optimizer, scheduler, device, config):
    wandb.login()

    torch.manual_seed(1234)

    net = config["net"]
    learning_rate = config["training"]["lr"]
    num_epochs = config["training"]["num_epochs"]
    patience = config["training"]["patience"]
    frozen_layers = config["training"]["frozen_layers"]
    momentum = config["training"]["optimizer"]["momentum"]
    weight_decay = config["training"]["optimizer"]["weight_decay"]
    step_size = config["training"]["scheduler"]["step_size"]
    gamma = config["training"]["scheduler"]["gamma"]
    batch_size = config["data"]["batch_size"]

    wandb.init(
        project = "Competition",
        name = f"{args.run_name}",
        config = {
        "learning_rate": learning_rate,
        "architecture": net,
        "dataset": dataset_name,
        "num_classes": num_classes,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "frozen_layers": frozen_layers,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "step_size": step_size,
        "gamma": gamma
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

    if config["data"]["pytorch"]:
        train_dataset, val_dataset, test_dataset = get_data(dataset_function = dataset_function,
                                              train_transforms = train_transforms,
                                              val_transforms = val_transforms)
    elif config["data"]["custom"]:
        download_url = config["data"]["download_url"]
        train_dataset, val_dataset = get_data_custom(dataset_name = dataset_name,
                                                     download_url = download_url,
                                                     num_classes = num_classes,
                                                     train_transforms = train_transforms,
                                                     val_transforms = val_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size = batch_size, num_workers = 4, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, num_workers = 4, shuffle = False)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, num_workers = 4, shuffle = False)

    optimal_model(model = model,
                  train_loader = train_loader,
                  val_loader = val_loader,
                  test_loader = test_loader,
                  num_epochs = num_epochs,
                  patience = patience,
                  device = device,
                  checkpoint_path = checkpoint_path,
                  criterion = criterion,
                  optimizer = optimizer,
                  scheduler = scheduler)
    
    # test_model(model, test_loader, device = device)

def calc_accuracy(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    print(f"Correct predictions: {correct} / {len(y_pred)}")
    return acc

def get_data(dataset_function, train_transforms, val_transforms):
    dataset_path = "../../data/"
    train_dataset = dataset_function(root = dataset_path, split = "train", transform = train_transforms, download = True)
    val_dataset = dataset_function(root = dataset_path, split = "val", transform = val_transforms, download = True)
    test_dataset = dataset_function(root = dataset_path, split = "test", transform = val_transforms, download = True)
    #train_dataset = dataset_function(root = dataset_path, train = True, transform = train_transforms, download = True)
    #val_dataset = dataset_function(root = dataset_path, train = False, transform = val_transforms, download = True)

    return train_dataset, val_dataset, test_dataset

def get_data_custom(dataset_name, download_url, num_classes, train_transforms, val_transforms):
    data_dir = os.path.join("../../data", dataset_name)
    if os.path.isdir(data_dir):
        print(f"Dataset {dataset_name} already exists at path {data_dir}. Not downloading")
    else:
        download_dataset_tgz(url = download_url)

    images_dir = os.path.join(data_dir, 'images')
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    with open(os.path.join(data_dir, 'images.txt')) as f:
        images = [line.strip().split() for line in f.readlines()]

    with open(os.path.join(data_dir, 'image_class_labels.txt')) as f:
        labels = [line.strip().split() for line in f.readlines()]

    with open(os.path.join(data_dir, 'train_test_split.txt')) as f:
        train_test_split_dataset = [line.strip().split() for line in f.readlines()]

    os.makedirs(train_dir, exist_ok = True)
    os.makedirs(val_dir, exist_ok = True)

    for i in range(1, num_classes):
        os.makedirs(os.path.join(train_dir, str(i)), exist_ok = True)
        os.makedirs(os.path.join(val_dir, str(i)), exist_ok = True)

    file_paths = [os.path.join(images_dir, img[1]) for img in images]
    labels = [int(label[1]) for label in labels]
    train_test_split_dataset = [int(split[1]) for split in train_test_split_dataset]

    data = list(zip(file_paths, labels, train_test_split_dataset))

    train_data, val_data = train_test_split([item for item in data if item[2] == 1], test_size = 0.3, stratify = [item[1] for item in data if item[2] == 1])

    def copy_files(data, target_dir):
        for file_path, label, _ in data:
            target_class_dir = os.path.join(target_dir, str(label))
            shutil.copy(file_path, target_class_dir)

    copy_files(train_data, train_dir)
    copy_files(val_data, val_dir)

    train_dataset = datasets.ImageFolder(train_dir, transform = train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform = val_transforms)

    return train_dataset, val_dataset

def download_dataset_zip(url: str, output_dir: str = "../../data") -> None:
    response = requests.get(url)
    
    if response.status_code == 200:
        with open("tmp.zip", "wb") as f:
            f.write(response.content)
        try:
            with zipfile.ZipFile("tmp.zip", "r") as zip_ref:
                zip_ref.extractall(output_dir)
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

def download_dataset_tgz(url: str, output_dir: str = "../../data") -> None:
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
