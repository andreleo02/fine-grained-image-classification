from kaggle.api.kaggle_api_extended import KaggleApi
import os, requests, zipfile, tarfile
import yaml
import argparse
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import SwinTransformer, swin_t, Swin_T_Weights
from torch import DeviceObjType
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import Flowers102
from pathlib import Path

def get_data(dataset_name, batch_size, transform):
    dataset_path = "../../data/" + dataset_name
    train_dataset = Flowers102(root = dataset_path, split = "train", transform = transform, download = True)
    val_dataset = Flowers102(root = dataset_path, split = "val", transform = transform, download = True)
    test_dataset = Flowers102(root = dataset_path, split = "test", transform = transform, download = True)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

    return train_loader, val_loader, test_loader

def train_model(model, train_loader: DataLoader, optimizer, criterion, device):
    model.train()
    model.to(device)
    train_loss = 0
    train_acc = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        train_loss += loss.item()
        train_acc += calc_accuracy(labels, outputs.argmax(dim = 1))
        
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

def test_model(model, val_loader: DataLoader, criterion, device):
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

def calc_accuracy(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

def download_dataset_from_kaggle(dataset_name: str, output_dir: str = "dataset") -> None:
    api = KaggleApi()
    api.authenticate()
    final_output_dir: str = "src/data/" + output_dir
    api.dataset_download_files(dataset = dataset_name, path = final_output_dir, unzip = True)
    print(f"Dataset {dataset_name} downloaded correctly from Kaggle")

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
