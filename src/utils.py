import wandb
import torch
import sys, os

from torch.utils.data import DataLoader
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from data_utils import get_data, get_data_custom
from training_utils import optimal_model

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

def main(args, model, dataset_function, num_classes, dataset_name, train_transforms, val_transforms, criterion, optimizer, scheduler, device, config):
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

    checkpoint_path = Path("./checkpoint")
    checkpoint_path = checkpoint_path / args.run_name
    checkpoint_path.mkdir(exist_ok = True, parents = True)

    if config["data"]["custom"]:
        download_url = config["data"]["download_url"]
        train_dataset, val_dataset = get_data_custom(dataset_name = dataset_name,
                                                     download_url = download_url,
                                                     num_classes = num_classes,
                                                     train_transforms = train_transforms,
                                                     val_transforms = val_transforms)
    else:
        train_dataset, val_dataset = get_data(dataset_function = dataset_function,
                                              train_transforms = train_transforms,
                                              val_transforms = val_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size = batch_size, num_workers = 4, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, num_workers = 4, shuffle = False)

    if config["wandb"]:
        wandb.login()
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
            "gamma": gamma,
            "train_size": len(train_dataset),
            "validation_size": len(val_dataset),
            })

    optimal_model(model = model,
                  train_loader = train_loader,
                  val_loader = val_loader,
                  num_epochs = num_epochs,
                  patience = patience,
                  device = device,
                  checkpoint_path = checkpoint_path,
                  criterion = criterion,
                  optimizer = optimizer,
                  scheduler = scheduler,
                  wandb_enabled = config["wandb"])
