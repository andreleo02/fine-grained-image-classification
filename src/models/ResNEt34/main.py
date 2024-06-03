import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import argparse, sys, os, yaml

from torchvision.models import resnet34
from torchvision.datasets import FGVCAircraft, Flowers102

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.utils import main

def freeze_resnet_layers(model, num_blocks_to_freeze):
    """
    Freeze the first num_blocks_to_freeze blocks of layers in a ResNet model.

    Parameters:
    model (torch.nn.Module): The ResNet model to be modified.
    num_blocks_to_freeze (int): The number of blocks to freeze.

    Returns:
    None
    """
    frozen_blocks_count = 0  # Counter for the number of frozen blocks

    # Iterate over the layers (children) in the model's features
    for layer_name, layer in model.named_children():
        if frozen_blocks_count < num_blocks_to_freeze:
            # Freeze all parameters in the current layer
            for param in layer.parameters():
                param.requires_grad = False
            frozen_blocks_count += 1  # Increment the counter
        else:
            break  # Exit the loop once the required number of blocks are frozen

if __name__ == "__main__":

    parser = argparse.ArgumentParser("")
    parser.add_argument("--config", required=True, type=str, help="Path to the configuration file")
    parser.add_argument("--run_name", required=False, type=str, help="Name of the run")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: '{device}'")

    model = resnet34(pretrained=True)  # Load ResNet34

    num_classes = 200

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    learning_rate = config["training"]["lr"]
    frozen_layers = config["training"]["frozen_layers"]
    momentum = config["training"]["optimizer"]["momentum"]
    weight_decay = config["training"]["optimizer"]["weight_decay"]
    step_size = config["training"]["scheduler"]["step_size"]
    gamma = config["training"]["scheduler"]["gamma"]
    dataset_name = config["data"]["dataset_name"]


    # Modify the fully connected layer to match ResNet34's classifier input size
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    freeze_resnet_layers(model=model, num_blocks_to_freeze=frozen_layers)

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
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=learning_rate,
                          momentum=momentum,
                          weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    main(args=args,
         model=model,
         dataset_function=FGVCAircraft,
         dataset_name=dataset_name,
         num_classes=num_classes,
         train_transforms=train_transforms,
         val_transforms=val_transforms,
         criterion=criterion,
         optimizer=optimizer,
         scheduler=scheduler,
         device=device,
         config=config)
