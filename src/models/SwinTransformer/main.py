import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import argparse, sys, os, yaml

from torchvision.models import swin_t, Swin_T_Weights
from torchvision.datasets import FGVCAircraft, Flowers102

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.utils import main, load_model, freeze_layers

if __name__ == "__main__":

    parser = argparse.ArgumentParser("")
    parser.add_argument("--config", required = True, type = str, help = "Path to the configuration file")
    parser.add_argument("--run_name", required = False, type = str, help = "Name of the run")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(swin_t, weights = Swin_T_Weights.IMAGENET1K_V1)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    learning_rate = config["training"]["lr"]
    frozen_layers = config["training"]["frozen_layers"]
    momentum = config["training"]["optimizer"]["momentum"]
    weight_decay = config["training"]["optimizer"]["weight_decay"]
    step_size = config["training"]["scheduler"]["step_size"]
    gamma = config["training"]["scheduler"]["gamma"]
    num_classes = config["data"]["num_classes"]
    dataset_name = config["data"]["dataset_name"]


    model.head = nn.Linear(model.head.in_features, num_classes)
    freeze_layers(model = model, num_blocks_to_freeze = frozen_layers)

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
                          lr = learning_rate,
                          momentum = momentum,
                          weight_decay = weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = step_size, gamma = gamma)

    main(args = args,
         model = model,
         dataset_function = FGVCAircraft,
         dataset_name = dataset_name,
         num_classes = num_classes,
         train_transforms = train_transforms,
         val_transforms = val_transforms,
         criterion = criterion,
         optimizer = optimizer,
         scheduler = scheduler,
         device = device,
         config = config)