import argparse, sys, os, yaml
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torchvision.datasets import Flowers102, ImageNet, CIFAR10, FGVCAircraft
from torchvision.transforms import transforms
import torch.optim as optim
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.utils import main, load_model, freeze_layers

if __name__ == "__main__":

    parser = argparse.ArgumentParser("")
    parser.add_argument("--config", required = True, type = str, help = "Path to the configuration file")
    parser.add_argument("--run_name", required = False, type = str, help = "Name of the run")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: '{device}'")

    model = load_model(efficientnet_v2_s, weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1)

    num_classes = 100

    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    freeze_layers(model = model, num_blocks_to_freeze = 3)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    learning_rate = config["training"]["lr"]
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr = learning_rate, momentum = 0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 3, gamma = 0.1)

    main(args = args,
         model = model,
         dataset_function = FGVCAircraft,
         dataset_name = "FGVCAircraft",
         criterion = criterion,
         optimizer = optimizer,
         scheduler = scheduler,
         device = device,
         config = config)