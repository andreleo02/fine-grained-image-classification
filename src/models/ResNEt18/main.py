import argparse, sys, os, yaml
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.datasets import FGVCAircraft
import torch.optim as optim
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.utils import main, load_model, freeze_layers

# Custom function to freeze layers in ResNet
def freeze_resnet_layers(model, num_blocks_to_freeze):
    frozen_blocks_count = 0  # Counter for the number of frozen blocks
    for layer_name, layer in model.named_children():
        if frozen_blocks_count < num_blocks_to_freeze:
            # Freeze all parameters in the current layer
            for param in layer.parameters():
                param.requires_grad = False
            frozen_blocks_count += 1  # Increment the counter
        else:
            break  


if __name__ == "__main__":

    parser = argparse.ArgumentParser("")
    parser.add_argument("--config", required=True, type=str, help="Path to the configuration file")
    parser.add_argument("--run_name", required=False, type=str, help="Name of the run")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(resnet18, weights=ResNet18_Weights.IMAGENET1K_V1)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    learning_rate = config["training"]["lr"]
    frozen_layers = config["training"]["frozen_layers"]
    momentum = config["training"]["optimizer"]["momentum"]
    weight_decay = config["training"]["optimizer"]["weight_decay"]
    step_size = config["training"]["scheduler"]["step_size"]
    gamma = config["training"]["scheduler"]["gamma"]
    num_classes = config["data"]["num_classes"]

    # This Doesn't work in ResNet
    #model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    #freeze_layers(model = model, num_blocks_to_freeze = frozen_layers)
    
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    freeze_resnet_layers(model = model, num_blocks_to_freeze = frozen_layers)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                          lr = learning_rate,
                          momentum = momentum,
                          weight_decay = weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = step_size, gamma = gamma)

    main(args = args,
         model = model,
         dataset_function = FGVCAircraft,
         dataset_name = "CUB_200_2011",
         num_classes = num_classes,
         criterion = criterion,
         optimizer = optimizer,
         scheduler = scheduler,
         device = device,
         config = config)
