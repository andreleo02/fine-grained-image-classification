import argparse, sys, os, yaml
from torchvision.models import swin_t, Swin_T_Weights
from torchvision.datasets import FGVCAircraft, Flowers102 
import torch.nn as nn
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

    num_classes = 102

    model = load_model(swin_t, weights = Swin_T_Weights.IMAGENET1K_V1)

    model.head = nn.Linear(model.head.in_features, num_classes)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    learning_rate = config["training"]["lr"]
    frozen_layers = config["training"]["frozen_layers"]
    momentum = config["training"]["optimizer"]["momentum"]
    step_size = config["training"]["scheduler"]["step_size"]
    gamma = config["training"]["scheduler"]["gamma"]

    freeze_layers(model = model, num_blocks_to_freeze = frozen_layers)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    learning_rate = config["training"]["lr"]
    

    criterion = nn.CrossEntropyLoss().to(device = device)
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr = learning_rate, momentum = momentum)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = step_size, gamma = gamma)
    
    main(args = args,
         model = model,
         dataset_function = Flowers102 ,
         dataset_name = "Flowers102 ",
         criterion = criterion,
         optimizer = optimizer,
         scheduler = scheduler,
         device = device,
         config = config)
    
