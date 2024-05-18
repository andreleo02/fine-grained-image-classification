import argparse, sys, os, yaml
from torchvision.models import swin_t, Swin_B_Weights
from torchvision.datasets import FGVCAircraft
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.utils import main, load_model

if __name__ == "__main__":

    parser = argparse.ArgumentParser("")
    parser.add_argument("--config", required = True, type = str, help = "Path to the configuration file")
    parser.add_argument("--run_name", required = False, type = str, help = "Name of the run")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(swin_t)

    num_classes = 100

    model.head = nn.Linear(model.head.in_features, num_classes)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    learning_rate = config["training"]["lr"]
    

    criterion = nn.CrossEntropyLoss().to(device = device)
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
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
    
