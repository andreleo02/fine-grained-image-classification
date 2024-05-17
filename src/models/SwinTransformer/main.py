import argparse, sys, os
from torchvision.models import swin_b, Swin_B_Weights
from torchvision.datasets import Flowers102

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.utils import main

if __name__ == "__main__":

    parser = argparse.ArgumentParser("")
    parser.add_argument("--config", required = True, type = str, help = "Path to the configuration file")
    parser.add_argument("--run_name", required = False, type = str, help = "Name of the run")
    args = parser.parse_args()
    main(args = args,
         model_function = swin_b,
         weights = Swin_B_Weights.DEFAULT,
         dataset_function = Flowers102,
         dataset_name = "Flowers102",
         num_classes = 102,
         fc = False)
