from utils import download_dataset_tgz, get_data

# download_dataset_tgz(url = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz", output_dir = "./data/cub")

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import FGVCAircraft, Flowers102

from torch.utils.data import Dataset
from PIL import Image
import os

class CustomDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = os.listdir(image_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, img_name


train_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
    ])

# val_transforms = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
# ])
# train_dataset, val_dataset = get_data(dataset_function = FGVCAircraft,
#                                               train_transforms = train_transforms,
#                                               val_transforms = val_transforms)
train_dataset = CustomDataset(image_dir = "./data/fgvc-aircraft-2013b/data/images", transform = train_transforms)
dataloader = DataLoader(train_dataset, batch_size = 32, num_workers = 4, shuffle = True)
res = {}
for images, img_ids in dataloader:
    for id in img_ids:
        res[id] = 1
    break
print(res)
