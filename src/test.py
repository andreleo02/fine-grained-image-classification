# from utils import download_dataset_tgz

# download_dataset_tgz(url = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz", output_dir = "./data/cub")
import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image


train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(size = (224, 224), antialias = True),
        transforms.RandomHorizontalFlip(p = 0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
    ])

test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
    ])


# Function to get label IDs from the training dataset
def get_label_ids(train_dir):
    label_ids = []
    classes = sorted(os.listdir(train_dir))
    for class_name in classes:
        if os.path.isdir(os.path.join(train_dir, class_name)):
            class_id = class_name.split('.')[0]
            label_ids.append(class_id)
    return label_ids
class TestDataset(Dataset):
    def __init__(self, test_dir, transform=None):
        self.test_dir = test_dir
        self.image_files = [f for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.test_dir, self.image_files[idx])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return image, self.image_files[idx]

train_dir = "./data/competition_cub/train"
test_dir = "./data/competition_cub/test"

train_data = datasets.ImageFolder(root = train_dir, transform = train_transforms)
num_train = int(len(train_data) * 0.8)
num_val = len(train_data) - num_train
train_dataset, val_dataset = random_split(train_data, [num_train, num_val])

test_dataset = TestDataset(test_dir, transform = test_transforms)

train_loader = DataLoader(train_data, batch_size = 1, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = 1, shuffle = False)
test_loader = DataLoader(test_dataset, batch_size = 1, shuffle = False)


for images, labels in train_loader:
    print("do training")

for images, labels in val_loader:
    print("do validation")

for images, image_file_names in test_loader:
    # for i in range(len(images)):
        # output = model(images[i])
        # preds[image_file_names[i]] = label_ids[output]
    print("do test")
