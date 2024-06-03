import os, requests, zipfile, tarfile, shutil, logging
import torchvision.datasets as datasets

from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def get_data(dataset_function, train_transforms, val_transforms):
    dataset_path = "../../data/"
    train_dataset = dataset_function(root = dataset_path, split = "train", transform = train_transforms, download = True)
    val_dataset = dataset_function(root = dataset_path, split = "val", transform = val_transforms, download = True)
    test_dataset = dataset_function(root = dataset_path, split = "test", transform = val_transforms, download = True)

    return train_dataset, val_dataset, test_dataset

def get_data_custom(dataset_name, download_url: str, num_classes, train_transforms, val_transforms):
    data_dir = os.path.join("../../data", dataset_name)
    if os.path.isdir(data_dir):
        logger.info(f"Dataset {dataset_name} already exists at path {data_dir}. Not downloading")
    else:
        if download_url.endswith(".tgz"):
            download_dataset_tgz(url = download_url)
        else:
            download_dataset_zip(url = download_url)

    images_dir = os.path.join(data_dir, 'images')
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    with open(os.path.join(data_dir, 'images.txt')) as f:
        images = [line.strip().split() for line in f.readlines()]

    with open(os.path.join(data_dir, 'image_class_labels.txt')) as f:
        labels = [line.strip().split() for line in f.readlines()]

    with open(os.path.join(data_dir, 'train_test_split.txt')) as f:
        train_test_split_dataset = [line.strip().split() for line in f.readlines()]

    os.makedirs(train_dir, exist_ok = True)
    os.makedirs(val_dir, exist_ok = True)

    for i in range(1, num_classes):
        os.makedirs(os.path.join(train_dir, str(i)), exist_ok = True)
        os.makedirs(os.path.join(val_dir, str(i)), exist_ok = True)

    file_paths = [os.path.join(images_dir, img[1]) for img in images]
    labels = [int(label[1]) for label in labels]
    train_test_split_dataset = [int(split[1]) for split in train_test_split_dataset]

    data = list(zip(file_paths, labels, train_test_split_dataset))

    train_data, val_data = train_test_split([item for item in data if item[2] == 1], test_size = 0.2, stratify = [item[1] for item in data if item[2] == 1])

    def copy_files(data, target_dir):
        for file_path, label, _ in data:
            target_class_dir = os.path.join(target_dir, str(label))
            shutil.copy(file_path, target_class_dir)

    copy_files(train_data, train_dir)
    copy_files(val_data, val_dir)

    train_dataset = datasets.ImageFolder(train_dir, transform = train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform = val_transforms)

    return train_dataset, val_dataset

def download_dataset_zip(url: str, output_dir: str = "../../data") -> None:
    response = requests.get(url)
    
    if response.status_code == 200:
        with open("tmp.zip", "wb") as f:
            f.write(response.content)
        try:
            with zipfile.ZipFile("tmp.zip", "r") as zip_ref:
                zip_ref.extractall(output_dir)
            logger.info(f"Download and extracted from {url} completed successfully!")
            pass
        except zipfile.BadZipFile as e: 
            logger.error(f"Exception while extracting files from zip. Zip downloaded from {url}. Exception {e}")
            pass
        try:
            os.remove("tmp.zip")
        except Exception as e:
            logger.error(f"Error occurred: {e}")
    else:
        logger.error(f"Failed to download file. Status code: {response.status_code}")

def download_dataset_tgz(url: str, output_dir: str = "../../data") -> None:
    logger.info(f"Downloading dataset from {url} ...")
    response = requests.get(url)
    
    if response.status_code == 200:
        logger.info("Download completed. Extracting files ...")
        with open("tmp.tgz", "wb") as f:
            f.write(response.content)
        try:
            with tarfile.open("tmp.tgz", "r:gz") as tgz_ref:
                tgz_ref.extractall(output_dir)
            logger.info(f"Download and extracted from {url} completed successfully! Extracted files in {output_dir}")
            pass
        except tarfile.TarError as e: 
            logger.error(f"Exception while extracting files from tgz. Tar downloaded from {url}. Exception {e}")
            pass
        try:
            os.remove("tmp.tgz")
        except Exception as e:
            logger.error(f"Error occurred: {e}")
    else:
        logger.error(f"Failed to download file. Status code: {response.status_code}")
