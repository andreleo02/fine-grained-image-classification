from kaggle.api.kaggle_api_extended import KaggleApi
import os, requests, zipfile, tarfile

def download_dataset_from_kaggle(dataset_name: str, output_dir: str = "dataset") -> None:
    api = KaggleApi()
    api.authenticate()
    final_output_dir: str = "src/data/" + output_dir
    api.dataset_download_files(dataset = dataset_name, path = final_output_dir, unzip = True)
    print(f"Dataset {dataset_name} downloaded correctly from Kaggle")

def download_dataset_zip(url: str, output_dir: str = "dataset") -> None:
    response = requests.get(url)
    final_output_dir: str = "src/data/" + output_dir
    
    if response.status_code == 200:
        with open("tmp.zip", "wb") as f:
            f.write(response.content)
        try:
            with zipfile.ZipFile("tmp.zip", "r") as zip_ref:
                zip_ref.extractall(final_output_dir)
            print(f"Download and extracted from {url} completed successfully!")
            pass
        except zipfile.BadZipFile as e: 
            print(f"Exception while extracting files from zip. Zip downloaded from {url}. Exception {e}")
            pass
        try:
            os.remove("tmp.zip")
        except Exception as e:
            print(f"Error occurred: {e}")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")

def download_dataset_tgz(url: str, output_dir: str = "dataset") -> None:
    response = requests.get(url)
    final_output_dir: str = "src/data/" + output_dir
    
    if response.status_code == 200:
        with open("tmp.tgz", "wb") as f:
            f.write(response.content)
        try:
            with tarfile.open("tmp.tgz", "r:gz") as tgz_ref:
                tgz_ref.extractall(final_output_dir)
            print(f"Download and extracted from {url} completed successfully!")
            pass
        except tarfile.TarError as e: 
            print(f"Exception while extracting files from tgz. Tar downloaded from {url}. Exception {e}")
            pass
        try:
            os.remove("tmp.tgz")
        except Exception as e:
            print(f"Error occurred: {e}")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")
