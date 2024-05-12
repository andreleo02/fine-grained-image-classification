# deep-dream-team

Project competition for the Introduction to Machine Learning course (2023/2024)

## Guidelines to download datasets

The datasets can be manually downloaded and added to the `src/data` folder. This folder is however **ignored by git** and so it will only exists in the local environment.

To keep the process of training the models as smooth as possible, some functions to download libraries directly from the code are defined in the `utils.py` file. Datasets can be downloaded in such these ways:

- plain download from web (`.zip` and `.tgz`)
- download from Kaggle (with Kaggle Api)

### Download from Kaggle

An extra step is required to download datasets from Kaggle. Follow these steps to use the Kaggle download.

1. **Install Kaggle** with pip.

```
pip install --upgrade kaggle
```

2. Create a **Kaggle account**.

3. In the account settings, look for **API** and click on **Create New Token**. Automatically, a file called `kaggle.json` will be downloaded.

4. Place this file in the location `~/.kaggle/kaggle.json` on your machine. You may need to create the directory and set the correct permissions.

```
mkdir ~/.kaggle
chmod 600 ~/.kaggle/kaggle.json
```

Finally, datasets from Kaggle can be downloaded calling the function `download_dataset_from_kaggle` and passing as argument the name of the dataset (`<author>/<name>`) and the name of the directory where the dataset will be saved. There is an example call in the `test.py` file.
