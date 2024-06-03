# Fine-grained image classification

## Project for the Introduction to Machine Learning course (2023/2024).

This project aims to conduct an exploratory analysis of fine-grained image classification, a complex task in computer vision requiring deep learning models to discern subtle differences between highly similar images. The performance of two convolutional neural network (CNN) architectures and two transformer-based architectures has been compared across three fine-grained datasets. The experiments were conducted using pre-trained models that had been fine-tuned on these datasets. The focus was on evaluating the accuracy and loss of the models. Our findings highlight the significance of model architecture and training strategies in attaining high performance in fine-grained visual tasks.

<br>

<p align="center">
  <img src="https://github.com/andreleo02/deep-dream-team/blob/1fcf8a483e30d40d6c5407b5db2c12fd56e27a82/fine%20grained%20image%20classification.jpeg?raw=true" width="512"/>  
</p>

<p align="center">
  <sub><em>Example of fine grained image classification from [Yafei Wang and Zepeng Wang](https://pdf.sciencedirectassets.com/272324/1-s2.0-S1047320319X00024/1-s2.0-S1047320318303754/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEAYaCXVzLWVhc3QtMSJHMEUCIQCz%2BKA5h9cwX2azQDfzLumdCE4cxGxhYAqJGERnjIjTcgIgHuAfmhWqCY09530%2FM5X9%2FHY3doFKjaov0SmXJnej0GUqvAUIjv%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAFGgwwNTkwMDM1NDY4NjUiDJ2e2phxxPpg1wZC2yqQBVsk0om3WoNQ0nKm3uIz7gWMZdmR%2F4qkuXHtD%2B8QiE2RsWP8o0iVhVkg89CK1fuCessuXDGzk9fWbr2NiH7AaViGoOKCggtHynUkqjh4V0YxYgn9s6%2FvJcmWL9vvG0lTqWEPwUIn9PUNWu8Ao7x30rCtAZTpWrmmnECMSCDRbvTnSSSPG3Ea5IerxDCGFzw7uXi7K6%2FGGsPb7sEHiiz%2F%2B2ZyRsOfQwpaKjvitJ%2FRWkZZePoNvAcSdfLAAhuTtXn9%2B6O1qMCMZIcMUtYk6i1IHz51y5sFP7Br1kVRtPk%2FV9g%2B5q3eSBjX5Pu3%2FKaGUKdkAFgY%2FqTeHMUO%2B3v4thqmGOWjSn8TFhjfQzw%2B76LPEW4ngAdl%2FgUYca0%2FxJsl0%2Bg1ewvBhtEoWfBSfBcgznDpfkeJjioqQXPMnnBflKX7a9TOMnXJnnZA%2FJ9fXApZRbguw3MlWUlYj9hIh2RfWhrvStzauHAHO6CwFTpK7tbuz0LBgTS0kIBAjCql2DTrA8tv%2BI%2Bys3yqT8EH6i3DGQPSZ1AM%2FvdA%2BW2XZFChVzXfLtBWF2sZfQmJsAo0SCHfNBkXQ%2BRRCINJCmeTYvbYteLKq%2B%2BpfaeTAucWlr%2FNn4BokfUFjkmoF5gx9VCtwXhV1HVsmEKAwvkvRDcOvt%2Bd1vn732YMbYDsNaSa9AgHcmy0KxdPEAjHTcWGQAHuxjfbod%2FVNUZioH5PZ0fF6EUEXFQ6esCe%2BUSgTDF8mmpJaNsoZkz5%2FLwiCdyrDpD7fML8U13qbLAMWYLoyk5aMIFG383OUPfGIjMHYL8Q6C9XXPzeLlZ%2Ffz%2BfJ3FUaKkXa%2Fjiqi5%2FhhhtrQz17qCM2zvJFlzZ%2FrmnQLqZWajJr81YrnkRfNLCMJeC97IGOrEBdmsel7yf9rz9M4hwqLt8GwUJFQhVDbhvsS02muW24ZB2nu4a3rXg2KSQkc1M37vzr2knyikwxCllhB57UqbE9uSqjZkCbJsBr%2BXp9CH4tZvwrw39wgOBjNK%2F1JBKDfOKcbiKq2g5uQSEq8JBN%2FB68NUG3UuQX1xe22EyAH3YWsBwwGUcOzrdSO7dqnLwdzIQBDWFHLpSd6hRlb49%2F9UL6tq%2FSgWeT0Eeaub%2BJqtv7s1g&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240603T140024Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY6KCGA5HN%2F20240603%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=ae0f74adfbeca781c732a86fe7f312273c1fd7608010528ea81c3b2296e364c4&hash=2050336335496316480631d7a859b6975a338853ccaeb43b0065beb4a231bbdb&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S1047320318303754&tid=spdf-9f16245d-d302-4e97-927d-cb784645bf9c&sid=0cdf1f931a8545400689b6a0b1593e1ca004gxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=0f10575e065152525d07&rr=88e0358d8b7759fb&cc=it) (2019),"A survey of recent work on fine-grained image classification techniques".</em></sub>
</p>


## Models 
To conduct our experiments on fine-grained image classification, we have selected four models, two belonging to the family of convolutional neural networks  and two belonging to the family of transformers, for comparative purposes:

- **[EfficientNetV2](https://github.com/andreleo02/deep-dream-team/tree/7964a7d63d8beab4f713f7030f3412d59899445c/src/models/EfficientNetV2)**
- **[ResNEt34](https://github.com/andreleo02/deep-dream-team/tree/c80422b86efe3ef2454dc738407a3fa4863da757/src/models/ResNEt34)**
- **[SwinTransformer](https://github.com/andreleo02/deep-dream-team/tree/c80422b86efe3ef2454dc738407a3fa4863da757/src/models/SwinTransformer)**
- **[ViT-16](https://github.com/andreleo02/deep-dream-team/tree/c80422b86efe3ef2454dc738407a3fa4863da757/src/models/ViT-16)**

If something is missing in this guide, please feel free to open an issue on this repo.


## Experiments 
To conduct this analysis on fine-grained visual classification, we evaluated the performance of our models on four very popular datasets in the field of computer vision, specifically chosen for fine-grained tasks like the present one.
- **[CUB 200 2011](https://www.vision.caltech.edu/datasets/cub_200_2011/)**
- **[Oxford Flowers 102](https://pytorch.org/vision/0.17/generated/torchvision.datasets.Flowers102.html)** 
- **[FGVC Aircraft](https://pytorch.org/vision/0.17/generated/torchvision.datasets.FGVCAircraft.html)** 

The results with the comment of the work can be found on the [paper]().

  
## Steps to follow to reproduce our experiments 

1. **Clone the repository**
    ```sh
    git clone https://github.com/andreleo02/deep-dream-team.git
    ```

2. **Install the requirements**:

    ```sh
    pip install -r requirements.txt
    ```

3. **Install the CUDA package** (if using GPU):

    Follow [instructions](https://pytorch.org/get-started/locally/).

4. **Track results with Weights & Biases (wandb)**:

    - Create a profile on [wandb](https://wandb.ai/).
    - On the first run with `wandb` config flag set to `True`, you'll be asked to insert an API KEY. Generate it from the `Settings` section of your wandb account.

5. **Run experiments**:

    To replicate experiments on these models and datasets iside a model folder, use       the following command:

    ```sh
    python main.py --config ./config.yml --run_name <run_name>
    ```

    Feel free to play with the parameters in the `config.yml` and have fun!

---


## Guidelines to download datasets
### Manual Download

The datasets can be manually downloaded and added to the `src/data` folder. This folder is however **ignored by git** and so it will only exists in the local environment.

To keep the process of training the models as smooth as possible, some functions to download libraries directly from the code are defined in the `utils.py` file. Datasets can be downloaded from web (`.zip` and `.tgz`).

> [!TIP]
> To enable the download of a custom dataset, in the `data` section of the `config.yml` file the field `custom` must be set to `True` and the url of the dataset must be specified in the `download_url` field. Specify also the `dataset_name` field with the name of the compressed download folder.

To choose a dataset from `torchvision`, set the `custom` field to `False`. The dataset function must be specified inside the `main.py` file of the model (see `SwinTransformer` model).

### Choosing a Dataset from `torchvision`

To choose a dataset from `torchvision`, set the `custom` field to `False`. The dataset function must be specified inside the `main.py` file of the model (see the `SwinTransformer` model for an example).

## Training a New Model (Pre-trained model from PyTorch is recommended)

Follow these steps:

1. **Select a pre-trained model** from [PyTorch](https://pytorch.org/vision/stable/models.html#classification) or create your own.

2. **Create a folder for the model** in the `models` directory.

3. **Inside the new folder, create three files**:
   - `config.yml`
   - `main.py`
   - `README.md` (to clarify what the model does)

4. **Optional**: Create a custom function to freeze some layers based on the model.

5. **Define the proper components**:
   - Loss function
   - Optimizer
   - Scheduler

6. **Specify the run parameters** in the `config.yml` file as preferred:
   - In the `data` section, choose to download a dataset not available directly from `torchvision`.
   - Set `wandb` parameter to `False` to avoid tracking results on the wandb personal profile.

7. **Call the main function** with the required parameters (example available in the `SwinTransformer` folder).

8. **Run the model**:
   - From the terminal, navigate to the model's folder.
   - Run the following command:

   ```sh
   python main.py --config ./config.yml --run_name <run_name>
   ```

## Authors

- [Borsi Sonia](https://github.com/SoniaBorsi/)
- [Leoni Andrea](https://github.com/andreleo02/)
- [Mbarki Mohamed ](https://github.com/mbarki-mohamed/)
