# Fine-grained image classification

## Project for the Introduction to Machine Learning course (2023/2024).

This project aims to conduct an exploratory analysis of fine-grained image classification, a complex task in computer vision requiring deep learning models to discern subtle differences between highly similar images. The performance of two convolutional neural network (CNN) architectures and two transformer-based architectures has been compared across three fine-grained datasets. The experiments were conducted using pre-trained models that had been fine-tuned on these datasets. The focus was on evaluating the accuracy and loss of the models. Our findings highlight the significance of model architecture and training strategies in attaining high performance in fine-grained visual tasks.

<br>

<p align="center">
  <img src="https://github.com/andreleo02/deep-dream-team/blob/939a8f82ba51ad641e39d62bb95e40f5309fd958/fine%20grained%20image%20classification%20on%20flowers.png?raw=true" width="512"/>  
</p>

<p align="center">
  <sub><em>Example of fine grained image classification on flowers: object present smaller differences between them and hace a high degree of similarity.</em></sub>
</p>


## Models 
To conduct our experiments on fine-grained image classification, we have selected four models, two belonging to the family of convolutional neural networks  and two belonging to the family of transformers, for comparative purposes:

- **[EfficientNetV2](https://github.com/andreleo02/deep-dream-team/tree/9027f3385f4c53f2c438b2e9372e96980558f2dc/src/models/EfficientNetV2)**
- **[ResNEt34](https://github.com/andreleo02/deep-dream-team/tree/9027f3385f4c53f2c438b2e9372e96980558f2dc/src/models/ResNEt34)**
- **[SwinTransformer](https://github.com/andreleo02/deep-dream-team/tree/9027f3385f4c53f2c438b2e9372e96980558f2dc/src/models/SwinTransformer)**
- **[ViT-16](https://github.com/andreleo02/deep-dream-team/tree/9027f3385f4c53f2c438b2e9372e96980558f2dc/src/models/ViT-16)**

If something is missing in this guide, please feel free to open an issue on this repo.


## Experiments 
To conduct this analysis on fine-grained visual classification, we evaluated the performance of our models on three very popular datasets in the field of computer vision, specifically chosen for fine-grained tasks like the present one.
- **[CUB 200 2011](https://www.vision.caltech.edu/datasets/cub_200_2011/)**
- **[Oxford Flowers 102](https://pytorch.org/vision/0.17/generated/torchvision.datasets.Flowers102.html)** 
- **[FGVC Aircraft](https://pytorch.org/vision/0.17/generated/torchvision.datasets.FGVCAircraft.html)**


The results with the comment of the work can be found on the [paper]().

  
## Repository guide

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
The datasets can be manually downloaded and added to the `src/data` folder. This folder is however **ignored by git** and so it will only exists in the local environment. To keep the process of training the models as smooth as possible, some functions to download libraries directly from the code are defined in the `utils.py` file. Datasets can be downloaded from web (`.zip` and `.tgz`).

### Torchvision datasets
To choose a dataset from `torchvision`, set the `custom` field to `False`. The dataset function must be specified inside the `main.py` file of the model (see `SwinTransformer` model).

> [!TIP]
> To enable the download of a custom dataset, in the `data` section of the `config.yml` file the field `custom` must be set to `True` and the url of the dataset must be specified in the `download_url` field. Specify also the `dataset_name` field with the name of the compressed download folder.


--- 

## Authors

- [Borsi Sonia](https://github.com/SoniaBorsi/)
- [Leoni Andrea](https://github.com/andreleo02/)
- [Mbarki Mohamed ](https://github.com/mbarki-mohamed/)
