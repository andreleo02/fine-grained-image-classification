# Fine-grained image classification

## Project for the Introduction to Machine Learning course (2023/2024).

This paper presents an exploratory analysis of fine-grained image classification, a complex task in computer vision requiring deep learning models to discern subtle differences between highly similar images. The performance of two convolutional neural network (CNN) architectures, ResNet34 and EfficientNetV2, and two transformer-based architectures, ViT16 and SwinT, is compared across three fine-grained datasets: The datasets used in this study were FGVC-Aircraft, Oxford 102 Flower, and CUB-200-2011. The experiments were conducted using pre-trained models that had been fine-tuned on these datasets. The focus was on evaluating the accuracy and loss of the models. EfficientNet consistently demonstrated the highest accuracy, indicating that it is a robust and efficient model for fine-grained classification tasks. SwinT also demonstrated promising results, achieving the highest accuracy on the Mammals dataset used for the Introduction to Machine Learning Competition. Our findings highlight the significance of model architecture and training strategies in attaining high performance in fine-grained visual tasks.

<br>

<p align="center">
  <img src="https://github.com/andreleo02/deep-dream-team/blob/9f9672f7b2fb71ec20b1a9eac890e3074ff0ddab/Fine%20grained%20vs%20image%20classification%20.jpeg?raw=true" width="512"/>  
</p>

<p align="center">
  <sub><em>Fine-grained classification vs. general image classification. Detailed description of the image.</em></sub>
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
- **CUB 200 2011**
- **Oxford Flowers 102** (from pytorch)
- **FGVC Aircraft** (from pytorch)
- **Mammalia**

The results with the comment of the work can be found on the [paper]().

  
# Steps to follow

## Prepare the environment

1. **Clone the repository**.

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

## How to train a new model (pre-trained model from pytorch is recommended)

Follow these steps:

1. Select one of the pre-trained models present in [pytorch](https://pytorch.org/vision/stable/models.html#classification) or create your own.
2. Create a folder for the model in the `models` folder.
3. Inside the new folder, create three files: `config.yml`, `main.py` and `README.md` (to clarify what the model does).
4. If needed, create a custom function to freeze some layers based on the model.
5. Define the proper _loss function_, _optimizer_ and _scheduler_.
6. Specify the run parameters in the `config.yml` file as preferred:
   - in the `data` section it can be chosen to download a dataset that is not available directly from `torchvision`
   - the parameter `wandb` can be set to `False` to avoid keeping track of the results on the wandb personal profile
7. Call the main function with the required parameters (there is an example in the `SwinTransformer` folder).
8. From the terminal, move to the folder of the model and run the following command

```
python main.py --config ./config.yml --run_name <run_name>
```

## Authors

- [Borsi Sonia](https://github.com/SoniaBorsi/)
- [Leoni Andrea](https://github.com/andreleo02/)
- [Mbarki Mohamed ](https://github.com/mbarki-mohamed/)
