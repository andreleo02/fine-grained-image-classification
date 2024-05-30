# Fine-grained image classification

Project competition for the Introduction to Machine Learning course (2023/2024).

Fine-grained image classification involves distinguishing between visually similar subcategories within a larger category. This task is particularly challenging due to the subtle differences in appearance that define each subcategory. Our project aims to address this challenge by employing state-of-the-art deep learning techniques and leveraging transfer learning from pre-trained models.
![Fine-grained-classification-vs-general-image-classification-Finegrained-classification png](https://github.com/andreleo02/deep-dream-team/assets/159782399/6194b503-d2fb-4af1-a558-ca13bae36efc)

# Steps to follow

## Prepare the environment

After cloning the repository, install the requirements using the following command.

```
pip install -r requirements.txt
```

If you want to keep track of the results, it is suggested to create a profile on [wandb](https://wandb.ai). On the first run with the `wandb` config flag set to `True`, will be asked to insert an API KEY for the profile. This can be generated from the `Settings` section of the account.

## Stable models and datasets

The authors of the repository, trained and validated the following models:

- **EfficientNetV2**
- **ResNEt34**
- **SwinTransformer**
- **ViT-16**

In particular, the datasets used for the experiments are:

- **CUB 200 2011**
- **Oxford Flowers 102** (from pytorch)
- **FGVC Aircraft** (from pytorch)

The results with the comment of the work can be found on the [paper]().

## How to train a new model (pre-trained model from pytorch is recommended)

Follow these steps:

1. Select one of the pre-trained models present in [pytorch](https://pytorch.org/vision/stable/models.html#classification).
2. Create a folder for the model in the `models` folder.
3. Inside the new folder, create three files: `config.yml`, `main.py` and `README.md` (to clarify what the model does).
4. If needed, create a custom function to freeze some layers based on the model.
5. Specify the run parameters in the `config.yml` file as preferred:
   - in the `data` section it can be chosen to download a dataset that is not available directly from `torchvision`
   - the parameter `wandb` can be set to `False` to avoid keeping track of the results on the wandb personal profile
6. Call the main function with the required parameters (there is an example in the `SwinTransformer` folder).
7. From the terminal, move to the folder of the model and run the following command

```
python main.py --config ./config.yml --run_name <run_name>
```

## Guidelines to download datasets

The datasets can be manually downloaded and added to the `src/data` folder. This folder is however **ignored by git** and so it will only exists in the local environment.

To keep the process of training the models as smooth as possible, some functions to download libraries directly from the code are defined in the `utils.py` file. Datasets can be downloaded from web (`.zip` and `.tgz`).

> To enable the download of a custom dataset, in the `data` section of the `config.yml` file the field `custom` must be set to `True` and the url of the dataset must be specified in the `download_url` field. Specify also the `dataset_name` field with the name of the compressed download folder.

To choose a dataset from `torchvision`, set the `custom` field to `False`. The dataset function must be specified inside the `main.py` file of the model (see `SwinTransformer` model).

## Authors

- [Borsi Sonia](https://github.com/SoniaBorsi/)
- [Leoni Andrea](https://github.com/andreleo02/)
- [Mbarki Mohamed ](https://github.com/mbarki-mohamed/)
