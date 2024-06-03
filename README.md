# Fine-grained image classification

## Project for the Introduction to Machine Learning course (2023/2024).

Fine-grained image classification involves distinguishing between visually similar subcategories within a larger category. This task is particularly challenging due to the subtle differences in appearance that define each subcategory. Our project aims to address this challenge by employing state-of-the-art deep learning techniques and leveraging transfer learning from pre-trained models.
![Fine-grained-classification-vs-general-image-classification-Finegrained-classification png](https://github.com/andreleo02/deep-dream-team/assets/159782399/6194b503-d2fb-4af1-a558-ca13bae36efc)

If something is missing in this guide, please feel free to open an issue on this repo.

## Experiments 
For this project we conducted experiments testing our four models on 3 different datasets:


- Oxford Flowers 102: The images have large scale, pose and light variations. In addition, there are categories that have large variations within the category and several very similar categories.

  
- FGVC-Aircraft dataset: Aircraft, and in particular airplanes, are alternative to objects typically considered for fine-grained categorization such as birds and pets.

  
# Steps to follow

## Prepare the environment

After cloning the repository, install the requirements using the following command.

```
pip install -r requirements.txt
```

Additionally, to run the experiments using _GPU_, download also the `cuda` package following these [instructions](https://pytorch.org/get-started/locally/).

If you want to keep track of the results, it is suggested to create a profile on [wandb](https://wandb.ai). On the first run with the `wandb` config flag set to `True`, will be asked to insert an API KEY for the profile. This can be generated from the `Settings` section of the account.

## Stable models and datasets

The authors of the repository, trained and validated the following models:

- **EfficientNetV2**
- **ResNEt34**
- **SwinTransformer**
- **ViT-16**

In particular, the datasets used for the experiments are:

- **CUB 200 2011**: one of the most widely used dataset for fine-grained visual categorisation tasks. It contains 11,788 images of 200 subcategories belonging to birds, 5,994 for training and 5,794 for testing. All images are annotated with bounding boxes, part locations, and attribute labels. Images and annotations were filtered by multiple users of Mechanical Turk.
![accuracy cub.jpg](https://github.com/andreleo02/deep-dream-team/blob/7b40d64b2caa0d20ed388f90ad845a18453d3956/accuracy%20cub.jpg)

- **Oxford Flowers 102** (from pytorch): It consists of 102 flower categories. The flowers selected for inclusion in the study are those that are commonly observed in the United Kingdom. Each class comprises between 40 and 258 images.  The images have large scale, pose and light variations. In addition, there are categories that have large variations within the category and several very similar categories.
![accuracy flowers.jpg](https://github.com/andreleo02/deep-dream-team/blob/7b40d64b2caa0d20ed388f90ad845a18453d3956/accuracy%20flowers.jpg)

- **FGVC Aircraft** (from pytorch): It contains 10,200 images of aircraft, with 100 images for each of 102 different aircraft model variants. Aircraft, and in particular airplanes, are alternative to objects typically considered for fine-grained categorization such as birds and pets.
![accuracy aircrafts.jpg](https://github.com/andreleo02/deep-dream-team/blob/7b40d64b2caa0d20ed388f90ad845a18453d3956/accuracy%20aircrafts.jpg)

-**Mammalia**: It contains 100 different classes of mammals. For the training set, each class of the dataset contained 50 images, giving a total of 5000 images, while the test set contained 10 images per class, giving 1000 test images. To obtain the validation set, 20\% of the training set was selected randomly.
![accuracy mammalia.jpg](https://github.com/andreleo02/deep-dream-team/blob/7b40d64b2caa0d20ed388f90ad845a18453d3956/accuracy%20mammalia.jpg)

The results with the comment of the work can be found on the [paper]().

To replicate experiments on these models and datasets, run the following command inside a model folder:

```
python main.py --config ./config.yml --run_name <run_name>
```

Play with the parameters in the `config.yml` and have fun!

## Guidelines to download datasets

The datasets can be manually downloaded and added to the `src/data` folder. This folder is however **ignored by git** and so it will only exists in the local environment.

To keep the process of training the models as smooth as possible, some functions to download libraries directly from the code are defined in the `utils.py` file. Datasets can be downloaded from web (`.zip` and `.tgz`).

> [!TIP]
> To enable the download of a custom dataset, in the `data` section of the `config.yml` file the field `custom` must be set to `True` and the url of the dataset must be specified in the `download_url` field. Specify also the `dataset_name` field with the name of the compressed download folder.

To choose a dataset from `torchvision`, set the `custom` field to `False`. The dataset function must be specified inside the `main.py` file of the model (see `SwinTransformer` model).

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
