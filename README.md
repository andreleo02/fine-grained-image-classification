# Fine-grained image classification

Project competition for the Introduction to Machine Learning course (2023/2024). 

Fine-grained image classification involves distinguishing between visually similar subcategories within a larger category. This task is particularly challenging due to the subtle differences in appearance that define each subcategory. Our project aims to address this challenge by employing state-of-the-art deep learning techniques and leveraging transfer learning from pre-trained models.


## How to test a model (pre-trained model from pytorch)

Follow these steps:

1. Select one of the pre-trained models present in [pytorch](https://pytorch.org/vision/stable/models.html#classification).
2. Create a folder for the model in the `models` folder.
3. Inside the new folder, create three files: `config.yml`, `main.py` and `README.md` (to clarify what the model does).
4. Call the main function with the required parameters (there is an example in the `SwinTransformer` folder).
5. Specify the run paramenters in the `config.yml` file.
6. From the terminal, move to the folder of the model and run the following command

```
python main.py --config ./config.yml --run_name <run_name>
```

## Guidelines to download datasets

The datasets can be manually downloaded and added to the `src/data` folder. This folder is however **ignored by git** and so it will only exists in the local environment.

To keep the process of training the models as smooth as possible, some functions to download libraries directly from the code are defined in the `utils.py` file. Datasets can be downloaded in such these ways:

- plain download from web (`.zip` and `.tgz`)
- download from Kaggle (with Kaggle Api)

## Authors
- [Borsi Sonia](https://github.com/SoniaBorsi/)
- [Leoni Andrea](https://github.com/andreleo02/)
- [Mbarki Mohamed ](https://github.com/mbarki-mohamed/)


Finally, datasets from Kaggle can be downloaded calling the function `download_dataset_from_kaggle` and passing as argument the name of the dataset (`<author>/<name>`) and the name of the directory where the dataset will be saved. There is an example call in the `test.py` file.
