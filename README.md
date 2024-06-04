# Fine-Grained Image Classification

## Computer Vision Project

This project aims to conduct an exploratory analysis of fine-grained image classification, a complex task in computer vision requiring deep learning models to discern subtle differences between highly similar images. The performance of two convolutional neural network (CNN) architectures and two transformer-based architectures has been compared across three fine-grained datasets. The experiments were conducted using pre-trained models that had been fine-tuned on these datasets. The focus was on evaluating the accuracy and loss of the models. Our findings highlight the significance of model architecture and training strategies in attaining high performance in fine-grained visual tasks.

<br>

<p align="center">
  <img src="https://github.com/andreleo02/deep-dream-team/blob/438b36ca8076da82890a56df2ff4e28dcfca60e4/Fine-grained-classification-vs-general-image-classification-Finegrained-classification.png.jpeg?raw=true" width="512"/>  
</p>

<p align="center">
  <sub><em>Fine grained classification vs general image classification: fine grained classification aims to distinguish between very similar object (red box), while general image classification usually aims to distinguish distinct objects. From Aoxue Li et al.(2017), "Zero-Shot Fine-Grained Classification by Deep Feature Learning with Semantics".</em></sub>
</p>

## Project structure

```
┌─ requirements.txt
├─ src/
│  ├─ models/
│  │  ├─ EfficientNetV2/
│  │  │  ├─ config.yml
│  │  │  ├─ EfficientNet.png
│  │  │  ├─ main.py
│  │  │  └─ README.md
│  │  ├─ ResNet34/
        ...
│  │  ├─ SwinTransformer/
        ...
│  │  ├─ ViT-16/
        ...
│  ├─ data_utils.py
│  ├─ training_utils.py
│  └─ utils.py

```
- **`requirements.txt`**: list of dependencies required to run the project.
  
In the **`src`** folder you will find: 

- **`models`**: containing subdirectories for different models used for our experiments; each model has its **`README.md`** with a brief description of its srchitecture, a **`main.py`** containing the script to run the model and a **`config.yml`** with all the paramters of the model.
  
- **`data_utils.py`**: utility functions for data handling and preprocessing.
  
- **`training_utils.py`**: utility functions for model training processes.
  
- **`utils.py`**: general utility functions used across the project.

## Models

To conduct our experiments on fine-grained image classification, we have selected four models, two belonging to the family of convolutional neural networks and two belonging to the family of transformers, for comparative purposes:

- **[EfficientNetV2](https://github.com/andreleo02/deep-dream-team/tree/9027f3385f4c53f2c438b2e9372e96980558f2dc/src/models/EfficientNetV2)**
- **[ResNEt34](https://github.com/andreleo02/deep-dream-team/tree/9027f3385f4c53f2c438b2e9372e96980558f2dc/src/models/ResNEt34)**
- **[SwinTransformer](https://github.com/andreleo02/deep-dream-team/tree/9027f3385f4c53f2c438b2e9372e96980558f2dc/src/models/SwinTransformer)**
- **[ViT-16](https://github.com/andreleo02/deep-dream-team/tree/9027f3385f4c53f2c438b2e9372e96980558f2dc/src/models/ViT-16)**

If something is missing in this guide, please feel free to open an issue on this repo.

## Experiments

To conduct this analysis on fine-grained visual classification, we evaluated the performance of our models on four very popular datasets in the field of computer vision, specifically chosen for fine-grained tasks like the present one.

- **[CUB 200 2011](https://www.vision.caltech.edu/datasets/cub_200_2011/)**
  <br>

<p align="center">
  <img src="https://github.com/andreleo02/deep-dream-team/blob/7cc62f96cf4fd7e4a9c9739abd99227fb38c140c/CUB-200-2011-0000000109-6e01ce73_vMleyYb.jpeg" width="350"/>  
</p>

<p align="center">
  <sub><em>CUB-200-2011 is an extended version of CUB-200. The extended version roughly doubles the number of images per category and adds new part localization annotations. All images are annotated with bounding boxes, part locations, and at- tribute labels. Images and annotations were filtered by multiple users of Mechanical Turk.</em></sub>
</p>

- **[Oxford Flowers 102](https://pytorch.org/vision/0.17/generated/torchvision.datasets.Flowers102.html)**
<br>

<p align="center">
  <img src="https://github.com/andreleo02/deep-dream-team/blob/d48b5728ed9e2bbc55d0092f96d69c0ceade84eb/Image-examples-from-Flowers102-dataset.jpg?raw=true" width="450"/>  
</p>

<p align="center">
  <sub><em>Flowers 102 dataset: The images have large scale, pose and light variations. In addition, there are categories that have large variations within the category and several very similar categories.</em></sub>
</p>
  
- **[FGVC Aircraft](https://pytorch.org/vision/0.17/generated/torchvision.datasets.FGVCAircraft.html)**
<br>

<p align="center">
  <img src="https://github.com/andreleo02/deep-dream-team/blob/49d700167aef624473f743a45e6aee2f117570c3/FGVC-Aircraft-0000003405-c35d29b7.jpg?raw=true" width="350"/>  
</p>

<p align="center">
  <sub><em>FGVC-Aircraft dataset: Aircraft, and in particular airplanes, are alternative to objects typically considered for fine-grained categorization such as birds and pets.</em></sub>
</p>

- **[Mammalia]()** 
<p>
  <sub><em> Note: Mammalia dataset is not publicly available but was used in the context of the competition of the Introduction to Machine Learning Course and provided by the University of Trento. This dataset contains 100 different classes of mammals. For more detailed information please consult our paper. </em></sub>
</p>


## Results 
During our experiments we trained and validated each ***model*** on each ***dataset*** and compared their performances. The results of our experiments demonstrated that EfficientNet consistently exhibited the highest accuracy and lowest loss across the different datasets. However, SwinT also exhibited promising results, indicating the potential of transformers for image classification. Both of these models exhibited an optimal balance between complexity and efficiency. SwinT also exhibited the best performance in comparison to ViT16. In contrast, ResNet, despite being a deep and effective architecture, exhibited poorer results compared to EfficientNet.


**- VALIDATION ACCURACY**


| Model | CUB | Flowers | Aircrafts | Mammalia|
| :---: | :---: | :---: | :---: | :---: | 
| ResNet34 | 97.77 | 94.22 | 66.69 | 50.58 | 
| EfficientNetV2 | 99.91 | 95.80 | 76.49 | 66.11 | 
| ViT16 | 98.36 | 88.01 | 44.99 | 59.96 | 
| Swin-T | 98.18 | 94.12 | 71.39 | 66.50 | 

**- VALIDATION LOSS**


| Model | CUB | Flowers | Aircrafts | Mammalia|
| :---: | :---: | :---: | :---: | :---: | 
| ResNet34 | 0.10 | 0.28 | 1.09 | 2.11 | 
| EfficientNetV2 | 0.015 | 0.18| 0.78 | 1.42 | 
| ViT16 | 0.13 | 0.46 | 2.23 | 1.70 | 
| Swin-T | 0.24 | 0.21 | 1.12 | 1.98 | 


<table border="0">
<tr>
    <td>
    <img src="https://github.com/andreleo02/deep-dream-team/blob/c8e259caa035a53fc7466147aefee016cd32efbf/accuracy%20aircrafts.jpg" width="100%" />
    </td>
    <td>
    <img src="https://github.com/andreleo02/deep-dream-team/blob/c8e259caa035a53fc7466147aefee016cd32efbf/accuracy%20cub.jpg", width="100%" />
    </td>
</tr>
</table>

 More details about the results of our experiments (including information about the training phase) for each model can be found in our [paper](https://github.com/andreleo02/deep-dream-team/blob/8e4b56d933e07fccc3834fa6e1ea1f2b7000dcdf/PAPER.pdf).

<p>
  <sub><em> Note: for the mammalia dataset, in the context of the competition of the Introduction to Machine Learning Course (University of Trento, y. 2024), we also trained ResNet50 and SwinB. More datails about these specific runs can be found in the paper in the section "Competition".</em></sub>
</p>

  
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

   To replicate experiments on these models and datasets iside a model folder, use the following command:

   ```sh
   python main.py --config ./config.yml --run_name <run_name>
   ```

   Feel free to play with the parameters in the `config.yml` and have fun!

---
## How to use different models

Follow these steps:

1. Choose one of the pre-trained models available in [PyTorch](https://pytorch.org/vision/stable/models.html#classification).

2. In the `models` directory, create a new folder for your selected model.

3. Inside the newly created folder, add the following files:
   - `config.yml`
   - `main.py`
   - `README.md`

4. Specify the run parameters in the `config.yml` file.

Here's a sample directory structure:
```
models/
├── YourModelName/
│   ├── config.yml
│   ├── main.py
│   └── README.md
```

---

## How to use different datasets

### Custom datasets
The datasets can be manually downloaded and added to the `src/data` folder. This folder is however **ignored by git** and so it will only exists in the local environment. To keep the process of training the models as smooth as possible, some functions to download libraries directly from the code are defined in the `utils.py` file. Datasets can be downloaded from web (`.zip` and `.tgz`).

> [!TIP]
> To enable the download of a custom dataset, in the `data` section of the `config.yml` file the field `custom` must be set to `True` and the url of the dataset must be specified in the `download_url` field. Specify also the `dataset_name` field with the name of the compressed download folder.

### Torchvision datasets
To choose a dataset from `torchvision`, set the `custom` field to `False`. The dataset function must be specified inside the `main.py` file of the model (see `SwinTransformer` model).


--- 

## Authors

- [Borsi Sonia](https://github.com/SoniaBorsi/)
- [Leoni Andrea](https://github.com/andreleo02/)
- [Mbarki Mohamed ](https://github.com/mbarki-mohamed/)
