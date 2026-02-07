# GroupCeptionNet: A Lightweight Model for Classifying Chili Seed Germination with Macro Images

This repository contains the **Official PyTorch implementation** of **GroupCeptionNet**.

> **GroupCeptionNet** is a lightweight CNN designed for classifying chili seed germination using macro images. It achieves **94.66% accuracy** with only **1.40M parameters**, offering an efficient solution for precision agriculture.

## Abstract

Chili seed germination potential is a significant element influencing planting yield. Many researchers focus on feature extraction analysis methods to detect undesirable seed germination characteristics and improve classification accuracy. In deep learning, data collection and model complexity often limit practical application in production environments. This work includes a dataset of macroscopic images of chili seeds. This paper proposes a lightweight model GroupCeptionNet to process macroscopic images for screening non-germinated chili seeds before planting. Experimental results show that GroupCeptionNet achieves 94.66% accuracy and 94.65% F1-score under 1.40M parameters, outperforming the classical CNN and Transformer models. We also explore the impact of background removal and GroupCeptionNet ablation and variant structures. The visualization results further validate the consistency of the model's focus areas with human-annotated regions. The proposed dataset and model provide technical references for a low-cost and efficient chili seed screening process.

**Keywords**: Chili Seeds, Germination Classification, Macro Imaging, Lightweight Model, Convolutional Neural Networks, Inception Module


## Dataset Preparation

To use this repository, please organize your dataset as follows:

1.  Create a folder named `full_original_dataset` in the root directory.
2.  Place your original, raw macro images into this folder.

```bash
/GroupCeptionNet
    /full_original_dataset
        /image1.jpg
        /image2.jpg
        ...
    /1_0_process_image.py
    /main.py
    ...
```

## Data Preprocessing

Before training, you need to preprocess the raw images.

### 1. Standard Preprocessing (With Background Removal)
Run the following script to process images and remove backgrounds (Recommended):
```bash
python 1_0_process_image.py
```

### 2. Preprocessing Without Background Removal
If you wish to retain the background (for ablation studies or comparison):
```bash
python 1_1_process_image_without_cutout.py
```

### 3. Dataset Organization (Optional)
The script `2_list_image.py` is used for internal dataset organization and list generation. You can skip this step if the preprocessing scripts handle the split automatically.
```bash
python 2_list_image.py
```

### 4. Dataset Statistics
To calculate and view statistical information about the dataset (e.g., mean, std, class distribution):
```bash
python 3_seed_statistics.py
```

## Training

You can start training the model using `main.py`. The script supports various command-line arguments to control the training process.

### Example Usage
To train the model with specific configurations:

```bash
python main.py --train \
    --batch_size 32 \
    --optimizer sgd \
    --model GroupCeptionNet \
    --pretrained False \
    --dataset process_dataset \
    --experiment_name 1_1_GroupCeptionNet_b32_sp622o_sgd1e-4_cutout
```

> **Note:** The specific arguments (e.g., `--model`, `--optimizer`, `--batch_size`) can be adjusted. Please refer to `main.py` for the complete definition of all available parameters and model options.

## Results

Comparison of the performance of various models on the chili seed germination dataset:

| Model | Parameters (M) | Accuracy (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: |
| **GroupCeptionNet** | **1.40** | **94.66** | **94.65** |
| VGG13 | 128.96 | 94.40 | 94.39 |
| VGG11 | 128.81 | 94.14 | 94.13 |
| VGG16 | 134.27 | 93.62 | 93.60 |
| ResNet18 | 11.17 | 93.62 | 93.60 |
| VGG19 | 139.58 | 93.49 | 93.47 |
| InceptionV3 | 21.78 | 93.23 | 93.21 |
| ResNeXt18 | 11.16 | 92.84 | 92.82 |
| AlexNet | 57.01 | 92.58 | 92.55 |
| SwinTransformerV2 | 27.52 | 92.45 | 92.44 |
| ResNet50 | 23.51 | 92.45 | 92.43 |
| ResNet34 | 21.28 | 92.45 | 92.42 |
| MobileNetV2 | 2.22 | 92.32 | 92.30 |
| ResNet101 | 42.50 | 92.06 | 92.03 |
| GoogLeNet | 5.60 | 91.93 | 91.89 |
| MobileNetV4_Large | 31.39 | 91.54 | 91.51 |
| MobileNetV4_Medium | 8.87 | 91.54 | 91.51 |
| ResNet152 | 58.14 | 91.54 | 91.49 |
| MobileNetV4_Small | 3.23 | 90.23 | 90.16 |
| VisionTransformer | 85.80 | 88.28 | 88.23 |

## Citation

If you find this code or dataset useful for your research, please cite our paper:

```bibtex
Coming soon...
```

## License

This project is licensed under the MIT License.
