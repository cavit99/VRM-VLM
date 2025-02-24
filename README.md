# UK Car Plate VRN Dataset & Generator

## Overview

This repository contains tools for generating synthetic UK vehicle registration number (VRN) images and compiling them into a dataset. The project demonstrates both the generation of realistic UK car plate images (front and rear), applying image augmentations, and the construction of a Hugging Face dataset from these images.

The repository includes:

- **create_VRN.py**  
  Creates UK VRNs following region-specific and age-related rules. It also optionally adds flash properties (flash colour, national flag, and country codes) to the VRNs.

- **create_plate_images.py**  
  Uses the VRN generator to render plate images. This script creates two images per VRN – a front image (with a white background) and a rear image (with a yellow background chosen at random). It optionally renders a flash area with a national flag and country text.

- **augmentation.py**  
  Augments license plate images using geometric (e.g., perspective, rotation) and photometric (e.g., brightness/contrast, blur, noise) transformations. This script reads images from the `created/` folder and outputs them to the `augmented/` folder.

- **create_dataset.py**  
  Processes the created images from the `created/` folder, builds a dataset with associated metadata (e.g., VRN, plate type), creates a README for the dataset, and pushes the dataset to the Hugging Face Hub.

- **media/**  
  Contains supporting media assets such as custom fonts (`CharlesWright-Bold.otf`) and flag images used during plate rendering.

## Installation

Ensure you are using Python 3 and it is recommended to use a virtual environment. Install the required packages with:

```bash
pip install -r requirements.txt
```

**Dependencies:**

- [Pillow](https://python-pillow.org/) – For image processing and rendering.
- [datasets](https://github.com/huggingface/datasets) – For dataset creation and management.

*You may create a `requirements.txt` file containing:*
```
pillow
datasets
```

## Usage

### 1. Create VRNs and License Plate Images

To create a number of license plate images, run:

```bash
python create_plate_images.py [number_of_plates]
```

- The script will create both front and rear JPEG images in the `generated/` folder.
- If no number is provided, the default is 1 plate.

### 2. Augment License Plate Images

Once you have generated the license plate images (stored in the `generated/` folder), run:

```bash
python augmentation.py
```

- This script applies geometric and photometric augmentations to each image in the `generated/` folder.
- Augmented images are saved in the `augmented/` folder.

### 3. Create and Push the Dataset

Once the images have been created (or augmented), run:

```bash
python create_dataset.py
```

- This script scans the `created/` folder, compiles the data (e.g., file name, VRN, plate type, and image path), and creates a Hugging Face dataset.
- A README is created for the dataset, and the dataset is pushed to the Hugging Face Hub.
- **Note:** Update the `repo_name` variable in `create_dataset.py` with your actual Hugging Face repository name before pushing.

## Customization

- **Media Assets:**  
  Make sure the `media/` directory contains the required resources (e.g., the custom font and flag images). If these files are missing or located elsewhere, adjust the file paths in the scripts accordingly.

- **Dataset Repository Name:**  
  In `create_dataset.py`, change the `repo_name` variable to match your Hugging Face repository name if needed.

## License

This project is licensed under the MIT License.

```
MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Contact

Dataset and image generation code by **Cavit Erginsoy - 2025**.  
For questions, suggestions, or contributions, please open an issue or submit a pull request.

## Hugging Face Hub

The created dataset is available on the Hugging Face Hub at:  
[https://huggingface.co/datasets/spawn99/UK-Car-Plate-VRN-Dataset](https://huggingface.co/datasets/spawn99/UK-Car-Plate-VRN-Dataset)