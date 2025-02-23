import os
from datasets import Dataset, Features, Value, Image
# The datasets library automatically handles image conversion using PIL under the hood
import datetime

# Folder where your images are located
GENERATED_FOLDER = 'generated'

# List to store our dataset rows
rows = []

# Iterate over all files in the 'generated' folder
for filename in os.listdir(GENERATED_FOLDER):
    if filename.endswith('.jpg'):
        # Expected filename format: "VRN_side.jpg" (e.g. "LA19VPZ_f.jpg")
        base, _ = os.path.splitext(filename)  # e.g., "LA19VPZ_f"
        parts = base.split('_')
        if len(parts) < 2:
            # Skip any file that does not match the expected format
            continue

        vrn = parts[0]
        plate_side_code = parts[1].lower()

        # Determine plate type based on the filename suffix
        if plate_side_code == 'f':
            plate_type = "front"
        elif plate_side_code == 'r':
            plate_type = "rear"
        else:
            # If there is an unexpected code, mark as unknown
            plate_type = "unknown"

        # Full path to the image file
        image_path = os.path.join(GENERATED_FOLDER, filename)

        # Append example dictionary
        rows.append({
            "file_name": filename,
            "vrn": vrn,
            "plate_type": plate_type,
            "image": image_path  # the Image() feature will read this file
        })

# Define the features (schema) for the dataset
features = Features({
    "file_name": Value("string"),
    "vrn": Value("string"),
    "plate_type": Value("string"),
    "image": Image()
})

# Create a dictionary with lists for each column from the rows
data_dict = {key: [row[key] for row in rows] for key in rows[0]}

# Create the Hugging Face dataset
ds = Dataset.from_dict(data_dict, features=features)

# Before pushing to hub, let's create a README.md file
readme_content = """# UK Car Plate VRN Dataset

## Dataset Description
This dataset contains images of UK car license plates (Vehicle Registration Numbers - VRNs) with their corresponding plate types (front/rear).

### Dataset Creation Date
23 February 2025

### Dataset Structure
- **file_name**: Name of the image file
- **vrn**: Vehicle Registration Number
- **plate_type**: Position of the plate (front/rear)
- **image**: Image file in jpg format

### File Naming Convention
Files are named using the format: `VRN_side.jpg`
where:
- VRN: The vehicle registration number
- side: 'f' for front plate, 'r' for rear plate

### Data Collection Method
Images are collected from the 'generated' folder. Each image is processed to extract the VRN and plate type information from the filename.

### Dataset Statistics
7500 Unique VRNs as both front and rear plates each
15,000 total images

### License
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

### Contact
Dataset created by: Cavit Erginsoy - 2025
Repository: https://huggingface.co/datasets/{repo_name}
""".format(
    datetime.datetime.now().strftime("%Y-%m-%d"),
    len(rows)
)

# Write README.md
with open("README.md", "w") as f:
    f.write(readme_content)

# Push the dataset to the Hugging Face Hub:
# Replace 'your_username/uk_plate_dataset' with your actual repository name.
repo_name = "spawn99/UK-Car-Plate-VRN-Dataset"
ds.push_to_hub(repo_name, private=False)

print(f"Dataset pushed to: https://huggingface.co/datasets/{repo_name}") 