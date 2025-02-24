import os
from datasets import Dataset, Features, Value, Image
import datetime
from huggingface_hub import Repository

# Folders with images
FOLDERS = ['created', 'augmented']

# Replace with your actual repository name.
repo_name = "spawn99/UK-Car-Plate-VRN-Dataset"

# Create a dictionary to accumulate records keyed by VRN.
data = {}

# Process images from both "created" and "augmented" folders.
for folder in FOLDERS:
    if not os.path.exists(folder):
        continue
    for filename in os.listdir(folder):
        if filename.endswith('.jpg'):
            # Expected filename formats:
            #   - Original: "VRN_side.jpg" (e.g. "LA19VPZ_f.jpg")
            #   - Augmented: "VRN_side_aug0.jpg" (or similar)
            base, _ = os.path.splitext(filename)
            parts = base.split('_')
            if len(parts) < 2:
                continue

            vrn = parts[0]
            plate_side_code = parts[1].lower()  # expecting 'f' or 'r'
            image_path = os.path.join(folder, filename)
            
            if vrn not in data:
                data[vrn] = {
                    "vrn": vrn,
                    "front_plate": None,
                    "rear_plate": None,
                    "augmented_front_plate": None,  # single augmented front image
                    "augmented_rear_plate": None    # single augmented rear image
                }
            
            # Handle created images
            if folder == "created":
                if plate_side_code == 'f':
                    data[vrn]["front_plate"] = image_path
                elif plate_side_code == 'r':
                    data[vrn]["rear_plate"] = image_path
            # Handle augmented images
            elif folder == "augmented":
                # Only set if not already set, as we intend to have just 1 augmentation per image.
                if plate_side_code == 'f' and data[vrn]["augmented_front_plate"] is None:
                    data[vrn]["augmented_front_plate"] = image_path
                elif plate_side_code == 'r' and data[vrn]["augmented_rear_plate"] is None:
                    data[vrn]["augmented_rear_plate"] = image_path

# Convert the data dictionary to a list of rows.
rows = list(data.values())

# Define the features (dataset schema) using the flat structure.
features = Features({
    "vrn": Value("string"),
    "front_plate": Image(),  # single front plate image
    "rear_plate": Image(),   # single rear plate image
    "augmented_front_plate": Image(),  # single augmented front image
    "augmented_rear_plate": Image()      # single augmented rear image
})

# Create the Hugging Face dataset. This converts each record field into a list.
ds = Dataset.from_dict({k: [row[k] for row in rows] for k in rows[0]}, features=features)

# Compute dataset statistics by counting the available images.
front_count = sum(1 for row in rows if row["front_plate"] is not None)
rear_count = sum(1 for row in rows if row["rear_plate"] is not None)
aug_front_count = sum(1 for row in rows if row["augmented_front_plate"] is not None)
aug_rear_count = sum(1 for row in rows if row["augmented_rear_plate"] is not None)

# Create a README.md file with improved clarity using the updated structure.
readme_content = """# UK Car Plate VRN Dataset

## Dataset Description
This dataset contains VRNs along with their respective UK car plate images. Each VRN record provides:
- **front_plate:** Synthetically created front license plate image.
- **rear_plate:** Synthetically created rear license plate image.
- **augmented_front_plate:** The augmented front license plate image.
- **augmented_rear_plate:** The augmented rear license plate image.

### Dataset Creation Date
{}

### Dataset Creation Method
See GitHub repository for details: [https://github.com/cavit99/VRM-VLM](https://github.com/cavit99/VRM-VLM)

### Dataset Structure
- **vrn:** Vehicle Registration Number.
- **front_plate:** The front view of the license plate.
- **rear_plate:** The rear view of the license plate.
- **augmented_front_plate:** An augmented front plate image.
- **augmented_rear_plate:** An augmented rear plate image.

### Dataset Statistics
- **Total distinct VRNs:** {}
- **Created Front Plates:** {}
- **Created Rear Plates:** {}
- **Augmented Front Plates:** {}
- **Augmented Rear Plates:** {}

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
Repository: https://huggingface.co/datasets/{}
""".format(
    datetime.datetime.now().strftime("%Y-%m-%d"),
    len(rows),
    front_count,
    rear_count,
    aug_front_count,
    aug_rear_count,
    repo_name
)

# Write README.md
with open("README.md", "w") as f:
    f.write(readme_content)

# Push the dataset and README to the Hugging Face Hub

# Initialize repository
repo = Repository(local_dir=".", clone_from=repo_name, use_auth_token=True)

# Push the dataset
ds.push_to_hub(repo_name, private=False)

# Add and push README
repo.push_to_hub(commit_message="Update README.md")

print(f"Dataset and README pushed to: https://huggingface.co/datasets/{repo_name}") 