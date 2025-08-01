import os 
import shutil
import random
from pathlib import Path
import tensorflow as tf

# List all physical devices recognized by TensorFlow
print("Physical devices:", tf.config.list_physical_devices())

# Specifically list GPU devices
print("GPU devices:", tf.config.list_physical_devices('GPU'))

# Set seed for reproducibility
random.seed(42)                                 # sets a fixed random so shuffling is repeatable

# Path to your original images + annotations
SOURCE_DIR = Path("images_and_anno_OG")         #sets the source directory to .jpg and .xml files

# Output folders
OUTPUT_DIR = SOURCE_DIR
TRAIN_DIR = OUTPUT_DIR / "train"
VAL_DIR = OUTPUT_DIR / "val"
TEST_DIR = OUTPUT_DIR / "test"

# Create output folders
for folder in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    folder.mkdir(parents=True, exist_ok=True) #parents=True creates any necessary parent folders.

# Gather all image files (assumes .jpg)
image_files = list(SOURCE_DIR.glob("*.jpg"))    #Gets a list of all .jpg image files in your source folder.
random.shuffle(image_files)                     #Randomly shuffles the list of images to ensure a good data split.

# Split 80/10/10
total = len(image_files)                                            # Total number of images 107
train_split = int(0.8 * total)                                      #0.8 x 107 images
val_split = int(0.1 * total)                                        #0.1 x 107 images

train_files = image_files[:train_split]                             # First 80% for training
val_files = image_files[train_split:train_split + val_split]        # Next 10% for validation
test_files = image_files[train_split + val_split:]                  # Last 10% for testing

def move_files(file_list, destination):
    for img_path in file_list:
        xml_path = img_path.with_suffix(".xml")                     #Generates the expected XML file path by changing .jpg to .xml
        shutil.move(str(img_path), destination / img_path.name)     #Moves the .jpg image to the new destination folder
        if xml_path.exists():                                       #Checks if the .xml file exists. If it does, move it alongside the image
            shutil.move(str(xml_path), destination / xml_path.name)
        else:
            print(f"⚠️ Warning: Missing XML for {img_path.name}")

# Moves the training, validation, and testing files to their respective folders by calling the function we defined.
move_files(train_files, TRAIN_DIR)
move_files(val_files, VAL_DIR)
move_files(test_files, TEST_DIR)

#Confirmation messages showing how many files were moved into each split
print(f"Split complete!")
print(f"Train: {len(train_files)} images")
print(f"Val:   {len(val_files)} images")
print(f"Test:  {len(test_files)} images")

