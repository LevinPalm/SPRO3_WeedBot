import os
import shutil
import random

# Path to the folder containing your new negative images
SOURCE_IMAGES_DIR = 'path\to\scaled\negative\images'

# Path to the root of your existing dataset structure
# This folder should contain 'train', 'valid', and 'test' folders
TARGET_DATASET_ROOT = 'path\to\dataset'

# Define the split ratio for your new negative images (must sum to 1)
TRAIN_RATIO = 0.7  # 70% of new images go to 'train'
VALID_RATIO = 0.2  # 20% of new images go to 'valid'
TEST_RATIO  = 0.1  # 10% of new images go to 'test'

# Creates a corresponding empty .txt label file for a negative image
def create_empty_label_file(target_labels_dir, image_filename):
    base_filename = os.path.splitext(image_filename)[0]
    label_filepath = os.path.join(target_labels_dir, f"{base_filename}.txt")
    
    # Create an empty file
    with open(label_filepath, 'w') as f:
        pass 

#Copies images and creates empty label files for a given split
def copy_negative_images_and_labels(images_list, target_split_dir):
    # Check for the expected folder structure inside the target split
    target_images_dir = os.path.join(TARGET_DATASET_ROOT, target_split_dir, 'images')
    target_labels_dir = os.path.join(TARGET_DATASET_ROOT, target_split_dir, 'labels')

    if not (os.path.isdir(target_images_dir) and os.path.isdir(target_labels_dir)):
        print(f"ERROR: Could not find '{target_images_dir}' or '{target_labels_dir}'.")
        print("Please check if TARGET_DATASET_ROOT is correct and contains the expected subfolders.")
        return

    print(f"\nCopying {len(images_list)} images to '{target_split_dir}':")
    
    for filename in images_list:
        source_path = os.path.join(SOURCE_IMAGES_DIR, filename)
        
        # Copy the image file
        shutil.copy2(source_path, target_images_dir)
        
        # Create the empty .txt label file
        create_empty_label_file(target_labels_dir, filename)
        
    print(f"Successfully added {len(images_list)} negative samples to '{target_split_dir}'.")


# Validate configuration
if abs(TRAIN_RATIO + VALID_RATIO + TEST_RATIO - 1.0) > 1e-6:
    raise ValueError("The split ratios (TRAIN_RATIO, VALID_RATIO, TEST_RATIO) must sum up to 1.0.")

if not os.path.isdir(SOURCE_IMAGES_DIR):
    raise FileNotFoundError(f"Source folder not found: {SOURCE_IMAGES_DIR}")

if not os.path.isdir(TARGET_DATASET_ROOT):
    raise FileNotFoundError(f"Target dataset root not found: {TARGET_DATASET_ROOT}")

# Get all image files from the source folder
print(f"Scanning source directory: {SOURCE_IMAGES_DIR}")
all_images = [f for f in os.listdir(SOURCE_IMAGES_DIR) 
              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]

if not all_images:
    print("No images found in the source directory. Exiting.")
    exit()

print(f"Found a total of {len(all_images)} images.")

# Randomly shuffle and split the images
random.shuffle(all_images)
total_count = len(all_images)

train_count = int(total_count * TRAIN_RATIO)
valid_count = int(total_count * VALID_RATIO)
# The remaining count ensures no rounding errors leave files out
test_count = total_count - train_count - valid_count 

train_images = all_images[:train_count]
valid_images = all_images[train_count : train_count + valid_count]
test_images  = all_images[train_count + valid_count :]

# Copy files to the respective target folders
copy_negative_images_and_labels(train_images, 'train')
copy_negative_images_and_labels(valid_images, 'valid')
copy_negative_images_and_labels(test_images, 'test')

print("The existing dataset has been updated with new negative samples.")