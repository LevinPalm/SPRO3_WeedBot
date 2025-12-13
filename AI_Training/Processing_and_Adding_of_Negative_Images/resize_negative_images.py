import os
from PIL import Image

# Path to the folder containing your new negative images (unprocessed)
SOURCE_IMAGES_DIR = 'path\to\unprocessed\negative\images'

# Path for the processed images to be saveds.
OUTPUT_IMAGES_DIR = 'path\to\processed\images'

# Target resolution for the square image
TARGET_SIZE = (640, 640)


def process_image(source_path, output_path, target_size):
    # Loads an image, center crops it to a square aspect ratio, and then resizes it to the target size
    try:
        img = Image.open(source_path).convert('RGB')
        width, height = img.size

        # Calculate the center crop box for a square aspect ratio
        if width > height:
            # Image is wider than it is tall -> crop the sides
            left = (width - height) // 2
            right = (width + height) // 2
            top = 0
            bottom = height
            crop_area = (left, top, right, bottom)
        else:
            # Image is taller than it is wide -> crop the top and bottom
            left = 0
            right = width
            top = (height - width) // 2
            bottom = (height + width) // 2
            crop_area = (left, top, right, bottom)
            
        # Perform the center crop
        img_cropped = img.crop(crop_area)
        
        # Resize the square image to the target size
        img_resized = img_cropped.resize(target_size)
        
        # Save the processed image
        img_resized.save(output_path)
        return True

    except Exception as e:
        print(f"Error processing {os.path.basename(source_path)}: {e}")
        return False


if not os.path.isdir(SOURCE_IMAGES_DIR):
    print(f"ERROR: Source folder not found: {SOURCE_IMAGES_DIR}")
    exit()

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)

print(f"Starting image processing to {TARGET_SIZE[0]}x{TARGET_SIZE[1]}...")

# Get all images
image_files = [f for f in os.listdir(SOURCE_IMAGES_DIR) 
               if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

if not image_files:
    print("No images found in the source directory. Exiting.")
    exit()

processed_count = 0
for filename in image_files:
    source_path = os.path.join(SOURCE_IMAGES_DIR, filename)
    output_path = os.path.join(OUTPUT_IMAGES_DIR, filename)
    
    if process_image(source_path, output_path, TARGET_SIZE):
        processed_count += 1

print(f"Successfully processed {processed_count} images and saved them to: {OUTPUT_IMAGES_DIR}")