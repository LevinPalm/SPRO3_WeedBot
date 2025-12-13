from ultralytics import YOLO

# Load your custom PyTorch model
model = YOLO("path\to\pytorch\model\best.pt")
# Export the model to NCNN format
# This creates a folder with .param and .bin files
model.export(format="ncnn") 