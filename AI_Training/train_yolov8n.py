from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n.pt') 

    results = model.train(
        data = r'path\to\data.yaml',
        epochs = 50,
        imgsz = 640,
        batch=4,
        device='cuda'
    )