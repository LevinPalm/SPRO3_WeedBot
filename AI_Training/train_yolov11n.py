from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolo11n.pt') 

    results = model.train(
        data = r'path\to\data.yaml',
        epochs = 100,
        conf = 0.75,
        iou = 0.5,
        imgsz = 640,
        batch=4,
        device='cuda'
    )