from ultralytics import YOLO

def load_model(model_name: str):
    if model_name == "yolov8n":
        return YOLO("yolov8n.pt")
    elif model_name == "yolov8s":
        return YOLO("yolov8s.pt")
    else:
        raise ValueError("Unsupported model. Use 'yolov8n' or 'yolov8s'.")

def run_inference(model, image):
    results = model(image)
    detections = []

    for r in results:
        for box in r.boxes:
            detections.append({
                "label": model.names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": [int(coord) for coord in box.xyxy[0].tolist()]
            })
    return detections
