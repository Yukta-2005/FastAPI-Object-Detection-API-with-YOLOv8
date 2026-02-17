from PIL import ImageDraw

def draw_boxes(image, detections):
    draw = ImageDraw.Draw(image)
    for det in detections:
        bbox = det["bbox"]
        label = det["label"]
        conf = det["confidence"]
        draw.rectangle(bbox, outline="red", width=2)
        draw.text((bbox[0], bbox[1]-10), f"{label} {conf:.2f}", fill="red")
    return image
