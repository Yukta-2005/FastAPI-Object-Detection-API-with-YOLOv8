# Object Detection API (Multiple Images)

A FastAPI service for object detection using pre-trained YOLOv8 models. Supports multiple image uploads and annotated image downloads.

## 🚀 Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt

## 🐳 Docker Usage

### Build & Run with Docker
```bash
docker build -t object-detection-api .
docker run -p 8000:8000 -v $(pwd)/annotated:/app/annotated object-detection-api
