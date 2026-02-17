# Object Detection API (Multiple Images)

A FastAPI service for object detection using pre-trained YOLOv8 models. Supports multiple image uploads and annotated image downloads.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------

A. Objective:
   1. Build a simple API using FastAPI that accepts multiple images uploaded and returns object detection results using a pre-trained model.
   2. Return annotated images (with bounding boxes drawn) as a second endpoint and a downloadable link.
   2. Requirements:
      1. Framework: FastAPI
      2. Task:
         ○ Accept an image file (JPEG/PNG) via an HTTP POST endpoint.
         ○ Run a simple object detection model on it (use a pre-trained YOLOv5 / YOLOv8 / MobileNet SSD — anything lightweight).
         ○ Return JSON with:
            ■ Detected object labels
            ■ Confidence scores
            ■ Bounding box coordinates
      3. Pre-trained Model Suggestions:
         ○ Any pre-trained model from Torchvision (like FasterRCNN, SSD) or
         ○ YOLOv8n (ultralytics repo) — very easy to load
         ○ Switch between models via API call
   4. Return JSON Example:
   {
      detections: [
      {
            label: dog,
            confidence: 0.92,
            bbox&: [100, 150, 200, 250]
      },
      {
            label: ball,
            confidence;: 0.87,
            bbox: [300, 400, 350, 450]
         }
      ]
   }
   
Here’s a full FastAPI implementation tailored to your updated objective: accept multiple images, run object detection, return JSON results, and provide annotated images with downloadable links.
--------------------------------------------------------------------------------------------------------------------------------------------------------------------

B. Project Structure

object-detection-api/
│── app.py
│── models.py
│── utils.py
│── requirements.txt
│── README.md
--------------------------------------------------------------------------------------------------------------------------------------------------------------------

C. Code
1. app.py

      import io, os, uuid
      from fastapi import FastAPI, File, UploadFile, HTTPException
      from fastapi.responses import JSONResponse, FileResponse
      from typing import List
      from PIL import Image
      from models import load_model, run_inference
      from utils import draw_boxes
      
      app = FastAPI(title="Object Detection API")
      
      # Default model
      current_model = load_model("yolov8n")
      
      # Directory to save annotated images
      ANNOTATED_DIR = "annotated"
      os.makedirs(ANNOTATED_DIR, exist_ok=True)
      
      @app.post("/detect")
      async def detect_objects(files: List[UploadFile] = File(...)):
          results = []
          for file in files:
              if file.content_type not in ["image/jpeg", "image/png"]:
                  raise HTTPException(status_code=400, detail="Invalid image format. Use JPEG/PNG.")
      
              image_bytes = await file.read()
              image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
      
              detections = run_inference(current_model, image)
              results.append({
                  "filename": file.filename,
                  "detections": detections
              })
      
          return JSONResponse(content={"results": results})
      
      
      @app.post("/detect/annotated")
      async def detect_and_annotate(files: List[UploadFile] = File(...)):
          results = []
          for file in files:
              if file.content_type not in ["image/jpeg", "image/png"]:
                  raise HTTPException(status_code=400, detail="Invalid image format. Use JPEG/PNG.")
      
              image_bytes = await file.read()
              image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
      
              detections = run_inference(current_model, image)
              annotated = draw_boxes(image, detections)
      
              # Save annotated image with unique filename
              filename = f"{uuid.uuid4().hex}_{file.filename}.png"
              filepath = os.path.join(ANNOTATED_DIR, filename)
              annotated.save(filepath)
      
              results.append({
                  "filename": file.filename,
                  "detections": detections,
                  "download_url": f"/download/{filename}"
              })
      
          return JSONResponse(content={"results": results})
      
      
      @app.get("/download/{filename}")
      async def download_file(filename: str):
          filepath = os.path.join(ANNOTATED_DIR, filename)
          if not os.path.exists(filepath):
              raise HTTPException(status_code=404, detail="File not found")
          return FileResponse(filepath, media_type="image/png", filename=filename)
      
      
      @app.post("/switch-model/{model_name}")
      async def switch_model(model_name: str):
          global current_model
          try:
              current_model = load_model(model_name)
              return {"message": f"Model switched to {model_name}"}
          except Exception as e:
              raise HTTPException(status_code=400, detail=str(e))
2. models.py

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

   
4. utils.py

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
5. requirements.txt

      fastapi
      uvicorn
      torch
      ultralytics
      pillow

6. README.md

      # Object Detection API (Multiple Images)
      
      A FastAPI service for object detection using pre-trained YOLOv8 models. Supports multiple image uploads and annotated image downloads.
      
      ## 🚀 Setup
      
      1. Install dependencies:
         ```bash
         pip install -r requirements.txt
      Run the API:
      
      bash
      uvicorn app:app --reload
--------------------------------------------------------------------------------------------------------------------------------------------------------------------
   
D. Endpoints
   POST /detect → Upload multiple images (JPEG/PNG) and get detections in JSON.
   
   POST /detect/annotated → Upload multiple images and get detections + downloadable links to annotated images.
   
   GET /download/{filename} → Download annotated images.
   
   POST /switch-model/{model_name} → Switch between models (yolov8n, yolov8s).
--------------------------------------------------------------------------------------------------------------------------------------------------------------------

E. Example Request
   Postman-->
   curl -X POST "http://127.0.0.1:8000/detect" \
     -H "Content-Type: multipart/form-data" \
     -F "files=@dog.png" \
     -F "files=@ball.png"
--------------------------------------------------------------------------------------------------------------------------------------------------------------------

F. Response:
   json
   {
     "results": [
       {
         "filename": "dog.png",
         "detections": [
           {
             "label": "dog",
             "confidence": 0.92,
             "bbox": [100, 150, 200, 250]
           }
         ]
       },
       {
         "filename": "ball.png",
         "detections": [
           {
             "label": "ball",
             "confidence": 0.87,
             "bbox": [300, 400, 350, 450]
           }
         ]
       }
     ]
   }
--------------------------------------------------------------------------------------------------------------------------------------------------------------------

G. This implementation now supports:
- Multiple image uploads  
- JSON detection results per image  
- Annotated image saving  
- Downloadable links for each annotated image  
- Model switching  

--------------------------------------------------------------------------------------------------------------------------------------------------------------------
Docker support with docker-compose so you can run your FastAPI object detection API with persistent storage for annotated images and easy environment configuration.

H. Dockerfile
   Place this in your project root:
   
   dockerfile
   FROM python:3.9-slim
   
   WORKDIR /app
   
   # Install system dependencies (needed for Pillow/Torch)
   RUN apt-get update && apt-get install -y \
       build-essential \
       libgl1 \
       && rm -rf /var/lib/apt/lists/*
   
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   
   COPY . .
   
   EXPOSE 8000

   CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

-------------------------------------------------------------------------------------------------------------------------------------------------------------------  
I. docker-compose.yml
   This file makes it easy to run the API and persist annotated images:
   
   yaml
   version: "3.9"
   
   services:
     object-detection-api:
       build: .
       container_name: object-detection-api
       ports:
         - "8000:8000"
       volumes:
         - ./annotated:/app/annotated   # persist annotated images
       environment:
         - MODEL_NAME=yolov8n           # default model (can override)
--------------------------------------------------------------------------------------------------------------------------------------------------------------------

J. README.md (add Docker instructions)
   Update your README with:
   
   markdown
   ## 🐳 Docker Usage
   
   ### Build & Run with Docker
   ```bash
   docker build -t object-detection-api .
   docker run -p 8000:8000 -v $(pwd)/annotated:/app/annotated object-detection-api
   Run with docker-compose
   bash
   docker-compose up --build
   This will:
   
   Expose the API on http://localhost:8000
   
   Persist annotated images in the local annotated/ folder
   
   Allow switching models via environment variable (MODEL_NAME)
--------------------------------------------------------------------------------------------------------------------------------------------------------------------

K. With this setup you now have:
- **Dockerfile** for containerization  
- **docker-compose.yml** for easy orchestration and persistent storage  
- Annotated images saved locally and accessible via `/download/{filename}`  

You can also add a **Kubernetes deployment manifest** so you can scale this API in a cluster

Extend your deployment setup with Kubernetes manifests so you can run and scale your FastAPI object detection API in a cluster.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------
L. Kubernetes Manifests:
   1. Create a folder k8s/ in your project root:
   
   Code
   object-detection-api/
   │── app.py
   │── models.py
   │── utils.py
   │── requirements.txt
   │── README.md
   │── Dockerfile
   │── docker-compose.yml
   │── k8s/
       │── deployment.yaml
       │── service.yaml
       │── ingress.yaml   (optional)
       │── pvc.yaml

   2. k8s/deployment.yaml
   yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: object-detection-api
     labels:
       app: object-detection-api
   spec:
     replicas: 2   # scale horizontally
     selector:
       matchLabels:
         app: object-detection-api
     template:
       metadata:
         labels:
           app: object-detection-api
       spec:
         containers:
           - name: object-detection-api
             image: object-detection-api:latest   # replace with your registry image
             ports:
               - containerPort: 8000
             env:
               - name: MODEL_NAME
                 value: yolov8n
             volumeMounts:
               - name: annotated-storage
                 mountPath: /app/annotated
         volumes:
           - name: annotated-storage
             persistentVolumeClaim:
               claimName: annotated-pvc

   3. k8s/service.yaml
   yaml
   apiVersion: v1
   kind: Service
   metadata:
     name: object-detection-api-service
   spec:
     selector:
       app: object-detection-api
     ports:
       - protocol: TCP
         port: 80
         targetPort: 8000
     type: ClusterIP

   4. k8s/ingress.yaml (optional, if you want external access)
   yaml
   apiVersion: networking.k8s.io/v1
   kind: Ingress
   metadata:
     name: object-detection-api-ingress
     annotations:
       kubernetes.io/ingress.class: nginx
   spec:
     rules:
       - host: object-detection.local
         http:
           paths:
             - path: /
               pathType: Prefix
               backend:
                 service:
                   name: object-detection-api-service
                   port:
                     number: 80
   5. k8s/pvc.yaml
   yaml
   apiVersion: v1
   kind: PersistentVolumeClaim
   metadata:
     name: annotated-pvc
   spec:
     accessModes:
       - ReadWriteOnce
     resources:
       requests:
         storage: 1Gi
--------------------------------------------------------------------------------------------------------------------------------------------------------------------

M. Deployment Steps
   Build & push your Docker image to a registry (e.g., Docker Hub, AWS ECR, GCP Artifact Registry):
   
   bash
   docker build -t your-dockerhub-username/object-detection-api:latest .
   docker push your-dockerhub-username/object-detection-api:latest
   Update deployment.yaml with your registry image:
   
   yaml
   image: your-dockerhub-username/object-detection-api:latest
   Apply manifests:
   
   bash
   kubectl apply -f k8s/pvc.yaml
   kubectl apply -f k8s/deployment.yaml
   kubectl apply -f k8s/service.yaml
   kubectl apply -f k8s/ingress.yaml   # optional
   Check pods:
   
   bash
   kubectl get pods
   Access the API:
   
   Inside cluster: http://object-detection-api-service

   Outside cluster (with Ingress): http://object-detection.local

--------------------------------------------------------------------------------------------------------------------------------------------------------------------
N. With this setup you now have:

   Deployment with scaling (replicas)
   Service for internal access   
   Ingress for external access (optional)   
   Persistent storage for annotated images
