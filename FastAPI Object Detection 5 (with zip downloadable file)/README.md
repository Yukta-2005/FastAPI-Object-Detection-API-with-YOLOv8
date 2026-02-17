# Object Detection API (Image Dataset)

A FastAPI service for object detection using pre-trained YOLOv8 models. Supports dataset uploads, JSON results, annotated images, and downloadable zip files.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------

A. Objective:

1. Build a simple API using FastAPI that accepts image dataset uploaded and returns object detection results using a pre-trained model.
2. Returns annotated image's dataset (with bounding boxes drawn) as a second endpoint and a downloadable link.
   
-------------------------------------------------------------------------------------------------------------------------------------------------------------------
B. Requirements:

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
            bbox: [100, 150, 200, 250]
         },
         {
            label: ball;,
            confidence: 0.87,
            bbox: [300, 400, 350, 450]
         }
      ]
   }
   
Here’s a full FastAPI implementation that handles image dataset uploads (multiple images), runs object detection with a pre‑trained YOLOv8 model, returns JSON results, and provides annotated images with downloadable links.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------
C. Project Structure

object-detection-api/
│── app.py
│── models.py
│── utils.py
│── requirements.txt
│── README.md

-------------------------------------------------------------------------------------------------------------------------------------------------------------------
D. Code
1. app.py
      python
      import io, os, uuid, zipfile
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
          zip_filename = f"{uuid.uuid4().hex}_dataset.zip"
          zip_filepath = os.path.join(ANNOTATED_DIR, zip_filename)
      
          with zipfile.ZipFile(zip_filepath, "w") as zipf:
              for file in files:
                  if file.content_type not in ["image/jpeg", "image/png"]:
                      raise HTTPException(status_code=400, detail="Invalid image format. Use JPEG/PNG.")
      
                  image_bytes = await file.read()
                  image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
      
                  detections = run_inference(current_model, image)
                  annotated = draw_boxes(image, detections)
      
                  # Save annotated image
                  filename = f"{uuid.uuid4().hex}_{file.filename}.png"
                  filepath = os.path.join(ANNOTATED_DIR, filename)
                  annotated.save(filepath)
      
                  # Add to zip
                  zipf.write(filepath, arcname=filename)
      
                  results.append({
                      "filename": file.filename,
                      "detections": detections,
                      "annotated_file": filename
                  })
      
          return {
              "results": results,
              "download_url": f"/download/{zip_filename}"
          }
      
      
      @app.get("/download/{filename}")
      async def download_file(filename: str):
          filepath = os.path.join(ANNOTATED_DIR, filename)
          if not os.path.exists(filepath):
              raise HTTPException(status_code=404, detail="File not found")
          return FileResponse(filepath, media_type="application/zip", filename=filename)
      
      
      @app.post("/switch-model/{model_name}")
      async def switch_model(model_name: str):
          global current_model
          try:
              current_model = load_model(model_name)
              return {"message": f"Model switched to {model_name}"}
          except Exception as e:
              raise HTTPException(status_code=400, detail=str(e))

   
3. models.py
      
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

   
3. utils.py

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


4.requirements.txt

   fastapi
   uvicorn
   torch
   ultralytics
   pillow

   
5. README.md
 
      # Object Detection API (Image Dataset)
      
      A FastAPI service for object detection using pre-trained YOLOv8 models. Supports dataset uploads, JSON results, annotated images, and downloadable zip files.
      
      ## 🚀 Setup
      
      1. Install dependencies:
         ```bash
         pip install -r requirements.txt
      Run the API:

      bash
      uvicorn app:app --reload

------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
E. Endpoints
POST /detect → Upload multiple images (JPEG/PNG) and get detections in JSON.

POST /detect/annotated → Upload multiple images, get detections + annotated images zipped with a download link.

GET /download/{filename} → Download annotated dataset zip.

POST /switch-model/{model_name} → Switch between models (yolov8n, yolov8s).

-------------------------------------------------------------------------------------------------------------------------------------------------------------------
F. Example Request
   1. Postman--->
   curl -X POST "http://127.0.0.1:8000/detect" \
     -H "Content-Type: multipart/form-data" \
     -F "files=@dog.png" \
     -F "files=@ball.png"
   
   2. Response:
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

   3. For annotated dataset:
   
   json
   {
     "results": [...],
     "download_url": "/download/abcd1234_dataset.zip"
   }

-------------------------------------------------------------------------------------------------------------------------------------------------------------------

G. This implementation now supports:
- Dataset uploads (multiple images)  
- JSON detection results per image  
- Annotated dataset zipped with a download link  
- Model switching capability  

Would you like me to also add **unit tests** for these dataset endpoints so you can validate them autom

