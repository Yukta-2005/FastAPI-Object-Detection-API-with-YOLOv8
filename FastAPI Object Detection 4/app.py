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
