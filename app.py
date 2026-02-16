import io
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from models import load_model, run_inference

app = FastAPI(title="Object Detection API")

# Default model
current_model = load_model("yolov8n")

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid image format. Use JPEG/PNG.")

    # Read image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Run inference
    detections = run_inference(current_model, image)

    return JSONResponse(content={"detections": detections})


@app.post("/switch-model/{model_name}")
async def switch_model(model_name: str):
    global current_model
    try:
        current_model = load_model(model_name)
        return {"message": f"Model switched to {model_name}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
