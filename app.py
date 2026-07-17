import os
import shutil
import uuid
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
import utils

# Initialize FastAPI App
app = FastAPI(title="Aircraft Engine Defect Detection Dashboard", version="1.0")

# Enable CORS for local development flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories for saving images
UPLOAD_DIR = Path("static/uploads")
RESULT_DIR = Path("static/results")

# Ensure folders exist and are clean on startup
if UPLOAD_DIR.exists():
    for f in UPLOAD_DIR.glob("*"):
        try:
            if f.is_file(): f.unlink()
        except Exception: pass
if RESULT_DIR.exists():
    for f in RESULT_DIR.glob("*"):
        try:
            if f.is_file(): f.unlink()
        except Exception: pass

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# YOLO11 Class details
CLASS_NAMES = {0: "burn", 1: "crack", 2: "turbine_blade"}
COLORS = {
    "turbine_blade": (255, 120, 0),  # Steel Blue (BGR)
    "crack": (0, 0, 255),           # Neon Red (BGR)
    "burn": (0, 165, 255)           # Glowing Amber (BGR)
}

# Global Model Variable
model = None
is_mock_mode = False

@app.on_event("startup")
def load_model():
    global model, is_mock_mode
    # Search for trained model weights
    default_trained = Path("runs/segment/train_turbine_defects/weights/best.pt")
    
    if default_trained.exists():
        model_path = str(default_trained)
        print(f"Server Startup: Loading custom trained weights from {model_path}...")
    elif os.path.exists("yolo11n-seg.pt"):
        model_path = "yolo11n-seg.pt"
        print(f"Server Startup: Trained weights not found. Loading base weights: {model_path}...")
    else:
        is_mock_mode = True
        model = None
        print("="*60)
        print("Server Startup WARNING: No YOLO weights found.")
        print("Running in DEMO/MOCK mode. Detections will be synthetically generated.")
        print("To run actual inference, run 'python train.py' or download 'yolo11n-seg.pt'.")
        print("="*60)
        return
        
    try:
        model = YOLO(model_path)
        is_mock_mode = False
        print("Server Startup: Model loaded successfully.")
    except Exception as e:
        print(f"Server Startup Error: Failed to load model: {e}")
        is_mock_mode = True
        model = None

@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    global model, is_mock_mode
    # 1. Validate file extension
    ext = Path(file.filename).suffix.lower()
    if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
        raise HTTPException(status_code=400, detail="Unsupported file format. Use JPG, PNG or WEBP.")
        
    # 2. Generate unique filename to avoid collisions
    unique_filename = f"{uuid.uuid4().hex}{ext}"
    orig_path = UPLOAD_DIR / unique_filename
    res_path = RESULT_DIR / unique_filename
    
    # 3. Save uploaded file to static/uploads/
    try:
        with orig_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save upload image: {e}")
        
    # 4. Load image using OpenCV
    img = cv2.imread(str(orig_path))
    if img is None:
        if orig_path.exists():
            orig_path.unlink()
        raise HTTPException(status_code=400, detail="Invalid image file or corrupted upload.")
        
    h, w, _ = img.shape
    
    # 5. Process detection (Inference or Mock)
    try:
        if is_mock_mode or model is None:
            # Run simulation
            metrics, annotated_img = utils.run_mock_inference(img, COLORS)
            metrics["is_mock"] = True
        else:
            # Run real YOLO inference
            results = model(str(orig_path), conf=0.25)[0]
            metrics, blades, defects, unassociated = utils.analyze_detections(
                results.boxes, results.masks, CLASS_NAMES, COLORS, w, h
            )
            annotated_img = utils.draw_overlays(img, blades, defects, unassociated, COLORS)
            metrics["is_mock"] = False
            
        # 6. Save annotated image to static/results/
        cv2.imwrite(str(res_path), annotated_img)
        
        # 7. Add file references to JSON response
        metrics["original_url"] = f"/static/uploads/{unique_filename}"
        metrics["result_url"] = f"/static/results/{unique_filename}"
        metrics["image_name"] = file.filename
        
        return metrics
        
    except Exception as e:
        # Fallback cleanup
        if orig_path.exists():
            orig_path.unlink()
        if res_path.exists():
            res_path.unlink()
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

# Serve Static UI Frontend
@app.get("/")
def get_index():
    # Return main dashboard page
    index_path = Path("static/index.html")
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="static/index.html not found.")
    return FileResponse(str(index_path))

# Mount static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    # Start FastAPI server on port 8000
    print("Launching Local Web Application...")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
