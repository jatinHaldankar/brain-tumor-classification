import os
import uuid
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.prediction import PredictionPipeline

app = FastAPI(title="Brain Tumor MRI Classifier")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Temporary folder for uploaded images — cleaned up after each prediction
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model once at startup so the first request isn't slow
pipeline = PredictionPipeline()
pipeline._load_model()

class ImageRequest(BaseModel):
    image: str

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main UI page."""
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/predict")
async def predict(data: ImageRequest):
    """
    Accepts a base64-encoded image in the request body,
    saves it temporarily, runs the model, deletes the file, and returns JSON.
    """
    try:
        # Save the base64 image to a unique temp file to avoid collisions
        img_filename = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}.jpg")
        decodeImage(data.image, img_filename)

        result = pipeline.predict(img_filename)

        # Clean up — don't keep user images on disk
        if os.path.exists(img_filename):
            os.remove(img_filename)

        return result

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
