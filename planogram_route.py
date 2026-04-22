from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import os
import asyncio

from planogram_service import PlanogramAnalyzer

router = APIRouter()

# Load model path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESNET_MODEL_PATH = os.path.join(BASE_DIR, "new_model.onnx")

# Initialize analyzer (global, so it loads once)
analyzer = PlanogramAnalyzer(resnet_model_path=RESNET_MODEL_PATH)

# Ensure model is loaded once at startup
initialized = False


async def get_analyzer():
    global initialized
    if not initialized:
        await analyzer.initialize()
        initialized = True
    return analyzer


@router.post("/analyze-planogram")
async def analyze_planogram(file: UploadFile = File(...)):
    """
    Upload an image and get planogram analysis
    """
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Get analyzer
        analyzer_instance = await get_analyzer()

        # Run analysis
        results = await analyzer_instance.results(image)

        return JSONResponse(content=results)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))