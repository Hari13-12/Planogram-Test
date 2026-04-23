from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import os
import asyncio
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi import Form
import os
from typing import Optional

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
# async def analyze_planogram(file: UploadFile = File(...)):
#     """
#     Upload an image and get planogram analysis
#     """
#     try:
#         # Validate file type
#         if not file.content_type.startswith("image/"):
#             raise HTTPException(status_code=400, detail="File must be an image")

#         # Read image
#         contents = await file.read()
#         image = Image.open(io.BytesIO(contents))

#         # Get analyzer
#         analyzer_instance = await get_analyzer()

#         # Run analysis
#         results = await analyzer_instance.results(image)

#         return JSONResponse(content=results)

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
async def get_planogram(org_id: str = Form(...), image: UploadFile = File(...)):
    if org_id != "test":
        return JSONResponse(
            status_code=400,
            content={"detail": "Invalid organization ID"}
        )
    try:
        # Validate image
        if not image.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an image"
            )
        try:
            planogram_analyzer = PlanogramAnalyzer(
                # yolo_model_path=YOLO_PATH,
                resnet_model_path=RESNET_PATH
            )
            # analyzer = await get_planogram_analyzer()
            await planogram_analyzer.initialize()
            analyzer = planogram_analyzer
            logger.info("ML")
            # Read the image content
            content = await image.read()
            # Convert to PIL Image
            pil_image = Image.open(io.BytesIO(content))
            # Process the image
            result = await analyzer.results(pil_image)
            # Force cleanup to free memory
            del pil_image
            del content
            return JSONResponse(
                status_code=200,
                content={
                    "data":{
                        "result": result
                    }
                }
            )
        except HTTPException as http_ex:
            raise http_ex
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error processing file: {str(e)}"
            )
        finally:
            gc.collect()
    except Exception as e:
        logger.error(f"Error in planogram API: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing file: {str(e)}"
        )
