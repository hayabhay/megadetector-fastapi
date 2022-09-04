import asyncio
import logging
import time
from typing import Optional

from config import DETECTOR_METADATA
from fastapi import FastAPI
from handlers.images import load_image
from handlers.models import get_megadetector_model
from pydantic import BaseModel, Field, validator

# Setup logger & environment variables
logger = logging.getLogger(__name__)

# Create a FastAPI app
app = FastAPI()

# Pydantic model for the request body
# -------------------------------------------------------------------------------------------
AVAILABLE_MD_MODELS = list(DETECTOR_METADATA.keys())


class AnnotateRequest(BaseModel):
    image: str = Field(
        default="https://raw.githubusercontent.com/microsoft/CameraTraps/main/images/nacti.jpg",
        description="Web url or a base64 encoded image to annotate",
    )
    megadetector_version: str = Field(AVAILABLE_MD_MODELS[0], description="Version of megadetector model to use")
    detection_threshold: Optional[float] = Field(default=None, description="Optional Detection threshold")
    load_multiple_models: bool = Field(default=False, description="Load multiple models in memory")

    @validator("megadetector_version")
    def megadetector_version_validator(cls, version):
        if version not in DETECTOR_METADATA:
            raise ValueError(f"Invalid megadetector version: {version}. Please choose from {AVAILABLE_MD_MODELS}")
        return version


# API endpoints
# -------------------------------------------------------------------------------------------
@app.post("/annotate/")
async def get_annotated_image(request: AnnotateRequest):
    """Main function that consumes an image list object and returns a list of Megadetector annotations"""
    # Load the megadetector model. To keep the memory footprint low, keep only one model at any time
    # For dev purposes, we can load multiple models in memory
    start = time.perf_counter()
    image, model = await asyncio.gather(
        load_image(request.image),
        get_megadetector_model(request.megadetector_version, load_multiple_models=request.load_multiple_models),
    )

    # Set the detection threshold if provided
    if request.detection_threshold is None:
        detection_threshold = model.detection_threshold
    else:
        detection_threshold = request.detection_threshold

    annotation = model.annotate_image(image, request.image[:255], detection_threshold)
    annotation_time = time.perf_counter() - start

    return {
        "annotation": annotation,
        "megadetector_version": request.megadetector_version,
        "detection_threshold": detection_threshold,
        "annotation_time": annotation_time,
    }


@app.get("/available_models/")
def get_available_models():
    """Returns a list of available models & its metadata"""
    return {"available_models": DETECTOR_METADATA}


@app.get("/")
def home():
    return "I'm alive & healthy! To interact with me, go to /docs for Swagger UI or /redoc for ReDoc."
