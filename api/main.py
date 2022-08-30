import asyncio
import logging
import time
from enum import Enum
from typing import List, Optional

import httpx
from config import DETECTOR_METADATA
from fastapi import FastAPI
from handlers.images import load_image
from handlers.models import MegaDetector, get_megadetector_model
from pydantic import AnyUrl, BaseModel, Field

# Setup logger & environment variables
logger = logging.getLogger(__name__)

# Create a FastAPI app
app = FastAPI()


# Main helper function
# -------------------------------------------------------------------------------------------
async def annotate_image(
    image_src: str,
    model: MegaDetector,
    detection_threshold: Optional[float] = None,
    client: httpx.AsyncClient = None,
) -> dict:
    """Download image, run inference, and return results"""
    # Download image from url
    image = await load_image(image_src, client)
    # Run inference on image. All image names are capped at 255 characters
    annotation = model.annotate_image(image, image_src[:255], detection_threshold)
    return annotation


# Pydantic model for the request body
# -------------------------------------------------------------------------------------------
# Available model versions are defined in config.py
# This is a just functional API call to Enum - https://docs.python.org/3/library/enum.html#functional-api
MD_VERSION = Enum("MD_VERSION", {version: version for version in DETECTOR_METADATA})


class ImagesToAnnotate(BaseModel):
    images: List[str] = Field(
        default=[
            "https://raw.githubusercontent.com/microsoft/CameraTraps/main/images/nacti.jpg",
            "https://raw.githubusercontent.com/microsoft/CameraTraps/main/images/detector_example.jpg",
        ],
        description="List of urls to images that need to be annotated",
    )
    megadetector_version: MD_VERSION = Field(..., description="Version of megadetector model to use")
    detection_threshold: Optional[float] = Field(default=None, description="Optional Detection threshold")
    load_multiple_models: bool = Field(default=False, description="Load multiple models in memory")


# API endpoints
# -------------------------------------------------------------------------------------------
@app.post("/annotate_images/")
async def get_annotated_images(images_to_annotate: ImagesToAnnotate):
    """Main function that consumes an image list object and returns a list of Megadetector annotations"""
    # Load the megadetector model. To keep the memory footprint low, keep only one model at any time
    # For dev purposes, we can load multiple models in memory
    start = time.perf_counter()
    model = get_megadetector_model(
        images_to_annotate.megadetector_version.value, load_multiple_models=images_to_annotate.load_multiple_models
    )
    model_load_time = time.perf_counter() - start

    # Set the detection threshold if provided
    if images_to_annotate.detection_threshold is None:
        detection_threshold = model.detection_threshold
    else:
        detection_threshold = images_to_annotate.detection_threshold

    start = time.perf_counter()
    # Create an async client to pull images in async
    # Note that currently this helps for http(s) images only
    async with httpx.AsyncClient() as client:
        # Create a list of tasks to run concurrently
        annotate_tasks = [
            annotate_image(image_src=image_src, model=model, detection_threshold=detection_threshold, client=client)
            for image_src in images_to_annotate.images
        ]
        # Run inference on each image in the list
        results = await asyncio.gather(*annotate_tasks)
    annotation_time = time.perf_counter() - start

    return {
        "annotations": results,
        "megadetector_version": images_to_annotate.megadetector_version.value,
        "detection_threshold": detection_threshold,
        "model_load_time": model_load_time,
        "annotation_time": annotation_time,
    }


@app.get("/available_models/")
def get_available_models():
    """Returns a list of available models & its metadata"""
    return {"available_models": DETECTOR_METADATA}


@app.get("/")
def home():
    return "I'm alive & healthy! To interact with me, go to /docs for Swagger UI or /redoc for ReDoc."
