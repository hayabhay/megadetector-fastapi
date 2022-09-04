"""
This module is the primary entry point for to interface with all the MegaDetector models.
"""
import logging
import time
from pathlib import Path
from typing import Any, Optional

import httpx
from config import DETECTOR_METADATA, MD_MODELS_DIR
from PIL import Image

logger = logging.getLogger(__name__)


class MegaDetector:
    """This is a thin wrapper class around MegaDetector's TFDetector and/or PTDetector"""

    def __init__(self, model_version: str) -> None:
        # If the model version is not in the metadata, raise an error
        if model_version not in DETECTOR_METADATA:
            raise ValueError(f"Model {model_version} is not supported")

        # Set model version & load metadata
        self.model_version = model_version
        self.model_metadata = DETECTOR_METADATA[self.model_version]
        self.model_url = self.model_metadata["source_url"]
        self.model_file = self.model_url.split("/")[-1]
        self.model_path = f"{MD_MODELS_DIR}/{self.model_file}"
        self.detection_threshold = self.model_metadata["typical_detection_threshold"]
        # Download megadetector model if it's not already downloaded & get its path
        self._download_megadetector_model()
        # Link down to the underlying model object
        self.model = self._get_model()

    def _download_megadetector_model(self) -> None:
        """Downloads the MegaDetector model to the specific cache location if it isn't already downloaded"""
        # Check the models directory cache for the specific model file. If it's not there, download it.
        if Path(self.model_path).exists():
            logger.info(f"{self.model_file} already downloaded!")
        else:
            # If not, download the model from the source url
            logger.info(f"Downloading {self.model_file} from {self.model_url}")
            try:
                start = time.perf_counter()
                # Download the model from the source url and save it to the cache
                # httpx will not follow redirects by default, so we need to explicitly set it to True
                response = httpx.get(self.model_url, follow_redirects=True)

                with open(self.model_path, "wb") as f:
                    f.write(response.content)
                logger.info(f"{self.model_file} downloaded in {time.perf_counter()-start}s")
            except Exception as e:
                logger.error(f"Error downloading model: {e}")
                raise e

    # Return left at Any to prevent tensorflow/pytorch specific import regardless of specified model version
    def _get_model(self) -> Any:
        """Loads the specific model from the model path & returns the model object"""
        # -------------------------------------------------------------------------------------------
        # This is paraphrased from https://github.com/microsoft/CameraTraps/blob/main/detection/run_detector.py
        # -------------------------------------------------------------------------------------------
        if self.model_file.endswith(".pb"):
            start = time.perf_counter()
            from detectors.tensorflow_detector import TFDetector

            model = TFDetector(self.model_path)
            logger.info(f"{self.model_file} loaded in {time.perf_counter()-start}s")
        elif self.model_file.endswith(".pt"):
            start = time.perf_counter()
            from detectors.pytorch_detector import PTDetector

            model = PTDetector(self.model_path)
            logger.info(f"{self.model_file} loaded in {time.perf_counter()-start}s")

        # Both models have the same function signature to run inference
        return model

    def annotate_image(self, image: Image, filename: str, detection_threshold: Optional[float] = None) -> dict:
        """Runs the "generation_detections_one_image" function on the model object"""
        # Failsafe: If no detection threshold is provided, use the model's default detection threshold
        if detection_threshold is None:
            detection_threshold = self.detection_threshold

        return self.model.generate_detections_one_image(image, filename, detection_threshold)


# This is a global variable to cache loaded models
# This can be set in the loader function to have multiple models in memory or just one
MODELS = {}


# Since the models are conditionally loaded to keep the dependencies & footprint light, the return type is Any.
async def get_megadetector_model(model_version: str, load_multiple_models: bool = False) -> Any:
    """Initializes a MegaDetector model and returns it if it's not already cached

    Args:
        model: MegaDetector model name

    Returns:
        One of TFDetector or PTDetector models.

    """
    # global so loaded models can be updated
    global MODELS

    if model_version not in MODELS:
        logger.info(f"{model_version} not in memory! Loading..")
        # Create a MegaDetector model object and cache it
        model = MegaDetector(model_version)

        if load_multiple_models:
            MODELS[model_version] = model
        else:
            MODELS = {model_version: model}
    else:
        logger.info(f"{model_version} already in memory!")
        model = MODELS[model_version]

    # Return the model
    return model
