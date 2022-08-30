"""
This has a mix of config variables & initial setup for FastAPI
"""
import importlib
import os
import pathlib

from dotenv import load_dotenv

# First load all environment variables from .env file before any additional import
load_dotenv()

# First, make sure path is set up accurately
APP_DIR = pathlib.Path(__file__).parent.absolute()
# Next set the project root directory.
# Model directory & yolov5 repo should be in the same level as the app directory.
PROJECT_DIR = APP_DIR.parent

# Set the model directory & create if it doesn't exist
MD_MODELS_DIR = PROJECT_DIR / "md_models"
MD_MODELS_DIR.mkdir(exist_ok=True)

# Set the yolov5 directory
YOLOV5_DIR = PROJECT_DIR / "yolov5"
# Set the yolov5 source url.
# Note that this is a specific commit that MegaDetector uses.
YOLOV5_REPO_URL = "https://github.com/ultralytics/yolov5"
YOLOV5_COMMIT_HASH = "c23a441c9df7ca9b1f275e8c8719c949269160d1"
YOLOV5_SRC_ZIP = f"{YOLOV5_REPO_URL}/archive/{YOLOV5_COMMIT_HASH}.zip"
YOLOV5_SRC_DIRNAME = f"yolov5-{YOLOV5_COMMIT_HASH}"

# Move env variables into config variables with meaingful defaults
GOOGLE_PROJECT_ID = os.environ.get("GOOGLE_PROJECT_ID", None)


# -------------------------------------------------------------------------------------------
# This is copied from https://github.com/microsoft/CameraTraps/blob/main/detection/run_detector.py
# -------------------------------------------------------------------------------------------

# An enumeration of failure reasons
FAILURE_INFER = "Failure inference"
FAILURE_IMAGE_OPEN = "Failure image access"

# Number of decimal places to round to for confidence and bbox coordinates
CONF_DIGITS = 3
COORD_DIGITS = 4

# Each version of the detector is associated with some "typical" values
# that are included in output files, so that downstream applications can
# use them as defaults.
DETECTOR_METADATA = {}

# Version availability is determined by the presence of tensorflow/pytorch availability

# Add pytorch model versions & metadata if present
if importlib.util.find_spec("torch") is not None:
    DETECTOR_METADATA.update(
        {
            "v5a.0.0": {
                "megadetector_version": "v5a.0.0",
                "source_url": "https://github.com/microsoft/CameraTraps/releases/download/v5.0/md_v5a.0.0.pt",
                "typical_detection_threshold": 0.2,
                "conservative_detection_threshold": 0.05,
            },
            "v5b.0.0": {
                "megadetector_version": "v5b.0.0",
                "source_url": "https://github.com/microsoft/CameraTraps/releases/download/v5.0/md_v5b.0.0.pt",
                "typical_detection_threshold": 0.2,
                "conservative_detection_threshold": 0.05,
            },
        }
    )

# Add tensorflow model versions & metadata if present
if importlib.util.find_spec("tensorflow") is not None:
    DETECTOR_METADATA.update(
        {
            "v4.1.0": {
                "megadetector_version": "v4.1.0",
                "source_url": "https://lilablobssc.blob.core.windows.net/models/camera_traps/megadetector/md_v4.1.0/md_v4.1.0.pb",
                "typical_detection_threshold": 0.8,
                "conservative_detection_threshold": 0.3,
            },
            "v3.0.0": {
                "megadetector_version": "v3.0.0",
                "source_url": "https://lilablobssc.blob.core.windows.net/models/camera_traps/megadetector/megadetector_v3.pb",
                "typical_detection_threshold": 0.8,
                "conservative_detection_threshold": 0.3,
            },
            "v2.0.0": {
                "megadetector_version": "v2.0.0",
                "source_url": "https://lilablobssc.blob.core.windows.net/models/camera_traps/megadetector/megadetector_v2.pb",
                "typical_detection_threshold": 0.8,
                "conservative_detection_threshold": 0.3,
            },
        }
    )


DEFAULT_RENDERING_CONFIDENCE_THRESHOLD = DETECTOR_METADATA["v5b.0.0"]["typical_detection_threshold"]
DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD = 0.005
