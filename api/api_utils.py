"""
This has a mix of functions from different repos.
NOTE: Naming this module "utils" at the top level will directly conflict with the "utils" module in the yolov5 repo
so this must be given a different name
"""
import logging
import math
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import httpx
import numpy as np
from config import YOLOV5_DIR, YOLOV5_SRC_ZIP

# Setup logger & environment variables
logger = logging.getLogger(__name__)


# A simple function to download the yolov5 repo & set HEAD to the specific commit required by MegaDetector
def download_yolov5():
    """
    This clones the yolov5 repo and sets it to a specific commit
    This requires git to be installed underneath
    """
    if Path(YOLOV5_DIR).exists():
        logger.info(f"{YOLOV5_DIR} already exists, skipping download")
    else:
        logger.info(f"{YOLOV5_DIR} already exists, skipping download")
        # Download a zipfile from the url and extract it in memory
        try:
            logger.info(f"Downloading yolov5 source zip from {YOLOV5_SRC_ZIP}")
            response = httpx.get(YOLOV5_SRC_ZIP, follow_redirects=True)
            zipfile = ZipFile(BytesIO(response.content))
            zipfile.extractall(YOLOV5_DIR)
        except Exception as e:
            logger.error(f"Error downloading yolov5 source zip: {e}")
            raise e


# -------------------------------------------------------------------------------------------
# This is copied directly from https://github.com/microsoft/CameraTraps/blob/main/ct_utils.py
# -------------------------------------------------------------------------------------------


def truncate_float_array(xs, precision=3):
    """
    Vectorized version of truncate_float(...)
    Args:
    x         (list of float) List of floats to truncate
    precision (int)           The number of significant digits to preserve, should be
                              greater or equal 1
    """

    return [truncate_float(x, precision=precision) for x in xs]


def truncate_float(x, precision=3):
    """
    Function for truncating a float scalar to the defined precision.
    For example: truncate_float(0.0003214884) --> 0.000321
    This function is primarily used to achieve a certain float representation
    before exporting to JSON
    Args:
    x         (float) Scalar to truncate
    precision (int)   The number of significant digits to preserve, should be
                      greater or equal 1
    """

    assert precision > 0

    if np.isclose(x, 0):
        return 0
    else:
        # Determine the factor, which shifts the decimal point of x
        # just behind the last significant digit
        factor = math.pow(10, precision - 1 - math.floor(math.log10(abs(x))))
        # Shift decimal point by multiplicatipon with factor, flooring, and
        # division by factor
        return math.floor(x * factor) / factor


def convert_yolo_to_xywh(yolo_box):
    """
    Converts a YOLO format bounding box to [x_min, y_min, width_of_box, height_of_box].
    Args:
        yolo_box: bounding box of format [x_center, y_center, width_of_box, height_of_box].
    Returns:
        bbox with coordinates represented as [x_min, y_min, width_of_box, height_of_box].
    """
    x_center, y_center, width_of_box, height_of_box = yolo_box
    x_min = x_center - width_of_box / 2.0
    y_min = y_center - height_of_box / 2.0
    return [x_min, y_min, width_of_box, height_of_box]
