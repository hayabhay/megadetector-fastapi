import base64
import logging
from io import BytesIO

import httpx
import numpy as np
from config import GOOGLE_PROJECT_ID
from google.cloud import storage
from PIL import Image

# Setup logger & environment variables
logger = logging.getLogger(__name__)

# Placeholder for Google storage client & buckets
storage_client = None
BUCKETS = {}


# Function to get a PIL Image from google cloud storage from a gs:// url
# This is mainly for buckets without public access
async def load_gs_image(image_gs_loc: str) -> Image:
    """Function to load an image directly from a google storage link"""
    global storage_client
    global BUCKETS

    if storage_client is None:
        if not GOOGLE_PROJECT_ID:
            raise ValueError("GOOGLE_PROJECT_ID environment variable must be set")
        storage_client = storage.Client(project=GOOGLE_PROJECT_ID)

    # First get the bucket name from the url.
    bucket_name = image_gs_loc[5:].split("/")[0]
    file_path = "/".join(image_gs_loc[5:].split("/")[1:])

    if bucket_name not in BUCKETS:
        BUCKETS[bucket_name] = storage_client.get_bucket(bucket_name)

    blob = BUCKETS[bucket_name].get_blob(file_path)
    try:
        # Load the blob as a binary string
        binary_string = blob.download_as_string()
        # Try to interpret the bytes as an image
        image = Image.open(BytesIO(binary_string))
    except Exception as e:
        logger.error(f"Error loading image from google storage: {e}")
        raise e

    return image


# Function to get a PIL Image from an HTTP url
async def load_web_image(image_url: str, client: httpx.AsyncClient = None) -> Image:
    """Function to load an image from a web url"""
    try:
        # If no client is passed, treat this as a one-off and create a new client
        if not client:
            async with httpx.AsyncClient() as client:
                response = await client.get(image_url, timeout=30)
        # Use the passed client to get the image
        else:
            response = await client.get(image_url)
        # Try to interpret the bytes as an image
        image = Image.open(BytesIO(response.content))
    except Exception as e:
        logger.error(f"Error loading image from web url: {e}")
        raise e

    return image


# Function to get a PIL Image from a string
async def load_image_from_binary_string(image_byte_string: str) -> Image:
    """Function to load an image directly from a binary string"""
    try:
        # Try to interpret the bytes as an image
        image_binary = base64.decodebytes(image_byte_string.encode())
        image = Image.open(BytesIO(image_binary))
    except Exception as e:
        logger.error(f"Error loading image from binary string: {e}")
        raise e

    return image


# Function to load an image from the web, gs or a binary string
async def load_image(image_src: str, client: httpx.AsyncClient = None) -> Image:
    """Function that takes in a string, determines the source and returns a correctly formatted PIL Image"""
    # Load the image depending on the type
    # TODO: This is a little shabby and fails if the file is not an image without a specific error message
    if image_src.startswith(("http://", "https://")):
        image = await load_web_image(image_src, client)
    elif image_src.startswith("gs://"):
        image = await load_gs_image(image_src)
    else:
        image = await load_image_from_binary_string(image_src)

    # ==========================================================================================
    # THIS CODE IS FROM
    # https://github.com/microsoft/CameraTraps/blob/master/visualization/visualization_utils.py
    if image.mode not in ("RGBA", "RGB", "L", "I;16"):
        raise AttributeError(f"Image {image_src} uses unsupported mode {image.mode}")

    if image.mode == "RGBA" or image.mode == "L":
        # PIL.Image.convert() returns a converted copy of this image
        image = image.convert(mode="RGB")
    # ==========================================================================================

    return image
