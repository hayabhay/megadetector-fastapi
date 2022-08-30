import os

from dotenv import load_dotenv

# First load all environment variables from .env file before any additional import
load_dotenv()

# Load MegaDetector url & validate
MEGADETECTOR_API_URL = os.environ.get("MEGADETECTOR_API_URL", "http://127.0.0.1:8000")
