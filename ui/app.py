import base64
import json
from io import BytesIO
from typing import Optional

import httpx
import streamlit as st
from config import MEGADETECTOR_API_URL
from image_utils import render_detection_bounding_boxes
from PIL import Image

# Set page config
st.set_page_config(
    page_title="MegaDetector Demo",
    layout="wide",
    menu_items={
        "Report a Bug": "https://github.com/abhay1/megadetector-fastapi/issues/new/choose",
        "About": """
        ## MegaDetector UI

        This is a visual demo of MegaDetector models.
        This is currently tightly coupled to the FastAPI endpoint but can be easily adapted to any other API service.

        For bugs & feature requests, please open an issue on [GitHub](https://github.com/abhay1/megadetector-fastapi/issues/new/choose).
        For feedback, suggestions & ideas, please leave a comment [here](https://github.com/abhay1/megadetector-fastapi/discussions).

        ---
        """,
    },
)


# Cached data loads
# -------------------------------------------------------------------------------------------
# Download an image
@st.experimental_memo
def get_image(image_url: str) -> Image:
    """Get image from url"""
    response = httpx.get(image_url)
    image = Image.open(BytesIO(response.content))
    return image


# Download an image
@st.experimental_memo(suppress_st_warning=True)
def get_available_models():
    """Get available models"""
    # Try to get the available models from the API
    try:
        endpoint = f"{MEGADETECTOR_API_URL}/available_models/"
        response = httpx.get(endpoint)
        available_models = response.json()["available_models"]
        return available_models
    except Exception as e:
        st.error(
            f"""
            `Exception: {e}`

            ### Error getting available MegaDetector models.
            ```
            This is likely because your server is either not running or the API URL in your `.env` file is incorrect.
            Please check your FastAPI server and click refresh to try again.
            ```
        """
        )
        return None


# Call API to annotate image
@st.experimental_memo
def get_annotations(
    image_src: str,
    megadetector_version: str,
    detection_threshold: Optional[float] = None,
    load_multiple_models: bool = True,
) -> dict:
    """Call MegaDetector API to annotate the image"""
    # Set payload with image source & megadetector version
    payload = {
        "image": image_src,
        "megadetector_version": megadetector_version,
        "load_multiple_models": load_multiple_models,
    }
    # Update detection threshold if it exists. If not, leave it as None.
    if detection_threshold is not None:
        payload["detection_threshold"] = detection_threshold

    # Send request to MegaDetector API
    endpoint = f"{MEGADETECTOR_API_URL}/annotate/"
    response = httpx.post(endpoint, json=payload, timeout=120)

    annotation = response.json()

    return annotation


# Get available models
# Note: This assumes that the API is already running. If it isn't, it'll throw an error
DETECTOR_METADATA = get_available_models()
# If fetching models fail, clear cache and halt page rendering.
if not DETECTOR_METADATA:
    st.experimental_memo.clear()
    st.stop()

# Render a sidebar menu to select an image first
# -------------------------------------------------------------------------------------------
st.sidebar.markdown(
    """
## Select an image to start
"""
)

# Select image input option
image_input_option = st.sidebar.radio("How would you like to add your image?", ("Upload", "Web link"))

# Set placeholder for image source
image_src = None
image = None
selected_model_versions = []

if image_input_option == "Upload":
    # Source file upload or image url
    uploaded_file = st.sidebar.file_uploader("Choose an image")
    if uploaded_file is not None:
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()
        image = Image.open(BytesIO(bytes_data))
        # To read file as base64:
        image_src = base64.b64encode(bytes_data).decode()
elif image_input_option == "Web link":
    # Image url
    image_src = st.sidebar.text_input("Image URL")
    if image_src:
        image = get_image(image_src)


# Select subset of models
model_list = list(DETECTOR_METADATA.keys())
selected_model_versions = st.sidebar.multiselect(
    "Select MegaDetector models", options=model_list, default=model_list[0]
)

# Populate detection tresholds dynamically
detection_thresholds = {}
for model_version in selected_model_versions:
    detection_treshold = st.sidebar.number_input(
        f"Detection threshold for {model_version}",
        key=f"{model_version}-threshold",
        min_value=0.0,
        step=0.2,
        max_value=1.0,
        value=DETECTOR_METADATA[model_version]["typical_detection_threshold"],
    )
    detection_thresholds[model_version] = round(detection_treshold, 2)

# Generic settings
with st.sidebar.expander("More settings"):
    load_multiple_models = st.checkbox("Keep multiple models in server memory?", value=True)


# Render forms to select version and detection threshold if image is selected
# -------------------------------------------------------------------------------------------
# If an image & a model version is selected, annotate & render.
if image_src and selected_model_versions:
    # Render a tab per model
    tabs = st.tabs(selected_model_versions + ["original"])
    tabs[-1].image(image)
    annotated_image = {}
    # For each model, render the annotated image
    for i, model_version in enumerate(selected_model_versions):
        # Get image annotation
        annotations = get_annotations(
            image_src=image_src,
            megadetector_version=model_version,
            detection_threshold=detection_thresholds[model_version],
            load_multiple_models=load_multiple_models,
        )

        # Get the first annotion since there is only one image
        annotation = annotations["annotation"]

        # First create a collapsible section raw output
        with tabs[i].expander("API Response (JSON)"):
            # Render the raw json
            # st.write(get_annotation(image_src, model))
            st.code(json.dumps(annotations, indent=4))

        # Since render function adds boxes in place, make a copy and pass it to render function
        annotated_image[model_version] = image.copy()

        # Add bounding boxes onto the image
        render_detection_bounding_boxes(
            annotation["detections"],
            annotated_image[model_version],
            label_map={"1": "Animal", "2": "Person", "3": "Vehicle"},
            confidence_threshold=detection_thresholds[model_version],
        )

        # Render the annotated image
        tabs[i].markdown(f"""#### `Annotation Time: {annotations["annotation_time"]:.2f}s`""")
        tabs[i].image(annotated_image[model_version], caption=f"Annotated image with {model_version}")
# Else render the homepage text
else:
    st.markdown(
        """
    ## Streamlit + MegaDetector = 🚀
    > **_Visualize, examine & compare annotations from different MegaDetector models._**
    ---

    *Steps in no particular order*

    - **Select one or more MegaDetector models to run on your image.**
    > Detection thresholds are set to the typical detection threshold for each model but you can adjust them.

    - **Either upload an image from your computer or enter a web link.**
    >If you choose the former, your image will be sent to the server as a `base64` encoded string. If you choose the latter, the image will be downloaded from the web link.

    Once you select the image, Streamlit will call the MegaDetector API with the selected models and detection thresholds.
    It will then render the annotated image for each model along with the raw JSON that you can easily toggle and compare.

    > For bugs & feature requests, please open an issue on [GitHub](https://github.com/abhay1/megadetector-fastapi/issues/new/choose).
    > For feedback, suggestions & ideas, please leave a comment [here](https://github.com/abhay1/megadetector-fastapi/discussions).
    """
    )
