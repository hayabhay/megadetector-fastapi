# MegaDetector with FastAPI & Streamlit

This repo packages [MegaDetector](https://github.com/microsoft/CameraTraps/blob/main/megadetector.md) models and serves them over [FastAPI](https://fastapi.tiangolo.com/). Additionally, it comes with a simple [Streamlit](https://streamlit.io/) UI to visualize & compare annotations from different models.

Most of the code is directly copied from MegaDetector's source repo. Any additional code form thin wrappers to streamline some interfaces & dependencies for the API.

**Why FastAPI?** - FastAPI is async-first and also comes with nice things like Pydantic validation & Swagger UI out of the box. Async in the context of cloud endpoints and large batches of images is particularly useful since it allows images to be downloaded in parallel significantly lowering CPU-idle time and its associated cloud costs.

**Why Streamlit** - Streamlit is an extremely simple, pure Python way to build web GUIs. While Swagger UI allows API interactions, the outputs are still raw JSON. Streamlit complements this with visualization of annotated images with trivial amounts of python code.

![MegaDetector Streamlit demo](/assets/streamlit_demo.gif)

## Getting Started

**Pre-requisite**: A fresh `python-3.9` virtual environment ([Pyenv](https://realpython.com/intro-to-pyenv/)+[Virtualenv](https://virtualenv.pypa.io/en/latest/). [Conda](https://docs.conda.io/en/latest/) etc.). If you're using a GPU, this also assumes you have set up the necessary drivers.

> **Note**: This code is tested on `python 3.9` and will likely work on `python 3.7+` (untested for now).

**Step 1**: Clone this repo.

```
git clone https://github.com/abhay1/megadetector-fastapi.git
cd megadetector-fastapi
```

**Step 2**: Install python dependencies.

```
pip install -r requirements/requirements-dev.txt
```

> **Note**: There are several requirements files to choose from. For local development purposes, we'll install the GPU enabled version of PyTorch & Tensorflow. Later, when dockerizing it, you can pick and choose from them depending on the model you wish to use.

**Step 3**: Run the FastAPI app.
```
cd api
uvicorn main:app --port 8000 --reload
```
You should now be able to navigate to [http://127.0.0.1:8000](http://127.0.0.1:8000) and see a welcome message. FastAPI comes with [Swagger UI](https://swagger.io/tools/swagger-ui/) and [ReDoc](https://redocly.com/) out of the box that you can find at [/docs](http://127.0.0.1:8000/docs) & [/redoc](http://127.0.0.1:8000/redoc). You can use this to interact with the server and you'll see it's JSON responses.

![MegaDetector FastAPI demo](/assets/api_demo.gif)

**Step 4**: Run the Streamlit app.
```
cd ../ui/
streamlit run app.py
```
You should now be able to navigate to [http://127.0.0.1:8501](http://127.0.0.1:8501) to see the Streamlit app live!
`8501` is the default starting port and if you want to change it, you can pass it in like this
`streamlit run app.py --server.port 8001` or [use other supported methods](https://docs.streamlit.io/library/advanced-features/configuration).

> **Note**: Streamlit reads the FastAPI server location from an environment variable `MEGADETECTOR_API_URL` and defaults to `http://127.0.0.1:8000/` if it is missing. If the server port is different, you can directly update it in the `config.py` file or put it in a `.env` file (like this `MEGADETECTOR_API_URL='http://127.0.0.1:8000'`) which might be preferrable if the URL is public and shouldn't be on git.

## Deployment

There is a docker template to containerize the application and deploy it. Currently, the configuration is tailored to create containers that run on CPUs. This is primarily for Cloud based APIs, specifically, Google Cloud Run (soon to be tested).

Deployment strategies like Google Cloud Run spin up instances on demand. In this setting, instead of downloading the MegaDetector model everytime,  it might be useful to create images with the MegaDetector models baked in the container images. However, to keep them light, it is best to only have a specific model in addition to its required dependencies installed (PyTorch & Tensorflow).

## Batch of Images

FastAPI, with uvicorn is async by default i.e. subsequent calls to the server aren't blocked by I/O from previous calls. For synchronous services like Flask, this is usually supported by adding threads/workers to gunicorn. Regardless, to leverage this, the calling application also needs to use an async client like `httpx` or `aiohttp` instead of sync clients like `requests`. Or, it needs to make requests in a threadpool. ([more details](https://www.twilio.com/blog/asynchronous-http-requests-in-python-with-httpx-and-asyncio))

Here is an example of calling this API with `httpx` for a list of image urls (or base64 encoded strings).

```
import asyncio
import httpx

# List of image urls to call the API with
images = [...]

# Async function to get annotation for an image
async def get_annotation(image, client):
    response = await client.post(url,json={"image": image})
    return response.json()["annotation"]

# Use asyncio to make concurrent requests with httpx
async with httpx.AsyncClient() as client:
    # Create a list of tasks to run concurrently
    tasks = [
        get_annotation(
            image=image,
            url="http://127.0.0.1:8000/annotate/",
            client=client
        )
        for image in images
    ]
    # Run inference on each image in the list
    # * operator here simply unpacks the list
    results = await asyncio.gather(*tasks)
```
That's it!

## Issues, Comments & Feedback

- For bugs & feature requests, please [open an issue](https://github.com/abhay1/megadetector-fastapi/issues/new/choose).
- For feedback, suggestions & ideas, please leave a comment on the [discussion forum](https://github.com/abhay1/megadetector-fastapi/discussions).
