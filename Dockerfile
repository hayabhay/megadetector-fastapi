# Start with official python base image
FROM python:3.9-slim

# Install wget
RUN apt-get -y update \
    && apt-get -y autoremove \
    && apt-get clean \
    && apt-get install \
    && apt-get install -y wget \
    zip \
    unzip

# Set working directory to app
WORKDIR /app

# Copy the entire requirements directory
COPY ./requirements /requirements

# Pip install requirements
RUN pip install --upgrade pip

# To keep the image light, install only version specfic requirements
# Megadetector v5 requires pytorch while v4 and below require tensorflow

# RUN pip install --no-cache-dir -r /requirements/requirements-pytorch-cpu.txt
RUN pip install --no-cache-dir -r /requirements/requirements-pytorch.txt
#RUN pip install --no-cache-dir -r /requirements/requirements-tensorflow-cpu.txt
#RUN pip install --no-cache-dir -r /requirements/requirements-tensorflow.txt

# Download yolov5 dependency as a zipfile & extract it
RUN mkdir /yolov5 \
    && cd /yolov5 \
    && wget https://github.com/ultralytics/yolov5/archive/c23a441c9df7ca9b1f275e8c8719c949269160d1.zip \
    && unzip c23a441c9df7ca9b1f275e8c8719c949269160d1.zip \
    && rm c23a441c9df7ca9b1f275e8c8719c949269160d1.zip


# Move specific models to md_models directory to bake them into the image
# This reduces the need for containers to download models everytime they're triggered in cloud run
# This assumes that md_models in the local project already has models downloaded
# Regardless, non-preloaded models will still be downloaded and installed
COPY ./md_models/md_v5a.0.0.pt /md_models/md_v5a.0.0.pt
# COPY ./md_models/md_v5b.0.1.pt /md_models/md_v5b.0.1.pt
# COPY ./md_models/md_v4.1.0.pb /md_models/md_v4.1.0.pb
# COPY ./md_models/megadetector_v3.pb /md_models/megadetector_v3.pb
# COPY ./md_models/megadetector_v2.pb /md_models/megadetector_v2.pb

# Copy app code
COPY ./api /api

# Run uvicorn
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# If you just want the API, uncomment the line above & comment out the line below
# ----------------------------------------------------------------------------------
# Copy ui code
COPY ./ui /ui
# Move the startup script
COPY ./start.sh /start.sh
RUN chmod +x /start.sh
# Run the startup script
CMD ["/start.sh"]
