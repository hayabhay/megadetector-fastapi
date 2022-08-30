#!/bin/bash

# Deploy uvicorn
cd /api
nohup uvicorn main:app --host 0.0.0.0 --port 8000 > /uvicorn.log 2>&1 &

# Deploy streamlit
cd /ui
nohup streamlit run app.py --server.port 8501 > /streamlit.log 2>&1 &
