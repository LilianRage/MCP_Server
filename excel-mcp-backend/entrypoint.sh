#!/bin/bash

# Lancer Xvfb pour permettre à Wine/Excel de fonctionner
Xvfb :99 -screen 0 1024x768x16 &
export DISPLAY=:99

export CUDA_VISIBLE_DEVICES=0  # Utilisez le GPU 0
export VLLM_USE_CUDA_GRAPH=1  # Activer CUDA Graph pour une inférence plus rapide
export VLLM_MAX_GPU_MEMORY="42GiB"

# Initialiser Wine (première exécution)
wineboot --init

# Attendre que Wine soit initialisé
sleep 5

# Lancer le serveur FastAPI
uvicorn handler:app --host 0.0.0.0 --port 8001