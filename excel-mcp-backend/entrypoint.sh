#!/bin/bash

# Lancer Xvfb pour permettre à Wine/Excel de fonctionner
Xvfb :99 -screen 0 1024x768x16 &
export DISPLAY=:99

# Initialiser Wine (première exécution)
wineboot --init

# Attendre que Wine soit initialisé
sleep 5

# Lancer le serveur FastAPI
uvicorn handler:app --host 0.0.0.0 --port 8001