FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Variables d'environnement
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV TORCH_COMPILE_DISABLE=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV HF_HOME=/workspace/model_cache
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Paris
ENV WINEARCH=win32
ENV WINEPREFIX=/root/.wine
ENV DISPLAY=:99
ENV MKL_SERVICE_FORCE_INTEL=1
ENV MKL_THREADING_LAYER=GNU

# Variables d'environnement pour le stockage persistant
ENV TRANSFORMERS_CACHE=/workspace/model_cache
ENV HF_HOME=/workspace/model_cache
ENV XDG_CACHE_HOME=/workspace/model_cache
ENV VLLM_CACHE_DIR=/workspace/model_cache/vllm

RUN mkdir -p /workspace/model_cache

# Répertoire de travail
WORKDIR /app

# Installation des dépendances système, Wine et Xvfb
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    tzdata \
    xvfb \
    xauth \
    libxrandr2 \
    libxcomposite1 \
    libxcursor1 \
    libxdamage1 \
    libxfixes3 \
    libxi6 \
    libxtst6 \
    libxss1 \
    libxft2 \
    wget \
    gnupg \
    && ln -fs /usr/share/zoneinfo/Europe/Paris /etc/localtime \
    && dpkg-reconfigure --frontend noninteractive tzdata \
    && rm -rf /var/lib/apt/lists/*

# Installation de Wine
RUN dpkg --add-architecture i386 && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    wine \
    wine32 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Création des répertoires nécessaires
RUN mkdir -p /app/.cache/huggingface && \
    mkdir -p /app/excel_files && \
    mkdir -p /app/model_cache && \
    chmod 777 /app/excel_files && \
    chmod 777 /app/model_cache

# Copie et installation des dépendances Python
COPY requirements.txt .

# Installation des dépendances en petits groupes pour éviter les problèmes d'espace
RUN pip install --no-cache-dir runpod
RUN pip install --no-cache-dir fastapi==0.95.0
RUN pip install --no-cache-dir uvicorn==0.21.0
RUN pip install --no-cache-dir pydantic==1.10.11 python-multipart==0.0.6
RUN pip install --no-cache-dir openpyxl==3.1.2
RUN pip install --no-cache-dir transformers>=4.36.0
RUN pip install --no-cache-dir sentence-transformers==2.2.2
RUN pip install --no-cache-dir numpy pandas
RUN pip install --no-cache-dir pyyaml requests
RUN pip install --no-cache-dir xlwings>=0.30.0
RUN pip install --no-cache-dir scikit-learn>=1.0.2
RUN pip install --no-cache-dir accelerate>=0.25.0
RUN pip install --no-cache-dir bitsandbytes>=0.41.0
RUN pip install --no-cache-dir --timeout 1200 vllm>=0.2.0
RUN pip install --no-cache-dir ray>=2.6.0
RUN pip install --no-cache-dir protobuf>=3.20.0
RUN pip install --no-cache-dir matplotlib>=3.5.0
RUN pip install --no-cache-dir seaborn>=0.11.2

# Copie du code source
COPY . .

# Rendre les scripts exécutables
RUN chmod +x /app/entrypoint.sh

# Script de démarrage pour RunPod dédié
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Créer et configurer les répertoires pour le stockage du modèle\n\
mkdir -p /workspace/model_cache/vllm\n\
mkdir -p /workspace/model_cache/huggingface\n\
chmod -R 777 /workspace/model_cache\n\
\n\
echo "🚀 Démarrage du service Excel MCP..."\n\
echo "📚 Versions installées:"\n\
python -c "import torch; print(f\"PyTorch: {torch.__version__}\")" || echo "❌ PyTorch non disponible"\n\
python -c "import transformers; print(f\"Transformers: {transformers.__version__}\")" || echo "❌ Transformers non disponible"\n\
python -c "import xlwings; print(f\"xlwings: {xlwings.__version__}\")" || echo "❌ xlwings non disponible"\n\
\n\
# Vérification GPU\n\
echo "🖥️ Informations GPU:"\n\
python -c "import torch; print(f\"CUDA disponible: {torch.cuda.is_available()}\"); print(f\"Appareil: {torch.cuda.get_device_name(0)}\") if torch.cuda.is_available() else print(\"GPU non disponible\")" || echo "❌ Erreur lors de la vérification GPU"\n\
\n\
# Lancer Xvfb pour permettre à Wine/Excel de fonctionner\n\
echo "🖥️ Démarrage de Xvfb..."\n\
Xvfb :99 -screen 0 1024x768x16 &\n\
export DISPLAY=:99\n\
\n\
# Initialiser Wine\n\
echo "🍷 Initialisation de Wine..."\n\
wineboot --init\n\
\n\
# Attendre quelques secondes pour l\'initialisation\n\
sleep 10\n\
\n\
# Exécuter le service principal\n\
echo "🔄 Démarrage du service handler.py sur le port 8001..."\n\
python -u handler.py\n\
' > /app/start.sh && chmod +x /app/start.sh

# Exposition du port
EXPOSE 8001

# Commande par défaut pour RunPod dédiée
CMD ["/app/start.sh"]