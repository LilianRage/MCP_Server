import os
import json
import logging
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import torch
from embedding_handler import retrieve_relevant_rows, get_embedding_status

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Excel Embedding API")

# Configuration CORS pour permettre les requêtes cross-origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

@app.get("/status")
async def get_status():
    """Vérifier l'état du service d'embedding"""
    embedding_status = get_embedding_status()
    
    # Vérifier si le GPU est disponible
    gpu_info = {
        "available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "memory": None
    }
    
    # Récupérer la mémoire GPU si disponible
    if torch.cuda.is_available():
        try:
            memory_allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)  # En Go
            memory_reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)  # En Go
            gpu_info["memory"] = {
                "allocated_gb": round(memory_allocated, 2),
                "reserved_gb": round(memory_reserved, 2)
            }
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des infos mémoire GPU: {str(e)}")
    
    return {
        "status": "online",
        "embedding": embedding_status,
        "gpu": gpu_info
    }

@app.post("/analyze_data")
async def analyze_data(request: Request):
    """
    Analyse les données Excel avec embedding
    
    Attend un JSON avec:
    - excel_data: données formatées avec headers, rows, etc.
    - query: requête de recherche
    - top_k: nombre de résultats à retourner
    """
    try:
        data = await request.json()
        excel_data = data.get("excel_data", {})
        query = data.get("query", "")
        top_k = data.get("top_k", 5)
        
        logger.info(f"Requête d'analyse reçue: query={query}, top_k={top_k}")
        
        if not excel_data or not query:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Paramètres manquants: excel_data et query sont requis"}
            )
        
        # Vérifier que les données Excel sont correctement formatées
        if not all(k in excel_data for k in ["headers", "rows"]):
            return JSONResponse(
                status_code=400, 
                content={"success": False, "error": "Format de données Excel invalide"}
            )
        
        logger.info(f"Analyse de données Excel: {len(excel_data.get('rows', []))} lignes, {len(excel_data.get('headers', []))} colonnes")
        
        # Utiliser les embeddings pour trouver les lignes pertinentes
        results = retrieve_relevant_rows(query, excel_data, top_k)
        
        return {
            "success": True,
            "result": results
        }
        
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse avec embedding: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

# Pour les tests locaux et le déploiement RunPod
if __name__ == "__main__":
    # Afficher les informations GPU au démarrage
    logger.info("=== INFORMATIONS GPU ===")
    logger.info(f"GPU disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"Nombre de GPU: {torch.cuda.device_count()}")
        logger.info(f"Nom du GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Index GPU actif: {torch.cuda.current_device()}")
    logger.info("=======================")
    
    # Démarrer le serveur
    port = int(os.environ.get("PORT", 8001))
    logger.info(f"Démarrage du service d'embedding sur le port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)