import os
import json
import logging
import threading
import uuid
import time
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from embedding_handler import retrieve_relevant_rows, get_embedding_status
from llm_handler import analyze_with_llm, get_llm_status, generate_excel_modification
from excel_processor_mcp import (
    update_cell_value,
    update_range_values,
    execute_excel_formula,
    get_sheet_data
)
import json

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Excel Analysis API")

# Configuration CORS pour permettre les requêtes du frontend ou de l'agent local
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Stockage en mémoire des tâches asynchrones
# Dans un environnement de production, utilisez une base de données ou Redis
TASKS = {}

# Période d'expiration des tâches (24 heures en secondes)
TASK_EXPIRY = 24 * 60 * 60

@app.get("/status")
async def get_status():
    """Vérifier l'état des services"""
    embedding_status = get_embedding_status()
    llm_status = get_llm_status()
    
    # Informations VLLM supplémentaires si disponible
    vllm_info = {}
    if llm_status.get("engine") == "VLLM" and llm_status.get("available"):
        try:
            vllm_info = {
                "engine": "VLLM",
                "version": "optimized",
                "max_batch_size": 32,  # Par défaut dans VLLM
                "tensor_parallel_size": 1  # Modifié si vous utilisez plusieurs GPUs
            }
        except:
            pass
    
    return {
        "status": "online",
        "embedding": embedding_status,
        "llm": {**llm_status, **vllm_info}
    }

@app.post("/")
async def run_pod_handler(request: Request):
    """Gestionnaire pour API RunPod Serverless"""
    try:
        # Récupérer la requête
        body = await request.json()
        input_data = body.get("input", {})
        
        # Extraire la commande et les paramètres
        command = input_data.get("command", "status")
        params = input_data.get("params", {})
        
        logger.info(f"RunPod API Request: command={command}, params={params}")
        
        # Traiter les différentes commandes
        if command == "status":
            embedding_status = get_embedding_status()
            llm_status = get_llm_status()
            result = {
                "status": "online",
                "embedding": embedding_status,
                "llm": llm_status
            }
        elif command == "analyze_data":
            result = await analyze_data(params)
        elif command == "analyze_with_llm":
            result = await analyze_data_llm(params)
        elif command == "submit_llm_task":
            result = await submit_llm_task(params)
        elif command == "get_task_status":
            result = get_task_status(params.get("task_id"))
        elif command == "modify_excel_with_llm":
            query = params.get("query", "")
            excel_data = params.get("excel_data", {})
            
            if not query or not excel_data:
                result = {
                    "success": False,
                    "error": "Paramètres manquants: query et excel_data sont requis"
                }
            else:
                result = generate_excel_modification(query, excel_data)
        else:
            result = {"success": False, "error": f"Commande inconnue: {command}"}
        
        # Renvoyer le résultat formaté pour RunPod
        return {"output": result}
        
    except Exception as e:
        logger.error(f"Erreur lors du traitement de la requête RunPod: {str(e)}")
        return {"output": {"success": False, "error": str(e)}}

async def analyze_data(params):
    """Analyse les données Excel avec embedding"""
    try:
        # Récupérer les paramètres
        excel_data = params.get("excel_data", {})
        query = params.get("query", "")
        top_k = params.get("top_k", 5)
        
        if not excel_data or not query:
            return {
                "success": False,
                "error": "Paramètres manquants: excel_data et query sont requis"
            }
        
        # Utiliser les embeddings pour trouver les lignes pertinentes
        results = retrieve_relevant_rows(query, excel_data, top_k)
        
        return {
            "success": True,
            "result": results
        }
        
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse avec embedding: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

async def analyze_data_llm(params):
    """
    Analyse les données Excel avec LLM et génération de code Python
    Cette méthode est maintenant déconseillée en faveur de l'approche asynchrone
    """
    try:
        # Récupérer les paramètres
        excel_data = params.get("excel_data", {})
        query = params.get("query", "")
        
        if not excel_data or not query:
            return {
                "success": False,
                "error": "Paramètres manquants: excel_data et query sont requis"
            }
        
        # Utiliser LLM pour analyser les données
        results = analyze_with_llm(query, excel_data)
        
        return results
        
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse avec LLM: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

async def submit_llm_task(params):
    """
    Soumet une tâche d'analyse LLM asynchrone
    """
    try:
        # Récupérer les paramètres
        excel_data = params.get("excel_data", {})
        query = params.get("query", "")
        
        if not excel_data or not query:
            return {
                "success": False,
                "error": "Paramètres manquants: excel_data et query sont requis"
            }
        
        # Générer un ID unique pour la tâche
        task_id = str(uuid.uuid4())
        
        # Créer une entrée pour cette tâche
        TASKS[task_id] = {
            "status": "pending",
            "created_at": time.time(),
            "excel_data": excel_data,
            "query": query,
            "result": None,
            "error": None
        }
        
        logger.info(f"Tâche d'analyse LLM créée: {task_id}")
        
        # Lancer l'analyse en arrière-plan dans un thread séparé
        def run_analysis_in_background(task_id, query, excel_data):
            try:
                logger.info(f"Démarrage de l'analyse en arrière-plan pour la tâche {task_id}")
                TASKS[task_id]["status"] = "processing"
                
                # Analyser les données avec LLM
                results = analyze_with_llm(query, excel_data)
                
                if results.get("success", False):
                    TASKS[task_id]["status"] = "completed"
                    TASKS[task_id]["result"] = results
                else:
                    TASKS[task_id]["status"] = "failed"
                    TASKS[task_id]["error"] = results.get("error", "Erreur inconnue")
                
                logger.info(f"Analyse terminée pour la tâche {task_id} avec statut {TASKS[task_id]['status']}")
            
            except Exception as e:
                logger.error(f"Erreur lors de l'analyse en arrière-plan pour la tâche {task_id}: {str(e)}")
                TASKS[task_id]["status"] = "failed"
                TASKS[task_id]["error"] = str(e)
        
        # Démarrer le thread d'analyse
        threading.Thread(
            target=run_analysis_in_background,
            args=(task_id, query, excel_data)
        ).start()
        
        # Retourner l'ID de la tâche
        return {
            "success": True,
            "task_id": task_id,
            "message": "Tâche d'analyse soumise avec succès"
        }
        
    except Exception as e:
        logger.error(f"Erreur lors de la soumission de la tâche LLM: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

def get_task_status(task_id):
    """Récupère le statut d'une tâche asynchrone"""
    if not task_id:
        return {
            "success": False,
            "error": "ID de tâche manquant"
        }
    
    if task_id not in TASKS:
        return {
            "success": False,
            "error": "Tâche non trouvée"
        }
    
    task = TASKS[task_id]
    
    # Nettoyer les tâches expirées
    cleanup_expired_tasks()
    
    # Construire la réponse en fonction du statut
    if task["status"] == "completed":
        return {
            "success": True,
            "status": task["status"],
            "result": task["result"]
        }
    elif task["status"] in ["failed", "error"]:
        return {
            "success": False,
            "status": task["status"],
            "error": task["error"]
        }
    else:
        # En attente ou en cours de traitement
        return {
            "success": True,
            "status": task["status"],
            "message": f"La tâche est {task['status']}"
        }

def cleanup_expired_tasks():
    """Nettoie les tâches expirées pour libérer la mémoire"""
    current_time = time.time()
    expired_tasks = [
        task_id for task_id, task in TASKS.items()
        if current_time - task.get("created_at", 0) > TASK_EXPIRY
    ]
    
    for task_id in expired_tasks:
        logger.info(f"Suppression de la tâche expirée {task_id}")
        del TASKS[task_id]



@app.post("/analyze_data")
async def analyze_data_endpoint(request: Request):
    try:
        data = await request.json()
        return await analyze_data(data)
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse directe: {str(e)}")
        return {"success": False, "error": str(e)}

@app.post("/analyze_data_llm")
async def analyze_data_llm_endpoint(request: Request):
    try:
        data = await request.json()
        return await analyze_data_llm(data)
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse avec LLM: {str(e)}")
        return {"success": False, "error": str(e)}

@app.post("/submit_llm_task")
async def submit_llm_task_endpoint(request: Request):
    try:
        data = await request.json()
        return await submit_llm_task(data)
    except Exception as e:
        logger.error(f"Erreur lors de la soumission de la tâche LLM: {str(e)}")
        return {"success": False, "error": str(e)}

@app.get("/task_status/{task_id}")
async def task_status_endpoint(task_id: str):
    return get_task_status(task_id)











@app.post("/update_excel_cell")
async def update_excel_cell(request: Request):
    """Mettre à jour une cellule dans Excel"""
    try:
        data = await request.json()
        workbook = data.get("workbook", "")
        sheet = data.get("sheet", "")
        cell = data.get("cell", "")
        value = data.get("value", "")
        
        if not all([workbook, sheet, cell]):
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Paramètres manquants"}
            )
        
        result = update_cell_value(workbook, sheet, cell, value)
        return result
    except Exception as e:
        logger.error(f"Erreur lors de la mise à jour de la cellule: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.post("/update_excel_range")
async def update_excel_range(request: Request):
    """Mettre à jour une plage de cellules dans Excel"""
    try:
        data = await request.json()
        workbook = data.get("workbook", "")
        sheet = data.get("sheet", "")
        start_cell = data.get("start_cell", "")
        values = data.get("values", [])
        
        if not all([workbook, sheet, start_cell]) or not isinstance(values, list):
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Paramètres manquants ou invalides"}
            )
        
        result = update_range_values(workbook, sheet, start_cell, values)
        return result
    except Exception as e:
        logger.error(f"Erreur lors de la mise à jour de la plage: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.post("/modify_excel_with_llm")
async def modify_excel_with_llm(request: Request):
    """Génère les commandes pour modifier Excel en utilisant une requête en langage naturel"""
    try:
        data = await request.json()
        query = data.get("query", "")
        excel_data = data.get("excel_data", {})
        
        if not query or not excel_data:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Paramètres manquants: query et excel_data sont requis"}
            )
        
        # Générer des commandes de modification avec le LLM - SEULEMENT générer, pas exécuter
        commands = generate_excel_modification(query, excel_data)
        
        # Retourner simplement les commandes générées, sans tenter d'exécuter quoi que ce soit
        return commands
            
    except Exception as e:
        logger.error(f"Erreur lors de la génération de commandes Excel via LLM: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

# Pour les tests locaux
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)