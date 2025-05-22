import os
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import xlwings as xw
import requests
import json
import time
from excel_processor_mcp import update_cell_value, update_range_values, execute_excel_formula, add_worksheet


# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Excel Local Bridge Agent")

# Configuration CORS pour permettre les requêtes du frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Remplacer par votre domaine en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# URL de votre service RunPod - à modifier selon votre déploiement
RUNPOD_API = "https://7yg89rdzg30tkn-8001.proxy.runpod.net"

# Cache des tâches pour éviter les requêtes inutiles
TASK_CACHE = {}

@app.get("/status")
async def get_status():
    """Vérifier l'état du service local et d'Excel"""
    excel_status = check_excel_status()
    
    # Vérifier également le statut du service RunPod
    try:
        runpod_status = requests.get(f"{RUNPOD_API}/status", timeout=5).json()
    except Exception as e:
        logger.error(f"Erreur lors de la vérification du service RunPod: {str(e)}")
        runpod_status = {"status": "offline", "error": str(e)}
    
    return {
        "status": "online",
        "excel": excel_status,
        "runpod": runpod_status
    }

def check_excel_status():
    """Vérifie si Excel est accessible via xlwings"""
    try:
        # Tenter de récupérer l'application Excel
        app = get_excel_instance()
        return {
            "success": True,
            "excel_running": True,
            "version": app.version
        }
    except Exception as e:
        logger.error(f"Erreur lors de la vérification d'Excel: {str(e)}")
        return {
            "success": False,
            "excel_running": False,
            "error": str(e)
        }

def get_excel_instance():
    """Obtient une instance d'Excel via xlwings"""
    try:
        logger.info("Tentative de connexion à une instance Excel existante...")
        # xlwings utilise une approche différente pour se connecter à Excel
        app = xw.apps.active
        if app is None:
            # Si aucune instance n'est active, en créer une nouvelle
            raise Exception("Pas d'instance Excel active")
        return app
    except Exception as e:
        logger.info(f"Pas d'instance Excel active trouvée, création d'une nouvelle instance: {str(e)}")
        # Si Excel n'est pas déjà ouvert, on peut l'ouvrir
        app = xw.App(visible=True)
        return app

@app.get("/workbooks")
async def get_workbooks():
    """Récupérer la liste des classeurs Excel ouverts"""
    try:
        # Obtenir une instance Excel
        app = get_excel_instance()
        workbooks = []
        
        for wb in app.books:
            sheets = []
            
            for sheet in wb.sheets:
                sheets.append(sheet.name)
            
            workbooks.append({
                "name": wb.name,
                "path": wb.fullname,
                "sheets": sheets
            })
        
        return {
            "success": True,
            "workbooks": workbooks
        }
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des classeurs: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.post("/read_sheet")
async def read_sheet_data(request: Request):
    """Lire les données d'une feuille d'un classeur ouvert"""
    try:
        data = await request.json()
        workbook = data.get("workbook", "")
        sheet = data.get("sheet", "")
        
        if not workbook or not sheet:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Paramètres manquants: workbook et sheet sont requis"}
            )
            
        # Obtenir une instance Excel
        app = get_excel_instance()
        
        # Trouver le classeur par son nom
        wb = None
        for book in app.books:
            if book.name == workbook:
                wb = book
                break
        
        if wb is None:
            return JSONResponse(
                status_code=404,
                content={"success": False, "error": f"Classeur non trouvé: {workbook}"}
            )
        
        # Trouver la feuille par son nom
        ws = None
        for s in wb.sheets:
            if s.name == sheet:
                ws = s
                break
        
        if ws is None:
            return JSONResponse(
                status_code=404,
                content={"success": False, "error": f"Feuille non trouvée: {sheet}"}
            )
        
        # Déterminer la plage de données utilisée
        used_range = ws.used_range
        
        # Récupérer toutes les données en une seule fois (plus efficace)
        all_data = used_range.value
        
        # Si aucune donnée n'a été trouvée
        if not all_data:
            return {
                "success": True,
                "result": {
                    "headers": [],
                    "rows": [],
                    "row_count": 0,
                    "column_count": 0
                }
            }
        
        # Si les données sont un seul scalaire, le convertir en liste de liste
        if not isinstance(all_data, list):
            all_data = [[all_data]]
        elif all_data and not isinstance(all_data[0], list):
            all_data = [all_data]
        
        # Extraire les en-têtes et les données
        headers = all_data[0] if all_data else []
        data = all_data[1:] if len(all_data) > 1 else []
        
        # Nettoyer les valeurs None dans les en-têtes
        headers = [header if header is not None else f"Column_{i+1}" for i, header in enumerate(headers)]
        
        # Convertir les données pour qu'elles soient sérialisables en JSON
        processed_data = []
        for row in data:
            processed_row = []
            for cell in row:
                if isinstance(cell, (int, float, str, bool, type(None))):
                    processed_row.append(cell)
                else:
                    # Convertir d'autres types en chaînes
                    processed_row.append(str(cell))
            processed_data.append(processed_row)
        
        excel_data = {
            "headers": headers,
            "rows": processed_data,
            "row_count": len(processed_data),
            "column_count": len(headers)
        }
        
        return {
            "success": True,
            "result": excel_data
        }
        
    except Exception as e:
        logger.error(f"Erreur lors de la lecture des données Excel: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.post("/analyze_sheet")
async def analyze_sheet(request: Request):
    """Analyser une feuille Excel avec le service RunPod"""
    try:
        data = await request.json()
        workbook = data.get("workbook", "")
        sheet = data.get("sheet", "")
        query = data.get("query", "")
        top_k = data.get("top_k", 5)
        
        if not workbook or not sheet or not query:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Paramètres manquants: workbook, sheet et query sont requis"}
            )
        
        logger.info(f"Requête d'analyse reçue: workbook={workbook}, sheet={sheet}, query={query}, top_k={top_k}")
        
        # Lire les données de la feuille Excel
        excel_data = await get_sheet_data(workbook, sheet)
        
        if not excel_data.get("success", False):
            return excel_data
        
        # Envoyer les données au service RunPod pour analyse
        try:
            runpod_response = requests.post(
                f"{RUNPOD_API}/analyze_data",
                json={
                    "excel_data": excel_data["result"],
                    "query": query,
                    "top_k": top_k
                },
                timeout=30  # Timeout de 30 secondes
            )
            
            if runpod_response.status_code != 200:
                logger.error(f"Erreur RunPod: {runpod_response.text}")
                return JSONResponse(
                    status_code=runpod_response.status_code,
                    content={"success": False, "error": f"Erreur du service RunPod: {runpod_response.text}"}
                )
                
            # Retourner les résultats
            return runpod_response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Erreur de communication avec RunPod: {str(e)}")
            return JSONResponse(
                status_code=503,
                content={"success": False, "error": f"Erreur de communication avec RunPod: {str(e)}"}
            )
            
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.post("/submit_llm_analysis")
async def submit_llm_analysis(request: Request):
    """Soumettre une tâche d'analyse LLM asynchrone"""
    try:
        data = await request.json()
        workbook = data.get("workbook", "")
        sheet = data.get("sheet", "")
        query = data.get("query", "")
        
        if not workbook or not sheet or not query:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Paramètres manquants: workbook, sheet et query sont requis"}
            )
        
        logger.info(f"Soumission d'une tâche d'analyse LLM: workbook={workbook}, sheet={sheet}, query={query}")
        
        # Lire les données de la feuille Excel
        excel_data = await get_sheet_data(workbook, sheet)
        
        if not excel_data.get("success", False):
            return excel_data
        
        # Soumettre la tâche au service RunPod
        try:
            runpod_response = requests.post(
                f"{RUNPOD_API}/submit_llm_task",
                json={
                    "excel_data": excel_data["result"],
                    "query": query
                },
                timeout=10  # Timeout court pour la soumission
            )
            
            if runpod_response.status_code != 200:
                logger.error(f"Erreur lors de la soumission de la tâche LLM: {runpod_response.text}")
                return JSONResponse(
                    status_code=runpod_response.status_code,
                    content={"success": False, "error": f"Erreur du service RunPod: {runpod_response.text}"}
                )
            
            # Récupérer l'ID de tâche et le stocker dans le cache
            response_data = runpod_response.json()
            task_id = response_data.get("task_id")
            
            if task_id:
                TASK_CACHE[task_id] = {
                    "workbook": workbook,
                    "sheet": sheet,
                    "query": query,
                    "submitted_at": time.time()
                }
            
            # Retourner l'ID de tâche au client
            return {
                "success": True,
                "task_id": task_id,
                "message": "Tâche d'analyse soumise avec succès"
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Erreur de communication avec RunPod: {str(e)}")
            return JSONResponse(
                status_code=503,
                content={"success": False, "error": f"Erreur de communication avec RunPod: {str(e)}"}
            )
            
    except Exception as e:
        logger.error(f"Erreur lors de la soumission de la tâche: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.get("/llm_task_status/{task_id}")
async def get_llm_task_status(task_id: str):
    """Vérifier le statut d'une tâche d'analyse LLM"""
    try:
        logger.info(f"Vérification du statut de la tâche {task_id}")
        
        # Vérifier si la tâche est dans le cache
        if task_id not in TASK_CACHE:
            return JSONResponse(
                status_code=404,
                content={"success": False, "error": "Tâche non trouvée dans le cache local"}
            )
        
        # Vérifier le statut auprès du service RunPod
        try:
            runpod_response = requests.get(
                f"{RUNPOD_API}/task_status/{task_id}",
                timeout=10  # Timeout court pour la vérification
            )
            
            if runpod_response.status_code != 200:
                logger.error(f"Erreur lors de la vérification du statut: {runpod_response.text}")
                return JSONResponse(
                    status_code=runpod_response.status_code,
                    content={"success": False, "error": f"Erreur du service RunPod: {runpod_response.text}"}
                )
            
            # Retourner le statut au client
            return runpod_response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Erreur de communication avec RunPod: {str(e)}")
            return JSONResponse(
                status_code=503,
                content={"success": False, "error": f"Erreur de communication avec RunPod: {str(e)}"}
            )
            
    except Exception as e:
        logger.error(f"Erreur lors de la vérification du statut: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.post("/analyze_sheet_llm")
async def analyze_sheet_llm(request: Request):
    """
    Version compatible de l'API synchrone qui utilise l'API asynchrone en arrière-plan
    Cette méthode est maintenue pour la compatibilité avec l'ancienne API
    """
    try:
        data = await request.json()
        workbook = data.get("workbook", "")
        sheet = data.get("sheet", "")
        query = data.get("query", "")
        
        if not workbook or not sheet or not query:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Paramètres manquants: workbook, sheet et query sont requis"}
            )
        
        logger.info(f"Requête d'analyse LLM reçue via API synchrone: workbook={workbook}, sheet={sheet}, query={query}")
        
        # Créer une nouvelle requête pour submit_llm_analysis
        submit_request = Request(scope={"type": "http"})
        submit_data = {"workbook": workbook, "sheet": sheet, "query": query}
        
        # Un peu de magie pour créer une nouvelle requête avec les bonnes données
        async def mock_json():
            return submit_data
        submit_request.json = mock_json
        
        # Soumettre la tâche en mode asynchrone
        submit_response = await submit_llm_analysis(submit_request)
        
        if not submit_response.get("success", False):
            return submit_response
        
        task_id = submit_response.get("task_id")
        
        # Attendre que la tâche soit terminée (polling)
        max_retries = 60  # 5 minutes max (5s * 60)
        retry_delay = 5  # 5 secondes entre les tentatives
        
        for attempt in range(max_retries):
            logger.info(f"Vérification du statut de la tâche {task_id} (tentative {attempt+1}/{max_retries})")
            
            # Vérifier le statut de la tâche
            status_response = await get_llm_task_status(task_id)
            
            # Si le statut n'est pas un succès, retourner l'erreur
            if not status_response.get("success", False):
                return status_response
            
            # Vérifier si la tâche est terminée
            if status_response.get("status") == "completed":
                logger.info(f"Tâche {task_id} terminée avec succès")
                return status_response.get("result", {})
            
            # Si la tâche a échoué, retourner l'erreur
            if status_response.get("status") in ["failed", "error"]:
                logger.error(f"Tâche {task_id} échouée: {status_response.get('error')}")
                return {
                    "success": False,
                    "error": status_response.get("error", "Erreur inconnue")
                }
            
            # Attendre avant la prochaine tentative
            await asyncio.sleep(retry_delay)  # Version asynchrone du sleep
        
        # Si on arrive ici, c'est que le délai maximum est atteint
        return {
            "success": False,
            "error": f"Délai d'attente dépassé après {max_retries * retry_delay} secondes"
        }
        
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse LLM: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

async def get_sheet_data(workbook, sheet):
    """Fonction utilitaire pour lire les données d'une feuille"""
    try:
        # Obtenir une instance Excel
        app = get_excel_instance()
        
        # Trouver le classeur par son nom
        wb = None
        for book in app.books:
            if book.name == workbook:
                wb = book
                break
        
        if wb is None:
            return {
                "success": False,
                "error": f"Classeur non trouvé: {workbook}"
            }
        
        # Trouver la feuille par son nom
        ws = None
        for s in wb.sheets:
            if s.name == sheet:
                ws = s
                break
        
        if ws is None:
            return {
                "success": False, 
                "error": f"Feuille non trouvée: {sheet}"
            }
        
        # Déterminer la plage de données utilisée
        used_range = ws.used_range
        
        # Récupérer toutes les données en une seule fois
        all_data = used_range.value
        
        # Si aucune donnée n'a été trouvée
        if not all_data:
            return {
                "success": True,
                "result": {
                    "headers": [],
                    "rows": [],
                    "row_count": 0,
                    "column_count": 0
                }
            }
        
        # Si les données sont un seul scalaire, le convertir en liste de liste
        if not isinstance(all_data, list):
            all_data = [[all_data]]
        elif all_data and not isinstance(all_data[0], list):
            all_data = [all_data]
        
        # Extraire les en-têtes et les données
        headers = all_data[0] if all_data else []
        data_rows = all_data[1:] if len(all_data) > 1 else []
        
        # Nettoyer les valeurs None dans les en-têtes
        headers = [header if header is not None else f"Column_{i+1}" for i, header in enumerate(headers)]
        
        # Convertir les données pour qu'elles soient sérialisables en JSON
        processed_data = []
        for row in data_rows:
            processed_row = []
            for cell in row:
                if isinstance(cell, (int, float, str, bool, type(None))):
                    processed_row.append(cell)
                else:
                    # Convertir d'autres types en chaînes
                    processed_row.append(str(cell))
            processed_data.append(processed_row)
        
        excel_data = {
            "headers": headers,
            "rows": processed_data,
            "row_count": len(processed_data),
            "column_count": len(headers)
        }
        
        return {
            "success": True,
            "result": excel_data
        }
    
    except Exception as e:
        logger.error(f"Erreur lors de la lecture des données Excel: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }




@app.post("/update_cell")
async def update_cell_endpoint(request: Request):
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

@app.post("/update_range")
async def update_range_endpoint(request: Request):
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

@app.post("/execute_formula")
async def execute_formula_endpoint(request: Request):
    """Exécuter une formule Excel dans une cellule"""
    try:
        data = await request.json()
        workbook = data.get("workbook", "")
        sheet = data.get("sheet", "")
        cell = data.get("cell", "")
        formula = data.get("formula", "")
        
        if not all([workbook, sheet, cell, formula]):
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Paramètres manquants"}
            )
        
        result = execute_excel_formula(workbook, sheet, cell, formula)
        return result
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution de la formule: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.post("/add_sheet")
async def add_sheet_endpoint(request: Request):
    """Ajouter une nouvelle feuille dans un classeur Excel"""
    try:
        data = await request.json()
        workbook = data.get("workbook", "")
        sheet = data.get("sheet", "")
        
        if not all([workbook, sheet]):
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Paramètres manquants"}
            )
        
        result = add_worksheet(workbook, sheet)
        return result
    except Exception as e:
        logger.error(f"Erreur lors de l'ajout de la feuille: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.post("/modify_excel_with_llm")
async def modify_excel_with_llm_endpoint(request: Request):
    """Modifie Excel en utilisant une requête en langage naturel"""
    try:
        data = await request.json()
        workbook = data.get("workbook", "")
        sheet = data.get("sheet", "")
        query = data.get("query", "")
        
        if not all([workbook, sheet, query]):
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Paramètres manquants"}
            )
        
        # Lire les données Excel pour les envoyer au service RunPod
        logger.info(f"Lecture des données Excel: workbook={workbook}, sheet={sheet}")
        excel_data_response = await get_sheet_data(workbook, sheet)
        
        if not excel_data_response.get("success", False):
            logger.error(f"La lecture de données a échoué: {excel_data_response.get('error')}")
            return excel_data_response
            
        logger.info("La lecture de données a fonctionné")
        excel_data = excel_data_response["result"]
        
        # Préparer le payload au format handler RunPod
        handler_payload = {
            "input": {
                "command": "modify_excel_with_llm",  # Cette commande doit être ajoutée au handler côté RunPod
                "params": {
                    "query": query,
                    "excel_data": excel_data
                }
            }
        }
        
        # URL du handler
        handler_url = f"{RUNPOD_API}/"
        
        logger.info(f"Envoi à RunPod (format handler): command=modify_excel_with_llm, query={query}")
        
        # Envoyer la requête au service RunPod
        try:
            runpod_response = requests.post(
                handler_url,
                json=handler_payload,
                timeout=30
            )
            
            logger.info(f"Réponse RunPod: statut={runpod_response.status_code}")
            
            if runpod_response.status_code != 200:
                logger.error(f"Erreur RunPod: {runpod_response.text}")
                return JSONResponse(
                    status_code=runpod_response.status_code,
                    content={"success": False, "error": f"Erreur du service RunPod: {runpod_response.text}"}
                )
            
            # Récupérer les commandes générées (encapsulées dans 'output' avec le format handler)
            response_json = runpod_response.json()
            response_data = response_json.get("output", response_json)
            
            logger.info(f"Données de réponse: {response_data}")
            
            if not response_data.get("success", False):
                return response_data
                
            # Exécuter les commandes LOCALEMENT avec xlwings
            command_data = response_data.get("command", {})
            command_type = command_data.get("command", "")
            
            logger.info(f"Commande à exécuter: type={command_type}, data={command_data}")
            
            if command_type == "update_cell":
                result = update_cell_value(
                    workbook, 
                    sheet, 
                    command_data.get("cell", ""), 
                    command_data.get("value", "")
                )
            elif command_type == "execute_formula":
                result = execute_excel_formula(
                    workbook, 
                    sheet, 
                    command_data.get("cell", ""), 
                    command_data.get("formula", "")
                )
            elif command_type == "update_range":
                result = update_range_values(
                    workbook, 
                    sheet, 
                    command_data.get("start_cell", ""), 
                    command_data.get("values", [])
                )
            else:
                return {
                    "success": False,
                    "error": f"Type de commande inconnu: {command_type}"
                }
            
            # Ajouter les informations de commande au résultat
            result["command_generated"] = command_data
            result["query"] = query
            
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Erreur de communication avec RunPod: {str(e)}")
            return JSONResponse(
                status_code=503,
                content={"success": False, "error": f"Erreur de communication avec RunPod: {str(e)}"}
            )
            
    except Exception as e:
        logger.error(f"Erreur lors de la modification Excel via LLM: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

# Pour les tests locaux
if __name__ == "__main__":
    # Importer asyncio uniquement ici car il est utilisé dans analyze_sheet_llm
    import asyncio
    
    port = int(os.environ.get("PORT", 8001))
    logger.info(f"Démarrage du service d'agent local sur le port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)