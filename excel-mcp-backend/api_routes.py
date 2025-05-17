from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional
import logging

# Importer les modules MCP pour Excel
from excel_processor_mcp import check_excel_status, get_open_workbooks, get_sheet_data,update_cell_value, update_range_values, execute_excel_formula, add_worksheet 
from embedding_handler import retrieve_relevant_rows, get_embedding_status

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/status")
async def get_status():
    """Vérifier l'état du service"""
    embedding_status = get_embedding_status()
    excel_status = check_excel_status()
    
    return {
        "status": "online",
        "embedding": embedding_status,
        "excel": excel_status
    }

# Routes pour l'intégration MCP (Microsoft COM for Excel)
@router.get("/mcp/status")
async def get_mcp_status():
    """Vérifier si Excel est accessible via MCP"""
    try:
        status = check_excel_status()
        return status
    except Exception as e:
        logger.error(f"Erreur de statut MCP: {str(e)}")
        return {
            "success": False,
            "excel_running": False,
            "error": str(e)
        }

@router.get("/mcp/workbooks")
async def get_workbooks():
    """Récupérer la liste des classeurs Excel ouverts"""
    try:
        workbooks = get_open_workbooks()
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

@router.post("/mcp/read_sheet")
async def read_sheet_data(
    workbook: str,
    sheet: str
):
    """Lire les données d'une feuille d'un classeur ouvert"""
    try:
        data = get_sheet_data(workbook, sheet)
        return {
            "success": True,
            "result": data
        }
    except Exception as e:
        logger.error(f"Erreur lors de la lecture de la feuille: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@router.post("/mcp/analyze")
async def analyze_with_embedding(
    workbook: str,
    sheet: str,
    query: str,
    top_k: int = 5
):
    """Analyser les données d'une feuille avec embedding"""
    try:
        # Récupérer les données de la feuille Excel
        excel_data = get_sheet_data(workbook, sheet)
        
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
    







@router.post("/mcp/update_cell")
async def update_cell(
    workbook: str,
    sheet: str,
    cell: str,
    value: Any
):
    """Mettre à jour la valeur d'une cellule dans Excel"""
    try:
        result = update_cell_value(workbook, sheet, cell, value)
        return result
    except Exception as e:
        logger.error(f"Erreur lors de la mise à jour de la cellule: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@router.post("/mcp/update_range")
async def update_range(
    workbook: str,
    sheet: str,
    start_cell: str,
    data: List[List[Any]]
):
    """Mettre à jour une plage de cellules dans Excel"""
    try:
        result = update_range_values(workbook, sheet, start_cell, data)
        return result
    except Exception as e:
        logger.error(f"Erreur lors de la mise à jour de la plage: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@router.post("/mcp/execute_formula")
async def execute_formula(
    workbook: str,
    sheet: str,
    cell: str,
    formula: str
):
    """Exécuter une formule Excel dans une cellule"""
    try:
        result = execute_excel_formula(workbook, sheet, cell, formula)
        return result
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution de la formule: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@router.post("/mcp/add_sheet")
async def add_sheet(
    workbook: str,
    sheet: str
):
    """Ajouter une nouvelle feuille dans un classeur Excel"""
    try:
        result = add_worksheet(workbook, sheet)
        return result
    except Exception as e:
        logger.error(f"Erreur lors de l'ajout de la feuille: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )