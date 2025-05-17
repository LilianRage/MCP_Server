import os
import xlwings as xw
from typing import Dict, List, Any, Optional
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_excel_instance():
    """
    Obtient une instance d'Excel via xlwings
    
    Returns:
        Instance Excel xlwings
    """
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

def check_excel_status() -> Dict[str, Any]:
    """
    Vérifie si Excel est accessible via xlwings
    
    Returns:
        Dictionnaire avec le statut d'Excel
    """
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

def get_open_workbooks() -> List[Dict[str, Any]]:
    """
    Récupère la liste des classeurs Excel ouverts
    
    Returns:
        Liste des classeurs avec leurs métadonnées
    """
    try:
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
        
        return workbooks
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des classeurs: {str(e)}")
        raise Exception(f"Erreur lors de la récupération des classeurs: {str(e)}")

def get_sheet_data(workbook_name: str, sheet_name: str) -> Dict[str, Any]:
    """
    Extrait les données d'une feuille Excel
    
    Args:
        workbook_name: Nom du classeur Excel
        sheet_name: Nom de la feuille
        
    Returns:
        Dictionnaire avec les en-têtes et les lignes
    """
    try:
        app = get_excel_instance()
        
        # Trouver le classeur par son nom
        wb = None
        for book in app.books:
            if book.name == workbook_name:
                wb = book
                break
        
        if wb is None:
            raise Exception(f"Classeur non trouvé: {workbook_name}")
        
        # Trouver la feuille par son nom
        ws = None
        for sheet in wb.sheets:
            if sheet.name == sheet_name:
                ws = sheet
                break
        
        if ws is None:
            raise Exception(f"Feuille non trouvée: {sheet_name}")
        
        # Déterminer la plage de données utilisée
        used_range = ws.used_range
        
        # Récupérer toutes les données en une seule fois (plus efficace)
        all_data = used_range.value
        
        # Si aucune donnée n'a été trouvée
        if not all_data:
            return {
                "headers": [],
                "rows": [],
                "row_count": 0,
                "column_count": 0
            }
        
        # Si les données sont un seul scalaire, le convertir en liste de liste
        if not isinstance(all_data, list):
            all_data = [[all_data]]
        elif all_data and not isinstance(all_data[0], list):
            all_data = [all_data]
        
        # Extrayez les en-têtes et les données
        headers = all_data[0] if all_data else []
        data = all_data[1:] if len(all_data) > 1 else []
        
        # Nettoyer les valeurs None dans les en-têtes
        headers = [header if header is not None else f"Column_{i+1}" for i, header in enumerate(headers)]
        
        # Convertir les données pour qu'elles soient sérialisables en JSON
        # (gérer les dates, les nombres complexes, etc.)
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
        
        return {
            "headers": headers,
            "rows": processed_data,
            "row_count": len(processed_data),
            "column_count": len(headers)
        }
        
    except Exception as e:
        logger.error(f"Erreur lors de la lecture des données Excel: {str(e)}")
        raise Exception(f"Erreur lors de la lecture des données Excel: {str(e)}")
    


    
    
def update_cell_value(workbook_name: str, sheet_name: str, cell_address: str, value: Any) -> Dict[str, Any]:
    """
    Met à jour la valeur d'une cellule spécifique dans une feuille Excel
    
    Args:
        workbook_name: Nom du classeur Excel
        sheet_name: Nom de la feuille
        cell_address: Adresse de la cellule (ex: "A1", "B5")
        value: Nouvelle valeur à placer dans la cellule
        
    Returns:
        Dictionnaire avec le statut de l'opération
    """
    try:
        app = get_excel_instance()
        
        # Trouver le classeur par son nom
        wb = None
        for book in app.books:
            if book.name == workbook_name:
                wb = book
                break
        
        if wb is None:
            raise Exception(f"Classeur non trouvé: {workbook_name}")
        
        # Trouver la feuille par son nom
        ws = None
        for sheet in wb.sheets:
            if sheet.name == sheet_name:
                ws = sheet
                break
        
        if ws is None:
            raise Exception(f"Feuille non trouvée: {sheet_name}")
        
        # Mettre à jour la cellule
        ws.range(cell_address).value = value
        
        # Enregistrer les modifications
        wb.save()
        
        return {
            "success": True,
            "message": f"Cellule {cell_address} mise à jour avec succès"
        }
        
    except Exception as e:
        logger.error(f"Erreur lors de la mise à jour de la cellule: {str(e)}")
        return {
            "success": False, 
            "error": str(e)
        }

def update_range_values(workbook_name: str, sheet_name: str, start_cell: str, data: List[List[Any]]) -> Dict[str, Any]:
    """
    Met à jour une plage de cellules avec de nouvelles valeurs
    
    Args:
        workbook_name: Nom du classeur Excel
        sheet_name: Nom de la feuille
        start_cell: Cellule de départ (ex: "A1")
        data: Liste de listes contenant les données à écrire
        
    Returns:
        Dictionnaire avec le statut de l'opération
    """
    try:
        app = get_excel_instance()
        
        # Trouver le classeur par son nom
        wb = None
        for book in app.books:
            if book.name == workbook_name:
                wb = book
                break
        
        if wb is None:
            raise Exception(f"Classeur non trouvé: {workbook_name}")
        
        # Trouver la feuille par son nom
        ws = None
        for sheet in wb.sheets:
            if sheet.name == sheet_name:
                ws = sheet
                break
        
        if ws is None:
            raise Exception(f"Feuille non trouvée: {sheet_name}")
        
        # Mettre à jour la plage de cellules
        ws.range(start_cell).value = data
        
        # Enregistrer les modifications
        wb.save()
        
        return {
            "success": True,
            "message": f"Plage de cellules mise à jour avec succès à partir de {start_cell}"
        }
        
    except Exception as e:
        logger.error(f"Erreur lors de la mise à jour de la plage de cellules: {str(e)}")
        return {
            "success": False, 
            "error": str(e)
        }

def execute_excel_formula(workbook_name: str, sheet_name: str, cell_address: str, formula: str) -> Dict[str, Any]:
    """
    Exécute une formule Excel dans une cellule spécifique
    
    Args:
        workbook_name: Nom du classeur Excel
        sheet_name: Nom de la feuille
        cell_address: Adresse de la cellule (ex: "A1", "B5")
        formula: Formule Excel (doit commencer par "=")
        
    Returns:
        Dictionnaire avec le statut de l'opération et le résultat
    """
    try:
        app = get_excel_instance()
        
        # Trouver le classeur par son nom
        wb = None
        for book in app.books:
            if book.name == workbook_name:
                wb = book
                break
        
        if wb is None:
            raise Exception(f"Classeur non trouvé: {workbook_name}")
        
        # Trouver la feuille par son nom
        ws = None
        for sheet in wb.sheets:
            if sheet.name == sheet_name:
                ws = sheet
                break
        
        if ws is None:
            raise Exception(f"Feuille non trouvée: {sheet_name}")
        
        # S'assurer que la formule commence par "="
        if not formula.startswith("="):
            formula = "=" + formula
            
        # Appliquer la formule
        ws.range(cell_address).formula = formula
        
        # Récupérer le résultat calculé
        result = ws.range(cell_address).value
        
        # Enregistrer les modifications
        wb.save()
        
        return {
            "success": True,
            "formula": formula,
            "result": result,
            "message": f"Formule appliquée avec succès à la cellule {cell_address}"
        }
        
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution de la formule: {str(e)}")
        return {
            "success": False, 
            "error": str(e)
        }

def add_worksheet(workbook_name: str, sheet_name: str) -> Dict[str, Any]:
    """
    Ajoute une nouvelle feuille au classeur Excel
    
    Args:
        workbook_name: Nom du classeur Excel
        sheet_name: Nom de la nouvelle feuille
        
    Returns:
        Dictionnaire avec le statut de l'opération
    """
    try:
        app = get_excel_instance()
        
        # Trouver le classeur par son nom
        wb = None
        for book in app.books:
            if book.name == workbook_name:
                wb = book
                break
        
        if wb is None:
            raise Exception(f"Classeur non trouvé: {workbook_name}")
        
        # Vérifier si une feuille avec ce nom existe déjà
        sheet_exists = False
        for sheet in wb.sheets:
            if sheet.name.lower() == sheet_name.lower():
                sheet_exists = True
                break
        
        if sheet_exists:
            return {
                "success": False,
                "error": f"Une feuille nommée '{sheet_name}' existe déjà"
            }
        
        # Ajouter une nouvelle feuille
        new_sheet = wb.sheets.add(name=sheet_name)
        
        # Enregistrer les modifications
        wb.save()
        
        return {
            "success": True,
            "message": f"Feuille '{sheet_name}' ajoutée avec succès"
        }
        
    except Exception as e:
        logger.error(f"Erreur lors de l'ajout de la feuille: {str(e)}")
        return {
            "success": False, 
            "error": str(e)
        }