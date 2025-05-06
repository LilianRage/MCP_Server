import os
import base64
import pandas as pd
import numpy as np
import xlwings as xw
from typing import Dict, List, Any, Optional
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Variables d'environnement
EXCEL_FILES_PATH = os.environ.get("EXCEL_FILES_PATH", "./excel_files")

def save_excel_file(filename: str, content_base64: str) -> str:
    """
    Sauvegarde un fichier Excel depuis du contenu base64
    
    Args:
        filename: Nom du fichier
        content_base64: Contenu du fichier encodé en base64
        
    Returns:
        Chemin complet du fichier sauvegardé
    """
    # Créer le répertoire s'il n'existe pas
    os.makedirs(EXCEL_FILES_PATH, exist_ok=True)
    
    # Générer un chemin de fichier unique si nécessaire
    base_name = os.path.basename(filename)
    file_path = os.path.join(EXCEL_FILES_PATH, base_name)
    
    # Sauvegarder le fichier
    try:
        with open(file_path, "wb") as f:
            f.write(base64.b64decode(content_base64))
        return file_path
    except Exception as e:
        raise Exception(f"Erreur lors de la sauvegarde du fichier: {str(e)}")

def get_excel_data(filepath: str, sheet_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Extrait les données d'un fichier Excel avec xlwings
    
    Args:
        filepath: Chemin du fichier Excel
        sheet_name: Nom de la feuille (optionnel)
        
    Returns:
        Dictionnaire avec les en-têtes et les lignes
    """
    try:
        # Vérifier si le fichier existe
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Fichier non trouvé: {filepath}")
        
        # Obtenir une nouvelle instance Excel
        app = xw.App(visible=False)
        
        try:
            # Ouvrir le classeur
            wb = app.books.open(filepath)
            
            # Sélectionner la feuille
            if sheet_name and sheet_name in [sheet.name for sheet in wb.sheets]:
                ws = wb.sheets[sheet_name]
            else:
                ws = wb.sheets[0]  # Première feuille par défaut
            
            # Récupérer les données utilisées
            used_range = ws.used_range
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
            
            return {
                "headers": headers,
                "rows": data,
                "row_count": len(data),
                "column_count": len(headers) if headers else 0
            }
        
        finally:
            # Fermer le classeur et quitter Excel
            wb.close()
            app.quit()
        
    except Exception as e:
        logger.error(f"Erreur lors de la lecture du fichier Excel: {str(e)}")
        raise Exception(f"Erreur lors de la lecture du fichier Excel: {str(e)}")

def create_excel_file(data: Dict[str, Any], filepath: Optional[str] = None) -> str:
    """
    Crée un fichier Excel à partir de données structurées avec xlwings
    
    Args:
        data: Dictionnaire avec headers et rows
        filepath: Chemin du fichier (optionnel)
        
    Returns:
        Chemin du fichier créé
    """
    try:
        # Créer le répertoire si nécessaire
        os.makedirs(EXCEL_FILES_PATH, exist_ok=True)
        
        # Générer un nom de fichier si non fourni
        if not filepath:
            filepath = os.path.join(EXCEL_FILES_PATH, f"export_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}.xlsx")
        elif not os.path.isabs(filepath):
            filepath = os.path.join(EXCEL_FILES_PATH, filepath)
        
        # Créer un DataFrame à partir des données
        headers = data.get("headers", [])
        rows = data.get("rows", [])
        
        df = pd.DataFrame(rows, columns=headers)
        
        # Utiliser xlwings pour créer le fichier Excel
        app = xw.App(visible=False)
        
        try:
            # Créer un nouveau classeur
            wb = app.books.add()
            
            # Sélectionner la première feuille
            ws = wb.sheets[0]
            
            # Écrire les en-têtes dans la première ligne
            for col, header in enumerate(headers):
                ws.cells(1, col + 1).value = header
            
            # Écrire les données
            if rows:
                ws.range(f"A2").options(index=False).value = df
            
            # Sauvegarder le fichier
            wb.save(filepath)
            
            return filepath
        
        finally:
            # Fermer le classeur et quitter Excel
            wb.close()
            app.quit()
        
    except Exception as e:
        logger.error(f"Erreur lors de la création du fichier Excel: {str(e)}")
        raise Exception(f"Erreur lors de la création du fichier Excel: {str(e)}")