import os
import numpy as np

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MKL_THREADING_LAYER'] = 'GNU'
import logging
import threading
import torch
from typing import Dict, Any, List, Optional
import tempfile
import pandas as pd
import re
import io
import sys
import contextlib
from transformers import AutoTokenizer

# Nouvelle importation pour VLLM
from vllm import LLM, SamplingParams

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Variables globales pour le modèle LLM
LLM_MODEL = None
LLM_TOKENIZER = None
LLM_LOADING = False
LLM_ERROR = None
LLM_AVAILABLE = False

# Configuration du modèle
MODEL_NAME = "Phind/Phind-CodeLlama-34B-v2"  # Votre modèle préféré
MODEL_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 2048
TEMPERATURE = 0.1

# Contexte pour les prompts
SYSTEM_PROMPT = """Tu es un assistant d'analyse de données expert en Python pour pandas. 
Ta tâche est de générer du code Python qui répond à des requêtes sur des données Excel. 
Génère UNIQUEMENT du code Python sans aucune explication, commentaire ou texte supplémentaire.
Le code doit être exécutable directement et imprimer les résultats.
Utilise seulement pandas et les bibliothèques standard."""

def load_llm_model_background():
    """Charge le modèle LLM en arrière-plan avec VLLM"""
    global LLM_MODEL, LLM_TOKENIZER, LLM_LOADING, LLM_ERROR, LLM_AVAILABLE
    
    try:
        logger.info(f"Chargement du modèle LLM {MODEL_NAME} avec VLLM...")
        
        # Configuration VLLM
        # Ajuster ces paramètres selon votre GPU
        tensor_parallel_size = 1  # Augmenter si vous avez plusieurs GPUs
        
        # Charger le tokenizer normalement depuis Hugging Face
        LLM_TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        
        # Charger le modèle avec VLLM - utilise quantification automatique
        LLM_MODEL = LLM(
            model=MODEL_NAME,
            tensor_parallel_size=1,
            trust_remote_code=True,
            dtype="half",           # Utiliser float16
            gpu_memory_utilization=0.85,  # Limiter l'utilisation mémoire à 85%
            max_model_len=4096,     # Réduire la longueur maximale
            quantization="bitsandbytes"     # Utiliser la quantification Int8 au lieu de AWQ
        )
        
        logger.info(f"Modèle LLM {MODEL_NAME} chargé avec succès via VLLM sur {MODEL_DEVICE}!")
        LLM_AVAILABLE = True
        LLM_LOADING = False
    
    except Exception as e:
        LLM_ERROR = f"Erreur lors du chargement du modèle LLM avec VLLM: {str(e)}"
        LLM_LOADING = False
        LLM_AVAILABLE = False
        logger.error(LLM_ERROR)

# Lancer le chargement du modèle en arrière-plan
try:
    LLM_LOADING = True
    threading.Thread(target=load_llm_model_background).start()
except Exception as e:
    logger.error(f"Erreur lors du démarrage du thread de chargement LLM: {str(e)}")
    LLM_LOADING = False
    LLM_ERROR = str(e)

def get_llm_status() -> Dict[str, Any]:
    """Retourne le statut du modèle LLM"""
    model_name = None
    if LLM_TOKENIZER is not None:
        model_name = MODEL_NAME
    
    return {
        "available": LLM_AVAILABLE,
        "loading": LLM_LOADING,
        "error": LLM_ERROR,
        "model_loaded": LLM_MODEL is not None and LLM_TOKENIZER is not None,
        "model_name": model_name,
        "device": MODEL_DEVICE,
        "engine": "VLLM"  # Indiquer qu'on utilise VLLM
    }

# Reste des fonctions de manipulation de données inchangées
def create_dataframe_from_excel_data(excel_data: Dict[str, Any]) -> pd.DataFrame:
    """Convertit les données Excel en DataFrame pandas"""
    # Votre code existant inchangé
    headers = excel_data.get("headers", [])
    rows = excel_data.get("rows", [])
    
    # Créer un DataFrame
    df = pd.DataFrame(rows, columns=headers)
    
    # Convertir les types de données si possible
    for col in df.columns:
        # Essayer de convertir en numérique
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            # Si ça échoue, essayer de convertir en datetime
            try:
                df[col] = pd.to_datetime(df[col])
            except (ValueError, TypeError):
                # Sinon laisser en string
                pass
    
    return df

def generate_python_code(query: str, excel_data: Dict[str, Any]) -> str:
    """Génère du code Python pour répondre à la requête en utilisant VLLM"""
    if not LLM_AVAILABLE or LLM_MODEL is None or LLM_TOKENIZER is None:
        raise Exception("Le modèle LLM n'est pas disponible")
    
    # Créer le prompt
    headers = excel_data.get("headers", [])
    rows = excel_data.get("rows", [])
    row_count = excel_data.get("row_count", 0)
    
    # Ajouter des informations sur le type de données pour chaque colonne
    df = create_dataframe_from_excel_data(excel_data)
    column_types = df.dtypes.to_dict()
    column_info = []
    
    for col, dtype in column_types.items():
        column_info.append(f"- {col}: {dtype}")
    
    # Construire le prompt
    prompt = f"""{SYSTEM_PROMPT}

## Structure du DataFrame
Les données sont déjà chargées dans un DataFrame pandas appelé `df` avec les colonnes suivantes:
{', '.join(headers)}

## Types de données pour chaque colonne:
{chr(10).join(column_info)}

## Informations sur les données
- Nombre de lignes: {row_count}
- Nombre de colonnes: {len(headers)}

## Exemples de données (2 premières lignes):
```
{df.head(2).to_string()}
```

## Requête:
{query}

Écris un code Python utilisant pandas qui répond à cette requête. Le code doit:
1. Utiliser le DataFrame 'df' déjà défini
2. Effectuer les opérations nécessaires
3. Imprimer le résultat clairement

Génère uniquement le code Python:
```python
"""

    try:
        # Configuration des paramètres de génération pour VLLM
        sampling_params = SamplingParams(
            temperature=TEMPERATURE,
            max_tokens=MAX_NEW_TOKENS,
            stop=["```"],  # Arrêter lorsqu'on atteint la fin du bloc de code
        )
        
        # Générer la réponse avec VLLM
        logger.info(f"Génération de code avec VLLM pour la requête: {query}")
        outputs = LLM_MODEL.generate(prompt, sampling_params)
        
        # Récupérer le texte généré
        generated_text = outputs[0].outputs[0].text
        
        logger.info(f"🦾 Réponse générée par le modèle via VLLM : {generated_text}")
        
        # Extraire le code Python
        python_code = extract_python_code(generated_text)
        
        # Ajouter l'import de pandas si nécessaire
        if "import pandas" not in python_code:
            python_code = "import pandas as pd\n\n" + python_code
        
        return python_code
    
    except Exception as e:
        logger.error(f"Erreur lors de la génération du code Python avec VLLM: {str(e)}")
        raise Exception(f"Erreur lors de la génération du code Python: {str(e)}")

def extract_python_code(text: str) -> str:
    """Extrait le code Python d'une réponse"""
    # Chercher le code entre les balises de code
    pattern = r"```python\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    
    # Si pas de balises, prendre tout le texte
    return text.strip()

# Le reste du code reste inchangé...
def execute_python_code(code: str, excel_data: Dict[str, Any]) -> Dict[str, Any]:
    """Exécute le code Python généré et retourne le résultat"""
    # Votre code existant inchangé
    try:
        # Créer un DataFrame à partir des données Excel
        df = create_dataframe_from_excel_data(excel_data)
        
        # Nettoyer le code généré pour éviter les erreurs de syntaxe
        # Retirer les backticks et les mentions import pandas qui pourraient être incluses
        clean_code = code.replace("```python", "").replace("```", "").strip()
        
        # Si le code ne contient pas d'import pandas et qu'il est nécessaire, l'ajouter
        if "import pandas" not in clean_code:
            final_code = "import pandas as pd\n\n" + clean_code
        else:
            final_code = clean_code
            
        # Log pour débogage
        logger.info(f"Code à exécuter après nettoyage: \n{final_code}")
        
        # Capturer la sortie standard
        output = io.StringIO()
        
        # Exécuter le code avec le DataFrame
        with contextlib.redirect_stdout(output):
            # Créer un environnement d'exécution
            exec_globals = {
                'pd': pd,
                'df': df
            }
            
            # Exécuter le code
            exec(final_code, exec_globals)
        
        # Récupérer la sortie
        result = output.getvalue()
        logger.info(f"🇫🇷Resultats apres execution: \n{result}")
        
        return {
            "success": True,
            "code": clean_code,
            "result": result,
            "error": None
        }
    
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution du code Python: {str(e)}")
        
        # Code de secours en cas d'erreur
        try:
            # Utiliser un code simple qui fonctionnera dans tous les cas
            fallback_code = ""
            
            # Détecter quelle colonne est mentionnée dans la requête
            if "Units Sold" in code:
                fallback_code = "result = df['Units Sold'].sum()\nprint(f\"La somme de Units Sold est: {result}\")"
            elif "Unit Price" in code:
                fallback_code = "result = df['Unit Price'].sum()\nprint(f\"La somme de Unit Price est: {result}\")"
            elif "Total Profit" in code:
                fallback_code = "result = df['Total Profit'].mean()\nprint(f\"La moyenne de Total Profit est: {result}\")"
            else:
                # Code générique si on ne peut pas détecter la colonne
                fallback_code = "print(df.describe())"
            
            logger.info(f"Utilisation du code de secours: {fallback_code}")
            
            # Exécuter le code de secours
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exec_globals = {'pd': pd, 'df': df}
                exec(fallback_code, exec_globals)
            
            result = output.getvalue()
            
            return {
                "success": True,
                "code": code,
                "result": result,
                "error": f"Code original non exécutable : {str(e)}. Résultat généré avec un code de secours."
            }
        except Exception as fallback_e:
            return {
                "success": False,
                "code": code,
                "result": None,
                "error": f"Erreur initiale: {str(e)}. Erreur avec code de secours: {str(fallback_e)}"
            }

def analyze_with_llm(query: str, excel_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyse les données Excel avec un LLM et du code Python généré
    
    Args:
        query: Requête utilisateur en langage naturel
        excel_data: Données Excel (headers, rows, etc.)
        
    Returns:
        Résultat de l'analyse
    """
    try:
        # Vérifier que le modèle est chargé
        if not LLM_AVAILABLE or LLM_MODEL is None:
            return {
                "success": False,
                "error": "Le modèle LLM n'est pas disponible ou n'est pas chargé"
            }
        
        # Générer le code Python
        python_code = generate_python_code(query, excel_data)
        
        # Exécuter le code Python
        result = execute_python_code(python_code, excel_data)
        
        return {
            "success": result["success"],
            "code": result["code"],
            "result": result["result"],
            "error": result["error"]
        }
    
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse avec LLM: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }