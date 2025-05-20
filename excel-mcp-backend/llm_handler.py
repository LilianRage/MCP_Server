import os
import numpy as np

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MKL_THREADING_LAYER'] = 'GNU'

#Storage
os.environ["TRANSFORMERS_CACHE"] = "/workspace/model_cache"
os.environ["HF_HOME"] = "/workspace/model_cache"
os.environ["VLLM_CACHE_DIR"] = "/workspace/model_cache/vllm"

import logging
import threading
import torch
from typing import Dict, Any, List, Optional, Tuple, Union
import tempfile
import pandas as pd
import re
import io
import sys
import contextlib
import base64
from transformers import AutoTokenizer
import json
# Nouvelle importation pour VLLM
from vllm import LLM, SamplingParams

# Importations pour les visualisations
import matplotlib
matplotlib.use('Agg')  # Utilisation du backend non-interactif
import matplotlib.pyplot as plt
import seaborn as sns

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
SYSTEM_PROMPT = """Tu es un assistant d'analyse de données expert en Python pour pandas et les visualisations. 
Ta tâche est de générer du code Python qui répond à des requêtes sur des données Excel.

DETECTION D'INTENTION:
1. Si la requête demande explicitement une visualisation (graphique, diagramme, tendance visuelle), ou pourrait bénéficier d'une représentation visuelle, tu DOIS générer une visualisation.
2. Si la requête est purement factuelle (somme, comptage simple, statistique unique), tu ne dois PAS générer de visualisation.

POUR LES VISUALISATIONS:
- Pour les tendances temporelles ou évolutions: utilise un graphique linéaire (plt.plot).
- Pour les comparaisons entre catégories: utilise un diagramme à barres (plt.bar).
- Pour les répartitions proportionnelles: utilise un diagramme circulaire (plt.pie).
- Utilise seaborn (sns) pour des visualisations plus avancées si nécessaire.
- Sauvegarde TOUJOURS l'image avec: plt.savefig('/tmp/visualization.png', dpi=100, bbox_inches='tight')
- IMPORTANT: Termine TOUJOURS par plt.close() pour libérer la mémoire.
- N'affiche JAMAIS le graphique avec plt.show().

Génère UNIQUEMENT du code Python sans aucune explication, commentaire ou texte supplémentaire.
Le code doit être exécutable directement et imprimer les résultats textuels comme avant.
Si tu génères une visualisation, tu dois aussi imprimer un résultat textuel pour assurer la compatibilité avec le système existant.

IMPORTANT: matplotlib et seaborn sont déjà importés, n'importe jamais ces bibliothèques.
"""

def check_model_cache():
    """Vérifie si le modèle est déjà en cache"""
    model_cache_path = "/workspace/model_cache/vllm/models/Phind/Phind-CodeLlama-34B-v2"
    if os.path.exists(model_cache_path):
        logger.info(f"✅ Modèle trouvé dans le cache: {model_cache_path}")
        return True
    else:
        logger.info(f"❌ Modèle non trouvé dans le cache, téléchargement nécessaire")
        return False

def load_llm_model_background():
    """Charge le modèle LLM en arrière-plan avec VLLM"""
    global LLM_MODEL, LLM_TOKENIZER, LLM_LOADING, LLM_ERROR, LLM_AVAILABLE
    
    try:
        # Vérifier si le modèle est déjà en cache
        model_cache_path = "/workspace/model_cache/vllm/models/Phind/Phind-CodeLlama-34B-v2"
        cache_exists = os.path.exists(model_cache_path)
        
        logger.info(f"Chargement du modèle LLM {MODEL_NAME} avec VLLM... (Cache {'existant' if cache_exists else 'inexistant'})")
        
        # Charger le tokenizer normalement depuis Hugging Face
        LLM_TOKENIZER = AutoTokenizer.from_pretrained(
            MODEL_NAME, 
            trust_remote_code=True,
            cache_dir="/workspace/model_cache"
        )
        
        # Charger le modèle avec VLLM - utilise quantification automatique
        LLM_MODEL = LLM(
            model=MODEL_NAME,
            tensor_parallel_size=1,
            trust_remote_code=True,
            dtype="half",
            gpu_memory_utilization=0.85,
            max_model_len=4096,
            quantization="bitsandbytes",
            download_dir="/workspace/model_cache",
        )
        
        logger.info(f"Modèle LLM {MODEL_NAME} chargé avec succès via VLLM sur {MODEL_DEVICE}!")
        LLM_AVAILABLE = True
        LLM_LOADING = False
    
    except Exception as e:
        LLM_ERROR = f"Erreur lors du chargement du modèle LLM avec VLLM: {str(e)}"
        LLM_LOADING = False
        LLM_AVAILABLE = False
        logger.error(LLM_ERROR)

try:
    # Ne lancer le chargement que si nécessaire
    if not check_model_cache():
        logger.info("Démarrage du chargement du modèle...")
        LLM_LOADING = True
        threading.Thread(target=load_llm_model_background).start()
    else:
        # Si le modèle est en cache, le charger directement en mémoire
        logger.info("Chargement du modèle depuis le cache...")
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
    headers = excel_data.get("headers", [])
    rows = excel_data.get("rows", [])
    
    logger.info(f"Création DataFrame: {len(rows)} lignes, {len(headers)} colonnes")
    
    # Créer un DataFrame
    df = pd.DataFrame(rows, columns=headers)
    
    # Convertir les types de données si possible
    for col in df.columns:
        # Essayer de convertir en numérique
        try:
            original_type = df[col].dtype
            df[col] = pd.to_numeric(df[col])
            if original_type != df[col].dtype:
                logger.info(f"Colonne '{col}' convertie en {df[col].dtype}")
        except (ValueError, TypeError):
            # Si ça échoue, essayer de convertir en datetime avec format='mixed'
            try:
                original_type = df[col].dtype
                df[col] = pd.to_datetime(df[col], format='mixed')
                if original_type != df[col].dtype:
                    logger.info(f"Colonne '{col}' convertie en datetime")
            except (ValueError, TypeError):
                # Sinon laisser en string
                pass
    
    logger.info(f"DataFrame créé avec succès. Types des colonnes importantes: "
                f"'Order Date': {df['Order Date'].dtype if 'Order Date' in df.columns else 'N/A'}, "
                f"'Units Sold': {df['Units Sold'].dtype if 'Units Sold' in df.columns else 'N/A'}")
    
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
4. Les colonnes de type datetime ont déjà été converties, NE PAS les reconvertir avec pd.to_datetime()
5. Utiliser directement les opérations de comparaison sur ces colonnes

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
    """Exécute le code Python généré et retourne le résultat avec éventuellement une visualisation"""
    visualization_path = '/tmp/visualization.png'
    has_visualization = False
    visualization_base64 = None
    
    # Supprimer toute visualisation précédente
    if os.path.exists(visualization_path):
        os.remove(visualization_path)
    
    try:
        logger.info("=== EXÉCUTION CODE PYTHON ===")
        
        # Créer un DataFrame à partir des données Excel
        df = create_dataframe_from_excel_data(excel_data)
        
        # Nettoyer le code généré pour éviter les erreurs de syntaxe
        clean_code = code.replace("```python", "").replace("```", "").strip()
        
        # Préparer les imports nécessaires
        imports = "import pandas as pd\n"
        
        # Vérifier si le code fait référence à des visualisations
        needs_visualization = any(x in clean_code for x in ['plt.', 'sns.', '.plot(', '.hist(', 'savefig'])
        
        if needs_visualization:
            logger.info("Visualisation détectée dans le code généré")
        
        # Assembler le code final
        if "import pandas" not in clean_code:
            final_code = imports + "\n" + clean_code
        else:
            final_code = clean_code
        
        logger.info(f"Code à exécuter:\n{final_code}")
        
        # Capturer la sortie standard
        output = io.StringIO()
        
        # Exécuter le code avec le DataFrame
        try:
            logger.info("Tentative d'exécution...")
            with contextlib.redirect_stdout(output):
                # Créer un environnement d'exécution avec matplotlib et seaborn
                exec_globals = {
                    'pd': pd, 
                    'df': df, 
                    'plt': plt, 
                    'sns': sns,
                    'np': np
                }
                exec(final_code, exec_globals)
            
            # Récupérer la sortie
            result = output.getvalue()
            logger.info(f"Exécution réussie! Résultat: {result}")
            
            # Vérifier si une visualisation a été générée
            if os.path.exists(visualization_path):
                logger.info("Visualisation détectée dans /tmp/visualization.png")
                has_visualization = True
                with open(visualization_path, 'rb') as img_file:
                    visualization_base64 = base64.b64encode(img_file.read()).decode('utf-8')
                    
                # Facultatif: supprimer le fichier après l'avoir lu
                os.remove(visualization_path)
            
            return {
                "success": True,
                "code": clean_code,
                "result": result,
                "has_visualization": has_visualization,
                "visualization": visualization_base64,
                "error": None
            }
        
        except Exception as exec_e:
            logger.error(f"Erreur d'exécution: {str(exec_e)}")
            
            # Si l'erreur est liée au format de date, réessayer avec format='mixed'
            if "time data" in str(exec_e) and "doesn't match format" in str(exec_e):
                logger.info("Détection erreur format de date. Correction avec format='mixed'...")
                
                # Modifier le code pour utiliser format='mixed'
                fixed_code = re.sub(
                    r"pd\.to_datetime\(([^)]+)\)",
                    r"pd.to_datetime(\1, format='mixed')",
                    final_code
                )
                
                logger.info(f"Code corrigé:\n{fixed_code}")
                
                # Réessayer avec le nouveau code
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    exec(fixed_code, exec_globals)
                
                # Récupérer la sortie
                result = output.getvalue()
                logger.info(f"Exécution avec format='mixed' réussie! Résultat: {result}")
                
                # Vérifier si une visualisation a été générée après correction
                if os.path.exists(visualization_path):
                    logger.info("Visualisation détectée après correction")
                    has_visualization = True
                    with open(visualization_path, 'rb') as img_file:
                        visualization_base64 = base64.b64encode(img_file.read()).decode('utf-8')
                        
                    # Facultatif: supprimer le fichier après l'avoir lu
                    os.remove(visualization_path)
                
                return {
                    "success": True,
                    "code": fixed_code,
                    "result": result,
                    "has_visualization": has_visualization,
                    "visualization": visualization_base64,
                    "error": None
                }
            else:
                # Pour les autres types d'erreurs
                return {
                    "success": False,
                    "code": final_code,
                    "result": f"L'exécution du code a échoué: {str(exec_e)}",
                    "has_visualization": False,
                    "visualization": None,
                    "error": str(exec_e)
                }
    
    except Exception as e:
        logger.error(f"Erreur générale: {str(e)}")
        return {
            "success": False,
            "code": code,
            "result": f"Erreur lors de l'exécution: {str(e)}",
            "has_visualization": False,
            "visualization": None,
            "error": str(e)
        }
    finally:
        # S'assurer que toutes les figures matplotlib sont fermées
        plt.close('all')
        logger.info("=== FIN EXÉCUTION CODE PYTHON ===")

def analyze_with_llm(query: str, excel_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyse les données Excel avec un LLM et du code Python généré
    Peut également générer des visualisations si approprié
    
    Args:
        query: Requête utilisateur en langage naturel
        excel_data: Données Excel (headers, rows, etc.)
        
    Returns:
        Résultat de l'analyse avec visualisation optionnelle
    """
    try:
        # Vérifier que le modèle est chargé
        if not LLM_AVAILABLE or LLM_MODEL is None:
            return {
                "success": False,
                "error": "Le modèle LLM n'est pas disponible ou n'est pas chargé"
            }
        
        # Détecter si la requête pourrait nécessiter une visualisation
        visualization_keywords = [
            "graphique", "graph", "plot", "visualise", "visualiser", "visualisation",
            "affiche", "montre", "représente", "tendance", "évolution", "historique",
            "comparer", "comparaison", "répartition", "distribution", "courbe", "histogramme",
            "camembert", "diagramme", "barre", "pie chart", "bar chart", "line chart"
        ]
        
        might_need_visualization = any(keyword in query.lower() for keyword in visualization_keywords)
        
        if might_need_visualization:
            logger.info(f"La requête '{query}' pourrait nécessiter une visualisation")
        
        # Générer le code Python
        python_code = generate_python_code(query, excel_data)
        
        # Exécuter le code Python
        result = execute_python_code(python_code, excel_data)
        
        # Construire la réponse en incluant les informations de visualisation
        response = {
            "success": result["success"],
            "code": result["code"],
            "result": result["result"],
            "error": result["error"],
            "has_visualization": result.get("has_visualization", False),
            "visualization": result.get("visualization")
        }
        
        return response
    
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse avec LLM: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "has_visualization": False,
            "visualization": None
        }
    
def generate_excel_modification(query: str, excel_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Génère des commandes pour modifier Excel basées sur une requête en langage naturel
    
    Args:
        query: Requête utilisateur en langage naturel (ex: "change le prix en B5 à 150")
        excel_data: Données Excel (headers, rows, etc.)
        
    Returns:
        Commandes de modification Excel
    """
    if not LLM_AVAILABLE or LLM_MODEL is None or LLM_TOKENIZER is None:
        raise Exception("Le modèle LLM n'est pas disponible")
    
    headers = excel_data.get("headers", [])
    rows = excel_data.get("rows", [])
    
    # Construire le prompt
    prompt = f"""Tu es un assistant qui génère des commandes pour modifier des fichiers Excel.
Basé sur la demande de l'utilisateur, génère un objet JSON qui contient les commandes nécessaires.

## Structure du fichier Excel
Les données ont les colonnes suivantes:
{', '.join(headers)}

## Exemples de données (2 premières lignes):
```{pd.DataFrame(rows[:2], columns=headers).to_string()}```
## Types de commandes disponibles:
1. Pour mettre à jour une cellule: {{"command": "update_cell", "cell": "A1", "value": 100}}
2. Pour exécuter une formule: {{"command": "execute_formula", "cell": "C5", "formula": "=SUM(C1:C4)"}}
3. Pour mettre à jour une plage: {{"command": "update_range", "start_cell": "A1", "values": [[1, 2], [3, 4]]}}

## Requête de l'utilisateur:
{query}

Réponds uniquement avec l'objet JSON approprié sans explications, commentaires, backticks ou formatage supplémentaire:
"""

    try:
        # Configuration des paramètres de génération pour VLLM
        sampling_params = SamplingParams(
            temperature=0.1,
            max_tokens=500,
        )
        
        # Générer la réponse avec VLLM
        logger.info(f"Génération de commandes Excel pour la requête: {query}")
        outputs = LLM_MODEL.generate(prompt, sampling_params)
        
        # Récupérer le texte généré
        generated_text = outputs[0].outputs[0].text
        
        logger.info(f"Commandes générées: {generated_text}")
        
        # Essayer d'extraire un JSON valide - VERSION CORRIGÉE
        try:
            # Nettoyer le texte pour s'assurer qu'il contient un JSON valide
            json_text = generated_text.strip()
            
            # Supprimer les backticks et les balises markdown
            json_text = re.sub(r'```(json)?|```', '', json_text, flags=re.IGNORECASE)
            
            # Suppression des espaces blancs au début et à la fin
            json_text = json_text.strip()
            
            logger.info(f"JSON nettoyé: {json_text}")
            
            # Analyser le JSON
            command = json.loads(json_text)
            
            return {
                "success": True,
                "command": command,
                "source_query": query
            }
        except json.JSONDecodeError as e:
            logger.error(f"Erreur de décodage JSON: {str(e)}")
            
            # Tentative de dernière chance: essayer d'extraire manuellement le JSON
            try:
                # Recherche de patterns avec regex pour extraire les valeurs
                cell_pattern = r'"cell":\s*"([A-Z0-9]+)"'
                value_pattern = r'"value":\s*"([^"]+)"'
                command_pattern = r'"command":\s*"([^"]+)"'
                
                cell_match = re.search(cell_pattern, json_text)
                value_match = re.search(value_pattern, json_text)
                command_match = re.search(command_pattern, json_text)
                
                if cell_match and value_match and command_match:
                    cell = cell_match.group(1)
                    value = value_match.group(1)
                    command_type = command_match.group(1)
                    
                    # Construire manuellement la commande
                    command = {
                        "command": command_type,
                        "cell": cell,
                        "value": value
                    }
                    
                    logger.info(f"JSON reconstruit manuellement: {command}")
                    
                    return {
                        "success": True,
                        "command": command,
                        "source_query": query
                    }
            except Exception as regex_error:
                logger.error(f"Tentative de reconstruction JSON échouée: {str(regex_error)}")
            
            # Si toutes les tentatives échouent
            return {
                "success": False,
                "error": f"Le modèle n'a pas généré de JSON valide: {str(e)}",
                "generated_text": generated_text
            }
        
    except Exception as e:
        logger.error(f"Erreur lors de la génération des commandes Excel: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }