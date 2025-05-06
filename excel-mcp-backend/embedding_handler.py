import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
import logging
import threading
import warnings
import torch

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Variables globales pour le modèle d'embedding
TEXT_MODEL = None
TOKENIZER = None
EMBEDDING_LOADING = False
EMBEDDING_ERROR = None
TRANSFORMERS_AVAILABLE = False

# Essayer d'importer les dépendances nécessaires
try:
    from transformers import AutoTokenizer, AutoModel
    from sklearn.metrics.pairwise import cosine_similarity
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    EMBEDDING_ERROR = "Transformers ou sklearn n'est pas installé."
    
    # Fonction de cosine similarity pour le mode dégradé
    def cosine_similarity(X, Y):
        return np.zeros((len(X), len(Y)))  # Retourner une matrice de zéros en mode dégradé

# Fonction pour calculer les embeddings avec les modèles Hugging Face
def mean_pooling(model_output, attention_mask):
    """
    Fonction d'agrégation des tokens pour obtenir un embedding de phrase
    """
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def load_embedding_model_background():
    """Charge le modèle d'embedding en arrière-plan"""
    global TEXT_MODEL, TOKENIZER, EMBEDDING_LOADING, EMBEDDING_ERROR
    
    try:
        logger.info("Chargement du modèle d'embedding...")
        
        # Utiliser un modèle léger adapté à un serveur serverless
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        
        # Déterminer le device (GPU si disponible)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Utilisation du device: {device}")
        
        # Charger le tokenizer et le modèle
        TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        TEXT_MODEL = AutoModel.from_pretrained(model_name).to(device)
        
        logger.info(f"Modèle d'embedding {model_name} chargé avec succès sur {device}!")
        EMBEDDING_LOADING = False
    
    except Exception as e:
        EMBEDDING_ERROR = f"Erreur lors du chargement du modèle d'embedding: {str(e)}"
        EMBEDDING_LOADING = False
        logger.error(EMBEDDING_ERROR)

# Lancer le chargement du modèle en arrière-plan si les dépendances sont disponibles
if TRANSFORMERS_AVAILABLE and not EMBEDDING_LOADING and TEXT_MODEL is None:
    try:
        EMBEDDING_LOADING = True
        threading.Thread(target=load_embedding_model_background).start()
    except Exception as e:
        logger.error(f"Erreur lors du démarrage du thread de chargement: {str(e)}")
        EMBEDDING_LOADING = False
        EMBEDDING_ERROR = str(e)

def get_embedding_status() -> Dict[str, Any]:
    """Retourne le statut du modèle d'embedding"""
    model_name = None
    if TOKENIZER is not None:
        model_name = TOKENIZER.name_or_path
    
    return {
        "available": TRANSFORMERS_AVAILABLE,
        "loading": EMBEDDING_LOADING,
        "error": EMBEDDING_ERROR,
        "model_loaded": TEXT_MODEL is not None and TOKENIZER is not None,
        "model_name": model_name
    }

def encode_text(texts: List[str]) -> np.ndarray:
    """
    Encode une liste de textes en vecteurs d'embedding
    """
    if TEXT_MODEL is None or TOKENIZER is None:
        # Mode dégradé - retourner des embeddings aléatoires
        warnings.warn("Modèle d'embedding non disponible, utilisation d'embeddings aléatoires")
        return np.random.rand(len(texts), 384)  # dimension standard pour MiniLM
    
    try:
        # Déterminer le device
        device = next(TEXT_MODEL.parameters()).device
        
        # Tokenization des textes
        encoded_input = TOKENIZER(texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
        
        # Déplacer les entrées vers le device approprié
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        
        # Compute embeddings
        with torch.no_grad():
            model_output = TEXT_MODEL(**encoded_input)
        
        # Agréger les tokens pour obtenir des embeddings de phrase
        embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        
        # Normaliser les embeddings
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # Convertir en numpy pour la compatibilité
        return embeddings.cpu().numpy()
    
    except Exception as e:
        logger.error(f"Erreur lors du calcul des embeddings: {str(e)}")
        # En cas d'erreur, retourner des embeddings aléatoires
        return np.random.rand(len(texts), 384)

def create_text_from_row(row: List[Any], headers: List[str]) -> str:
    """Crée une représentation textuelle d'une ligne d'Excel"""
    text_parts = []
    for i, value in enumerate(row):
        if i < len(headers):  # Vérification pour éviter les erreurs d'index
            # Ignorer les valeurs None ou vides
            if value is not None and value != "":
                text_parts.append(f"{headers[i]}: {value}")
    
    return ", ".join(text_parts)

def embed_excel_data(excel_data: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
    """
    Crée des vecteurs d'embedding pour chaque ligne du DataFrame Excel
    """
    headers = excel_data.get("headers", [])
    rows = excel_data.get("rows", [])
    
    # Créer des représentations textuelles de chaque ligne
    row_texts = [create_text_from_row(row, headers) for row in rows]
    
    # Encoder les textes
    embeddings = encode_text(row_texts)
    
    return embeddings, row_texts

def retrieve_relevant_rows(query: str, excel_data: Dict[str, Any], top_k: int = 5) -> Dict[str, Any]:
    """
    Récupère les lignes les plus pertinentes par rapport à la requête
    """
    # Si pas assez de données, retourner toutes les données
    if excel_data.get("row_count", 0) <= top_k:
        return excel_data
    
    try:
        # Créer les embeddings des lignes
        row_embeddings, row_texts = embed_excel_data(excel_data)
        
        if TEXT_MODEL is None:
            # En mode dégradé, sélectionner des lignes aléatoires
            row_count = excel_data.get("row_count", 0)
            top_indices = np.random.choice(row_count, min(top_k, row_count), replace=False)
            similarities = {i: 0.5 for i in top_indices}  # Similarités arbitraires
        else:
            # Encoder la requête
            query_embedding = encode_text([query])[0]
            
            # Calculer les similarités cosinus
            similarities = cosine_similarity([query_embedding], row_embeddings)[0]
            
            # Trier les indices par similarité décroissante
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
        # Créer un nouveau dictionnaire de données avec seulement les lignes pertinentes
        filtered_data = {
            "headers": excel_data.get("headers", []),
            "rows": [excel_data.get("rows", [])[i] for i in top_indices],
            "row_count": len(top_indices),
            "column_count": excel_data.get("column_count", 0),
            "similarity_scores": {int(i): float(similarities[i]) for i in top_indices},
            "original_indices": top_indices.tolist()
        }
        
        return filtered_data
    
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des lignes pertinentes: {str(e)}")
        # En cas d'erreur, retourner les données complètes
        return excel_data