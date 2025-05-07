import axios from 'axios';

// URL de l'agent local qui détecte les fichiers Excel ouverts
const LOCAL_AGENT_URL = 'http://localhost:8001';

// URL du service RunPod qui fait tourner les modèles d'embedding et LLM
// Utilisé en interne par l'agent local, pas directement par le frontend
const RUNPOD_URL = 'https://523ryay9qiglbv-8001.proxy.runpod.net';

// Client API pour communiquer avec l'agent local
const localApi = axios.create({
  baseURL: LOCAL_AGENT_URL,
  headers: {
    'Content-Type': 'application/json'
  }
});

/**
 * Vérifie le statut de l'agent local et la disponibilité d'Excel
 * @returns {Promise} - Statut du service local et Excel
 */
export const checkLocalStatus = async () => {
  try {
    const response = await localApi.get('/status');
    return response.data;
  } catch (error) {
    console.error('Status check error:', error);
    return { 
      success: false, 
      error: error.message || 'Erreur de connexion au serveur local'
    };
  }
};

/**
 * Récupère la liste des classeurs Excel ouverts
 * @returns {Promise} - Liste des classeurs
 */
export const getOpenWorkbooks = async () => {
  try {
    const response = await localApi.get('/workbooks');
    return response.data;
  } catch (error) {
    console.error('Workbooks fetch error:', error);
    return {
      success: false,
      error: error.message || 'Erreur lors de la récupération des classeurs'
    };
  }
};

/**
 * Lit les données d'une feuille dans un classeur ouvert
 * @param {string} workbook - Nom du classeur
 * @param {string} sheet - Nom de la feuille
 * @returns {Promise} - Données de la feuille
 */
export const readSheetData = async (workbook, sheet) => {
  try {
    const response = await localApi.post('/read_sheet', {
      workbook: workbook,
      sheet: sheet
    });
    return response.data;
  } catch (error) {
    console.error('Sheet read error:', error);
    return {
      success: false,
      error: error.message || 'Erreur lors de la lecture de la feuille'
    };
  }
};

/**
 * Analyse une feuille Excel avec embedding sur RunPod
 * @param {string} workbook - Nom du classeur
 * @param {string} sheet - Nom de la feuille
 * @param {string} query - Requête utilisateur
 * @param {number} topK - Nombre de résultats à retourner
 * @returns {Promise} - Résultat de l'analyse
 */
export const analyzeSheetWithEmbedding = async (workbook, sheet, query, topK = 5) => {
  try {
    const response = await localApi.post('/analyze_sheet', {
      workbook: workbook,
      sheet: sheet,
      query: query,
      top_k: topK
    });
    return response.data;
  } catch (error) {
    console.error('Embedding analysis error:', error);
    return {
      success: false,
      error: error.message || 'Erreur lors de l\'analyse avec embedding'
    };
  }
};

/**
 * Soumet une tâche d'analyse LLM asynchrone
 * @param {string} workbook - Nom du classeur
 * @param {string} sheet - Nom de la feuille
 * @param {string} query - Requête utilisateur
 * @returns {Promise} - ID de la tâche soumise
 */
export const submitLLMAnalysis = async (workbook, sheet, query) => {
  try {
    const response = await localApi.post('/submit_llm_analysis', {
      workbook: workbook,
      sheet: sheet,
      query: query
    });
    return response.data;
  } catch (error) {
    console.error('LLM task submission error:', error);
    return {
      success: false,
      error: error.message || 'Erreur lors de la soumission de l\'analyse LLM'
    };
  }
};

/**
 * Vérifie le statut d'une tâche d'analyse LLM
 * @param {string} taskId - ID de la tâche à vérifier
 * @returns {Promise} - Statut de la tâche
 */
export const checkLLMTaskStatus = async (taskId) => {
  try {
    const response = await localApi.get(`/llm_task_status/${taskId}`);
    return response.data;
  } catch (error) {
    console.error('Task status check error:', error);
    return {
      success: false,
      error: error.message || 'Erreur lors de la vérification du statut de la tâche'
    };
  }
};

/**
 * Analyse une feuille Excel avec LLM et génération de code Python
 * Cette méthode est maintenue pour la compatibilité avec l'ancienne API
 * Elle utilise l'API synchrone qui fait du polling en interne
 * @param {string} workbook - Nom du classeur
 * @param {string} sheet - Nom de la feuille
 * @param {string} query - Requête utilisateur
 * @returns {Promise} - Résultat de l'analyse LLM
 */
export const analyzeSheetWithLLM = async (workbook, sheet, query) => {
  try {
    const response = await localApi.post('/analyze_sheet_llm', {
      workbook: workbook,
      sheet: sheet,
      query: query
    });
    return response.data;
  } catch (error) {
    console.error('LLM analysis error:', error);
    return {
      success: false,
      error: error.message || 'Erreur lors de l\'analyse avec LLM'
    };
  }
};

// Exporter toutes les fonctions
export default {
  checkLocalStatus,
  getOpenWorkbooks,
  readSheetData,
  analyzeSheetWithEmbedding,
  analyzeSheetWithLLM,
  submitLLMAnalysis,
  checkLLMTaskStatus
};