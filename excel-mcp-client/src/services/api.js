import axios from 'axios';

// URL de l'agent local qui détecte les fichiers Excel ouverts
const LOCAL_AGENT_URL = 'http://localhost:8001';

// URL du service RunPod qui fait tourner les modèles d'embedding et LLM
// Utilisé en interne par l'agent local, pas directement par le frontend
const RUNPOD_URL = 'https://k994j50z5ge66t-8001.proxy.runpod.net';

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
 * @returns {Promise} - Statut de la tâche et visualisations éventuelles
 */
export const checkLLMTaskStatus = async (taskId) => {
  try {
    const response = await localApi.get(`/llm_task_status/${taskId}`);
    const data = response.data;
    
    // Si la tâche est terminée et contient une visualisation, la transmettre au frontend
    if (data.status === 'completed' && data.result && data.result.has_visualization) {
      return {
        ...data,
        has_visualization: data.result.has_visualization,
        visualization: data.result.visualization
      };
    }
    
    return data;
  } catch (error) {
    console.error('Task status check error:', error);
    return {
      success: false,
      error: error.message || 'Erreur lors de la vérification du statut de la tâche',
      has_visualization: false,
      visualization: null
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

/**
 * Met à jour une cellule dans Excel
 * @param {string} workbook - Nom du classeur
 * @param {string} sheet - Nom de la feuille
 * @param {string} cell - Adresse de la cellule (ex: "A1")
 * @param {any} value - Nouvelle valeur
 * @returns {Promise} - Résultat de l'opération
 */
export const updateExcelCell = async (workbook, sheet, cell, value) => {
  try {
    const response = await localApi.post('/update_cell', {
      workbook: workbook,
      sheet: sheet,
      cell: cell,
      value: value
    });
    return response.data;
  } catch (error) {
    console.error('Cell update error:', error);
    return {
      success: false,
      error: error.message || 'Erreur lors de la mise à jour de la cellule'
    };
  }
};

/**
 * Met à jour une plage de cellules dans Excel
 * @param {string} workbook - Nom du classeur
 * @param {string} sheet - Nom de la feuille
 * @param {string} startCell - Cellule de départ (ex: "A1")
 * @param {Array} values - Tableau 2D de valeurs
 * @returns {Promise} - Résultat de l'opération
 */
export const updateExcelRange = async (workbook, sheet, startCell, values) => {
  try {
    const response = await localApi.post('/update_range', {
      workbook: workbook,
      sheet: sheet,
      start_cell: startCell,
      values: values
    });
    return response.data;
  } catch (error) {
    console.error('Range update error:', error);
    return {
      success: false,
      error: error.message || 'Erreur lors de la mise à jour de la plage'
    };
  }
};

/**
 * Exécute une formule Excel dans une cellule
 * @param {string} workbook - Nom du classeur
 * @param {string} sheet - Nom de la feuille
 * @param {string} cell - Adresse de la cellule (ex: "A1")
 * @param {string} formula - Formule Excel (ex: "=SUM(A1:A10)")
 * @returns {Promise} - Résultat de l'opération
 */
export const executeExcelFormula = async (workbook, sheet, cell, formula) => {
  try {
    const response = await localApi.post('/execute_formula', {
      workbook: workbook,
      sheet: sheet,
      cell: cell,
      formula: formula
    });
    return response.data;
  } catch (error) {
    console.error('Formula execution error:', error);
    return {
      success: false,
      error: error.message || 'Erreur lors de l\'exécution de la formule'
    };
  }
};

/**
 * Ajoute une nouvelle feuille à un classeur Excel
 * @param {string} workbook - Nom du classeur
 * @param {string} sheet - Nom de la nouvelle feuille
 * @returns {Promise} - Résultat de l'opération
 */
export const addExcelSheet = async (workbook, sheet) => {
  try {
    const response = await localApi.post('/add_sheet', {
      workbook: workbook,
      sheet: sheet
    });
    return response.data;
  } catch (error) {
    console.error('Sheet addition error:', error);
    return {
      success: false,
      error: error.message || 'Erreur lors de l\'ajout de la feuille'
    };
  }
};

/**
 * Modifie Excel en utilisant une requête en langage naturel
 * Le LLM génère la commande à exécuter en fonction de la requête
 * @param {string} workbook - Nom du classeur
 * @param {string} sheet - Nom de la feuille
 * @param {string} query - Requête en langage naturel
 * @returns {Promise} - Résultat de l'opération
 */
export const modifyExcelWithLLM = async (workbook, sheet, query) => {
  try {
    const response = await localApi.post('/modify_excel_with_llm', {
      workbook: workbook,
      sheet: sheet,
      query: query
    });
    return response.data;
  } catch (error) {
    console.error('LLM modification error:', error);
    return {
      success: false,
      error: error.message || 'Erreur lors de la modification avec LLM'
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
  checkLLMTaskStatus,
  updateExcelCell,
  updateExcelRange,
  executeExcelFormula,
  addExcelSheet,
  modifyExcelWithLLM
};