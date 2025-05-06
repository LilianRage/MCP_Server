import React, { useState, useEffect, useRef } from 'react';
import { 
  checkLocalStatus, 
  getOpenWorkbooks, 
  readSheetData, 
  analyzeSheetWithEmbedding,
  analyzeSheetWithLLM,
  submitLLMAnalysis,
  checkLLMTaskStatus
} from '../../services/api';
import './ExcelConnector.css';

function ExcelConnector() {
  // États pour les statuts et données
  const [status, setStatus] = useState({
    connected: false,
    error: null,
    excelRunning: false
  });
  const [workbooks, setWorkbooks] = useState([]);
  const [selectedWorkbook, setSelectedWorkbook] = useState('');
  const [selectedSheet, setSelectedSheet] = useState('');
  const [sheets, setSheets] = useState([]);
  const [excelData, setExcelData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [query, setQuery] = useState('');
  const [results, setResults] = useState(null);
  const [llmResults, setLlmResults] = useState(null);
  const [topK, setTopK] = useState(5);
  const [analysisMode, setAnalysisMode] = useState('embedding'); // 'embedding' ou 'llm'
  
  // Nouveaux états pour le mode asynchrone
  const [taskId, setTaskId] = useState(null);
  const [taskStatus, setTaskStatus] = useState(null);
  const [pollingCount, setPollingCount] = useState(0);
  const [taskSubmittedTime, setTaskSubmittedTime] = useState(null);
  const [estimatedTimeRemaining, setEstimatedTimeRemaining] = useState(null);
  
  // Référence pour annuler le polling si le composant est démonté
  const pollingIntervalRef = useRef(null);
  
  // Vérifier la connexion et Excel au chargement
  useEffect(() => {
    checkServerStatus();
    
    // Nettoyer les intervalles lors du démontage
    return () => {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
      }
    };
  }, []);
  
  // Vérifier le statut du serveur local
  const checkServerStatus = async () => {
    setLoading(true);
    setStatus({...status, error: null});
    
    try {
      // Vérifier si le serveur est accessible et si Excel est en cours d'exécution
      const statusResponse = await checkLocalStatus();
      
      if (statusResponse.status === "online") {
        const excelStatus = statusResponse.excel || {};
        
        setStatus({
          connected: true,
          error: null,
          excelRunning: excelStatus.success || false
        });
        
        // Si Excel est en cours d'exécution, récupérer les classeurs ouverts
        if (excelStatus.success) {
          fetchOpenWorkbooks();
        }
      } else {
        setStatus({
          connected: true,
          error: "Le service local n'est pas en ligne",
          excelRunning: false
        });
      }
    } catch (error) {
      console.error('Erreur de connexion:', error);
      setStatus({
        connected: false,
        error: "Impossible de se connecter au serveur local",
        excelRunning: false
      });
    } finally {
      setLoading(false);
    }
  };
  
  // Récupérer les classeurs Excel ouverts
  const fetchOpenWorkbooks = async () => {
    setLoading(true);
    
    try {
      const response = await getOpenWorkbooks();
      
      if (response && response.success && response.workbooks) {
        setWorkbooks(response.workbooks);
        
        if (response.workbooks.length > 0) {
          // Présélectionner le premier classeur
          const firstWorkbook = response.workbooks[0];
          setSelectedWorkbook(firstWorkbook.name);
          
          // Charger les feuilles du classeur
          setSheets(firstWorkbook.sheets || []);
          
          // Présélectionner la première feuille si disponible
          if (firstWorkbook.sheets && firstWorkbook.sheets.length > 0) {
            setSelectedSheet(firstWorkbook.sheets[0]);
          }
        }
      } else {
        throw new Error(response?.error || "Erreur lors de la récupération des classeurs");
      }
    } catch (error) {
      console.error('Erreur lors de la récupération des classeurs:', error);
      setStatus({
        ...status,
        error: `Erreur lors de la récupération des classeurs: ${error.message}`
      });
    } finally {
      setLoading(false);
    }
  };
  
  // Charger les données d'une feuille Excel
  const loadSheetData = async () => {
    if (!selectedWorkbook || !selectedSheet) return;
    
    setLoading(true);
    setExcelData(null);
    setResults(null);
    setLlmResults(null);
    setTaskId(null);
    setTaskStatus(null);
    
    try {
      const response = await readSheetData(selectedWorkbook, selectedSheet);
      
      if (response && response.success && response.result) {
        setExcelData({
          headers: response.result.headers || [],
          rows: response.result.rows || [],
          rowCount: response.result.row_count || 0,
          columnCount: response.result.column_count || 0,
          sheetName: selectedSheet
        });
      } else {
        throw new Error(response?.error || "Erreur lors du chargement des données");
      }
    } catch (error) {
      console.error('Erreur lors du chargement des données:', error);
      setStatus({
        ...status,
        error: `Erreur lors du chargement des données: ${error.message}`
      });
    } finally {
      setLoading(false);
    }
  };
  
  // Vérifier périodiquement le statut d'une tâche LLM
  const startTaskPolling = (taskId) => {
    // Arrêter tout polling existant
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current);
    }
    
    setPollingCount(0);
    
    // Démarrer un nouveau polling
    const interval = setInterval(async () => {
      try {
        const response = await checkLLMTaskStatus(taskId);
        
        // Incrémenter le compteur de polling
        setPollingCount(prev => prev + 1);
        
        if (response.success) {
          // Mettre à jour le statut de la tâche
          setTaskStatus(response.status);
          
          // Si la tâche est terminée, arrêter le polling et afficher les résultats
          if (response.status === 'completed' && response.result) {
            clearInterval(interval);
            pollingIntervalRef.current = null;
            setLlmResults(response.result);
            setLoading(false);
          }
          // Si la tâche a échoué, arrêter le polling et afficher l'erreur
          else if (response.status === 'failed' || response.status === 'error') {
            clearInterval(interval);
            pollingIntervalRef.current = null;
            setStatus({
              ...status,
              error: response.error || "L'analyse a échoué"
            });
            setLoading(false);
          }
          // Mettre à jour le temps estimé restant
          else if (response.status === 'processing' || response.status === 'pending') {
            // Estimer le temps restant basé sur l'expérience passée
            const elapsedTime = (Date.now() - taskSubmittedTime) / 1000; // en secondes
            
            // Supposons qu'une analyse LLM typique prend environ 2 minutes
            const totalEstimatedTime = 120; // 2 minutes en secondes
            const estimatedTimeLeft = Math.max(0, totalEstimatedTime - elapsedTime);
            
            setEstimatedTimeRemaining(Math.round(estimatedTimeLeft));
          }
        } else {
          // En cas d'erreur dans la requête de statut
          console.error('Erreur lors de la vérification du statut:', response.error);
          
          // Si on obtient trop d'erreurs consécutives, arrêter le polling
          if (pollingCount > 20) {
            clearInterval(interval);
            pollingIntervalRef.current = null;
            setStatus({
              ...status,
              error: "Impossible de vérifier le statut de l'analyse après plusieurs tentatives"
            });
            setLoading(false);
          }
        }
      } catch (error) {
        console.error('Erreur lors du polling:', error);
        
        // Si on obtient trop d'erreurs consécutives, arrêter le polling
        if (pollingCount > 20) {
          clearInterval(interval);
          pollingIntervalRef.current = null;
          setStatus({
            ...status,
            error: `Erreur lors de la vérification du statut: ${error.message}`
          });
          setLoading(false);
        }
      }
    }, 3000); // Vérifier toutes les 3 secondes
    
    // Stocker l'intervalle dans la référence pour pouvoir le nettoyer
    pollingIntervalRef.current = interval;
  };
  
  // Analyser une feuille avec embedding ou soumettre une tâche LLM
  const handleAnalyze = async () => {
    if (!selectedWorkbook || !selectedSheet || !query.trim()) return;
    
    setLoading(true);
    setResults(null);
    setLlmResults(null);
    setTaskId(null);
    setTaskStatus(null);
    setEstimatedTimeRemaining(null);
    setStatus({...status, error: null});
    
    try {
      if (analysisMode === 'embedding') {
        // Analyse avec embedding - mode synchrone
        const response = await analyzeSheetWithEmbedding(
          selectedWorkbook, 
          selectedSheet, 
          query, 
          topK
        );
        
        if (response && response.success && response.result) {
          setResults(response.result);
        } else {
          throw new Error(response?.error || "Erreur lors de l'analyse");
        }
      } else {
        // Analyse avec LLM - mode asynchrone
        const response = await submitLLMAnalysis(
          selectedWorkbook,
          selectedSheet,
          query
        );
        
        if (response && response.success && response.task_id) {
          // Stocker l'ID de tâche
          setTaskId(response.task_id);
          setTaskStatus('pending');
          setTaskSubmittedTime(Date.now());
          
          // Démarrer le polling pour vérifier l'état de la tâche
          startTaskPolling(response.task_id);
        } else {
          throw new Error(response?.error || "Erreur lors de la soumission de l'analyse LLM");
        }
      }
    } catch (error) {
      console.error('Erreur lors de l\'analyse:', error);
      setStatus({
        ...status,
        error: `Erreur lors de l'analyse: ${error.message}`
      });
      setLoading(false);
    } finally {
      // On ne termine pas le chargement ici pour le mode LLM, 
      // car il se termine dans le polling lorsque la tâche est terminée
      if (analysisMode === 'embedding') {
        setLoading(false);
      }
    }
  };
  
  // Annuler une tâche en cours
  const handleCancelTask = () => {
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current);
      pollingIntervalRef.current = null;
    }
    
    setTaskId(null);
    setTaskStatus(null);
    setLoading(false);
    setPollingCount(0);
    setEstimatedTimeRemaining(null);
  };
  
  // Changer de classeur sélectionné
  const handleWorkbookChange = (e) => {
    const workbookName = e.target.value;
    setSelectedWorkbook(workbookName);
    
    // Mettre à jour les feuilles disponibles
    const workbook = workbooks.find(wb => wb.name === workbookName);
    if (workbook) {
      setSheets(workbook.sheets || []);
      // Présélectionner la première feuille
      setSelectedSheet(workbook.sheets && workbook.sheets.length > 0 ? workbook.sheets[0] : '');
    } else {
      setSheets([]);
      setSelectedSheet('');
    }
    
    // Réinitialiser les données et résultats
    setExcelData(null);
    setResults(null);
    setLlmResults(null);
    setTaskId(null);
    setTaskStatus(null);
  };
  
  // Changer de feuille sélectionnée
  const handleSheetChange = (e) => {
    setSelectedSheet(e.target.value);
    // Réinitialiser les données et résultats
    setExcelData(null);
    setResults(null);
    setLlmResults(null);
    setTaskId(null);
    setTaskStatus(null);
  };
  
  // Changer de mode d'analyse
  const handleAnalysisModeChange = (e) => {
    setAnalysisMode(e.target.value);
    // Réinitialiser les résultats
    setResults(null);
    setLlmResults(null);
    setTaskId(null);
    setTaskStatus(null);
  };
  
  // Formater le temps estimé
  const formatEstimatedTime = (seconds) => {
    if (!seconds) return '';
    
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    
    if (minutes > 0) {
      return `${minutes} min ${remainingSeconds} s`;
    } else {
      return `${remainingSeconds} secondes`;
    }
  };
  
  return (
    <div className="excel-connector">
      <header>
        <h1>Excel Analyzer</h1>
        <button 
          className="refresh-button" 
          onClick={checkServerStatus}
          disabled={loading}
        >
          Rafraîchir
        </button>
      </header>
      
      {status.error && <div className="error-message">{status.error}</div>}
      
      {loading && !taskId && <div className="loading-indicator">Chargement en cours...</div>}
      
      {/* Indicateur spécial pour les tâches LLM asynchrones */}
      {taskId && (taskStatus === 'pending' || taskStatus === 'processing') && (
        <div className="task-status-indicator">
          <div className="task-status-header">
            <h3>Analyse IA en cours</h3>
            <button 
              className="cancel-task-button" 
              onClick={handleCancelTask}
            >
              Annuler
            </button>
          </div>
          <div className="task-progress">
            <div className="progress-bar">
              <div 
                className="progress-bar-inner" 
                style={{ 
                  width: taskStatus === 'processing' ? '50%' : '20%',
                  animation: 'pulse 2s infinite'
                }}
              ></div>
            </div>
            <div className="task-status-details">
              <p>Statut: <span className="status-badge">{taskStatus === 'pending' ? 'En attente' : 'En cours de traitement'}</span></p>
              <p>ID de tâche: <span className="task-id">{taskId}</span></p>
              {estimatedTimeRemaining && (
                <p>Temps estimé restant: <span className="time-remaining">{formatEstimatedTime(estimatedTimeRemaining)}</span></p>
              )}
              <p className="status-message">
                {taskStatus === 'pending' 
                  ? 'Votre analyse est en file d\'attente et sera traitée dès que possible...' 
                  : 'Le modèle d\'IA analyse actuellement vos données, veuillez patienter...'}
              </p>
            </div>
          </div>
        </div>
      )}
      
      <div className="main-content">
        <div className="sidebar">
          <div className="workbook-selector">
            <h2>Sélection du classeur</h2>
            
            {workbooks.length > 0 ? (
              <>
                <div className="select-group">
                  <label>Classeur Excel:</label>
                  <select 
                    value={selectedWorkbook} 
                    onChange={handleWorkbookChange}
                    disabled={loading}
                  >
                    {workbooks.map(wb => (
                      <option key={wb.name} value={wb.name}>{wb.name}</option>
                    ))}
                  </select>
                </div>
                
                {selectedWorkbook && (
                  <div className="select-group">
                    <label>Feuille:</label>
                    <select 
                      value={selectedSheet} 
                      onChange={handleSheetChange}
                      disabled={loading}
                    >
                      {sheets.map(sheet => (
                        <option key={sheet} value={sheet}>{sheet}</option>
                      ))}
                    </select>
                  </div>
                )}
                
                <button 
                  onClick={loadSheetData} 
                  disabled={!selectedWorkbook || !selectedSheet || loading}
                  className="load-button"
                >
                  Charger les données
                </button>
              </>
            ) : (
              <div className="no-workbooks-message">
                {status.connected ? (
                  status.excelRunning ? 
                    "Aucun classeur Excel n'est ouvert. Veuillez ouvrir un fichier Excel." : 
                    "Excel n'est pas en cours d'exécution. Veuillez ouvrir Excel."
                ) : (
                  "Connexion au serveur local non établie."
                )}
              </div>
            )}
          </div>
          
          <div className="options-panel">
            <h2>Options d'analyse</h2>
            
            <div className="option-group">
              <label>Mode d'analyse:</label>
              <div className="radio-group">
                <label>
                  <input
                    type="radio"
                    name="analysisMode"
                    value="embedding"
                    checked={analysisMode === 'embedding'}
                    onChange={handleAnalysisModeChange}
                    disabled={loading}
                  />
                  Recherche par similarité
                </label>
                <label>
                  <input
                    type="radio"
                    name="analysisMode"
                    value="llm"
                    checked={analysisMode === 'llm'}
                    onChange={handleAnalysisModeChange}
                    disabled={loading}
                  />
                  Analyse par IA (LLM)
                </label>
              </div>
            </div>
            
            {analysisMode === 'embedding' && (
              <div className="option-group">
                <label>
                  Nombre de résultats (top-k):
                  <input 
                    type="number" 
                    min="1" 
                    max="20" 
                    value={topK} 
                    onChange={(e) => setTopK(Math.max(1, parseInt(e.target.value) || 1))}
                    disabled={loading}
                  />
                </label>
              </div>
            )}
            
            {analysisMode === 'llm' && (
              <div className="llm-info">
                <p>L'analyse par IA peut prendre jusqu'à 2-3 minutes selon la complexité de la requête.</p>
                <p>Vous pourrez continuer à utiliser l'application pendant le traitement.</p>
              </div>
            )}
          </div>
        </div>
        
        <div className="content-area">
          {excelData ? (
            <div className="data-explorer">
              <div className="data-info">
                <h2>Données chargées</h2>
                <p>
                  <strong>Classeur:</strong> {selectedWorkbook} | 
                  <strong>Feuille:</strong> {excelData.sheetName} | 
                  <strong>Lignes:</strong> {excelData.rowCount} | 
                  <strong>Colonnes:</strong> {excelData.columnCount}
                </p>
              </div>
              
              <div className="data-preview">
                <h3>Aperçu des données</h3>
                
                <div className="table-container">
                  <table>
                    <thead>
                      <tr>
                        <th className="row-index">#</th>
                        {excelData.headers.map((header, index) => (
                          <th key={index}>{header}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {excelData.rows.slice(0, 5).map((row, rowIndex) => (
                        <tr key={rowIndex}>
                          <td className="row-index">{rowIndex + 1}</td>
                          {row.map((cell, cellIndex) => (
                            <td key={cellIndex}>{cell !== null ? cell : ''}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                
                {excelData.rowCount > 5 && (
                  <p className="more-rows-note">
                    + {excelData.rowCount - 5} lignes supplémentaires
                  </p>
                )}
              </div>
              
              <div className="query-section">
                <h3>
                  {analysisMode === 'embedding' 
                    ? "Rechercher des lignes similaires" 
                    : "Poser une question sur ces données"}
                </h3>
                
                <div className="query-input">
                  <input
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder={analysisMode === 'embedding' 
                      ? "Ex: Quelles lignes contiennent des informations sur..." 
                      : "Ex: Calcule la somme de la colonne Montant"}
                    disabled={loading}
                    onKeyPress={(e) => e.key === 'Enter' && handleAnalyze()}
                  />
                  <button 
                    onClick={handleAnalyze} 
                    disabled={!query.trim() || loading}
                    className="analyze-button"
                  >
                    {analysisMode === 'embedding' ? "Rechercher" : "Analyser"}
                  </button>
                </div>
              </div>
              
              {results && analysisMode === 'embedding' && (
                <div className="results-section">
                  <h3>Résultats de la recherche</h3>
                  
                  <div className="table-container">
                    <table>
                      <thead>
                        <tr>
                          <th className="score-column">Score</th>
                          <th className="row-index">Ligne</th>
                          {results.headers.map((header, index) => (
                            <th key={index}>{header}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {results.rows.map((row, rowIndex) => {
                          const originalIndex = results.original_indices[rowIndex];
                          const score = results.similarity_scores[originalIndex];
                          
                          return (
                            <tr key={rowIndex} className={rowIndex === 0 ? "top-result" : ""}>
                              <td className="similarity-score">
                                {(score * 100).toFixed(2)}%
                              </td>
                              <td className="row-index">{originalIndex + 1}</td>
                              {row.map((cell, cellIndex) => (
                                <td key={cellIndex}>{cell !== null ? cell : ''}</td>
                              ))}
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
              
              {llmResults && analysisMode === 'llm' && (
                <div className="llm-results-section">
                  <h3>Résultat de l'analyse</h3>
                  
                  <div className="result-container">
                    <pre className="result-text">{llmResults.result}</pre>
                  </div>
                  
                  {llmResults.code && (
                    <div className="code-container">
                      <h4>Code Python généré:</h4>
                      <pre className="python-code">{llmResults.code}</pre>
                    </div>
                  )}
                </div>
              )}
            </div>
          ) : (
            <div className="no-data-message">
              {workbooks.length > 0 ? (
                "Sélectionnez un classeur et une feuille, puis cliquez sur 'Charger les données'"
              ) : (
                status.excelRunning ? 
                  "Aucun classeur Excel ouvert détecté" : 
                  "Excel n'est pas en cours d'exécution"
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default ExcelConnector;