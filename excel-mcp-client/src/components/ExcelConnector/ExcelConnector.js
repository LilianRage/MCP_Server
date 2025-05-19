import React, { useState, useEffect, useRef } from 'react';
import { 
  checkLocalStatus, 
  getOpenWorkbooks, 
  readSheetData, 
  analyzeSheetWithLLM,
  submitLLMAnalysis,
  checkLLMTaskStatus,
  modifyExcelWithLLM
} from '../../services/api';
import './ExcelConnector.css';
// Commentaire test
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
  const [llmResults, setLlmResults] = useState(null);
  
  // États pour le mode asynchrone
  const [taskId, setTaskId] = useState(null);
  const [taskStatus, setTaskStatus] = useState(null);
  const [pollingCount, setPollingCount] = useState(0);
  const [taskSubmittedTime, setTaskSubmittedTime] = useState(null);
  const [estimatedTimeRemaining, setEstimatedTimeRemaining] = useState(null);
  
  // Référence pour annuler le polling si le composant est démonté
  const pollingIntervalRef = useRef(null);

  // Ajouter un état pour la modification
  const [modifyQuery, setModifyQuery] = useState('');
  const [modificationResult, setModificationResult] = useState(null);

  
  
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
  
  // Soumettre une tâche LLM
  const handleAnalyze = async () => {
    if (!selectedWorkbook || !selectedSheet || !query.trim()) return;
    
    setLoading(true);
    setLlmResults(null);
    setTaskId(null);
    setTaskStatus(null);
    setEstimatedTimeRemaining(null);
    setStatus({...status, error: null});
    
    try {
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
    } catch (error) {
      console.error('Erreur lors de l\'analyse:', error);
      setStatus({
        ...status,
        error: `Erreur lors de l'analyse: ${error.message}`
      });
      setLoading(false);
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
  // Fonction pour modifier Excel avec LLM
  const handleModifyExcel = async () => {
    if (!selectedWorkbook || !selectedSheet || !modifyQuery.trim()) return;
    
    setLoading(true);
    setModificationResult(null);
    
    try {
      const response = await modifyExcelWithLLM(selectedWorkbook, selectedSheet, modifyQuery);
      
      if (response && response.success) {
        setModificationResult(response);
        // Recharger les données après modification
        await loadSheetData();
      } else {
        setStatus({
          ...status,
          error: response?.error || "Erreur lors de la modification Excel"
        });
      }
    } catch (error) {
      console.error('Erreur lors de la modification Excel:', error);
      setStatus({
        ...status,
        error: `Erreur lors de la modification Excel: ${error.message}`
      });
    } finally {
      setLoading(false);
    }
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
    setLlmResults(null);
    setTaskId(null);
    setTaskStatus(null);
  };
  
  // Changer de feuille sélectionnée
  const handleSheetChange = (e) => {
    setSelectedSheet(e.target.value);
    // Réinitialiser les données et résultats
    setExcelData(null);
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
        <div className="header-content">
          <h1>Excel Analyzer</h1>
          <p className="app-subtitle">Analyse intelligente de vos données Excel</p>
        </div>
        <button 
          className="refresh-button" 
          onClick={checkServerStatus}
          disabled={loading}
        >
          <span className="button-icon">↻</span>
          {loading ? 'Actualisation...' : 'Actualiser'}
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
                  width: taskStatus === 'processing' ? '50%' : '20%'
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
                  <span className="button-icon">↓</span>
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
          
          <div className="file-stats">
            <h2>Statistiques du fichier</h2>
            <div className="stats-grid">
              <div className="stat-item">
                <div className="stat-label">Lignes</div>
                <div className="stat-value">{excelData ? excelData.rowCount : '-'}</div>
              </div>
              <div className="stat-item">
                <div className="stat-label">Colonnes</div>
                <div className="stat-value">{excelData ? excelData.columnCount : '-'}</div>
              </div>
              <div className="stat-item">
                <div className="stat-label">Feuille</div>
                <div className="stat-value">{excelData ? excelData.sheetName : '-'}</div>
              </div>
              <div className="stat-item">
                <div className="stat-label">Classeur</div>
                <div className="stat-value">{selectedWorkbook || '-'}</div>
              </div>
            </div>
          </div>
        </div>
        
        <div className="content-area">
          {excelData ? (
            <div className="data-explorer">
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
                <h3>Poser une question sur ces données</h3>
                
                <div className="query-input">
                  <input
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="Ex: Calcule la somme de la colonne Montant par catégorie"
                    disabled={loading}
                    onKeyPress={(e) => e.key === 'Enter' && handleAnalyze()}
                  />
                  <button 
                    onClick={handleAnalyze} 
                    disabled={!query.trim() || loading}
                    className="analyze-button"
                  >
                    <span className="button-icon">✨</span>
                    Analyser
                  </button>
                </div>
              </div>
              
              {llmResults && (
                <div className="llm-results-section">
                  <h3>Résultat de l'analyse</h3>
                  
                  <div className="result-container">
                    <pre className="result-text">{llmResults.result}</pre>
                  </div>
                  
                  {/*llmResults.code && (
                    <div className="code-container">
                      <h4>Code Python généré:</h4>
                      <pre className="python-code">{llmResults.code}</pre>
                    </div>
                  )*/}
                </div>
              )}
              {excelData && (
                <div className="modify-excel-section">
                  <h3>Modifier les données Excel</h3>
                  <p className="section-intro">
                    Décrivez en langage naturel la modification que vous souhaitez apporter
                  </p>
                  
                  <div className="modify-input">
                    <input
                      type="text"
                      value={modifyQuery}
                      onChange={(e) => setModifyQuery(e.target.value)}
                      placeholder="Ex: Changer le prix en B5 à 150€"
                      disabled={loading}
                      onKeyPress={(e) => e.key === 'Enter' && handleModifyExcel()}
                    />
                    <button 
                      onClick={handleModifyExcel} 
                      disabled={!modifyQuery.trim() || loading}
                      className="modify-button"
                    >
                      <span className="button-icon">✏️</span>
                      Modifier
                    </button>
                  </div>
                  
                  {modificationResult && (
                    <div className="modification-result">
                      <h4>Modification effectuée</h4>
                      <div className="result-details">
                        <p><strong>Action réalisée:</strong> {modificationResult.command_generated?.command}</p>
                        {modificationResult.command_generated?.cell && (
                          <p><strong>Cellule modifiée:</strong> {modificationResult.command_generated.cell}</p>
                        )}
                        {modificationResult.command_generated?.value !== undefined && (
                          <p><strong>Nouvelle valeur:</strong> {modificationResult.command_generated.value}</p>
                        )}
                        {modificationResult.command_generated?.formula && (
                          <p><strong>Formule appliquée:</strong> {modificationResult.command_generated.formula}</p>
                        )}
                        <p className="success-message">{modificationResult.message}</p>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          ) : (
            <div className="no-data-message">
              {workbooks.length > 0 ? (
                <div className="no-data-content">
                  <div className="no-data-icon">📊</div>
                  <h3>Aucune donnée chargée</h3>
                  <p>Sélectionnez un classeur et une feuille, puis cliquez sur 'Charger les données'</p>
                </div>
              ) : (
                <div className="no-data-content">
                  <div className="no-data-icon">📑</div>
                  <h3>{status.excelRunning ? "Aucun classeur Excel ouvert" : "Excel n'est pas en cours d'exécution"}</h3>
                  <p>{status.excelRunning ? "Veuillez ouvrir un fichier Excel pour commencer" : "Veuillez lancer Excel et ouvrir un classeur"}</p>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
      
      <footer className="app-footer">
        <div className="footer-content">
          <p className="copyright">Excel Analyzer © {new Date().getFullYear()}</p>
          <div className="footer-links">
            <a href="#" className="footer-link">À propos</a>
            <a href="#" className="footer-link">Documentation</a>
            <a href="#" className="footer-link">Aide</a>
          </div>
        </div>
      </footer>
    </div>
  );
  
}

export default ExcelConnector;