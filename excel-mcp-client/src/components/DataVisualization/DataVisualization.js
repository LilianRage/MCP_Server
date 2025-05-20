import React, { useState } from 'react';
import './DataVisualization.css';

/**
 * Composant de visualisation de données
 * 
 * Ce composant affiche une visualisation basée sur une image encodée en base64
 * Il est conçu pour fonctionner avec les visualisations générées par matplotlib/seaborn
 * 
 * @param {Object} props - Propriétés du composant
 * @param {string} props.base64Image - Image encodée en base64
 * @param {string} props.title - Titre de la visualisation
 * @param {string} props.description - Description de la visualisation
 */
const DataVisualization = ({ base64Image, title = "Visualisation des données", description = "" }) => {
  const [isVisible, setIsVisible] = useState(true);

  if (!base64Image || !isVisible) {
    return null;
  }

  return (
    <div className="visualization-container">
      <div className="visualization-header">
        <h4>{title}</h4>
        <button 
          className="toggle-button"
          onClick={() => setIsVisible(false)}
          title="Masquer la visualisation"
        >
          × Fermer
        </button>
      </div>
      
      {description && (
        <p className="visualization-description">
          {description}
        </p>
      )}

      <img 
        src={`data:image/png;base64,${base64Image}`}
        alt="Visualisation des données"
        className="visualization-image"
      />
    </div>
  );
};

export default DataVisualization;
