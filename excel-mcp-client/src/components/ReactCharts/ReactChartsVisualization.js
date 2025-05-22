import React from 'react';
import LineChartComponent from './LineChartComponent';
import BarChartComponent from './BarChartComponent';
import PieChartComponent from './PieChartComponent';
import StatsTableComponent from './StatsTableComponent';
import './ReactChartsStyles.css';

/**
 * Composant principal de visualisation interactive avec Recharts
 * 
 * Ce composant détermine quel type de graphique afficher en fonction des données reçues
 * et délègue le rendu au composant spécifique approprié
 * 
 * @param {Object} props - Propriétés du composant
 * @param {Object} props.data - Données structurées pour la visualisation
 * @param {string} props.title - Titre optionnel pour la visualisation (remplace celui dans data)
 */
const ReactChartsVisualization = ({ data, title }) => {
  const [isVisible, setIsVisible] = React.useState(true);

  if (!data || !isVisible) {
    return null;
  }

  // Utiliser le titre passé en props ou celui dans les données
  const chartTitle = title || data.title || "Visualisation des données";
  const visualizationType = data.visualization_type;

  const renderVisualization = () => {
    switch (visualizationType) {
      case 'line':
        return <LineChartComponent data={data} />;
      case 'bar':
        return <BarChartComponent data={data} />;
      case 'pie':
        return <PieChartComponent data={data} />;
      case 'stats':
        return <StatsTableComponent data={data} />;
      default:
        return <div className="visualization-error">Type de visualisation non pris en charge: {visualizationType}</div>;
    }
  };

  return (
    <div className="recharts-visualization-container">
      <div className="recharts-visualization-header">
        <h4>{chartTitle}</h4>
        <button 
          className="toggle-button"
          onClick={() => setIsVisible(false)}
          title="Masquer la visualisation"
        >
          × Fermer
        </button>
      </div>

      <div className="recharts-visualization-content">
        {renderVisualization()}
      </div>
    </div>
  );
};

export default ReactChartsVisualization;
