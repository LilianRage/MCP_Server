import React from 'react';

/**
 * Composant de tableau de statistiques
 * 
 * @param {Object} props - Propriétés du composant
 * @param {Object} props.data - Données structurées pour le tableau de statistiques
 */
const StatsTableComponent = ({ data }) => {
  if (!data || !data.data || !data.data.length) {
    return <div className="visualization-error">Données insuffisantes pour le tableau de statistiques</div>;
  }

  return (
    <div className="stats-table-wrapper">
      <table className="stats-table">
        <thead>
          <tr>
            <th>Libellé</th>
            <th>Valeur</th>
          </tr>
        </thead>
        <tbody>
          {data.data.map((item, index) => (
            <tr key={`stat-${index}`}>
              <td>{item.label}</td>
              <td>{item.value}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default StatsTableComponent;
