import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';

/**
 * Composant de graphique linéaire basé sur Recharts
 * 
 * @param {Object} props - Propriétés du composant
 * @param {Object} props.data - Données structurées pour le graphique linéaire
 */
const LineChartComponent = ({ data }) => {
  if (!data || !data.data || !data.data.length) {
    return <div className="visualization-error">Données insuffisantes pour le graphique linéaire</div>;
  }

  // Extraire les clés d'axe
  const xAxisKey = data.x_axis?.key || Object.keys(data.data[0])[0];
  const yAxisKey = data.y_axis?.key || Object.keys(data.data[0])[1];
  
  // Extraire les libellés d'axe
  const xAxisLabel = data.x_axis?.label || xAxisKey;
  const yAxisLabel = data.y_axis?.label || yAxisKey;

  return (
    <div className="recharts-component-wrapper">
      <ResponsiveContainer width="100%" height={300}>
        <LineChart
          data={data.data}
          margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis 
            dataKey={xAxisKey} 
            label={{ value: xAxisLabel, position: 'insideBottomRight', offset: -10 }} 
          />
          <YAxis 
            label={{ value: yAxisLabel, angle: -90, position: 'insideLeft' }} 
          />
          <Tooltip />
          <Legend />
          <Line 
            type="monotone" 
            dataKey={yAxisKey} 
            stroke="#8884d8" 
            activeDot={{ r: 8 }} 
            name={yAxisLabel}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default LineChartComponent;
