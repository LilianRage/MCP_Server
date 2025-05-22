import React from 'react';
import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';

/**
 * Composant de diagramme circulaire basé sur Recharts
 * 
 * @param {Object} props - Propriétés du composant
 * @param {Object} props.data - Données structurées pour le diagramme circulaire
 */
const PieChartComponent = ({ data }) => {
  if (!data || !data.data || !data.data.length) {
    return <div className="visualization-error">Données insuffisantes pour le diagramme circulaire</div>;
  }

  // Couleurs par défaut pour les segments
  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#82CA9D', '#A4DE6C'];

  return (
    <div className="recharts-component-wrapper">
      <ResponsiveContainer width="100%" height={300}>
        <PieChart>
          <Pie
            data={data.data}
            cx="50%"
            cy="50%"
            labelLine={true}
            outerRadius={80}
            fill="#8884d8"
            dataKey="value"
            nameKey="name"
            label={({name, percent}) => `${name}: ${(percent * 100).toFixed(0)}%`}
          >
            {data.data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
            ))}
          </Pie>
          <Tooltip formatter={(value) => [value, 'Valeur']} />
          <Legend />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
};

export default PieChartComponent;
