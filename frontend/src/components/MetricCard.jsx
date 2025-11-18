import React from 'react';

const MetricCard = ({ title, value, icon, trend, trendLabel, subtitle, variant = 'primary' }) => {
  const variantStyles = {
    primary: 'border-blue-600 bg-slate-900',
    success: 'border-green-500 bg-slate-900',
    warning: 'border-yellow-500 bg-slate-900',
    info: 'border-blue-400 bg-slate-900',
  };

  const iconColors = {
    primary: 'text-blue-500',
    success: 'text-green-400',
    warning: 'text-yellow-500',
    info: 'text-blue-400',
  };

  return (
    <div className={`${variantStyles[variant]} border-2 rounded-xl p-6 shadow-lg`}>
      <div className="flex items-start justify-between mb-4">
        <div className={iconColors[variant]}>{icon}</div>
        {trend !== undefined && (
          <span className={`text-sm font-semibold ${trend >= 0 ? 'text-red-500' : 'text-green-400'}`}>
            {trend >= 0 ? '+' : ''}{trend.toFixed(1)}%
          </span>
        )}
      </div>
      <h3 className="text-gray-400 text-sm mb-2">{title}</h3>
      <p className="text-3xl font-bold text-white mb-1">{value}</p>
      {(trendLabel || subtitle) && (
        <p className="text-xs text-gray-500">{trendLabel || subtitle}</p>
      )}
    </div>
  );
};

export default MetricCard;

