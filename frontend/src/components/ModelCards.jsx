import React from 'react';
import { formatPHP } from '../utils/formatters';

const ModelCards = ({ predictions }) => {
  return (
    <div className="mt-8 grid grid-cols-1 lg:grid-cols-3 gap-6">
      <div className="bg-slate-900 rounded-xl p-6 border-2 border-[#60A5FA]">
        <h4 className="text-lg font-bold text-white mb-3">Moving Average</h4>
        <p className="text-gray-400 text-sm mb-4">
          Simple historical trend analysis based on recent spending patterns.
        </p>
        <div className="bg-slate-800 rounded-lg p-4">
          <p className="text-2xl font-bold text-[#60A5FA]">
            {formatPHP(predictions.models.moving_average[0]?.value || 0)}
          </p>
          <p className="text-xs text-gray-500 mt-1">Current prediction</p>
        </div>
      </div>

      <div className="bg-slate-900 rounded-xl p-6 border-2 border-[#A78BFA]">
        <h4 className="text-lg font-bold text-white mb-3">Holt-Winters</h4>
        <p className="text-gray-400 text-sm mb-4">
          Exponential smoothing with seasonal patterns and trend components.
        </p>
        <div className="bg-slate-800 rounded-lg p-4">
          <p className="text-2xl font-bold text-[#A78BFA]">
            {formatPHP(predictions.models.holtwinters[0]?.value || 0)}
          </p>
          <p className="text-xs text-gray-500 mt-1">Current prediction</p>
        </div>
      </div>

      <div className="bg-slate-900 rounded-xl p-6 border-2 border-[#34D399]">
        <h4 className="text-lg font-bold text-white mb-3">LSTM Neural Network</h4>
        <p className="text-gray-400 text-sm mb-4">
          Deep learning model that captures complex, non-linear patterns.
        </p>
        <div className="bg-slate-800 rounded-lg p-4">
          <p className="text-2xl font-bold text-[#34D399]">
            {formatPHP(predictions.models.lstm[0]?.value || 0)}
          </p>
          <p className="text-xs text-gray-500 mt-1">Current prediction</p>
        </div>
      </div>
    </div>
  );
};

export default ModelCards;

