import React from 'react';
import { ComposedChart, Line, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { formatPHP, formatPeriodShort } from '../utils/formatters';

const UnifiedForecastChart = ({ predictions }) => {
  const chartData = predictions.ensemble.map((ensemblePoint) => {
    const period = ensemblePoint.period;
    const maPoint = predictions.models.moving_average.find(p => p.period === period);
    const hwPoint = predictions.models.holtwinters.find(p => p.period === period);
    const lstmPoint = predictions.models.lstm.find(p => p.period === period);

    return {
      period,
      ensemble: ensemblePoint.value,
      ci_lower: ensemblePoint.ci_lower,
      ci_upper: ensemblePoint.ci_upper,
      moving_average: maPoint?.value || 0,
      holtwinters: hwPoint?.value || 0,
      lstm: lstmPoint?.value || 0,
    };
  });

  const CustomTooltip = ({ active, payload, label }) => {
    if (!active || !payload || !payload.length) return null;

    return (
      <div className="bg-slate-900 border-2 border-blue-600 p-4 rounded-lg shadow-xl">
        <p className="text-yellow-500 font-bold mb-2">{formatPeriodShort(label)}</p>
        {payload.map((entry, index) => {
          if (entry.dataKey === 'ci_lower' || entry.dataKey === 'ci_upper') return null;
          
          return (
            <div key={index} className="flex justify-between gap-4 mb-1">
              <span className="text-gray-300" style={{ color: entry.color }}>
                {entry.name}:
              </span>
              <span className="text-white font-semibold">
                {formatPHP(entry.value)}
              </span>
            </div>
          );
        })}
        {payload[0]?.payload.ci_lower && (
          <div className="mt-2 pt-2 border-t border-blue-800">
            <p className="text-xs text-gray-400">Confidence Interval</p>
            <div className="flex justify-between gap-2 text-xs">
              <span className="text-blue-300">Lower:</span>
              <span className="text-white">{formatPHP(payload[0].payload.ci_lower)}</span>
            </div>
            <div className="flex justify-between gap-2 text-xs">
              <span className="text-blue-300">Upper:</span>
              <span className="text-white">{formatPHP(payload[0].payload.ci_upper)}</span>
            </div>
          </div>
        )}
      </div>
    );
  };

  const CustomLegend = (props) => {
    const { payload } = props;
    return (
      <div className="flex flex-wrap justify-center gap-6 mt-4">
        {payload.map((entry, index) => (
          <div key={index} className="flex items-center gap-2">
            <div className="w-8 h-0.5 rounded" style={{ backgroundColor: entry.color }} />
            <span className="text-sm text-gray-300">{entry.value}</span>
          </div>
        ))}
      </div>
    );
  };

  return (
    <div className="bg-slate-900 rounded-xl p-6 shadow-xl border border-slate-800">
      <div className="mb-4">
        <h3 className="text-2xl font-bold text-white">
          Multi-Model Forecast Analysis
        </h3>
        <p className="text-gray-400 mt-1">
          Ensemble prediction with confidence intervals
        </p>
      </div>

      <ResponsiveContainer width="100%" height={500}>
        <ComposedChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
          <defs>
            <linearGradient id="confidenceGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#2563EB" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#2563EB" stopOpacity={0.05} />
            </linearGradient>
          </defs>

          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
          
          <XAxis
            dataKey="period"
            tickFormatter={formatPeriodShort}
            stroke="#9CA3AF"
            style={{ fontSize: '12px' }}
          />
          
          <YAxis
            tickFormatter={(value) => `₱${(value / 1000).toFixed(0)}K`}
            stroke="#9CA3AF"
            style={{ fontSize: '12px' }}
          />

          <Tooltip content={<CustomTooltip />} />
          <Legend content={<CustomLegend />} />

          <Area
            type="monotone"
            dataKey="ci_upper"
            stroke="none"
            fill="url(#confidenceGradient)"
            fillOpacity={1}
            name="Confidence Interval"
          />
          <Area
            type="monotone"
            dataKey="ci_lower"
            stroke="none"
            fill="#0f172a"
            fillOpacity={1}
          />

          <Line
            type="monotone"
            dataKey="moving_average"
            stroke="#60A5FA"
            strokeWidth={2}
            dot={{ fill: '#60A5FA', r: 3 }}
            activeDot={{ r: 5 }}
            name="Moving Average"
          />
          
          <Line
            type="monotone"
            dataKey="holtwinters"
            stroke="#A78BFA"
            strokeWidth={2}
            dot={{ fill: '#A78BFA', r: 3 }}
            activeDot={{ r: 5 }}
            name="Holt-Winters"
          />
          
          <Line
            type="monotone"
            dataKey="lstm"
            stroke="#34D399"
            strokeWidth={2}
            dot={{ fill: '#34D399', r: 3 }}
            activeDot={{ r: 5 }}
            name="LSTM"
          />

          <Line
            type="monotone"
            dataKey="ensemble"
            stroke="#FDB913"
            strokeWidth={4}
            dot={{ fill: '#FDB913', r: 5, strokeWidth: 2, stroke: '#DC0000' }}
            activeDot={{ r: 7 }}
            name="Ensemble Prediction"
          />
        </ComposedChart>
      </ResponsiveContainer>

      <div className="mt-6 grid grid-cols-4 gap-4">
        <div className="bg-slate-800 rounded-lg p-3 border border-blue-900">
          <div className="flex items-center gap-2 mb-1">
            <div className="w-3 h-3 rounded-full bg-[#60A5FA]"></div>
            <span className="text-xs text-gray-400">Moving Avg</span>
          </div>
          <p className="text-sm text-white">Simple trend</p>
        </div>
        
        <div className="bg-slate-800 rounded-lg p-3 border border-blue-900">
          <div className="flex items-center gap-2 mb-1">
            <div className="w-3 h-3 rounded-full bg-[#A78BFA]"></div>
            <span className="text-xs text-gray-400">Holt-Winters</span>
          </div>
          <p className="text-sm text-white">Seasonal patterns</p>
        </div>
        
        <div className="bg-slate-800 rounded-lg p-3 border border-blue-900">
          <div className="flex items-center gap-2 mb-1">
            <div className="w-3 h-3 rounded-full bg-[#34D399]"></div>
            <span className="text-xs text-gray-400">LSTM</span>
          </div>
          <p className="text-sm text-white">Deep learning</p>
        </div>
        
        <div className="bg-gradient-to-r from-red-600 to-yellow-600 rounded-lg p-3">
          <div className="flex items-center gap-2 mb-1">
            <div className="w-3 h-3 rounded-full bg-white"></div>
            <span className="text-xs text-white font-bold">Ensemble</span>
          </div>
          <p className="text-sm text-white font-semibold">Best combined</p>
        </div>
      </div>
    </div>
  );
};

export default UnifiedForecastChart;

