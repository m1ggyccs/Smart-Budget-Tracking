import React from 'react';
import { Loader2, RefreshCw } from 'lucide-react';
import CategorySelector from './CategorySelector';

const InputControls = ({ 
  months, 
  setMonths, 
  budget, 
  setBudget, 
  categories, 
  selectedCategory, 
  setSelectedCategory,
  onRefresh,
  isRefreshing 
}) => {
  return (
    <div className="mb-8 bg-slate-900 rounded-xl p-6 border-2 border-blue-600">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <div>
          <label className="block text-sm font-semibold text-gray-300 mb-2">
            Months Ahead
          </label>
          <input
            type="number"
            min="1"
            max="12"
            value={months}
            onChange={(e) => setMonths(Math.max(1, Math.min(12, parseInt(e.target.value) || 6)))}
            className="w-full px-4 py-2 bg-slate-800 border border-blue-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
        <div>
          <label className="block text-sm font-semibold text-gray-300 mb-2">
            Monthly Budget (₱)
          </label>
          <input
            type="number"
            min="0"
            step="100"
            value={budget}
            onChange={(e) => setBudget(Math.max(0, parseFloat(e.target.value) || 25000))}
            className="w-full px-4 py-2 bg-slate-800 border border-blue-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
        <div className="flex items-end">
          <button
            onClick={onRefresh}
            disabled={isRefreshing}
            className="w-full px-6 py-2 bg-gradient-to-r from-red-600 to-yellow-500 text-white rounded-lg font-semibold hover:from-red-700 hover:to-yellow-600 transition disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            {isRefreshing ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Refreshing...
              </>
            ) : (
              <>
                <RefreshCw className="w-4 h-4" />
                Refresh Predictions
              </>
            )}
          </button>
        </div>
      </div>
      <CategorySelector
        categories={categories}
        selectedCategory={selectedCategory}
        onSelectCategory={setSelectedCategory}
      />
    </div>
  );
};

export default InputControls;

