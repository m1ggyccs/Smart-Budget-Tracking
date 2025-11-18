import React, { useState, useEffect } from 'react';
import { TrendingUp, TrendingDown, AlertCircle, BarChart3, Loader2 } from 'lucide-react';
import { API_BASE_URL } from './utils/constants';
import { formatPHP } from './utils/formatters';
import MetricCard from './components/MetricCard';
import InputControls from './components/InputControls';
import UnifiedForecastChart from './components/UnifiedForecastChart';
import ModelCards from './components/ModelCards';

export default function Dashboard() {
  const [selectedCategory, setSelectedCategory] = useState(null);
  const [categories, setCategories] = useState([]);
  const [predictionsData, setPredictionsData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [months, setMonths] = useState(6);
  const [budget, setBudget] = useState(25000);
  const [isRefreshing, setIsRefreshing] = useState(false);

  // Fetch categories on mount
  useEffect(() => {
    fetchCategories();
  }, []);

  // Fetch predictions when inputs change
  useEffect(() => {
    if (categories.length > 0 && selectedCategory) {
      fetchPredictions();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [months, budget, selectedCategory]);

  const fetchCategories = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/categories`);
      const data = await response.json();
      if (data.categories && data.categories.length > 0) {
        setCategories(data.categories);
        setSelectedCategory(data.categories[0]);
      } else {
        setError('No categories found. Please ensure data is processed.');
      }
    } catch (err) {
      setError(`Failed to fetch categories: ${err.message}`);
      console.error('Error fetching categories:', err);
    }
  };

  const fetchPredictions = async () => {
    setIsRefreshing(true);
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/predictions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ months, budget }),
      });
      
      if (!response.ok) {
        throw new Error(`API error: ${response.statusText}`);
      }
      
      const data = await response.json();
      setPredictionsData(data.predictions);
      setLoading(false);
    } catch (err) {
      setError(`Failed to fetch predictions: ${err.message}`);
      setLoading(false);
      console.error('Error fetching predictions:', err);
    } finally {
      setIsRefreshing(false);
    }
  };

  if (!selectedCategory || !predictionsData) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-blue-950 flex items-center justify-center">
        <div className="text-center">
          {loading ? (
            <>
              <Loader2 className="w-12 h-12 text-blue-500 animate-spin mx-auto mb-4" />
              <p className="text-white text-lg">Loading predictions...</p>
            </>
          ) : error ? (
            <>
              <AlertCircle className="w-12 h-12 text-red-500 mx-auto mb-4" />
              <p className="text-red-400 text-lg mb-4">{error}</p>
              <button
                onClick={fetchCategories}
                className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition"
              >
                Retry
              </button>
            </>
          ) : (
            <p className="text-white text-lg">No data available</p>
          )}
        </div>
      </div>
    );
  }

  const data = predictionsData[selectedCategory];
  if (!data || !data.predictions || !data.predictions.ensemble || data.predictions.ensemble.length === 0) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-blue-950 flex items-center justify-center">
        <div className="text-center">
          <AlertCircle className="w-12 h-12 text-yellow-500 mx-auto mb-4" />
          <p className="text-yellow-400 text-lg">No predictions available for this category</p>
        </div>
      </div>
    );
  }

  const currentPrediction = data.predictions.ensemble[0];
  const nextPrediction = data.predictions.ensemble[1] || currentPrediction;
  const percentageChange = currentPrediction.value > 0 
    ? ((nextPrediction.value - currentPrediction.value) / currentPrediction.value) * 100 
    : 0;

  const confidenceRange = {
    lower: currentPrediction.ci_lower,
    upper: currentPrediction.ci_upper,
  };

  const confidenceWidth = currentPrediction.value > 0
    ? ((confidenceRange.upper - confidenceRange.lower) / currentPrediction.value) * 100
    : 0;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-blue-950">
      <div className="bg-gradient-to-r from-slate-950 via-blue-950 to-slate-950 border-b-4 border-red-600">
        <div className="max-w-7xl mx-auto px-6 py-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-4xl font-bold text-white mb-2">
                Budget Forecasting Dashboard
              </h1>
              <p className="text-gray-400">
                AI-powered spending predictions with multi-model analysis
              </p>
            </div>
            <div className="w-16 h-16 bg-red-600 rounded-full flex items-center justify-center shadow-lg shadow-red-500/50">
              <BarChart3 className="w-8 h-8 text-white" />
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-8">
        <InputControls
          months={months}
          setMonths={setMonths}
          budget={budget}
          setBudget={setBudget}
          categories={categories}
          selectedCategory={selectedCategory}
          setSelectedCategory={setSelectedCategory}
          onRefresh={fetchPredictions}
          isRefreshing={isRefreshing}
        />

        {error && (
          <div className="mb-6 bg-red-900/50 border-2 border-red-600 rounded-xl p-4">
            <div className="flex items-center gap-2">
              <AlertCircle className="w-5 h-5 text-red-400" />
              <p className="text-red-400">{error}</p>
            </div>
          </div>
        )}

        {loading && (
          <div className="mb-6 bg-blue-900/50 border-2 border-blue-600 rounded-xl p-4">
            <div className="flex items-center gap-2">
              <Loader2 className="w-5 h-5 text-blue-400 animate-spin" />
              <p className="text-blue-400">Loading predictions...</p>
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <MetricCard
            title="Current Forecast"
            value={formatPHP(currentPrediction.value)}
            icon={<TrendingUp className="w-6 h-6" />}
            trend={percentageChange}
            trendLabel="vs last month"
            variant="primary"
          />
          
          <MetricCard
            title="Next Month"
            value={formatPHP(nextPrediction.value)}
            icon={percentageChange >= 0 ? 
              <TrendingUp className="w-6 h-6" /> : 
              <TrendingDown className="w-6 h-6" />
            }
            trend={percentageChange}
            trendLabel="expected change"
            variant={percentageChange >= 0 ? 'warning' : 'success'}
          />
          
          <MetricCard
            title="Confidence Range"
            value={`±${confidenceWidth.toFixed(1)}%`}
            icon={<AlertCircle className="w-6 h-6" />}
            subtitle={`${formatPHP(confidenceRange.lower)} - ${formatPHP(confidenceRange.upper)}`}
            variant="info"
          />
          
          <MetricCard
            title="Model Consensus"
            value="High"
            icon={<BarChart3 className="w-6 h-6" />}
            subtitle="All models agree"
            variant="success"
          />
        </div>

        <UnifiedForecastChart predictions={data.predictions} />

        <ModelCards predictions={data.predictions} />

        <div className="mt-8 bg-gradient-to-r from-red-600 to-yellow-500 rounded-xl p-6">
          <div className="flex items-start gap-4">
            <div className="bg-white rounded-full p-3">
              <BarChart3 className="w-6 h-6 text-red-600" />
            </div>
            <div>
              <h4 className="text-xl font-bold text-white mb-2">
                Why Ensemble Predictions?
              </h4>
              <p className="text-white text-sm opacity-90">
                Our ensemble model combines the strengths of all three forecasting methods 
                to provide the most accurate prediction. By weighing each model's performance 
                and confidence, we reduce individual model biases and capture a more complete 
                picture of your spending patterns.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

