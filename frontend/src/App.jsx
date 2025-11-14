import React, { useEffect, useState } from 'react';
import {
  AlertCircle,
  BarChart3,
  Calendar,
  CheckCircle,
  DollarSign,
  FileText,
  LogOut,
  PieChart,
  PlusCircle,
  Settings,
  Tag,
  TrendingDown,
  TrendingUp,
  User,
  Wallet,
} from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8001/v1';

const categories = ['Groceries', 'Transport', 'Entertainment', 'Shopping', 'Bills', 'Healthcare', 'Other'];

const navItems = ['dashboard', 'budgets', 'transactions', 'analytics', 'forecast'];

const initialBudgetForm = {
  amount: '',
  period: 'monthly',
  alerts_enabled: true,
  start_date: '',
  end_date: '',
};

const initialTransactionForm = {
  amount: '',
  category: categories[0],
  transaction_type: 'expense',
  occurred_at: new Date().toISOString().split('T')[0],
  notes: '',
};

function App() {
  const [currentView, setCurrentView] = useState('login');
  const [token, setToken] = useState(null);
  const [user, setUser] = useState(null);

  const [budgets, setBudgets] = useState([]);
  const [transactions, setTransactions] = useState([]);
  const [analytics, setAnalytics] = useState(null);
  const [forecastDetails, setForecastDetails] = useState([]);

  const [authForm, setAuthForm] = useState({ email: '', password: '', full_name: '' });
  const [budgetForm, setBudgetForm] = useState(initialBudgetForm);
  const [transactionForm, setTransactionForm] = useState(initialTransactionForm);
  const [forecastCategory, setForecastCategory] = useState(categories[0]);

  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (token) {
      fetchDashboardData();
    }
  }, [token]);

  const apiCall = async (endpoint, method = 'GET', body = null, useAuth = true) => {
    const headers = { 'Content-Type': 'application/json' };
    if (useAuth && token) {
      headers.Authorization = `Bearer ${token}`;
    }

    const res = await fetch(`${API_BASE}${endpoint}`, {
      method,
      headers,
      body: body ? JSON.stringify(body) : undefined,
    });

    if (!res.ok) {
      const detail = await res.text();
      throw new Error(detail || 'Request failed');
    }

    return res.json();
  };

  const handleLogin = async () => {
    setLoading(true);
    try {
      const formData = new URLSearchParams();
      formData.append('username', authForm.email);
      formData.append('password', authForm.password);

      const res = await fetch(`${API_BASE}/auth/token`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: formData,
      });

      if (!res.ok) {
        throw new Error('Invalid email or password');
      }

      const data = await res.json();
      setToken(data.access_token);
      setUser({ email: authForm.email });
      setCurrentView('dashboard');
    } catch (err) {
      alert(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleRegister = async () => {
    setLoading(true);
    try {
      await apiCall(
        '/auth/register',
        'POST',
        {
          email: authForm.email,
          password: authForm.password,
          full_name: authForm.full_name,
        },
        false
      );
      alert('Registration successful. Please log in.');
      setCurrentView('login');
    } catch (err) {
      alert(err.message);
    } finally {
      setLoading(false);
    }
  };

  const fetchDashboardData = async () => {
    setLoading(true);
    try {
      const [budgetsData, transactionsData, insightsData, summaryData] = await Promise.all([
        apiCall('/budgets'),
        apiCall('/transactions'),
        apiCall('/analytics/insights'),
        apiCall('/transactions/summary'),
      ]);

      setBudgets(budgetsData);
      setTransactions(transactionsData);
      setAnalytics({ insights: insightsData, summary: summaryData });
    } catch (err) {
      console.error('Failed to load dashboard data', err);
    } finally {
      setLoading(false);
    }
  };

  const handleAddBudget = async () => {
    setLoading(true);
    try {
      await apiCall('/budgets', 'POST', {
        amount: parseFloat(budgetForm.amount),
        period: budgetForm.period,
        start_date: budgetForm.start_date || null,
        end_date: budgetForm.end_date || null,
        alerts_enabled: budgetForm.alerts_enabled,
      });
      setBudgetForm(initialBudgetForm);
      await fetchDashboardData();
      setCurrentView('dashboard');
    } catch (err) {
      alert(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleAddTransaction = async () => {
    setLoading(true);
    try {
      await apiCall('/transactions', 'POST', {
        amount: parseFloat(transactionForm.amount),
        category: transactionForm.category,
        transaction_type: transactionForm.transaction_type,
        occurred_at: transactionForm.occurred_at,
        notes: transactionForm.notes || null,
      });
      setTransactionForm(initialTransactionForm);
      await fetchDashboardData();
      setCurrentView('dashboard');
    } catch (err) {
      alert(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleGetForecast = async () => {
    setLoading(true);
    try {
      const data = await apiCall('/analytics/forecast', 'POST', {
        category: forecastCategory.toLowerCase(),
        periods: 3,
        use_user_data: true,
      });
      setForecastDetails(data.forecasts ?? []);
    } catch (err) {
      alert('Failed to get forecast: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = () => {
    setToken(null);
    setUser(null);
    setCurrentView('login');
    setBudgets([]);
    setTransactions([]);
    setAnalytics(null);
    setForecastDetails([]);
  };

  if (!token || currentView === 'login' || currentView === 'register') {
    return (
      <div className="min-h-screen bg-gradient-to-br from-purple-50 via-pink-50 to-blue-50 flex items-center justify-center p-4">
        <div className="bg-white rounded-3xl shadow-2xl p-8 w-full max-w-md">
          <div className="text-center mb-8">
            <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-r from-purple-500 to-pink-500 rounded-2xl mb-4">
              <Wallet className="w-8 h-8 text-white" />
            </div>
            <h1 className="text-3xl font-bold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">
              Smart Budget Tracker
            </h1>
            <p className="text-gray-500 mt-2">Track your finances with ease</p>
          </div>

          <div className="space-y-4">
            {currentView === 'register' && (
              <input
                type="text"
                placeholder="Full Name"
                value={authForm.full_name}
                onChange={(e) => setAuthForm({ ...authForm, full_name: e.target.value })}
                className="w-full px-4 py-3 bg-gray-50 border-2 border-gray-100 rounded-xl focus:border-purple-500 focus:outline-none"
                required
              />
            )}
            <input
              type="email"
              placeholder="Email"
              value={authForm.email}
              onChange={(e) => setAuthForm({ ...authForm, email: e.target.value })}
              className="w-full px-4 py-3 bg-gray-50 border-2 border-gray-100 rounded-xl focus:border-purple-500 focus:outline-none"
              required
            />
            <input
              type="password"
              placeholder="Password"
              value={authForm.password}
              onChange={(e) => setAuthForm({ ...authForm, password: e.target.value })}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  currentView === 'login' ? handleLogin() : handleRegister();
                }
              }}
              className="w-full px-4 py-3 bg-gray-50 border-2 border-gray-100 rounded-xl focus:border-purple-500 focus:outline-none"
              required
            />
            <button
              onClick={currentView === 'login' ? handleLogin : handleRegister}
              disabled={loading}
              className="w-full py-3 bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-xl font-semibold hover:shadow-lg transition disabled:opacity-50"
            >
              {loading ? 'Processing…' : currentView === 'login' ? 'Login' : 'Register'}
            </button>
          </div>

          <div className="text-center mt-6">
            <button
              onClick={() => setCurrentView(currentView === 'login' ? 'register' : 'login')}
              className="text-purple-600 hover:text-purple-700 font-medium"
            >
              {currentView === 'login' ? 'Create an account' : 'Already have an account?'}
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-pink-50 to-blue-50">
      <header className="bg-white border-b border-gray-100 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-gradient-to-r from-purple-500 to-pink-500 rounded-xl flex items-center justify-center">
              <Wallet className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">
                Smart Budget Tracker
              </h1>
              {user && <p className="text-sm text-gray-500">{user.email}</p>}
            </div>
          </div>
          <button onClick={handleLogout} className="flex items-center space-x-2 px-4 py-2 text-gray-600 hover:bg-gray-50 rounded-xl">
            <LogOut className="w-5 h-5" />
            <span>Logout</span>
          </button>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 py-6 space-y-6">
        <nav className="flex space-x-2 overflow-x-auto pb-2">
          {navItems.map((view) => (
            <button
              key={view}
              onClick={() => setCurrentView(view)}
              className={`px-6 py-2 rounded-xl font-medium whitespace-nowrap transition ${
                currentView === view
                  ? 'bg-gradient-to-r from-purple-500 to-pink-500 text-white shadow-lg'
                  : 'bg-white text-gray-600 hover:bg-gray-50'
              }`}
            >
              {view.charAt(0).toUpperCase() + view.slice(1)}
            </button>
          ))}
        </nav>

        {currentView === 'dashboard' && (
          <DashboardView analytics={analytics} transactions={transactions} />
        )}

        {currentView === 'budgets' && (
          <BudgetsView
            budgets={budgets}
            budgetForm={budgetForm}
            setBudgetForm={setBudgetForm}
            loading={loading}
            onSubmit={handleAddBudget}
          />
        )}

        {currentView === 'transactions' && (
          <TransactionsView
            transactions={transactions}
            transactionForm={transactionForm}
            setTransactionForm={setTransactionForm}
            loading={loading}
            onSubmit={handleAddTransaction}
          />
        )}

        {currentView === 'analytics' && <AnalyticsView analytics={analytics} />}

        {currentView === 'forecast' && (
          <ForecastView
            loading={loading}
            forecastCategory={forecastCategory}
            setForecastCategory={setForecastCategory}
            forecastDetails={forecastDetails}
            onGenerate={handleGetForecast}
          />
        )}
      </div>
    </div>
  );
}

function DashboardView({ analytics, transactions }) {
  if (!analytics) {
    return (
      <div className="bg-white rounded-2xl p-6 shadow-lg text-gray-500">No analytics yet.</div>
    );
  }

  const { summary, insights } = analytics;

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <SummaryCard
          label="Total Income"
          value={summary?.total_income}
          icon={<TrendingUp className="w-5 h-5 text-green-500" />}
          color="text-green-600"
        />
        <SummaryCard
          label="Total Expenses"
          value={summary?.total_expense}
          icon={<TrendingDown className="w-5 h-5 text-red-500" />}
          color="text-red-600"
        />
        <SummaryCard
          label="Net Balance"
          value={summary?.net}
          icon={<DollarSign className="w-5 h-5 text-blue-500" />}
          color="text-blue-600"
        />
      </div>

      {insights?.top_categories?.length > 0 && (
        <div className="bg-white rounded-2xl p-6 shadow-lg">
          <h2 className="text-xl font-bold mb-4 flex items-center">
            <PieChart className="w-6 h-6 mr-2 text-purple-500" />
            Top Spending Categories
          </h2>
          <div className="space-y-3">
            {insights.top_categories.map((cat, idx) => (
              <div key={cat.category} className="flex items-center justify-between p-3 bg-gray-50 rounded-xl">
                <div className="flex items-center space-x-3">
                  <div className="w-8 h-8 bg-gradient-to-r from-purple-400 to-pink-400 rounded-lg flex items-center justify-center text-white font-bold">
                    {idx + 1}
                  </div>
                  <span className="font-medium capitalize">{cat.category}</span>
                </div>
                <span className="font-bold text-purple-600">${Number(cat.amount).toFixed(2)}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="bg-white rounded-2xl p-6 shadow-lg">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold flex items-center">
            <FileText className="w-6 h-6 mr-2 text-purple-500" />
            Recent Transactions
          </h2>
          <span className="text-sm text-gray-500">Showing latest 5</span>
        </div>
        <div className="space-y-2">
          {transactions.slice(0, 5).map((txn) => (
            <div key={txn.id} className="flex items-center justify-between p-4 bg-gray-50 rounded-xl">
              <div className="flex items-center space-x-3">
                <div
                  className={`w-10 h-10 rounded-xl flex items-center justify-center ${
                    txn.transaction_type === 'income' ? 'bg-green-100' : 'bg-red-100'
                  }`}
                >
                  {txn.transaction_type === 'income' ? (
                    <TrendingUp className="w-5 h-5 text-green-600" />
                  ) : (
                    <TrendingDown className="w-5 h-5 text-red-600" />
                  )}
                </div>
                <div>
                  <p className="font-medium capitalize">{txn.category}</p>
                  <p className="text-sm text-gray-500">{new Date(txn.occurred_at).toLocaleDateString()}</p>
                </div>
              </div>
              <span className={`font-bold ${txn.transaction_type === 'income' ? 'text-green-600' : 'text-red-600'}`}>
                {txn.transaction_type === 'income' ? '+' : '-'}${Number(txn.amount).toFixed(2)}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function SummaryCard({ label, value, icon, color }) {
  return (
    <div className="bg-white rounded-2xl p-6 shadow-lg">
      <div className="flex items-center justify-between mb-2">
        <span className="text-gray-500">{label}</span>
        {icon}
      </div>
      <p className={`text-3xl font-bold ${color}`}>${Number(value ?? 0).toFixed(2)}</p>
    </div>
  );
}

function BudgetsView({ budgets, budgetForm, setBudgetForm, loading, onSubmit }) {
  return (
    <div className="space-y-6">
      <div className="bg-white rounded-2xl p-6 shadow-lg">
        <h2 className="text-xl font-bold mb-4">Add New Budget</h2>
        <div className="grid gap-4 md:grid-cols-2">
          <input
            type="number"
            step="0.01"
            placeholder="Amount"
            value={budgetForm.amount}
            onChange={(e) => setBudgetForm({ ...budgetForm, amount: e.target.value })}
            className="w-full px-4 py-3 bg-gray-50 border-2 border-gray-100 rounded-xl focus:border-purple-500"
            required
          />
          <select
            value={budgetForm.period}
            onChange={(e) => setBudgetForm({ ...budgetForm, period: e.target.value })}
            className="w-full px-4 py-3 bg-gray-50 border-2 border-gray-100 rounded-xl focus:border-purple-500"
          >
            <option value="weekly">Weekly</option>
            <option value="monthly">Monthly</option>
          </select>
          <input
            type="date"
            value={budgetForm.start_date}
            onChange={(e) => setBudgetForm({ ...budgetForm, start_date: e.target.value })}
            className="w-full px-4 py-3 bg-gray-50 border-2 border-gray-100 rounded-xl focus:border-purple-500"
          />
          <input
            type="date"
            value={budgetForm.end_date}
            onChange={(e) => setBudgetForm({ ...budgetForm, end_date: e.target.value })}
            className="w-full px-4 py-3 bg-gray-50 border-2 border-gray-100 rounded-xl focus:border-purple-500"
          />
          <label className="flex items-center space-x-3">
            <input
              type="checkbox"
              checked={budgetForm.alerts_enabled}
              onChange={(e) => setBudgetForm({ ...budgetForm, alerts_enabled: e.target.checked })}
              className="w-5 h-5 text-purple-600"
            />
            <span className="text-gray-700">Enable Alerts</span>
          </label>
          <button
            onClick={onSubmit}
            disabled={loading}
            className="w-full py-3 bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-xl font-semibold hover:shadow-lg"
          >
            {loading ? 'Saving…' : 'Add Budget'}
          </button>
        </div>
      </div>

      <div className="bg-white rounded-2xl p-6 shadow-lg">
        <h2 className="text-xl font-bold mb-4">Your Budgets</h2>
        {budgets.length === 0 && <p className="text-gray-500">No budgets yet.</p>}
        <div className="space-y-3">
          {budgets.map((budget) => (
            <div key={budget.id} className="p-4 bg-gray-50 rounded-xl">
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-bold text-lg">${Number(budget.amount).toFixed(2)}</p>
                  <p className="text-sm text-gray-500 capitalize">{budget.period}</p>
                  {budget.start_date && (
                    <p className="text-xs text-gray-500">
                      {budget.start_date} → {budget.end_date || 'ongoing'}
                    </p>
                  )}
                </div>
                {budget.alerts_enabled && (
                  <div className="flex items-center space-x-1 text-green-600">
                    <CheckCircle className="w-4 h-4" />
                    <span className="text-sm">Alerts On</span>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function TransactionsView({ transactions, transactionForm, setTransactionForm, loading, onSubmit }) {
  return (
    <div className="space-y-6">
      <div className="bg-white rounded-2xl p-6 shadow-lg">
        <h2 className="text-xl font-bold mb-4">Add Transaction</h2>
        <div className="grid gap-4 md:grid-cols-2">
          <input
            type="number"
            step="0.01"
            placeholder="Amount"
            value={transactionForm.amount}
            onChange={(e) => setTransactionForm({ ...transactionForm, amount: e.target.value })}
            className="w-full px-4 py-3 bg-gray-50 border-2 border-gray-100 rounded-xl"
            required
          />
          <select
            value={transactionForm.category}
            onChange={(e) => setTransactionForm({ ...transactionForm, category: e.target.value })}
            className="w-full px-4 py-3 bg-gray-50 border-2 border-gray-100 rounded-xl"
          >
            {categories.map((cat) => (
              <option key={cat} value={cat.toLowerCase()}>
                {cat}
              </option>
            ))}
          </select>
          <select
            value={transactionForm.transaction_type}
            onChange={(e) => setTransactionForm({ ...transactionForm, transaction_type: e.target.value })}
            className="w-full px-4 py-3 bg-gray-50 border-2 border-gray-100 rounded-xl"
          >
            <option value="expense">Expense</option>
            <option value="income">Income</option>
          </select>
          <input
            type="date"
            value={transactionForm.occurred_at}
            onChange={(e) => setTransactionForm({ ...transactionForm, occurred_at: e.target.value })}
            className="w-full px-4 py-3 bg-gray-50 border-2 border-gray-100 rounded-xl"
            required
          />
          <textarea
            placeholder="Notes (optional)"
            value={transactionForm.notes}
            onChange={(e) => setTransactionForm({ ...transactionForm, notes: e.target.value })}
            rows={3}
            className="w-full px-4 py-3 bg-gray-50 border-2 border-gray-100 rounded-xl md:col-span-2"
          />
          <button
            onClick={onSubmit}
            disabled={loading}
            className="w-full py-3 bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-xl font-semibold hover:shadow-lg"
          >
            {loading ? 'Saving…' : 'Add Transaction'}
          </button>
        </div>
      </div>

      <div className="bg-white rounded-2xl p-6 shadow-lg">
        <h2 className="text-xl font-bold mb-4">All Transactions</h2>
        {transactions.length === 0 && <p className="text-gray-500">No transactions yet.</p>}
        <div className="space-y-2">
          {transactions.map((txn) => (
            <div key={txn.id} className="flex items-center justify-between p-4 bg-gray-50 rounded-xl">
              <div>
                <p className="font-medium capitalize">{txn.category}</p>
                <p className="text-sm text-gray-500">{new Date(txn.occurred_at).toLocaleDateString()}</p>
                {txn.notes && <p className="text-sm text-gray-600 mt-1">{txn.notes}</p>}
              </div>
              <span className={`font-bold ${txn.transaction_type === 'income' ? 'text-green-600' : 'text-red-600'}`}>
                {txn.transaction_type === 'income' ? '+' : '-'}${Number(txn.amount).toFixed(2)}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function AnalyticsView({ analytics }) {
  if (!analytics) {
    return <div className="bg-white rounded-2xl p-6 shadow-lg text-gray-500">No analytics yet.</div>;
  }
  const { insights, summary } = analytics;
  return (
    <div className="space-y-6">
      <div className="bg-white rounded-2xl p-6 shadow-lg">
        <h2 className="text-xl font-bold mb-4">Financial Insights</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="p-4 bg-purple-50 rounded-xl">
            <p className="text-sm text-gray-600 mb-1">Average Monthly Spend</p>
            <p className="text-2xl font-bold text-purple-600">
              ${Number(insights?.average_monthly ?? 0).toFixed(2)}
            </p>
          </div>
          <div className="p-4 bg-pink-50 rounded-xl">
            <p className="text-sm text-gray-600 mb-1">Spending Trend</p>
            <p className="text-2xl font-bold text-pink-600 capitalize">{insights?.trend ?? 'stable'}</p>
          </div>
        </div>
      </div>

      <div className="bg-white rounded-2xl p-6 shadow-lg grid grid-cols-1 md:grid-cols-3 gap-4">
        <InsightCard title="Total Spending" value={insights?.total_spending} icon={<DollarSign />} />
        <InsightCard title="Period (months)" value={insights?.period_months} icon={<Calendar />} suffix="mo" />
        <InsightCard title="Net" value={summary?.net} icon={<BarChart3 />} />
      </div>
    </div>
  );
}

function InsightCard({ title, value, icon, suffix }) {
  return (
    <div className="p-4 bg-gray-50 rounded-xl">
      <div className="flex items-center space-x-2 text-gray-500 mb-1">
        {icon}
        <span>{title}</span>
      </div>
      <p className="text-2xl font-bold text-gray-900">
        {suffix ? `${Number(value ?? 0).toFixed(0)} ${suffix}` : `$${Number(value ?? 0).toFixed(2)}`}
      </p>
    </div>
  );
}

function ForecastView({ loading, forecastCategory, setForecastCategory, forecastDetails, onGenerate }) {
  return (
    <div className="space-y-6">
      <div className="bg-white rounded-2xl p-6 shadow-lg">
        <h2 className="text-xl font-bold mb-4">Spending Forecast</h2>
        <div className="space-y-4">
          <select
            value={forecastCategory}
            onChange={(e) => setForecastCategory(e.target.value)}
            className="w-full px-4 py-3 bg-gray-50 border-2 border-gray-100 rounded-xl"
          >
            {categories.map((cat) => (
              <option key={cat} value={cat}>
                {cat}
              </option>
            ))}
          </select>
          <button
            onClick={onGenerate}
            disabled={loading}
            className="w-full py-3 bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-xl font-semibold hover:shadow-lg"
          >
            {loading ? 'Generating…' : 'Generate Forecast'}
          </button>
        </div>
      </div>

      {forecastDetails.length > 0 && (
        <div className="bg-white rounded-2xl p-6 shadow-lg">
          <h3 className="text-lg font-bold mb-4">Forecast Results for {forecastCategory}</h3>
          <div className="space-y-4">
            {forecastDetails.map((model) => (
              <div key={model.model} className="p-4 bg-gray-50 rounded-xl">
                <div className="flex items-center justify-between mb-2">
                  <div className="font-semibold capitalize">{model.model.replace(/_/g, ' ')}</div>
                  <span className="text-sm text-gray-500">{model.data_source}</span>
                </div>
                {model.forecast && model.forecast.length > 0 ? (
                  <div>
                    <p className="text-sm text-gray-600 mb-2">Next periods forecast:</p>
                    <div className="flex flex-wrap gap-2">
                      {model.forecast.map((val, idx) => (
                        <div key={`${model.model}-${idx}`} className="px-3 py-1 bg-white rounded-lg text-sm font-medium">
                          ${Number(val).toFixed(2)}
                        </div>
                      ))}
                    </div>
                  </div>
                ) : (
                  <p className="text-sm text-gray-500">No forecast available.</p>
                )}
                {model.notes && <p className="text-sm text-gray-600 mt-2">{model.notes}</p>}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
