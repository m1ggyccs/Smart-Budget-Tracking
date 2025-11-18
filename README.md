# Smart Budget Tracking Dashboard

AI-powered budget forecasting dashboard with multi-model analysis (Moving Average, Holt-Winters, LSTM).

## Features

- **Multi-Model Forecasting**: Combines Moving Average, Holt-Winters, and LSTM models
- **Ensemble Predictions**: Weighted ensemble for improved accuracy
- **Interactive Dashboard**: Real-time predictions with confidence intervals
- **Category Analysis**: View predictions by spending category
- **Budget Tracking**: Compare predictions against monthly budget

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Ensure you have processed data:
- Run `processing.py` to preprocess raw transactions
- Run `training.py` to train models for each category

### 3. Start the API Server

```bash
python api.py
```

The API will be available at `http://localhost:8001`

### 4. Frontend Setup

The frontend is a React component. To use it:

1. Set up a React project (e.g., using Vite, Create React App, or Next.js)
2. Copy the `frontend` file content into your React component
3. Install required dependencies:
   ```bash
   npm install recharts lucide-react
   ```
4. Ensure Tailwind CSS is configured in your project

## Streamlit Dashboard

Prefer a ready-made dashboard without setting up the React frontend? Launch the Streamlit experience included in this repo:

```bash
streamlit run streamlit_app.py
```

Requirements:
- Processed data at `data/processed/cleaned_transactions.csv` (run `processing.py`)
- Trained models inside `models/` (run `training.py`)

Once running, open the provided local URL to interactively:
- Select forecast horizon (1-12 months) and monthly budget
- View ensemble vs budget summaries per month
- Drill into per-category model outputs
- Inspect predictive insights and download the raw CSV

## API Endpoints

### `GET /api/health`
Health check endpoint.

### `GET /api/categories`
Returns list of available spending categories.

### `POST /api/predictions`
Get predictions for all categories.

**Request Body:**
```json
{
  "months": 6,
  "budget": 25000
}
```

**Response:**
```json
{
  "predictions": {
    "category_name": {
      "category": "category_name",
      "predictions": {
        "ensemble": [...],
        "models": {
          "moving_average": [...],
          "holtwinters": [...],
          "lstm": [...]
        }
      }
    }
  },
  "budget_summary": {...},
  "total_budget": 25000,
  "months": 6
}
```

### `POST /api/insights`
Get predictive insights and risk analysis.

**Request Body:**
```json
{
  "months": 6
}
```

## Usage

1. **Start the API**: Run `python api.py` in the project root
2. **Open Frontend**: The frontend will automatically connect to `http://localhost:5000`
3. **Configure Inputs**:
   - Set "Months Ahead" (1-12) to determine forecast horizon
   - Set "Monthly Budget" (₱) for budget comparison
   - Click "Refresh Predictions" to update
4. **View Predictions**: Select a category to see detailed forecasts

## Input Parameters (from simulation.py)

- **Months Ahead**: Number of months to predict (1-12)
- **Monthly Budget**: Your monthly budget in PHP (₱)

These match the inputs from `simulation.py`'s CLI interface.

## Project Structure

```
Smart-Budget-Tracking/
├── api.py                 # Flask API server
├── frontend               # React dashboard component
├── simulation.py          # Prediction logic
├── training.py            # Model training
├── processing.py          # Data preprocessing
├── data/
│   ├── raw/              # Raw transaction data
│   └── processed/        # Cleaned data
├── models/               # Trained model files
└── outputs/             # Prediction outputs
```

## Notes

- Ensure models are trained before using the API
- The frontend expects the API to be running on `http://localhost:8001`
- CORS is enabled for local development
- Confidence intervals are calculated based on model variance

