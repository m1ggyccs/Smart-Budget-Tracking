"""
Flask API server for Smart Budget Tracking Dashboard
Provides endpoints for predictions and insights
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings

# Import simulation functions
from simulation import (
    load_cleaned_data,
    monthly_series,
    load_moving_average,
    load_holt_winters,
    load_lstm_and_scaler,
    forecast_moving_average,
    forecast_holt_winters,
    forecast_lstm,
    best_model_for_category,
    safe_name
)

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = ROOT / "outputs"


def calculate_confidence_intervals(ensemble_pred, ma_pred, hw_pred, lstm_pred):
    """Calculate confidence intervals based on model variance"""
    # Convert to numpy arrays
    ensemble_arr = np.array(ensemble_pred)
    predictions = np.array([ma_pred, hw_pred, lstm_pred])
    std_dev = np.std(predictions, axis=0)
    mean_pred = np.mean(predictions, axis=0)
    
    # 95% confidence interval (approximately 2 standard deviations)
    ci_lower = mean_pred - 2 * std_dev
    ci_upper = mean_pred + 2 * std_dev
    
    # Ensure CI doesn't go negative and is reasonable
    ci_lower = np.maximum(ci_lower, ensemble_arr * 0.7)
    ci_upper = np.maximum(ci_upper, ensemble_arr * 1.3)
    
    return ci_lower.tolist(), ci_upper.tolist()


def format_predictions_for_api(df, months, budget):
    """Format predictions in the structure expected by the frontend - based on simulation.py logic"""
    categories = sorted(df["category"].unique())
    result = {}
    
    if len(categories) == 0:
        print("Warning: No categories found in data")
        return result
    
    for cat in categories:
        try:
            print(f"\n🔹 Predicting for category: {cat}")
            series = monthly_series(df, cat)
            if series.empty:
                print(f"  (no data for this category; skipping)")
                continue

            # Load models - same as simulation.py
            ma_params = load_moving_average(cat)
            hw_model = load_holt_winters(cat)
            lstm_model, scaler = load_lstm_and_scaler(cat)

            # Compute predictions - same as simulation.py
            ma_pred = forecast_moving_average(series, months, ma_params) if ma_params else [0.0] * months
            hw_pred = forecast_holt_winters(hw_model, months) if hw_model else [0.0] * months
            lstm_pred = forecast_lstm(series, months, lstm_model, scaler) if lstm_model and scaler else ma_pred

            # Ensemble (weighted) - same as simulation.py
            ensemble_pred = (0.5 * np.array(lstm_pred)) + (0.3 * np.array(hw_pred)) + (0.2 * np.array(ma_pred))
            ensemble_pred = [float(x) for x in ensemble_pred]
            
            # Calculate confidence intervals
            ci_lower, ci_upper = calculate_confidence_intervals(ensemble_pred, ma_pred, hw_pred, lstm_pred)
            
            # Generate period labels (YYYY-MM format)
            last_date = series.index[-1] if not series.empty else datetime.now()
            periods = []
            for i in range(months):
                next_month = last_date + pd.DateOffset(months=i+1)
                periods.append(next_month.strftime("%Y-%m"))
            
            # Format predictions for frontend
            ensemble_list = []
            ma_list = []
            hw_list = []
            lstm_list = []
            
            for i in range(months):
                ensemble_list.append({
                    "period": periods[i],
                    "value": float(ensemble_pred[i]),
                    "ci_lower": float(ci_lower[i]),
                    "ci_upper": float(ci_upper[i])
                })
                ma_list.append({
                    "period": periods[i],
                    "value": float(ma_pred[i])
                })
                hw_list.append({
                    "period": periods[i],
                    "value": float(hw_pred[i])
                })
                lstm_list.append({
                    "period": periods[i],
                    "value": float(lstm_pred[i])
                })
            
            result[cat] = {
                "category": cat,
                "predictions": {
                    "ensemble": ensemble_list,
                    "models": {
                        "moving_average": ma_list,
                        "holtwinters": hw_list,
                        "lstm": lstm_list
                    }
                }
            }
        except Exception as e:
            print(f"Error processing category {cat}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return result


@app.route('/', methods=['GET'])
def root():
    """Root endpoint with API information"""
    return jsonify({
        "message": "Smart Budget Tracking API",
        "version": "1.0",
        "endpoints": {
            "health": "/api/health",
            "categories": "/api/categories",
            "predictions": "/api/predictions (POST)",
            "insights": "/api/insights (POST)"
        },
        "status": "running"
    })


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "API is running"})


@app.route('/api/predictions', methods=['POST'])
def get_predictions():
    """Get predictions for all categories"""
    try:
        data = request.get_json() or {}
        months = int(data.get('months', 6))
        budget = float(data.get('budget', 25000.0))
        
        # Load data
        df = load_cleaned_data()
        
        # Generate predictions
        predictions = format_predictions_for_api(df, months, budget)
        
        # Calculate budget summary
        budget_summary = {}
        for cat, pred_data in predictions.items():
            ensemble = pred_data['predictions']['ensemble']
            if ensemble:
                current = ensemble[0]['value']
                next_month = ensemble[1]['value'] if len(ensemble) > 1 else current
                percentage_change = ((next_month - current) / current * 100) if current > 0 else 0
                
                budget_summary[cat] = {
                    "current": current,
                    "next_month": next_month,
                    "percentage_change": percentage_change,
                    "confidence_range": {
                        "lower": ensemble[0]['ci_lower'],
                        "upper": ensemble[0]['ci_upper']
                    }
                }
        
        return jsonify({
            "predictions": predictions,
            "budget_summary": budget_summary,
            "total_budget": budget,
            "months": months
        })
    
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in /api/predictions: {str(e)}")
        print(f"Traceback: {error_trace}")
        return jsonify({"error": str(e), "traceback": error_trace}), 500


@app.route('/api/categories', methods=['GET'])
def get_categories():
    """Get list of available categories"""
    try:
        df = load_cleaned_data()
        categories = sorted(df["category"].unique().tolist())
        return jsonify({"categories": categories})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/insights', methods=['POST'])
def get_insights():
    """Get predictive insights"""
    try:
        data = request.get_json() or {}
        months = int(data.get('months', 6))
        
        df = load_cleaned_data()
        predictions = format_predictions_for_api(df, months, 25000.0)
        
        insights = []
        for cat, pred_data in predictions.items():
            ensemble = pred_data['predictions']['ensemble']
            if not ensemble:
                continue
            
            # Calculate total predicted spend
            total_predicted = sum(p['value'] for p in ensemble)
            
            # Calculate average historical spending
            series = monthly_series(df, cat)
            avg_historical = float(series.mean()) if not series.empty else 0.0
            
            # Calculate percentage change
            if avg_historical > 0:
                pct_change = ((total_predicted / months - avg_historical) / avg_historical) * 100
            else:
                pct_change = 0.0
            
            # Determine risk level
            if avg_historical == 0:
                risk = "neutral"
                trend = "new category (no past data)"
            elif pct_change > 20:
                risk = "high"
                trend = f"Overspending risk (↑ {pct_change:.1f}%)"
            elif pct_change > -10:
                risk = "medium"
                trend = f"Stable spending ({pct_change:.1f}%)"
            else:
                risk = "low"
                trend = f"Improving (↓ {abs(pct_change):.1f}%)"
            
            insights.append({
                "category": cat,
                "past_avg": avg_historical,
                "predicted_total": total_predicted,
                "predicted_monthly": total_predicted / months,
                "trend": trend,
                "risk_level": risk,
                "percentage_change": pct_change
            })
        
        # Sort by predicted total
        insights.sort(key=lambda x: x['predicted_total'], reverse=True)
        
        return jsonify({"insights": insights})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("Starting Smart Budget Tracking API server...")
    print("API will be available at http://localhost:8001")
    app.run(debug=True, host='0.0.0.0', port=8001)

