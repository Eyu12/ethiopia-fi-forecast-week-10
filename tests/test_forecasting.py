"""
Tests for Financial Inclusion Forecasting module.
"""

import os
import sys
from pathlib import Path
PROJECT_ROOT = Path().resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import pytest
from src.forecasting import FinancialInclusionForecaster

def create_test_forecasting_data():
    """Create test data for forecasting."""
    # Create time series data
    dates = pd.date_range(start='2011-01-01', end='2024-01-01', freq='3Y')
    
    # Account ownership data (mimicking Ethiopia)
    acc_ownership = pd.Series([14, 22, 35, 46, 49], index=dates)
    
    # Create DataFrame
    data = []
    
    for date, value in acc_ownership.items():
        data.append({
            'record_type': 'observation',
            'indicator_code': 'ACC_OWNERSHIP',
            'value_numeric': value,
            'observation_date': date
        })
    
    # Add an event
    data.append({
        'record_type': 'event',
        'event_name': 'Test Event',
        'event_date': '2023-01-01',
        'category': 'test'
    })
    
    return pd.DataFrame(data)

def test_forecaster_initialization():
    """Test FinancialInclusionForecaster initialization."""
    test_data = create_test_forecasting_data()
    forecaster = FinancialInclusionForecaster(test_data)
    
    assert forecaster is not None
    assert forecaster.data is not None
    assert 'ACC_OWNERSHIP' in forecaster.time_series
    assert len(forecaster.time_series['ACC_OWNERSHIP']) == 5

def test_fit_trend_model():
    """Test trend model fitting."""
    test_data = create_test_forecasting_data()
    forecaster = FinancialInclusionForecaster(test_data)
    
    # Test linear model
    model = forecaster.fit_trend_model('ACC_OWNERSHIP', 'linear')
    
    assert model is not None
    assert 'model_type' in model
    assert model['model_type'] == 'linear'
    assert 'slope' in model
    assert 'intercept' in model
    assert 'rmse' in model
    assert callable(model['predict_function'])

def test_select_best_trend_model():
    """Test best trend model selection."""
    test_data = create_test_forecasting_data()
    forecaster = FinancialInclusionForecaster(test_data)
    
    best_model = forecaster.select_best_trend_model('ACC_OWNERSHIP')
    
    assert best_model is not None
    assert 'model_type' in best_model
    assert 'rmse' in best_model
    assert best_model['rmse'] >= 0

def test_generate_forecast():
    """Test forecast generation."""
    test_data = create_test_forecasting_data()
    forecaster = FinancialInclusionForecaster(test_data)
    
    forecast = forecaster.generate_forecast('ACC_OWNERSHIP', [2025, 2026])
    
    assert forecast is not None
    assert 'indicator' in forecast
    assert forecast['indicator'] == 'ACC_OWNERSHIP'
    assert 'annual_forecasts' in forecast
    assert 2025 in forecast['annual_forecasts']
    assert 2026 in forecast['annual_forecasts']
    assert 'final_forecast' in forecast
    assert len(forecast['final_forecast']) > 0

def test_create_scenarios():
    """Test scenario creation."""
    test_data = create_test_forecasting_data()
    forecaster = FinancialInclusionForecaster(test_data)
    
    scenarios = forecaster.create_scenarios('ACC_OWNERSHIP', [2025])
    
    assert scenarios is not None
    assert 'optimistic' in scenarios
    assert 'base' in scenarios
    assert 'pessimistic' in scenarios
    
    for scenario in ['optimistic', 'base', 'pessimistic']:
        assert 'annual_forecasts' in scenarios[scenario]
        assert 2025 in scenarios[scenario]['annual_forecasts']

def test_visualize_forecast():
    """Test forecast visualization (no errors)."""
    test_data = create_test_forecasting_data()
    forecaster = FinancialInclusionForecaster(test_data)
    
    # Generate forecast first
    forecaster.generate_forecast('ACC_OWNERSHIP', [2025])
    
    # Test visualization doesn't crash
    try:
        forecaster.visualize_forecast('ACC_OWNERSHIP', show_scenarios=False)
        assert True
    except Exception as e:
        pytest.fail(f"Visualization failed: {e}")

if __name__ == '__main__':
    # Run tests
    test_forecaster_initialization()
    print("✓ test_forecaster_initialization passed")
    
    test_fit_trend_model()
    print("✓ test_fit_trend_model passed")
    
    test_select_best_trend_model()
    print("✓ test_select_best_trend_model passed")
    
    test_generate_forecast()
    print("✓ test_generate_forecast passed")
    
    test_create_scenarios()
    print("✓ test_create_scenarios passed")
    
    test_visualize_forecast()
    print("✓ test_visualize_forecast passed")
    
    print("\nAll forecasting tests passed!")