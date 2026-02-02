"""
Tests for Event Impact Modeling module.
"""

import os
import sys
from pathlib import Path
PROJECT_ROOT = Path().resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import pytest
from src.impact_model import EventImpactModeler

def create_test_data():
    """Create test data for impact modeling."""
    # Create events
    events_data = {
        'record_type': ['event', 'event', 'event'],
        'event_name': ['Telebirr Launch', 'M-Pesa Launch', 'Interoperability'],
        'event_date': ['2021-05-01', '2023-08-01', '2024-01-01'],
        'category': ['product_launch', 'product_launch', 'infrastructure'],
        'event_id': ['EV001', 'EV002', 'EV003']
    }
    
    # Create impact links
    impact_data = {
        'record_type': ['impact_link', 'impact_link', 'impact_link'],
        'parent_id': ['EV001', 'EV001', 'EV002'],
        'related_indicator': ['ACC_MM_ACCOUNT', 'ACC_OWNERSHIP', 'ACC_MM_ACCOUNT'],
        'impact_direction': ['positive', 'positive', 'positive'],
        'impact_magnitude': [0.15, 0.10, 0.08],
        'lag_months': [12, 18, 12]
    }
    
    # Create observations
    obs_data = {
        'record_type': ['observation', 'observation', 'observation', 'observation'],
        'indicator_code': ['ACC_MM_ACCOUNT', 'ACC_MM_ACCOUNT', 'ACC_OWNERSHIP', 'ACC_OWNERSHIP'],
        'value_numeric': [4.7, 9.45, 46.0, 49.0],
        'observation_date': ['2021-01-01', '2024-01-01', '2021-01-01', '2024-01-01']
    }
    
    # Combine data
    events_df = pd.DataFrame(events_data)
    impact_df = pd.DataFrame(impact_data)
    obs_df = pd.DataFrame(obs_data)
    
    # Combine all data
    test_data = pd.concat([events_df, impact_df, obs_df], ignore_index=True)
    
    return test_data

def test_impact_modeler_initialization():
    """Test EventImpactModeler initialization."""
    test_data = create_test_data()
    modeler = EventImpactModeler(test_data)
    
    assert modeler is not None
    assert modeler.events is not None
    assert modeler.impact_links is not None
    assert len(modeler.events) == 3
    assert len(modeler.impact_links) == 3

def test_build_association_matrix():
    """Test association matrix building."""
    test_data = create_test_data()
    modeler = EventImpactModeler(test_data)
    
    matrix = modeler.build_event_indicator_matrix()
    
    assert matrix is not None
    assert not matrix.empty
    assert 'ACC_MM_ACCOUNT' in matrix.columns
    assert 'Telebirr Launch' in matrix.index.get_level_values(0)

def test_estimate_event_impact():
    """Test event impact estimation."""
    test_data = create_test_data()
    modeler = EventImpactModeler(test_data)
    
    impact = modeler.estimate_event_impact('Telebirr Launch', 'ACC_MM_ACCOUNT')
    
    assert impact is not None
    assert 'magnitude' in impact
    assert 'direction' in impact
    assert 'confidence' in impact
    assert impact['event_name'] == 'Telebirr Launch'
    assert impact['indicator'] == 'ACC_MM_ACCOUNT'

def test_calculate_historical_impact():
    """Test historical impact calculation."""
    test_data = create_test_data()
    modeler = EventImpactModeler(test_data)
    
    historical = modeler.calculate_historical_impact('Telebirr Launch', 'ACC_MM_ACCOUNT')
    
    assert historical is not None
    if 'error' not in historical:
        assert 'observed_impact' in historical
        assert 'pre_event' in historical
        assert 'post_event' in historical

def test_create_impact_function():
    """Test impact function creation."""
    test_data = create_test_data()
    modeler = EventImpactModeler(test_data)
    
    impact_func = modeler.create_impact_function('Telebirr Launch', 'ACC_MM_ACCOUNT', 'sigmoid')
    
    assert callable(impact_func)
    
    # Test function at different time points
    assert impact_func(0) >= 0
    assert impact_func(12) > impact_func(0)  # Should increase over time

def test_simulate_composite_impact():
    """Test composite impact simulation."""
    test_data = create_test_data()
    modeler = EventImpactModeler(test_data)
    
    events = ['Telebirr Launch', 'M-Pesa Launch']
    composite = modeler.simulate_composite_impact(events, 'ACC_MM_ACCOUNT', time_horizon=24)
    
    assert composite is not None
    assert not composite.empty
    assert 'composite_impact' in composite.columns
    assert len(composite) == 24

def test_validation_against_history():
    """Test model validation against historical data."""
    test_data = create_test_data()
    modeler = EventImpactModeler(test_data)
    
    validation = modeler.validate_model_against_history()
    
    assert validation is not None
    if not validation.empty:
        assert 'event' in validation.columns
        assert 'historical_impact' in validation.columns
        assert 'model_estimate' in validation.columns

if __name__ == '__main__':
    # Run tests
    test_impact_modeler_initialization()
    print("✓ test_impact_modeler_initialization passed")
    
    test_build_association_matrix()
    print("✓ test_build_association_matrix passed")
    
    test_estimate_event_impact()
    print("✓ test_estimate_event_impact passed")
    
    test_calculate_historical_impact()
    print("✓ test_calculate_historical_impact passed")
    
    test_create_impact_function()
    print("✓ test_create_impact_function passed")
    
    test_simulate_composite_impact()
    print("✓ test_simulate_composite_impact passed")
    
    test_validation_against_history()
    print("✓ test_validation_against_history passed")
    
    print("\nAll tests passed!")