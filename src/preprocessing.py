
"""
Data preprocessing utilities for financial inclusion forecasting.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
from typing import Dict, List, Optional, Tuple, Union


class DataPreprocessor:
    """Preprocess financial inclusion data for analysis and modeling."""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize preprocessor with raw data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Raw financial inclusion data
        """
        self.data = data.copy()
        self.processed_data = None
        
    def clean_dates(self) -> None:
        """Clean and standardize date columns."""
        date_columns = ['observation_date', 'event_date']
        
        for col in date_columns:
            if col in self.data.columns:
                self.data[col] = pd.to_datetime(self.data[col], errors='coerce')
        
        # Extract year from dates
        if 'observation_date' in self.data.columns:
            self.data['year'] = self.data['observation_date'].dt.year
        if 'event_date' in self.data.columns:
            self.data['event_year'] = self.data['event_date'].dt.year
    
    def handle_missing_values(self, strategy: str = 'forward_fill') -> None:
        """
        Handle missing values in the dataset.
        
        Parameters:
        -----------
        strategy : str
            Strategy for handling missing values:
            - 'forward_fill': Forward fill within groups
            - 'interpolate': Linear interpolation
            - 'drop': Drop rows with missing values
        """
        if strategy == 'forward_fill':
            # Group by indicator and forward fill numeric values
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col in ['value_numeric']:
                    self.data[col] = self.data.groupby('indicator_code')[col].ffill()
        
        elif strategy == 'interpolate':
            # Linear interpolation for time series data
            self.data = self.data.sort_values(['indicator_code', 'observation_date'])
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            
            for indicator in self.data['indicator_code'].unique():
                mask = self.data['indicator_code'] == indicator
                for col in numeric_cols:
                    if col in self.data.columns:
                        self.data.loc[mask, col] = self.data.loc[mask, col].interpolate(
                            method='linear', limit_direction='forward'
                        )
        
        elif strategy == 'drop':
            # Drop rows with missing critical values
            critical_cols = ['indicator_code', 'value_numeric', 'observation_date']
            self.data = self.data.dropna(subset=critical_cols)
    
    def standardize_indicators(self) -> None:
        """Standardize indicator names and codes."""
        # Clean indicator names
        if 'indicator' in self.data.columns:
            self.data['indicator'] = self.data['indicator'].str.strip()
            self.data['indicator'] = self.data['indicator'].str.replace('\s+', ' ', regex=True)
        
        # Create standardized indicator categories
        def categorize_indicator(indicator_code: str) -> str:
            """Categorize indicators into broader groups."""
            if pd.isna(indicator_code):
                return 'other'
            
            indicator_code = str(indicator_code).upper()
            
            if indicator_code.startswith('ACC_'):
                return 'access'
            elif indicator_code.startswith('USG_'):
                return 'usage'
            elif indicator_code.startswith('EN_'):
                return 'enabler'
            elif indicator_code.startswith('INF_'):
                return 'infrastructure'
            else:
                return 'other'
        
        self.data['indicator_category'] = self.data['indicator_code'].apply(categorize_indicator)
    
    def validate_data_quality(self) -> Dict:
        """
        Validate data quality and return quality metrics.
        
        Returns:
        --------
        Dict with quality metrics
        """
        quality_metrics = {
            'total_records': len(self.data),
            'missing_values': {},
            'data_types': {},
            'value_ranges': {},
            'confidence_distribution': {}
        }
        
        # Check missing values
        for col in self.data.columns:
            missing_count = self.data[col].isna().sum()
            missing_pct = (missing_count / len(self.data)) * 100
            quality_metrics['missing_values'][col] = {
                'count': int(missing_count),
                'percentage': float(round(missing_pct, 2))
            }
        
        # Check data types
        for col in self.data.columns:
            quality_metrics['data_types'][col] = str(self.data[col].dtype)
        
        # Check numeric value ranges
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in self.data.columns:
                quality_metrics['value_ranges'][col] = {
                    'min': float(self.data[col].min()),
                    'max': float(self.data[col].max()),
                    'mean': float(self.data[col].mean()),
                    'std': float(self.data[col].std())
                }
        
        # Check confidence distribution
        if 'confidence' in self.data.columns:
            confidence_dist = self.data['confidence'].value_counts(normalize=True) * 100
            quality_metrics['confidence_distribution'] = confidence_dist.round(2).to_dict()
        
        return quality_metrics
    
    def create_time_series_format(self, indicator_codes: List[str] = None) -> pd.DataFrame:
        """
        Convert data to time series format for analysis.
        
        Parameters:
        -----------
        indicator_codes : List[str], optional
            Specific indicator codes to include
            
        Returns:
        --------
        pd.DataFrame in time series format
        """
        # Filter observations
        obs_data = self.data[self.data['record_type'] == 'observation'].copy()
        
        # Filter specific indicators if provided
        if indicator_codes:
            obs_data = obs_data[obs_data['indicator_code'].isin(indicator_codes)]
        
        # Pivot to time series format
        ts_data = obs_data.pivot_table(
            index='observation_date',
            columns='indicator_code',
            values='value_numeric',
            aggfunc='mean'
        )
        
        # Sort by date
        ts_data = ts_data.sort_index()
        
        # Forward fill missing values for time series
        ts_data = ts_data.ffill().bfill()
        
        return ts_data
    
    def extract_events_timeline(self) -> pd.DataFrame:
        """
        Extract and format events for timeline analysis.
        
        Returns:
        --------
        pd.DataFrame with event information
        """
        if 'event' not in self.data['record_type'].values:
            return pd.DataFrame()
        
        events = self.data[self.data['record_type'] == 'event'].copy()
        
        # Select relevant columns
        event_cols = ['event_name', 'event_date', 'category', 'source_name', 'confidence']
        event_cols = [col for col in event_cols if col in events.columns]
        
        events_timeline = events[event_cols].copy()
        events_timeline = events_timeline.sort_values('event_date')
        
        return events_timeline
    
    def calculate_growth_metrics(self, indicator_code: str) -> pd.DataFrame:
        """
        Calculate growth metrics for a specific indicator.
        
        Parameters:
        -----------
        indicator_code : str
            Indicator code to analyze
            
        Returns:
        --------
        pd.DataFrame with growth metrics
        """
        # Filter data for indicator
        indicator_data = self.data[
            (self.data['indicator_code'] == indicator_code) & 
            (self.data['record_type'] == 'observation')
        ].copy()
        
        if len(indicator_data) < 2:
            return pd.DataFrame()
        
        # Sort by date
        indicator_data = indicator_data.sort_values('observation_date')
        
        # Calculate growth metrics
        indicator_data['value_lag'] = indicator_data['value_numeric'].shift(1)
        indicator_data['absolute_growth'] = indicator_data['value_numeric'] - indicator_data['value_lag']
        indicator_data['growth_rate'] = (indicator_data['absolute_growth'] / indicator_data['value_lag']) * 100
        
        # Calculate time between observations
        indicator_data['days_between'] = indicator_data['observation_date'].diff().dt.days
        indicator_data['annualized_growth'] = (
            indicator_data['growth_rate'] * (365 / indicator_data['days_between'])
        )
        
        return indicator_data[['observation_date', 'value_numeric', 'absolute_growth', 
                              'growth_rate', 'annualized_growth']].dropna()
    
    def prepare_analysis_dataset(self) -> pd.DataFrame:
        """
        Prepare final dataset for analysis.
        
        Returns:
        --------
        pd.DataFrame ready for analysis
        """
        # Apply all preprocessing steps
        self.clean_dates()
        self.handle_missing_values(strategy='forward_fill')
        self.standardize_indicators()
        
        # Create analysis dataset with derived features
        analysis_data = self.data.copy()
        
        # Add derived features
        if 'value_numeric' in analysis_data.columns:
            # Normalize values for comparison
            numeric_indicators = analysis_data['indicator_code'].unique()
            for indicator in numeric_indicators:
                mask = analysis_data['indicator_code'] == indicator
                values = analysis_data.loc[mask, 'value_numeric']
                if len(values) > 1:
                    analysis_data.loc[mask, 'value_normalized'] = (
                        (values - values.mean()) / values.std()
                    )
        
        # Add time-based features
        if 'observation_date' in analysis_data.columns:
            analysis_data['year'] = analysis_data['observation_date'].dt.year
            analysis_data['month'] = analysis_data['observation_date'].dt.month
            analysis_data['quarter'] = analysis_data['observation_date'].dt.quarter
        
        self.processed_data = analysis_data
        return analysis_data
    
    def save_processed_data(self, output_path: str) -> None:
        """
        Save processed data to file.
        
        Parameters:
        -----------
        output_path : str
            Path to save processed data
        """
        if self.processed_data is not None:
            self.processed_data.to_csv(output_path, index=False)
            print(f"Processed data saved to {output_path}")
        else:
            print("No processed data available. Run prepare_analysis_dataset() first.")