# src/data_loader.py 
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os

class DataLoader:
    def __init__(self, raw_data_path: str = None, reference_codes_path: str = None):
        """
        Initialize DataLoader with file paths.
        
        Parameters:
        -----------
        raw_data_path : str, optional
            Path to raw data CSV
        reference_codes_path : str, optional
            Path to reference codes CSV
        """
        self.raw_data_path = raw_data_path
        self.reference_codes_path = reference_codes_path
        self.data = None
        self.reference_codes = None
        self.loaded = False
        
    def load_data(self, data_path: str = None) -> pd.DataFrame:
        """
        Load and validate the unified dataset.
        
        Parameters:
        -----------
        data_path : str, optional
            Alternative data path
            
        Returns:
        --------
        pd.DataFrame: Loaded data
        """
        if data_path:
            self.raw_data_path = data_path
        
        if not self.raw_data_path or not os.path.exists(self.raw_data_path):
            raise FileNotFoundError(f"Data file not found: {self.raw_data_path}")
        
        # Load data
        self.data = pd.read_csv(self.raw_data_path)
        
        # Load reference codes if available
        if self.reference_codes_path and os.path.exists(self.reference_codes_path):
            self.reference_codes = pd.read_csv(self.reference_codes_path)
        
        # Validate data structure
        self._validate_data()
        
        self.loaded = True
        return self.data
    
    def _validate_data(self) -> None:
        """Validate data structure and required columns."""
        required_columns = ['record_type', 'indicator', 'indicator_code']
        
        if not all(col in self.data.columns for col in required_columns):
            missing = [col for col in required_columns if col not in self.data.columns]
            raise ValueError(f"Missing required columns: {missing}")
        
        # Check record type values
        valid_record_types = ['observation', 'event', 'impact_link', 'target']
        invalid_records = self.data[~self.data['record_type'].isin(valid_record_types)]
        
        if len(invalid_records) > 0:
            print(f"Warning: {len(invalid_records)} records with invalid record_type")
    
    def get_record_counts(self) -> pd.DataFrame:
        """Count records by type and pillar."""
        if not self.loaded:
            self.load_data()
        
        if 'pillar' in self.data.columns:
            counts = self.data.groupby(['record_type', 'pillar']).size().unstack(fill_value=0)
        else:
            counts = self.data['record_type'].value_counts().to_frame('count')
        
        return counts
    
    def get_indicators_summary(self) -> pd.DataFrame:
        """Get summary of all indicators in the dataset."""
        if not self.loaded:
            self.load_data()
        
        indicators_summary = self.data.groupby('indicator_code').agg({
            'indicator': 'first',
            'record_type': lambda x: ', '.join(x.unique()),
            'value_numeric': ['count', 'mean', 'min', 'max'],
            'source_name': lambda x: ', '.join(x.dropna().unique()[:3])
        }).round(2)
        
        indicators_summary.columns = ['indicator_name', 'record_types', 
                                     'count', 'mean', 'min', 'max', 'sources']
        
        return indicators_summary
    
    def filter_by_pillar(self, pillar: str) -> pd.DataFrame:
        """Filter data by pillar."""
        if not self.loaded:
            self.load_data()
        
        if 'pillar' not in self.data.columns:
            raise ValueError("Pillar column not found in data")
        
        return self.data[self.data['pillar'] == pillar].copy()
    
    def filter_by_indicator(self, indicator_code: str) -> pd.DataFrame:
        """Filter data by indicator code."""
        if not self.loaded:
            self.load_data()
        
        return self.data[self.data['indicator_code'] == indicator_code].copy()
    
    def get_temporal_range(self) -> dict:
        """Get temporal range of the data."""
        if not self.loaded:
            self.load_data()
        
        date_columns = [col for col in self.data.columns if 'date' in col.lower()]
        temporal_info = {}
        
        for col in date_columns:
            if col in self.data.columns and self.data[col].notna().any():
                dates = pd.to_datetime(self.data[col], errors='coerce')
                temporal_info[col] = {
                    'min': dates.min(),
                    'max': dates.max(),
                    'range_days': (dates.max() - dates.min()).days
                }
        
        return temporal_info
    
    def save_sample(self, n: int = 100, output_path: str = None) -> None:
        """
        Save a sample of the data.
        
        Parameters:
        -----------
        n : int
            Number of samples to save
        output_path : str
            Path to save sample
        """
        if not self.loaded:
            self.load_data()
        
        sample = self.data.sample(min(n, len(self.data)))
        
        if output_path:
            sample.to_csv(output_path, index=False)
            print(f"Sample saved to {output_path}")
        
        return sample