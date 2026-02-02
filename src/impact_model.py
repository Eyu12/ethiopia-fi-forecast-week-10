
"""
Event Impact Modeling for Financial Inclusion Forecasting.
Quantifies how events (policies, product launches, infrastructure investments)
affect financial inclusion indicators.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class EventImpactModeler:
    """Model the impact of events on financial inclusion indicators."""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize EventImpactModeler with processed data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Processed financial inclusion data with events and impact_links
        """
        self.data = data.copy()
        self.events = None
        self.impact_links = None
        self.association_matrix = None
        self.event_effects = {}
        
        # Initialize
        self._extract_events_and_impacts()
    
    def _extract_events_and_impacts(self) -> None:
        """Extract events and impact links from data."""
        # Extract events
        if 'record_type' in self.data.columns:
            self.events = self.data[self.data['record_type'] == 'event'].copy()
            if 'event_date' in self.events.columns:
                self.events['event_date'] = pd.to_datetime(self.events['event_date'], errors='coerce')
        
        # Extract impact links
        self.impact_links = self.data[self.data['record_type'] == 'impact_link'].copy()
        
        print(f"Extracted {len(self.events)} events and {len(self.impact_links)} impact links")
    
    def build_event_indicator_matrix(self) -> pd.DataFrame:
        """
        Build event-indicator association matrix.
        
        Returns:
        --------
        pd.DataFrame: Association matrix with events as rows and indicators as columns
        """
        if self.impact_links.empty:
            print("No impact links found. Creating empty matrix.")
            return pd.DataFrame()
        
        # Create association matrix
        association_data = []
        
        for _, link in self.impact_links.iterrows():
            parent_id = link.get('parent_id')
            related_indicator = link.get('related_indicator')
            impact_direction = link.get('impact_direction', 'positive')
            impact_magnitude = link.get('impact_magnitude', 0.1)
            lag_months = link.get('lag_months', 12)
            evidence_basis = link.get('evidence_basis', 'modeled')
            
            # Get event details
            event_details = {}
            if not self.events.empty and pd.notna(parent_id):
                event = self.events[self.events.get('event_id', '') == parent_id]
                if not event.empty:
                    event_details = {
                        'event_name': event.iloc[0].get('event_name', 'Unknown'),
                        'event_date': event.iloc[0].get('event_date'),
                        'category': event.iloc[0].get('category', 'unknown')
                    }
            
            association_data.append({
                'event_id': parent_id,
                'event_name': event_details.get('event_name', 'Unknown'),
                'event_category': event_details.get('category', 'unknown'),
                'indicator': related_indicator,
                'impact_direction': impact_direction,
                'impact_magnitude': float(impact_magnitude) if pd.notna(impact_magnitude) else 0.1,
                'lag_months': int(lag_months) if pd.notna(lag_months) else 12,
                'evidence_basis': evidence_basis
            })
        
        # Create association matrix
        association_df = pd.DataFrame(association_data)
        
        if not association_df.empty:
            # Pivot to create matrix
            self.association_matrix = association_df.pivot_table(
                index=['event_name', 'event_category'],
                columns='indicator',
                values='impact_magnitude',
                aggfunc='mean'
            )
            
            # Add direction information
            direction_matrix = association_df.pivot_table(
                index=['event_name', 'event_category'],
                columns='indicator',
                values='impact_direction',
                aggfunc=lambda x: x.mode()[0] if len(x.mode()) > 0 else 'positive'
            )
            
            # Combine magnitude and direction
            for idx in self.association_matrix.index:
                for col in self.association_matrix.columns:
                    if pd.notna(self.association_matrix.loc[idx, col]):
                        direction = direction_matrix.loc[idx, col] if pd.notna(direction_matrix.loc[idx, col]) else 'positive'
                        sign = 1 if direction == 'positive' else -1
                        self.association_matrix.loc[idx, col] *= sign
        
        return self.association_matrix
    
    def get_comparable_country_evidence(self) -> Dict:
        """
        Get comparable country evidence for event impacts.
        Based on documented impacts from similar contexts (Kenya, Ghana, Tanzania, etc.)
        
        Returns:
        --------
        Dict: Comparable country evidence
        """
        comparable_evidence = {
            'mobile_money_launch': {
                'countries': ['Kenya (M-Pesa 2007)', 'Tanzania (M-Pesa 2008)', 'Ghana (MTN Mobile Money 2009)'],
                'impacts': {
                    'ACC_OWNERSHIP': {'magnitude': 0.15, 'lag': 24, 'description': '+15pp in 2 years'},
                    'ACC_MM_ACCOUNT': {'magnitude': 0.25, 'lag': 12, 'description': '+25pp in 1 year'},
                    'USG_DIGITAL_PAYMENT': {'magnitude': 0.20, 'lag': 18, 'description': '+20pp in 1.5 years'}
                },
                'sources': ['GSMA State of the Industry Reports', 'World Bank Findex', 'Academic Studies']
            },
            'interoperability_launch': {
                'countries': ['Ghana (2015)', 'Tanzania (2014)', 'Rwanda (2015)'],
                'impacts': {
                    'USG_DIGITAL_PAYMENT': {'magnitude': 0.12, 'lag': 12, 'description': '+12pp in 1 year'},
                    'ACC_OWNERSHIP': {'magnitude': 0.08, 'lag': 18, 'description': '+8pp in 1.5 years'}
                },
                'sources': ['CGAP Reports', 'National Switch Reports']
            },
            'agent_network_expansion': {
                'countries': ['Kenya (2010-2015)', 'Bangladesh (bKash)', 'Pakistan (Easypaisa)'],
                'impacts': {
                    'ACC_OWNERSHIP': {'magnitude': 0.10, 'lag': 12, 'description': '1pp per 10 agents/10k adults'},
                    'USG_DIGITAL_PAYMENT': {'magnitude': 0.08, 'lag': 6, 'description': '+8pp usage increase'}
                },
                'sources': ['GSMA Agent Network Accelerator', 'IMF Working Papers']
            },
            'digital_id_rollout': {
                'countries': ['India (Aadhaar)', 'Pakistan (NADRA)', 'Ghana (Ghana Card)'],
                'impacts': {
                    'ACC_OWNERSHIP': {'magnitude': 0.20, 'lag': 24, 'description': '+20pp in 2 years'},
                    'USG_DIGITAL_PAYMENT': {'magnitude': 0.15, 'lag': 18, 'description': '+15pp in 1.5 years'}
                },
                'sources': ['World Bank ID4D', 'IMF Financial Access Survey']
            },
            'smartphone_penetration': {
                'countries': ['Global averages', 'East Africa region'],
                'impacts': {
                    'USG_DIGITAL_PAYMENT': {'magnitude': 0.30, 'lag': 12, 'description': '+0.3pp per 1% smartphone increase'},
                    'ACC_MM_ACCOUNT': {'magnitude': 0.25, 'lag': 12, 'description': '+0.25pp per 1% smartphone increase'}
                },
                'sources': ['GSMA Intelligence', 'ITU Digital Trends']
            }
        }
        
        return comparable_evidence
    
    def estimate_event_impact(self, event_name: str, indicator: str, 
                            use_comparable: bool = True) -> Dict:
        """
        Estimate impact of a specific event on an indicator.
        
        Parameters:
        -----------
        event_name : str
            Name of the event
        indicator : str
            Target indicator code
        use_comparable : bool
            Whether to use comparable country evidence
            
        Returns:
        --------
        Dict: Impact estimate with confidence
        """
        impact = {
            'event_name': event_name,
            'indicator': indicator,
            'magnitude': None,
            'direction': 'positive',
            'lag_months': 12,
            'confidence': 'low',
            'evidence': 'modeled',
            'notes': ''
        }
        
        # Check if we have direct impact link
        if not self.impact_links.empty:
            # Try to find matching impact link
            event_match = self.events[self.events['event_name'] == event_name]
            if not event_match.empty:
                event_id = event_match.iloc[0].get('event_id', '')
                impact_link = self.impact_links[
                    (self.impact_links['parent_id'] == event_id) & 
                    (self.impact_links['related_indicator'] == indicator)
                ]
                
                if not impact_link.empty:
                    link = impact_link.iloc[0]
                    impact['magnitude'] = float(link.get('impact_magnitude', 0.1))
                    impact['direction'] = link.get('impact_direction', 'positive')
                    impact['lag_months'] = int(link.get('lag_months', 12))
                    impact['evidence'] = link.get('evidence_basis', 'direct_link')
                    impact['confidence'] = 'medium'
                    impact['notes'] = 'Based on provided impact link'
                    return impact
        
        # Use comparable country evidence if enabled
        if use_comparable:
            comparable_evidence = self.get_comparable_country_evidence()
            
            # Map event to comparable evidence category
            event_category_map = {
                'Telebirr': 'mobile_money_launch',
                'M-Pesa': 'mobile_money_launch',
                'Safaricom': 'mobile_money_launch',
                'interoperability': 'interoperability_launch',
                'EthSwitch': 'interoperability_launch',
                'agent': 'agent_network_expansion',
                'Fayda': 'digital_id_rollout',
                'smartphone': 'smartphone_penetration',
                '4G': 'infrastructure',
                'policy': 'policy_change'
            }
            
            for keyword, category in event_category_map.items():
                if keyword.lower() in event_name.lower():
                    if category in comparable_evidence:
                        if indicator in comparable_evidence[category]['impacts']:
                            evidence = comparable_evidence[category]['impacts'][indicator]
                            impact['magnitude'] = evidence['magnitude']
                            impact['lag_months'] = evidence['lag']
                            impact['evidence'] = f"Comparable: {category}"
                            impact['confidence'] = 'medium'
                            impact['notes'] = evidence['description']
                            break
        
        # Default estimation if no other evidence
        if impact['magnitude'] is None:
            # Base magnitude on indicator type
            base_magnitudes = {
                'ACC_OWNERSHIP': 0.10,
                'ACC_MM_ACCOUNT': 0.15,
                'USG_DIGITAL_PAYMENT': 0.12,
                'USG_RECEIVE_WAGES': 0.08
            }
            
            impact['magnitude'] = base_magnitudes.get(indicator, 0.10)
            impact['confidence'] = 'low'
            impact['notes'] = 'Default estimation based on indicator type'
        
        return impact
    
    def calculate_historical_impact(self, event_name: str, 
                                  indicator_code: str) -> Dict:
        """
        Calculate actual historical impact of an event using pre/post analysis.
        
        Parameters:
        -----------
        event_name : str
            Name of the event
        indicator_code : str
            Indicator to analyze
        
        Returns:
        --------
        Dict: Historical impact analysis
        """
        # Get event details
        event = self.events[self.events['event_name'] == event_name]
        if event.empty:
            return {'error': f"Event '{event_name}' not found"}
        
        event_date = pd.to_datetime(event.iloc[0]['event_date'])
        
        # Get indicator data
        indicator_data = self.data[
            (self.data['record_type'] == 'observation') & 
            (self.data['indicator_code'] == indicator_code)
        ].copy()
        
        if indicator_data.empty:
            return {'error': f"No data for indicator '{indicator_code}'"}
        
        # Convert dates
        indicator_data['observation_date'] = pd.to_datetime(indicator_data['observation_date'])
        
        # Find pre and post event observations
        pre_event = indicator_data[indicator_data['observation_date'] < event_date]
        post_event = indicator_data[indicator_data['observation_date'] > event_date]
        
        if len(pre_event) == 0 or len(post_event) == 0:
            return {'error': 'Insufficient data for pre/post analysis'}
        
        # Get closest observations before and after event
        pre_value = pre_event.iloc[-1]['value_numeric']
        pre_date = pre_event.iloc[-1]['observation_date']
        post_value = post_event.iloc[0]['value_numeric']
        post_date = post_event.iloc[0]['observation_date']
        
        # Calculate impact
        time_diff = (post_date - pre_date).days / 30  # Convert to months
        value_diff = post_value - pre_value
        
        # Annualize impact
        if time_diff > 0:
            annualized_impact = (value_diff / time_diff) * 12
        else:
            annualized_impact = value_diff
        
        analysis = {
            'event_name': event_name,
            'event_date': event_date.strftime('%Y-%m-%d'),
            'indicator': indicator_code,
            'pre_event': {
                'date': pre_date.strftime('%Y-%m-%d'),
                'value': float(pre_value)
            },
            'post_event': {
                'date': post_date.strftime('%Y-%m-%d'),
                'value': float(post_value)
            },
            'observed_impact': {
                'absolute_change': float(value_diff),
                'time_months': float(round(time_diff, 1)),
                'monthly_rate': float(value_diff / time_diff) if time_diff > 0 else 0,
                'annualized_impact': float(annualized_impact)
            },
            'data_quality': {
                'pre_observations': len(pre_event),
                'post_observations': len(post_event),
                'confidence': 'high' if len(pre_event) > 1 and len(post_event) > 1 else 'medium'
            }
        }
        
        return analysis
    
    def validate_model_against_history(self) -> pd.DataFrame:
        """
        Validate impact model against historical data where possible.
        
        Returns:
        --------
        pd.DataFrame: Validation results
        """
        validation_results = []
        
        # Key events to validate
        key_events = [
            ('Telebirr Launch', 'ACC_MM_ACCOUNT'),
            ('Telebirr Launch', 'ACC_OWNERSHIP'),
            ('M-Pesa Ethiopia Launch', 'ACC_MM_ACCOUNT'),
            ('Safaricom Ethiopia License', 'ACC_OWNERSHIP')
        ]
        
        for event_name, indicator in key_events:
            # Get historical impact
            historical = self.calculate_historical_impact(event_name, indicator)
            
            # Get model estimate
            model_estimate = self.estimate_event_impact(event_name, indicator)
            
            if 'error' not in historical:
                validation = {
                    'event': event_name,
                    'indicator': indicator,
                    'historical_impact': historical['observed_impact']['absolute_change'],
                    'model_estimate': model_estimate['magnitude'],
                    'difference': None,
                    'match_quality': 'unknown'
                }
                
                if model_estimate['magnitude'] is not None:
                    validation['difference'] = abs(
                        historical['observed_impact']['absolute_change'] - 
                        model_estimate['magnitude']
                    )
                    
                    # Assess match quality
                    diff_pct = validation['difference'] / max(abs(historical['observed_impact']['absolute_change']), 0.01)
                    if diff_pct < 0.3:
                        validation['match_quality'] = 'good'
                    elif diff_pct < 0.5:
                        validation['match_quality'] = 'moderate'
                    else:
                        validation['match_quality'] = 'poor'
                
                validation_results.append(validation)
        
        return pd.DataFrame(validation_results)
    
    def create_impact_function(self, event_name: str, indicator: str, 
                             function_type: str = 'sigmoid') -> callable:
        """
        Create mathematical function representing event impact over time.
        
        Parameters:
        -----------
        event_name : str
            Name of the event
        indicator : str
            Target indicator
        function_type : str
            Type of impact function: 'sigmoid', 'linear', 'exponential'
        
        Returns:
        --------
        callable: Function f(t) that returns impact at time t (months after event)
        """
        # Get impact parameters
        impact = self.estimate_event_impact(event_name, indicator)
        magnitude = impact['magnitude'] or 0.1
        lag = impact['lag_months'] or 12
        
        if function_type == 'sigmoid':
            # Sigmoid function: gradual build-up, plateau
            def sigmoid_impact(t):
                # t: months after event
                if t < 0:
                    return 0
                # Sigmoid parameters
                k = 0.5  # Steepness
                t0 = lag  # Time to reach half impact
                return magnitude / (1 + np.exp(-k * (t - t0)))
            return sigmoid_impact
        
        elif function_type == 'linear':
            # Linear function: constant impact after lag
            def linear_impact(t):
                if t < lag:
                    return magnitude * (t / lag)
                else:
                    return magnitude
            return linear_impact
        
        elif function_type == 'exponential':
            # Exponential function: rapid initial impact, then slower
            def exponential_impact(t):
                if t < 0:
                    return 0
                tau = lag / 3  # Time constant
                return magnitude * (1 - np.exp(-t / tau))
            return exponential_impact
        
        else:
            # Default: immediate impact
            def default_impact(t):
                return magnitude if t >= 0 else 0
            return default_impact
    
    def simulate_composite_impact(self, events: List[str], 
                                indicator: str, 
                                time_horizon: int = 36) -> pd.DataFrame:
        """
        Simulate composite impact of multiple events over time.
        
        Parameters:
        -----------
        events : List[str]
            List of event names
        indicator : str
            Target indicator
        time_horizon : int
            Time horizon in months
        
        Returns:
        --------
        pd.DataFrame: Time series of composite impact
        """
        # Get event dates
        event_dates = {}
        for event_name in events:
            event = self.events[self.events['event_name'] == event_name]
            if not event.empty:
                event_dates[event_name] = pd.to_datetime(event.iloc[0]['event_date'])
        
        # Create time index
        start_date = min(event_dates.values()) if event_dates else pd.Timestamp('2021-01-01')
        dates = pd.date_range(start=start_date, periods=time_horizon, freq='M')
        
        # Initialize impact matrix
        impact_df = pd.DataFrame(index=dates)
        impact_df['composite_impact'] = 0
        
        # Add individual event impacts
        for event_name, event_date in event_dates.items():
            impact_func = self.create_impact_function(event_name, indicator, 'sigmoid')
            
            # Calculate impact at each time point
            event_impacts = []
            for date in dates:
                months_since = (date - event_date).days / 30.44
                impact = impact_func(months_since)
                event_impacts.append(impact)
            
            impact_df[f'impact_{event_name}'] = event_impacts
            impact_df['composite_impact'] += event_impacts
        
        # Cap at reasonable maximum
        impact_df['composite_impact'] = impact_df['composite_impact'].clip(upper=1.0)
        
        return impact_df
    
    def visualize_impact_timeline(self, events: List[str], 
                                indicator: str,
                                save_path: str = None) -> None:
        """
        Visualize impact timeline for multiple events.
        
        Parameters:
        -----------
        events : List[str]
            List of event names
        indicator : str
            Target indicator
        save_path : str, optional
            Path to save visualization
        """
        # Get simulation data
        impact_df = self.simulate_composite_impact(events, indicator, time_horizon=48)
        
        # Create visualization
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot individual impacts
        impact_cols = [col for col in impact_df.columns if col.startswith('impact_')]
        for col in impact_cols:
            axes[0].plot(impact_df.index, impact_df[col], label=col.replace('impact_', ''), linewidth=2)
        
        axes[0].set_title(f'Individual Event Impacts on {indicator}', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Impact Magnitude', fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot composite impact
        axes[1].plot(impact_df.index, impact_df['composite_impact'], 
                    color='darkred', linewidth=3, label='Composite Impact')
        axes[1].fill_between(impact_df.index, 0, impact_df['composite_impact'], 
                            alpha=0.3, color='darkred')
        
        # Add event markers
        event_dates = {}
        for event_name in events:
            event = self.events[self.events['event_name'] == event_name]
            if not event.empty:
                event_date = pd.to_datetime(event.iloc[0]['event_date'])
                event_dates[event_name] = event_date
                axes[1].axvline(x=event_date, color='gray', linestyle='--', alpha=0.7)
                axes[1].text(event_date, axes[1].get_ylim()[1]*0.9, 
                           event_name, rotation=45, ha='right', fontsize=9)
        
        axes[1].set_title(f'Composite Impact Timeline on {indicator}', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Date', fontsize=12)
        axes[1].set_ylabel('Total Impact', fontsize=12)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def generate_impact_report(self, output_path: str = None) -> Dict:
        """
        Generate comprehensive impact modeling report.
        
        Parameters:
        -----------
        output_path : str, optional
            Path to save report
        
        Returns:
        --------
        Dict: Report data
        """
        # Build association matrix if not exists
        if self.association_matrix is None:
            self.build_event_indicator_matrix()
        
        # Validate model
        validation_results = self.validate_model_against_history()
        
        # Key events analysis
        key_events = ['Telebirr Launch', 'M-Pesa Ethiopia Launch', 
                     'Safaricom Ethiopia License', 'EthSwitch Interoperability']
        key_indicators = ['ACC_OWNERSHIP', 'ACC_MM_ACCOUNT', 'USG_DIGITAL_PAYMENT']
        
        detailed_impacts = []
        for event in key_events:
            for indicator in key_indicators:
                impact = self.estimate_event_impact(event, indicator)
                historical = self.calculate_historical_impact(event, indicator)
                
                if 'error' not in historical:
                    detailed_impacts.append({
                        'event': event,
                        'indicator': indicator,
                        'model_estimate': impact['magnitude'],
                        'historical_actual': historical['observed_impact']['absolute_change'],
                        'confidence': impact['confidence'],
                        'lag_months': impact['lag_months']
                    })
        
        # Create report
        report = {
            'generated_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'summary': {
                'total_events': len(self.events),
                'total_impact_links': len(self.impact_links),
                'key_events_analyzed': len(key_events),
                'indicators_covered': len(key_indicators)
            },
            'association_matrix': self.association_matrix.to_dict() if self.association_matrix is not None else {},
            'validation_results': validation_results.to_dict('records') if not validation_results.empty else [],
            'detailed_impacts': detailed_impacts,
            'methodology': {
                'impact_estimation': 'Combination of direct impact links and comparable country evidence',
                'validation_approach': 'Pre/post analysis where historical data available',
                'impact_functions': 'Sigmoid functions for gradual impact build-up',
                'composite_impact': 'Additive model of individual event impacts'
            },
            'assumptions': [
                'Event impacts are independent and additive',
                'Impact functions follow sigmoid pattern (gradual build-up, plateau)',
                'Comparable country evidence applicable to Ethiopia context',
                'No significant negative interactions between events'
            ],
            'limitations': [
                'Limited historical data for validation',
                'Sparse impact link data in provided dataset',
                'Event impacts may not be perfectly independent',
                'External factors (economic conditions, regulations) not modeled'
            ]
        }
        
        # Save report if path provided
        if output_path:
            import json
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Impact report saved to {output_path}")
        
        return report