# src/forecasting.py
"""
Forecasting module for financial inclusion indicators.
Combines trend analysis with event impacts to forecast Access and Usage.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class FinancialInclusionForecaster:
    """Forecast financial inclusion indicators using trend + event impacts."""

    def __init__(self, data: pd.DataFrame, impact_parameters: Dict = None):
        self.data = data.copy()
        self.impact_params = impact_parameters or {}

        # Core indicators to forecast
        self.target_indicators = {
            'access': 'ACC_OWNERSHIP',
            'usage': 'USG_DIGITAL_PAYMENT'
        }

        # Extract time series and events
        self.time_series = self._prepare_time_series()
        self.events = self._extract_events()

        # Forecasting results
        self.forecasts = {}
        self.scenarios = {}

    def _prepare_time_series(self) -> Dict[str, pd.Series]:
        """Prepare time series data for each indicator."""
        ts_data = {}
        obs_data = self.data[self.data['record_type'] == 'observation'].copy()
        obs_data['observation_date'] = pd.to_datetime(obs_data['observation_date'])

        for indicator_code in obs_data['indicator_code'].unique():
            indicator_obs = obs_data[obs_data['indicator_code'] == indicator_code]
            if len(indicator_obs) > 1:
                indicator_obs = indicator_obs.sort_values('observation_date')
                ts = indicator_obs.set_index('observation_date')['value_numeric']
                ts_data[indicator_code] = ts

        return ts_data

    def _extract_events(self) -> pd.DataFrame:
        """Extract events for forecasting timeline."""
        events = self.data[self.data['record_type'] == 'event'].copy()
        events['observation_date'] = pd.to_datetime(events['observation_date'])
        events = events.sort_values('observation_date')
        return events

    def fit_trend_model(self, indicator: str, model_type: str = 'linear') -> Dict:
        """Fit trend model to historical data."""
        if indicator not in self.time_series:
            raise ValueError(f"No time series data for indicator: {indicator}")

        ts = self.time_series[indicator]
        if len(ts) < 3:
            raise ValueError(f"Insufficient data points for {indicator}: {len(ts)}")

        dates_numeric = (ts.index - ts.index[0]).days / 365.25
        values = ts.values

        model_results = {
            'indicator': indicator,
            'model_type': model_type,
            'n_observations': len(ts),
            'dates': ts.index.tolist(),
            'values': values.tolist()
        }

        if model_type == 'linear':
            slope, intercept, r_value, p_value, std_err = stats.linregress(dates_numeric, values)
            model_results.update({
                'intercept': float(intercept),
                'slope': float(slope),
                'r_squared': float(r_value ** 2),
                'std_err': float(std_err),
                'predict_function': lambda t: intercept + slope * t,
                'annual_growth_rate': float(slope)
            })

        elif model_type == 'exponential':
            log_values = np.log(values)
            slope, intercept, r_value, p_value, std_err = stats.linregress(dates_numeric, log_values)
            a = np.exp(intercept)
            b = slope
            model_results.update({
                'a': float(a),
                'b': float(b),
                'r_squared': float(r_value ** 2),
                'std_err': float(std_err),
                'predict_function': lambda t: a * np.exp(b * t),
                'annual_growth_rate': float((np.exp(b) - 1) * 100)
            })

        elif model_type == 'logistic':
            L = 100
            logit_values = np.log((L - values) / values)
            mask = np.isfinite(logit_values)
            if np.sum(mask) >= 2:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    dates_numeric[mask], logit_values[mask]
                )
                k = -slope
                t0 = intercept / k if k != 0 else 0
                model_results.update({
                    'L': float(L),
                    'k': float(k),
                    't0': float(t0),
                    'r_squared': float(r_value ** 2),
                    'predict_function': lambda t: L / (1 + np.exp(-k * (t - t0)))
                })
                model_results['inflection_point'] = ts.index[0] + timedelta(days=t0 * 365.25)
            else:
                return self.fit_trend_model(indicator, 'linear')

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        predicted = [model_results['predict_function'](t) for t in dates_numeric]
        residuals = values - predicted
        model_results.update({
            'predicted': predicted,
            'residuals': residuals.tolist(),
            'rmse': float(np.sqrt(np.mean(residuals ** 2))),
            'mae': float(np.mean(np.abs(residuals)))
        })

        return model_results

    def select_best_trend_model(self, indicator: str) -> Dict:
        """Select best trend model based on RMSE."""
        model_types = ['linear', 'exponential', 'logistic']
        models = {}
        for model_type in model_types:
            try:
                model = self.fit_trend_model(indicator, model_type)
                models[model_type] = model
            except Exception as e:
                print(f"Could not fit {model_type} model for {indicator}: {e}")

        if not models:
            raise ValueError(f"Could not fit any model for {indicator}")

        best_model_type = min(models.keys(), key=lambda x: models[x]['rmse'])
        best_model = models[best_model_type]
        print(f"Selected {best_model_type} model for {indicator} (RMSE: {best_model['rmse']:.3f})")
        return best_model

    def apply_event_impacts(self, base_trend: np.ndarray, 
                            forecast_dates: pd.DatetimeIndex,
                            indicator: str) -> np.ndarray:
        """Apply event impacts to base trend forecast."""
        if not self.impact_params:
            print(f"No impact parameters provided for {indicator}, using base trend only")
            return base_trend

        adjusted_forecast = base_trend.copy()

        for event_key, impact_params in self.impact_params.items():
            if indicator in event_key:
                event_name = event_key.split('_')[0] + ' ' + event_key.split('_')[1]
                event = self.events[self.events['indicator'].str.contains(event_name, na=False)]
                if event.empty:
                    continue
                event_date = pd.to_datetime(event.iloc[0]['observation_date'])
                magnitude = impact_params.get('magnitude', 0)
                lag_months = impact_params.get('lag_months', 12)
                function_type = impact_params.get('function_type', 'sigmoid')

                for i, date in enumerate(forecast_dates):
                    months_since = (date - event_date).days / 30.44
                    if months_since >= 0:
                        if function_type == 'sigmoid':
                            k = 0.5
                            t0 = lag_months
                            impact = magnitude / (1 + np.exp(-k * (months_since - t0)))
                        elif function_type == 'linear':
                            impact = magnitude * (months_since / lag_months) if months_since < lag_months else magnitude
                        elif function_type == 'exponential':
                            tau = lag_months / 3
                            impact = magnitude * (1 - np.exp(-months_since / tau))
                        else:
                            impact = magnitude
                        adjusted_forecast[i] += impact

        adjusted_forecast = np.clip(adjusted_forecast, 0, 100)
        return adjusted_forecast

    def generate_forecast(self, indicator: str, 
                          forecast_years: List[int] = [2025, 2026, 2027],
                          include_events: bool = True) -> Dict:
        """Generate forecast for a specific indicator."""
        if indicator not in self.time_series:
            raise ValueError(f"No data available for indicator: {indicator}")

        ts = self.time_series[indicator]
        trend_model = self.select_best_trend_model(indicator)
        forecast_start = pd.Timestamp(f'{forecast_years[0]}-01-01')
        forecast_end = pd.Timestamp(f'{forecast_years[-1]}-12-31')
        forecast_dates = pd.date_range(start=forecast_start, end=forecast_end, freq='M')
        reference_date = ts.index[0]
        dates_numeric = [(date - reference_date).days / 365.25 for date in forecast_dates]

        base_forecast = np.array([trend_model['predict_function'](t) for t in dates_numeric])
        final_forecast = self.apply_event_impacts(base_forecast, forecast_dates, indicator) if include_events else base_forecast

        ci_lower, ci_upper = self._calculate_confidence_intervals(base_forecast, trend_model['rmse'], len(ts))

        annual_forecasts = {}
        for year in forecast_years:
            year_mask = forecast_dates.year == year
            if year_mask.any():
                annual_forecasts[year] = {
                    'mean': float(np.mean(final_forecast[year_mask])),
                    'min': float(np.min(final_forecast[year_mask])),
                    'max': float(np.max(final_forecast[year_mask])),
                    'quarterly': {
                        f'Q{i}': float(np.mean(final_forecast[year_mask & (forecast_dates.quarter == i)]))
                        for i in range(1, 5)
                    }
                }

        forecast_result = {
            'indicator': indicator,
            'trend_model': trend_model,
            'forecast_dates': forecast_dates.tolist(),
            'base_forecast': base_forecast.tolist(),
            'final_forecast': final_forecast.tolist(),
            'confidence_intervals': {'lower': ci_lower.tolist(), 'upper': ci_upper.tolist()},
            'annual_forecasts': annual_forecasts,
            'include_events': include_events,
            'last_historical_value': float(ts.iloc[-1]),
            'last_historical_date': ts.index[-1].strftime('%Y-%m-%d')
        }

        self.forecasts[indicator] = forecast_result
        return forecast_result

    def _calculate_confidence_intervals(self, forecast: np.ndarray, rmse: float, n: int, confidence_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        se = rmse / np.sqrt(n)
        margin = z_score * se
        lower = np.clip(forecast - margin, 0, 100)
        upper = np.clip(forecast + margin, 0, 100)
        return lower, upper

    def create_scenarios(self, indicator: str,
                        forecast_years: List[int] = [2025, 2026, 2027]) -> Dict:
        """
        Create multiple forecast scenarios.
        
        Parameters:
        -----------
        indicator : str
            Indicator code
        forecast_years : List[int]
            Forecast years
            
        Returns:
        --------
        Dict: Scenario forecasts
        """
        scenarios = {}
        
        # Scenario 1: Optimistic (high growth + all events)
        optimistic_params = self._adjust_parameters_for_scenario('optimistic')
        temp_forecaster = FinancialInclusionForecaster(self.data, optimistic_params)
        optimistic = temp_forecaster.generate_forecast(indicator, forecast_years, include_events=True)
        
        # Scenario 2: Base (most likely)
        base = self.generate_forecast(indicator, forecast_years, include_events=True)
        
        # Scenario 3: Pessimistic (low growth, limited events)
        pessimistic_params = self._adjust_parameters_for_scenario('pessimistic')
        temp_forecaster = FinancialInclusionForecaster(self.data, pessimistic_params)
        pessimistic = temp_forecaster.generate_forecast(indicator, forecast_years, include_events=False)
        
        scenarios = {
            'optimistic': optimistic,
            'base': base,
            'pessimistic': pessimistic
        }
        
        self.scenarios[indicator] = scenarios
        return scenarios
    
    def _adjust_parameters_for_scenario(self, scenario: str) -> Dict:
        """
        Adjust impact parameters for different scenarios.
        
        Parameters:
        -----------
        scenario : str
            'optimistic', 'base', or 'pessimistic'
            
        Returns:
        --------
        Dict: Adjusted impact parameters
        """
        adjusted_params = self.impact_params.copy()
        
        if scenario == 'optimistic':
            # Increase impact magnitudes by 25%
            for key in adjusted_params:
                if 'magnitude' in adjusted_params[key]:
                    adjusted_params[key]['magnitude'] *= 1.25
                # Reduce lag times
                if 'lag_months' in adjusted_params[key]:
                    adjusted_params[key]['lag_months'] *= 0.75
        
        elif scenario == 'pessimistic':
            # Decrease impact magnitudes by 25%
            for key in adjusted_params:
                if 'magnitude' in adjusted_params[key]:
                    adjusted_params[key]['magnitude'] *= 0.75
                # Increase lag times
                if 'lag_months' in adjusted_params[key]:
                    adjusted_params[key]['lag_months'] *= 1.25
        
        return adjusted_params
    
    def forecast_all_targets(self, forecast_years: List[int] = [2025, 2026, 2027]) -> Dict:
        """
        Forecast all target indicators.
        
        Parameters:
        -----------
        forecast_years : List[int]
            Years to forecast
            
        Returns:
        --------
        Dict: All forecasts
        """
        all_forecasts = {}
        
        for indicator_name, indicator_code in self.target_indicators.items():
            print(f"\nForecasting {indicator_name} ({indicator_code})...")
            
            try:
                # Generate forecasts
                forecast = self.generate_forecast(indicator_code, forecast_years, include_events=True)
                
                # Create scenarios
                scenarios = self.create_scenarios(indicator_code, forecast_years)
                
                all_forecasts[indicator_name] = {
                    'indicator_code': indicator_code,
                    'forecast': forecast,
                    'scenarios': scenarios,
                    'summary': self._create_forecast_summary(forecast, scenarios)
                }
                
            except Exception as e:
                print(f"Error forecasting {indicator_name}: {e}")
                all_forecasts[indicator_name] = {
                    'error': str(e),
                    'indicator_code': indicator_code
                }
        
        return all_forecasts
    
    def _create_forecast_summary(self, forecast: Dict, scenarios: Dict) -> Dict:
        """Create summary statistics for forecast."""
        annual_data = forecast['annual_forecasts']
        
        summary = {
            'historical_trend': {
                'last_value': forecast['last_historical_value'],
                'last_date': forecast['last_historical_date'],
                'model_type': forecast['trend_model']['model_type'],
                'annual_growth': forecast['trend_model'].get('annual_growth_rate', 0)
            },
            'projections': {},
            'milestones': []
        }
        
        # Annual projections
        for year, data in annual_data.items():
            summary['projections'][year] = {
                'base': data['mean'],
                'optimistic': scenarios['optimistic']['annual_forecasts'][year]['mean'],
                'pessimistic': scenarios['pessimistic']['annual_forecasts'][year]['mean'],
                'range': f"{scenarios['pessimistic']['annual_forecasts'][year]['mean']:.1f} - {scenarios['optimistic']['annual_forecasts'][year]['mean']:.1f}%"
            }
        
        # Calculate milestones
        target_levels = [50, 60, 70, 80]  # Key percentage milestones
        
        for target in target_levels:
            # Find when base forecast reaches target
            forecast_values = forecast['final_forecast']
            forecast_dates = [pd.Timestamp(d) for d in forecast['forecast_dates']]
            
            for i, value in enumerate(forecast_values):
                if value >= target:
                    milestone_date = forecast_dates[i]
                    summary['milestones'].append({
                        'target': target,
                        'date': milestone_date.strftime('%Y-%m'),
                        'scenario': 'base'
                    })
                    break
        
        return summary
    
    def visualize_forecast(self, indicator: str, 
                         save_path: str = None,
                         show_scenarios: bool = True) -> None:
        """
        Visualize forecast with historical data.
        
        Parameters:
        -----------
        indicator : str
            Indicator code
        save_path : str, optional
            Path to save visualization
        show_scenarios : bool
            Whether to show scenario ranges
        """
        if indicator not in self.forecasts:
            print(f"No forecast available for {indicator}")
            return
        
        forecast = self.forecasts[indicator]
        ts = self.time_series[indicator]
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot historical data
        ax.plot(ts.index, ts.values, 'ko-', linewidth=2, markersize=8, 
               label='Historical', zorder=5)
        
        # Convert forecast dates
        forecast_dates = [pd.Timestamp(d) for d in forecast['forecast_dates']]
        forecast_values = forecast['final_forecast']
        
        # Plot forecast
        ax.plot(forecast_dates, forecast_values, 'b-', linewidth=2.5, 
               label='Base Forecast', zorder=4)
        
        # Plot confidence interval
        ci_lower = forecast['confidence_intervals']['lower']
        ci_upper = forecast['confidence_intervals']['upper']
        
        ax.fill_between(forecast_dates, ci_lower, ci_upper, 
                       alpha=0.2, color='blue', label='95% CI')
        
        if show_scenarios and indicator in self.scenarios:
            scenarios = self.scenarios[indicator]
            
            # Plot optimistic and pessimistic
            opt_dates = [pd.Timestamp(d) for d in scenarios['optimistic']['forecast_dates']]
            opt_values = scenarios['optimistic']['final_forecast']
            pess_values = scenarios['pessimistic']['final_forecast']
            
            ax.plot(opt_dates, opt_values, 'g--', linewidth=1.5, 
                   alpha=0.7, label='Optimistic')
            ax.plot(opt_dates, pess_values, 'r--', linewidth=1.5, 
                   alpha=0.7, label='Pessimistic')
            
            # Fill between scenarios
            ax.fill_between(opt_dates, pess_values, opt_values, 
                           alpha=0.1, color='gray', label='Scenario Range')
        
        # Add events as vertical lines
        for _, event in self.events.iterrows():
            event_date = event['event_date']
            if isinstance(event_date, pd.Timestamp):
                ax.axvline(x=event_date, color='gray', linestyle=':', alpha=0.5)
                ax.text(event_date, ax.get_ylim()[1]*0.95, 
                       event['event_name'][:20], 
                       rotation=45, ha='right', va='top', fontsize=8)
        
        # Formatting
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Percentage (%)', fontsize=12)
        ax.set_title(f'Forecast: {indicator}', fontsize=16, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Add target lines
        for target in [50, 60, 70]:
            ax.axhline(y=target, color='red', linestyle='--', alpha=0.3, linewidth=1)
            ax.text(ax.get_xlim()[1], target, f' {target}%', 
                   va='center', ha='left', fontsize=9, color='red')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def generate_forecast_report(self, forecast_years: List[int] = [2025, 2026, 2027],
                               output_path: str = None) -> Dict:
        """
        Generate comprehensive forecast report.
        
        Parameters:
        -----------
        forecast_years : List[int]
            Years to forecast
        output_path : str, optional
            Path to save report
            
        Returns:
        --------
        Dict: Forecast report
        """
        print("Generating comprehensive forecast report...")
        
        # Forecast all targets
        all_forecasts = self.forecast_all_targets(forecast_years)
        
        # Calculate key metrics
        access_forecast = all_forecasts.get('access', {})
        usage_forecast = all_forecasts.get('usage', {})
        
        report = {
            'generated_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'forecast_years': forecast_years,
            'methodology': {
                'trend_modeling': 'Best-fit trend (linear/exponential/logistic) selected by RMSE',
                'event_integration': 'Sigmoid impact functions from Task 3',
                'scenario_analysis': 'Optimistic/Base/Pessimistic scenarios',
                'confidence_intervals': 'Based on model RMSE and observation count'
            },
            'key_assumptions': [
                'Continuation of current policy environment',
                'Event impacts as estimated in Task 3',
                'No major economic disruptions',
                'Infrastructure development continues as planned'
            ],
            'limitations': [
                'Limited historical data (only 5 Findex points)',
                'Event impact estimates have uncertainty',
                'External economic factors not modeled',
                'Assumes no major policy changes beyond cataloged events'
            ],
            'forecasts': {},
            'policy_implications': [],
            'recommendations': []
        }
        
        # Add detailed forecasts
        for target_name, target_data in all_forecasts.items():
            if 'error' not in target_data:
                report['forecasts'][target_name] = target_data['summary']
        
        # Calculate policy implications
        if 'access' in report['forecasts']:
            access_2027 = report['forecasts']['access']['projections'].get(2027, {})
            if access_2027:
                base_2027 = access_2027.get('base', 0)
                
                if base_2027 >= 60:
                    implication = "Ethiopia likely to exceed NFIS-II 60% target"
                elif base_2027 >= 55:
                    implication = "Ethiopia on track to approach NFIS-II target"
                else:
                    implication = "Additional interventions needed to reach NFIS-II target"
                
                report['policy_implications'].append(implication)
        
        # Generate recommendations
        recommendations = [
            "Focus on converting mobile money registrations to active usage",
            "Accelerate digital ID (Fayda) rollout to reduce KYC barriers",
            "Enhance interoperability to maximize network effects",
            "Target gender gap reduction through women-focused programs",
            "Invest in digital literacy and merchant acceptance"
        ]
        
        report['recommendations'] = recommendations
        
        # Save report if path provided
        if output_path:
            import json
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Forecast report saved to {output_path}")
        
        return report