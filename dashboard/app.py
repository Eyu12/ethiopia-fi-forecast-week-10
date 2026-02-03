"""
Streamlit Dashboard for Ethiopia Financial Inclusion Forecasting.
Interactive dashboard for stakeholders to explore data, understand event impacts,
and view forecasts.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import pickle
import os
import sys
from pathlib import Path

# Add src to path for module imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # parent of dashboard
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

from data_loader import DataLoader
from preprocessing import DataPreprocessor
from impact_model import EventImpactModeler
from forecasting import FinancialInclusionForecaster

# Page configuration
st.set_page_config(
    page_title="Ethiopia Financial Inclusion Forecast",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #374151;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3B82F6;
    }
    .stProgress > div > div > div > div {
        background-color: #10B981;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'forecasts_loaded' not in st.session_state:
    st.session_state.forecasts_loaded = False

@st.cache_data
def load_data():
    """Load and preprocess data."""
    try:
        # Load enriched data
        data_path = Path(__file__).parent.parent / 'data' / 'processed' / 'analysis_ready_data.csv'
        data = pd.read_csv(data_path)
        data['observation_date'] = pd.to_datetime(data['observation_date'], errors='coerce')
        data['event_date'] = pd.to_datetime(data['event_date'], errors='coerce')
        
        # Load impact parameters
        impact_path = Path(__file__).parent.parent / 'models' / 'event_impact_parameters.json'
        with open(impact_path, 'r') as f:
            impact_parameters = json.load(f)
        
        # Load forecast results
        forecast_path = Path(__file__).parent.parent / 'models' / 'forecast_results.pkl'
        with open(forecast_path, 'rb') as f:
            forecast_results = pickle.load(f)
        
        return data, impact_parameters, forecast_results
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

def initialize_analysis(data, impact_parameters):
    """Initialize analysis components."""
    # Initialize forecaster
    forecaster = FinancialInclusionForecaster(data, impact_parameters)
    
    # Generate forecasts if not already in cache
    forecast_years = [2025, 2026, 2027]
    
    # Forecast access
    access_forecast = forecaster.generate_forecast('ACC_OWNERSHIP', forecast_years, include_events=True)
    access_scenarios = forecaster.create_scenarios('ACC_OWNERSHIP', forecast_years)
    
    # Forecast usage
    usage_forecast = forecaster.generate_forecast('USG_DIGITAL_PAYMENT', forecast_years, include_events=True)
    usage_scenarios = forecaster.create_scenarios('USG_DIGITAL_PAYMENT', forecast_years)
    
    return {
        'forecaster': forecaster,
        'access_forecast': access_forecast,
        'usage_forecast': usage_forecast,
        'access_scenarios': access_scenarios,
        'usage_scenarios': usage_scenarios,
        'forecast_years': forecast_years
    }

def create_metric_card(title, value, change=None, change_label=None):
    """Create a metric card for dashboard."""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(f"**{title}**")
        st.markdown(f"<h2 style='margin: 0;'>{value}</h2>", unsafe_allow_html=True)
    
    if change and change_label:
        with col2:
            change_color = "green" if change >= 0 else "red"
            change_icon = "▲" if change >= 0 else "▼"
            st.markdown(f"<p style='color: {change_color}; margin: 0;'>{change_icon} {change} {change_label}</p>", 
                       unsafe_allow_html=True)
    
    st.markdown("---")

def plot_time_series_interactive(data, indicator_code, title):
    """Create interactive time series plot."""
    # Filter data for indicator
    indicator_data = data[
        (data['record_type'] == 'observation') & 
        (data['indicator_code'] == indicator_code)
    ].sort_values('observation_date')
    
    if indicator_data.empty:
        st.warning(f"No data available for {indicator_code}")
        return None
    
    fig = go.Figure()
    
    # Add line for indicator
    fig.add_trace(go.Scatter(
        x=indicator_data['observation_date'],
        y=indicator_data['value_numeric'],
        mode='lines+markers',
        name=indicator_code,
        line=dict(color='#3B82F6', width=3),
        marker=dict(size=8),
        hovertemplate='%{x|%Y}: %{y:.1f}%<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=16, family="Arial", color="black")
        ),
        xaxis=dict(
            title='Year',
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray'
        ),
        yaxis=dict(
            title='Percentage (%)',
            range=[0, max(indicator_data['value_numeric'].max() * 1.2, 100)],
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray'
        ),
        hovermode='x unified',
        template='plotly_white',
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def plot_forecast_comparison(access_forecast, usage_forecast, access_scenarios, usage_scenarios):
    """Create forecast comparison visualization."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Account Ownership Forecast', 'Digital Payments Forecast'),
        horizontal_spacing=0.15
    )
    
    # Helper function to add forecast to subplot
    def add_forecast_to_subplot(fig, forecast, scenarios, row, col, color):
        # Get historical data points
        hist_dates = [pd.Timestamp(d) for d in forecast['trend_model']['dates']]
        hist_values = forecast['trend_model']['values']
        
        # Add historical
        fig.add_trace(
            go.Scatter(
                x=hist_dates,
                y=hist_values,
                mode='markers+lines',
                name='Historical',
                line=dict(color='black', width=2),
                marker=dict(size=6),
                showlegend=(row == 1 and col == 1)
            ),
            row=row, col=col
        )
        
        # Add forecast
        forecast_dates = [pd.Timestamp(d) for d in forecast['forecast_dates']]
        fig.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=forecast['final_forecast'],
                mode='lines',
                name='Base Forecast',
                line=dict(color=color, width=3),
                showlegend=(row == 1 and col == 1)
            ),
            row=row, col=col
        )
        
        # Add confidence interval
        fig.add_trace(
            go.Scatter(
                x=forecast_dates + forecast_dates[::-1],
                y=forecast['confidence_intervals']['upper'] + forecast['confidence_intervals']['lower'][::-1],
                fill='toself',
                fillcolor=f'rgba{tuple(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + (0.2,)}',
                line=dict(color='rgba(255, 255, 255, 0)'),
                name='95% CI',
                showlegend=(row == 1 and col == 1)
            ),
            row=row, col=col
        )
        
        # Add scenario ranges
        if scenarios:
            opt_dates = [pd.Timestamp(d) for d in scenarios['optimistic']['forecast_dates']]
            fig.add_trace(
                go.Scatter(
                    x=opt_dates,
                    y=scenarios['optimistic']['final_forecast'],
                    mode='lines',
                    name='Optimistic',
                    line=dict(color='green', width=1, dash='dash'),
                    showlegend=(row == 1 and col == 1)
                ),
                row=row, col=col
            )
            
            fig.add_trace(
                go.Scatter(
                    x=opt_dates,
                    y=scenarios['pessimistic']['final_forecast'],
                    mode='lines',
                    name='Pessimistic',
                    line=dict(color='red', width=1, dash='dash'),
                    showlegend=(row == 1 and col == 1)
                ),
                row=row, col=col
            )
    
    # Add access forecast
    add_forecast_to_subplot(fig, access_forecast, access_scenarios, 1, 1, '#3B82F6')
    
    # Add usage forecast
    add_forecast_to_subplot(fig, usage_forecast, usage_scenarios, 1, 2, '#10B981')
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='Financial Inclusion Forecasts: 2025-2027',
            font=dict(size=20, family="Arial", color="black")
        ),
        height=500,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02
        )
    )
    
    # Update axes
    fig.update_xaxes(title_text="Year", row=1, col=1)
    fig.update_xaxes(title_text="Year", row=1, col=2)
    fig.update_yaxes(title_text="Account Ownership (%)", row=1, col=1)
    fig.update_yaxes(title_text="Digital Payments (%)", row=1, col=2)
    
    return fig

def plot_scenario_comparison(access_scenarios, usage_scenarios, forecast_years):
    """Create scenario comparison visualization."""
    # Prepare data
    scenario_data = []
    
    for year in forecast_years:
        # Access scenarios
        scenario_data.append({
            'Year': year,
            'Indicator': 'Account Ownership',
            'Scenario': 'Optimistic',
            'Value': access_scenarios['optimistic']['annual_forecasts'][year]['mean']
        })
        scenario_data.append({
            'Year': year,
            'Indicator': 'Account Ownership',
            'Scenario': 'Base',
            'Value': access_scenarios['base']['annual_forecasts'][year]['mean']
        })
        scenario_data.append({
            'Year': year,
            'Indicator': 'Account Ownership',
            'Scenario': 'Pessimistic',
            'Value': access_scenarios['pessimistic']['annual_forecasts'][year]['mean']
        })
        
        # Usage scenarios
        scenario_data.append({
            'Year': year,
            'Indicator': 'Digital Payments',
            'Scenario': 'Optimistic',
            'Value': usage_scenarios['optimistic']['annual_forecasts'][year]['mean']
        })
        scenario_data.append({
            'Year': year,
            'Indicator': 'Digital Payments',
            'Scenario': 'Base',
            'Value': usage_scenarios['base']['annual_forecasts'][year]['mean']
        })
        scenario_data.append({
            'Year': year,
            'Indicator': 'Digital Payments',
            'Scenario': 'Pessimistic',
            'Value': usage_scenarios['pessimistic']['annual_forecasts'][year]['mean']
        })
    
    scenario_df = pd.DataFrame(scenario_data)
    
    # Create grouped bar chart
    fig = px.bar(
        scenario_df,
        x='Year',
        y='Value',
        color='Scenario',
        facet_col='Indicator',
        barmode='group',
        color_discrete_map={
            'Optimistic': '#10B981',
            'Base': '#3B82F6',
            'Pessimistic': '#EF4444'
        },
        title='Scenario Comparison by Year and Indicator'
    )
    
    fig.update_layout(
        height=400,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02
        )
    )
    
    fig.update_yaxes(title_text="Percentage (%)")
    
    return fig

def plot_event_timeline(data):
    """Create event timeline visualization."""
    # Extract events
    events = data[data['record_type'] == 'event'].copy()
    events = events.sort_values('event_date')
    
    if events.empty:
        st.warning("No events found in data")
        return None
    
    # Create timeline
    fig = go.Figure()
    
    for i, (_, event) in enumerate(events.iterrows()):
        event_date = pd.to_datetime(event['event_date'])
        event_name = event.get('event_name', 'Unknown Event')
        event_category = event.get('category', 'unknown')
        
        # Add event as a point on timeline
        fig.add_trace(go.Scatter(
            x=[event_date],
            y=[i],
            mode='markers+text',
            name=event_category,
            marker=dict(size=12, symbol='diamond'),
            text=[event_name],
            textposition="top center",
            hovertemplate=f"<b>{event_name}</b><br>Date: %{{x|%Y-%m-%d}}<br>Category: {event_category}<extra></extra>"
        ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='Financial Inclusion Event Timeline',
            font=dict(size=16, family="Arial", color="black")
        ),
        xaxis=dict(
            title='Date',
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray'
        ),
        yaxis=dict(
            title='',
            showticklabels=False,
            showgrid=False
        ),
        hovermode='closest',
        template='plotly_white',
        height=300,
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def create_impact_matrix_heatmap(impact_parameters):
    """Create heatmap of event impact matrix."""
    if not impact_parameters:
        return None
    
    # Extract unique events and indicators
    events = set()
    indicators = set()
    
    for key in impact_parameters.keys():
        parts = key.split('_')
        if len(parts) >= 2:
            event = ' '.join(parts[:2])
            indicator = '_'.join(parts[2:])
            events.add(event)
            indicators.add(indicator)
    
    # Create matrix
    matrix_data = pd.DataFrame(index=list(events), columns=list(indicators))
    
    for key, params in impact_parameters.items():
        parts = key.split('_')
        if len(parts) >= 2:
            event = ' '.join(parts[:2])
            indicator = '_'.join(parts[2:])
            if event in matrix_data.index and indicator in matrix_data.columns:
                matrix_data.loc[event, indicator] = params.get('magnitude', 0)
    
    matrix_data = matrix_data.fillna(0)
    
    # Create heatmap
    fig = px.imshow(
        matrix_data,
        text_auto='.2f',
        aspect="auto",
        color_continuous_scale='RdBu',
        title='Event-Impact Association Matrix',
        labels=dict(color="Impact Magnitude")
    )
    
    fig.update_layout(
        height=400,
        xaxis_title="Indicator",
        yaxis_title="Event"
    )
    
    return fig

def main():
    """Main dashboard application."""
    # Sidebar
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/7/71/Flag_of_Ethiopia.svg", width=100)
        st.title("Ethiopia FI Forecast")
        
        st.markdown("---")
        st.markdown("### Navigation")
        page = st.radio(
            "Select Page",
            ["Overview", "Trend Analysis", "Event Impacts", "Forecasts", "Scenario Explorer", "Data & Reports"]
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This dashboard provides forecasts for financial inclusion in Ethiopia, 
        focusing on Account Ownership (Access) and Digital Payment Usage.
        
        **Developed by:** Selam Analytics
        **For:** Consortium of stakeholders
        **Last Updated:** January 2026
        """)
    
    # Load data
    with st.spinner("Loading data..."):
        data, impact_parameters, forecast_results = load_data()
        
        if data is not None:
            st.session_state.data_loaded = True
            st.session_state.data = data
            st.session_state.impact_parameters = impact_parameters
            st.session_state.forecast_results = forecast_results
            
            # Initialize analysis
            analysis = initialize_analysis(data, impact_parameters)
            st.session_state.analysis = analysis
    
    # Page routing
    if page == "Overview":
        show_overview_page()
    elif page == "Trend Analysis":
        show_trend_analysis_page()
    elif page == "Event Impacts":
        show_event_impacts_page()
    elif page == "Forecasts":
        show_forecasts_page()
    elif page == "Scenario Explorer":
        show_scenario_explorer_page()
    elif page == "Data & Reports":
        show_data_reports_page()

def show_overview_page():
    """Show overview/dashboard page."""
    st.markdown("<h1 class='main-header'>📊 Ethiopia Financial Inclusion Forecasting System</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background-color: #F0F9FF; padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 2rem;'>
    <h3 style='margin-top: 0; color: #1E3A8A;'>Overview</h3>
    <p>This dashboard provides forecasts for Ethiopia's financial inclusion progress, 
    tracking Account Ownership (Access) and Digital Payment Usage (Usage) through 2027.</p>
    <p>The forecasting system combines historical trend analysis with event impact modeling 
    to predict how policies, product launches, and infrastructure investments affect inclusion outcomes.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics row
    st.markdown("<h3 class='sub-header'>Key Metrics</h3>", unsafe_allow_html=True)
    
    if st.session_state.data_loaded:
        data = st.session_state.data
        analysis = st.session_state.analysis
        
        # Calculate key metrics
        # Latest account ownership
        acc_data = data[
            (data['record_type'] == 'observation') & 
            (data['indicator_code'] == 'ACC_OWNERSHIP')
        ].sort_values('observation_date')
        
        latest_acc = acc_data.iloc[-1]['value_numeric'] if not acc_data.empty else "N/A"
        
        # Latest digital payments
        usage_data = data[
            (data['record_type'] == 'observation') & 
            (data['indicator_code'] == 'USG_DIGITAL_PAYMENT')
        ].sort_values('observation_date')
        
        latest_usage = usage_data.iloc[-1]['value_numeric'] if not usage_data.empty else "N/A"
        
        # Number of events
        num_events = len(data[data['record_type'] == 'event'])
        
        # Forecast for 2027
        acc_2027 = analysis['access_forecast']['annual_forecasts'][2027]['mean'] if analysis else "N/A"
        usage_2027 = analysis['usage_forecast']['annual_forecasts'][2027]['mean'] if analysis else "N/A"
        
        # Create metric columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            create_metric_card(
                "Current Account Ownership",
                f"{latest_acc:.1f}%" if isinstance(latest_acc, (int, float)) else latest_acc,
                change=acc_2027 - latest_acc if isinstance(acc_2027, (int, float)) and isinstance(latest_acc, (int, float)) else None,
                change_label="to 2027"
            )
        
        with col2:
            create_metric_card(
                "Current Digital Payments",
                f"{latest_usage:.1f}%" if isinstance(latest_usage, (int, float)) else latest_usage,
                change=usage_2027 - latest_usage if isinstance(usage_2027, (int, float)) and isinstance(latest_usage, (int, float)) else None,
                change_label="to 2027"
            )
        
        with col3:
            create_metric_card(
                "Cataloged Events",
                f"{num_events}",
                change_label="policy/product/infrastructure"
            )
        
        with col4:
            nfis_gap = 60 - acc_2027 if isinstance(acc_2027, (int, float)) else "N/A"
            status = "On Track" if isinstance(nfis_gap, (int, float)) and nfis_gap <= 0 else "Gap"
            create_metric_card(
                "NFIS-II 2030 Target",
                f"{status}",
                change=f"{abs(nfis_gap):.1f}pp" if isinstance(nfis_gap, (int, float)) and nfis_gap > 0 else "0pp",
                change_label="gap" if isinstance(nfis_gap, (int, float)) and nfis_gap > 0 else "achieved"
            )
    
    # Quick insights
    st.markdown("<h3 class='sub-header'>Quick Insights</h3>", unsafe_allow_html=True)
    
    insights_col1, insights_col2 = st.columns(2)
    
    with insights_col1:
        st.markdown("""
        <div class='metric-card'>
        <h4>📈 Growth Acceleration</h4>
        <p>Digital payments growing faster (+4.4pp/year) than account ownership (+2.8pp/year), 
        suggesting increased usage among existing account holders.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='metric-card'>
        <h4>🤝 Competition Benefits</h4>
        <p>M-Pesa entry estimated to add +5.8pp to digital payments by 2027, 
        demonstrating the positive impact of market competition.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with insights_col2:
        st.markdown("""
        <div class='metric-card'>
        <h4>🎯 Target Progress</h4>
        <p>Account ownership projected to reach 57.5% by 2027, approaching 
        the NFIS-II 2030 target of 60%. Digital payments need acceleration.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='metric-card'>
        <h4>⚠️ Key Uncertainty</h4>
        <p>Limited historical data (only 5 Findex points) creates significant 
        forecast uncertainty. Confidence intervals: ±3.2pp for account ownership.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Forecast preview
    st.markdown("<h3 class='sub-header'>Forecast Preview</h3>", unsafe_allow_html=True)
    
    if st.session_state.data_loaded and 'analysis' in st.session_state:
        analysis = st.session_state.analysis
        
        fig = plot_forecast_comparison(
            analysis['access_forecast'],
            analysis['usage_forecast'],
            analysis['access_scenarios'],
            analysis['usage_scenarios']
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Event timeline
    st.markdown("<h3 class='sub-header'>Recent Events Timeline</h3>", unsafe_allow_html=True)
    
    if st.session_state.data_loaded:
        fig = plot_event_timeline(st.session_state.data)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

def show_trend_analysis_page():
    """Show trend analysis page."""
    st.markdown("<h1 class='main-header'>📈 Trend Analysis</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background-color: #F0F9FF; padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 2rem;'>
    <h3 style='margin-top: 0; color: #1E3A8A;'>Historical Trends Analysis</h3>
    <p>Explore historical trends in financial inclusion indicators. Analyze growth patterns, 
    identify turning points, and understand the factors driving inclusion in Ethiopia.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("Please wait for data to load or check data availability.")
        return
    
    data = st.session_state.data
    
    # Indicator selection
    st.markdown("<h3 class='sub-header'>Select Indicators to Analyze</h3>", unsafe_allow_html=True)
    
    # Get available indicators
    available_indicators = data[
        data['record_type'] == 'observation'
    ]['indicator_code'].unique()
    
    selected_indicators = st.multiselect(
        "Choose indicators",
        options=available_indicators,
        default=['ACC_OWNERSHIP', 'USG_DIGITAL_PAYMENT', 'ACC_MM_ACCOUNT']
    )
    
    # Time range selection
    col1, col2 = st.columns(2)
    
    with col1:
        start_year = st.slider(
            "Start Year",
            min_value=2011,
            max_value=2024,
            value=2011,
            step=1
        )
    
    with col2:
        end_year = st.slider(
            "End Year",
            min_value=2011,
            max_value=2024,
            value=2024,
            step=1
        )
    
    # Filter data by selected indicators and time range
    filtered_data = data[
        (data['record_type'] == 'observation') &
        (data['indicator_code'].isin(selected_indicators)) &
        (data['observation_date'].dt.year >= start_year) &
        (data['observation_date'].dt.year <= end_year)
    ]
    
    if filtered_data.empty:
        st.warning("No data available for the selected indicators and time range.")
        return
    
    # Visualization options
    viz_type = st.radio(
        "Visualization Type",
        ["Time Series", "Growth Rates", "Comparison Chart"]
    )
    
    if viz_type == "Time Series":
        # Create individual time series plots
        for indicator in selected_indicators:
            indicator_data = filtered_data[filtered_data['indicator_code'] == indicator]
            
            if not indicator_data.empty:
                st.markdown(f"<h4>{indicator}</h4>", unsafe_allow_html=True)
                
                fig = plot_time_series_interactive(
                    filtered_data, 
                    indicator, 
                    f"{indicator} Trend"
                )
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show statistics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Latest Value",
                            f"{indicator_data.iloc[-1]['value_numeric']:.1f}%"
                        )
                    
                    with col2:
                        if len(indicator_data) > 1:
                            growth = indicator_data.iloc[-1]['value_numeric'] - indicator_data.iloc[0]['value_numeric']
                            st.metric(
                                "Total Growth",
                                f"{growth:+.1f}pp"
                            )
                    
                    with col3:
                        if len(indicator_data) > 1:
                            years = (indicator_data.iloc[-1]['observation_date'] - indicator_data.iloc[0]['observation_date']).days / 365.25
                            annual_growth = growth / years if years > 0 else 0
                            st.metric(
                                "Annual Growth",
                                f"{annual_growth:+.1f}pp/year"
                            )
                    
                    with col4:
                        st.metric(
                            "Data Points",
                            len(indicator_data)
                        )
    
    elif viz_type == "Growth Rates":
        # Calculate and display growth rates
        growth_data = []
        
        for indicator in selected_indicators:
            indicator_data = filtered_data[filtered_data['indicator_code'] == indicator]
            indicator_data = indicator_data.sort_values('observation_date')
            
            if len(indicator_data) > 1:
                for i in range(1, len(indicator_data)):
                    period_start = indicator_data.iloc[i-1]['observation_date'].year
                    period_end = indicator_data.iloc[i]['observation_date'].year
                    growth = indicator_data.iloc[i]['value_numeric'] - indicator_data.iloc[i-1]['value_numeric']
                    years = (indicator_data.iloc[i]['observation_date'] - indicator_data.iloc[i-1]['observation_date']).days / 365.25
                    annual_rate = growth / years if years > 0 else 0
                    
                    growth_data.append({
                        'Indicator': indicator,
                        'Period': f"{period_start}-{period_end}",
                        'Growth (pp)': growth,
                        'Annual Rate (pp/year)': annual_rate,
                        'Start Value': indicator_data.iloc[i-1]['value_numeric'],
                        'End Value': indicator_data.iloc[i]['value_numeric']
                    })
        
        if growth_data:
            growth_df = pd.DataFrame(growth_data)
            
            # Display as table
            st.dataframe(
                growth_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Create growth visualization
            fig = px.bar(
                growth_df,
                x='Period',
                y='Growth (pp)',
                color='Indicator',
                barmode='group',
                title='Growth by Period and Indicator'
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Comparison Chart":
        # Create comparison line chart
        pivot_data = filtered_data.pivot_table(
            index='observation_date',
            columns='indicator_code',
            values='value_numeric',
            aggfunc='mean'
        ).reset_index()
        
        fig = go.Figure()
        
        for indicator in selected_indicators:
            if indicator in pivot_data.columns:
                fig.add_trace(go.Scatter(
                    x=pivot_data['observation_date'],
                    y=pivot_data[indicator],
                    mode='lines+markers',
                    name=indicator,
                    hovertemplate='%{x|%Y}: %{y:.1f}%<extra></extra>'
                ))
        
        fig.update_layout(
            title='Indicator Comparison',
            xaxis_title='Year',
            yaxis_title='Percentage (%)',
            height=500,
            hovermode='x unified',
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation analysis
    st.markdown("<h3 class='sub-header'>Correlation Analysis</h3>", unsafe_allow_html=True)
    
    if len(selected_indicators) >= 2:
        # Prepare correlation matrix
        corr_data = filtered_data.pivot_table(
            index='observation_date',
            columns='indicator_code',
            values='value_numeric',
            aggfunc='mean'
        )
        
        # Calculate correlation matrix
        correlation_matrix = corr_data.corr()
        
        # Display correlation matrix
        fig = px.imshow(
            correlation_matrix,
            text_auto='.2f',
            aspect="auto",
            color_continuous_scale='RdBu',
            title='Correlation Matrix',
            labels=dict(color="Correlation")
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show key correlations
        st.markdown("**Key Correlations:**")
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr = correlation_matrix.iloc[i, j]
                if abs(corr) > 0.7:
                    strength = "Strong" if abs(corr) > 0.8 else "Moderate"
                    color = "🟢" if corr > 0 else "🔴"
                    st.markdown(f"{color} **{correlation_matrix.columns[i]}** ↔ **{correlation_matrix.columns[j]}**: {corr:.3f} ({strength})")

def show_event_impacts_page():
    """Show event impacts page."""
    st.markdown("<h1 class='main-header'>⚡ Event Impact Analysis</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background-color: #F0F9FF; padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 2rem;'>
    <h3 style='margin-top: 0; color: #1E3A8A;'>Modeling Event Impacts</h3>
    <p>Analyze how specific events (policy changes, product launches, infrastructure investments) 
    affect financial inclusion indicators. The impact model combines direct evidence with 
    comparable country data to estimate effects.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("Please wait for data to load or check data availability.")
        return
    
    data = st.session_state.data
    impact_parameters = st.session_state.impact_parameters
    
    # Event timeline
    st.markdown("<h3 class='sub-header'>Event Timeline</h3>", unsafe_allow_html=True)
    
    fig = plot_event_timeline(data)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    # Impact matrix
    st.markdown("<h3 class='sub-header'>Event-Impact Association Matrix</h3>", unsafe_allow_html=True)
    
    fig = create_impact_matrix_heatmap(impact_parameters)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    # Event impact explorer
    st.markdown("<h3 class='sub-header'>Event Impact Explorer</h3>", unsafe_allow_html=True)
    
    # Get unique events
    events = data[data['record_type'] == 'event']
    event_names = events['event_name'].unique().tolist()
    
    # Get indicators
    indicators = data[
        data['record_type'] == 'observation'
    ]['indicator_code'].unique().tolist()
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_event = st.selectbox(
            "Select Event",
            options=event_names,
            index=0 if event_names else None
        )
    
    with col2:
        selected_indicator = st.selectbox(
            "Select Indicator",
            options=indicators,
            index=indicators.index('ACC_OWNERSHIP') if 'ACC_OWNERSHIP' in indicators else 0
        )
    
    if selected_event and selected_indicator:
        # Get event details
        event_details = events[events['event_name'] == selected_event].iloc[0]
        
        # Try to find impact parameter
        impact_key = None
        for key in impact_parameters.keys():
            if selected_event.replace(' ', '_').split('_')[0] in key and selected_indicator in key:
                impact_key = key
                break
        
        # Display event details
        st.markdown("---")
        st.markdown(f"### {selected_event}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Event Date",
                event_details.get('event_date', 'Unknown').strftime('%Y-%m-%d') 
                if isinstance(event_details.get('event_date'), pd.Timestamp) 
                else 'Unknown'
            )
        
        with col2:
            st.metric(
                "Category",
                event_details.get('category', 'Unknown')
            )
        
        with col3:
            st.metric(
                "Confidence",
                event_details.get('confidence', 'Medium').title()
            )
        
        # Display impact estimate
        st.markdown("#### Impact Estimate")
        
        if impact_key and impact_key in impact_parameters:
            impact = impact_parameters[impact_key]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Impact Magnitude",
                    f"{impact.get('magnitude', 0):.3f}",
                    help="Maximum impact on indicator (percentage points)"
                )
            
            with col2:
                st.metric(
                    "Lag Time",
                    f"{impact.get('lag_months', 12)} months",
                    help="Time to reach half of maximum impact"
                )
            
            with col3:
                st.metric(
                    "Function Type",
                    impact.get('function_type', 'sigmoid').title(),
                    help="Shape of impact over time"
                )
            
            # Impact visualization
            st.markdown("#### Impact Over Time")
            
            # Create impact function visualization
            months = np.linspace(0, 48, 100)  # 0-48 months
            
            if impact.get('function_type') == 'sigmoid':
                k = 0.5
                t0 = impact.get('lag_months', 12)
                impact_values = impact.get('magnitude', 0) / (1 + np.exp(-k * (months - t0)))
            elif impact.get('function_type') == 'linear':
                lag = impact.get('lag_months', 12)
                impact_values = np.where(
                    months < lag,
                    impact.get('magnitude', 0) * (months / lag),
                    impact.get('magnitude', 0)
                )
            elif impact.get('function_type') == 'exponential':
                tau = impact.get('lag_months', 12) / 3
                impact_values = impact.get('magnitude', 0) * (1 - np.exp(-months / tau))
            else:  # immediate
                impact_values = np.where(months >= 0, impact.get('magnitude', 0), 0)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=months,
                y=impact_values,
                mode='lines',
                name='Impact',
                line=dict(color='#3B82F6', width=3),
                fill='tozeroy',
                fillcolor='rgba(59, 130, 246, 0.2)'
            ))
            
            # Add lag line
            fig.add_vline(
                x=impact.get('lag_months', 12),
                line_width=1,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Lag: {impact.get('lag_months', 12)} months",
                annotation_position="top right"
            )
            
            # Add magnitude line
            fig.add_hline(
                y=impact.get('magnitude', 0),
                line_width=1,
                line_dash="dash",
                line_color="green",
                annotation_text=f"Max: {impact.get('magnitude', 0):.3f}",
                annotation_position="bottom right"
            )
            
            fig.update_layout(
                title=f"Impact of {selected_event} on {selected_indicator}",
                xaxis_title="Months After Event",
                yaxis_title="Impact (Percentage Points)",
                height=400,
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("No specific impact estimate available for this event-indicator pair. Using comparable country evidence.")
            
            # Show comparable evidence
            st.markdown("#### Comparable Country Evidence")
            
            comparable_evidence = {
                'mobile_money_launch': '+15-25pp on mobile money accounts over 2 years (Kenya, Tanzania)',
                'interoperability': '+8-12pp on digital payments over 18 months (Ghana, Rwanda)',
                'digital_id': '+15-20pp on account ownership over 2 years (India, Pakistan)',
                'agent_expansion': '+1pp per 10 agents/10k adults (Kenya, Bangladesh)'
            }
            
            for event_type, evidence in comparable_evidence.items():
                if event_type in selected_event.lower():
                    st.success(f"**Based on {event_type.replace('_', ' ').title()}:** {evidence}")
                    break
    
    # Composite impact simulation
    st.markdown("<h3 class='sub-header'>Composite Impact Simulation</h3>", unsafe_allow_html=True)
    
    # Let user select multiple events
    selected_events = st.multiselect(
        "Select multiple events to simulate combined impact",
        options=event_names,
        default=['Telebirr Launch', 'M-Pesa Ethiopia Launch'] if 'Telebirr Launch' in event_names else event_names[:2]
    )
    
    if selected_events and 'analysis' in st.session_state:
        analysis = st.session_state.analysis
        forecaster = analysis['forecaster']
        
        # Simulate composite impact on account ownership
        try:
            composite_df = forecaster.simulate_composite_impact(
                selected_events, 
                'ACC_OWNERSHIP', 
                time_horizon=48
            )
            
            # Create visualization
            fig = go.Figure()
            
            # Add individual event impacts
            impact_cols = [col for col in composite_df.columns if col.startswith('impact_')]
            for col in impact_cols:
                event_name = col.replace('impact_', '')
                fig.add_trace(go.Scatter(
                    x=composite_df.index,
                    y=composite_df[col],
                    mode='lines',
                    name=event_name,
                    line=dict(width=2),
                    stackgroup='one'
                ))
            
            # Add total impact
            fig.add_trace(go.Scatter(
                x=composite_df.index,
                y=composite_df['composite_impact'],
                mode='lines',
                name='Total Impact',
                line=dict(color='black', width=3)
            ))
            
            fig.update_layout(
                title='Composite Impact of Selected Events on Account Ownership',
                xaxis_title='Date',
                yaxis_title='Cumulative Impact (Percentage Points)',
                height=400,
                hovermode='x unified',
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show summary statistics
            st.markdown("#### Impact Summary")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Peak Impact",
                    f"{composite_df['composite_impact'].max():.2f} pp"
                )
            
            with col2:
                st.metric(
                    "Time to 50% Impact",
                    f"{np.argmax(composite_df['composite_impact'] >= composite_df['composite_impact'].max() * 0.5)} months"
                )
            
            with col3:
                st.metric(
                    "Events",
                    len(selected_events)
                )
        
        except Exception as e:
            st.error(f"Error simulating composite impact: {e}")

def show_forecasts_page():
    """Show forecasts page."""
    st.markdown("<h1 class='main-header'>🔮 Forecasts</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background-color: #F0F9FF; padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 2rem;'>
    <h3 style='margin-top: 0; color: #1E3A8A;'>Financial Inclusion Forecasts</h3>
    <p>View forecasts for Account Ownership (Access) and Digital Payment Usage (Usage) 
    through 2027. The forecasts combine historical trends with event impact modeling, 
    providing confidence intervals and scenario analysis.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.data_loaded or 'analysis' not in st.session_state:
        st.warning("Please wait for analysis to complete or check data availability.")
        return
    
    analysis = st.session_state.analysis
    
    # Forecast type selection
    forecast_type = st.radio(
        "Select Forecast View",
        ["Combined View", "Account Ownership", "Digital Payments", "Comparison Table"]
    )
    
    if forecast_type == "Combined View":
        # Show combined forecast visualization
        st.markdown("<h3 class='sub-header'>Combined Forecast Visualization</h3>", unsafe_allow_html=True)
        
        fig = plot_forecast_comparison(
            analysis['access_forecast'],
            analysis['usage_forecast'],
            analysis['access_scenarios'],
            analysis['usage_scenarios']
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Key takeaways
        st.markdown("#### Key Takeaways")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class='metric-card'>
            <h4>Account Ownership</h4>
            <p><strong>2027 Forecast:</strong> {:.1f}% (Range: {:.1f}-{:.1f}%)</p>
            <p><strong>Annual Growth:</strong> +{:.1f} pp/year</p>
            <p><strong>2030 Target Gap:</strong> {:.1f} pp</p>
            </div>
            """.format(
                analysis['access_forecast']['annual_forecasts'][2027]['mean'],
                analysis['access_scenarios']['pessimistic']['annual_forecasts'][2027]['mean'],
                analysis['access_scenarios']['optimistic']['annual_forecasts'][2027]['mean'],
                analysis['access_forecast']['trend_model'].get('annual_growth_rate', 0),
                60 - analysis['access_forecast']['annual_forecasts'][2027]['mean']
            ), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class='metric-card'>
            <h4>Digital Payments</h4>
            <p><strong>2027 Forecast:</strong> {:.1f}% (Range: {:.1f}-{:.1f}%)</p>
            <p><strong>Annual Growth:</strong> +{:.1f} pp/year</p>
            <p><strong>2030 Target Gap:</strong> {:.1f} pp</p>
            </div>
            """.format(
                analysis['usage_forecast']['annual_forecasts'][2027]['mean'],
                analysis['usage_scenarios']['pessimistic']['annual_forecasts'][2027]['mean'],
                analysis['usage_scenarios']['optimistic']['annual_forecasts'][2027]['mean'],
                analysis['usage_forecast']['trend_model'].get('annual_growth_rate', 0),
                60 - analysis['usage_forecast']['annual_forecasts'][2027]['mean']
            ), unsafe_allow_html=True)
    
    elif forecast_type == "Account Ownership":
        # Detailed account ownership forecast
        st.markdown("<h3 class='sub-header'>Account Ownership (Access) Forecast</h3>", unsafe_allow_html=True)
        
        # Show visualization
        analysis['forecaster'].visualize_forecast('ACC_OWNERSHIP', show_scenarios=True)
        
        # Detailed forecast table
        st.markdown("#### Detailed Forecast Table")
        
        forecast_data = []
        for year in analysis['forecast_years']:
            forecast_data.append({
                'Year': year,
                'Base Forecast': f"{analysis['access_forecast']['annual_forecasts'][year]['mean']:.1f}%",
                'Optimistic': f"{analysis['access_scenarios']['optimistic']['annual_forecasts'][year]['mean']:.1f}%",
                'Pessimistic': f"{analysis['access_scenarios']['pessimistic']['annual_forecasts'][year]['mean']:.1f}%",
                '95% CI Range': f"±{analysis['access_forecast']['trend_model']['rmse']:.1f} pp"
            })
        
        forecast_df = pd.DataFrame(forecast_data)
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)
        
        # Quarterly breakdown for 2027
        st.markdown("#### 2027 Quarterly Breakdown")
        
        q_data = analysis['access_forecast']['annual_forecasts'][2027]['quarterly']
        q_df = pd.DataFrame({
            'Quarter': ['Q1', 'Q2', 'Q3', 'Q4'],
            'Forecast': [f"{q_data['Q1']:.1f}%", f"{q_data['Q2']:.1f}%", 
                        f"{q_data['Q3']:.1f}%", f"{q_data['Q4']:.1f}%"]
        })
        
        st.dataframe(q_df, use_container_width=True, hide_index=True)
        
        # Model information
        st.markdown("#### Model Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Trend Model",
                analysis['access_forecast']['trend_model']['model_type'].title()
            )
        
        with col2:
            st.metric(
                "Model RMSE",
                f"{analysis['access_forecast']['trend_model']['rmse']:.3f}"
            )
        
        with col3:
            st.metric(
                "R²",
                f"{analysis['access_forecast']['trend_model'].get('r_squared', 0):.3f}"
            )
    
    elif forecast_type == "Digital Payments":
        # Detailed digital payments forecast
        st.markdown("<h3 class='sub-header'>Digital Payment Usage (Usage) Forecast</h3>", unsafe_allow_html=True)
        
        # Show visualization
        analysis['forecaster'].visualize_forecast('USG_DIGITAL_PAYMENT', show_scenarios=True)
        
        # Detailed forecast table
        st.markdown("#### Detailed Forecast Table")
        
        forecast_data = []
        for year in analysis['forecast_years']:
            forecast_data.append({
                'Year': year,
                'Base Forecast': f"{analysis['usage_forecast']['annual_forecasts'][year]['mean']:.1f}%",
                'Optimistic': f"{analysis['usage_scenarios']['optimistic']['annual_forecasts'][year]['mean']:.1f}%",
                'Pessimistic': f"{analysis['usage_scenarios']['pessimistic']['annual_forecasts'][year]['mean']:.1f}%",
                '95% CI Range': f"±{analysis['usage_forecast']['trend_model']['rmse']:.1f} pp"
            })
        
        forecast_df = pd.DataFrame(forecast_data)
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)
        
        # Model information
        st.markdown("#### Model Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Trend Model",
                analysis['usage_forecast']['trend_model']['model_type'].title()
            )
        
        with col2:
            st.metric(
                "Model RMSE",
                f"{analysis['usage_forecast']['trend_model']['rmse']:.3f}"
            )
        
        with col3:
            st.metric(
                "R²",
                f"{analysis['usage_forecast']['trend_model'].get('r_squared', 0):.3f}"
            )
    
    elif forecast_type == "Comparison Table":
        # Comparison table
        st.markdown("<h3 class='sub-header'>Forecast Comparison Table</h3>", unsafe_allow_html=True)
        
        comparison_data = []
        
        for year in analysis['forecast_years']:
            # Account ownership
            acc_base = analysis['access_forecast']['annual_forecasts'][year]['mean']
            acc_opt = analysis['access_scenarios']['optimistic']['annual_forecasts'][year]['mean']
            acc_pess = analysis['access_scenarios']['pessimistic']['annual_forecasts'][year]['mean']
            
            # Digital payments
            usage_base = analysis['usage_forecast']['annual_forecasts'][year]['mean']
            usage_opt = analysis['usage_scenarios']['optimistic']['annual_forecasts'][year]['mean']
            usage_pess = analysis['usage_scenarios']['pessimistic']['annual_forecasts'][year]['mean']
            
            comparison_data.append({
                'Year': year,
                'Account Ownership (Base)': f"{acc_base:.1f}%",
                'Account Ownership (Range)': f"{acc_pess:.1f}-{acc_opt:.1f}%",
                'Digital Payments (Base)': f"{usage_base:.1f}%",
                'Digital Payments (Range)': f"{usage_pess:.1f}-{usage_opt:.1f}%",
                'Growth from 2024 (Acc)': f"{(acc_base - 49):+.1f} pp",
                'Growth from 2024 (Usage)': f"{(usage_base - 35):+.1f} pp"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        # Download button
        csv = comparison_df.to_csv(index=False)
        st.download_button(
            label="Download Forecast Data",
            data=csv,
            file_name="ethiopia_fi_forecasts.csv",
            mime="text/csv"
        )
    
    # Uncertainty information
    st.markdown("---")
    st.markdown("<h3 class='sub-header'>Forecast Uncertainty</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='metric-card'>
    <h4>Key Sources of Uncertainty</h4>
    <ul>
    <li><strong>Historical Data Sparsity:</strong> Only 5 Findex data points (2011-2024)</li>
    <li><strong>Event Impact Estimates:</strong> Based on comparable country evidence</li>
    <li><strong>External Factors:</strong> Economic conditions, policy changes not modeled</li>
    <li><strong>Model Specification:</strong> Choice of trend model and impact functions</li>
    </ul>
    <p><strong>Confidence Intervals:</strong> Represent statistical uncertainty from model fit (±{:.1f} pp for Account Ownership, ±{:.1f} pp for Digital Payments)</p>
    <p><strong>Scenario Ranges:</strong> Represent parameter uncertainty (optimistic/pessimistic assumptions)</p>
    </div>
    """.format(
        analysis['access_forecast']['trend_model']['rmse'],
        analysis['usage_forecast']['trend_model']['rmse']
    ), unsafe_allow_html=True)

def show_scenario_explorer_page():
    """Show scenario explorer page."""
    st.markdown("<h1 class='main-header'>🎭 Scenario Explorer</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background-color: #F0F9FF; padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 2rem;'>
    <h3 style='margin-top: 0; color: #1E3A8A;'>Explore Different Futures</h3>
    <p>Compare different scenarios for Ethiopia's financial inclusion trajectory. 
    Adjust key parameters to see how different assumptions affect the forecasts.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.data_loaded or 'analysis' not in st.session_state:
        st.warning("Please wait for analysis to complete or check data availability.")
        return
    
    analysis = st.session_state.analysis
    
    # Scenario definition
    st.markdown("<h3 class='sub-header'>Scenario Definitions</h3>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='background-color: #D1FAE5; padding: 1rem; border-radius: 0.5rem;'>
        <h4 style='color: #065F46; margin: 0;'>Optimistic</h4>
        <p style='margin: 0.5rem 0 0 0; font-size: 0.9rem;'>
        • Event impacts +25%<br>
        • Lag times -25%<br>
        • All events included<br>
        • Strong economic growth
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background-color: #DBEAFE; padding: 1rem; border-radius: 0.5rem;'>
        <h4 style='color: #1E40AF; margin: 0;'>Base</h4>
        <p style='margin: 0.5rem 0 0 0; font-size: 0.9rem;'>
        • Most likely estimates<br>
        • Task 3 impact parameters<br>
        • Current trajectory<br>
        • Moderate assumptions
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='background-color: #FEE2E2; padding: 1rem; border-radius: 0.5rem;'>
        <h4 style='color: #991B1B; margin: 0;'>Pessimistic</h4>
        <p style='margin: 0.5rem 0 0 0; font-size: 0.9rem;'>
        • Event impacts -25%<br>
        • Lag times +25%<br>
        • Some events excluded<br>
        • Economic challenges
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Scenario comparison visualization
    st.markdown("<h3 class='sub-header'>Scenario Comparison</h3>", unsafe_allow_html=True)
    
    fig = plot_scenario_comparison(
        analysis['access_scenarios'],
        analysis['usage_scenarios'],
        analysis['forecast_years']
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Custom scenario builder
    st.markdown("<h3 class='sub-header'>Custom Scenario Builder</h3>", unsafe_allow_html=True)
    
    with st.expander("Adjust Scenario Parameters"):
        col1, col2 = st.columns(2)
        
        with col1:
            event_impact_factor = st.slider(
                "Event Impact Factor",
                min_value=0.5,
                max_value=1.5,
                value=1.0,
                step=0.1,
                help="Multiply all event impacts by this factor"
            )
            
            include_mobile_money = st.checkbox(
                "Include Mobile Money Events",
                value=True,
                help="Include Telebirr, M-Pesa, and other mobile money events"
            )
        
        with col2:
            lag_time_factor = st.slider(
                "Lag Time Factor",
                min_value=0.5,
                max_value=2.0,
                value=1.0,
                step=0.1,
                help="Multiply all lag times by this factor"
            )
            
            include_infrastructure = st.checkbox(
                "Include Infrastructure Events",
                value=True,
                help="Include interoperability, 4G expansion, etc."
            )
        
        # Create custom scenario
        if st.button("Generate Custom Scenario"):
            st.info("Custom scenario generation would require re-running the forecast model with adjusted parameters.")
            st.markdown(f"""
            **Custom Scenario Parameters:**
            - Event Impact Factor: {event_impact_factor}
            - Lag Time Factor: {lag_time_factor}
            - Include Mobile Money: {include_mobile_money}
            - Include Infrastructure: {include_infrastructure}
            
            *Note: This is a demonstration. In a full implementation, these parameters would be fed back into the forecasting model.*
            """)
    
    # Target achievement analysis
    st.markdown("<h3 class='sub-header'>Target Achievement Analysis</h3>", unsafe_allow_html=True)
    
    # NFIS-II target: 60% by 2030
    target_year = 2030
    target_value = 60
    
    # Calculate when each scenario reaches target
    target_analysis = []
    
    for scenario_name, scenario_data in [('Optimistic', analysis['access_scenarios']['optimistic']),
                                        ('Base', analysis['access_forecast']),
                                        ('Pessimistic', analysis['access_scenarios']['pessimistic'])]:
        
        # Get forecast values for 2027
        forecast_2027 = scenario_data['annual_forecasts'][2027]['mean']
        
        # Estimate growth rate
        historical_value = scenario_data['last_historical_value']
        annual_growth = (forecast_2027 - historical_value) / 3  # 2024-2027
        
        # Project to 2030
        projected_2030 = forecast_2027 + (annual_growth * 3)  # 2027-2030
        
        # Calculate gap
        gap = projected_2030 - target_value
        
        target_analysis.append({
            'Scenario': scenario_name,
            '2027 Forecast': f"{forecast_2027:.1f}%",
            'Projected 2030': f"{projected_2030:.1f}%",
            '2030 Target Gap': f"{gap:+.1f} pp",
            'Status': 'Above Target' if gap >= 0 else 'Below Target'
        })
    
    target_df = pd.DataFrame(target_analysis)
    st.dataframe(target_df, use_container_width=True, hide_index=True)
    
    # Visualization of target progress
    st.markdown("#### Target Progress Visualization")
    
    fig = go.Figure()
    
    # Add current value
    fig.add_trace(go.Bar(
        x=['Current (2024)'],
        y=[49],  # Latest account ownership
        name='Current',
        marker_color='gray'
    ))
    
    # Add forecast values for each scenario
    scenarios = ['Optimistic', 'Base', 'Pessimistic']
    colors = ['#10B981', '#3B82F6', '#EF4444']
    
    for i, scenario in enumerate(scenarios):
        forecast_2027 = float(target_analysis[i]['2027 Forecast'].replace('%', ''))
        fig.add_trace(go.Bar(
            x=[f'2027 ({scenario})'],
            y=[forecast_2027],
            name=scenario,
            marker_color=colors[i]
        ))
    
    # Add target line
    fig.add_hline(
        y=target_value,
        line_width=2,
        line_dash="dash",
        line_color="red",
        annotation_text=f"NFIS-II 2030 Target: {target_value}%",
        annotation_position="top right"
    )
    
    fig.update_layout(
        title='Progress Toward NFIS-II 2030 Target (Account Ownership)',
        xaxis_title="Scenario",
        yaxis_title="Account Ownership (%)",
        height=400,
        barmode='group',
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Policy implications
    st.markdown("#### Policy Implications by Scenario")
    
    implications = {
        'Optimistic': [
            "Continue current policies and investments",
            "Focus on converting growth to sustainable usage",
            "Address remaining gender and regional gaps"
        ],
        'Base': [
            "Accelerate digital ID (Fayda) rollout",
            "Enhance merchant acceptance infrastructure",
            "Implement targeted financial literacy programs"
        ],
        'Pessimistic': [
            "Urgent interventions needed to reach 2030 targets",
            "Review and potentially revise NFIS-II targets",
            "Consider additional incentives for adoption"
        ]
    }
    
    for scenario, scenario_implications in implications.items():
        with st.expander(f"{scenario} Scenario Implications"):
            for implication in scenario_implications:
                st.markdown(f"• {implication}")

def show_data_reports_page():
    """Show data and reports page."""
    st.markdown("<h1 class='main-header'>📁 Data & Reports</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background-color: #F0F9FF; padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 2rem;'>
    <h3 style='margin-top: 0; color: #1E3A8A;'>Data Access and Reports</h3>
    <p>Access the underlying data, download reports, and explore methodology details.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("Please wait for data to load or check data availability.")
        return
    
    data = st.session_state.data
    
    # Data explorer
    st.markdown("<h3 class='sub-header'>Data Explorer</h3>", unsafe_allow_html=True)
    
    # Data type filter
    data_type = st.selectbox(
        "Select Data Type",
        ["All", "Observations", "Events", "Impact Links"]
    )
    
    # Filter data
    if data_type == "Observations":
        filtered_data = data[data['record_type'] == 'observation']
    elif data_type == "Events":
        filtered_data = data[data['record_type'] == 'event']
    elif data_type == "Impact Links":
        filtered_data = data[data['record_type'] == 'impact_link']
    else:
        filtered_data = data
    
    # Show data preview
    st.dataframe(
        filtered_data.head(100),
        use_container_width=True,
        hide_index=True
    )
    
    # Download data
    st.markdown("#### Download Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv = filtered_data.to_csv(index=False)
        st.download_button(
            label=f"Download {data_type} Data",
            data=csv,
            file_name=f"ethiopia_fi_{data_type.lower().replace(' ', '_')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Download all data
        csv_all = data.to_csv(index=False)
        st.download_button(
            label="Download All Data",
            data=csv_all,
            file_name="ethiopia_fi_complete_dataset.csv",
            mime="text/csv"
        )
    
    with col3:
        # Download forecast results
        if 'analysis' in st.session_state:
            analysis = st.session_state.analysis
            
            # Create forecast summary
            forecast_summary = []
            for year in analysis['forecast_years']:
                forecast_summary.append({
                    'Year': year,
                    'Account_Ownership_Base': analysis['access_forecast']['annual_forecasts'][year]['mean'],
                    'Account_Ownership_Optimistic': analysis['access_scenarios']['optimistic']['annual_forecasts'][year]['mean'],
                    'Account_Ownership_Pessimistic': analysis['access_scenarios']['pessimistic']['annual_forecasts'][year]['mean'],
                    'Digital_Payments_Base': analysis['usage_forecast']['annual_forecasts'][year]['mean'],
                    'Digital_Payments_Optimistic': analysis['usage_scenarios']['optimistic']['annual_forecasts'][year]['mean'],
                    'Digital_Payments_Pessimistic': analysis['usage_scenarios']['pessimistic']['annual_forecasts'][year]['mean']
                })
            
            forecast_df = pd.DataFrame(forecast_summary)
            forecast_csv = forecast_df.to_csv(index=False)
            
            st.download_button(
                label="Download Forecasts",
                data=forecast_csv,
                file_name="ethiopia_fi_forecasts_2025_2027.csv",
                mime="text/csv"
            )
    
    # Reports section
    st.markdown("<h3 class='sub-header'>Reports</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='metric-card'>
        <h4>📊 EDA Report</h4>
        <p>Comprehensive exploratory data analysis with insights on trends, patterns, and data quality.</p>
        <p><strong>Includes:</strong> Trend analysis, correlation matrices, event timeline, key insights</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Load and display EDA report
        try:
            eda_report_path = Path(__file__).parent.parent / 'reports' / 'eda_summary_report.json'
            if eda_report_path.exists():
                with open(eda_report_path, 'r') as f:
                    eda_report = json.load(f)
                
                with st.expander("View EDA Report Summary"):
                    st.json(eda_report, expanded=False)
        except:
            pass
    
    with col2:
        st.markdown("""
        <div class='metric-card'>
        <h4>📈 Forecast Report</h4>
        <p>Detailed forecast report with methodology, results, and policy implications.</p>
        <p><strong>Includes:</strong> 2025-2027 forecasts, scenario analysis, uncertainty quantification, recommendations</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Load and display forecast report
        try:
            forecast_report_path = Path(__file__).parent.parent / 'reports' / 'forecast_report.json'
            if forecast_report_path.exists():
                with open(forecast_report_path, 'r') as f:
                    forecast_report = json.load(f)
                
                with st.expander("View Forecast Report Summary"):
                    st.json(forecast_report, expanded=False)
        except:
            pass
    
    # Methodology
    st.markdown("<h3 class='sub-header'>Methodology</h3>", unsafe_allow_html=True)
    
    with st.expander("View Methodology Details"):
        st.markdown("""
        ### Forecasting Methodology
        
        The forecasting system combines multiple approaches:
        
        **1. Trend Analysis**
        - Historical data fitted with linear, exponential, and logistic models
        - Best model selected based on RMSE
        - Accounts for natural growth patterns
        
        **2. Event Impact Modeling**
        - Events categorized as policy changes, product launches, infrastructure
        - Impact estimates from direct evidence and comparable countries
        - Sigmoid impact functions for realistic adoption curves
        
        **3. Composite Forecasting**
        - Base trend + event impacts = final forecast
        - Additive model of independent event effects
        - Confidence intervals based on model uncertainty
        
        **4. Scenario Analysis**
        - Optimistic: +25% event impacts, -25% lag times
        - Base: Most likely estimates
        - Pessimistic: -25% event impacts, +25% lag times
        
        ### Data Sources
        
        **Primary Sources:**
        - World Bank Global Findex Database (2011-2024)
        - National Bank of Ethiopia reports
        - Mobile money operator reports (Telebirr, M-Pesa)
        - GSMA Mobile Economy reports
        
        **Supplementary Sources:**
        - ITU digital development data
        - IMF Financial Access Survey
        - Comparable country evidence (Kenya, Ghana, Tanzania)
        
        ### Limitations
        
        1. **Sparse Historical Data:** Only 5 Findex data points
        2. **Event Impact Uncertainty:** Estimates based on comparable evidence
        3. **External Factors:** Economic conditions not modeled
        4. **Independence Assumption:** Event impacts treated as additive
        5. **Data Quality:** Variations in measurement methodologies
        
        ### Validation
        
        - Model validation against historical data where available
        - Comparison with comparable country trajectories
        - Expert review of event impact estimates
        - Sensitivity analysis for key parameters
        """)
    
    # About section
    st.markdown("<h3 class='sub-header'>About This Dashboard</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='metric-card'>
    <h4>Project Information</h4>
    
    **Project:** Ethiopia Financial Inclusion Forecasting System<br>
    **Client:** Consortium of stakeholders (development finance institutions, mobile money operators, National Bank of Ethiopia)<br>
    **Developer:** Selam Analytics<br>
    **Version:** 1.0<br>
    **Last Updated:** January 2026<br>
    
    <h4>Contact Information</h4>
    
    For questions, feedback, or additional analysis requests:<br>
    • Email: insights@selamanalytics.com<br>
    • Website: www.selamanalytics.com<br>
    
    <h4>Acknowledgments</h4>
    
    • World Bank for the Global Findex Database<br>
    • National Bank of Ethiopia for policy context<br>
    • Telebirr and M-Pesa for market data<br>
    • GSMA for mobile money insights<br>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()