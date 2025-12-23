"""
Revenue & COGS Forecasting Streamlit App
Interactive web app for forecasting 2026 revenue, COGS, and profit

Installation:
pip install streamlit pandas numpy statsmodels plotly gspread oauth2client

To run:
streamlit run forecast_app.py

For Google Sheets integration:
1. Go to https://console.cloud.google.com/
2. Create a new project
3. Enable Google Sheets API
4. Create Service Account credentials
5. Download JSON key file
6. Share your Google Sheet with the service account email
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
import warnings
import io
import json
warnings.filterwarnings('ignore')

# Try to import gspread for Google Sheets
try:
    import gspread
    from oauth2client.service_account import ServiceAccountCredentials
    GSPREAD_AVAILABLE = True
except ImportError:
    GSPREAD_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Revenue & COGS Forecasting",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1e40af;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f9ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3b82f6;
    }
    .profit-positive {
        color: #16a34a;
        font-weight: bold;
    }
    .profit-negative {
        color: #dc2626;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">Revenue, COGS & Profit Forecasting</p>', unsafe_allow_html=True)
st.markdown("---")

# Initialize session state
if 'forecasts' not in st.session_state:
    st.session_state.forecasts = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'parsed_data' not in st.session_state:
    st.session_state.parsed_data = None
if 'gsheet_connected' not in st.session_state:
    st.session_state.gsheet_connected = False

# Helper function for Google Sheets
def load_from_google_sheets(sheet_url, credentials_json):
    """Load data from Google Sheets"""
    try:
        # Parse credentials
        creds_dict = json.loads(credentials_json)
        
        # Setup credentials
        scope = ['https://spreadsheets.google.com/feeds',
                'https://www.googleapis.com/auth/drive']
        
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        
        # Open the spreadsheet
        sheet = client.open_by_url(sheet_url)
        worksheet = sheet.get_worksheet(0)  # Get first sheet
        
        # Get all values
        data = worksheet.get_all_values()
        
        # Convert to DataFrame
        df = pd.DataFrame(data[1:], columns=data[0])
        
        return df, None
    except Exception as e:
        return None, str(e)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Data source selection
    data_source = st.radio(
        "Select Data Source:",
        ["Upload CSV File", "Google Sheets"],
        help="Choose how to load your data"
    )
    
    if data_source == "Upload CSV File":
        # File upload
        uploaded_file = st.file_uploader(
            "Upload your CSV file", 
            type=['csv'],
            help="First N rows = Revenue, Next N rows = COGS (same category order)"
        )
        
        if uploaded_file is not None:
            st.session_state.df = pd.read_csv(uploaded_file)
            st.session_state.gsheet_connected = False
            st.success(f"File uploaded: {uploaded_file.name}")
            st.info(f"Total rows: {len(st.session_state.df)}")
            st.info(f"Months: {len(st.session_state.df.columns) - 1}")
    
    else:  # Google Sheets
        if not GSPREAD_AVAILABLE:
            st.error("Google Sheets integration not available. Install: pip install gspread oauth2client")
        else:
            st.markdown("### Connect to Google Sheets")
            
            # Two methods
            connection_method = st.radio(
                "Connection Method:",
                ["Public Sheet (Read-only)", "Private Sheet (Credentials)"],
                help="Public sheets don't need credentials"
            )
            
            if connection_method == "Public Sheet (Read-only)":
                st.info("Make your Google Sheet public: Share → Anyone with link can view")
                
                sheet_url = st.text_input(
                    "Google Sheet URL:",
                    placeholder="https://docs.google.com/spreadsheets/d/...",
                    help="Paste the full URL of your Google Sheet"
                )
                
                if st.button("🔗 Connect to Sheet"):
                    if sheet_url:
                        with st.spinner("Connecting to Google Sheets..."):
                            try:
                                # For public sheets, use simpler method
                                sheet_id = sheet_url.split('/d/')[1].split('/')[0]
                                csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
                                df = pd.read_csv(csv_url)
                                
                                st.session_state.df = df
                                st.session_state.gsheet_connected = True
                                st.success("Connected to Google Sheets!")
                                st.info(f"Total rows: {len(df)}")
                                st.info(f"Columns: {len(df.columns)}")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Connection failed: {e}")
                                st.info("Make sure your sheet is publicly accessible")
                    else:
                        st.warning("Please enter a Google Sheet URL")
            
            else:  # Private Sheet
                st.markdown("""
                **Setup Steps:**
                1. Go to [Google Cloud Console](https://console.cloud.google.com/)
                2. Create a project & enable Google Sheets API
                3. Create Service Account credentials
                4. Download JSON key file
                5. Share your sheet with the service account email
                """)
                
                credentials_file = st.file_uploader(
                    "Upload Service Account JSON:",
                    type=['json'],
                    help="Upload your Google Cloud credentials JSON file"
                )
                
                sheet_url = st.text_input(
                    "Google Sheet URL:",
                    placeholder="https://docs.google.com/spreadsheets/d/...",
                    key="private_sheet_url"
                )
                
                if st.button("Connect with Credentials"):
                    if credentials_file and sheet_url:
                        with st.spinner("Connecting to Google Sheets..."):
                            credentials_json = credentials_file.read().decode('utf-8')
                            df, error = load_from_google_sheets(sheet_url, credentials_json)
                            
                            if df is not None:
                                st.session_state.df = df
                                st.session_state.gsheet_connected = True
                                st.success("Connected to Google Sheets!")
                                st.info(f"Total rows: {len(df)}")
                                st.info(f"Columns: {len(df.columns)}")
                                st.rerun()
                            else:
                                st.error(f"Connection failed: {error}")
                    else:
                        st.warning("Please provide both credentials and sheet URL")
            
            # Refresh button if connected
            if st.session_state.gsheet_connected:
                if st.button("🔄 Refresh Data from Sheet"):
                    st.rerun()
    
    # Use sample data button
    st.markdown("---")
    if st.button("Use Sample Data"):
        sample_data = """RevenueUnit,January-2022,February-2022,March-2022,April-2022,May-2022,June-2022,July-2022,August-2022,September-2022,October-2022,November-2022,December-2022,January-2023,February-2023,March-2023,April-2023,May-2023,June-2023,July-2023,August-2023,September-2023,October-2023,November-2023,December-2023,January-2024,February-2024,March-2024,April-2024,May-2024,June-2024,July-2024,August-2024,September-2024,October-2024,November-2024,December-2024
ALLERGY COUGH & FLU,168050,966433.08,1029628.06,1027067.74,1375472.5,1439849.28,1685388.7,1172299.73,1530216.38,1446259.76,1403527.8,1560990.46,1063906.37,1010453.78,911667.5,1157105.23,1309143.75,1342215.67,1851251.05,1221026.56,958779.95,1506295.52,1003153.85,2184740.59,1527837.01,2013113.03,1456502.68,1571843.35,1697217.52,1991981.27,2175297.24,1764480.46,1764457.31,2691198.68,2201113.61,3490623.49
ANTIMALARIAL,103530,738359.43,804780.69,923147.21,1951351,441909.14,1438585,1238352.31,1205506.26,1447037.98,1342536.41,1339327.5,1208032.1,1064672.81,1220059.07,1455216.77,1484055.77,1881336.73,1737407.27,1300902.41,1095995.45,1183575.79,974535.96,1681690.6,1538909.03,2001271.06,1370931.06,1530208,1541444.25,2017119.01,2295928.58,1831249.18,1776821.73,1735575.29,1776464.82,2396447.59
ALLERGY COUGH & FLU,120000,690000,735000,732000,981000,1026000,1200000,837000,1092000,1032000,1002000,1113000,759000,720000,650000,825000,933000,957000,1320000,871000,684000,1074000,716000,1557000,1090000,1436000,1038000,1121000,1211000,1421000,1552000,1259000,1259000,1919000,1569000,2488000
ANTIMALARIAL,72000,515000,561000,643000,1360000,308000,1003000,864000,840000,1009000,936000,933000,842000,742000,850000,1014000,1034000,1311000,1211000,907000,764000,825000,679000,1172000,1073000,1395000,956000,1067000,1075000,1406000,1601000,1276000,1238000,1210000,1238000,1671000"""
        
        st.session_state.df = pd.read_csv(io.StringIO(sample_data))
        st.session_state.gsheet_connected = False
        st.success("Sample data loaded!")
        st.rerun()
    
    st.markdown("---")
    
    # Model selection
    st.subheader("🔧 Model Settings")
    forecast_method = st.selectbox(
        "Forecasting Method",
        ["SARIMA", "Exponential Smoothing"],
        help="SARIMA is better for data with clear seasonal patterns"
    )
    
    show_confidence = st.checkbox("Show Confidence Intervals", value=True)
    show_trends = st.checkbox("Show Trend Analysis", value=True)
    
    st.markdown("---")
    st.markdown("### How to Use")
    
    if st.session_state.gsheet_connected:
        st.success("Connected to Google Sheets")
        st.markdown("""
        - Data auto-syncs from your sheet
        - Click 'Refresh' to update
        - Run forecast when ready
        """)
    else:
        st.markdown("""
        1. Upload CSV or connect to Google Sheets
        2. Click 'Run Forecast'
        3. View trends & forecasts
        4. Analyze profit margins
        5. Download results
        """)

# Helper functions
def parse_revenue_cogs_data(df):
    """Parse CSV where first N rows are Revenue, next N rows are COGS"""
    categories = []
    category_names = []
    
    # Get unique category names in order of first appearance
    seen = set()
    for cat in df.iloc[:, 0]:
        if cat not in seen:
            category_names.append(cat)
            seen.add(cat)
    
    # Group by category
    for cat_name in category_names:
        cat_rows = df[df.iloc[:, 0] == cat_name]
        
        if len(cat_rows) >= 2:
            # First occurrence = Revenue, Second = COGS
            revenue_row = cat_rows.iloc[0, 1:].values
            cogs_row = cat_rows.iloc[1, 1:].values
            
            categories.append({
                'name': cat_name,
                'revenue': revenue_row.astype(float),
                'cogs': cogs_row.astype(float),
                'profit': revenue_row.astype(float) - cogs_row.astype(float)
            })
        elif len(cat_rows) == 1:
            # Only revenue available
            revenue_row = cat_rows.iloc[0, 1:].values
            categories.append({
                'name': cat_name,
                'revenue': revenue_row.astype(float),
                'cogs': None,
                'profit': None
            })
    
    return categories, df.columns[1:]  # Return categories and month names

@st.cache_data
def prepare_time_series(values, start_date='2022-01'):
    """Convert array to time series"""
    dates = pd.date_range(start=start_date, periods=len(values), freq='MS')
    ts = pd.Series(values, index=dates, dtype=float)
    return ts

def forecast_series(ts, method='SARIMA'):
    """Forecast using selected method"""
    try:
        if method == 'SARIMA':
            model = SARIMAX(ts, 
                           order=(1, 1, 1),
                           seasonal_order=(1, 1, 1, 12),
                           enforce_stationarity=False,
                           enforce_invertibility=False)
            
            fitted_model = model.fit(disp=False)
            forecast = fitted_model.forecast(steps=12)
            forecast_obj = fitted_model.get_forecast(steps=12)
            conf_int = forecast_obj.conf_int()
            
            metrics = {
                'AIC': fitted_model.aic,
                'BIC': fitted_model.bic
            }
            
        else:  # Exponential Smoothing
            model = ExponentialSmoothing(ts, 
                                        seasonal='add',
                                        seasonal_periods=12,
                                        trend='add')
            
            fitted_model = model.fit()
            forecast = fitted_model.forecast(steps=12)
            conf_int = pd.DataFrame({
                'lower': forecast * 0.9,
                'upper': forecast * 1.1
            }, index=forecast.index)
            
            metrics = {'Method': 'Exponential Smoothing'}
        
        return forecast, conf_int, metrics
        
    except Exception as e:
        st.error(f"Error in forecasting: {e}")
        return None, None, None

def create_trend_and_forecast_chart(historical_ts, forecast, conf_int, title, metric_type, show_ci=True):
    """Create comprehensive trend + forecast chart with growth indicators"""
    fig = go.Figure()
    
    # Calculate trend line for historical data
    x_hist = np.arange(len(historical_ts))
    z = np.polyfit(x_hist, historical_ts.values, 1)
    p = np.poly1d(z)
    trend_line = p(x_hist)
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=historical_ts.index,
        y=historical_ts.values,
        mode='lines+markers',
        name='Historical',
        line=dict(color='#3b82f6', width=2),
        marker=dict(size=4),
        hovertemplate='%{x|%b %Y}<br>%{y:,.2f}<extra></extra>'
    ))
    
    # Trend line
    fig.add_trace(go.Scatter(
        x=historical_ts.index,
        y=trend_line,
        mode='lines',
        name='Historical Trend',
        line=dict(color='#93c5fd', width=2, dash='dot'),
        hovertemplate='Trend: %{y:,.2f}<extra></extra>'
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast.index,
        y=forecast.values,
        mode='lines+markers',
        name='2026 Forecast',
        line=dict(color='#ef4444', width=3, dash='dash'),
        marker=dict(size=6),
        hovertemplate='%{x|%b %Y}<br>Forecast: %{y:,.2f}<extra></extra>'
    ))
    
    # Confidence interval
    if show_ci and conf_int is not None:
        fig.add_trace(go.Scatter(
            x=conf_int.index.tolist() + conf_int.index.tolist()[::-1],
            y=conf_int.iloc[:, 1].tolist() + conf_int.iloc[:, 0].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(239, 68, 68, 0.15)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% Confidence',
            showlegend=True,
            hoverinfo='skip'
        ))
    
    # Calculate growth
    last_value = historical_ts.iloc[-1]
    forecast_avg = forecast.mean()
    growth_pct = ((forecast_avg - last_value) / last_value) * 100
    
    # Add annotation for growth
    fig.add_annotation(
        x=forecast.index[6],
        y=forecast.values.max(),
        text=f"Expected Growth: {growth_pct:+.1f}%",
        showarrow=True,
        arrowhead=2,
        bgcolor='#fef3c7' if growth_pct > 0 else '#fee2e2',
        bordercolor='#f59e0b' if growth_pct > 0 else '#ef4444',
        borderwidth=2,
        font=dict(size=12, color='#000')
    )
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, weight='bold')),
        xaxis_title='Date',
        yaxis_title=metric_type,
        hovermode='x unified',
        height=450,
        template='plotly_white',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_combined_chart(revenue_ts, cogs_ts, profit_ts, 
                         revenue_forecast, cogs_forecast, profit_forecast,
                         category_name):
    """Create combined chart showing Revenue, COGS, and Profit"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Revenue Trend', 'COGS Trend', 'Gross Profit Trend', 'Profit Margin %'),
        specs=[[{}, {}], [{}, {}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # Revenue
    fig.add_trace(go.Scatter(x=revenue_ts.index, y=revenue_ts.values, 
                            name='Revenue (Hist)', line=dict(color='#3b82f6')),
                 row=1, col=1)
    fig.add_trace(go.Scatter(x=revenue_forecast.index, y=revenue_forecast.values,
                            name='Revenue (Forecast)', line=dict(color='#ef4444', dash='dash')),
                 row=1, col=1)
    
    # COGS
    if cogs_ts is not None:
        fig.add_trace(go.Scatter(x=cogs_ts.index, y=cogs_ts.values,
                                name='COGS (Hist)', line=dict(color='#f97316')),
                     row=1, col=2)
        fig.add_trace(go.Scatter(x=cogs_forecast.index, y=cogs_forecast.values,
                                name='COGS (Forecast)', line=dict(color='#dc2626', dash='dash')),
                     row=1, col=2)
    
    # Profit
    if profit_ts is not None:
        fig.add_trace(go.Scatter(x=profit_ts.index, y=profit_ts.values,
                                name='Profit (Hist)', line=dict(color='#16a34a')),
                     row=2, col=1)
        fig.add_trace(go.Scatter(x=profit_forecast.index, y=profit_forecast.values,
                                name='Profit (Forecast)', line=dict(color='#15803d', dash='dash')),
                     row=2, col=1)
        
        # Profit Margin %
        margin_hist = (profit_ts / revenue_ts * 100).dropna()
        margin_forecast = (profit_forecast / revenue_forecast * 100).dropna()
        
        fig.add_trace(go.Scatter(x=margin_hist.index, y=margin_hist.values,
                                name='Margin % (Hist)', line=dict(color='#8b5cf6')),
                     row=2, col=2)
        fig.add_trace(go.Scatter(x=margin_forecast.index, y=margin_forecast.values,
                                name='Margin % (Forecast)', line=dict(color='#7c3aed', dash='dash')),
                     row=2, col=2)
    
    fig.update_layout(
        height=700,
        showlegend=False,
        title_text=f"{category_name} - Complete Financial Overview",
        title_font_size=18
    )
    
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Amount", row=1, col=1)
    fig.update_yaxes(title_text="Amount", row=1, col=2)
    fig.update_yaxes(title_text="Amount", row=2, col=1)
    fig.update_yaxes(title_text="Margin %", row=2, col=2)
    
    return fig

# Main app
if st.session_state.df is not None:
    df = st.session_state.df
    
    # Parse data
    if st.session_state.parsed_data is None:
        parsed_categories, month_names = parse_revenue_cogs_data(df)
        st.session_state.parsed_data = {
            'categories': parsed_categories,
            'months': month_names
        }
    
    parsed_data = st.session_state.parsed_data
    categories = parsed_data['categories']
    
    # Data preview
    with st.expander("View Parsed Data Preview", expanded=False):
        preview_data = []
        for cat in categories:
            preview_data.append({
                'Category': cat['name'],
                'Data Type': 'Revenue',
                'First 3 Months': f"{cat['revenue'][:3]}"
            })
            if cat['cogs'] is not None:
                preview_data.append({
                    'Category': cat['name'],
                    'Data Type': 'COGS',
                    'First 3 Months': f"{cat['cogs'][:3]}"
                })
        st.dataframe(pd.DataFrame(preview_data), use_container_width=True)
        st.info(f"Found {len(categories)} categories with Revenue data. {sum(1 for c in categories if c['cogs'] is not None)} have COGS data.")
    
    # Run forecast button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Run Forecast for All Categories", type="primary", use_container_width=True):
            with st.spinner("Forecasting... Please wait..."):
                forecasts = {}
                progress_bar = st.progress(0)
                
                for idx, cat in enumerate(categories):
                    try:
                        # Forecast Revenue
                        revenue_ts = prepare_time_series(cat['revenue'])
                        revenue_forecast, revenue_ci, revenue_metrics = forecast_series(revenue_ts, forecast_method)
                        
                        # Forecast COGS
                        cogs_forecast, cogs_ci, cogs_metrics = None, None, None
                        if cat['cogs'] is not None:
                            cogs_ts = prepare_time_series(cat['cogs'])
                            cogs_forecast, cogs_ci, cogs_metrics = forecast_series(cogs_ts, forecast_method)
                        
                        # Calculate Profit forecast
                        profit_forecast, profit_ci = None, None
                        if cogs_forecast is not None and revenue_forecast is not None:
                            profit_forecast = revenue_forecast - cogs_forecast
                            # Simple CI for profit
                            profit_ci = pd.DataFrame({
                                'lower': revenue_ci.iloc[:, 0] - cogs_ci.iloc[:, 1],
                                'upper': revenue_ci.iloc[:, 1] - cogs_ci.iloc[:, 0]
                            }, index=profit_forecast.index)
                        
                        forecasts[cat['name']] = {
                            'revenue': {
                                'historical': revenue_ts,
                                'forecast': revenue_forecast,
                                'ci': revenue_ci,
                                'metrics': revenue_metrics
                            },
                            'cogs': {
                                'historical': prepare_time_series(cat['cogs']) if cat['cogs'] is not None else None,
                                'forecast': cogs_forecast,
                                'ci': cogs_ci,
                                'metrics': cogs_metrics
                            } if cat['cogs'] is not None else None,
                            'profit': {
                                'historical': prepare_time_series(cat['profit']) if cat['profit'] is not None else None,
                                'forecast': profit_forecast,
                                'ci': profit_ci
                            } if cat['profit'] is not None else None
                        }
                        
                        progress_bar.progress((idx + 1) / len(categories))
                        
                    except Exception as e:
                        st.warning(f"Could not forecast {cat['name']}: {e}")
                        continue
                
                st.session_state.forecasts = forecasts
                st.success(f"Successfully forecasted {len(forecasts)} categories!")
                st.rerun()
    
    # Display results
    if st.session_state.forecasts:
        forecasts = st.session_state.forecasts
        
        st.markdown("---")
        st.header("Forecast Results")
        
        # Summary metrics
        st.subheader("Overall 2026 Summary")
        
        total_revenue_2026 = sum([data['revenue']['forecast'].sum() for data in forecasts.values()])
        total_cogs_2026 = sum([data['cogs']['forecast'].sum() for data in forecasts.values() if data['cogs'] is not None and data['cogs']['forecast'] is not None])
        total_profit_2026 = total_revenue_2026 - total_cogs_2026 if total_cogs_2026 > 0 else 0
        overall_margin = (total_profit_2026 / total_revenue_2026 * 100) if total_revenue_2026 > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Revenue 2026", f"₦{total_revenue_2026:,.0f}")
        with col2:
            st.metric("Total COGS 2026", f"₦{total_cogs_2026:,.0f}")
        with col3:
            st.metric("Gross Profit 2026", f"₦{total_profit_2026:,.0f}", 
                     delta=f"{overall_margin:.1f}% margin")
        with col4:
            st.metric("Categories", len(forecasts))
        
        # Category selector
        st.markdown("---")
        st.subheader("Detailed Category Analysis")
        
        selected_category = st.selectbox(
            "Select a category to view details:",
            options=list(forecasts.keys())
        )
        
        if selected_category:
            data = forecasts[selected_category]
            
            # Show trend analysis if enabled
            if show_trends:
                st.subheader("Complete Financial Overview")
                combined_chart = create_combined_chart(
                    data['revenue']['historical'],
                    data['cogs']['historical'] if data['cogs'] else None,
                    data['profit']['historical'] if data['profit'] else None,
                    data['revenue']['forecast'],
                    data['cogs']['forecast'] if data['cogs'] and data['cogs']['forecast'] is not None else None,
                    data['profit']['forecast'] if data['profit'] and data['profit']['forecast'] is not None else None,
                    selected_category
                )
                st.plotly_chart(combined_chart, use_container_width=True)
            
            # Individual charts
            st.markdown("---")
            st.subheader("📈 Detailed Trend & Forecast Analysis")
            
            # Revenue Chart
            revenue_chart = create_trend_and_forecast_chart(
                data['revenue']['historical'],
                data['revenue']['forecast'],
                data['revenue']['ci'],
                f"{selected_category} - Revenue",
                "Revenue",
                show_confidence
            )
            st.plotly_chart(revenue_chart, use_container_width=True)
            
            # COGS Chart
            if data['cogs'] and data['cogs']['forecast'] is not None:
                cogs_chart = create_trend_and_forecast_chart(
                    data['cogs']['historical'],
                    data['cogs']['forecast'],
                    data['cogs']['ci'],
                    f"{selected_category} - COGS",
                    "COGS",
                    show_confidence
                )
                st.plotly_chart(cogs_chart, use_container_width=True)
            
            # Profit Chart
            if data['profit'] and data['profit']['forecast'] is not None:
                profit_chart = create_trend_and_forecast_chart(
                    data['profit']['historical'],
                    data['profit']['forecast'],
                    data['profit']['ci'],
                    f"{selected_category} - Gross Profit",
                    "Gross Profit",
                    show_confidence
                )
                st.plotly_chart(profit_chart, use_container_width=True)
            
            # Detailed Tables
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("2026 Monthly Forecast")
                months = ['January', 'February', 'March', 'April', 'May', 'June',
                         'July', 'August', 'September', 'October', 'November', 'December']
                
                forecast_table_data = {
                    'Month': months,
                    'Revenue': data['revenue']['forecast'].values
                }
                
                if data['cogs'] and data['cogs']['forecast'] is not None:
                    forecast_table_data['COGS'] = data['cogs']['forecast'].values
                    forecast_table_data['Gross Profit'] = data['profit']['forecast'].values
                    forecast_table_data['Margin %'] = (data['profit']['forecast'] / data['revenue']['forecast'] * 100).values
                
                forecast_df = pd.DataFrame(forecast_table_data)
                
                st.dataframe(
                    forecast_df.style.format({
                        'Revenue': '₦{:,.2f}',
                        'COGS': '₦{:,.2f}' if 'COGS' in forecast_df.columns else None,
                        'Gross Profit': '₦{:,.2f}' if 'Gross Profit' in forecast_df.columns else None,
                        'Margin %': '{:.1f}%' if 'Margin %' in forecast_df.columns else None
                    }),
                    use_container_width=True,
                    height=460
                )
            
            with col2:
                st.subheader("2026 Statistics")
                
                revenue_values = data['revenue']['forecast'].values
                stats_data = {
                    'Metric': ['Total Revenue', 'Avg Monthly Revenue', 'Min Month', 'Max Month']
                }
                stats_data['Value'] = [
                    f"₦{revenue_values.sum():,.2f}",
                    f"₦{revenue_values.mean():,.2f}",
                    f"₦{revenue_values.min():,.2f}",
                    f"₦{revenue_values.max():,.2f}"
                ]
                
                if data['cogs'] and data['cogs']['forecast'] is not None:
                    cogs_values = data['cogs']['forecast'].values
                    profit_values = data['profit']['forecast'].values
                    
                    stats_data['Metric'].extend(['Total COGS', 'Total Gross Profit', 'Avg Margin %'])
                    stats_data['Value'].extend([
                        f"₦{cogs_values.sum():,.2f}",
                        f"₦{profit_values.sum():,.2f}",
                        f"{(profit_values.sum() / revenue_values.sum() * 100):.1f}%"
                    ])
                
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
                
                # Model metrics
                if data['revenue']['metrics']:
                    st.markdown("**Revenue Model Performance:**")
                    for key, value in data['revenue']['metrics'].items():
                        if isinstance(value, (int, float)):
                            st.metric(key, f"{value:.2f}")
        
        # Comparison charts
        st.markdown("---")
        st.subheader("Category Comparison - 2026 Forecast")
        
        comparison_data = []
        for category, data in forecasts.items():
            row = {
                'Category': category,
                'Revenue': data['revenue']['forecast'].sum()
            }
            if data['cogs'] and data['cogs']['forecast'] is not None:
                row['COGS'] = data['cogs']['forecast'].sum()
                row['Gross Profit'] = data['profit']['forecast'].sum()
                row['Margin %'] = (row['Gross Profit'] / row['Revenue'] * 100)
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data).sort_values('Revenue', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Revenue comparison
            fig_revenue = px.bar(
                comparison_df,
                x='Category',
                y='Revenue',
                title='Total 2026 Revenue by Category',
                color='Revenue',
                color_continuous_scale='Blues'
            )
            fig_revenue.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_revenue, use_container_width=True)
        
        with col2:
            if 'Margin %' in comparison_df.columns:
                fig_margin = px.bar(
                    comparison_df,
                    x='Category',
                    y='Margin %',
                    title='Profit Margin % by Category (2026)',
                    color='Margin %',
                    color_continuous_scale='Greens'
                )
                fig_margin.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_margin, use_container_width=True)
        
        # Export section
        st.markdown("---")
        st.subheader("Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Export Revenue forecast
            revenue_export = []
            months = ['January-2026', 'February-2026', 'March-2026', 'April-2026',
                     'May-2026', 'June-2026', 'July-2026', 'August-2026',
                     'September-2026', 'October-2026', 'November-2026', 'December-2026']
            
            for category, data in forecasts.items():
                row = {'Category': category}
                for i, month in enumerate(months):
                    row[month] = data['revenue']['forecast'].values[i]
                revenue_export.append(row)
            
            revenue_export_df = pd.DataFrame(revenue_export)
            csv_revenue = revenue_export_df.to_csv(index=False)
            
            st.download_button(
                label="Download Revenue Forecast",
                data=csv_revenue,
                file_name="revenue_forecast_2026.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # Export COGS forecast
            cogs_export = []
            for category, data in forecasts.items():
                if data['cogs'] and data['cogs']['forecast'] is not None:
                    row = {'Category': category}
                    for i, month in enumerate(months):
                        row[month] = data['cogs']['forecast'].values[i]
                    cogs_export.append(row)
            
            if cogs_export:
                cogs_export_df = pd.DataFrame(cogs_export)
                csv_cogs = cogs_export_df.to_csv(index=False)
                
                st.download_button(
                    label="Download COGS Forecast",
                    data=csv_cogs,
                    file_name="cogs_forecast_2026.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col3:
            # Export Profit forecast
            profit_export = []
            for category, data in forecasts.items():
                if data['profit'] and data['profit']['forecast'] is not None:
                    row = {'Category': category}
                    for i, month in enumerate(months):
                        row[month] = data['profit']['forecast'].values[i]
                    profit_export.append(row)
            
            if profit_export:
                profit_export_df = pd.DataFrame(profit_export)
                csv_profit = profit_export_df.to_csv(index=False)
                
                st.download_button(
                    label="Download Profit Forecast",
                    data=csv_profit,
                    file_name="profit_forecast_2026.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        # Summary table export
        st.markdown("---")
        summary_csv = comparison_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Complete Summary",
            data=summary_csv,
            file_name="forecast_summary_2026.csv",
            mime="text/csv",
            use_container_width=True
        )
else:
    # Welcome screen
    st.info("Welcome! Please upload your CSV file, connect to Google Sheets, or use sample data to get started.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### CSV Format Requirements:
        
        Your CSV should have categories listed **twice** in the same order:
        - **First N rows**: Revenue data for each category
        - **Next N rows**: COGS data for the same categories (same order)
        
        #### Example Structure:
        ```
        RevenueUnit,January-2022,February-2022,...
        ALLERGY COUGH & FLU,168050,966433.08,...    ← Revenue
        ANTIMALARIAL,103530,738359.43,...           ← Revenue
        BEAUTY,29750,20452,...                      ← Revenue
        ALLERGY COUGH & FLU,120000,690000,...       ← COGS
        ANTIMALARIAL,72000,515000,...               ← COGS
        BEAUTY,21000,14500,...                      ← COGS
        ```
        """)
    
    with col2:
        st.markdown("""
        ###  Google Sheets Integration:
        
        **Option 1: Public Sheet (Easiest)**
        1. Open your Google Sheet
        2. Click Share → Change to "Anyone with link"
        3. Set permission to "Viewer"
        4. Copy the URL
        5. Paste in the sidebar
        
        **Option 2: Private Sheet**
        1. Create a Google Cloud project
        2. Enable Google Sheets API
        3. Create Service Account
        4. Download JSON credentials
        5. Share sheet with service account email
        6. Upload credentials in sidebar
        
        [📖 Detailed Setup Guide](https://docs.gspread.org/en/latest/oauth2.html)
        """)
    
    st.markdown("""
    ### What This App Does:
    - Forecasts 2026 Revenue for each category
    - Forecasts 2026 COGS for each category
    - Calculates Gross Profit automatically
    - Shows profit margins and trends
    - Provides growth indicators
    - Creates interactive charts with trend analysis
    - Exports all results to CSV
    - **Syncs with Google Sheets in real-time!**
    
    ### Advanced Features:
    - **Trend Analysis**: See historical trends with forecast projections
    - **Growth Indicators**: Automatic calculation of expected growth rates
    - **Confidence Intervals**: Understand forecast uncertainty
    - **Margin Analysis**: Track profitability by category
    - **Complete Dashboard**: Revenue, COGS, Profit, and Margins in one view
    - **Google Sheets Sync**: Connect directly to your live data
    """)
    
    st.markdown("---")
    st.markdown("*Powered by statsmodels, plotly, gspread, and Streamlit*")


