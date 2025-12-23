"""
Revenue & COGS Forecasting Streamlit App
Interactive web app for forecasting 2026 revenue, COGS, and profit

Installation:
pip install streamlit pandas numpy statsmodels plotly gspread oauth2client

To run:
streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
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
        creds_dict = json.loads(credentials_json)
        scope = ['https://spreadsheets.google.com/feeds',
                'https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        sheet = client.open_by_url(sheet_url)
        worksheet = sheet.get_worksheet(0)
        data = worksheet.get_all_values()
        df = pd.DataFrame(data[1:], columns=data[0])
        return df, None
    except Exception as e:
        return None, str(e)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    data_source = st.radio(
        "Select Data Source:",
        ["Upload CSV File", "Google Sheets"],
        help="Choose how to load your data"
    )
    
    if data_source == "Upload CSV File":
        uploaded_file = st.file_uploader(
            "Upload your CSV file", 
            type=['csv'],
            help="First N rows = Revenue, Next N rows = COGS"
        )
        
        if uploaded_file is not None:
            st.session_state.df = pd.read_csv(uploaded_file)
            st.session_state.gsheet_connected = False
            st.success(f"File uploaded: {uploaded_file.name}")
            st.info(f"Total rows: {len(st.session_state.df)}")
    
    else:
        if not GSPREAD_AVAILABLE:
            st.error("Install: pip install gspread oauth2client")
        else:
            st.markdown("### Connect to Google Sheets")
            
            connection_method = st.radio(
                "Connection Method:",
                ["Public Sheet (Read-only)", "Private Sheet (Credentials)"]
            )
            
            if connection_method == "Public Sheet (Read-only)":
                st.info("Make sheet public: Share → Anyone with link")
                
                sheet_url = st.text_input(
                    "Google Sheet URL:",
                    placeholder="https://docs.google.com/spreadsheets/d/..."
                )
                
                if st.button("🔗 Connect to Sheet"):
                    if sheet_url:
                        with st.spinner("Connecting..."):
                            try:
                                sheet_id = sheet_url.split('/d/')[1].split('/')[0]
                                csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
                                df = pd.read_csv(csv_url)
                                st.session_state.df = df
                                st.session_state.gsheet_connected = True
                                st.success("Connected!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed: {e}")
            
            else:
                credentials_file = st.file_uploader("Upload Service Account JSON:", type=['json'])
                sheet_url = st.text_input("Google Sheet URL:", key="private_sheet_url")
                
                if st.button("Connect with Credentials"):
                    if credentials_file and sheet_url:
                        with st.spinner("Connecting..."):
                            credentials_json = credentials_file.read().decode('utf-8')
                            df, error = load_from_google_sheets(sheet_url, credentials_json)
                            if df is not None:
                                st.session_state.df = df
                                st.session_state.gsheet_connected = True
                                st.success("Connected!")
                                st.rerun()
                            else:
                                st.error(f"Failed: {error}")
            
            if st.session_state.gsheet_connected:
                if st.button("🔄 Refresh Data"):
                    st.rerun()
    
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
    st.subheader("🔧 Model Settings")
    forecast_method = st.selectbox(
        "Forecasting Method",
        ["SARIMA", "Exponential Smoothing"],
        help="SARIMA is better for seasonal data"
    )
    
    show_confidence = st.checkbox("Show Confidence Intervals", value=True)
    show_trends = st.checkbox("Show Trend Analysis", value=True)

# Helper functions
def parse_revenue_cogs_data(df):
    """Parse CSV where first N rows are Revenue, next N rows are COGS"""
    categories = []
    category_names = []
    seen = set()
    
    for cat in df.iloc[:, 0]:
        if cat not in seen:
            category_names.append(cat)
            seen.add(cat)
    
    for cat_name in category_names:
        cat_rows = df[df.iloc[:, 0] == cat_name]
        
        if len(cat_rows) >= 2:
            revenue_row = cat_rows.iloc[0, 1:].values
            cogs_row = cat_rows.iloc[1, 1:].values
            
            categories.append({
                'name': cat_name,
                'revenue': revenue_row.astype(float),
                'cogs': cogs_row.astype(float),
                'profit': revenue_row.astype(float) - cogs_row.astype(float)
            })
        elif len(cat_rows) == 1:
            revenue_row = cat_rows.iloc[0, 1:].values
            categories.append({
                'name': cat_name,
                'revenue': revenue_row.astype(float),
                'cogs': None,
                'profit': None
            })
    
    return categories, df.columns[1:]

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
            
            metrics = {'AIC': fitted_model.aic, 'BIC': fitted_model.bic}
            
        else:
            model = ExponentialSmoothing(ts, seasonal='add', seasonal_periods=12, trend='add')
            fitted_model = model.fit()
            forecast = fitted_model.forecast(steps=12)
            conf_int = pd.DataFrame({
                'lower': forecast * 0.9,
                'upper': forecast * 1.1
            }, index=forecast.index)
            metrics = {'Method': 'Exponential Smoothing'}
        
        return forecast, conf_int, metrics
    except Exception as e:
        st.error(f"Error: {e}")
        return None, None, None

def create_trend_and_forecast_chart(historical_ts, forecast, conf_int, title, metric_type, show_ci=True):
    """Create trend + forecast chart"""
    fig = go.Figure()
    
    x_hist = np.arange(len(historical_ts))
    z = np.polyfit(x_hist, historical_ts.values, 1)
    p = np.poly1d(z)
    trend_line = p(x_hist)
    
    fig.add_trace(go.Scatter(x=historical_ts.index, y=historical_ts.values,
                            mode='lines+markers', name='Historical',
                            line=dict(color='#3b82f6', width=2), marker=dict(size=4)))
    
    fig.add_trace(go.Scatter(x=historical_ts.index, y=trend_line,
                            mode='lines', name='Trend',
                            line=dict(color='#93c5fd', width=2, dash='dot')))
    
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values,
                            mode='lines+markers', name='Forecast',
                            line=dict(color='#ef4444', width=3, dash='dash'),
                            marker=dict(size=6)))
    
    if show_ci and conf_int is not None:
        fig.add_trace(go.Scatter(
            x=conf_int.index.tolist() + conf_int.index.tolist()[::-1],
            y=conf_int.iloc[:, 1].tolist() + conf_int.iloc[:, 0].tolist()[::-1],
            fill='toself', fillcolor='rgba(239, 68, 68, 0.15)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% Confidence'))
    
    last_value = historical_ts.iloc[-1]
    forecast_avg = forecast.mean()
    growth_pct = ((forecast_avg - last_value) / last_value) * 100
    
    fig.add_annotation(x=forecast.index[6], y=forecast.values.max(),
                      text=f"Growth: {growth_pct:+.1f}%", showarrow=True,
                      bgcolor='#fef3c7' if growth_pct > 0 else '#fee2e2')
    
    fig.update_layout(title=title, xaxis_title='Date', yaxis_title=metric_type,
                     height=450, template='plotly_white', hovermode='x unified')
    return fig

def create_combined_chart(revenue_ts, cogs_ts, profit_ts, 
                         revenue_forecast, cogs_forecast, profit_forecast, category_name):
    """Create combined 4-panel chart"""
    fig = make_subplots(rows=2, cols=2,
                       subplot_titles=('Revenue', 'COGS', 'Profit', 'Margin %'),
                       vertical_spacing=0.12, horizontal_spacing=0.1)
    
    fig.add_trace(go.Scatter(x=revenue_ts.index, y=revenue_ts.values,
                            name='Rev (Hist)', line=dict(color='#3b82f6')), row=1, col=1)
    fig.add_trace(go.Scatter(x=revenue_forecast.index, y=revenue_forecast.values,
                            name='Rev (Forecast)', line=dict(color='#ef4444', dash='dash')), row=1, col=1)
    
    if cogs_ts is not None:
        fig.add_trace(go.Scatter(x=cogs_ts.index, y=cogs_ts.values,
                                name='COGS (Hist)', line=dict(color='#f97316')), row=1, col=2)
        fig.add_trace(go.Scatter(x=cogs_forecast.index, y=cogs_forecast.values,
                                name='COGS (Forecast)', line=dict(color='#dc2626', dash='dash')), row=1, col=2)
    
    if profit_ts is not None:
        fig.add_trace(go.Scatter(x=profit_ts.index, y=profit_ts.values,
                                name='Profit (Hist)', line=dict(color='#16a34a')), row=2, col=1)
        fig.add_trace(go.Scatter(x=profit_forecast.index, y=profit_forecast.values,
                                name='Profit (Forecast)', line=dict(color='#15803d', dash='dash')), row=2, col=1)
        
        margin_hist = (profit_ts / revenue_ts * 100).dropna()
        margin_forecast = (profit_forecast / revenue_forecast * 100).dropna()
        
        fig.add_trace(go.Scatter(x=margin_hist.index, y=margin_hist.values,
                                name='Margin (Hist)', line=dict(color='#8b5cf6')), row=2, col=2)
        fig.add_trace(go.Scatter(x=margin_forecast.index, y=margin_forecast.values,
                                name='Margin (Forecast)', line=dict(color='#7c3aed', dash='dash')), row=2, col=2)
    
    fig.update_layout(height=700, showlegend=False,
                     title_text=f"{category_name} - Financial Overview")
    return fig

# Main app
if st.session_state.df is not None:
    df = st.session_state.df
    
    if st.session_state.parsed_data is None:
        parsed_categories, month_names = parse_revenue_cogs_data(df)
        st.session_state.parsed_data = {
            'categories': parsed_categories,
            'months': month_names
        }
    
    parsed_data = st.session_state.parsed_data
    categories = parsed_data['categories']
    
    with st.expander("View Data Preview"):
        st.info(f"Found {len(categories)} categories")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Run Forecast", type="primary", use_container_width=True):
            with st.spinner("Forecasting..."):
                forecasts = {}
                progress_bar = st.progress(0)
                
                for idx, cat in enumerate(categories):
                    try:
                        revenue_ts = prepare_time_series(cat['revenue'])
                        revenue_forecast, revenue_ci, revenue_metrics = forecast_series(revenue_ts, forecast_method)
                        
                        cogs_forecast, cogs_ci, cogs_metrics = None, None, None
                        if cat['cogs'] is not None:
                            cogs_ts = prepare_time_series(cat['cogs'])
                            cogs_forecast, cogs_ci, cogs_metrics = forecast_series(cogs_ts, forecast_method)
                        
                        profit_forecast, profit_ci = None, None
                        if cogs_forecast is not None and revenue_forecast is not None:
                            profit_forecast = revenue_forecast - cogs_forecast
                            profit_ci = pd.DataFrame({
                                'lower': revenue_ci.iloc[:, 0] - cogs_ci.iloc[:, 1],
                                'upper': revenue_ci.iloc[:, 1] - cogs_ci.iloc[:, 0]
                            }, index=profit_forecast.index)
                        
                        forecasts[cat['name']] = {
                            'revenue': {'historical': revenue_ts, 'forecast': revenue_forecast,
                                      'ci': revenue_ci, 'metrics': revenue_metrics},
                            'cogs': {'historical': prepare_time_series(cat['cogs']) if cat['cogs'] is not None else None,
                                    'forecast': cogs_forecast, 'ci': cogs_ci, 'metrics': cogs_metrics} if cat['cogs'] is not None else None,
                            'profit': {'historical': prepare_time_series(cat['profit']) if cat['profit'] is not None else None,
                                     'forecast': profit_forecast, 'ci': profit_ci} if cat['profit'] is not None else None
                        }
                        
                        progress_bar.progress((idx + 1) / len(categories))
                    except Exception as e:
                        st.warning(f"Could not forecast {cat['name']}: {e}")
                
                st.session_state.forecasts = forecasts
                st.success(f"Forecasted {len(forecasts)} categories!")
                st.rerun()
    
    if st.session_state.forecasts:
        forecasts = st.session_state.forecasts
        
        st.markdown("---")
        st.header("Forecast Results")
        
        total_revenue_2026 = sum([data['revenue']['forecast'].sum() for data in forecasts.values()])
        total_cogs_2026 = sum([data['cogs']['forecast'].sum() for data in forecasts.values() 
                              if data['cogs'] is not None and data['cogs']['forecast'] is not None])
        total_profit_2026 = total_revenue_2026 - total_cogs_2026
        overall_margin = (total_profit_2026 / total_revenue_2026 * 100) if total_revenue_2026 > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Revenue 2026", f"₦{total_revenue_2026:,.0f}")
        with col2:
            st.metric("Total COGS 2026", f"₦{total_cogs_2026:,.0f}")
        with col3:
            st.metric("Gross Profit 2026", f"₦{total_profit_2026:,.0f}")
        with col4:
            st.metric("Profit Margin", f"{overall_margin:.1f}%")
        
        st.markdown("---")
        st.subheader("Category Analysis")
        
        selected_category = st.selectbox("Select category:", options=list(forecasts.keys()))
        
        if selected_category:
            data = forecasts[selected_category]
            
            if show_trends:
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
            
            st.markdown("---")
            
            revenue_chart = create_trend_and_forecast_chart(
                data['revenue']['historical'], data['revenue']['forecast'],
                data['revenue']['ci'], f"{selected_category} - Revenue",
                "Revenue", show_confidence
            )
            st.plotly_chart(revenue_chart, use_container_width=True)
            
            if data['cogs'] and data['cogs']['forecast'] is not None:
                cogs_chart = create_trend_and_forecast_chart(
                    data['cogs']['historical'], data['cogs']['forecast'],
                    data['cogs']['ci'], f"{selected_category} - COGS",
                    "COGS", show_confidence
                )
                st.plotly_chart(cogs_chart, use_container_width=True)
            
            if data['profit'] and data['profit']['forecast'] is not None:
                profit_chart = create_trend_and_forecast_chart(
                    data['profit']['historical'], data['profit']['forecast'],
                    data['profit']['ci'], f"{selected_category} - Profit",
                    "Profit", show_confidence
                )
                st.plotly_chart(profit_chart, use_container_width=True)
            
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("2026 Monthly Forecast")
                months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                
                forecast_table_data = {
                    'Month': months,
                    'Revenue': data['revenue']['forecast'].values
                }
                
                if data['cogs'] and data['cogs']['forecast'] is not None:
                    forecast_table_data['COGS'] = data['cogs']['forecast'].values
                    forecast_table_data['Profit'] = data['profit']['forecast'].values
                
                forecast_df = pd.DataFrame(forecast_table_data)
                st.dataframe(forecast_df.style.format({
                    'Revenue': '₦{:,.0f}',
                    'COGS': '₦{:,.0f}' if 'COGS' in forecast_df.columns else None,
                    'Profit': '₦{:,.0f}' if 'Profit' in forecast_df.columns else None
                }), use_container_width=True)
            
            with col2:
                st.subheader("Statistics")
                if data['revenue']['metrics']:
                    for key, value in data['revenue']['metrics'].items():
                        if isinstance(value, (int, float)):
                            st.metric(key, f"{value:.2f}")
        
        st.markdown("---")
        st.subheader("Category Comparison")
        
        comparison_data = []
        for category, data in forecasts.items():
            row = {'Category': category, 'Revenue': data['revenue']['forecast'].sum()}
            if data['cogs'] and data['cogs']['forecast'] is not None:
                row['COGS'] = data['cogs']['forecast'].sum()
                row['Profit'] = data['profit']['forecast'].sum()
                row['Margin %'] = (row['Profit'] / row['Revenue'] * 100)
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data).sort_values('Revenue', ascending=False)
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(comparison_df, x='Category', y='Revenue',
                        title='2026 Revenue by Category', color='Revenue')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'Margin %' in comparison_df.columns:
                fig = px.bar(comparison_df, x='Category', y='Margin %',
                           title='Profit Margin %', color='Margin %')
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            revenue_export = []
            months = ['Jan-26', 'Feb-26', 'Mar-26', 'Apr-26', 'May-26', 'Jun-26',
                     'Jul-26', 'Aug-26', 'Sep-26', 'Oct-26', 'Nov-26', 'Dec-26']
            
            for category, data in forecasts.items():
                row = {'Category': category}
                for i, month in enumerate(months):
                    row[month] = data['revenue']['forecast'].values[i]
                revenue_export.append(row)
            
            csv_revenue = pd.DataFrame(revenue_export).to_csv(index=False)
            st.download_button("Download Revenue", csv_revenue,
                             "revenue_2026.csv", "text/csv", use_container_width=True)
        
        with col2:
            cogs_export = []
            for category, data in forecasts.items():
                if data['cogs'] and data['cogs']['forecast'] is not None:
                    row = {'Category': category}
                    for i, month in enumerate(months):
                        row[month] = data['cogs']['forecast'].values[i]
                    cogs_export.append(row)
            
            if cogs_export:
                csv_cogs = pd.DataFrame(cogs_export).to_csv(index=False)
                st.download_button("Download COGS", csv_cogs,
                                 "cogs_2026.csv", "text/csv", use_container_width=True)
        
        with col3:
            profit_export = []
            for category, data in forecasts.items():
                if data['profit'] and data['profit']['forecast'] is not None:
                    row = {'Category': category}
                    for i, month in enumerate(months):
                        row[month] = data['profit']['forecast'].values[i]
                    profit_export.append(row)
            
            if profit_export:
                csv_profit = pd.DataFrame(profit_export).to_csv(index=False)
                st.download_button("Download Profit", csv_profit,
                                 "profit_2026.csv", "text/csv", use_container_width=True)
        
        st.markdown("---")
        csv_summary = comparison_df.to_csv(index=False)
        st.download_button("📥 Download Summary", csv_summary,
                         "summary_2026.csv", "text/csv", use_container_width=True)
        
        st.markdown("---")
        st.subheader("📊 Month-on-Month Revenue")
        
        mom_data = []
        for category, data in forecasts.items():
            hist_ts = data['revenue']['historical']
            forecast_series = data['revenue']['forecast']
            all_dates = list(hist_ts.index) + list(forecast_series.index)
            all_values = list(hist_ts.values) + list(forecast_series.values)
            
            for date, value in zip(all_dates, all_values):
                mom_data.append({
                    'Department': category,
                    'Month': date.strftime('%B'),
                    'Year': date.year,
                    'Date': date.strftime('%Y-%m'),
                    'Revenue': value
                })
        
        mom_df = pd.DataFrame(mom_data)
        
        with st.expander("📋 Preview Month-on-Month"):
            pivot_df = mom_df.pivot(index='Department', columns='Date', values='Revenue')
            st.dataframe(pivot_df.style.format('₦{:,.2f}'), use_container_width=True, height=300)
        
        csv_mom = mom_df.to_csv(index=False)
        st.download_button("📥 Download Month-on-Month (All Periods)", csv_mom,
                         "month_on_month_revenue.csv", "text/csv",
                         use_container_width=True, type="primary")
        
        st.info("💡 Includes historical (2022-2024) AND 2026 forecasts")

else:
    st.info("👋 Upload CSV, connect to Google Sheets, or use sample data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### CSV Format
        Categories listed twice:
        - First N rows: Revenue
        - Next N rows: COGS (same order)
        
        ```
        RevenueUnit,Jan-2022,Feb-2022,...
        CATEGORY_1,100,200,...    ← Revenue
        CATEGORY_2,150,250,...    ← Revenue
        CATEGORY_1,80,150,...     ← COGS
        CATEGORY_2,100,180,...    ← COGS
        ```
        """)
    
    with col2:
        st.markdown("""
        ### Features
        - Forecasts 2026 Revenue & COGS
        - Calculates Gross Profit
        - Trend analysis with growth indicators
        - Confidence intervals
        - Interactive charts
        - Google Sheets integration
        - Month-on-month reports
        - Export to CSV
        """)
    
    st.markdown("---")
    st.markdown("*Powered by statsmodels, plotly, and Streamlit*")
