import streamlit as st
import pandas as pd
import google.generativeai as genai
import json
from datetime import datetime
import plotly.graph_objects as go

st.set_page_config(page_title="Revenue Forecast 2026", layout="wide")

st.title("2026 Revenue Forecast")
st.markdown("Sales prediction based on your Google Sheet data")

with st.sidebar:
    st.header("Configuration")
    sheet_url = st.text_input(
        "Google Sheet URL",
        placeholder="https://docs.google.com/spreadsheets/d/...",
        help="Paste your Google Sheet URL here"
    )
    sheet_name = st.text_input("Sheet Name", value="Sheet1")
    gemini_key = st.text_input("Google Gemini API Key", type="password", help="Get free key from https://aistudio.google.com/app/apikeys")

if st.button("Load Data & Generate Forecast", use_container_width=True):
    if not sheet_url or not gemini_key:
        st.error("Please provide both Google Sheet URL and Gemini API Key")
    else:
        with st.spinner("Loading data from Google Sheet..."):
            try:
                sheet_id = sheet_url.split('/d/')[1].split('/')[0]
                url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
                df = pd.read_csv(url)
                
                df.columns = df.columns.str.strip()
                
                st.write("**Sheet Preview:**")
                st.write(f"Total rows: {len(df)}, Total columns: {len(df.columns)}")
                st.write("**Column headers:**")
                st.write(df.columns.tolist())
                st.write("**First few rows:**")
                st.write(df.head())
                
                # Get revenue row (first data row after headers)
                revenue_row = df.iloc[0]
                cogs_row = df.iloc[51] if len(df) > 51 else None
                
                # Extract months starting from column C (index 2)
                monthly_totals = []
                
                # Get column names starting from index 2
                for col_idx in range(2, len(df.columns)):
                    col_name = str(df.columns[col_idx]).strip()
                    
                    # Check if column name matches format like "March-2026"
                    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
                    has_month = any(month in col_name for month in months)
                    has_hyphen = '-' in col_name
                    has_year = '20' in col_name
                    
                    if has_month and has_hyphen and has_year:
                        try:
                            revenue_value = pd.to_numeric(revenue_row.iloc[col_idx], errors='coerce')
                            cogs_value = pd.to_numeric(cogs_row.iloc[col_idx], errors='coerce') if cogs_row is not None else None
                            
                            if pd.notna(revenue_value) and revenue_value > 0:
                                monthly_totals.append({
                                    'Month': col_name, 
                                    'Revenue': revenue_value,
                                    'COGS': cogs_value if pd.notna(cogs_value) else 0
                                })
                        except Exception as e:
                            st.warning(f"Could not parse {col_name}: {str(e)}")
                
                if not monthly_totals:
                    st.error("No valid monthly data found. Check that your sheet has month columns in format like 'March-2026' starting from column C")
                    st.stop()
                
                df_processed = pd.DataFrame(monthly_totals)
                
                if len(df_processed) >= 2:
                    
                    st.success(f"Loaded {len(df_processed)} months of data")
                    
                    with st.expander("View Historical Data"):
                        st.dataframe(df_processed, use_container_width=True)
                    
                    with st.spinner("Analyzing your data..."):
                        genai.configure(api_key=gemini_key)
                        model = genai.GenerativeModel('gemini-pro')
                        
                        data_str = "\n".join([
                            f"{row['Month']}: Revenue=${row['Revenue']:,.0f}, COGS=${row['COGS']:,.0f}" 
                            for _, row in df_processed.iterrows()
                        ])
                        
                        prompt = f"""Analyze this monthly revenue and COGS data and forecast both for all 12 months of 2026.

Historical Data (monthly revenue and COGS):
{data_str}

Please analyze trends, seasonality, and patterns for both revenue and COGS. Generate separate forecasts for each metric for 2026 with confidence intervals.

Return ONLY a JSON object with these exact fields:
- methodology: string explaining your approach
- keyInsights: array of 3 strings with key findings
- forecast: array of exactly 12 objects with these fields for each month (Jan 2026 through Dec 2026): month (string like "Jan 2026"), revenue_predicted (number), revenue_lower (number), revenue_upper (number), cogs_predicted (number), cogs_lower (number), cogs_upper (number)
- annualRevenue: number (sum of all 12 predicted revenue months)
- annualCOGS: number (sum of all 12 predicted COGS months)
- annualProfit: number (annualRevenue - annualCOGS)
- profitMargin: string (like "35%")
- growthRate: string (like "10%" or "-5%")
- avgHistoricalRevenue: number
- avgHistoricalCOGS: number

Make sure forecast array has exactly 12 months. Return only valid JSON, no other text."""
                        
                        response = model.generate_content(prompt)
                        response_text = response.text
                        
                        json_start = response_text.find('{')
                        json_end = response_text.rfind('}') + 1
                        forecast_data = json.loads(response_text[json_start:json_end])
                        
                        st.session_state.forecast = forecast_data
                        st.session_state.historical = df_processed
                        st.success("Forecast generated successfully!")
                else:
                    st.error("Sheet must have at least 2 months of data")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")

if 'forecast' in st.session_state:
    forecast_data = st.session_state.forecast
    df_historical = st.session_state.historical
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("2026 Annual Revenue", f"${forecast_data['annualRevenue']:,.0f}")
    with col2:
        st.metric("2026 Annual COGS", f"${forecast_data['annualCOGS']:,.0f}")
    with col3:
        st.metric("2026 Annual Profit", f"${forecast_data['annualProfit']:,.0f}")
    with col4:
        st.metric("Profit Margin", forecast_data['profitMargin'])
    
    st.divider()
    
    st.subheader("Revenue Trend & 2026 Forecast")
    
    df_forecast = pd.DataFrame(forecast_data['forecast'])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_historical['Month'],
        y=df_historical['Revenue'],
        mode='lines+markers',
        name='Historical Revenue',
        line=dict(color='#3b82f6', width=2),
        marker=dict(size=6)
    ))
    
    fig.add_trace(go.Scatter(
        x=df_historical['Month'],
        y=df_historical['COGS'],
        mode='lines+markers',
        name='Historical COGS',
        line=dict(color='#ef4444', width=2),
        marker=dict(size=6)
    ))
    
    fig.add_trace(go.Scatter(
        x=df_forecast['month'],
        y=df_forecast['revenue_predicted'],
        mode='lines+markers',
        name='2026 Revenue Forecast',
        line=dict(color='#3b82f6', width=2, dash='dash'),
        marker=dict(size=6)
    ))
    
    fig.add_trace(go.Scatter(
        x=df_forecast['month'],
        y=df_forecast['cogs_predicted'],
        mode='lines+markers',
        name='2026 COGS Forecast',
        line=dict(color='#ef4444', width=2, dash='dash'),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        hovermode='x unified',
        height=500,
        template='plotly_white',
        xaxis_title='Month',
        yaxis_title='Revenue ($)',
        yaxis=dict(tickformat='$,.0f')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    st.subheader("Key Insights")
    for insight in forecast_data['keyInsights']:
        st.info(insight)
    
    st.divider()
    
    st.subheader("2026 Monthly Forecast Details")
    df_display = pd.DataFrame(forecast_data['forecast'])
    df_display.columns = ['Month', 'Revenue', 'Rev Lower', 'Rev Upper', 'COGS', 'COGS Lower', 'COGS Upper']
    
    df_display['Revenue'] = df_display['Revenue'].apply(lambda x: f"${x:,.0f}")
    df_display['Rev Lower'] = df_display['Rev Lower'].apply(lambda x: f"${x:,.0f}")
    df_display['Rev Upper'] = df_display['Rev Upper'].apply(lambda x: f"${x:,.0f}")
    df_display['COGS'] = df_display['COGS'].apply(lambda x: f"${x:,.0f}")
    df_display['COGS Lower'] = df_display['COGS Lower'].apply(lambda x: f"${x:,.0f}")
    df_display['COGS Upper'] = df_display['COGS Upper'].apply(lambda x: f"${x:,.0f}")
    
    st.dataframe(df_display, use_container_width=True, hide_index=True)
    
    st.divider()
    
    with st.expander("Forecast Methodology"):
        st.write(forecast_data['methodology'])
    
    st.download_button(
        label="Download Forecast as CSV",
        data=df_display.to_csv(index=False),
        file_name=f"revenue_forecast_2026_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True
    )
