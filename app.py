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
    sheet_name = st.text_input("Sheet Name", value="DEPARTMENT COGS")
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
                
                # Your data has Revenue/Unit in first 2 columns, then months
                # Sum all the monthly data across all product categories
                monthly_columns = df.columns[2:]  # All columns after "Revenue" and "Unit"
                
                # Calculate total revenue per month
                monthly_totals = []
                for col in monthly_columns:
                    try:
                        total = pd.to_numeric(df[col], errors='coerce').sum()
                        monthly_totals.append({'Month': col, 'Revenue': total})
                    except:
                        pass
                
                df_processed = pd.DataFrame(monthly_totals)
                df_processed = df_processed.dropna()
                
                if len(df_processed) >= 2:
                    
                    st.success(f"Loaded {len(df_processed)} months of data")
                    
                    with st.expander("View Historical Data"):
                        st.dataframe(df_processed, use_container_width=True)
                    
                    with st.spinner("Analyzing your data..."):
                        genai.configure(api_key=gemini_key)
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        
                        data_str = "\n".join([
                            f"{row['Month']}: ${row['Revenue']:,.0f}" 
                            for _, row in df_processed.iterrows()
                        ])
                        
                        prompt = f"""Analyze this monthly revenue data and forecast the revenue for all 12 months of 2026.

Historical Data (monthly revenue):
{data_str}

Please analyze trends, seasonality, and patterns. Generate forecasts for each month of 2026 with confidence intervals.

Return ONLY a JSON object with these exact fields:
- methodology: string explaining your approach
- keyInsights: array of 3 strings with key findings
- forecast: array of exactly 12 objects with these fields for each month (Jan 2026 through Dec 2026): month (string like "Jan 2026"), predicted (number), lower_bound (number), upper_bound (number)
- annualTotal: number (sum of all 12 predicted months)
- growthRate: string (like "10%" or "-5%")
- avgHistorical: number (average of historical data)

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
                    st.error("Sheet must have at least 2 columns")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")

if 'forecast' in st.session_state:
    forecast_data = st.session_state.forecast
    df_historical = st.session_state.historical
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("2026 Annual Total", f"${forecast_data['annualTotal']:,.0f}")
    with col2:
        st.metric("Growth Rate", forecast_data['growthRate'])
    with col3:
        st.metric("Avg Historical", f"${forecast_data['avgHistorical']:,.0f}")
    
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
        x=df_forecast['month'],
        y=df_forecast['predicted'],
        mode='lines+markers',
        name='2026 Forecast',
        line=dict(color='#f59e0b', width=2, dash='dash'),
        marker=dict(size=6)
    ))
    
    fig.add_trace(go.Scatter(
        x=df_forecast['month'],
        y=df_forecast['upper_bound'],
        fill=None,
        mode='lines',
        line_color='rgba(245,158,11,0)',
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=df_forecast['month'],
        y=df_forecast['lower_bound'],
        fill='tonexty',
        mode='lines',
        line_color='rgba(245,158,11,0)',
        name='Confidence Interval',
        fillcolor='rgba(245,158,11,0.2)'
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
    df_display.columns = ['Month', 'Predicted Revenue', 'Lower Bound', 'Upper Bound']
    
    df_display['Predicted Revenue'] = df_display['Predicted Revenue'].apply(lambda x: f"${x:,.0f}")
    df_display['Lower Bound'] = df_display['Lower Bound'].apply(lambda x: f"${x:,.0f}")
    df_display['Upper Bound'] = df_display['Upper Bound'].apply(lambda x: f"${x:,.0f}")
    
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
