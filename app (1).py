import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Inventory Demand Forecasting",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Premium Look ---
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        font-weight: 800;
        margin-bottom: 20px;
        padding-top: 20px;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
        border-top: 4px solid #3b82f6;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-label {
        font-size: 1rem;
        color: #64748b;
        font-weight: 500;
        margin-top: 5px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f1f5f9;
        border-radius: 5px;
        color: #1e293b;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6;
        color: white;
    }
    div[data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---

def load_data(uploaded_file):
    """Loads dataset from uploaded file or default file."""
    try:
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        else:
            # Fallback to default
            df = pd.read_csv("inventory_demand_data.csv.xls")
            
        if 'Date' not in df.columns or 'Demand' not in df.columns:
            st.error("Dataset must contain 'Date' and 'Demand' columns.")
            return None
            
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Handle missing values using linear interpolation
        if df['Demand'].isnull().sum() > 0:
            df['Demand'] = df['Demand'].interpolate(method='linear')
            
        # Add Demand Change metrics
        df['Previous_Demand'] = df['Demand'].shift(1)
        df['Change_%'] = ((df['Demand'] - df['Previous_Demand']) / df['Previous_Demand']) * 100
        df['Change_%'] = df['Change_%'].fillna(0) # Handle NaN
            
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def create_features(df):
    """Creates time-based and lag features for modeling."""
    df = df.copy()
    
    # Time-based features
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    df['DayOfMonth'] = df['Date'].dt.day
    df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
    
    # Lag features (past values)
    df['Lag_1'] = df['Demand'].shift(1)
    df['Lag_7'] = df['Demand'].shift(7)
    
    # Rolling averages
    df['Rolling_7_Mean'] = df['Demand'].shift(1).rolling(window=7).mean()
    df['Rolling_14_Mean'] = df['Demand'].shift(1).rolling(window=14).mean()
    
    # Drop rows with NaN values created by shifting
    df = df.dropna().reset_index(drop=True)
    
    return df

@st.cache_resource
def train_models(df):
    """Trains multiple machine learning models and evaluates them."""
    feature_cols = ['DayOfWeek', 'Month', 'Quarter', 'DayOfMonth', 'IsWeekend', 
                    'Lag_1', 'Lag_7', 'Rolling_7_Mean', 'Rolling_14_Mean']
    
    X = df[feature_cols]
    y = df['Demand']
    
    # Use standard train-test split for time series
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    dates_test = df['Date'].iloc[split_idx:]
    
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    trained_models = {}
    metrics = {}
    predictions = {'Date': dates_test, 'Actual': y_test}
    
    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Predict
        preds = model.predict(X_test)
        predictions[name] = preds
        
        # Evaluate
        metrics[name] = {
            'MAE': mean_absolute_error(y_test, preds),
            'RMSE': np.sqrt(mean_squared_error(y_test, preds)),
            'R2': r2_score(y_test, preds)
        }
        
    return trained_models, pd.DataFrame(predictions), metrics, feature_cols

def forecast_future(model, last_known_data, days_to_predict, feature_cols):
    """Generates future forecasts using recursive prediction."""
    # Start with the last known data points
    current_data = last_known_data.copy()
    future_dates = pd.date_range(start=current_data['Date'].iloc[-1] + pd.Timedelta(days=1), periods=days_to_predict)
    
    forecasts = []
    
    for date in future_dates:
        # Create a new row for the future date
        new_row = pd.DataFrame({'Date': [date]})
        new_row['DayOfWeek'] = date.dayofweek
        new_row['Month'] = date.month
        new_row['Quarter'] = date.quarter
        new_row['DayOfMonth'] = date.day
        new_row['IsWeekend'] = int(date.dayofweek >= 5)
        
        # Calculate lags based on current_data
        new_row['Lag_1'] = current_data['Demand'].iloc[-1]
        new_row['Lag_7'] = current_data['Demand'].iloc[-7]
        new_row['Rolling_7_Mean'] = current_data['Demand'].iloc[-7:].mean()
        new_row['Rolling_14_Mean'] = current_data['Demand'].iloc[-14:].mean()
        
        # Predict
        X_future = new_row[feature_cols]
        pred = max(0, model.predict(X_future)[0]) # Ensure no negative demand
        
        forecasts.append(pred)
        
        # Append prediction to current_data to be used for next lag calculations
        new_row['Demand'] = pred
        current_data = pd.concat([current_data, new_row], ignore_index=True)
        
    future_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted Demand': np.round(forecasts).astype(int)
    })
    
    return future_df

# --- Sidebar Configuration ---
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>⚙️ Control Panel</h2>", unsafe_allow_html=True)
    st.write("---")
    
    uploaded_file = st.file_uploader("📂 Upload Dataset (CSV/Excel)", type=['csv', 'xlsx', 'xls'])
    if uploaded_file is None:
        st.info("Using default project dataset (`inventory_demand_data.csv.xls`).")
        
    st.write("---")
    menu = st.radio("Navigation", ["🏠 Dashboard", "📊 Model Evaluation", "📈 Future Forecast", "ℹ️ About Project"])

# --- Main Application Logic ---

df_raw = load_data(uploaded_file)

if df_raw is not None:
    df_features = create_features(df_raw)
    trained_models, predictions_df, metrics, feature_cols = train_models(df_features)
    
    if menu == "🏠 Dashboard":
        st.markdown("<div class='main-title'>Inventory Demand Dashboard</div>", unsafe_allow_html=True)
        
        # Top Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"<div class='metric-card'><div class='metric-value'>{len(df_raw)}</div><div class='metric-label'>Total Days Logged</div></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='metric-card'><div class='metric-value'>{df_raw['Demand'].mean():.0f}</div><div class='metric-label'>Avg Daily Demand</div></div>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<div class='metric-card'><div class='metric-value'>{df_raw['Demand'].max():.0f}</div><div class='metric-label'>Peak Demand</div></div>", unsafe_allow_html=True)
        with col4:
            best_model_name = max(metrics, key=lambda k: metrics[k]['R2'])
            st.markdown(f"<div class='metric-card'><div class='metric-value'>{metrics[best_model_name]['R2']*100:.1f}%</div><div class='metric-label'>Best R² ({best_model_name})</div></div>", unsafe_allow_html=True)
            
        st.write("---")
        
        # Alert System
        st.subheader("🔔 Recent Demand Alerts")
        latest_row = df_raw.iloc[-1]
        change_pct = latest_row['Change_%']
        
        if change_pct > 30:
            st.error(f"🚨 Demand Spike Alert: Demand increased by {change_pct:.1f}%. Suggestion: Increase stock to meet unexpected high demand.")
        elif change_pct < -30:
            st.warning(f"⚠️ Demand Drop Alert: Demand decreased by {change_pct:.1f}%. Suggestion: Reduce stock or run promotions to clear inventory.")
        else:
            st.success(f"✅ Demand Stable: Demand changed by {change_pct:.1f}%. Suggestion: Continue normal operations.")
            
        st.write("---")
        
        # Visualizations
        tab1, tab2, tab3 = st.tabs(["📉 Demand Trend", "📅 Seasonality", "📊 Distribution"])
        
        with tab1:
            fig_trend = px.line(df_raw, x='Date', y='Demand', title="Historical Demand Trend", 
                                line_shape='spline', color_discrete_sequence=['#3b82f6'])
            fig_trend.update_layout(xaxis_title="", yaxis_title="Units Sold", hovermode="x unified")
            st.plotly_chart(fig_trend, use_container_width=True)
            
        with tab2:
            seasonality_df = df_raw.copy()
            seasonality_df['Month_Name'] = seasonality_df['Date'].dt.month_name()
            monthly_avg = seasonality_df.groupby('Month_Name')['Demand'].mean().reset_index()
            fig_season = px.bar(monthly_avg, x='Month_Name', y='Demand', title="Average Monthly Demand (Seasonality)",
                                category_orders={"Month_Name": ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]},
                                color_discrete_sequence=['#10b981'], text_auto='.0f')
            fig_season.update_layout(xaxis_title="", yaxis_title="Average Units Sold")
            st.plotly_chart(fig_season, use_container_width=True)
            
        with tab3:
            sorted_demand = df_raw['Demand'].sort_values().reset_index(drop=True)
            fig_dist = px.area(x=sorted_demand.index, y=sorted_demand, title="Demand Profile (Lowest to Highest Days)",
                               color_discrete_sequence=['#8b5cf6'])
            fig_dist.update_layout(xaxis_title="Number of Days", yaxis_title="Demand Quantity")
            st.plotly_chart(fig_dist, use_container_width=True)

    elif menu == "📊 Model Evaluation":
        st.markdown("<div class='main-title'>Machine Learning Performance</div>", unsafe_allow_html=True)
        
        # Metrics Table
        st.subheader("Model Comparison Matrix")
        metrics_df = pd.DataFrame(metrics).T.sort_values(by='R2', ascending=False)
        st.dataframe(metrics_df.style.highlight_max(subset=['R2'], color='lightgreen')
                                        .highlight_min(subset=['MAE', 'RMSE'], color='lightgreen'), 
                     use_container_width=True)
        
        st.write("---")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Prediction Accuracy (Test Set)")
            selected_model = st.selectbox("Select Model to Visualize:", list(trained_models.keys()), index=list(trained_models.keys()).index('Random Forest') if 'Random Forest' in trained_models else 0)
            
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(x=predictions_df['Date'], y=predictions_df['Actual'], mode='lines', name='Actual', line=dict(color='gray', width=2)))
            fig_pred.add_trace(go.Scatter(x=predictions_df['Date'], y=predictions_df[selected_model], mode='lines', name=f'Predicted', line=dict(color='#ef4444', width=2)))
            fig_pred.update_layout(title=f"Actual vs Predicted - {selected_model}", hovermode="x unified")
            st.plotly_chart(fig_pred, use_container_width=True)
            
        with col2:
            st.subheader("Feature Importance")
            if selected_model in ['Random Forest']:
                model = trained_models[selected_model]
                importance = pd.DataFrame({
                    'Feature': feature_cols,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=True)
                
                fig_imp = px.bar(importance, x='Importance', y='Feature', orientation='h', title=f"Drivers ({selected_model})", color_discrete_sequence=['#f59e0b'])
                fig_imp.update_layout(margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.info(f"Feature importance is not supported for {selected_model}.")
                
        # --- NEW ENHANCEMENT 5: Model Comparison Chart ---
        st.write("---")
        st.subheader("📊 R² Score Comparison")
        fig_comp = px.bar(metrics_df.reset_index(), x='index', y='R2', title="Model Performance (R² Score)", color='R2', color_continuous_scale='viridis')
        fig_comp.update_layout(xaxis_title="Model", yaxis_title="R² Score")
        st.plotly_chart(fig_comp, use_container_width=True)

    elif menu == "📈 Future Forecast":
        st.markdown("<div class='main-title'>Business Forecasting Engine</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### Forecast Parameters")
            forecast_horizon = st.slider("Forecast Horizon (Days):", min_value=1, max_value=30, value=7)
            model_to_use = st.selectbox("Select Forecasting Algorithm:", list(trained_models.keys()), index=list(trained_models.keys()).index('Random Forest') if 'Random Forest' in trained_models else 0)
            
            st.markdown("### Actionable Insights")
            generate_btn = st.button("Generate Forecast 🚀", use_container_width=True, type="primary")
            
        with col2:
            if generate_btn:
                with st.spinner("Running recursive forecasting algorithm..."):
                    time.sleep(1) # Visual feedback
                    
                    model = trained_models[model_to_use]
                    future_df = forecast_future(model, df_features, forecast_horizon, feature_cols)
                    
                    # Confidence bounds approximation (using RMSE)
                    rmse = metrics[model_to_use]['RMSE']
                    future_df['Upper Bound'] = future_df['Predicted Demand'] + (rmse * 1.96) # 95% CI approx
                    future_df['Lower Bound'] = np.maximum(0, future_df['Predicted Demand'] - (rmse * 1.96))
                    
                    # Plot
                    fig_forecast = go.Figure()
                    
                    # Add historical data (last 30 days context)
                    historical_context = df_features.tail(30)
                    fig_forecast.add_trace(go.Scatter(x=historical_context['Date'], y=historical_context['Demand'], mode='lines', name='Historical Demand', line=dict(color='gray')))
                    
                    # Add bounds
                    fig_forecast.add_trace(go.Scatter(x=future_df['Date'], y=future_df['Upper Bound'], mode='lines', line=dict(width=0), showlegend=False))
                    fig_forecast.add_trace(go.Scatter(x=future_df['Date'], y=future_df['Lower Bound'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(59, 130, 246, 0.2)', name='95% Confidence Interval'))
                    
                    # Add prediction
                    fig_forecast.add_trace(go.Scatter(x=future_df['Date'], y=future_df['Predicted Demand'], mode='lines+markers', name='Forecasted Demand', line=dict(color='#3b82f6', width=3)))
                    
                    fig_forecast.update_layout(title=f"{forecast_horizon}-Day Forecast using {model_to_use}", hovermode="x unified")
                    st.plotly_chart(fig_forecast, use_container_width=True)
                    
        if generate_btn:
            st.write("---")
            total_predicted = future_df['Predicted Demand'].sum()
            avg_predicted = future_df['Predicted Demand'].mean()
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Forecasted Demand", f"{total_predicted:,} units")
            c2.metric("Average Daily Demand", f"{avg_predicted:.1f} units")
            c3.metric("Recommended Safety Stock", f"{int(total_predicted + (rmse * np.sqrt(forecast_horizon)))} units", help="Forecast + RMSE adjusted for time")
            
            st.success(f"💡 **Recommendation:** To maintain optimal inventory levels, ensure at least **{int(total_predicted + (rmse * np.sqrt(forecast_horizon)))} units** are available in the warehouse over the next {forecast_horizon} days to cover predicted demand and potential volatility.")
            
            # Forecast Alerts
            historical_avg = df_raw['Demand'].mean()
            max_future = future_df['Predicted Demand'].max()
            min_future = future_df['Predicted Demand'].min()
            
            if max_future > (historical_avg * 1.30):
                st.error(f"🚨 Forecast Spike Alert: A predicted demand of {max_future:.0f} units exceeds 130% of historical average ({historical_avg:.0f}). Suggestion: Increase upcoming stock orders.")
            elif min_future < (historical_avg * 0.70):
                st.warning(f"⚠️ Forecast Drop Alert: A predicted demand of {min_future:.0f} units falls below 70% of historical average ({historical_avg:.0f}). Suggestion: Reduce upcoming stock orders.")
            
            with st.expander("View Raw Forecast Data"):
                st.dataframe(future_df[['Date', 'Predicted Demand', 'Lower Bound', 'Upper Bound']], use_container_width=True)

    elif menu == "ℹ️ About Project":
        st.markdown("<div class='main-title'>Project Abstract</div>", unsafe_allow_html=True)
        st.write("""
        ### Inventory Demand Forecasting using Machine Learning
        
        Inventory demand forecasting helps businesses predict future product demand using past sales data. In this project, Machine Learning techniques are used to improve prediction accuracy. The system analyzes historical data, identifies patterns, and forecasts demand. This helps reduce overstock and stock shortages, improving overall business efficiency.
        
        #### Architecture Flow:
        `Data Collection` → `Preprocessing` → `Model Training` → `Prediction` → `Output`
        
        #### Algorithms Used:
        - Linear Regression (Baseline)
        - **Random Forest (Proposed - Higher Accuracy)**
        """)
