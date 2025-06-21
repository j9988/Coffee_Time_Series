import numpy as np
import pandas as pd
import seaborn as sns
import logging
import warnings
import calendar
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import zscore
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from streamlit_option_menu import option_menu
from pmdarima import auto_arima

# --- Suppress Warnings ---
logging.getLogger("cmdstanpy").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

st.set_page_config(
    page_title="Coffee Time Series Analysis and Forecasting",
    page_icon="☕",  
    layout="wide"   
)

# Load the dataset
@st.cache_data
def load_cleaned_data():
    df = pd.read_csv("Dataset/cleaned_dataset.csv")
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df

@st.cache_data
def load_count_data():
    df = pd.read_csv("Dataset/daily_coffee_count.csv")
    df['date'] = pd.to_datetime(df['date'])
    return df

@st.cache_data
def load_sales_data():
    df = pd.read_csv("Dataset/daily_coffee_sales.csv")
    df['date'] = pd.to_datetime(df['date'])
    return df

# --- MAPE Function ---
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

## -------------------------------------------------------------------------------

# Sidebar navigation
with st.sidebar:
    page = option_menu(
        "Navigation",  # Menu title
        ["EDA - Coffee Data", "Traditional - SARIMA", "Machine Learning - FB Prophet"],  # Menu items
        icons=["bar-chart", "activity", "robot"],
        menu_icon="list",  # Top icon
        default_index=0  # Default selected page
    )

## -------------------------------------------------------------------------------

if page == "EDA - Coffee Data":
    st.title("Exploratory Data Analysis - Coffee Data")
    df_cleaned = load_cleaned_data()

    st.write("## Coffee Dataset Sample")
    st.dataframe(df_cleaned.head())

    # Payment Method Pie Chart
    st.write("## Payment Method Distribution")
    payment_method = df_cleaned['payment_method'].value_counts()
    fig1, ax1 = plt.subplots(figsize=(5, 5))
    colors = sns.color_palette('pastel')[0:len(payment_method)]
    ax1.pie(payment_method.values, labels=payment_method.index, autopct='%1.1f%%', startangle=90, colors=colors, wedgeprops={'edgecolor': 'black'})
    ax1.set_title('Payment Method')
    st.pyplot(fig1)

    # Hourly Sales Line Chart
    st.write("## Hourly Sales Distribution")
    hourly_sales = df_cleaned['hour'].value_counts().sort_index()
    fig2, ax2 = plt.subplots(figsize=(10,4))
    ax2.plot(hourly_sales.index, hourly_sales.values, marker='o')
    ax2.set_xticks(range(24))
    ax2.set_xlabel('Hour of the day')
    ax2.set_ylabel('Number of sales')
    ax2.set_title('Hourly sales distribution')
    ax2.grid(True)
    st.pyplot(fig2)

    # Orders by Day of Week
    st.write("## Orders by Day of the Week")
    weekday_types = df_cleaned.groupby('day_of_week').size().reset_index(name='count')
    fig3, ax3 = plt.subplots(figsize=(8,4))
    barplot = sns.barplot(data=weekday_types, y='day_of_week', x='count', palette='pastel', ax=ax3)
    # Add bar labels
    for i, bar in enumerate(barplot.patches):
        width = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2
        ax3.text(width + 1, y, f'{int(width)}', va='center')
    ax3.set_title('Number of Orders by Day of the Week')
    ax3.set_xlabel('Number of Orders')
    ax3.set_ylabel('Day of the Week')
    st.pyplot(fig3)

    # Sales by Year-Month
    st.write("## Sales by Year-Month")
    sales_month = df_cleaned.groupby(['year', 'month'])['money'].sum().reset_index()
    sales_month['year_month'] = sales_month['year'].astype(str) + '-' + sales_month['month'].astype(str).str.zfill(2)
    fig4, ax4 = plt.subplots(figsize=(12,4))
    ax4.plot(sales_month['year_month'], sales_month['money'], marker='o', linestyle='-', color='b')
    ax4.set_xticklabels(sales_month['year_month'], rotation=45)
    ax4.set_xlabel('Year-Month')
    ax4.set_ylabel('Sales')
    ax4.set_title('Sales by Year-Month')
    ax4.grid(True)
    st.pyplot(fig4)

    # Top 10 Coffee by Orders
    st.write("## Top 10 Coffee by Number of Orders")
    top10_orders = df_cleaned.groupby('coffee_name').size().reset_index(name='num_orders').sort_values(by='num_orders', ascending=False).head(10)
    fig5, ax5 = plt.subplots(figsize=(10,4))
    sns.barplot(data=top10_orders, x='coffee_name', y='num_orders', palette='pastel', ax=ax5)
    ax5.set_title('Top 10 Coffee by Number of Orders')
    ax5.set_ylabel('Number of Orders')
    ax5.set_xlabel('Coffee Name')
    ax5.set_xticklabels(ax5.get_xticklabels(), rotation=45, ha='right')
    st.pyplot(fig5)
    
    # Top 10 Coffee by Sales
    st.write("## Top 10 Coffee by Sales")
    top10_sales = df_cleaned.groupby('coffee_name')['money'].sum().reset_index().sort_values(by='money', ascending=False).head(10)
    fig5b, ax5b = plt.subplots(figsize=(10,4))
    sns.barplot(data=top10_sales, x='coffee_name', y='money', palette='pastel', ax=ax5b)
    ax5b.set_title('Top 10 Coffee by Sales')
    ax5b.set_ylabel('Total Sales')
    ax5b.set_xlabel('Coffee Name')
    ax5b.set_xticklabels(ax5b.get_xticklabels(), rotation=45, ha='right')
    st.pyplot(fig5b)

    # Daily Sales Line Chart
    st.write("## Daily Sales for the Year")
    daily_sales = df_cleaned.groupby(df_cleaned['datetime'].dt.date)['money'].sum()
    fig6, ax6 = plt.subplots(figsize=(12,4))
    daily_sales.plot(kind='line', ax=ax6)
    ax6.set_title('Daily Sales for the Year')
    ax6.set_xlabel('Date')
    ax6.set_ylabel('Total Sales')
    ax6.grid(True)
    st.pyplot(fig6)

    # Total Sales by Month
    st.write("## Total Sales by Month")
    fig, ax = plt.subplots(figsize=(12, 5))  
    monthly_sales = df_cleaned.groupby(df_cleaned['datetime'].dt.month)['money'].sum().reset_index()
    monthly_sales['month_name'] = monthly_sales['datetime'].apply(lambda x: calendar.month_name[x])
    monthly_sales = monthly_sales.sort_values('datetime')
    sns.barplot(data=monthly_sales, x='month_name', y='money', ax=ax, palette='pastel')
    ax.set_xlabel('Month')
    ax.set_ylabel('Total Sales')
    ax.set_title('Total Sales of Coffee per Month')
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'${height:,.0f}', 
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom',
                    xytext=(0, 10), textcoords='offset points')
    y_max = monthly_sales['money'].max()
    ax.set_ylim(0, y_max * 1.10)
    st.pyplot(fig)
    
    # Total Sales by Time of Day
    st.write("## Total Sales by Time of Day")
    fig, ax = plt.subplots(figsize=(12, 5))  
    hourly_sales = df_cleaned.groupby(df_cleaned['datetime'].dt.hour)['money'].sum().reset_index()
    sns.barplot(data=hourly_sales, x='datetime', y='money', ax=ax, palette='pastel')
    ax.set_xlabel('Hour')
    ax.set_ylabel('Total Sales')
    ax.set_title('Total Sales of Coffee per Hour')
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'${height:,.0f}', 
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom',
                    xytext=(0, 10), textcoords='offset points')
    y_max = hourly_sales['money'].max()
    ax.set_ylim(0, y_max * 1.10)
    st.pyplot(fig)

    # Smoothing Process
    st.write("## Smoothing Process")
    df_sales = df_cleaned.groupby('datetime').size().reset_index(name='num_orders')
    df_sales['datetime'] = pd.to_datetime(df_sales['datetime'])
    df_sales = df_sales.set_index('datetime').resample('D').sum().fillna(0).reset_index()
    df_sales['money_ma7'] = df_sales['num_orders'].rolling(window=7, min_periods=1).mean()
    df_sales['EMA_0.3'] = df_sales['num_orders'].ewm(alpha=0.3, adjust=False).mean()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=df_sales, x='datetime', y='num_orders', label='Daily Sales', ax=ax, color='skyblue')
    sns.lineplot(data=df_sales, x='datetime', y='money_ma7', label='7-Day Moving Average', ax=ax, color='orange')
    sns.lineplot(data=df_sales, x='datetime', y='EMA_0.3', label='EMA (α=0.3)', ax=ax, color='lightseagreen')
    ax.set_title('Coffee Sales with Smoothing Process')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales (Money)')
    ax.legend()
    st.pyplot(fig)

    # Anomaly Detection using Z-Score Method
    st.write("## Anomaly Detection in Daily Orders (Z-Score Method)")
    df_sales = df_cleaned.groupby('datetime').size().reset_index(name='num_orders')
    df_sales['datetime'] = pd.to_datetime(df_sales['datetime'])
    df_sales = df_sales.set_index('datetime').resample('D').sum().fillna(0).reset_index()
    df_sales['z_score'] = zscore(df_sales['num_orders'].fillna(0))
    anomaly_threshold = 3
    df_sales['anomaly'] = df_sales['z_score'].apply(lambda x: 1 if abs(x) > anomaly_threshold else 0)
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(df_sales['datetime'], df_sales['num_orders'], label='Daily Orders', linewidth=1)
    ax.scatter(
        df_sales[df_sales['anomaly'] == 1]['datetime'],
        df_sales[df_sales['anomaly'] == 1]['num_orders'],
        color='red', label='Anomalies', s=80
    )
    ax.set_title('Anomaly Detection in Daily Orders using Z-score')
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of Orders')
    ax.legend()
    st.pyplot(fig)

elif page == "Traditional - SARIMA":
    st.title("Traditional Time Series Forecasting - SARIMA")
    st.sidebar.header("🔧 Forecast Settings")
    target = st.sidebar.selectbox("What would you like to forecast?", ["Orders", "Sales"])
    forecast_days = st.sidebar.slider("Forecast Horizon (Days)", min_value=21, max_value=180, value=21, step=7)

    # --- Load data based on selection ---
    if target == "Orders":
        df = load_count_data()
        df = df.rename(columns={"date": "ds", "order_count": "y"})
        y_label = "Cups Ordered"
    else:
        df = load_sales_data()
        df = df.rename(columns={"date": "ds", "total_sales": "y"})
        y_label = "Sales Revenue"
        
    df = df.sort_values('ds')
    df.set_index('ds', inplace=True)
    df.index = pd.to_datetime(df.index)
    df = df.asfreq('D')
    df = df.fillna(0)

    recent_df = df.tail(21)
    # --- Fit SARIMA ---
    st.write("### Fitting SARIMA Model")
    with st.spinner("Fitting SARIMA model..."):
        sarima_model = auto_arima(
            recent_df['y'],
            seasonal=True,
            m=7,
            trace=False,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True
        )

    st.code(sarima_model.summary())

## ---------------------------------------------------------------------------------
elif page == "Machine Learning - FB Prophet":
    st.title("Machine Learning Time Series Forecasting - FB Prophet")
    
    st.sidebar.header("🔧 Forecast Settings")

    target = st.sidebar.selectbox("What would you like to forecast?", ["Orders", "Sales"])
    forecast_days = st.sidebar.slider("Forecast Horizon (Days)", min_value=21, max_value=180, value=21, step=7)

    # --- Load data based on selection ---
    if target == "Orders":
        df = load_count_data()
        df = df.rename(columns={"date": "ds", "order_count": "y"})
        y_label = "Cups Ordered"
    else:
        df = load_sales_data()
        df = df.rename(columns={"date": "ds", "total_sales": "y"})
        y_label = "Sales Revenue"
        
    df['ds'] = pd.to_datetime(df['ds'])
    df = df[['ds', 'y']].copy()
    df['y'] = np.log1p(df['y'])  # log-transform
    
    test_days = 21
    train_df = df[:-test_days]
    test_df = df[-test_days:]
    
    # --- Build and Train Prophet ---
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=15 if target == "Orders" else 1
    )
    model.add_country_holidays(country_name='MY')
    model.fit(train_df)
    
    # --- Forecast ---
    future = model.make_future_dataframe(periods=forecast_days, freq='D')
    forecast = model.predict(future)
    forecast['yhat'] = np.expm1(forecast['yhat']).clip(lower=0)
    
    # --- Plot forecast vs actual (only last 21 days if available) ---
    st.subheader(f"📈 Forecast vs Actual ({target})")

    if forecast_days == 0:
        y_true = np.expm1(test_df['y'].values)
        y_pred = forecast['yhat'].iloc[-test_days:].values

        fig1, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(test_df['ds'], y_true, label='Actual', marker='o')
        ax1.plot(test_df['ds'], y_pred, label='Predicted', marker='x')
        ax1.set_title(f"Prophet Forecast vs Actual ({target})")
        ax1.set_xlabel("Date")
        ax1.set_ylabel(y_label)
        ax1.legend()
        ax1.grid()
        st.pyplot(fig1)

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = mean_absolute_percentage_error(y_true, y_pred)

        st.markdown("**📊 Forecast Accuracy:**")
        st.write(f"MAE: {mae:.2f}")
        st.write(f"RMSE: {rmse:.2f}")
        st.write(f"MAPE: {mape:.2f}%")

    else:
        # Plot full forecast
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.plot(df['ds'], np.expm1(df['y']), label="Historical", color='dodgerblue')
        ax2.plot(forecast['ds'], forecast['yhat'], label="Forecast", color='violet')
        start_date = df['ds'].max()
        end_date = forecast['ds'].max()
        ax2.axvspan(start_date, end_date, color='lightgray', alpha=0.3, label='Forecast Period')
        ax2.set_title(f"{target} Forecast for Next {forecast_days} Days")
        ax2.set_xlabel("Date")
        ax2.set_ylabel(y_label)
        ax2.legend()
        ax2.grid()
        st.pyplot(fig2)

    # --- Component Plots ---
    st.subheader("🔍 Seasonal Components")
    fig3 = model.plot_components(forecast)
    st.pyplot(fig3)


