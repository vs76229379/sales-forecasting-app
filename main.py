import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import plotly.express as px

@st.cache_data
def load_data():
    df = pd.read_csv('sales_data_sample.csv', encoding='latin1')
    df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'])
    return df

df = load_data()

# Sidebar filters
st.sidebar.header("Filters")
regions = df['TERRITORY'].dropna().unique()
selected_region = st.sidebar.selectbox("Region", regions)
forecast_months = st.sidebar.slider("Months to Forecast", 1, 12, 3)

# Filter data
region_data = df[df['TERRITORY'] == selected_region]

# Main interface
st.title("📈 Sales Dashboard (Kaggle Sample Data)")

st.header("Data Overview")
st.write(f"Showing data for **{selected_region}** region")
st.write(region_data.head())

st.subheader("Sales Trend")
monthly_sales = region_data.resample('M', on='ORDERDATE')['SALES'].sum()
fig = px.line(x=monthly_sales.index, y=monthly_sales.values, title="Monthly Sales Trend")
st.plotly_chart(fig)

st.header("Sales Forecasting")
if len(region_data) > 1:
    features = region_data[['QUANTITYORDERED', 'PRICEEACH']]
    target = region_data['SALES']
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    future_dates = pd.date_range(start=df['ORDERDATE'].max(), periods=forecast_months, freq='M')
    avg_quantity = X_train['QUANTITYORDERED'].mean()
    avg_price = X_train['PRICEEACH'].mean()
    forecast = model.predict([[avg_quantity, avg_price]] * forecast_months)
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Forecasted Sales': forecast.round(2)
    })
    st.write(forecast_df.set_index('Date'))
    st.download_button(
        label="Download Forecast",
        data=forecast_df.to_csv(),
        file_name='sales_forecast.csv'
    )

st.header("Sales vs Quantity")
if len(region_data) > 1:
    fig = px.scatter(region_data, x='SALES', y='QUANTITYORDERED', color='PRODUCTLINE')
    st.plotly_chart(fig)
