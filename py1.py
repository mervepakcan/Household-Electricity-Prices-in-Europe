import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error

monthly_data = pd.read_csv("EMBER european_wholesale_electricity_price_data_monthly.csv")
monthly_data['Date'] = pd.to_datetime(monthly_data['Date'])
monthly_data = monthly_data.sort_values(by=['Country', 'Date'])
monthly_data['Price (EUR/MWhe)'] = (
    monthly_data.groupby('Country')['Price (EUR/MWhe)']
    .apply(lambda group: group.ffill().bfill())
    .reset_index(level=0, drop=True)
)

results = []

for country in monthly_data['Country'].unique():
    print(f"Processing {country}...")

    data = monthly_data[monthly_data['Country'] == country].copy()
    data.set_index('Date', inplace=True)
    data.index = pd.date_range(start=data.index[0], periods=len(data), freq='MS')

    data = data.dropna()

    train_data = data[:'2023'].copy()
    test_data = data['2024':].copy()

    arima_model = SARIMAX(train_data['Price (EUR/MWhe)'], order=(1, 1, 1), seasonal_order=(1, 1, 0, 12))
    arima_fit = arima_model.fit(disp=False)

    future_steps = len(test_data) + (11 * 12)
    arima_forecast = arima_fit.get_forecast(steps=future_steps).predicted_mean
    arima_residuals = train_data['Price (EUR/MWhe)'] - arima_fit.predict(start=train_data.index[0], end=train_data.index[-1])

    amplification_factor = 2
    residuals_amplified = (amplification_factor * arima_residuals).values.reshape(-1, 1)
    scaler = StandardScaler()
    residuals_scaled = scaler.fit_transform(residuals_amplified)

    sequence_length = 36
    X_lstm, y_lstm = [], []
    for i in range(sequence_length, len(residuals_scaled)):
        X_lstm.append(residuals_scaled[i-sequence_length:i])
        y_lstm.append(residuals_scaled[i])

    X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)

    lstm_model = Sequential([
        Input(shape=(sequence_length, 1)),
        LSTM(128, activation='relu', return_sequences=True),
        Dropout(0.3),
        LSTM(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mse')
    lstm_model.fit(X_lstm, y_lstm, epochs=50, batch_size=32, verbose=0)

    last_sequence = residuals_scaled[-sequence_length:]
    lstm_predictions = []
    for _ in range(future_steps):
        lstm_pred = lstm_model.predict(last_sequence.reshape(1, sequence_length, 1))
        lstm_predictions.append(lstm_pred[0][0])
        last_sequence = np.append(last_sequence[1:], lstm_pred, axis=0)

    lstm_predictions = scaler.inverse_transform(np.array(lstm_predictions).reshape(-1, 1)).flatten()

    alpha = 0.7
    hybrid_forecast = alpha * arima_forecast + (1 - alpha) * lstm_predictions
    hybrid_forecast = np.maximum(hybrid_forecast, 0)

    train_data.loc[:, 'Year'] = train_data.index.year
    train_data.loc[:, 'Month'] = train_data.index.month
    test_data.loc[:, 'Year'] = test_data.index.year
    test_data.loc[:, 'Month'] = test_data.index.month

    X_train = train_data[['Year', 'Month']]
    y_train = train_data['Price (EUR/MWhe)']
    X_test = test_data[['Year', 'Month']]

    scaler_lr = StandardScaler()
    X_train_scaled = scaler_lr.fit_transform(X_train)
    X_test_scaled = scaler_lr.transform(X_test)

    lr_model = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    lr_forecast = lr_model.predict(X_test_scaled)

    future_dates = pd.date_range(start="2024-01-01", periods=future_steps, freq="MS")

    country_results = pd.DataFrame({
        'Country': country,
        'Date': future_dates,
        'ARIMA_Forecast': arima_forecast.values,
        'Hybrid_Forecast': hybrid_forecast,
        'Linear_Regression_Forecast': np.pad(lr_forecast, (0, future_steps - len(lr_forecast)), 'constant', constant_values=np.nan),
        'Actual': test_data['Price (EUR/MWhe)'].reindex(future_dates).values
    })
    results.append(country_results)

all_results = pd.concat(results, ignore_index=True)
all_results.to_csv("forecast_results_all_countries_with_lr.csv", index=False)

print("Forecast results saved to 'forecast_results_all_countries_with_lr.csv'.")

comparison_results = []

for country in monthly_data['Country'].unique():
    print(f"Evaluating performance for {country}...")
    country_data = all_results[all_results['Country'] == country]
    
    actual = country_data['Actual'].dropna()
    arima_forecast = country_data['ARIMA_Forecast'].iloc[:len(actual)]
    hybrid_forecast = country_data['Hybrid_Forecast'].iloc[:len(actual)]
    lr_forecast = country_data['Linear_Regression_Forecast'].iloc[:len(actual)]

    metrics = {
        'Country': country,
        'MAE_ARIMA': mean_absolute_error(actual, arima_forecast),
        'MAE_Hybrid': mean_absolute_error(actual, hybrid_forecast),
        'MAE_LR': mean_absolute_error(actual, lr_forecast),
        'MSE_ARIMA': mean_squared_error(actual, arima_forecast),
        'MSE_Hybrid': mean_squared_error(actual, hybrid_forecast),
        'MSE_LR': mean_squared_error(actual, lr_forecast),
        'MAPE_ARIMA': np.mean(np.abs((actual - arima_forecast) / actual)) * 100,
        'MAPE_Hybrid': np.mean(np.abs((actual - hybrid_forecast) / actual)) * 100,
        'MAPE_LR': np.mean(np.abs((actual - lr_forecast) / actual)) * 100,
    }
    comparison_results.append(metrics)

comparison_df = pd.DataFrame(comparison_results)
comparison_df.to_csv("model_comparison_metrics.csv", index=False)

print("Comparison metrics saved to 'model_comparison_metrics.csv'.")

