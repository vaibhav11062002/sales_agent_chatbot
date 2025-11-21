import pandas as pd
import numpy as np
import logging
from typing import Dict, Any
from data_connector import mcp_store

logger = logging.getLogger(__name__)

# Machine learning imports
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pmdarima as pm
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping

class ForecastingAgent:
    """Dynamic, AutoML forecasting with XGBoost, LinearReg, Deep, and ARIMA fallback. Pre-trains at startup."""

    def __init__(self):
        self.name = "ForecastingAgent"
        self.trained_models = {}  # key: target_col, value: dict of trained models and metadata
        self.last_train_timestamp = None
        logger.info(f"{self.name}: Initializing and pre-training models")
        self._initialize_and_train()

    def _initialize_and_train(self):
        df = mcp_store.get_sales_data()
        if df is None or len(df) < 6:
            logger.error("Insufficient data for training any models!")
            return
        # Always forecast target variable 'NetAmount' (modify here if you want more)
        self.target_col = 'NetAmount'
        ts = df.groupby(df['CreationDate'].dt.to_period('M')).agg({self.target_col: 'sum'}).reset_index()
        ts['CreationDate'] = ts['CreationDate'].dt.to_timestamp()
        ts = ts.sort_values('CreationDate').set_index('CreationDate')
        y = ts[self.target_col].values
        # Features: lag values for ML
        lags = 6
        X, y_ML = [], []
        for i in range(lags, len(y)):
            X.append(y[i-lags:i])
            y_ML.append(y[i])
        X, y_ML = np.array(X), np.array(y_ML)
        # Train-test split: 80/20
        split = int(0.8 * len(X))
        Xtr, Xval = X[:split], X[split:]
        ytr, yval = y_ML[:split], y_ML[split:]
        # XGBoost
        try:
            xgb = XGBRegressor(n_estimators=100)
            xgb.fit(Xtr, ytr)
            preds = xgb.predict(Xval)
            mae_xgb = mean_absolute_error(yval, preds)
        except Exception as e:
            xgb, mae_xgb = None, float('inf')
            logger.warning(f"XGBoost training failed: {e}")
        # Linear Regression
        try:
            lr = LinearRegression()
            lr.fit(Xtr, ytr)
            preds = lr.predict(Xval)
            mae_lr = mean_absolute_error(yval, preds)
        except Exception as e:
            lr, mae_lr = None, float('inf')
            logger.warning(f"LinearRegression training failed: {e}")
        # Deep (LSTM, if enough data)
        if len(Xtr) > 40:
            try:
                model = Sequential()
                model.add(LSTM(32, activation='relu', input_shape=(lags, 1)))
                model.add(Dense(1))
                model.compile(loss='mae', optimizer='adam')
                Xtr_r = Xtr.reshape((-1, lags, 1))
                Xval_r = Xval.reshape((-1, lags, 1))
                model.fit(Xtr_r, ytr, epochs=30, verbose=0, validation_data=(Xval_r, yval),
                          callbacks=[EarlyStopping('val_loss', patience=3)])
                preds = model.predict(Xval_r).flatten()
                mae_lstm = mean_absolute_error(yval, preds)
            except Exception as e:
                model, mae_lstm = None, float('inf')
                logger.warning(f"LSTM training failed: {e}")
        else:
            model, mae_lstm = None, float('inf')
        # ARIMA fallback
        try:
            arima = pm.auto_arima(ts[self.target_col].values, seasonal=True, m=12, error_action='ignore', suppress_warnings=True)
            preds = arima.predict(n_periods=len(yval))
            mae_arima = mean_absolute_error(yval, preds)
        except Exception as e:
            arima, mae_arima = None, float('inf')
            logger.warning(f"ARIMA training failed: {e}")
        # Cache all
        self.trained_models[self.target_col] = {
            'xgb': (xgb, mae_xgb),
            'lr': (lr, mae_lr),
            'lstm': (model, mae_lstm),
            'arima': (arima, mae_arima),
            'lags': lags,
            'last_value': y[-1] if len(y) else 0,
            'last_train_index': ts.index[-1] if len(ts) else None
        }
        logger.info(f"Initial model training complete: MAE (xgb): {mae_xgb:.2f}, (lr): {mae_lr:.2f}, (lstm): {mae_lstm:.2f}, (arima): {mae_arima:.2f}")

    def execute(self, query: str, forecast_periods: int = 3) -> Dict[str, Any]:
        logger.info(f"{self.name}: Executing dynamic forecast for {forecast_periods} months")
        try:
            target_col = self.target_col
            models = self.trained_models.get(target_col)
            if not models:
                raise RuntimeError("No models available. Did training fail on startup?")
            # Auto-pick best model
            model_order = sorted(
                [('xgb', models['xgb'][1]), ('lr', models['lr'][1]), ('lstm', models['lstm'][1]), ('arima', models['arima'][1])],
                key=lambda x: x[1]
            )
            for model_name, mae in model_order:
                if models[model_name][0] is not None and np.isfinite(mae):
                    best_model_name = model_name
                    break
            conf = str(max(0.99 - mae/(models['last_value']+1e-8), 0.65))[:4]
            # Create feature for last known period
            last_values = []
            df = mcp_store.get_sales_data().groupby(mcp_store.get_sales_data()['CreationDate'].dt.to_period('M')).agg({target_col: 'sum'}).reset_index()
            series = df.sort_values('CreationDate')[target_col].values
            lags = models['lags']
            if len(series) < lags:
                seed = np.pad(series, (lags-len(series), 0), 'constant', constant_values=0)
            else:
                seed = series[-lags:]
            preds = []
            # Forecast
            for _ in range(forecast_periods):
                x_input = np.array(seed[-lags:]).reshape(1, -1)
                if best_model_name == 'xgb':
                    y_pred = float(models['xgb'][0].predict(x_input)[0])
                elif best_model_name == 'lr':
                    y_pred = float(models['lr'][0].predict(x_input)[0])
                elif best_model_name == 'lstm':
                    y_pred = float(models['lstm'][0].predict(x_input.reshape((1, lags, 1)))[0][0])
                elif best_model_name == 'arima':
                    y_pred = float(models['arima'][0].predict(n_periods=1)[0])
                    models['arima'][0].update(np.array([y_pred]))
                else:
                    y_pred = 0.0
                preds.append(y_pred)
                seed = np.append(seed, y_pred)
            # Dates
            last_date = models['last_train_index']
            forecast_dates = pd.date_range(start=last_date+pd.offsets.MonthBegin(), periods=forecast_periods, freq='MS')
            # Format output
            forecasts = []
            for d, val in zip(forecast_dates, preds):
                forecasts.append({
                    "date": pd.Timestamp(d).strftime('%Y-%m'),
                    "forecasted_sales": round(float(val), 2),
                    "confidence": conf
                })
            trend = models['last_value'] - (seed[0] if len(seed) > 0 else 0)
            # Update context
            mcp_store.update_agent_context(self.name, {
                "forecasts": forecasts,
                "trend": float(trend),
                "model_used": best_model_name,
                "accuracy": mae,
                "query": query
            })
            return {
                "status": "success",
                "model_used": best_model_name,
                "model_params": "lags=%s" % lags,
                "forecasts": forecasts,
                "historical_trend": float(trend),
                "accuracy": {"mae": mae},
                "message": f"Best model: {best_model_name} (MAE={mae:.2f})"
            }
        except Exception as e:
            logger.error(f"Error in {self.name}: {e}")
            return {"status": "error", "message": str(e)}
