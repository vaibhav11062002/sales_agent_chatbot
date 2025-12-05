import pandas as pd
import numpy as np
import logging
from typing import Dict, Any
from data_connector import mcp_store

logger = logging.getLogger(__name__)


class ForecastingAgent:
    """
    Lightweight forecasting agent.

    - No heavy ML training at startup.
    - Uses a simple baseline: extrapolates recent monthly NetAmount trend.
    """

    def __init__(self):
        self.name = "ForecastingAgent"
        self.target_col = "NetAmount"
        self.last_train_timestamp = None
        logger.info(f"{self.name}: Initialized in lightweight mode (no pre-training)")

    def _build_monthly_series(self) -> pd.Series:
        """Aggregate sales_df to monthly NetAmount series."""
        df = mcp_store.get_sales_data()
        if df is None or len(df) == 0:
            raise ValueError("No sales data available from mcp_store")

        if "CreationDate" not in df.columns or self.target_col not in df.columns:
            raise ValueError("Required columns missing in sales data")

        ts = (
            df.groupby(df["CreationDate"].dt.to_period("M"))
            .agg({self.target_col: "sum"})
            .reset_index()
        )
        ts["CreationDate"] = ts["CreationDate"].dt.to_timestamp()
        ts = ts.sort_values("CreationDate").set_index("CreationDate")
        return ts[self.target_col]

    def execute(self, query: str, forecast_periods: int = 3) -> Dict[str, Any]:
        logger.info(f"{self.name}: Executing lightweight forecast for {forecast_periods} months")
        try:
            y = self._build_monthly_series().values
            if len(y) < 2:
                raise ValueError("Not enough history for baseline forecast (need at least 2 months).")

            # Baseline: average month-over-month change over last N points
            lookback = min(6, len(y) - 1)
            diffs = np.diff(y[-(lookback + 1):])
            avg_delta = float(diffs.mean())
            last_value = float(y[-1])

            preds = []
            current = last_value
            for _ in range(forecast_periods):
                current = current + avg_delta
                preds.append(max(current, 0.0))  # no negative sales

            # Build dates
            ts = self._build_monthly_series()
            last_date = ts.index[-1]
            forecast_dates = pd.date_range(
                start=last_date + pd.offsets.MonthBegin(),
                periods=forecast_periods,
                freq="MS",
            )

            # Very rough "confidence" based on volatility
            volatility = float(np.std(diffs)) if len(diffs) > 1 else 0.0
            base_conf = 0.9
            conf = max(0.6, base_conf - volatility / (abs(last_value) + 1e-8))
            conf_str = f"{conf:.2f}"

            forecasts = []
            for d, val in zip(forecast_dates, preds):
                forecasts.append(
                    {
                        "date": pd.Timestamp(d).strftime("%Y-%m"),
                        "forecasted_sales": round(float(val), 2),
                        "confidence": conf_str,
                    }
                )

            trend = float(last_value - y[-min(lookback + 1, len(y))])

            # Store in shared context
            mcp_store.update_agent_context(
                self.name,
                {
                    "forecasts": forecasts,
                    "trend": trend,
                    "model_used": "baseline_avg_delta",
                    "accuracy": None,
                    "query": query,
                },
            )

            return {
                "status": "success",
                "model_used": "baseline_avg_delta",
                "model_params": f"lookback={lookback}",
                "forecasts": forecasts,
                "historical_trend": trend,
                "accuracy": None,
                "message": "Baseline forecast (no heavy ML training).",
            }

        except Exception as e:
            logger.error(f"Error in {self.name}: {e}")
            return {"status": "error", "message": str(e)}
