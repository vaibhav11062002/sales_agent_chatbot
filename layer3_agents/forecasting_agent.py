import pandas as pd
import logging
from typing import Dict, Any
from data_connector import mcp_store

logger = logging.getLogger(__name__)

class ForecastingAgent:
    """Forecasting Agent using MCP-backed data store"""
    
    def __init__(self):
        self.name = "ForecastingAgent"
    
    def execute(self, query: str, forecast_periods: int = 3) -> Dict[str, Any]:
        """Execute forecasting using shared data store"""
        logger.info(f"{self.name}: Forecasting {forecast_periods} periods")
        
        try:
            # Access shared DataFrame
            df = mcp_store.get_sales_data()
            
            # Check context from Analysis Agent if available
            analysis_context = mcp_store.get_agent_context("AnalysisAgent")
            if analysis_context:
                logger.info("Using context from AnalysisAgent for better forecasting")
            
            # Perform forecasting
            df_monthly = df.groupby(df['CreationDate'].dt.to_period('M')).agg({
                'NetAmount': 'sum'
            }).reset_index()
            
            df_monthly['CreationDate'] = df_monthly['CreationDate'].dt.to_timestamp()
            df_monthly = df_monthly.sort_values('CreationDate')
            
            # Simple moving average forecast
            window = min(3, len(df_monthly))
            df_monthly['MA'] = df_monthly['NetAmount'].rolling(window=window).mean()
            
            recent_data = df_monthly.tail(6)
            if len(recent_data) >= 2:
                trend = (recent_data['NetAmount'].iloc[-1] - recent_data['NetAmount'].iloc[0]) / len(recent_data)
            else:
                trend = 0
            
            last_date = df_monthly['CreationDate'].max()
            last_value = df_monthly['NetAmount'].iloc[-1]
            
            forecasts = []
            for i in range(1, forecast_periods + 1):
                forecast_date = last_date + pd.DateOffset(months=i)
                forecast_value = last_value + (trend * i)
                
                forecasts.append({
                    "date": forecast_date.strftime('%Y-%m'),
                    "forecasted_sales": round(float(forecast_value), 2),
                    "confidence": "medium"
                })
            
            results = {
                "status": "success",
                "forecasts": forecasts,
                "historical_trend": float(trend),
                "message": f"Forecasted {forecast_periods} periods"
            }
            
            # Share results with other agents
            mcp_store.update_agent_context(self.name, {
                "forecasts": forecasts,
                "trend": float(trend),
                "query": query
            })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in {self.name}: {str(e)}")
            return {"status": "error", "message": str(e)}
