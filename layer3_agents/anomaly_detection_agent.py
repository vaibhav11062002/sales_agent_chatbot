import pandas as pd
import numpy as np
import logging
from typing import Dict, Any
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from data_connector import mcp_store

logger = logging.getLogger(__name__)

class AnomalyDetectionAgent:
    """Anomaly Detection Agent using MCP-backed data store"""
    
    def __init__(self):
        self.name = "AnomalyDetectionAgent"
        self.model = None
    
    def execute(self, query: str, contamination: float = 0.05) -> Dict[str, Any]:
        """Execute anomaly detection using shared data store"""
        logger.info(f"{self.name}: Detecting anomalies")
        
        try:
            # Access shared DataFrame
            df = mcp_store.get_sales_data()
            
            # Prepare features for anomaly detection
            features = ['NetAmount', 'OrderQuantity', 'TaxAmount', 'CostAmount']
            available_features = [f for f in features if f in df.columns]
            
            df_features = df[available_features].copy()
            df_features = df_features.fillna(0)
            
            # Scale features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(df_features)
            
            # Train Isolation Forest
            self.model = IsolationForest(contamination=contamination, random_state=42)
            predictions = self.model.fit_predict(scaled_features)
            
            # Add anomaly flag to dataframe
            df['is_anomaly'] = predictions == -1
            
            anomalies = df[df['is_anomaly'] == True]
            
            # Analyze anomalies
            anomaly_stats = {
                "total_anomalies": len(anomalies),
                "percentage": (len(anomalies) / len(df)) * 100,
                "anomaly_sales_total": float(anomalies['NetAmount'].sum()) if len(anomalies) > 0 else 0,
                "top_anomalies": anomalies.nlargest(5, 'NetAmount')[
                    ['SalesDocument', 'NetAmount', 'OrderQuantity', 'Product']
                ].to_dict('records') if len(anomalies) > 0 else []
            }
            
            results = {
                "status": "success",
                "anomalies": anomaly_stats,
                "message": f"Detected {len(anomalies)} anomalies"
            }
            
            # Share results with other agents
            mcp_store.update_agent_context(self.name, {
                "anomaly_stats": anomaly_stats,
                "query": query
            })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in {self.name}: {str(e)}")
            return {"status": "error", "message": str(e)}
