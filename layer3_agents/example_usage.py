"""
Example usage of the Sales AI Agents
"""

import pandas as pd
from datetime import datetime

# Import all agents
from data_fetching_agent import DataFetchingAgent
from analysis_agent import AnalysisAgent
from forecasting_agent import ForecastingAgent
from anomaly_detection_agent import AnomalyDetectionAgent
from explanation_agent import ExplanationAgent

# Or import from package (if using __init__.py)
# from agents import (
#     DataFetchingAgent,
#     AnalysisAgent,
#     ForecastingAgent,
#     AnomalyDetectionAgent,
#     ExplanationAgent
# )

def main():
    # Initialize agents
    data_agent = DataFetchingAgent()
    analysis_agent = AnalysisAgent()
    forecast_agent = ForecastingAgent()
    anomaly_agent = AnomalyDetectionAgent()
    explanation_agent = ExplanationAgent()

    # Example 1: Fetch filtered data
    print("\n=== Example 1: Data Fetching ===")
    filters = {
        'year': 2024,
        'sales_org': ''1000'
    }
    result = data_agent.execute("Get sales data for 2024", filters=filters)
    print(f"Status: {result['status']}")
    print(f"Message: {result['message']}")
    print(f"Rows retrieved: {result['row_count']}")

    # Example 2: Summary analysis
    print("\n=== Example 2: Summary Analysis ===")
    result = analysis_agent.execute("Show me summary statistics", analysis_type="summary")
    print(f"Status: {result['status']}")
    print(f"Total Sales: {result['results']['total_sales']}")
    print(f"Total Orders: {result['results']['total_orders']}")
    print(f"Unique Customers: {result['results']['unique_customers']}")

    # Example 3: Trend analysis
    print("\n=== Example 3: Trend Analysis ===")
    result = analysis_agent.execute("Analyze sales trends", analysis_type="trend")
    print(f"Status: {result['status']}")
    print(f"Average Growth Rate: {result['results']['avg_growth_rate']:.2f}%")

    # Example 4: Comparison analysis
    print("\n=== Example 4: Comparison Analysis ===")
    result = analysis_agent.execute("Compare top products and customers", analysis_type="comparison")
    print(f"Status: {result['status']}")
    print(f"Top 3 Products: {list(result['results']['top_products'].keys())[:3]}")

    # Example 5: Forecasting
    print("\n=== Example 5: Sales Forecasting ===")
    result = forecast_agent.execute("Forecast next 3 months", forecast_periods=3)
    print(f"Status: {result['status']}")
    print(f"Forecasts:")
    for forecast in result['forecasts']:
        print(f"  {forecast['date']}: ${forecast['forecasted_sales']:,.2f}")

    # Example 6: Anomaly detection
    print("\n=== Example 6: Anomaly Detection ===")
    result = anomaly_agent.execute("Detect anomalies in sales", contamination=0.05)
    print(f"Status: {result['status']}")
    print(f"Total Anomalies: {result['anomalies']['total_anomalies']}")
    print(f"Percentage: {result['anomalies']['percentage']:.2f}%")

    # Example 7: Generate explanation
    print("\n=== Example 7: Natural Language Explanation ===")
    analysis_results = {
        "total_sales": 1000000,
        "growth_rate": 15.5,
        "anomalies_detected": 23
    }
    result = explanation_agent.execute(
        "What are the key insights from this quarter?",
        analysis_results=analysis_results
    )
    print(f"Status: {result['status']}")
    print(f"Explanation: {result['explanation']}")


if __name__ == "__main__":
    # Note: Make sure global_state.sales_df is loaded before running
    # from config import global_state
    # global_state.sales_df = pd.read_csv('your_data.csv')
    # global_state.data_loaded = True

    main()
