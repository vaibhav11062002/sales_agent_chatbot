from typing_extensions import Annotated
from typing import TypedDict
from langgraph.prebuilt import InjectedState, create_react_agent
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
import pandas as pd
import numpy as np


class State(TypedDict):
    messages: Annotated[list, add_messages]
    df: pd.DataFrame
    remaining_steps: int


@tool
def auto_train_predict(target_col: str, state: Annotated[dict, InjectedState]) -> str:
    """Automatically train ML model on dimensions to predict target measure.
    Excludes other measures to prevent data leakage. Uses log transformation for skewed data."""
    from autogluon.tabular import TabularPredictor

    df = state["df"].copy()
    
    # Define all measure columns
    MEASURE_COLUMNS = ['kwmeng', 'netwr']
    
    # Handle case-insensitive column matching
    df_columns_lower = {col.lower(): col for col in df.columns}
    target_col_lower = target_col.lower().strip()
    
    if target_col_lower not in df_columns_lower:
        available_cols = ', '.join(df.columns)
        return f"Error: Column '{target_col}' not found. Available columns: {available_cols}"
    
    actual_target_col = df_columns_lower[target_col_lower]
    
    if actual_target_col not in MEASURE_COLUMNS:
        return f"Error: '{actual_target_col}' is not a valid measure column. Valid measures: {', '.join(MEASURE_COLUMNS)}"
    
    # Exclude other measures
    columns_to_exclude = [col for col in MEASURE_COLUMNS if col != actual_target_col]
    feature_columns = [col for col in df.columns if col not in columns_to_exclude]
    train_df = df[feature_columns].copy()
    
    # Get statistics BEFORE transformation
    original_rows = len(train_df)
    target_missing = train_df[actual_target_col].isna().sum()
    
    # Calculate original statistics (for the actual target column)
    orig_min = train_df[actual_target_col].min()
    orig_max = train_df[actual_target_col].max()
    orig_mean = train_df[actual_target_col].mean()
    orig_median = train_df[actual_target_col].median()
    orig_std = train_df[actual_target_col].std()
    
    print(f"\n📊 Data Preprocessing for '{actual_target_col}':")
    print(f"   Total Rows: {original_rows:,}")
    print(f"   Missing values: {target_missing:,} ({target_missing/original_rows*100:.1f}%)")
    print(f"\n   Original Statistics:")
    print(f"     Min: {orig_min:,.2f}")
    print(f"     Max: {orig_max:,.2f}")
    print(f"     Mean: {orig_mean:,.2f}")
    print(f"     Median: {orig_median:,.2f}")
    print(f"     Std Dev: {orig_std:,.2f}")
    
    # Handle target column
    if train_df[actual_target_col].dtype in ['float64', 'float32', 'int64', 'int32']:
        # Replace Inf/-Inf with NaN
        train_df[actual_target_col] = train_df[actual_target_col].replace([np.inf, -np.inf], np.nan)
        
        # Handle negative values (set to 0 for log transform)
        negative_count = (train_df[actual_target_col] < 0).sum()
        if negative_count > 0:
            print(f"     Warning: {negative_count} negative values set to 0")
            train_df.loc[train_df[actual_target_col] < 0, actual_target_col] = 0
        
        # Fill NaN with median
        target_median = train_df[actual_target_col].median()
        if pd.isna(target_median):
            target_median = 0
        train_df[actual_target_col] = train_df[actual_target_col].fillna(target_median)
        
        # LOG TRANSFORMATION for skewed data
        print(f"\n   ✓ Applying log transformation: log(1 + x)")
        train_df[actual_target_col] = np.log1p(train_df[actual_target_col])
    
    # Handle feature columns
    for col in train_df.columns:
        if col == actual_target_col:
            continue
        
        missing_count = train_df[col].isna().sum()
        if missing_count > 0:
            if train_df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                train_df[col] = train_df[col].fillna(-999)
            else:
                train_df[col] = train_df[col].fillna('MISSING')
    
    print(f"   Feature Columns: {len(feature_columns) - 1} (excluding target)")
    
    try:
        predictor = TabularPredictor(
            label=actual_target_col, 
            verbosity=1,
            eval_metric='root_mean_squared_error'  # This will be RMSLE due to log transform
        ).fit(
            train_df, 
            time_limit=120, 
            presets='good'  # Use 'good' instead of 'medium' for better memory management
        )
        
        leaderboard = predictor.leaderboard(silent=True)
        best_model = predictor.model_best
        best_score = leaderboard[leaderboard['model'] == best_model]['score_val'].values[0]
        
        # Convert RMSLE back to interpretable RMSE
        rmsle = abs(best_score)
        
        # Calculate approximate RMSE on original scale
        # RMSE ≈ mean * RMSLE (rough approximation)
        approx_rmse = orig_mean * rmsle
        rmse_pct_of_mean = (approx_rmse / orig_mean) * 100
        rmse_pct_of_median = (approx_rmse / orig_median) * 100 if orig_median > 0 else 0
        
        return f"""Successfully trained model on '{actual_target_col}' using log-transformed values.

🎯 Model Performance (Log Scale):
   Best Model: {best_model}
   RMSLE (Root Mean Squared Log Error): {rmsle:.4f}
   
📈 Top 3 Models:
{leaderboard.head(3)[['model', 'score_val']].to_string(index=False)}

📊 Original Scale Statistics for '{actual_target_col}':
   Min: {orig_min:,.2f}
   Max: {orig_max:,.2f}
   Mean: {orig_mean:,.2f}
   Median: {orig_median:,.2f}
   Std Dev: {orig_std:,.2f}
   
   Approximate RMSE (original scale): {approx_rmse:,.2f}
   RMSE as % of mean: {rmse_pct_of_mean:.2f}%
   RMSE as % of median: {rmse_pct_of_median:.2f}%

✓ Log Transformation Applied: Better for skewed data with outliers
✓ Data Leakage Prevention: Excluded {', '.join(columns_to_exclude)}
✓ All {original_rows:,} rows retained after NaN/Inf handling

Note: When making predictions, apply inverse transform: np.expm1(predictions)"""
        
    except Exception as e:
        return f"Error during training: {str(e)}"


# Initialize
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key="AIzaSyCM4wrMOFS2dL4U850zv7rlu9Oz2QnwvrI"
)

agent = create_react_agent(llm, tools=[auto_train_predict], state_schema=State)
df = pd.read_csv("D:/sales_agent_chatbot/filtered_sales_data.csv", low_memory=False)

# Test cases
test_prompts = [
    "Train a model to predict kwmeng",
    "Train a model to predict netwr"
]

print("="*70)
print("TESTING AI AGENT WITH LOG-TRANSFORMED PREDICTIONS")
print("="*70)

for i, prompt in enumerate(test_prompts, 1):
    print(f"\n\n{'='*70}")
    print(f"TEST {i}: {prompt}")
    print('='*70)
    
    try:
        result = agent.invoke({
            "messages": [("user", prompt)],
            "df": df,
            "remaining_steps": 25
        })
        print(f"\n✓ Agent Response:\n{result['messages'][-1].content}")
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
