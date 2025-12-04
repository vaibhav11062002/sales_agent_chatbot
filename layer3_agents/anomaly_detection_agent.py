import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional
from data_connector import mcp_store

logger = logging.getLogger(__name__)

try:
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("sklearn not available. Install with: pip install scikit-learn")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("shap not available. Install with: pip install shap")


class AnomalyDetectionAgent:
    """
    Advanced Anomaly Detection Agent with SHAP-based Explainability
    - Detects anomalies using IsolationForest
    - Generates CONCISE reasons using SHAP feature attributions
    - Stores enriched data for other agents (Analysis → Explanation)
    - DYNAMICALLY adapts to available columns
    - ✅ PRESERVES JOIN KEY for dashboard LEFT JOIN integration
    """

    def __init__(self):
        self.name = "AnomalyDetectionAgent"
        
        # Column mappings (will be detected dynamically)
        self.date_column: Optional[str] = None
        self.revenue_column: Optional[str] = None
        self.quantity_column: Optional[str] = None
        self.tax_column: Optional[str] = None
        self.cost_column: Optional[str] = None
        self.customer_column: Optional[str] = None
        self.product_column: Optional[str] = None
        self.order_id_column: Optional[str] = None  # ✅ NEW: Join key
        
        self.feature_columns: List[str] = []
        self.model: Optional[IsolationForest] = None
        self.explainer: Optional[shap.TreeExplainer] = None
        
        logger.info(f"{self.name} initialized")

    def _detect_columns(self, df: pd.DataFrame):
        """Dynamically detect available columns including join key"""
        # Date column
        for col in ['Date', 'CreationDate', 'SalesDocumentDate', 'TransactionDate']:
            if col in df.columns:
                self.date_column = col
                break
        
        # Revenue column
        for col in ['Revenue', 'NetAmount', 'Sales', 'TotalSales']:
            if col in df.columns:
                self.revenue_column = col
                break
        
        # Quantity column
        for col in ['Volume', 'OrderQuantity', 'Quantity']:
            if col in df.columns:
                self.quantity_column = col
                break
        
        # Tax column
        for col in ['TaxAmount', 'Tax']:
            if col in df.columns:
                self.tax_column = col
                break
        
        # Cost column
        for col in ['COGS', 'CostAmount', 'Cost', 'COGS_SP']:
            if col in df.columns:
                self.cost_column = col
                break
        
        # Customer column
        for col in ['Customer', 'SoldToParty', 'CustomerID']:
            if col in df.columns:
                self.customer_column = col
                break
        
        # Product column
        for col in ['Product', 'ProductID', 'Material']:
            if col in df.columns:
                self.product_column = col
                break
        
        # ✅ NEW: Order ID / Join Key column
        for col in ['SalesDocument', 'OrderID', 'TransactionID', 'InvoiceID', 'DocumentNumber', 'OrderNumber']:
            if col in df.columns:
                self.order_id_column = col
                logger.info(f"[Column Detection] ✅ Join key detected: {self.order_id_column}")
                break
        
        # Build feature columns list (only numeric columns that exist)
        self.feature_columns = []
        for col in [self.revenue_column, self.quantity_column, self.tax_column, self.cost_column]:
            if col and col in df.columns:
                self.feature_columns.append(col)
        
        logger.info(f"[Column Detection] Date: {self.date_column}, Revenue: {self.revenue_column}, "
                   f"Quantity: {self.quantity_column}, Tax: {self.tax_column}, Cost: {self.cost_column}")
        logger.info(f"[Column Detection] Customer: {self.customer_column}, Product: {self.product_column}, "
                   f"Join Key: {self.order_id_column}")
        logger.info(f"[Column Detection] Feature columns for anomaly detection: {self.feature_columns}")

    def execute(
        self,
        query: str,
        entities: dict = None,
        contamination: float = 0.05
    ) -> Dict[str, Any]:
        """
        Main execution method for anomaly detection

        Args:
            query: User query
            entities: Optional entity filters (year, customer, product)
            contamination: Expected proportion of anomalies (0.01 to 0.5)

        Returns:
            Dict with status, anomalies, and summary
        """
        # 🐛 DEBUG: Log incoming parameters
        logger.info(f"🐛 DEBUG [execute] - START")
        logger.info(f"🐛 DEBUG [execute] - query type: {type(query)}, value: {query}")
        logger.info(f"🐛 DEBUG [execute] - entities type: {type(entities)}, value: {entities}")
        logger.info(f"🐛 DEBUG [execute] - contamination: {contamination}")
        
        logger.info(f"{self.name}: Executing anomaly detection")
        logger.info(f"Query: {query}")
        logger.info(f"Contamination: {contamination}")

        if not SKLEARN_AVAILABLE:
            return {
                "status": "error",
                "message": "scikit-learn not installed. Please install: pip install scikit-learn"
            }

        if not SHAP_AVAILABLE:
            logger.warning("⚠️ SHAP not available - using fallback statistical reasoning")

        try:
            # Get data and detect columns
            df = mcp_store.get_sales_data()
            self._detect_columns(df)
            
            logger.info(f"Loaded {len(df)} records")

            # Validate we have at least some feature columns
            if not self.feature_columns:
                return {
                    "status": "error",
                    "message": f"No numeric feature columns found for anomaly detection. "
                              f"Need at least one of: Revenue, Volume, Tax, Cost columns."
                }

            # Apply entity filters if provided
            if entities:
                df = self._apply_filters(df, entities)
                logger.info(f"After filters: {len(df)} records")

            if len(df) < 10:
                return {
                    "status": "error",
                    "message": f"Insufficient data for anomaly detection ({len(df)} records). "
                               f"Need at least 10 records."
                }

            # STEP 1: Detect anomalies
            df_enriched, model, features = self._detect_anomalies(df, contamination)

            # STEP 2: Generate CONCISE reasons
            if SHAP_AVAILABLE:
                if self.explainer is None or self.model is not model:
                    logger.info("🔧 Initializing SHAP TreeExplainer")
                    self.explainer = shap.TreeExplainer(
                        model,
                        data=features,
                        feature_perturbation="interventional",
                        model_output="raw"
                    )
                    self.model = model
                
                df_enriched = self._generate_reasons_shap(df_enriched, features)
            else:
                df_enriched = self._generate_reasons_statistical(df_enriched, features)

            # STEP 3: Extract anomalies
            anomalies_df = df_enriched[df_enriched['is_anomaly']].copy()

            # ✅ STEP 4: Store enriched data with ALL columns (including join key)
            # CRITICAL: This ensures dashboard can perform LEFT JOIN
            logger.info(f"[Storage] Storing enriched data with {len(df_enriched.columns)} columns")
            logger.info(f"[Storage] Join key column '{self.order_id_column}' included: {self.order_id_column in df_enriched.columns}")
            
            mcp_store.set_enriched_data('anomalies', df_enriched)
            mcp_store.set_enriched_data('anomaly_records', anomalies_df)

            logger.info(
                f"✅ Detected {len(anomalies_df)} anomalies "
                f"({len(anomalies_df) / len(df) * 100:.2f}%)"
            )

            # STEP 5: Generate summary
            summary = self._generate_summary(anomalies_df, df)

            # 🐛 DEBUG: Entity conversion
            logger.info(f"🐛 DEBUG [execute] - BEFORE conversion: entities = {entities}, type = {type(entities)}")
            
            # ✅ FIX: Convert None entities to empty dict before storing
            safe_entities = entities if entities is not None else {}
            
            logger.info(f"🐛 DEBUG [execute] - AFTER conversion: safe_entities = {safe_entities}, type = {type(safe_entities)}")
            
            # Prepare context data
            context_data = {
                "query": query,
                "entities": safe_entities,  # ✅ Now always a dict, never None
                "results": {
                    "status": "success",
                    "total_records": len(df),
                    "total_anomalies": len(anomalies_df),
                    "anomaly_rate": f"{(len(anomalies_df) / len(df) * 100):.2f}%",
                    "contamination_used": contamination
                }
            }
            
            # 🐛 DEBUG: Log what we're about to store
            logger.info(f"🐛 DEBUG [execute] - context_data to store:")
            logger.info(f"🐛 DEBUG [execute] -   query: {context_data['query']}")
            logger.info(f"🐛 DEBUG [execute] -   entities: {context_data['entities']}")
            logger.info(f"🐛 DEBUG [execute] -   entities type: {type(context_data['entities'])}")
            
            # Update agent context
            mcp_store.update_agent_context(self.name, context_data)
            
            # 🐛 DEBUG: Verify what was actually stored
            stored_context = mcp_store.agent_contexts.get(self.name, {})
            logger.info(f"🐛 DEBUG [execute] - VERIFICATION: What's actually in mcp_store:")
            logger.info(f"🐛 DEBUG [execute] -   stored query: {stored_context.get('query')}")
            logger.info(f"🐛 DEBUG [execute] -   stored entities: {stored_context.get('entities')}")
            logger.info(f"🐛 DEBUG [execute] -   stored entities type: {type(stored_context.get('entities'))}")

            # Update dialogue state
            mcp_store.update_dialogue_state(
                {"anomalies_detected": True, "anomaly_count": len(anomalies_df)},
                query,
                f"Detected {len(anomalies_df)} anomalies"
            )

            # Return results
            return {
                "status": "success",
                "total_records": len(df),
                "total_anomalies": len(anomalies_df),
                "anomaly_rate": f"{(len(anomalies_df) / len(df) * 100):.2f}%",
                "anomalies": self._format_anomalies_for_display(anomalies_df),
                "summary": summary,
                "message": (
                    f"Found {len(anomalies_df)} anomalies out of {len(df)} records "
                    f"({len(anomalies_df) / len(df) * 100:.1f}%)"
                )
            }

        except Exception as e:
            logger.error(f"{self.name} Error: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"Anomaly detection failed: {str(e)}"
            }

    def _apply_filters(self, df: pd.DataFrame, entities: dict) -> pd.DataFrame:
        """Apply entity filters to dataframe - DYNAMIC"""
        filtered = df.copy()

        # Ensure date column is datetime
        if self.date_column and self.date_column in filtered.columns:
            if not np.issubdtype(filtered[self.date_column].dtype, np.datetime64):
                filtered[self.date_column] = pd.to_datetime(filtered[self.date_column], errors='coerce')

        # Year filter
        if entities.get('year') and self.date_column:
            year = int(entities['year']) if isinstance(entities['year'], str) else entities['year']
            filtered = filtered[filtered[self.date_column].dt.year == year]
            logger.info(f"Filtered by year: {year}")

        # Customer filter
        if entities.get('customer_id') and self.customer_column:
            filtered = filtered[filtered[self.customer_column] == str(entities['customer_id'])]
            logger.info(f"Filtered by customer: {entities['customer_id']}")

        # Product filter
        if entities.get('product_id') and self.product_column:
            filtered = filtered[filtered[self.product_column] == str(entities['product_id'])]
            logger.info(f"Filtered by product: {entities['product_id']}")

        return filtered

    def _detect_anomalies(
        self,
        df: pd.DataFrame,
        contamination: float
    ) -> Tuple[pd.DataFrame, IsolationForest, pd.DataFrame]:
        """
        Detect anomalies using IsolationForest - DYNAMIC
        ✅ Returns FULL dataframe with ALL columns preserved

        Args:
            df: Input dataframe (with ALL columns)
            contamination: Expected proportion of anomalies

        Returns:
            Tuple of (enriched_df_with_ALL_columns, trained_model, features_dataframe)
        """
        # ✅ CRITICAL: Work on a copy that preserves ALL original columns
        df_enriched = df.copy()

        # Use dynamically detected feature columns
        if not self.feature_columns:
            raise ValueError("No feature columns available for anomaly detection")

        # Extract only feature columns for model training
        features = df[self.feature_columns].astype(float)

        # Safer imputation: use median instead of 0
        features = features.fillna(features.median())

        logger.info(f"Features shape: {features.shape}")
        logger.info(f"Feature columns: {self.feature_columns}")

        # Train IsolationForest
        model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100,
            max_samples='auto',
            bootstrap=False,
            n_jobs=-1
        )

        model.fit(features)

        # Predict: -1 = anomaly, 1 = normal
        predictions = model.predict(features)
        
        # ✅ Add anomaly columns to FULL dataframe (preserving ALL original columns)
        df_enriched['is_anomaly'] = predictions == -1

        # Use decision_function as anomaly score (lower = more anomalous)
        scores = model.decision_function(features)
        df_enriched['anomaly_score'] = scores

        logger.info(f"Anomalies detected: {df_enriched['is_anomaly'].sum()}")
        logger.info(f"[Enriched DF] Total columns: {len(df_enriched.columns)} (includes join key: {self.order_id_column in df_enriched.columns})")

        return df_enriched, model, features

    def _generate_reasons_shap(
        self,
        df: pd.DataFrame,
        features: pd.DataFrame,
        top_k: int = 2
    ) -> pd.DataFrame:
        """
        Generate CONCISE reasons using SHAP (for storage, not display)
        ✅ Preserves ALL columns in dataframe
        
        Format: "High Revenue (3.2σ); Low Volume (2.1σ)"
        """
        df = df.copy()

        if self.explainer is None:
            logger.warning("SHAP explainer not initialized; using fallback")
            return self._generate_reasons_statistical(df, features)

        # Compute SHAP values for all rows
        logger.info("🔧 Computing SHAP values...")
        shap_values = self.explainer.shap_values(features)
        
        # Calculate statistical context
        feature_stats = {
            col: {
                'mean': features[col].mean(),
                'std': features[col].std(),
                'median': features[col].median()
            }
            for col in features.columns
        }

        reasons = []
        for idx, row in df.iterrows():
            if not row['is_anomaly']:
                reasons.append(None)
                continue

            # Get the position in features dataframe (not the index)
            position = features.index.get_loc(idx)
            shap_row = shap_values[position]
            feature_values = features.iloc[position]

            # Rank by absolute SHAP contribution
            abs_order = np.argsort(np.abs(shap_row))[::-1]

            parts = []
            for rank in abs_order[:top_k]:
                feat_name = features.columns[rank]
                feat_val = feature_values.iloc[rank]
                shap_contrib = shap_row[rank]
                
                mean = feature_stats[feat_name]['mean']
                std = feature_stats[feat_name]['std']
                
                if std > 0:
                    z_score = (feat_val - mean) / std
                    
                    if abs(z_score) > 2:
                        direction = "High" if z_score > 0 else "Low"
                        parts.append(f"{direction} {feat_name} ({abs(z_score):.1f}σ)")
                    elif shap_contrib < -0.05:
                        parts.append(f"Unusual {feat_name}")

            if not parts:
                parts.append("Multi-factor anomaly")

            reasons.append("; ".join(parts))

        df['anomaly_reason'] = reasons

        logger.info(f"✅ Generated SHAP-based reasons for {df['is_anomaly'].sum()} anomalies")

        return df

    def _generate_reasons_statistical(
        self,
        df: pd.DataFrame,
        features: pd.DataFrame,
        top_k: int = 2
    ) -> pd.DataFrame:
        """
        Fallback: Generate CONCISE reasons using statistical analysis
        ✅ Preserves ALL columns in dataframe
        """
        df = df.copy()
        
        # Calculate statistics
        feature_stats = {}
        for col in features.columns:
            feature_stats[col] = {
                'mean': features[col].mean(),
                'std': features[col].std(),
                'median': features[col].median()
            }

        reasons = []
        for idx, row in df.iterrows():
            if not row['is_anomaly']:
                reasons.append(None)
                continue

            # Get the position in features dataframe
            position = features.index.get_loc(idx)
            feature_values = features.iloc[position]
            
            # Calculate z-scores for all features
            z_scores = {}
            for col in features.columns:
                mean = feature_stats[col]['mean']
                std = feature_stats[col]['std']
                
                if std > 0:
                    z_scores[col] = abs((feature_values[col] - mean) / std)
                else:
                    z_scores[col] = 0

            # Get top contributors
            top_features = sorted(z_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

            parts = []
            for feat_name, z_score in top_features:
                if z_score > 1.5:
                    feat_val = feature_values[feat_name]
                    median = feature_stats[feat_name]['median']
                    direction = "High" if feat_val > median else "Low"
                    parts.append(f"{direction} {feat_name} ({z_score:.1f}σ)")

            if not parts:
                parts.append("Statistical anomaly")

            reasons.append("; ".join(parts))

        df['anomaly_reason'] = reasons

        logger.info(f"✅ Generated statistical reasons for {df['is_anomaly'].sum()} anomalies")

        return df

    def _generate_summary(self, anomalies_df: pd.DataFrame, full_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive summary - DYNAMIC"""
        if len(anomalies_df) == 0:
            return {
                "message": "No anomalies detected",
                "recommendation": "Data appears normal. Consider lowering contamination parameter."
            }

        anomalies_df = anomalies_df.copy()
        anomalies_df['reason_category'] = anomalies_df['anomaly_reason'].apply(self._categorize_reason)

        summary = {
            "total_anomalies": len(anomalies_df),
            "anomaly_rate": f"{(len(anomalies_df)/len(full_df)*100):.2f}%",
            "by_reason_category": anomalies_df['reason_category'].value_counts().to_dict(),
        }

        # Customer stats (if available)
        if self.customer_column and self.customer_column in anomalies_df.columns:
            summary["top_customers_with_anomalies"] = anomalies_df[self.customer_column].value_counts().head(5).to_dict()

        # Product stats (if available)
        if self.product_column and self.product_column in anomalies_df.columns:
            summary["top_products_with_anomalies"] = anomalies_df[self.product_column].value_counts().head(5).to_dict()

        # Financial impact (if available)
        if self.revenue_column and self.revenue_column in anomalies_df.columns:
            summary["total_anomaly_revenue"] = float(anomalies_df[self.revenue_column].sum())
            summary["avg_anomaly_revenue"] = float(anomalies_df[self.revenue_column].mean())
            summary["median_anomaly_revenue"] = float(anomalies_df[self.revenue_column].median())

        # Severity distribution
        summary["severe_anomalies"] = int((anomalies_df['anomaly_score'] < -0.5).sum())
        summary["moderate_anomalies"] = int((anomalies_df['anomaly_score'] >= -0.5).sum())

        # Date range (if available)
        if self.date_column and self.date_column in anomalies_df.columns:
            summary["earliest_anomaly"] = anomalies_df[self.date_column].min().strftime('%Y-%m-%d')
            summary["latest_anomaly"] = anomalies_df[self.date_column].max().strftime('%Y-%m-%d')

        return summary

    def _categorize_reason(self, reason: str) -> str:
        """Categorize anomaly reason"""
        if reason is None or not isinstance(reason, str):
            return "Unknown"

        reason_lower = reason.lower()

        if any(word in reason_lower for word in ["revenue", "netamount", "sales"]):
            return "Revenue Anomaly"
        elif any(word in reason_lower for word in ["volume", "quantity", "orderquantity"]):
            return "Quantity Anomaly"
        elif "tax" in reason_lower:
            return "Tax Anomaly"
        elif any(word in reason_lower for word in ["cost", "cogs"]):
            return "Cost Anomaly"
        elif "multi-factor" in reason_lower:
            return "Multi-Factor Anomaly"
        else:
            return "Complex Anomaly"

    def _format_anomalies_for_display(
        self,
        anomalies_df: pd.DataFrame,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Format anomalies for display - DYNAMIC"""
        if len(anomalies_df) == 0:
            return []

        # Build display columns dynamically
        display_columns = []
        
        # Add join key first if available
        if self.order_id_column and self.order_id_column in anomalies_df.columns:
            display_columns.append(self.order_id_column)
        
        if self.customer_column:
            display_columns.append(self.customer_column)
        if self.product_column:
            display_columns.append(self.product_column)
        if self.date_column:
            display_columns.append(self.date_column)
        
        # Add feature columns
        display_columns.extend(self.feature_columns)
        
        # Add anomaly metadata
        display_columns.extend(['anomaly_score', 'anomaly_reason'])

        # Filter to only existing columns
        display_columns = [col for col in display_columns if col in anomalies_df.columns]

        # Sort by severity
        sorted_df = anomalies_df.nsmallest(limit, 'anomaly_score')

        records = sorted_df[display_columns].to_dict('records')

        formatted = []
        for record in records:
            formatted_record = record.copy()

            # Format date
            if self.date_column in formatted_record and isinstance(formatted_record[self.date_column], pd.Timestamp):
                formatted_record[self.date_column] = formatted_record[self.date_column].strftime('%Y-%m-%d')

            # Format numeric columns
            for col in self.feature_columns:
                if col in formatted_record and pd.notnull(formatted_record[col]):
                    formatted_record[col] = round(float(formatted_record[col]), 2)

            # Format anomaly score
            if 'anomaly_score' in formatted_record and pd.notnull(formatted_record['anomaly_score']):
                formatted_record['anomaly_score'] = round(float(formatted_record['anomaly_score']), 4)

            formatted.append(formatted_record)

        return formatted
