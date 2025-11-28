import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Tuple
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
    """


    def __init__(self):
        self.name = "AnomalyDetectionAgent"
        self.feature_columns = ['NetAmount', 'OrderQuantity', 'TaxAmount', 'CostAmount']
        self.model: IsolationForest | None = None
        self.explainer: shap.TreeExplainer | None = None
        logger.info(f"{self.name} initialized")


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
            # Don't fail, just log warning - we'll use fallback method

        try:
            # Get data
            df = mcp_store.get_sales_data()
            logger.info(f"Loaded {len(df)} records")

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
                # Initialize SHAP explainer if needed
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
                # Fallback to statistical reasoning
                df_enriched = self._generate_reasons_statistical(df_enriched, features)

            # STEP 3: Extract anomalies
            anomalies_df = df_enriched[df_enriched['is_anomaly']].copy()

            # STEP 4: Store enriched data for other agents
            mcp_store.set_enriched_data('anomalies', df_enriched)
            mcp_store.set_enriched_data('anomaly_records', anomalies_df)

            logger.info(
                f"✅ Detected {len(anomalies_df)} anomalies "
                f"({len(anomalies_df) / len(df) * 100:.2f}%)"
            )

            # STEP 5: Generate summary
            summary = self._generate_summary(anomalies_df, df)

            # Update agent context
            mcp_store.update_agent_context(
                self.name,
                query=query,
                entities=entities,
                results={
                    "status": "success",
                    "total_records": len(df),
                    "total_anomalies": len(anomalies_df),
                    "anomaly_rate": f"{(len(anomalies_df) / len(df) * 100):.2f}%",
                    "contamination_used": contamination
                }
            )

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
        """Apply entity filters to dataframe"""
        filtered = df.copy()

        if 'CreationDate' in filtered.columns and not np.issubdtype(filtered['CreationDate'].dtype, np.datetime64):
            filtered['CreationDate'] = pd.to_datetime(filtered['CreationDate'], errors='coerce')

        if entities.get('year'):
            year = int(entities['year']) if isinstance(entities['year'], str) else entities['year']
            if 'CreationDate' in filtered.columns:
                filtered = filtered[filtered['CreationDate'].dt.year == year]
                logger.info(f"Filtered by year: {year}")

        if entities.get('customer_id') and 'SoldToParty' in filtered.columns:
            filtered = filtered[filtered['SoldToParty'] == str(entities['customer_id'])]
            logger.info(f"Filtered by customer: {entities['customer_id']}")

        if entities.get('product_id') and 'Product' in filtered.columns:
            filtered = filtered[filtered['Product'] == str(entities['product_id'])]
            logger.info(f"Filtered by product: {entities['product_id']}")

        return filtered


    def _detect_anomalies(
        self,
        df: pd.DataFrame,
        contamination: float
    ) -> Tuple[pd.DataFrame, IsolationForest, pd.DataFrame]:
        """
        Detect anomalies using IsolationForest

        Args:
            df: Input dataframe
            contamination: Expected proportion of anomalies

        Returns:
            Tuple of (enriched_df, trained_model, features_dataframe)
        """
        df = df.copy()

        # Prepare features (only keep columns that exist)
        available_features = [c for c in self.feature_columns if c in df.columns]
        if not available_features:
            raise ValueError(
                f"None of the expected feature columns are present: {self.feature_columns}"
            )

        features = df[available_features].astype(float)

        # Safer imputation: use median instead of 0
        features = features.fillna(features.median())

        logger.info(f"Features shape: {features.shape}")
        logger.info(f"Feature columns: {available_features}")

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
        df['is_anomaly'] = predictions == -1

        # Use decision_function as anomaly score (lower = more anomalous)
        scores = model.decision_function(features)
        df['anomaly_score'] = scores

        logger.info(f"Anomalies detected: {df['is_anomaly'].sum()}")

        return df, model, features


    def _generate_reasons_shap(
        self,
        df: pd.DataFrame,
        features: pd.DataFrame,
        top_k: int = 2
    ) -> pd.DataFrame:
        """
        Generate CONCISE reasons using SHAP (for storage, not display)
        
        Format: "High NetAmount (3.2σ); Low TaxAmount (2.1σ)"
        
        Args:
            df: DataFrame with is_anomaly column
            features: Feature matrix used for training
            top_k: Number of top features to mention (default: 2)

        Returns:
            DataFrame with anomaly_reason column
        """
        df = df.copy()

        if self.explainer is None:
            logger.warning("SHAP explainer not initialized; using fallback")
            return self._generate_reasons_statistical(df, features)

        # Compute SHAP values for all rows
        logger.info("🔧 Computing SHAP values...")
        shap_values = self.explainer.shap_values(features)
        
        # Calculate statistical context for interpretation
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

            # Get SHAP values and actual feature values
            shap_row = shap_values[idx]
            feature_values = features.iloc[idx]

            # Rank by absolute SHAP contribution (importance)
            abs_order = np.argsort(np.abs(shap_row))[::-1]

            parts = []
            for rank in abs_order[:top_k]:
                feat_name = features.columns[rank]
                feat_val = feature_values.iloc[rank]
                shap_contrib = shap_row[rank]
                
                # Calculate z-score for context
                mean = feature_stats[feat_name]['mean']
                std = feature_stats[feat_name]['std']
                
                if std > 0:
                    z_score = (feat_val - mean) / std
                    
                    # Determine direction
                    if abs(z_score) > 2:  # Significant deviation
                        direction = "High" if z_score > 0 else "Low"
                        parts.append(f"{direction} {feat_name} ({abs(z_score):.1f}σ)")
                    elif shap_contrib < -0.05:  # Strong SHAP contribution
                        parts.append(f"Unusual {feat_name}")

            if not parts:
                # Fallback if no clear contributors
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
        
        Args:
            df: DataFrame with is_anomaly column
            features: Feature matrix
            top_k: Number of top features to mention

        Returns:
            DataFrame with anomaly_reason column
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

            feature_values = features.iloc[idx]
            
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
                if z_score > 1.5:  # Significant deviation
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
        """Generate comprehensive summary of detected anomalies"""
        if len(anomalies_df) == 0:
            return {
                "message": "No anomalies detected",
                "recommendation": "Data appears normal. Consider lowering contamination parameter if you expect more anomalies."
            }

        anomalies_df = anomalies_df.copy()
        anomalies_df['reason_category'] = anomalies_df['anomaly_reason'].apply(self._categorize_reason)

        summary = {
            "total_anomalies": len(anomalies_df),
            "anomaly_rate": f"{(len(anomalies_df)/len(full_df)*100):.2f}%",

            # By reason category
            "by_reason_category": anomalies_df['reason_category'].value_counts().to_dict(),

            # Top customers with anomalies
            "top_customers_with_anomalies": anomalies_df['SoldToParty'].value_counts().head(5).to_dict()
            if 'SoldToParty' in anomalies_df.columns else {},

            # Top products with anomalies
            "top_products_with_anomalies": anomalies_df['Product'].value_counts().head(5).to_dict()
            if 'Product' in anomalies_df.columns else {},

            # Financial impact
            "total_anomaly_revenue": float(anomalies_df['NetAmount'].sum()) if 'NetAmount' in anomalies_df.columns else 0.0,
            "avg_anomaly_revenue": float(anomalies_df['NetAmount'].mean()) if 'NetAmount' in anomalies_df.columns else 0.0,
            "median_anomaly_revenue": float(anomalies_df['NetAmount'].median()) if 'NetAmount' in anomalies_df.columns else 0.0,

            # Severity distribution
            "severe_anomalies": int((anomalies_df['anomaly_score'] < -0.5).sum()),
            "moderate_anomalies": int((anomalies_df['anomaly_score'] >= -0.5).sum()),

            # Date range
            "earliest_anomaly": anomalies_df['CreationDate'].min().strftime('%Y-%m-%d')
            if 'CreationDate' in anomalies_df.columns else None,
            "latest_anomaly": anomalies_df['CreationDate'].max().strftime('%Y-%m-%d')
            if 'CreationDate' in anomalies_df.columns else None,
        }

        return summary


    def _categorize_reason(self, reason: str) -> str:
        """Categorize anomaly reason into broad categories"""
        if reason is None or not isinstance(reason, str):
            return "Unknown"

        reason_lower = reason.lower()

        if "netamount" in reason_lower or "revenue" in reason_lower:
            return "Revenue Anomaly"
        elif "orderquantity" in reason_lower or "quantity" in reason_lower:
            return "Quantity Anomaly"
        elif "taxamount" in reason_lower or "tax" in reason_lower:
            return "Tax Anomaly"
        elif "costamount" in reason_lower or "cost" in reason_lower:
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
        """Format anomalies for clean display"""
        if len(anomalies_df) == 0:
            return []

        display_columns = [
            'SoldToParty', 'Product', 'CreationDate',
            'NetAmount', 'OrderQuantity', 'TaxAmount', 'CostAmount',
            'anomaly_score', 'anomaly_reason'
        ]

        display_columns = [col for col in display_columns if col in anomalies_df.columns]

        # Sort by severity (anomaly_score ascending = more severe first)
        sorted_df = anomalies_df.nsmallest(limit, 'anomaly_score')

        records = sorted_df[display_columns].to_dict('records')

        formatted = []
        for record in records:
            formatted_record = record.copy()

            if 'CreationDate' in formatted_record and isinstance(formatted_record['CreationDate'], pd.Timestamp):
                formatted_record['CreationDate'] = formatted_record['CreationDate'].strftime('%Y-%m-%d')

            for col in ['NetAmount', 'TaxAmount', 'CostAmount']:
                if col in formatted_record and pd.notnull(formatted_record[col]):
                    formatted_record[col] = round(float(formatted_record[col]), 2)

            if 'OrderQuantity' in formatted_record and pd.notnull(formatted_record['OrderQuantity']):
                formatted_record['OrderQuantity'] = int(formatted_record['OrderQuantity'])

            if 'anomaly_score' in formatted_record and pd.notnull(formatted_record['anomaly_score']):
                formatted_record['anomaly_score'] = round(float(formatted_record['anomaly_score']), 4)

            formatted.append(formatted_record)

        return formatted
