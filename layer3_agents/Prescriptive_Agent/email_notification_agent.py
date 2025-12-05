# email_notification_agent.py

import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, List, Optional
from datetime import datetime
import os
from dotenv import load_dotenv
import pandas as pd
from data_connector import mcp_store
from config import GMAIL_USER, GMAIL_APP_PASSWORD, RECIPIENT_EMAIL


# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class EmailNotificationAgent:
    """
    Email Notification Agent for Anomaly Alerts
    - Fetches anomalies from MCP store
    - Sends HTML email with top 5 affected customers
    - Uses Gmail SMTP (free)
    """

    def __init__(self):
        self.name = "EmailNotificationAgent"
        
        # Gmail SMTP configuration
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587  # TLS port
        self.sender_email = GMAIL_USER
        self.sender_password = GMAIL_APP_PASSWORD
        self.recipient_email = RECIPIENT_EMAIL
        
        # Validate configuration
        if not all([self.sender_email, self.sender_password, self.recipient_email]):
            logger.error("❌ Email credentials not configured in .env file")
            raise ValueError(
                "Missing email configuration. Set GMAIL_USER, GMAIL_APP_PASSWORD, "
                "and RECIPIENT_EMAIL in .env file"
            )
        
        logger.info(f"{self.name} initialized with sender: {self.sender_email}")


    def execute(self, trigger_threshold: int = 1) -> Dict[str, Any]:
        """
        Main execution method - checks for anomalies and sends email
        
        Args:
            trigger_threshold: Minimum number of anomalies to trigger email (default: 1)
        
        Returns:
            Dict with status and message
        """
        logger.info(f"{self.name}: Checking for anomalies...")
        
        try:
            # ✅ STEP 1: Fetch anomalies from MCP store (not from raw detection)
            anomalies_df = mcp_store.get_enriched_data('anomaly_records')
            
            if anomalies_df is None or len(anomalies_df) == 0:
                logger.info("No anomalies found in MCP store")
                return {
                    "status": "success",
                    "message": "No anomalies detected - no email sent",
                    "anomalies_count": 0
                }
            
            total_anomalies = len(anomalies_df)
            
            # Check if threshold met
            if total_anomalies < trigger_threshold:
                logger.info(f"Only {total_anomalies} anomalies found (threshold: {trigger_threshold})")
                return {
                    "status": "success",
                    "message": f"Anomalies below threshold ({total_anomalies} < {trigger_threshold})",
                    "anomalies_count": total_anomalies
                }
            
            logger.info(f"📧 Threshold met: {total_anomalies} anomalies detected")
            
            # ✅ STEP 2: Calculate top 5 affected customers
            top_customers = self._get_top_affected_customers(anomalies_df)
            
            # ✅ STEP 3: Generate email content
            email_html = self._generate_email_html(total_anomalies, top_customers, anomalies_df)
            
            # ✅ STEP 4: Send email
            self._send_email(
                subject=f"🚨 Anomaly Alert: {total_anomalies} Anomalies Detected",
                html_content=email_html
            )
            
            logger.info(f"✅ Email sent successfully to {self.recipient_email}")
            
            return {
                "status": "success",
                "message": f"Email sent: {total_anomalies} anomalies detected",
                "anomalies_count": total_anomalies,
                "email_sent_to": self.recipient_email,
                "top_customers": top_customers
            }
            
        except Exception as e:
            logger.error(f"{self.name} Error: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"Email notification failed: {str(e)}"
            }


    def _get_top_affected_customers(
        self, 
        anomalies_df: pd.DataFrame, 
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get top N customers affected by anomalies with revenue impact
        
        Args:
            anomalies_df: DataFrame with anomaly records
            top_n: Number of top customers to return
        
        Returns:
            List of dicts with customer info and revenue impact
        """
        # Dynamically detect customer and revenue columns
        customer_col = None
        revenue_col = None
        
        for col in ['Customer', 'SoldToParty', 'CustomerID', 'customer_id']:
            if col in anomalies_df.columns:
                customer_col = col
                break
        
        for col in ['Revenue', 'NetAmount', 'Sales', 'TotalSales', 'revenue']:
            if col in anomalies_df.columns:
                revenue_col = col
                break
        
        if not customer_col:
            logger.warning("No customer column found - returning empty list")
            return []
        
        if not revenue_col:
            logger.warning("No revenue column found - using count only")
            # Count-based ranking
            customer_counts = anomalies_df[customer_col].value_counts().head(top_n)
            return [
                {
                    "customer_id": str(customer),
                    "anomaly_count": int(count),
                    "total_revenue": 0.0
                }
                for customer, count in customer_counts.items()
            ]
        
        # Group by customer and aggregate
        customer_summary = (
            anomalies_df
            .groupby(customer_col)
            .agg({
                revenue_col: ['sum', 'count']
            })
            .reset_index()
        )
        
        # Flatten multi-level columns
        customer_summary.columns = [customer_col, 'total_revenue', 'anomaly_count']
        
        # Sort by total revenue (descending)
        customer_summary = customer_summary.nlargest(top_n, 'total_revenue')
        
        # Convert to list of dicts
        top_customers = []
        for _, row in customer_summary.iterrows():
            top_customers.append({
                "customer_id": str(row[customer_col]),
                "anomaly_count": int(row['anomaly_count']),
                "total_revenue": float(row['total_revenue'])
            })
        
        logger.info(f"Top {len(top_customers)} affected customers calculated")
        return top_customers


    def _generate_email_html(
        self, 
        total_anomalies: int, 
        top_customers: List[Dict[str, Any]],
        anomalies_df: pd.DataFrame
    ) -> str:
        """
        Generate HTML email content with professional styling
        
        Args:
            total_anomalies: Total number of anomalies detected
            top_customers: List of top affected customers
            anomalies_df: Full anomalies dataframe
        
        Returns:
            HTML string for email body
        """
        # Calculate additional metrics
        total_revenue_affected = sum(c['total_revenue'] for c in top_customers)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Get anomaly reason breakdown
        reason_breakdown = ""
        if 'anomaly_reason' in anomalies_df.columns:
            reason_counts = anomalies_df['anomaly_reason'].value_counts().head(3)
            reason_breakdown = "<ul style='margin: 10px 0;'>"
            for reason, count in reason_counts.items():
                if pd.notna(reason):
                    reason_breakdown += f"<li><strong>{reason}</strong>: {count} occurrences</li>"
            reason_breakdown += "</ul>"
        
        # Build customer table rows
        customer_rows = ""
        for idx, customer in enumerate(top_customers, 1):
            customer_rows += f"""
            <tr style="background-color: {'#f9f9f9' if idx % 2 == 0 else '#ffffff'};">
                <td style="padding: 12px; text-align: center; border: 1px solid #ddd;">{idx}</td>
                <td style="padding: 12px; border: 1px solid #ddd;">{customer['customer_id']}</td>
                <td style="padding: 12px; text-align: center; border: 1px solid #ddd;">{customer['anomaly_count']}</td>
                <td style="padding: 12px; text-align: right; border: 1px solid #ddd; color: #d9534f; font-weight: bold;">
                    ${customer['total_revenue']:,.2f}
                </td>
            </tr>
            """
        
        # HTML email template
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Anomaly Detection Alert</title>
        </head>
        <body style="margin: 0; padding: 0; background-color: #f4f4f4; font-family: Arial, sans-serif;">
            <table width="100%" cellpadding="0" cellspacing="0" style="background-color: #f4f4f4; padding: 20px;">
                <tr>
                    <td align="center">
                        <table width="600" cellpadding="0" cellspacing="0" style="background-color: #ffffff; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                            
                            <!-- Header -->
                            <tr>
                                <td style="background: linear-gradient(135deg, #d9534f 0%, #c9302c 100%); padding: 30px; text-align: center;">
                                    <h1 style="color: #ffffff; margin: 0; font-size: 28px;">
                                        🚨 Anomaly Detection Alert
                                    </h1>
                                    <p style="color: #ffffff; margin: 10px 0 0 0; font-size: 14px; opacity: 0.9;">
                                        Detected: {current_time}
                                    </p>
                                </td>
                            </tr>
                            
                            <!-- Summary Section -->
                            <tr>
                                <td style="padding: 30px;">
                                    <div style="background-color: #fcf8e3; border-left: 4px solid #f0ad4e; padding: 15px; margin-bottom: 20px; border-radius: 4px;">
                                        <h2 style="color: #8a6d3b; margin: 0 0 10px 0; font-size: 18px;">
                                            📊 Summary
                                        </h2>
                                        <p style="margin: 0; color: #333; font-size: 16px; line-height: 1.6;">
                                            <strong style="font-size: 36px; color: #d9534f;">{total_anomalies}</strong> 
                                            anomalies detected in the system.
                                        </p>
                                        <p style="margin: 10px 0 0 0; color: #666; font-size: 14px;">
                                            Total revenue impact: <strong style="color: #d9534f;">${total_revenue_affected:,.2f}</strong>
                                        </p>
                                    </div>
                                    
                                    <!-- Top Anomaly Reasons -->
                                    {f'''
                                    <div style="margin-bottom: 25px;">
                                        <h3 style="color: #333; margin: 0 0 10px 0; font-size: 16px; border-bottom: 2px solid #e0e0e0; padding-bottom: 8px;">
                                            🔍 Top Anomaly Reasons
                                        </h3>
                                        {reason_breakdown}
                                    </div>
                                    ''' if reason_breakdown else ''}
                                    
                                    <!-- Top Affected Customers -->
                                    <h3 style="color: #333; margin: 0 0 15px 0; font-size: 16px; border-bottom: 2px solid #e0e0e0; padding-bottom: 8px;">
                                        👥 Top 5 Affected Customers
                                    </h3>
                                    
                                    <table width="100%" cellpadding="0" cellspacing="0" style="border-collapse: collapse; margin-bottom: 20px;">
                                        <thead>
                                            <tr style="background-color: #337ab7; color: #ffffff;">
                                                <th style="padding: 12px; text-align: center; border: 1px solid #2e6da4;">#</th>
                                                <th style="padding: 12px; text-align: left; border: 1px solid #2e6da4;">Customer ID</th>
                                                <th style="padding: 12px; text-align: center; border: 1px solid #2e6da4;">Anomaly Count</th>
                                                <th style="padding: 12px; text-align: right; border: 1px solid #2e6da4;">Revenue Impact</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {customer_rows if customer_rows else '<tr><td colspan="4" style="padding: 20px; text-align: center; color: #999;">No customer data available</td></tr>'}
                                        </tbody>
                                    </table>
                                    
                                    <!-- Call to Action -->
                                    <div style="text-align: center; margin-top: 30px;">
                                        <a href="http://localhost:8501" 
                                           style="display: inline-block; background-color: #5cb85c; color: #ffffff; padding: 12px 30px; text-decoration: none; border-radius: 4px; font-weight: bold; font-size: 16px;">
                                            View Dashboard
                                        </a>
                                    </div>
                                </td>
                            </tr>
                            
                            <!-- Footer -->
                            <tr>
                                <td style="background-color: #f9f9f9; padding: 20px; text-align: center; border-top: 1px solid #e0e0e0;">
                                    <p style="margin: 0; color: #999; font-size: 12px;">
                                        This is an automated alert from your Anomaly Detection System.<br>
                                        Please investigate these anomalies promptly.
                                    </p>
                                </td>
                            </tr>
                            
                        </table>
                    </td>
                </tr>
            </table>
        </body>
        </html>
        """
        
        return html


    def _send_email(self, subject: str, html_content: str):
        """
        Send HTML email via Gmail SMTP
        
        Args:
            subject: Email subject line
            html_content: HTML body content
        """
        # Create message
        message = MIMEMultipart('alternative')
        message['Subject'] = subject
        message['From'] = self.sender_email
        message['To'] = self.recipient_email
        
        # Attach HTML content
        html_part = MIMEText(html_content, 'html')
        message.attach(html_part)
        
        # Send via Gmail SMTP
        try:
            logger.info(f"Connecting to {self.smtp_server}:{self.smtp_port}...")
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()  # Enable TLS encryption
                logger.info("TLS enabled, logging in...")
                
                server.login(self.sender_email, self.sender_password)
                logger.info("Login successful, sending email...")
                
                server.sendmail(
                    self.sender_email, 
                    self.recipient_email, 
                    message.as_string()
                )
                
            logger.info("✅ Email sent successfully")
            
        except smtplib.SMTPAuthenticationError:
            logger.error("❌ Authentication failed - check GMAIL_APP_PASSWORD")
            raise ValueError(
                "Gmail authentication failed. Verify your App Password is correct. "
                "Generate new one at: https://myaccount.google.com/apppasswords"
            )
        except Exception as e:
            logger.error(f"❌ SMTP Error: {e}")
            raise


# ============================================
# Integration with Anomaly Detection Agent
# ============================================

def trigger_email_if_anomalies():
    """
    Convenience function to check and send email notifications
    Call this AFTER AnomalyDetectionAgent.execute()
    """
    try:
        email_agent = EmailNotificationAgent()
        result = email_agent.execute(trigger_threshold=1)
        return result
    except Exception as e:
        logger.error(f"Email notification trigger failed: {e}")
        return {"status": "error", "message": str(e)}
