import pandas as pd
from dataclasses import dataclass
from typing import Optional
from datetime import datetime
import os

# Gemini API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyAyr2sLxZNrIpPj0f1-3ytTfZhv7T8FlUg")

# Database Configuration
DB_CONFIG = {
    "address": "c0c94ed5-bef0-4ca4-95f8-55cf5a4ecbdc.hana.prod-us10.hanacloud.ondemand.com",
    "port": 443,
    "user": "DSP_CUST_CONTENT#DSP_CUST_CONTENT",
    "password": "g6D,$a%@D`3$!)-GaVO#_[]T+=3z~[Z6",
    "encrypt": True,
    "sslValidateCertificate": False
}

TABLE_NAME = "DSP_CUST_CONTENT.SALES_DATA_VIEW"

@dataclass
class GlobalState:
    """Global state to store data across all agents"""
    sales_df: Optional[pd.DataFrame] = None
    data_loaded: bool = False
    data_timestamp: Optional[datetime] = None
    user_context: dict = None
    
    def __post_init__(self):
        if self.user_context is None:
            self.user_context = {}

# Singleton instance
global_state = GlobalState()

# Column metadata for better understanding
# Column metadata for better understanding
COLUMN_DESCRIPTIONS = {
    # Document Header Fields
    "SalesDocument": "Unique sales document ID",
    "SalesDocumentItem": "Line item number within the sales document",
    "SDDocumentCategory": "Sales and distribution document category (e.g., order, quotation, contract)",
    "SalesDocumentType": "Type of sales document (e.g., standard order, rush order, credit memo)",
    "SalesDocumentItemCategory": "Category of the item within the sales document",
    "IsReturnsItem": "Indicator whether the item is a returns item",
    "CreationTime": "Time when the sales document was created",
    "CreationDate": "Date when the sales document was created",
    "LastChangeDate": "Date when the sales document was last modified",
    "SalesDocumentDate": "Document date of the sales transaction",
    
    # Organizational Data
    "SalesOrganization": "Sales organization responsible for the sale",
    "DistributionChannel": "Distribution channel through which the product is sold",
    "Division": "Product division or business unit",
    "SalesGroup": "Group of sales personnel responsible for the sale",
    "SalesOffice": "Sales office handling the transaction",
    "SalesDistrict": "Geographic sales district",
    "BusinessArea": "Business area for financial reporting",
    "ProfitCenter": "Profit center for cost and revenue tracking",
    "ControllingArea": "Controlling area for management accounting",
    
    # Product Information
    "Product": "Product ID or material number",
    "ProductDescription": "Product name or description",
    "InternationalArticleNumber": "EAN/UPC barcode number",
    "Batch": "Batch number for batch-managed materials",
    "OriginallyRequestedMaterial": "Original material requested by customer before substitution",
    "MaterialSubstitutionReason": "Reason for material substitution",
    "ProductGroup": "Product group classification",
    "AdditionalMaterialGroup1": "Additional material grouping level 1",
    "AdditionalMaterialGroup2": "Additional material grouping level 2",
    "AdditionalMaterialGroup3": "Additional material grouping level 3",
    "AdditionalMaterialGroup4": "Additional material grouping level 4",
    "AdditionalMaterialGroup5": "Additional material grouping level 5",
    
    # Customer/Partner Information
    "SoldToParty": "Customer ID - who purchased the product",
    "ShipToParty": "Customer ID - where the product is shipped",
    "PayerParty": "Customer ID - who pays for the product",
    "BillToParty": "Customer ID - who receives the invoice",
    "CustomerGroup": "Customer group classification",
    "AdditionalCustomerGroup1": "Additional customer grouping level 1",
    "AdditionalCustomerGroup2": "Additional customer grouping level 2",
    "AdditionalCustomerGroup3": "Additional customer grouping level 3",
    "AdditionalCustomerGroup4": "Additional customer grouping level 4",
    "AdditionalCustomerGroup5": "Additional customer grouping level 5",
    "CustomerAccountAssignmentGroup": "Account assignment group for customer",
    
    # Quantity Fields
    "OrderQuantity": "Quantity ordered by customer",
    "OrderQuantityUnit": "Unit of measure for order quantity",
    "TargetQuantity": "Target quantity for delivery",
    "TargetQuantityUnit": "Unit of measure for target quantity",
    "TargetToBaseQuantityDnmntr": "Denominator for target to base quantity conversion",
    "TargetToBaseQuantityNmrtr": "Numerator for target to base quantity conversion",
    "OrderToBaseQuantityDnmntr": "Denominator for order to base quantity conversion",
    "OrderToBaseQuantityNmrtr": "Numerator for order to base quantity conversion",
    "ConfdDelivQtyInOrderQtyUnit": "Confirmed delivery quantity in order quantity unit",
    "TargetDelivQtyInOrderQtyUnit": "Target delivery quantity in order quantity unit",
    "ConfdDeliveryQtyInBaseUnit": "Confirmed delivery quantity in base unit",
    "BaseUnit": "Base unit of measure for the product",
    "RequestedQuantityInBaseUnit": "Requested quantity converted to base unit",
    "MinDeliveryQtyInBaseUnit": "Minimum delivery quantity in base unit",
    "No_of_OrderItems": "Number of items in the order",
    
    # Weight and Volume
    "ItemGrossWeight": "Gross weight of the item",
    "ItemNetWeight": "Net weight of the item",
    "ItemWeightUnit": "Unit of measure for weight",
    "ItemVolume": "Volume of the item",
    "ItemVolumeUnit": "Unit of measure for volume",
    
    # Pricing and Financial Fields
    "NetAmount": "Net sales amount (price × quantity - discounts)",
    "TransactionCurrency": "Currency code for the transaction",
    "SalesOrganizationCurrency": "Currency of the sales organization",
    "Currency": "General currency field",
    "NetPriceAmount": "Net price per unit",
    "NetPriceQuantity": "Quantity used for pricing",
    "NetPriceQuantityUnit": "Unit of measure for pricing quantity",
    "TaxAmount": "Tax amount",
    "TAXAMOUNT_CC_CUR": "Tax amount in company code currency",
    "CostAmount": "Cost of goods sold",
    "COSTAMOUNT_CC_CUR": "Cost amount in company code currency",
    "NETAMOUNT_CC_CUR": "Net amount in company code currency",
    "Subtotal1Amount": "Subtotal 1 for pricing calculations",
    "Subtotal2Amount": "Subtotal 2 for pricing calculations",
    "Subtotal3Amount": "Subtotal 3 for pricing calculations",
    "Subtotal4Amount": "Subtotal 4 for pricing calculations",
    "Subtotal5Amount": "Subtotal 5 for pricing calculations",
    "Subtotal6Amount": "Subtotal 6 for pricing calculations",
    "SUBTOTAL1AMOUNT_CC_CUR": "Subtotal 1 in company code currency",
    "SUBTOTAL2AMOUNT_CC_CUR": "Subtotal 2 in company code currency",
    "SUBTOTAL3AMOUNT_CC_CUR": "Subtotal 3 in company code currency",
    "SUBTOTAL4AMOUNT_CC_CUR": "Subtotal 4 in company code currency",
    "SUBTOTAL5AMOUNT_CC_CUR": "Subtotal 5 in company code currency",
    "SUBTOTAL6AMOUNT_CC_CUR": "Subtotal 6 in company code currency",
    "OutlineAgreementTargetAmount": "Target amount for outline agreement (contract)",
    "OUTLINEAGREEMENTTARGAMT_CC_CUR": "Target amount in company code currency",
    
    # Pricing Control
    "PricingDate": "Date used for pricing determination",
    "ExchangeRateDate": "Date used for exchange rate determination",
    "PriceDetnExchangeRate": "Exchange rate used for pricing",
    "ExchangeRateType": "Type of exchange rate (e.g., average, buying, selling)",
    "StatisticalValueControl": "Statistical value control indicator",
    
    # Delivery and Shipping
    "ShippingPoint": "Location from where goods are shipped",
    "ShippingType": "Type of shipping method",
    "DeliveryPriority": "Priority level for delivery",
    "RequestedDeliveryDate": "Date requested by customer for delivery",
    "ShippingCondition": "Shipping conditions (e.g., standard, express)",
    "DeliveryBlockReason": "Reason for delivery block if applicable",
    "Plant": "Manufacturing or distribution plant",
    "StorageLocation": "Storage location within the plant",
    "Route": "Shipping route",
    "InventorySpecialStockType": "Type of special stock (e.g., consignment, project stock)",
    "ServicesRenderedDate": "Date when services were rendered",
    
    # Incoterms
    "IncotermsClassification": "Incoterms code (e.g., FOB, CIF, EXW)",
    "IncotermsVersion": "Version of Incoterms used",
    "IncotermsTransferLocation": "Location where ownership transfers",
    "IncotermsLocation1": "Incoterms location field 1",
    "IncotermsLocation2": "Incoterms location field 2",
    
    # Delivery Tolerances
    "UnlimitedOverdeliveryIsAllowed": "Indicator if unlimited over-delivery is allowed",
    "OverdelivTolrtdLmtRatioInPct": "Over-delivery tolerance limit as percentage",
    "UnderdelivTolrtdLmtRatioInPct": "Under-delivery tolerance limit as percentage",
    "PartialDeliveryIsAllowed": "Indicator if partial delivery is allowed",
    
    # Contract/Agreement Fields
    "BindingPeriodValStartDate": "Start date of binding period for validity",
    "BindingPeriodValEndDate": "End date of binding period for validity",
    "CompletionRule": "Rule for contract completion",
    
    # Billing Information
    "BillingDocumentDate": "Date of billing document (invoice date)",
    "BillingCompanyCode": "Company code for billing",
    "HeaderBillingBlockReason": "Reason for billing block at header level",
    "ItemBillingBlockReason": "Reason for billing block at item level",
    "ItemIsBillingRelevant": "Indicator whether item is billing relevant",
    
    # Financial Accounting
    "FiscalYear": "Fiscal year of the transaction",
    "FiscalPeriod": "Fiscal period within the fiscal year",
    "FiscalYearVariant": "Fiscal year variant used",
    
    # Reference Documents
    "OrderID": "Order ID reference",
    "ReferenceSDDocument": "Reference sales document number",
    "ReferenceSDDocumentItem": "Reference sales document item number",
    "ReferenceSDDocumentCategory": "Reference sales document category",
    "OriginSDDocument": "Original sales document number",
    "OriginSDDocumentItem": "Original sales document item number",
    "SDDocumentReason": "Reason for creating the sales document",
    "SalesDocumentRjcnReason": "Reason for sales document rejection",
    
    # Status Fields - Overall
    "OverallSDProcessStatus": "Overall sales document processing status",
    "OverallTotalDeliveryStatus": "Overall total delivery status",
    "OverallOrdReltdBillgStatus": "Overall order-related billing status",
    "TotalCreditCheckStatus": "Total credit check status",
    "OverallSDDocumentRejectionSts": "Overall sales document rejection status",
    "OverallTotalSDDocRefStatus": "Overall total sales document reference status",
    "OverallSDDocReferenceStatus": "Overall sales document reference status",
    "OverallDelivConfStatus": "Overall delivery confirmation status",
    "OverallDeliveryStatus": "Overall delivery status",
    "DeliveryBlockStatus": "Delivery block status",
    "BillingBlockStatus": "Billing block status",
    
    # Status Fields - Item Level
    "TotalSDDocReferenceStatus": "Total sales document reference status for item",
    "SDDocReferenceStatus": "Sales document reference status for item",
    "SDDocumentRejectionStatus": "Sales document rejection status for item",
    "ItemGeneralIncompletionStatus": "Item general incompletion status",
    "ItemBillingIncompletionStatus": "Item billing incompletion status",
    "PricingIncompletionStatus": "Pricing incompletion status",
    "ItemDeliveryIncompletionStatus": "Item delivery incompletion status",
    "DeliveryConfirmationStatus": "Delivery confirmation status for item",
    "OrderRelatedBillingStatus": "Order-related billing status for item",
    "SDProcessStatus": "Sales document processing status for item",
    "TotalDeliveryStatus": "Total delivery status for item",
    "DeliveryStatus": "Delivery status for item",
    
    # Probability Fields
    "HdrOrderProbabilityInPercent": "Probability percentage at header level (for quotations)",
    "ItemOrderProbabilityInPercent": "Probability percentage at item level (for quotations)",
    
    # Credit/Debit Indicators
    "CreditDebit_Posting_CD": "Credit/debit posting indicator at document level",
    "Creditdebit_posting_Item_Level": "Credit/debit posting indicator at item level",
    
    # Additional Categorization
    "Document_CategoryQuotationOrde": "Document category - whether quotation or order",
    "Relevant_for_Sales": "Indicator whether the record is relevant for sales reporting"
}
