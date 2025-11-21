"""
MCP Context Manager - Checks if query can be answered from existing context
"""
import logging
from typing import Dict, Any, Optional
from data_connector import mcp_store
import re

logger = logging.getLogger(__name__)

class ContextManager:
    """Manages context retrieval and determines if new computation is needed"""
    
    def __init__(self):
        self.name = "ContextManager"
        logger.info(f"✅ {self.name} initialized")
    
    def check_context_for_answer(self, query: str, entities: dict) -> Optional[Dict[str, Any]]:
        """
        Check if query can be answered from existing MCP context
        
        Returns:
            Answer dict if found in context, None if computation needed
        """
        logger.info(f"🔍 {self.name}: Checking MCP context for query: '{query}'")
        logger.debug(f"📋 Entities to search: {entities}")
        
        # Get all agent contexts
        all_contexts = mcp_store.get_all_contexts()
        logger.info(f"📊 Available agent contexts: {list(all_contexts.keys())}")
        
        if not all_contexts:
            logger.warning("⚠️  No contexts available in MCP store")
            logger.info("❌ CACHE MISS: No contexts, computation needed")
            return None
        
        # Strategy 1: Check for specific year data
        if entities and 'year' in entities and not entities.get('comparison'):
            year = entities['year']
            logger.info(f"🔎 Strategy 1: Looking for year {year} data...")
            cached_result = self._find_year_in_context(year, all_contexts)
            if cached_result:
                logger.info(f"✅ CACHE HIT: Found cached data for year {year}")
                return cached_result
            else:
                logger.debug(f"❌ No cached data found for year {year}")
        
        # Strategy 2: Check for comparison data
        if entities and entities.get('comparison') and 'years' in entities:
            years = entities['years']
            logger.info(f"🔎 Strategy 2: Looking for comparison data for years {years}...")
            cached_result = self._find_comparison_in_context(years, all_contexts)
            if cached_result:
                logger.info(f"✅ CACHE HIT: Found cached comparison for years {years}")
                return cached_result
            else:
                logger.debug(f"❌ No cached comparison found for years {years}")
        
        # Strategy 3: Check conversation history
        logger.info("🔎 Strategy 3: Checking conversation history...")
        conversation_history = mcp_store.conversation_history if hasattr(mcp_store, 'conversation_history') else []
        logger.debug(f"📜 Conversation history length: {len(conversation_history)}")
        
        if len(conversation_history) >= 2:
            last_user = conversation_history[-2] if len(conversation_history) >= 2 else None
            last_assistant = conversation_history[-1] if len(conversation_history) >= 1 else None
            
            if (last_user and last_assistant and 
                last_user.get('role') == 'user' and 
                last_assistant.get('role') == 'assistant'):
                
                similarity = self._calculate_similarity(query.lower(), last_user['content'].lower())
                logger.debug(f"📏 Query similarity with last query: {similarity:.2f}")
                
                if similarity > 0.85:
                    logger.info(f"✅ CACHE HIT: Very similar recent query (similarity: {similarity:.2f})")
                    logger.debug(f"   └─ Previous: '{last_user['content']}'")
                    logger.debug(f"   └─ Current:  '{query}'")
                    return {
                        "from_cache": True,
                        "response": last_assistant['content']
                    }
                else:
                    logger.debug(f"❌ Similarity {similarity:.2f} below threshold 0.85")
        
        logger.info("❌ CACHE MISS: No suitable cached data found, computation needed")
        return None
    
    def _find_year_in_context(self, year: int, all_contexts: dict) -> Optional[Dict[str, Any]]:
        """Find cached data for a specific year in agent contexts"""
        logger.debug(f"🔍 Searching for year {year} in {len(all_contexts)} agent contexts...")
        
        # Check AnalysisAgent context
        if 'AnalysisAgent' in all_contexts:
            logger.debug("   └─ Checking AnalysisAgent context...")
            agent_data = all_contexts['AnalysisAgent'].get('data', {})
            filtered_by = agent_data.get('filtered_by', {})
            
            logger.debug(f"      └─ Filtered by: {filtered_by}")
            
            # Check if this agent has data for the requested year
            if filtered_by.get('year') == year:
                logger.info(f"   ✅ Found matching year {year} in AnalysisAgent")
                results = agent_data.get('results', {})
                
                logger.debug(f"      └─ Results status: {results.get('status')}")
                
                if results and results.get('status') == 'success':
                    analysis_results = results.get('results', {})
                    
                    logger.debug(f"      └─ Analysis results keys: {list(analysis_results.keys())}")
                    
                    if analysis_results:
                        response_lines = []
                        
                        if 'total_sales' in analysis_results:
                            response_lines.append(f"💰 **Total Sales:** ${analysis_results['total_sales']:,.2f}\n")
                        if 'total_orders' in analysis_results:
                            response_lines.append(f"📦 **Total Orders:** {analysis_results['total_orders']:,}\n")
                        if 'avg_order_value' in analysis_results:
                            response_lines.append(f"📊 **Average Order Value:** ${analysis_results['avg_order_value']:,.2f}\n")
                        if 'unique_customers' in analysis_results:
                            response_lines.append(f"👥 **Unique Customers:** {analysis_results['unique_customers']:,}\n")
                        if 'unique_products' in analysis_results:
                            response_lines.append(f"🏷️ **Unique Products:** {analysis_results['unique_products']:,}\n")
                        
                        if response_lines:
                            logger.info(f"   ✅ Generated {len(response_lines)} response lines from cache")
                            return {
                                "from_cache": True,
                                "response": "".join(response_lines),
                                "cached_data": analysis_results
                            }
                        else:
                            logger.warning("   ⚠️  No data to format in cached results")
                else:
                    logger.debug("   ❌ Results empty or not successful")
            else:
                logger.debug(f"   ❌ Year mismatch: cached={filtered_by.get('year')}, requested={year}")
        else:
            logger.debug("   ❌ AnalysisAgent not in contexts")
        
        return None
    
    def _find_comparison_in_context(self, years: list, all_contexts: dict) -> Optional[Dict[str, Any]]:
        """Find cached comparison data for multiple years"""
        logger.debug(f"🔍 Searching for comparison data for years {years}...")
        
        # Check if AnalysisAgent has comparison data
        if 'AnalysisAgent' in all_contexts:
            logger.debug("   └─ Checking AnalysisAgent context...")
            agent_data = all_contexts['AnalysisAgent'].get('data', {})
            filtered_by = agent_data.get('filtered_by', {})
            
            logger.debug(f"      └─ Filtered by: {filtered_by}")
            
            # Check if this is a comparison with matching years
            if filtered_by.get('comparison') and 'years' in filtered_by:
                cached_years = set(filtered_by['years'])
                requested_years = set(years)
                
                logger.debug(f"      └─ Cached years: {cached_years}")
                logger.debug(f"      └─ Requested years: {requested_years}")
                
                if cached_years == requested_years:
                    logger.info(f"   ✅ Found matching comparison for years {years}")
                    results = agent_data.get('results', {})
                    
                    if results and results.get('status') == 'success':
                        comparison_results = results.get('results', {})
                        
                        logger.debug(f"      └─ Comparison results keys: {list(comparison_results.keys())}")
                        
                        if 'comparison' in comparison_results:
                            response_lines = []
                            comp = comparison_results['comparison']
                            
                            response_lines.append(f"📊 **Comparison Results:**\n")
                            response_lines.append(f"  • Sales Difference: ${comp.get('sales_difference', 0):,.2f}\n")
                            response_lines.append(f"  • Growth: {comp.get('growth_percentage', 0):.2f}%\n")
                            
                            for year_key, year_data in comparison_results.items():
                                if year_key.startswith('year_') and isinstance(year_data, dict):
                                    year = year_key.split('_')[1]
                                    response_lines.append(f"\n**Year {year}:**\n")
                                    response_lines.append(f"  • Total Sales: ${year_data.get('total_sales', 0):,.2f}\n")
                                    if 'total_orders' in year_data:
                                        response_lines.append(f"  • Total Orders: {year_data['total_orders']:,}\n")
                                    if 'avg_order_value' in year_data:
                                        response_lines.append(f"  • Avg Order Value: ${year_data['avg_order_value']:,.2f}\n")
                            
                            if response_lines:
                                logger.info(f"   ✅ Generated {len(response_lines)} response lines from comparison cache")
                                return {
                                    "from_cache": True,
                                    "response": "".join(response_lines),
                                    "cached_data": comparison_results
                                }
                        else:
                            logger.warning("   ⚠️  No comparison key in results")
                    else:
                        logger.debug("   ❌ Results empty or not successful")
                else:
                    logger.debug(f"   ❌ Year mismatch: cached={cached_years}, requested={requested_years}")
            else:
                logger.debug("   ❌ Not a comparison or no years in filtered_by")
        else:
            logger.debug("   ❌ AnalysisAgent not in contexts")
        
        return None
    
    def _calculate_similarity(self, query1: str, query2: str) -> float:
        """Calculate similarity between two queries"""
        stop_words = {'what', 'are', 'is', 'the', 'in', 'for', 'of', 'to', 'a', 'an'}
        
        words1 = set(query1.split()) - stop_words
        words2 = set(query2.split()) - stop_words
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        similarity = len(intersection) / len(union) if union else 0.0
        
        logger.debug(f"   └─ Words 1: {words1}")
        logger.debug(f"   └─ Words 2: {words2}")
        logger.debug(f"   └─ Intersection: {intersection}")
        logger.debug(f"   └─ Similarity: {similarity:.2f}")
        
        return similarity
