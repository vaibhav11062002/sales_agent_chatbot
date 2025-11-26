"""
Dynamic MCP Context Manager - Uses LLM for context resolution with entity-aware matching
"""
import logging
import os
import re
import json
from typing import Dict, Any, Optional, List
from data_connector import mcp_store
from config import GEMINI_API_KEY

logger = logging.getLogger(__name__)

class ContextManager:
    """Dynamic context manager using LLM for entity resolution and semantic matching"""
    
    def __init__(self):
        self.name = "ContextManager"
        
        # Initialize LLM for context resolution
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0,
                api_key=GEMINI_API_KEY
            )
            logger.info(f"✅ {self.name} initialized with LLM-powered context resolution")
        except Exception as e:
            logger.warning(f"⚠️ Could not initialize LLM: {e}")
            self.llm = None
    
    def check_context_for_answer(self, query: str, entities: dict) -> Optional[Dict[str, Any]]:
        """
        Dynamically check if query can be answered from context using LLM
        with entity-aware matching to prevent false cache hits
        """
        logger.info(f"🔍 {self.name}: Checking context for query: '{query}'")
        logger.info(f"   └─ Query entities: {entities}")
        
        # Get context stack
        context_stack = mcp_store.get_context_stack()
        if not context_stack:
            logger.info("❌ CACHE MISS: No context history")
            return None
        
        # Strategy 1: LLM-powered semantic context matching
        if self.llm:
            logger.info("🤖 Using LLM for dynamic context resolution...")
            llm_result = self._llm_resolve_context(query, context_stack, entities)
            if llm_result:
                return llm_result
        
        # Strategy 2: Similarity-based matching with entity checking
        logger.info("📊 Using similarity-based matching...")
        similar_contexts = mcp_store.get_similar_contexts(query, top_k=3)
        
        if similar_contexts:
            best_match = similar_contexts[0]
            best_match_entities = best_match.get('entities', {})
            
            # Calculate similarity with entity awareness
            similarity_score = self._calculate_query_similarity(
                query, 
                best_match['query'],
                entities,
                best_match_entities
            )
            
            logger.info(f"📏 Best match similarity: {similarity_score:.2f}")
            logger.info(f"   └─ Matched query: '{best_match['query']}'")
            
            if similarity_score > 0.6:
                logger.info(f"✅ CACHE HIT: High similarity match")
                return {
                    "from_cache": True,
                    "response": best_match['response'],
                    "cached_query": best_match['query'],
                    "similarity": similarity_score
                }
        
        # Strategy 3: Check agent contexts with entity validation
        all_contexts = mcp_store.get_all_contexts()
        if all_contexts:
            dynamic_result = self._dynamic_agent_context_search(query, entities, all_contexts)
            if dynamic_result:
                return dynamic_result
        
        logger.info("❌ CACHE MISS: No suitable context found")
        return None
    
    def _llm_resolve_context(self, query: str, context_stack: List[Dict], entities: dict) -> Optional[Dict[str, Any]]:
        """Use LLM to intelligently match query with past contexts"""
        try:
            # Build context summary
            context_summary = "\n".join([
                f"{i+1}. Query: '{ctx['query']}' | Entities: {ctx.get('entities', {})} | Type: {ctx.get('query_type', 'N/A')}"
                for i, ctx in enumerate(context_stack[:5])
            ])
            
            prompt = f"""You are a context resolution assistant. Determine if a new query can be answered using past contexts.

**Current Query:** "{query}"
**Current Entities:** {entities}

**Past Contexts:**
{context_summary}

**CRITICAL RULES:**
1. If year/customer/product entities differ → can_reuse = false
2. "total sales 2024" ≠ "total sales 2025" (different years!)
3. "customer 1002" ≠ "customer 1003" (different customers!)
4. "product OX140" ≠ "product OX141" (different products!)
5. Check if pronouns or references (like "that", "it", "those") refer to entities from past contexts
6. Only reuse if asking about EXACTLY the same data

**Response Format (JSON only, no markdown):**
{{
  "can_reuse": true/false,
  "matched_context_index": <1-5 or null>,
  "reasoning": "why or why not",
  "resolved_entities": {{"entity": "value"}}
}}

Analyze carefully and respond with JSON only."""

            response = self.llm.invoke(prompt)
            content = response.content
            
            # Parse LLM response
            json_match = re.search(r'\{[^}]+\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                
                if result.get('can_reuse') and result.get('matched_context_index'):
                    idx = result['matched_context_index'] - 1
                    if 0 <= idx < len(context_stack):
                        matched_ctx = context_stack[idx]
                        
                        # Double-check: Verify entities match
                        matched_entities = matched_ctx.get('entities', {})
                        if not self._entities_compatible(entities, matched_entities):
                            logger.info(f"   ❌ LLM suggested reuse but entities don't match - rejecting")
                            return None
                        
                        logger.info(f"✅ LLM matched context #{idx+1}")
                        logger.info(f"   └─ Reasoning: {result.get('reasoning', 'N/A')}")
                        
                        # Fetch full response
                        full_response = self._get_full_response_for_context(matched_ctx)
                        
                        return {
                            "from_cache": True,
                            "response": full_response or matched_ctx['response'],
                            "cached_query": matched_ctx['query'],
                            "llm_reasoning": result.get('reasoning'),
                            "resolved_entities": result.get('resolved_entities', {})
                        }
            
            logger.info("🤖 LLM: No reusable context found")
            return None
            
        except Exception as e:
            logger.error(f"LLM context resolution failed: {e}")
            return None
    
    def _entities_compatible(self, entities1: dict, entities2: dict) -> bool:
        """
        Check if two entity sets are compatible (no critical mismatches).
        Returns False if any critical entity has different values.
        """
        critical_keys = ['year', 'customer_id', 'product_id', 'sales_org']
        
        for key in critical_keys:
            val1 = entities1.get(key)
            val2 = entities2.get(key)
            
            # If both have this key but values differ → incompatible
            if val1 is not None and val2 is not None and str(val1) != str(val2):
                logger.info(f"   ❌ Entity incompatibility: {key} ({val1} ≠ {val2})")
                return False
        
        return True
    
    def _calculate_query_similarity(self, query1: str, query2: str, entities1: dict = None, entities2: dict = None) -> float:
        """
        Calculate semantic similarity between queries with entity-aware matching.
        Returns 0.0 if critical entities (year, customer, product) don't match.
        """
        
        # CRITICAL: Check entity mismatches first
        if entities1 and entities2:
            critical_entities = ['year', 'customer_id', 'product_id', 'sales_org']
            
            for entity_key in critical_entities:
                val1 = entities1.get(entity_key)
                val2 = entities2.get(entity_key)
                
                # If both queries have this entity but values differ → NO MATCH
                if val1 is not None and val2 is not None and str(val1) != str(val2):
                    logger.info(f"   ❌ Entity mismatch: {entity_key} ({val1} ≠ {val2}) → similarity = 0.0")
                    return 0.0
        
        # Extract years from query text and compare
        years1 = re.findall(r'\b(20\d{2})\b', query1)
        years2 = re.findall(r'\b(20\d{2})\b', query2)
        
        if years1 and years2:
            year1 = int(years1[0])
            year2 = int(years2[0])
            if year1 != year2:
                logger.info(f"   ❌ Year mismatch in query text: {year1} ≠ {year2} → similarity = 0.0")
                return 0.0
        
        # Extract customer IDs from query text
        customer1 = re.findall(r'customer\s+(\d+)', query1.lower())
        customer2 = re.findall(r'customer\s+(\d+)', query2.lower())
        
        if customer1 and customer2 and customer1[0] != customer2[0]:
            logger.info(f"   ❌ Customer mismatch: {customer1[0]} ≠ {customer2[0]} → similarity = 0.0")
            return 0.0
        
        # Extract product IDs from query text
        product1 = re.findall(r'product\s+([A-Z0-9]+)', query1, re.IGNORECASE)
        product2 = re.findall(r'product\s+([A-Z0-9]+)', query2, re.IGNORECASE)
        
        if product1 and product2 and product1[0].upper() != product2[0].upper():
            logger.info(f"   ❌ Product mismatch: {product1[0]} ≠ {product2[0]} → similarity = 0.0")
            return 0.0
        
        # Only if no critical mismatches, calculate word similarity
        stop_words = {'what', 'are', 'is', 'the', 'in', 'for', 'of', 'to', 'a', 'an', 'was', 'were', 'will', 'be', 'year', 'customer', 'product'}
        
        # Remove numbers from word comparison (already checked above)
        words1 = set(w for w in query1.lower().split() if w not in stop_words and not w.isdigit())
        words2 = set(w for w in query2.lower().split() if w not in stop_words and not w.isdigit())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        similarity = len(intersection) / len(union) if union else 0.0
        logger.info(f"   📊 Word similarity (after entity check): {similarity:.2f}")
        
        return similarity
    
    def _dynamic_agent_context_search(self, query: str, entities: dict, all_contexts: dict) -> Optional[Dict[str, Any]]:
        """Dynamically search agent contexts with entity validation"""
        
        for agent_name, agent_ctx in all_contexts.items():
            agent_data = agent_ctx.get('data', {})
            
            past_query = agent_data.get('query', '')
            
            # Get past entities from multiple possible locations
            past_entities = (
                agent_data.get('filtered_by', {}) or 
                agent_data.get('extracted_entities', {}) or 
                agent_data.get('entities', {})
            )
            
            # Check entity compatibility first
            if not self._entities_compatible(entities, past_entities):
                continue
            
            # Check query similarity
            similarity = self._calculate_query_similarity(query, past_query, entities, past_entities)
            
            if similarity > 0.5:
                logger.info(f"   ✅ Found similar query in {agent_name} (similarity: {similarity:.2f})")
                
                results = agent_data.get('results', {})
                if results.get('status') == 'success':
                    analysis_results = results.get('results', {})
                    
                    if 'llm_raw' in analysis_results:
                        clean_response = self._extract_clean_llm_response(analysis_results['llm_raw'])
                        return {
                            "from_cache": True,
                            "response": clean_response,
                            "cached_query": past_query,
                            "similarity": similarity
                        }
        
        return None
    
    def _get_full_response_for_context(self, context: Dict) -> Optional[str]:
        """Retrieve full response from agent contexts based on matched context"""
        all_contexts = mcp_store.get_all_contexts()
        
        for agent_name, agent_ctx in all_contexts.items():
            agent_data = agent_ctx.get('data', {})
            if agent_data.get('query') == context['query']:
                results = agent_data.get('results', {})
                if results.get('status') == 'success':
                    analysis_results = results.get('results', {})
                    
                    if 'llm_raw' in analysis_results:
                        return self._extract_clean_llm_response(analysis_results['llm_raw'])
        
        return None
    
    def _extract_clean_llm_response(self, llm_raw: str) -> str:
        """Extract clean answer from LLM raw output"""
        
        # Remove code blocks
        cleaned = re.sub(r'``````', '', llm_raw, flags=re.DOTALL)
        
        # Extract bullet points and formatted text
        lines = cleaned.split('\n')
        answer_lines = []
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('*') or stripped.startswith('-') or '**' in stripped:
                answer_lines.append(stripped)
            elif stripped and len(stripped) > 20 and not stripped.startswith('#'):
                answer_lines.append(stripped)
        
        if answer_lines:
            return '\n'.join(answer_lines)
        
        # Fallback: return cleaned text
        return cleaned.strip()[:500] if cleaned.strip() else llm_raw[:500]
