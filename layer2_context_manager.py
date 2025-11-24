"""
Dynamic MCP Context Manager - Uses LLM for context resolution
"""
import logging
import os
from typing import Dict, Any, Optional, List
from data_connector import mcp_store

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
                api_key="AIzaSyBvGk-pDi2hqdq0CLSoKV2Sa8TH5IWShtE"
            )
            logger.info(f"✅ {self.name} initialized with LLM-powered context resolution")
        except Exception as e:
            logger.warning(f"⚠️ Could not initialize LLM: {e}")
            self.llm = None
    
    def check_context_for_answer(self, query: str, entities: dict) -> Optional[Dict[str, Any]]:
        """
        Dynamically check if query can be answered from context using LLM
        """
        logger.info(f"🔍 {self.name}: Checking context for query: '{query}'")
        
        # Get context stack
        context_stack = mcp_store.get_context_stack()
        if not context_stack:
            logger.info("❌ CACHE MISS: No context history")
            return None
        
        # Strategy 1: LLM-powered semantic context matching
        if self.llm:
            logger.info("🤖 Using LLM for dynamic context resolution...")
            llm_result = self._llm_resolve_context(query, context_stack)
            if llm_result:
                return llm_result
        
        # Strategy 2: Similarity-based matching (fallback)
        logger.info("📊 Using similarity-based matching...")
        similar_contexts = mcp_store.get_similar_contexts(query, top_k=3)
        
        if similar_contexts:
            best_match = similar_contexts[0]
            similarity_score = self._calculate_query_similarity(query, best_match['query'])
            
            logger.info(f"📏 Best match similarity: {similarity_score:.2f}")
            logger.info(f"   └─ Matched query: '{best_match['query']}'")
            
            if similarity_score > 0.6:  # Lower threshold for dynamic matching
                logger.info(f"✅ CACHE HIT: High similarity match")
                return {
                    "from_cache": True,
                    "response": best_match['response'],
                    "cached_query": best_match['query'],
                    "similarity": similarity_score
                }
        
        # Strategy 3: Check agent contexts with dynamic extraction
        all_contexts = mcp_store.get_all_contexts()
        if all_contexts:
            dynamic_result = self._dynamic_agent_context_search(query, entities, all_contexts)
            if dynamic_result:
                return dynamic_result
        
        logger.info("❌ CACHE MISS: No suitable context found")
        return None
    
    def _llm_resolve_context(self, query: str, context_stack: List[Dict]) -> Optional[Dict[str, Any]]:
        """Use LLM to intelligently match query with past contexts"""
        try:
            # Build context summary
            context_summary = "\n".join([
                f"{i+1}. Query: '{ctx['query']}' | Entities: {ctx['entities']} | Type: {ctx['query_type']}"
                for i, ctx in enumerate(context_stack[:5])
            ])
            
            prompt = f"""You are a context resolution assistant. Given a new query and past conversation contexts, determine if the new query can be answered using information from past contexts.

**Current Query:** "{query}"

**Past Contexts:**
{context_summary}

**Task:**
1. Analyze if the current query is asking about the SAME information as any past query
2. Check if pronouns or references (like "that", "it", "those") refer to entities from past contexts
3. Determine if we can reuse a past response

**Response Format (JSON):**
{{
  "can_reuse": true/false,
  "matched_context_index": <index from 1-5 or null>,
  "reasoning": "brief explanation",
  "resolved_entities": {{"entity_type": "value"}}
}}

Think carefully about semantic similarity, not just exact matches."""

            response = self.llm.invoke(prompt)
            content = response.content
            
            # Parse LLM response
            import json
            import re
            
            json_match = re.search(r'\{[^}]+\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                
                if result.get('can_reuse') and result.get('matched_context_index'):
                    idx = result['matched_context_index'] - 1
                    if 0 <= idx < len(context_stack):
                        matched_ctx = context_stack[idx]
                        
                        logger.info(f"✅ LLM matched context #{idx+1}")
                        logger.info(f"   └─ Reasoning: {result.get('reasoning', 'N/A')}")
                        
                        # Check if we need to fetch full response from agent context
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
                        # Extract clean response
                        return self._extract_clean_llm_response(analysis_results['llm_raw'])
        
        return None
    
    def _extract_clean_llm_response(self, llm_raw: str) -> str:
        """Extract clean answer from LLM raw output"""
        import re
        
        # Remove code blocks
        cleaned = re.sub(r'``````', '', llm_raw, flags=re.DOTALL)
        
        # Extract bullet points
        lines = cleaned.split('\n')
        answer_lines = []
        
        for line in lines:
            if line.strip().startswith('*') or line.strip().startswith('-'):
                answer_lines.append(line.strip())
        
        if answer_lines:
            return '\n'.join(answer_lines)
        
        # Fallback: return cleaned text
        return cleaned.strip()[:500]
    
    def _dynamic_agent_context_search(self, query: str, entities: dict, all_contexts: dict) -> Optional[Dict[str, Any]]:
        """Dynamically search agent contexts without field restrictions"""
        
        for agent_name, agent_ctx in all_contexts.items():
            agent_data = agent_ctx.get('data', {})
            
            # Check if query is similar enough
            past_query = agent_data.get('query', '')
            similarity = self._calculate_query_similarity(query, past_query)
            
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
    
    def _calculate_query_similarity(self, query1: str, query2: str) -> float:
        """Calculate semantic similarity between queries"""
        stop_words = {'what', 'are', 'is', 'the', 'in', 'for', 'of', 'to', 'a', 'an', 'was', 'were', 'will', 'be'}
        
        words1 = set(query1.lower().split()) - stop_words
        words2 = set(query2.lower().split()) - stop_words
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0
