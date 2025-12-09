"""
Answer Synthesizer Module
Combines model knowledge with web search results to create accurate answers.
"""

from typing import Dict, List, Optional


class AnswerSynthesizer:
    """Synthesizes answers from multiple sources with citations."""
    
    def __init__(self):
        pass
    
    def create_prompt(self, query: str, analysis: Dict, assessment: Dict, 
                     search_results: Optional[Dict] = None, use_history: bool = False,
                     conversation_history: List[str] = None) -> str:
        """
        Create an optimized prompt combining all information.
        
        Args:
            query: User's question
            analysis: Query analysis
            assessment: Knowledge assessment
            search_results: Web search results (if available)
            use_history: Whether to include conversation history
            conversation_history: Previous conversation
            
        Returns:
            Optimized prompt string
        """
        prompt_parts = []
        
        # System instruction
        prompt_parts.append("You are a helpful AI assistant providing accurate, well-researched answers.")
        
        # Add conversation history if requested
        if use_history and conversation_history:
            context = "\n".join(conversation_history[-4:])
            prompt_parts.append(f"\nConversation History:\n{context}")
        
        # Add search results if available
        if search_results and search_results.get('results'):
            prompt_parts.append(f"\nWeb Search Results ({search_results.get('count', 0)} sources):")
            prompt_parts.append(search_results['results'])
            prompt_parts.append("\nIMPORTANT: Base your answer primarily on the above search results. Cite sources when possible.")
        
        # Add query with context
        prompt_parts.append(f"\nUser Question: {query}")
        
        # Add specific instructions based on assessment
        if assessment.get('knowledge_level') == 'high':
            prompt_parts.append("\nYou have strong knowledge on this topic. Provide a comprehensive answer.")
        elif assessment.get('knowledge_level') in ['low', 'outdated']:
            prompt_parts.append("\nThis requires current information. Use the search results above to provide an up-to-date answer.")
        else:
            prompt_parts.append("\nCombine your knowledge with the search results to provide a well-rounded answer.")
        
        # Response format
        prompt_parts.append("\nProvide a clear, accurate answer. If citing search results, mention them naturally in your response.")
        prompt_parts.append("\nAssistant:")
        
        return '\n'.join(prompt_parts)
    
    def calculate_confidence(self, assessment: Dict, has_search_results: bool) -> str:
        """
        Calculate overall confidence in the answer.
        
        Args:
            assessment: Knowledge assessment
            has_search_results: Whether web search was performed
            
        Returns:
            Confidence level string
        """
        knowledge_level = assessment.get('knowledge_level', 'medium')
        
        if has_search_results:
            # High confidence with search verification
            if knowledge_level in ['high', 'medium']:
                return 'Very High (Knowledge + Search Verified)'
            else:
                return 'High (Search Verified)'
        else:
            # Confidence based on knowledge alone
            if knowledge_level == 'high':
                return 'High (Knowledge Based)'
            elif knowledge_level == 'medium':
                return 'Medium (Knowledge Based)'
            else:
                return 'Low (Limited Knowledge)'
    
    def format_answer_with_metadata(self, answer: str, confidence: str, 
                                    sources: List[Dict] = None) -> str:
        """
        Format final answer with confidence and sources.
        
        Args:
            answer: Generated answer
            confidence: Confidence level
            sources: Citation sources
            
        Returns:
            Formatted answer with metadata
        """
        output = [answer]
        
        # Add confidence
        output.append(f"\n\n📊 Confidence: {confidence}")
        
        # Add sources if available
        if sources and len(sources) > 0:
            output.append("\n📚 Sources:")
            for i, source in enumerate(sources[:5], 1):
                output.append(f"   {i}. {source.get('title', 'Unknown')}")
                if source.get('url'):
                    output.append(f"      {source['url']}")
        
        return '\n'.join(output)
    
    def synthesize(self, query: str, analysis: Dict, assessment: Dict,
                  search_results: Optional[Dict], model_answer: str) -> Dict:
        """
        Synthesize final answer with all metadata.
        
        Args:
            query: User query
            analysis: Query analysis
            assessment: Knowledge assessment
            search_results: Search results (if any)
            model_answer: Generated answer from model
            
        Returns:
            Complete answer package
        """
        # Calculate confidence
        has_search = search_results is not None and search_results.get('results')
        confidence = self.calculate_confidence(assessment, has_search)
        
       # Get sources
        sources = search_results.get('sources', []) if search_results else []
        
        # Format complete answer
        formatted_answer = self.format_answer_with_metadata(
            model_answer, 
            confidence,
            sources
        )
        
        return {
            'answer': formatted_answer,
            'confidence': confidence,
            'sources': sources,
            'used_search': has_search,
            'query_type': analysis.get('query_type', 'general')
        }
