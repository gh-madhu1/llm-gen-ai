"""
Knowledge Verifier Module
Verifies what the model knows and shows confidence before answering.
"""

from typing import Dict, Tuple


class KnowledgeVerifier:
    """Verifies model knowledge and shows transparency to user."""
    
    def __init__(self):
        # Model knowledge cutoff
        self.knowledge_cutoff_year = 2022
        self.current_year = 2024
        
    def assess_knowledge(self, analysis: Dict) -> Dict:
        """
        Assess what the model knows about this query.
        
        Args:
            analysis: Query analysis from QueryAnalyzer
            
        Returns:
            Knowledge assessment dictionary
        """
        assessment = {
            'can_answer_from_knowledge': False,
            'knowledge_level': 'none',
            'needs_search': analysis['needs_search'],
            'reason': '',
            'recommendation': ''
        }
        
        #Check if time-sensitive
        if analysis['time_sensitive']:
            assessment['can_answer_from_knowledge'] = False
            assessment['knowledge_level'] = 'outdated'
            assessment['reason'] = f"Query requires information after {self.knowledge_cutoff_year}"
            assessment['recommendation'] = "Web search required for current information"
            return assessment
        
        # Check query type
        query_type = analysis['query_type']
        
        if query_type in ['definition', 'explanation', 'how-to']:
            # High confidence for general knowledge
            assessment['can_answer_from_knowledge'] = True
            assessment['knowledge_level'] = 'high'
            assessment['reason'] = "General knowledge within training data"
            assessment['recommendation'] = "Can answer from model knowledge, web search for verification"
        
        elif query_type == 'current-events':
            assessment['can_answer_from_knowledge'] = False
            assessment['knowledge_level'] = 'none'
            assessment['reason'] = "Current events beyondtraining data"
            assessment['recommendation'] = "Web search required"
        
        elif query_type == 'comparison':
            assessment['can_answer_from_knowledge'] = True
            assessment['knowledge_level'] = 'medium'
            assessment['reason'] = "Can compare concepts, but may need current data"
            assessment['recommendation'] = "Model knowledge + web search for latest info"
        
        else:
            # General query
            confidence = analysis.get('confidence', 'medium')
            if confidence == 'high':
                assessment['can_answer_from_knowledge'] = True
                assessment['knowledge_level'] = 'high'
                assessment['reason'] = "High confidence in model knowledge"
                assessment['recommendation'] = "Answer from knowledge, optional search for verification"
            elif confidence == 'medium':
                assessment['can_answer_from_knowledge'] = True
                assessment['knowledge_level'] = 'medium'
                assessment['reason'] = "Moderate confidence, verification recommended"
                assessment['recommendation'] = "Model knowledge + web search recommended"
            else:
                assessment['can_answer_from_knowledge'] = False
                assessment['knowledge_level'] = 'low'
                assessment['reason'] = "Low confidence in answer accuracy"
                assessment['recommendation'] = "Web search required"
        
        return assessment
    
    def format_verification(self, assessment: Dict, query_analysis: Dict) -> str:
        """
        Format verification results for display.
        
        Args:
            assessment: Knowledge assessment
            query_analysis: Original query analysis
            
        Returns:
            Formatted verification string
        """
        output = []
        output.append("\n🔍 Knowledge Verification:")
        output.append(f"   My knowledge cutoff: {self.knowledge_cutoff_year}")
        output.append(f"   Query requires info from: {self.current_year if query_analysis['time_sensitive'] else 'General knowledge'}")
        output.append(f"   Knowledge level: {assessment['knowledge_level'].title()}")
        output.append(f"   Reason: {assessment['reason']}")
        output.append(f"   ✅ Recommendation: {assessment['recommendation']}")
        
        return '\n'.join(output)
    
    def should_search(self, assessment: Dict) -> bool:
        """
        Determine if web search should be performed.
        
        Args:
            assessment: Knowledge assessment
            
        Returns:
            bool: True if search recommended
        """
        return assessment['needs_search'] or assessment['knowledge_level'] in ['low', 'outdated', 'none']
