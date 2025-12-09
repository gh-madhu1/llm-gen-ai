"""
Clarification Handler Module
Detects when clarification is needed and generates relevant questions.
"""

import re
from typing import Dict, List, Optional, Tuple


class ClarificationHandler:
    """Handles query clarification through targeted questions."""
    
    def __init__(self):
        # Vague pronouns that signal ambiguity
        self.vague_pronouns = ['it', 'that', 'this', 'those', 'these', 'them', 'they']
        
        # Broad terms needing context
        self.broad_terms = ['thing', 'stuff', 'one', 'latest', 'best', 'good', 'better']
        
        # Question starters indicating potential ambiguity
        self.ambiguous_starters = [
            'tell me about', 'what about', 'how about', 'explain'
        ]
        
    def detect_ambiguity(self, query: str, analysis: Dict) -> bool:
        """
        Detect if query is ambiguous and needs clarification.
        
        Args:
            query: User's original question
            analysis: Query analysis from QueryAnalyzer
            
        Returns:
            True if clarification is needed
        """
        query_lower = query.lower().strip()
        words = query_lower.split()
        
        # Very short queries are often ambiguous
        if len(words) <= 3:
            # Check for vague pronouns
            if any(pronoun in words for pronoun in self.vague_pronouns):
                return True
            
            # Check for broad terms without context
            if any(term in words for term in self.broad_terms):
                return True
        
        # Queries starting with ambiguous patterns
        if any(query_lower.startswith(starter) for starter in self.ambiguous_starters):
            # "Tell me about X" - check if X is specific enough
            if len(words) <= 4:
                return True
        
        # Missing entities for queries that need them
        if analysis.get('query_type') in ['definition', 'explanation']:
            if not analysis.get('entities') and len(words) <= 5:
                return True
        
        # Comparison queries missing one side
        if 'vs' in query_lower or 'versus' in query_lower:
            # Should have at least 2 entities or keywords
            if len(analysis.get('entities', [])) + len(analysis.get('keywords', [])) < 3:
                return True
        
        # "Latest" without specific topic
        if 'latest' in query_lower or 'recent' in query_lower:
            if len(analysis.get('keywords', [])) <= 2:
                return True
        
        return False
    
    def calculate_ambiguity_score(self, query: str, analysis: Dict) -> float:
        """
        Calculate ambiguity score from 0.0 (clear) to 1.0 (very ambiguous).
        
        Args:
            query: User's query
            analysis: Query analysis
            
        Returns:
            Ambiguity score
        """
        score = 0.0
        query_lower = query.lower()
        words = query_lower.split()
        
        # Factor 1: Query length (shorter = more ambiguous)
        if len(words) <= 3:
            score += 0.3
        elif len(words) <= 5:
            score += 0.15
        
        # Factor 2: Vague pronouns
        vague_count = sum(1 for word in words if word in self.vague_pronouns)
        score += min(vague_count * 0.2, 0.4)
        
        # Factor 3: Lack of entities
        if not analysis.get('entities'):
            score += 0.2
        
        # Factor 4: Broad terms
        broad_count = sum(1 for word in words if word in self.broad_terms)
        score += min(broad_count * 0.15, 0.3)
        
        # Factor 5: Missing context for "latest" queries
        if ('latest' in query_lower or 'recent' in query_lower) and len(words) <= 4:
            score += 0.25
        
        return min(score, 1.0)
    
    def identify_missing_context(self, query: str, analysis: Dict) -> List[str]:
        """
        Identify what context is missing from the query.
        
        Args:
            query: User's query
            analysis: Query analysis
            
        Returns:
            List of missing context types
        """
        missing = []
        query_lower = query.lower()
        words = query_lower.split()
        
        # Check for vague references
        if any(pronoun in words for pronoun in self.vague_pronouns):
            missing.append('subject_clarification')
        
        # Check for missing topic in "latest" queries
        if ('latest' in query_lower or 'recent' in query_lower) and len(analysis.get('keywords', [])) <= 2:
            missing.append('topic_specification')
        
        # Check for missing scope
        if any(term in query_lower for term in ['best', 'good', 'better']) and 'for' not in query_lower:
            missing.append('use_case_context')
        
        # Check for incomplete comparison
        if 'vs' in query_lower or 'versus' in query_lower or 'compare' in query_lower:
            if len(analysis.get('entities', [])) < 2:
                missing.append('comparison_items')
        
        # Check for missing time frame
        if analysis.get('time_sensitive') and not any(year in query_lower for year in ['2024', '2025', '2023']):
            missing.append('time_frame')
        
        # Generic explanation requests
        if query_lower.startswith('explain') and len(words) <= 3:
            missing.append('explanation_scope')
        
        return missing
    
    def generate_clarifying_questions(self, query: str, analysis: Dict) -> List[str]:
        """
        Generate relevant clarifying questions based on ambiguity.
        
        Args:
            query: User's query
            analysis: Query analysis
            
        Returns:
            List of 2-3 clarifying questions
        """
        questions = []
        missing_context = self.identify_missing_context(query, analysis)
        query_lower = query.lower()
        
        # Subject clarification
        if 'subject_clarification' in missing_context:
            questions.append("What specific topic or subject are you asking about?")
        
        # Topic specification for "latest" queries
        if 'topic_specification' in missing_context:
            if 'ai' in query_lower or 'ml' in query_lower:
                questions.append("Are you asking about AI models, research papers, tools, or applications?")
            elif 'tech' in query_lower:
                questions.append("Which technology area are you interested in (e.g., software, hardware, frameworks)?")
            else:
                questions.append("Could you specify the domain or field you're interested in?")
        
        # Use case context
        if 'use_case_context' in missing_context:
            questions.append("What is your intended use case or application?")
        
        # Comparison items
        if 'comparison_items' in missing_context:
            questions.append("What specific items or options would you like to compare?")
        
        # Time frame
        if 'time_frame' in missing_context:
            questions.append("What time period are you interested in (e.g., current year, last 6 months)?")
        
        # Explanation scope
        if 'explanation_scope' in missing_context:
            questions.append("Would you like a high-level overview or detailed technical explanation?")
        
        # Fallback generic questions based on query type
        if not questions:
            if analysis.get('query_type') == 'how-to':
                questions.append("What is your experience level with this topic (beginner, intermediate, advanced)?")
            elif analysis.get('query_type') == 'definition':
                questions.append("Are you looking for a simple definition or detailed explanation with examples?")
            else:
                questions.append("Could you provide more context about what you're trying to learn or accomplish?")
        
        # Limit to 2-3 most relevant questions
        return questions[:3]
    
    def collect_user_responses(
        self, 
        questions: List[str], 
        interactive: bool = True
    ) -> Optional[Dict[str, str]]:
        """
        Collect responses to clarifying questions.
        
        Args:
            questions: List of clarifying questions
            interactive: Whether to prompt user interactively
            
        Returns:
            Dictionary mapping questions to answers, or None if skipped
        """
        if not interactive:
            return None
        
        print("\n" + "="*60)
        print("🤔 I need some clarification to provide the best answer:")
        print("="*60)
        
        responses = {}
        
        for i, question in enumerate(questions, 1):
            print(f"\n{i}. {question}")
            
        print("\nYou can:")
        print("  - Answer any/all questions (one per line)")
        print("  - Type 'skip' to proceed without clarification")
        print("  - Press Enter after each answer (empty line when done)")
        print("-"*60)
        
        try:
            for i, question in enumerate(questions, 1):
                response = input(f"\nAnswer {i} (or 'skip'): ").strip()
                
                if response.lower() == 'skip':
                    print("⏭️  Skipping clarification, proceeding with original query...")
                    return None
                
                if response:
                    responses[question] = response
            
            if responses:
                print(f"\n✅ Got {len(responses)} clarification(s), thank you!")
                return responses
            else:
                print("⏭️  No clarifications provided, proceeding with original query...")
                return None
                
        except (KeyboardInterrupt, EOFError):
            print("\n⏭️  Skipping clarification...")
            return None
    
    def augment_query_with_responses(
        self, 
        query: str, 
        responses: Optional[Dict[str, str]]
    ) -> str:
        """
        Augment original query with clarification responses.
        
        Args:
            query: Original query
            responses: User's responses to clarifying questions
            
        Returns:
            Augmented query with additional context
        """
        if not responses:
            return query
        
        # Build context from responses
        context_parts = []
        
        for question, answer in responses.items():
            # Extract key context from answer
            context_parts.append(answer)
        
        # Combine original query with context
        if context_parts:
            augmented = f"{query}. Context: {' '.join(context_parts)}"
            return augmented
        
        return query
    
    def format_clarification_summary(
        self, 
        original_query: str, 
        augmented_query: str,
        responses: Optional[Dict[str, str]]
    ) -> str:
        """
        Format a summary of the clarification process.
        
        Args:
            original_query: Original query
            augmented_query: Query with clarifications
            responses: Clarification responses
            
        Returns:
            Formatted summary string
        """
        output = []
        output.append("\n📝 Clarification Summary:")
        output.append(f"   Original: \"{original_query}\"")
        
        if responses:
            output.append(f"   Clarifications: {len(responses)}")
            for q, a in responses.items():
                output.append(f"      • {a}")
            output.append(f"   Enhanced query: \"{augmented_query}\"")
        else:
            output.append("   Status: Proceeding without clarification")
        
        return '\n'.join(output)
