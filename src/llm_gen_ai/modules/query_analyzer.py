"""
Query Analyzer Module
Analyzes, rewrites, and optimizes user queries for better understanding and search.
"""

import re
from typing import Dict, List, Tuple


class QueryAnalyzer:
    """Analyzes and rewrites queries for better accuracy."""
    
    def __init__(self):
        self.current_year = 2024
        
        # Keywords for different categories
        self.time_keywords = ['latest', 'recent', 'current', 'today', 'now', 'this year', '2024', '2025']
        self.comparison_keywords = ['vs', 'versus', 'difference', 'compare', 'better']
        self.how_keywords = ['how', 'why', 'what', 'when', 'where', 'who']
        
    def analyze_query(self, query: str) -> Dict:
        """
        Analyze query and return structured information.
        
        Args:
            query: User's original question
            
        Returns:
            Dict with analysis results
        """
        query_lower = query.lower()
        
        analysis = {
            'original': query,
            'rewritten': self.rewrite_query(query),
            'needs_search': self._needs_web_search(query_lower),
            'query_type': self._classify_query_type(query_lower),
            'keywords': self._extract_keywords(query),
            'entities': self._extract_entities(query),
            'time_sensitive': self._is_time_sensitive(query_lower),
            'complexity': self._assess_complexity(query_lower),
            'confidence': self._estimate_confidence(query_lower),
            'needs_clarification': self._needs_clarification(query, query_lower),
            'ambiguity_score': self._calculate_ambiguity_score(query, query_lower)
        }
        
        return analysis
    
    def rewrite_query(self, query: str) -> str:
        """
        Rewrite query for better clarity and search results.
        
        Args:
            query: Original query
            
        Returns:
            Rewritten query
        """
        query_lower = query.lower()
        rewritten = query
        
        # Expand ambiguous terms
        if 'ai' in query_lower and len(query_lower.split()) <= 4:
            rewritten = query.replace('AI', 'artificial intelligence').replace('ai', 'artificial intelligence')
        
        if 'ml' in query_lower and 'html' not in query_lower:
            rewritten = rewritten.replace('ML', 'machine learning').replace('ml', 'machine learning')
        
        # Add year context for "latest" queries
        if any(kw in query_lower for kw in self.time_keywords):
            if str(self.current_year) not in query_lower:
                rewritten += f" {self.current_year}"
        
        # Make comparison queries more explicit
        if any(kw in query_lower for kw in self.comparison_keywords):
            if 'difference between' not in query_lower:
                rewritten = f"difference between {rewritten}"
        
        # Add context for vague queries
        if query_lower.strip() in ['how does it work?', 'how does it work']:
            rewritten = query  # Keep as is - will ask for clarification
        elif query_lower.startswith('how does ') and query_lower.count(' ') <= 3:
            rewritten += " - detailed explanation"
        
        return rewritten.strip()
    
    def _needs_web_search(self, query_lower: str) -> bool:
        """Determine if query needs web search."""
        # Time-sensitive queries need search
        if any(kw in query_lower for kw in self.time_keywords):
            return True
        
        # Price, stock, weather need real-time data
        if any(kw in query_lower for kw in ['price', 'stock', 'weather', 'score', 'result']):
            return True
        
        # News and events
        if any(kw in query_lower for kw in ['news', 'event', 'happened', 'breaking']):
            return True
        
        return False
    
    def _classify_query_type(self, query_lower: str) -> str:
        """Classify type of query."""
        if any(query_lower.startswith(kw) for kw in ['what is', 'what are', 'define']):
            return 'definition'
        elif any(query_lower.startswith(kw) for kw in ['how to', 'how do', 'how does']):
            return 'how-to'
        elif any(query_lower.startswith(kw) for kw in ['why', 'why is', 'why does']):
            return 'explanation'
        elif any(kw in query_lower for kw in self.comparison_keywords):
            return 'comparison'
        elif any(kw in query_lower for kw in self.time_keywords):
            return 'current-events'
        elif '?' in query_lower:
            return 'question'
        else:
            return 'general'
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query."""
        # Remove common stop words
        stop_words = {'the', 'is', 'are', 'what', 'how', 'why', 'when', 'where', 'who', 
                     'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'about'}
        
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        return keywords[:5]  # Top 5 keywords
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract potential entities (capitalized words, acronyms)."""
        # Find capitalized words (potential entities)
        entities = re.findall(r'\b[A-Z][A-Z]+\b|\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
        
        # Find common acronyms
        acronyms = re.findall(r'\b[A-Z]{2,}\b', query)
        
        return list(set(entities + acronyms))
    
    def _is_time_sensitive(self, query_lower: str) -> bool:
        """Check if query is time-sensitive."""
        return any(kw in query_lower for kw in self.time_keywords)
    
    def _assess_complexity(self, query_lower: str) -> str:
        """Assess query complexity."""
        word_count = len(query_lower.split())
        
        if word_count <= 3:
            return 'simple'
        elif word_count <= 10:
            return 'moderate'
        else:
            return 'complex'
    
    def _estimate_confidence(self, query_lower: str) -> str:
        """Estimate how confident we can be in answering without search."""
        # Low confidence if time-sensitive
        if self._is_time_sensitive(query_lower):
            return 'low'
        
        # Low confidence for current events
        if any(kw in query_lower for kw in ['price', 'stock', 'news', 'weather']):
            return 'low'
        
        # High confidence for definitions and how-to
        if any(query_lower.startswith(kw) for kw in ['what is', 'define', 'how to', 'explain']):
            return 'high'
        
        return 'medium'
    
    def _needs_clarification(self, query: str, query_lower: str) -> bool:
        """Determine if query needs clarification."""
        words = query_lower.split()
        
        # Very short queries often need clarification
        if len(words) <= 3:
            vague_pronouns = ['it', 'that', 'this', 'those', 'them']
            if any(pronoun in words for pronoun in vague_pronouns):
                return True
        
        # Queries with broad terms and no specifics
        broad_terms = ['thing', 'stuff', 'one', 'latest', 'best']
        if any(term in words for term in broad_terms) and len(words) <= 5:
            return True
        
        return False
    
    def _calculate_ambiguity_score(self, query: str, query_lower: str) -> float:
        """Calculate ambiguity score (0.0 = clear, 1.0 = very ambiguous)."""
        score = 0.0
        words = query_lower.split()
        
        # Factor 1: Query length
        if len(words) <= 3:
            score += 0.3
        elif len(words) <= 5:
            score += 0.15
        
        # Factor 2: Vague pronouns
        vague_pronouns = ['it', 'that', 'this', 'those', 'them']
        if any(pronoun in words for pronoun in vague_pronouns):
            score += 0.25
        
        # Factor 3: Broad terms
        broad_terms = ['thing', 'stuff', 'one', 'best', 'good']
        if any(term in words for term in broad_terms):
            score += 0.2
        
        # Factor 4: Missing context for comparisons
        if ('vs' in query_lower or 'versus' in query_lower) and len(words) <= 4:
            score += 0.25
        
        return min(score, 1.0)
    
    def format_analysis(self, analysis: Dict) -> str:
        """
        Format analysis results for display to user.
        
        Args:
            analysis: Analysis dictionary
            
        Returns:
            Formatted string for display
        """
        output = []
        output.append("📋 Query Analysis:")
        output.append(f"   Original: \"{analysis['original']}\"")
        
        if analysis['original'] != analysis['rewritten']:
            output.append(f"   Rewritten: \"{analysis['rewritten']}\"")
        
        output.append(f"   Type: {analysis['query_type'].replace('-', ' ').title()}")
        output.append(f"   Time-sensitive: {'Yes' if analysis['time_sensitive'] else 'No'}")
        output.append(f"   Needs search: {'Yes' if analysis['needs_search'] else 'No'}")
        
        if analysis['keywords']:
            output.append(f"   Key terms: {', '.join(analysis['keywords'])}")
        
        if analysis['entities']:
            output.append(f"   Entities: {', '.join(analysis['entities'])}")
        
        output.append(f"   Confidence: {analysis['confidence'].title()}")
        
        return '\n'.join(output)
