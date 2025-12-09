"""
Parallel Search Module
Executes web searches in parallel for faster results.
"""

from typing import List, Dict, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import re
from llm_gen_ai.core.search_engine import SearchEngine


class ParallelSearchEngine:
    """Executes multiple searches in parallel for comprehensive results."""
    
    def __init__(self):
        self.search_engine = SearchEngine()
        self.max_workers = 3
        self.time_window_months = 6  # Default time window for time-sensitive queries
        
    def search_parallel(self, queries: List[str], max_results: int = 3) -> Dict:
        """
        Search multiple queries in parallel.
        
        Args:
            queries: List of search queries
            max_results: Results per query
            
        Returns:
            Combined search results
        """
        if not queries:
            return {'results': [], 'sources': []}
        
        # Use ThreadPoolExecutor for parallel execution
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self._search_single, query, max_results)
                for query in queries
            ]
            
            # Collect results
            all_results = []
            for future in futures:
                try:
                    result = future.result(timeout=10)  # 10 second timeout
                    if result:
                        all_results.append(result)
                except Exception as e:
                    print(f"⚠️  Search error: {e}")
                    continue
        
        # Combine and deduplicate
        return self._combine_results(all_results)
    
    def _search_single(self, query: str, max_results: int) -> str:
        """Execute a single search."""
        try:
            return self.search_engine.search(query, max_results)
        except Exception as e:
            print(f"⚠️  Search failed for '{query}': {e}")
            return None
    
    def _combine_results(self, results: List[str]) -> Dict:
        """
        Combine multiple search results.
        
        Args:
            results: List of search result strings
            
        Returns:
            Combined and deduplicated results
        """
        if not results:
            return {'results': '', 'sources': []}
        
        # Simple combination - join all results
        combined_text = '\n\n'.join([r for r in results if r])
        
        # Get unique citations
        citations = self.search_engine.get_citations()
        
        return {
            'results': combined_text,
            'sources': citations,
            'count': len(citations)
        }
    
    def search_with_rewrites(self, original_query: str, rewritten_query: str, 
                           max_results: int = 3, time_sensitive: bool = False) -> Dict:
        """
        Search using both original and rewritten queries.
        
        Args:
            original_query: User's original query
            rewritten_query: Rewritten/optimized query
            max_results: Results per query
            time_sensitive: Apply time-based filtering
            
        Returns:
            Combined search results with freshness metadata
        """
        queries = []
        
        # Add time filter to queries if time-sensitive
        if time_sensitive:
            original_query = self._add_time_filter(original_query)
            rewritten_query = self._add_time_filter(rewritten_query)
        
        # Add original if different from rewritten
        if original_query.lower().strip() != rewritten_query.lower().strip():
            queries.append(original_query)
        
        # Always add rewritten
        queries.append(rewritten_query)
        
        print(f"🌐 Searching {len(queries)} query variation(s)...")
        if time_sensitive:
            print(f"⏰ Filtering for results from last {self.time_window_months} months")
        
        results = self.search_parallel(queries, max_results)
        
        # Add freshness scoring if time-sensitive
        if time_sensitive and results.get('sources'):
            results['sources'] = self._score_freshness(results['sources'])
        
        results['search_date'] = datetime.now().strftime('%Y-%m-%d')
        
        return results
    
    def get_citations(self) -> List[Dict]:
        """Get all citations from searches."""
        return self.search_engine.get_citations()
    
    def _add_time_filter(self, query: str) -> str:
        """
        Add time-based filter to search query.
        
        Args:
            query: Original search query
            
        Returns:
            Query with time filter
        """
        # Calculate date threshold
        cutoff_date = datetime.now() - timedelta(days=30 * self.time_window_months)
        year = cutoff_date.year
        
        # Add year to query if not already present
        if str(year) not in query and str(year + 1) not in query:
            query = f"{query} {datetime.now().year}"
        
        return query
    
    def _score_freshness(self, sources: List[Dict]) -> List[Dict]:
        """
        Score and sort sources by freshness.
        
        Args:
            sources: List of source dictionaries
            
        Returns:
            Sources sorted by freshness score
        """
        current_year = datetime.now().year
        
        for source in sources:
            freshness_score = 0.5  # Base score
            
            # Extract date from title or snippet if available
            text = f"{source.get('title', '')} {source.get('snippet', '')}"
            
            # Look for year mentions
            years = re.findall(r'\b(20\d{2})\b', text)
            if years:
                latest_year = max(int(y) for y in years)
                # Score based on recency
                years_old = current_year - latest_year
                if years_old == 0:
                    freshness_score = 1.0
                elif years_old == 1:
                    freshness_score = 0.8
                elif years_old == 2:
                    freshness_score = 0.6
                else:
                    freshness_score = 0.4
            
            # Look for recent time indicators
            recent_indicators = ['today', 'yesterday', 'this week', 'this month', 'latest', 'current']
            if any(indicator in text.lower() for indicator in recent_indicators):
                freshness_score = min(freshness_score + 0.2, 1.0)
            
            source['freshness_score'] = freshness_score
        
        # Sort by freshness score (descending)
        return sorted(sources, key=lambda x: x.get('freshness_score', 0), reverse=True)
    
    def format_sources(self, sources: List[Dict]) -> str:
        """
        Format citations for display.
        
        Args:
            sources: List of citation dictionaries
            
        Returns:
            Formatted sources string
        """
        if not sources:
            return ""
        
        output = ["\n📚 Sources:"]
        for i, source in enumerate(sources[:5], 1):  # Show top 5
            title = source.get('title', 'Unknown')
            freshness = source.get('freshness_score')
            
            # Add freshness indicator for recent sources
            if freshness and freshness >= 0.8:
                title = f"🆕 {title}"
            
            output.append(f"{i}. {title}")
            if source.get('url'):
                output.append(f"   {source['url']}")
        
        return '\n'.join(output)
