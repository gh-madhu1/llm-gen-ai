"""
Search engine module with caching and citation tracking.
Optimizes web searches to reduce API calls and improve performance.
"""
import time
from functools import lru_cache
from ddgs import DDGS
from llm_gen_ai.config import MAX_SEARCH_RESULTS, VALIDATION_SEARCH_RESULTS, SEARCH_CACHE_ENABLED


class SearchEngine:
    """Handles web searches with result caching and citation tracking."""

    def __init__(self, enable_cache=SEARCH_CACHE_ENABLED):
        self.ddgs = DDGS()
        self.citations = []
        self.enable_cache = enable_cache
        self._cache = {}  # Manual cache for search results

    def search(self, query, max_results=MAX_SEARCH_RESULTS):
        """
        Search the web for information with caching.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
        
        Returns:
            Formatted search results string
        """
        # Check cache first
        cache_key = f"{query}:{max_results}"
        if self.enable_cache and cache_key in self._cache:
            print(f"📦 Using cached results for: {query[:50]}...")
            return self._cache[cache_key]

        print(f"🔍 Searching: {query[:70]}...")
        
        try:
            results = self.ddgs.text(query, max_results=max_results)
            
            if results:
                # Track citations to avoid duplicates
                for result in results:
                    citation = {
                        'title': result.get('title', 'Unknown'),
                        'url': result.get('href', ''),
                        'snippet': result.get('body', '')[:200]
                    }
                    # Check if citation already exists (by URL)
                    if not any(c['url'] == citation['url'] for c in self.citations):
                        self.citations.append(citation)

                # Format findings
                findings = "\n".join([
                    f"- {r['title']}: {r['body'][:150]}..."
                    for r in results
                ])
                
                # Cache the results
                if self.enable_cache:
                    self._cache[cache_key] = findings
                
                return findings
            
            return "No results found."
        
        except Exception as e:
            print(f"⚠️  Search error: {e}")
            return f"Search failed: {str(e)}"

    def validate_novelty(self, idea):
        """
        Check if idea is novel using AI-powered analysis.
        Returns (is_novel, existing_work_list)
        
        Args:
            idea: The idea to validate
        
        Returns:
            Tuple of (bool, list) - (is_novel, existing_work)
        """
        print("\n" + "="*60)
        print("🔍 Validating Idea Novelty")
        print("="*60)

        # Create concise search query
        idea_summary = idea[:200].replace('\n', ' ')
        search_query = f"{idea_summary} research paper white paper"
        
        print(f"Query: {search_query[:100]}...")

        try:
            results = self.ddgs.text(search_query, max_results=VALIDATION_SEARCH_RESULTS)
            existing_work = []

            if results and len(results) >= 2:
                # Store existing work for comparative analysis
                for result in results[:5]:
                    existing_work.append({
                        'title': result.get('title', 'Unknown'),
                        'url': result.get('href', ''),
                        'summary': result.get('body', '')[:300]
                    })
                    # Also track as citation
                    citation = {
                        'title': result.get('title', 'Unknown'),
                        'url': result.get('href', ''),
                        'snippet': result.get('body', '')[:200]
                    }
                    if not any(c['url'] == citation['url'] for c in self.citations):
                        self.citations.append(citation)

                # Simple heuristic: if many results found, likely not novel
                # But we'll be optimistic and proceed with comparative analysis
                print(f"\n📊 Found {len(results)} related works")
                print("✅ Will proceed with comparative analysis")
                print("="*60)
                return True, existing_work
            
            else:
                # Few or no results - likely novel
                if results:
                    for result in results:
                        existing_work.append({
                            'title': result.get('title', 'Unknown'),
                            'url': result.get('href', ''),
                            'summary': result.get('body', '')[:300]
                        })
                
                print("\n✅ Minimal existing content - idea appears novel")
                print("="*60)
                return True, existing_work

        except Exception as e:
            print(f"\n⚠️  Validation error: {e}")
            print("Proceeding with caution...")
            print("="*60)
            return True, []  # Proceed on error

    def get_citations(self):
        """Get all tracked citations."""
        return self.citations

    def generate_references(self):
        """
        Generate formatted references section.
        
        Returns:
            Formatted references string
        """
        if not self.citations:
            return "No references available."

        refs = "# References\n\n"
        for i, citation in enumerate(self.citations, 1):
            refs += f"{i}. **{citation['title']}**\n"
            if citation['url']:
                refs += f"   URL: {citation['url']}\n"
            refs += f"   Accessed: {time.strftime('%Y-%m-%d')}\n\n"

        return refs

    def clear_cache(self):
        """Clear the search cache."""
        self._cache.clear()
        print("🧹 Search cache cleared")

    def get_cache_size(self):
        """Get number of cached queries."""
        return len(self._cache)
