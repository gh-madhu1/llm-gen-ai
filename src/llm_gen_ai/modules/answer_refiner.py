"""
Answer Refiner Module
Analyzes and refines answers through reflection for improved quality.
"""

from typing import Dict, List, Optional, Any


class AnswerRefiner:
    """Refines answers through quality assessment and reflection."""
    
    def __init__(self):
        # Quality thresholds
        self.min_answer_length = 50  # characters
        self.ideal_answer_length = 200
        
        # Key quality indicators
        self.quality_indicators = {
            'has_examples': ['example', 'for instance', 'such as', 'like'],
            'has_explanation': ['because', 'therefore', 'thus', 'since', 'due to'],
            'has_structure': ['first', 'second', 'finally', 'additionally', 'however'],
            'has_specifics': ['specifically', 'particularly', 'namely', 'including']
        }
    
    def assess_answer_quality(
        self, 
        answer: str, 
        query: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Assess the quality of an answer.
        
        Args:
            answer: Generated answer
            query: Original query
            context: Additional context (analysis, search results, etc.)
            
        Returns:
            Quality assessment dictionary
        """
        answer_lower = answer.lower()
        query_lower = query.lower()
        
        assessment = {
            'length_score': self._score_length(answer),
            'completeness_score': self._score_completeness(answer, query, context),
            'specificity_score': self._score_specificity(answer, context),
            'structure_score': self._score_structure(answer),
            'relevance_score': self._score_relevance(answer, query),
            'quality_indicators': self._check_quality_indicators(answer_lower),
            'overall_score': 0.0
        }
        
        # Calculate overall score (weighted average)
        assessment['overall_score'] = (
            assessment['length_score'] * 0.15 +
            assessment['completeness_score'] * 0.30 +
            assessment['specificity_score'] * 0.25 +
            assessment['structure_score'] * 0.15 +
            assessment['relevance_score'] * 0.15
        )
        
        return assessment
    
    def _score_length(self, answer: str) -> float:
        """Score answer length (0.0-1.0)."""
        length = len(answer)
        
        if length < self.min_answer_length:
            return length / self.min_answer_length
        elif length >= self.ideal_answer_length:
            return 1.0
        else:
            # Linear interpolation between min and ideal
            return 0.5 + 0.5 * (length - self.min_answer_length) / (self.ideal_answer_length - self.min_answer_length)
    
    def _score_completeness(self, answer: str, query: str, context: Dict) -> float:
        """Score how completely the answer addresses the query."""
        score = 0.5  # Base score
        answer_lower = answer.lower()
        query_lower = query.lower()
        
        # Check if query keywords are addressed
        analysis = context.get('analysis', {})
        keywords = analysis.get('keywords', [])
        
        if keywords:
            addressed_keywords = sum(1 for kw in keywords if kw in answer_lower)
            score += 0.3 * (addressed_keywords / len(keywords))
        
        # Check if entities are mentioned
        entities = analysis.get('entities', [])
        if entities:
            mentioned_entities = sum(1 for entity in entities if entity.lower() in answer_lower)
            score += 0.2 * (mentioned_entities / len(entities))
        
        return min(score, 1.0)
    
    def _score_specificity(self, answer: str, context: Dict) -> float:
        """Score how specific and detailed the answer is."""
        score = 0.3  # Base score
        answer_lower = answer.lower()
        
        # Check for specific details
        has_numbers = bool(any(char.isdigit() for char in answer))
        if has_numbers:
            score += 0.2
        
        # Check for quality indicators
        for indicator_type, keywords in self.quality_indicators.items():
            if any(kw in answer_lower for kw in keywords):
                score += 0.15
        
        # Check if search results were used (indicates recent, specific info)
        if context.get('search_results') and context['search_results'].get('count', 0) > 0:
            score += 0.2
        
        return min(score, 1.0)
    
    def _score_structure(self, answer: str) -> float:
        """Score the structural quality of the answer."""
        score = 0.3
        answer_lower = answer.lower()
        
        # Check for structural indicators
        structure_keywords = self.quality_indicators['has_structure']
        structure_count = sum(1 for kw in structure_keywords if kw in answer_lower)
        score += min(structure_count * 0.15, 0.4)
        
        # Check for multiple sentences
        sentence_count = answer.count('.') + answer.count('!') + answer.count('?')
        if sentence_count >= 3:
            score += 0.3
        elif sentence_count >= 2:
            score += 0.15
        
        return min(score, 1.0)
    
    def _score_relevance(self, answer: str, query: str) -> float:
        """Score how relevant the answer is to the query."""
        score = 0.4  # Base score
        answer_lower = answer.lower()
        query_lower = query.lower()
        
        # Check if answer starts appropriately
        if query_lower.startswith('what is') or query_lower.startswith('what are'):
            # Should provide definition
            if 'is' in answer_lower[:100] or 'are' in answer_lower[:100]:
                score += 0.2
        elif query_lower.startswith('how'):
            # Should provide explanation or steps
            if any(kw in answer_lower for kw in ['first', 'step', 'process', 'by']):
                score += 0.2
        elif query_lower.startswith('why'):
            # Should provide reasoning
            if any(kw in answer_lower for kw in ['because', 'due to', 'reason', 'since']):
                score += 0.2
        
        # Check for direct query word overlap
        query_words = set(query_lower.split())
        answer_words = set(answer_lower.split())
        overlap = len(query_words & answer_words) / max(len(query_words), 1)
        score += min(overlap * 0.4, 0.4)
        
        return min(score, 1.0)
    
    def _check_quality_indicators(self, answer_lower: str) -> Dict[str, bool]:
        """Check which quality indicators are present."""
        return {
            indicator: any(kw in answer_lower for kw in keywords)
            for indicator, keywords in self.quality_indicators.items()
        }
    
    def identify_gaps(
        self,
        answer: str,
        query: str,
        context: Dict[str, Any]
    ) -> List[str]:
        """
        Identify gaps or missing information in the answer.
        
        Args:
            answer: Generated answer
            query: Original query
            context: Context including analysis, search results, clarifications
            
        Returns:
            List of identified gaps
        """
        gaps = []
        assessment = self.assess_answer_quality(answer, query, context)
        answer_lower = answer.lower()
        query_lower = query.lower()
        
        # Gap 1: Too short
        if assessment['length_score'] < 0.5:
            gaps.append("Answer is too brief and lacks detail")
        
        # Gap 2: Missing query keywords
        analysis = context.get('analysis', {})
        keywords = analysis.get('keywords', [])
        if keywords:
            missing_keywords = [kw for kw in keywords if kw not in answer_lower]
            if len(missing_keywords) >= len(keywords) / 2:
                gaps.append(f"Key concepts not addressed: {', '.join(missing_keywords[:3])}")
        
        # Gap 3: Missing examples for "how" questions
        if query_lower.startswith('how') and 'example' not in answer_lower:
            gaps.append("Missing concrete examples or illustrations")
        
        # Gap 4: Missing sources for time-sensitive queries
        if analysis.get('time_sensitive') and context.get('search_results'):
            search_results = context['search_results']
            if search_results.get('count', 0) > 0:
                # Check if answer mentions recent info
                if not any(term in answer_lower for term in ['2024', '2025', 'recent', 'latest', 'current']):
                    gaps.append("Doesn't incorporate latest information from search results")
        
        # Gap 5: Missing comparison elements
        if 'vs' in query_lower or 'compare' in query_lower:
            if not any(term in answer_lower for term in ['whereas', 'while', 'compared to', 'unlike', 'similar']):
                gaps.append("Missing comparative analysis")
        
        # Gap 6: Missing context from clarifications
        if context.get('clarifications'):
            # Check if clarification context is used
            clarifications = context['clarifications']
            clarification_text = ' '.join(clarifications.values()).lower()
            # Extract key terms from clarifications
            clarification_words = set(clarification_text.split())
            answer_words = set(answer_lower.split())
            overlap = len(clarification_words & answer_words) / max(len(clarification_words), 1)
            
            if overlap < 0.3:
                gaps.append("Clarification context not fully incorporated")
        
        # Gap 7: Lack of structure
        if assessment['structure_score'] < 0.4 and len(answer) > 100:
            gaps.append("Answer lacks clear structure or organization")
        
        return gaps
    
    def refine_answer(
        self,
        original_answer: str,
        gaps: List[str],
        context: Dict[str, Any],
        model_generate_func: Optional[callable] = None
    ) -> str:
        """
        Refine the answer to address identified gaps.
        
        Args:
            original_answer: Original generated answer
            gaps: Identified gaps
            context: Full context
            model_generate_func: Optional function to regenerate with better prompt
            
        Returns:
            Refined answer
        """
        if not gaps:
            return original_answer
        
        # If no model function provided, do basic refinement
        if not model_generate_func:
            return self._basic_refinement(original_answer, gaps, context)
        
        # Create refinement prompt
        refinement_prompt = self._create_refinement_prompt(
            original_answer, gaps, context
        )
        
        # Generate refined answer
        try:
            refined = model_generate_func(refinement_prompt)
            return refined
        except Exception:
            # Fallback to basic refinement
            return self._basic_refinement(original_answer, gaps, context)
    
    def _basic_refinement(
        self, 
        original_answer: str, 
        gaps: List[str], 
        context: Dict[str, Any]
    ) -> str:
        """Basic refinement without model regeneration."""
        refined = original_answer
        
        # Add context from search results if missing recent info
        if any('latest information' in gap for gap in gaps):
            search_results = context.get('search_results', {})
            if search_results.get('results'):
                # Append note about search results
                refined += f"\n\nNote: Based on recent search results, this information is current as of {search_results.get('search_date', 'today')}."
        
        # Add clarification context if missing
        if any('Clarification context' in gap for gap in gaps):
            clarifications = context.get('clarifications', {})
            if clarifications:
                context_note = ' '.join(clarifications.values())
                refined += f"\n\nConsidering your clarification: {context_note}"
        
        return refined
    
    def _create_refinement_prompt(
        self,
        original_answer: str,
        gaps: List[str],
        context: Dict[str, Any]
    ) -> str:
        """Create prompt for answer refinement."""
        query = context.get('query', '')
        
        prompt = f"""Given this question: "{query}"

Initial answer:
{original_answer}

This answer has the following gaps:
{chr(10).join(f"- {gap}" for gap in gaps)}

Please provide an improved, more complete answer that addresses these gaps."""
        
        # Add search results context if available
        search_results = context.get('search_results', {})
        if search_results and search_results.get('sources'):
            prompt += "\n\nRecent information from web search:\n"
            for i, source in enumerate(search_results['sources'][:3], 1):
                prompt += f"{i}. {source.get('title', 'Source')}: {source.get('url', '')}\n"
        
        # Add clarification context
        clarifications = context.get('clarifications')
        if clarifications:
            prompt += f"\n\nAdditional context from user: {' '.join(clarifications.values())}"
        
        prompt += "\n\nProvide a comprehensive, well-structured answer:"
        
        return prompt
    
    def calculate_refined_confidence(
        self,
        original_assessment: Dict[str, Any],
        refined_assessment: Dict[str, Any],
        context: Dict[str, Any]
    ) -> float:
        """
        Calculate confidence score for refined answer.
        
        Args:
            original_assessment: Quality assessment of original answer
            refined_assessment: Quality assessment of refined answer
            context: Full context
            
        Returns:
            Confidence score (0.0-1.0)
        """
        base_confidence = refined_assessment['overall_score']
        
        # Boost confidence if improvement was made
        improvement = refined_assessment['overall_score'] - original_assessment['overall_score']
        if improvement > 0.1:
            base_confidence += 0.1
        
        # Boost if search results were used
        if context.get('search_results') and context['search_results'].get('count', 0) > 0:
            base_confidence += 0.05
        
        # Boost if clarifications were provided
        if context.get('clarifications'):
            base_confidence += 0.05
        
        return min(base_confidence, 1.0)
    
    def format_refinement_summary(
        self,
        gaps: List[str],
        original_score: float,
        refined_score: float,
        confidence: float
    ) -> str:
        """
        Format refinement summary for display.
        
        Args:
            gaps: Identified gaps
            original_score: Original quality score
            refined_score: Refined quality score
            confidence: Final confidence score
            
        Returns:
            Formatted summary
        """
        output = []
        output.append("\n🔍 Answer Refinement:")
        
        if gaps:
            output.append(f"   Gaps identified: {len(gaps)}")
            for gap in gaps[:3]:  # Show top 3
                output.append(f"      • {gap}")
            
            output.append(f"   Quality improvement: {original_score:.2f} → {refined_score:.2f}")
        else:
            output.append("   Status: No significant gaps found")
        
        output.append(f"   Final confidence: {confidence:.2f}")
        
        return '\n'.join(output)
    
    def refine_with_feedback(
        self,
        original_answer: str,
        feedback: Dict[str, Any],
        context: Dict[str, Any],
        model_generate_func: Optional[callable] = None
    ) -> str:
        """
        Refine answer based on human feedback.
        
        Args:
            original_answer: Original generated answer
            feedback: Parsed feedback from user
            context: Full context
            model_generate_func: Function to regenerate answer
            
        Returns:
            Refined answer
        """
        feedback_type = feedback.get('type', 'unclear')
        
        # If confirmed, return original
        if feedback_type == 'confirmation':
            return original_answer
        
        # For rejection or correction, create new answer
        if feedback_type in ['rejection', 'correction']:
            return self._refine_with_correction(
                original_answer, feedback, context, model_generate_func
            )
        
        # For enhancement, augment existing answer
        if feedback_type == 'enhancement':
            return self._refine_with_enhancement(
                original_answer, feedback, context, model_generate_func
            )
        
        # Unclear feedback - try best effort refinement
        return self._refine_with_unclear_feedback(
            original_answer, feedback, context, model_generate_func
        )
    
    def _refine_with_correction(
        self,
        original_answer: str,
        feedback: Dict[str, Any],
        context: Dict[str, Any],
        model_generate_func: Optional[callable]
    ) -> str:
        """Refine answer with user corrections."""
        if not model_generate_func:
            return original_answer
        
        query = context.get('query', '')
        corrections = feedback.get('corrections', [])
        
        # Create correction-focused prompt
        prompt = f"""The user asked: "{query}"

Previous answer was incorrect:
{original_answer}

User corrections:
{chr(10).join(f'- {corr}' for corr in corrections)}

Please provide a new, correct answer that addresses the user's corrections."""
        
        # Add search context if correction indicates new search needed
        if context.get('needs_new_search') and context.get('correction_query'):
            prompt += f"\n\nNote: Focus on: {context['correction_query']}"
        
        # Add clarification context
        if context.get('clarifications'):
            clarifications = context['clarifications']
            prompt += f"\n\nOriginal clarifications: {' '.join(clarifications.values())}"
        
        prompt += "\n\nProvide the corrected answer:"
        
        try:
            refined = model_generate_func(prompt)
            return refined
        except Exception:
            return original_answer
    
    def _refine_with_enhancement(
        self,
        original_answer: str,
        feedback: Dict[str, Any],
        context: Dict[str, Any],
        model_generate_func: Optional[callable]
    ) -> str:
        """Enhance answer based on user guidance."""
        if not model_generate_func:
            # Basic enhancement
            guidance = feedback.get('guidance', [])
            if guidance:
                enhanced = original_answer + "\n\nAdditional information:\n"
                enhanced += "\n".join(f"- {g}" for g in guidance)
                return enhanced
            return original_answer
        
        query = context.get('query', '')
        guidance = feedback.get('guidance', [])
        
        # Create enhancement prompt
        prompt = f"""Given this question: "{query}"

Current answer:
{original_answer}

User requests these enhancements:
{chr(10).join(f'- {g}' for g in guidance)}

Please enhance the answer to include the requested information while keeping the existing good parts."""
        
        # Add search results if available
        search_results = context.get('search_results', {})
        if search_results and search_results.get('sources'):
            prompt += "\n\nYou can use these search results:\n"
            for i, source in enumerate(search_results['sources'][:3], 1):
                prompt += f"{i}. {source.get('title', 'Source')}: {source.get('url', '')}\n"
        
        prompt += "\n\nProvide the enhanced answer:"
        
        try:
            refined = model_generate_func(prompt)
            return refined
        except Exception:
            return original_answer
    
    def _refine_with_unclear_feedback(
        self,
        original_answer: str,
        feedback: Dict[str, Any],
        context: Dict[str, Any],
        model_generate_func: Optional[callable]
    ) -> str:
        """Handle unclear feedback with best effort."""
        # Try to extract any useful information
        raw_text = feedback.get('raw_text', '')
        
        if not raw_text or not model_generate_func:
            return original_answer
        
        query = context.get('query', '')
        
        prompt = f"""Given this question: "{query}"

Current answer:
{original_answer}

User feedback (unclear): {raw_text}

Please try to improve the answer based on this feedback."""
        
        try:
            refined = model_generate_func(prompt)
            return refined
        except Exception:
            return original_answer
    
    def apply_corrections(
        self,
        answer: str,
        corrections: List[str]
    ) -> str:
        """Apply specific corrections to answer (basic text replacement)."""
        corrected = answer
        
        # Simple correction application
        for correction in corrections:
            # This is a placeholder for more sophisticated correction logic
            # In practice, you'd need more context to apply corrections accurately
            pass
        
        return corrected
    
    def incorporate_guidance(
        self,
        context: Dict[str, Any],
        guidance: List[str]
    ) -> Dict[str, Any]:
        """Incorporate user guidance into context."""
        context = context.copy()
        
        if 'user_guidance' not in context:
            context['user_guidance'] = []
        
        context['user_guidance'].extend(guidance)
        
        return context
