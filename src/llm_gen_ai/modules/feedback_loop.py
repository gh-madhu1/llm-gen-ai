"""
Human Feedback Loop Module
Collects and integrates human feedback for iterative answer refinement.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime


class FeedbackLoop:
    """Handles human feedback collection and integration for answer refinement."""
    
    def __init__(self, max_iterations: int = 3):
        self.max_iterations = max_iterations
        self.feedback_history = []
        self.current_iteration = 0
        
        # Feedback type keywords
        self.confirmation_keywords = ['correct', 'good', 'yes', 'perfect', 'right', 'accurate']
        self.rejection_keywords = ['wrong', 'incorrect', 'no', 'bad', 'false', 'inaccurate']
        self.enhancement_keywords = ['add', 'include', 'also', 'more', 'detail', 'explain']
        
    def present_draft_answer(
        self,
        answer: str,
        query: str,
        context: Dict[str, Any],
        iteration: int = 0
    ) -> None:
        """
        Present draft answer to user for feedback.
        
        Args:
            answer: Draft answer to present
            query: Original query
            context: Answer context
            iteration: Current iteration number
        """
        print("\n" + "="*60)
        if iteration == 0:
            print("📝 DRAFT ANSWER (awaiting your feedback)")
        else:
            print(f"📝 REFINED ANSWER - Iteration {iteration} (awaiting your feedback)")
        print("="*60)
        print(f"\n{answer}\n")
        
        # Show quality indicators if available
        if context.get('quality_score'):
            quality = context['quality_score']
            print(f"📊 Quality Score: {quality:.2f}/1.0")
            
            if quality < 0.6:
                print("⚠️  This answer may need improvement")
            elif quality >= 0.8:
                print("✅ This answer appears comprehensive")
        
        print("="*60)
    
    def collect_feedback(self, interactive: bool = True) -> Optional[Dict[str, Any]]:
        """
        Collect feedback from user on draft answer.
        
        Args:
            interactive: Whether to prompt interactively
            
        Returns:
            Feedback dictionary or None if skipped
        """
        if not interactive:
            return None
        
        print("\n🤔 Your Feedback:")
        print("  - Type 'yes' or 'correct' if the answer is good")
        print("  - Provide corrections or additional requirements")
        print("  - Type 'skip' to accept the current answer")
        print("-"*60)
        
        try:
            feedback_text = input("\nFeedback: ").strip()
            
            if not feedback_text or feedback_text.lower() == 'skip':
                print("⏭️  Proceeding with current answer...")
                return None
            
            # Parse feedback
            feedback = self._parse_feedback(feedback_text)
            feedback['raw_text'] = feedback_text
            feedback['timestamp'] = datetime.now().isoformat()
            
            # Store in history
            self.feedback_history.append(feedback)
            
            return feedback
            
        except (KeyboardInterrupt, EOFError):
            print("\n⏭️  Skipping feedback...")
            return None
    
    def _parse_feedback(self, feedback_text: str) -> Dict[str, Any]:
        """
        Parse feedback text to determine type and extract information.
        
        Args:
            feedback_text: Raw feedback text
            
        Returns:
            Parsed feedback dictionary
        """
        feedback_lower = feedback_text.lower()
        
        # Determine feedback type
        feedback_type = self._classify_feedback_type(feedback_lower)
        
        # Extract corrections and guidance
        corrections = []
        guidance = []
        
        if feedback_type in ['correction', 'enhancement']:
            # Extract specific corrections
            corrections = self._extract_corrections(feedback_text)
            
            # Extract guidance/requirements
            guidance = self._extract_guidance(feedback_text)
        
        return {
            'type': feedback_type,
            'corrections': corrections,
            'guidance': guidance,
            'sentiment': self._analyze_sentiment(feedback_lower)
        }
    
    def _classify_feedback_type(self, feedback_lower: str) -> str:
        """Classify the type of feedback."""
        # Check for confirmation
        if any(kw in feedback_lower for kw in self.confirmation_keywords):
            return 'confirmation'
        
        # Check for rejection
        if any(kw in feedback_lower for kw in self.rejection_keywords):
            # Distinguish between correction and complete rejection
            if len(feedback_lower.split()) > 3:
                return 'correction'  # Has additional info
            else:
                return 'rejection'
        
        # Check for enhancement
        if any(kw in feedback_lower for kw in self.enhancement_keywords):
            return 'enhancement'
        
        # Default to correction if specific feedback given
        if len(feedback_lower.split()) > 2:
            return 'correction'
        
        return 'unclear'
    
    def _extract_corrections(self, feedback_text: str) -> List[str]:
        """Extract specific corrections from feedback."""
        corrections = []
        
        # Look for correction patterns
        patterns = [
            "no,", "wrong,", "incorrect,", "actually,", "should be",
            "not about", "it's about", "i mean", "looking for"
        ]
        
        for pattern in patterns:
            if pattern in feedback_text.lower():
                # Extract the part after the pattern
                parts = feedback_text.lower().split(pattern)
                if len(parts) > 1:
                    correction = parts[1].strip()
                    if correction:
                        corrections.append(correction)
        
        # If no pattern matched, treat entire feedback as correction
        if not corrections and len(feedback_text.split()) > 3:
            corrections.append(feedback_text)
        
        return corrections
    
    def _extract_guidance(self, feedback_text: str) -> List[str]:
        """Extract guidance or requirements from feedback."""
        guidance = []
        
        # Look for guidance patterns
        patterns = [
            "add", "include", "explain", "provide", "show",
            "need", "want", "should have", "focus on"
        ]
        
        for pattern in patterns:
            if pattern in feedback_text.lower():
                # This is guidance for what to add/change
                guidance.append(feedback_text)
                break
        
        return guidance
    
    def _analyze_sentiment(self, feedback_lower: str) -> str:
        """Analyze sentiment of feedback."""
        if any(kw in feedback_lower for kw in self.confirmation_keywords):
            return 'positive'
        elif any(kw in feedback_lower for kw in self.rejection_keywords):
            return 'negative'
        else:
            return 'neutral'
    
    def integrate_feedback_into_context(
        self,
        feedback: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Integrate feedback into context for refinement.
        
        Args:
            feedback: Parsed feedback
            context: Existing context
            
        Returns:
            Updated context with feedback
        """
        context = context.copy()
        
        # Add feedback to context
        context['feedback'] = feedback
        
        # Add corrections to search queries if needed
        if feedback['corrections'] and feedback['type'] in ['correction', 'rejection']:
            # Create new search query based on correction
            correction_text = ' '.join(feedback['corrections'])
            context['correction_query'] = correction_text
            context['needs_new_search'] = True
        
        # Add guidance to refinement requirements
        if feedback['guidance']:
            context['refinement_guidance'] = feedback['guidance']
        
        return context
    
    def should_continue_iteration(self) -> bool:
        """Check if more iterations are allowed."""
        return self.current_iteration < self.max_iterations
    
    def increment_iteration(self) -> None:
        """Increment iteration counter."""
        self.current_iteration += 1
    
    def reset_iteration(self) -> None:
        """Reset iteration counter."""
        self.current_iteration = 0
    
    def format_feedback_summary(self, feedback: Dict[str, Any]) -> str:
        """
        Format feedback summary for display.
        
        Args:
            feedback: Feedback dictionary
            
        Returns:
            Formatted summary
        """
        output = []
        output.append("\n📋 Feedback Summary:")
        output.append(f"   Type: {feedback['type'].title()}")
        output.append(f"   Sentiment: {feedback['sentiment'].title()}")
        
        if feedback['corrections']:
            output.append(f"   Corrections: {len(feedback['corrections'])}")
            for corr in feedback['corrections'][:2]:
                output.append(f"      • {corr[:80]}...")
        
        if feedback['guidance']:
            output.append(f"   Guidance: {len(feedback['guidance'])}")
            for guide in feedback['guidance'][:2]:
                output.append(f"      • {guide[:80]}...")
        
        return '\n'.join(output)
    
    def get_feedback_history(self) -> List[Dict[str, Any]]:
        """Get all feedback history."""
        return self.feedback_history
    
    def save_feedback_history(self, filepath: str) -> None:
        """
        Save feedback history to file.
        
        Args:
            filepath: Path to save file
        """
        import json
        
        with open(filepath, 'w') as f:
            json.dump({
                'feedback_history': self.feedback_history,
                'total_iterations': self.current_iteration
            }, f, indent=2)
        
        print(f"💾 Feedback history saved to {filepath}")
    
    def create_refinement_instruction(self, feedback: Dict[str, Any], query: str) -> str:
        """
        Create instruction for refinement based on feedback.
        
        Args:
            feedback: Parsed feedback
            query: Original query
            
        Returns:
            Refinement instruction
        """
        instruction = f"Based on user feedback, please refine the answer to: '{query}'\n\n"
        
        if feedback['type'] == 'correction':
            instruction += "User corrections:\n"
            for corr in feedback['corrections']:
                instruction += f"- {corr}\n"
            instruction += "\nPlease correct the answer based on this feedback.\n"
        
        elif feedback['type'] == 'enhancement':
            instruction += "User requested enhancements:\n"
            for guide in feedback['guidance']:
                instruction += f"- {guide}\n"
            instruction += "\nPlease enhance the answer with these additions.\n"
        
        elif feedback['type'] == 'rejection':
            instruction += "The previous answer was incorrect. "
            if feedback['corrections']:
                instruction += f"User indicated: {feedback['corrections'][0]}\n"
            instruction += "Please provide a completely new answer addressing the correct topic.\n"
        
        return instruction
