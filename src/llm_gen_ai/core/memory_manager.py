"""
Memory management module for context tracking and compression.
Prevents memory overflow and infinite loops.
"""
from llm_gen_ai.config import (
    MAX_ACTIONS_MEMORY,
    REQUIRED_SECTIONS,
    RESEARCH_SUMMARY_LENGTH
)


class ContextMemory:
    """Manages context to prevent memory overflow and maintain coherence."""

    def __init__(self, max_actions=MAX_ACTIONS_MEMORY):
        self.completed_sections = set()
        self.research_summary = {}
        self.recent_actions = []
        self.max_actions = max_actions
        self.plan = None  # Store the agent's plan
        self.action_counts = {}  # Track action frequency for loop detection

    def add_section(self, section_name):
        """Track completed section."""
        self.completed_sections.add(section_name.lower())

    def add_research(self, query, findings):
        """
        Store summarized research.
        
        Args:
            query: Search query
            findings: Research findings (will be truncated)
        """
        # Keep only key points to save memory
        self.research_summary[query] = findings[:RESEARCH_SUMMARY_LENGTH]

    def add_action(self, action_type, details):
        """
        Add action to sliding window and track frequency.
        
        Args:
            action_type: Type of action (SEARCH, WRITE, etc.)
            details: Action details
        """
        action_key = f"{action_type}:{details}"
        
        # Update action frequency for loop detection
        self.action_counts[action_key] = self.action_counts.get(action_key, 0) + 1
        
        self.recent_actions.append((action_type, details))
        
        # Keep only last N actions (sliding window)
        if len(self.recent_actions) > self.max_actions:
            removed_action = self.recent_actions.pop(0)
            # Clean up old action counts
            old_key = f"{removed_action[0]}:{removed_action[1]}"
            if old_key in self.action_counts:
                self.action_counts[old_key] -= 1
                if self.action_counts[old_key] <= 0:
                    del self.action_counts[old_key]

    def set_plan(self, plan):
        """Store the agent's plan."""
        self.plan = plan

    def get_context_summary(self):
        """
        Generate concise context summary for prompt.
        Optimized to be token-efficient.
        
        Returns:
            Formatted context summary string
        """
        summary = []

        # Show plan if exists (truncated)
        if self.plan:
            plan_preview = self.plan[:150] + "..." if len(self.plan) > 150 else self.plan
            summary.append(f"📋 PLAN: {plan_preview}")

        # Completed sections (concise)
        if self.completed_sections:
            summary.append(
                f"\n✅ Completed ({len(self.completed_sections)}): {', '.join(sorted(list(self.completed_sections)[:5]))}"
            )
            if len(self.completed_sections) > 5:
                summary.append(f" +{len(self.completed_sections) - 5} more")

        # Remaining sections (only show if less than 5 remaining)
        remaining = REQUIRED_SECTIONS - self.completed_sections
        if remaining and len(remaining) <= 5:
            summary.append(f"\n⏳ Remaining: {', '.join(sorted(remaining))}")

        # Research topics (show only count to save tokens)
        if self.research_summary:
            summary.append(f"\n📚 Researched: {len(self.research_summary)} topics")

        return "".join(summary)

    def should_skip_section(self, section_name):
        """
        Check if section already completed.
        
        Args:
            section_name: Name of the section
        
        Returns:
            True if section should be skipped
        """
        return section_name.lower() in self.completed_sections

    def is_action_repeating(self, action_type, details, threshold=3):
        """
        Check if an action is being repeated too frequently.
        
        Args:
            action_type: Type of action
            details: Action details
            threshold: Max repetitions before flagging as loop
        
        Returns:
            True if action is repeating excessively
        """
        action_key = f"{action_type}:{details}"
        return self.action_counts.get(action_key, 0) >= threshold

    def get_completion_percentage(self):
        """
        Calculate completion percentage based on required sections.
        
        Returns:
            Completion percentage (0-100)
        """
        if not REQUIRED_SECTIONS:
            return 0
        completed_required = len(self.completed_sections & REQUIRED_SECTIONS)
        return int((completed_required / len(REQUIRED_SECTIONS)) * 100)

    def reset(self):
        """Reset all context memory."""
        self.completed_sections.clear()
        self.research_summary.clear()
        self.recent_actions.clear()
        self.action_counts.clear()
        self.plan = None
