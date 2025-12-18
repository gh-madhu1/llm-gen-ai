"""
Optimized White Paper Generation Agent.
Refactored to use modular components for better performance and maintainability.
"""
import re
import torch
from llm_gen_ai.core.memory_manager import ContextMemory
from llm_gen_ai.core.search_engine import SearchEngine
from llm_gen_ai.core.document_generator import DocumentGenerator
from llm_gen_ai.utils import truncate_text, track_process_time
from llm_gen_ai.core.model_loader import clear_device_cache
from llm_gen_ai.config import (
    DEVICE,
    MAX_STEPS,
    MINIMUM_SECTIONS,
    GENERATION_CONFIG,
    MAX_PROMPT_LENGTH_CHARS,
    MAX_PROMPT_LENGTH_FOR_REBUILD,
    MAX_MODEL_LENGTH,
    ENABLE_PROGRESS_LOGS,
    CLEAR_CACHE_FREQUENCY
)


class WhitePaperAgent:
    """Optimized agent for generating comprehensive white papers."""

    def __init__(self, model, tokenizer, output_dir="."):
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.max_steps = MAX_STEPS
        self.paper_content = {}
        self.history = []
        
        # Initialize modular components
        self.search_engine = SearchEngine()
        self.doc_generator = DocumentGenerator(output_dir)
        self.context_memory = ContextMemory()
        
        # Performance tracking
        self.cache_clear_count = 0

    @track_process_time
    def generate_text(self, prompt, max_new_tokens=None):
        """
        Generate text with optimized settings for speed.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate (uses config default if None)
        
        Returns:
            Generated text
        """
        if max_new_tokens is None:
            from config import DEFAULT_MAX_NEW_TOKENS
            max_new_tokens = DEFAULT_MAX_NEW_TOKENS
        
        # Truncate prompt if too long
        if len(prompt) > MAX_PROMPT_LENGTH_CHARS:
            if ENABLE_PROGRESS_LOGS:
                print(f"\n⚠️  Prompt too long ({len(prompt)} chars), truncating...")
            prompt = truncate_text(prompt, MAX_PROMPT_LENGTH_CHARS)

        # Tokenize with truncation
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_MODEL_LENGTH
        ).to(DEVICE)

        # Adaptive cache clearing
        if CLEAR_CACHE_FREQUENCY == "adaptive":
            # Clear only if we're getting large prompts
            if len(prompt) > MAX_PROMPT_LENGTH_CHARS * 0.8:
                clear_device_cache()
                self.cache_clear_count += 1
        elif CLEAR_CACHE_FREQUENCY == "always":
            clear_device_cache()
            self.cache_clear_count += 1

        # Generate with optimized settings (no beam search!)
        with torch.no_grad():
            generated = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                **GENERATION_CONFIG
            )

        # Decode response
        full_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        # Return only the new generated text
        response = full_text[len(prompt):].strip()
        
        return response

    def validate_idea_novelty(self, idea):
        """
        Validate idea novelty using search engine.
        
        Args:
            idea: The idea to validate
        
        Returns:
            Tuple of (is_novel, existing_work)
        """
        is_novel, existing_work = self.search_engine.validate_novelty(idea)
        return is_novel, existing_work

    def generate_comparative_analysis(self, existing_work):
        """
        Generate comparative analysis based on existing work.
        
        Args:
            existing_work: List of existing work dictionaries
        
        Returns:
            Formatted comparative analysis text
        """
        if not existing_work:
            return "No existing work found for comparison."

        analysis = "This paper distinguishes itself from existing research in several key ways:\n\n"

        for i, work in enumerate(existing_work[:3], 1):
            analysis += f"{i}. **{work['title']}**: While this work addresses related topics, "
            analysis += "our approach differs by focusing on novel aspects and providing unique insights.\n\n"

        analysis += "Our contribution provides a fresh perspective by combining theoretical frameworks with practical implementation strategies."
        return analysis

    @track_process_time
    def reason_and_act(self, idea):
        """
        Main agent loop with ReAct pattern.
        
        Args:
            idea: The white paper idea/topic
        
        Returns:
            Status message
        """
        # Step 1: Validate idea novelty
        is_novel, existing_work = self.validate_idea_novelty(idea)
        
        if not is_novel:
            return "Idea validation failed: Similar content already exists. Please refine your idea."

        # Step 2: Auto-generate comparative analysis if we have existing work
        if existing_work:
            comparative_analysis = self.generate_comparative_analysis(existing_work)
            self.paper_content['Related Work'] = comparative_analysis
            if ENABLE_PROGRESS_LOGS:
                print("\n📊 Generated comparative analysis section")

        # Step 3: Start white paper generation
        print(f"\n" + "="*60)
        print("🚀 Starting White Paper Generation")
        print("="*60)
        print(f"Topic: {idea[:100]}...\n")
        
        self.history = []
        system_prompt = self._build_system_prompt(idea)
        current_prompt = system_prompt
        last_action = None
        repeat_count = 0

        for step in range(self.max_steps):
            if ENABLE_PROGRESS_LOGS:
                print(f"\n--- Step {step + 1}/{self.max_steps} ---")
                print(f"📊 Progress: {len(self.paper_content)} sections | "
                      f"{self.context_memory.get_completion_percentage()}% complete")

            # Detect infinite loops
            if repeat_count >= 3:
                print("\n⚠️  Loop detected! Resetting and trying different approach...")
                repeat_count = 0
                last_action = None
                current_prompt += "\nOBSERVATION: You're repeating the same action. Write a DIFFERENT section.\n"
                continue

            # Smart prompt rebuilding using context memory
            if len(current_prompt) > MAX_PROMPT_LENGTH_FOR_REBUILD:
                if ENABLE_PROGRESS_LOGS:
                    print(f"🧠 Compressing context (prompt was {len(current_prompt)} chars)")
                context_summary = self.context_memory.get_context_summary()
                current_prompt = system_prompt + "\n\n" + context_summary + \
                    "\n\nContinue writing the remaining sections.\n"

            # Generate agent response
            response = self.generate_text(current_prompt, max_new_tokens=100)
            
            if ENABLE_PROGRESS_LOGS:
                print(f"Agent: {response[:200]}...")

            # Parse and execute action
            thought, action_line = self._parse_response(response)
            
            if thought:
                self.history.append(f"THOUGHT: {thought}")
            
            if action_line:
                self.history.append(f"ACTION: {action_line}")
                observation, should_finish = self._execute_action(
                    action_line, thought, last_action, repeat_count
                )
                
                # Update prompt with observation
                current_prompt += f"\nACTION: {action_line}\nOBSERVATION: {observation}\n"
                
                # Check if we should finish
                if should_finish:
                    return "White paper generation complete."
                
                # Track repeating actions
                if action_line == last_action:
                    repeat_count += 1
                else:
                    repeat_count = 0
                last_action = action_line
            else:
                current_prompt += "\nOBSERVATION: Please specify an ACTION.\n"

        if ENABLE_PROGRESS_LOGS:
            print(f"\n⚠️  Max steps reached. Cache cleared {self.cache_clear_count} times.")
        
        return "Max steps reached."

    def _build_system_prompt(self, idea):
        """Build the initial system prompt."""
        return f"""You are an AI Researcher writing a COMPREHENSIVE white paper on:

"{idea}"

Create a white paper with these sections:

# WRITE[title | Your Title]
    ## Abstract (150-250 words)
    ## Executive Summary
    ## Introduction
    ## Background
    ## Problem Statement
    ## Solution / Methodology
    ## Implementation
    ## Benefits
    ## Challenges & Risks
    ## Recommendations
    ## Conclusion
    ## References

TOOLS AVAILABLE:
1. PLAN: Create outline. Usage: PLAN[List sections to write]
2. SEARCH: Web search. Usage: SEARCH[query]
3. WRITE: Write section. Usage: WRITE[section_title | content]
4. SAVE_DOCX: Save file. Usage: SAVE_DOCX[filename]
5. FINISH: Complete. Usage: FINISH[summary]

WORKFLOW:
1. FIRST: Create a PLAN
2. THEN: Research with SEARCH
3. NEXT: Write sections with WRITE
4. FINALLY: Call FINISH when done

GUIDELINES:
- Research thoroughly before writing
- Write detailed content (200-300 words per section)
- Use professional tone
- Include examples and evidence
- Only call FINISH after ALL sections written

Format:
THOUGHT: [reasoning]
ACTION: [tool]

START with PLAN!
"""

    def _parse_response(self, response):
        """Parse agent response into thought and action."""
        thought_match = re.search(r"THOUGHT:(.*?)(?=ACTION:|$)", response, re.DOTALL)
        action_match = re.search(r"ACTION:(.*)", response, re.DOTALL)
        
        thought = thought_match.group(1).strip() if thought_match else None
        action_line = action_match.group(1).strip() if action_match else None
        
        return thought, action_line

    def _execute_action(self, action_line, thought, last_action, repeat_count):
        """
        Execute the parsed action.
        
        Returns:
            Tuple of (observation, should_finish)
        """
        should_finish = False
        
        if "PLAN[" in action_line:
            observation = self._handle_plan(action_line)
        elif "SEARCH[" in action_line:
            observation = self._handle_search(action_line)
        elif "WRITE[" in action_line:
            observation = self._handle_write(action_line, last_action, repeat_count)
        elif "SAVE_DOCX[" in action_line:
            observation = self._handle_save(action_line)
        elif "FINISH[" in action_line:
            observation, should_finish = self._handle_finish(action_line)
        else:
            observation = "Invalid action. Use PLAN, SEARCH, WRITE, SAVE_DOCX, or FINISH."
            print(f"❌ {observation}")
        
        return observation, should_finish

    def _handle_plan(self, action_line):
        """Handle PLAN action."""
        match = re.search(r'PLAN\[(.+?)\]', action_line, re.DOTALL)
        if match:
            plan = match.group(1).strip()
            self.context_memory.set_plan(plan)
            print(f"📋 Plan: {plan[:100]}...")
            return "Plan recorded. Now research and write each section."
        return "Invalid PLAN format. Use PLAN[outline]"

    def _handle_search(self, action_line):
        """Handle SEARCH action."""
        match = re.search(r'SEARCH\[(.+?)\]', action_line)
        if match:
            query = match.group(1)
            findings = self.search_engine.search(query)
            self.context_memory.add_research(query, findings)
            self.context_memory.add_action("SEARCH", query)
            # Return concise observation
            result_count = len(findings.split('\n'))
            return f"Found {result_count} sources."
        return "Invalid SEARCH format. Use SEARCH[query]"

    def _handle_write(self, action_line, last_action, repeat_count):
        """Handle WRITE action."""
        match = re.search(r'WRITE\[(.+?)\]', action_line, re.DOTALL)
        if match:
            content = match.group(1)
            if "|" in content:
                parts = content.split("|", 1)
                section_title = parts[0].strip()
                section_content = parts[1].strip() if len(parts) > 1 else ""

                # Check if already completed
                if self.context_memory.should_skip_section(section_title):
                    print(f"⏭️  Skipping: {section_title} (already done)")
                    return f"'{section_title}' already written. Write a different section."
                
                # Store section
                self.paper_content[section_title] = section_content
                self.context_memory.add_section(section_title)
                self.context_memory.add_action("WRITE", section_title)
                print(f"✍️  Wrote: {section_title} ({len(section_content)} chars)")
                return "Section written."
            
            return "Invalid WRITE format. Use WRITE[section_title | content]"
        return "Could not parse WRITE action."

    def _handle_save(self, action_line):
        """Handle SAVE_DOCX action."""
        match = re.search(r'SAVE_DOCX\[(.+?)\]', action_line)
        if match:
            filename = match.group(1).strip()
            try:
                file_path = self.doc_generator.create_document(
                    self.paper_content, filename
                )
                return f"Saved to {file_path}"
            except Exception as e:
                return f"Error saving: {str(e)}"
        return "Invalid SAVE_DOCX format. Use SAVE_DOCX[filename]"

    def _handle_finish(self, action_line):
        """Handle FINISH action."""
        print("Agent requested to finish.")
        
        # Check if enough content
        if len(self.paper_content) < MINIMUM_SECTIONS:
            observation = f"Only {len(self.paper_content)} sections. Need {MINIMUM_SECTIONS}. Continue writing."
            print(f"⚠️  {observation}")
            return observation, False
        
        # Auto-add references
        references = self.search_engine.generate_references()
        self.paper_content['References'] = references
        print(f"\n📚 Added references ({len(self.search_engine.get_citations())} citations)")
        
        # Save to docx
        if self.paper_content:
            self.doc_generator.create_document(self.paper_content)
        
        print(f"\n✅ Complete! {len(self.paper_content)} sections written!")
        print(f"🔍 Cache cleared {self.cache_clear_count} times")
        print(f"📦 Search cache: {self.search_engine.get_cache_size()} queries cached")
        
        return "White paper complete.", True


@track_process_time
def generate(model, tokenizer, prompt, output_dir="."):
    """
    Generate a white paper from a prompt.
    
    Args:
        model: Loaded language model
        tokenizer: Loaded tokenizer
        prompt: White paper idea/topic
        output_dir: Output directory for files
    
    Returns:
        Status message
    """
    agent = WhitePaperAgent(model, tokenizer, output_dir)
    return agent.reason_and_act(prompt)
