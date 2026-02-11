"""
DOVA Interactive CLI Session.

Provides a Claude Code-like continuous interaction experience with:
- Chain-of-thought reasoning for sophisticated responses
- Short-term and long-term memory integration
- Multi-turn conversation with context preservation
- Tool/action execution with result feedback
"""

import asyncio
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import structlog

from dova.agents.orchestrator import EVALUATIVE_PATTERNS

logger = structlog.get_logger(__name__)


class ActionType(Enum):
    """Types of actions the agent can take."""
    RESEARCH = "research"
    DEBATE = "debate"
    VALIDATE = "validate"
    SYNTHESIZE = "synthesize"
    REMEMBER = "remember"
    RECALL = "recall"
    THINK = "think"
    RESPOND = "respond"


@dataclass
class ThoughtStep:
    """A single step in the chain-of-thought reasoning."""
    step_type: str  # observation, reasoning, plan, action, reflection
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""
    role: str  # user, assistant, system
    content: str
    timestamp: float = field(default_factory=time.time)
    thought_chain: list[ThoughtStep] = field(default_factory=list)
    action_taken: str | None = None
    action_result: dict[str, Any] | None = None


@dataclass
class SessionState:
    """State for an interactive session."""
    session_id: str
    user_id: str
    started_at: float
    conversation: list[ConversationTurn] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    memory_refs: list[str] = field(default_factory=list)  # IDs of relevant memories
    # Enhanced context tracking
    last_topic: str = ""  # The main topic being discussed
    pending_suggestions: list[str] = field(default_factory=list)  # Suggestions offered to user
    last_question_to_user: str = ""  # Last question DOVA asked
    entities_discussed: dict[str, Any] = field(default_factory=dict)  # Papers, repos, etc.


class InteractiveSession:
    """
    Interactive CLI session with chain-of-thought reasoning.

    Provides Claude Code-like experience with:
    - Continuous multi-turn interaction
    - Memory-augmented responses
    - Transparent reasoning process
    - Action execution and feedback

    Supports two orchestration modes:
    - "standard": Task-graph based orchestration (default)
    - "thinking": Deliberation-first orchestration
    """

    def __init__(
        self,
        user_id: str = "interactive-user",
        show_thinking: bool = True,
        verbose: bool = False,
        orchestrator_type: str = "standard",
    ):
        self.user_id = user_id
        self.show_thinking = show_thinking
        self.verbose = verbose
        self.orchestrator_type = orchestrator_type

        # Session state
        self.state: SessionState | None = None

        # Components (initialized lazily)
        self._llm_router = None
        self._mcp_client = None
        self._memory_service = None
        self._research_agent = None
        self._debate_agent = None
        self._thinking_orchestrator = None
        self._settings = None

    @property
    def llm_router(self):
        """Lazy-load LLM router."""
        if self._llm_router is None:
            from dova.config.providers import create_llm_router_from_settings
            self._llm_router = create_llm_router_from_settings()
        return self._llm_router

    @property
    def settings(self):
        """Lazy-load settings."""
        if self._settings is None:
            from dova.config.settings import get_settings
            self._settings = get_settings()
        return self._settings

    @property
    def mcp_client(self):
        """Lazy-load MCP client."""
        if self._mcp_client is None:
            from dova.tools.mcp_registry import MCPClient
            self._mcp_client = MCPClient()
        return self._mcp_client

    @property
    def memory_service(self):
        """Lazy-load enhanced memory service."""
        if self._memory_service is None:
            from dova.services.memory_enhanced import EnhancedMemoryService
            from dova.utils.cache import InMemoryCache

            cache = InMemoryCache()
            self._memory_service = EnhancedMemoryService(
                cache=cache,
                llm_router=self.llm_router,
            )
        return self._memory_service

    @property
    def research_agent(self):
        """Lazy-load research agent."""
        if self._research_agent is None:
            from dova.agents.research import ResearchAgent
            self._research_agent = ResearchAgent(
                llm_router=self.llm_router,
                mcp_client=self.mcp_client,
                tavily_api_key=self.settings.mcp.tavily_api_key,
                enhanced_memory_service=self.memory_service,
            )
        return self._research_agent

    @property
    def debate_agent(self):
        """Lazy-load debate agent."""
        if self._debate_agent is None:
            from dova.agents.debate import DebateAgent
            self._debate_agent = DebateAgent(
                llm_router=self.llm_router,
                mcp_client=self.mcp_client,
                num_rounds=2,
            )
        return self._debate_agent

    @property
    def thinking_orchestrator(self):
        """Lazy-load thinking orchestrator."""
        if self._thinking_orchestrator is None:
            from dova.agents.thinking_orchestrator import ThinkingOrchestrator
            self._thinking_orchestrator = ThinkingOrchestrator(
                llm_router=self.llm_router,
                mcp_client=self.mcp_client,
                memory_service=self.memory_service,
            )
            # Register available agents
            self._thinking_orchestrator.register_agent("research", self.research_agent)
            self._thinking_orchestrator.register_agent("debate", self.debate_agent)
        return self._thinking_orchestrator

    def _is_evaluative_query(self, query: str) -> bool:
        """Check if query requires evaluative/debate analysis."""
        # Check if auto-debate is disabled
        if hasattr(self, "_auto_debate") and not self._auto_debate:
            return False
        query_lower = query.lower()
        return any(re.search(p, query_lower) for p in EVALUATIVE_PATTERNS)

    def start_session(self) -> str:
        """Start a new interactive session."""
        session_id = str(uuid.uuid4())[:8]
        self.state = SessionState(
            session_id=session_id,
            user_id=self.user_id,
            started_at=time.time(),
        )
        logger.info(
            "interactive_session_started",
            session_id=session_id,
            user_id=self.user_id,
        )
        return session_id

    async def process_input(self, user_input: str) -> str:
        """
        Process user input with chain-of-thought reasoning.

        Steps:
        0. Expand: Handle short follow-ups with context
        1. Observe: Understand the input and context
        2. Recall: Retrieve relevant memories
        3. Reason: Chain-of-thought analysis
        4. Plan: Determine best action
        5. Act: Execute action if needed
        6. Reflect: Evaluate result and learn
        7. Respond: Generate final response

        When orchestrator_type is "thinking", delegates to ThinkingOrchestrator
        for deliberation-first processing.
        """
        if not self.state:
            self.start_session()

        # Use ThinkingOrchestrator if configured
        if self.orchestrator_type == "thinking":
            return await self._process_with_thinking_orchestrator(user_input)

        start_time = time.time()
        thought_chain: list[ThoughtStep] = []

        # Store original input
        original_input = user_input

        # Step 0: Expand short follow-up responses
        if self._is_short_followup(user_input):
            user_input = await self._expand_followup(user_input)
            logger.debug("followup_expanded", original=original_input, expanded=user_input)
            if self.show_thinking:
                self._print_thought("Expansion", f"'{original_input}' -> '{user_input[:100]}'")

        # Add user turn to conversation (store original)
        user_turn = ConversationTurn(role="user", content=original_input)
        self.state.conversation.append(user_turn)

        try:
            # Step 1: Observe - Understand input and classify intent
            observation = await self._observe_direct(user_input, original_input)
            thought_chain.append(ThoughtStep("observation", observation))
            if self.show_thinking:
                self._print_thought("Observation", observation)

            # Step 2: Recall - Retrieve relevant memories
            memories = await self._recall(user_input)
            if memories:
                memory_summary = f"Found {len(memories)} relevant memories"
                thought_chain.append(ThoughtStep("recall", memory_summary))
                if self.show_thinking:
                    self._print_thought("Memory", memory_summary)

            # Step 3: Reason - Chain-of-thought analysis
            reasoning = await self._reason(user_input, observation, memories)
            thought_chain.append(ThoughtStep("reasoning", reasoning))
            if self.show_thinking:
                self._print_thought("Reasoning", reasoning)

            # Step 4: Plan - Determine action
            action_plan = await self._plan(user_input, reasoning)
            thought_chain.append(ThoughtStep("plan", json.dumps(action_plan)))
            if self.show_thinking:
                self._print_thought("Plan", f"Action: {action_plan.get('action', 'respond')}")

            # Step 5: Act - Execute action
            action_result = None
            if action_plan.get("action") != "respond":
                action_result = await self._act(action_plan)
                thought_chain.append(ThoughtStep(
                    "action",
                    f"Executed {action_plan['action']}: {action_result.get('status', 'unknown')}"
                ))
                if self.show_thinking:
                    self._print_thought("Action", f"Completed: {action_plan['action']}")

            # Step 6: Reflect - Evaluate and learn
            reflection = await self._reflect(user_input, action_plan, action_result)
            thought_chain.append(ThoughtStep("reflection", reflection))

            # Step 7: Respond - Generate final response
            response = await self._respond(
                user_input, reasoning, action_plan, action_result, memories
            )

            # Store in memory for future reference
            await self._remember(user_input, response, action_result)

            # Create assistant turn
            assistant_turn = ConversationTurn(
                role="assistant",
                content=response,
                thought_chain=thought_chain,
                action_taken=action_plan.get("action"),
                action_result=action_result,
            )
            self.state.conversation.append(assistant_turn)

            # Update context
            self.state.context["last_query"] = user_input
            self.state.context["last_action"] = action_plan.get("action")
            self.state.context["turn_count"] = len(self.state.conversation) // 2

            # Extract and track topic
            self._update_topic_tracking(user_input, action_result, response)

            # Extract entities from results for follow-up queries
            self._extract_entities(action_result)

            # Track any suggestions/questions offered to user
            self._track_pending_suggestions(response)

            execution_time = time.time() - start_time
            if self.verbose:
                self._print_thought("Time", f"{execution_time:.2f}s")

            return response

        except Exception as e:
            logger.exception("interactive_process_error", error=str(e))
            return f"I encountered an error: {str(e)}. Please try rephrasing your request."

    async def _process_with_thinking_orchestrator(self, user_input: str) -> str:
        """
        Process input using the ThinkingOrchestrator (deliberation-first).

        The ThinkingOrchestrator reasons about user needs before deciding
        whether to use tools, providing more intelligent source selection.
        """
        from dova.agents.base import AgentTask

        start_time = time.time()

        # Store original for conversation tracking
        original_input = user_input

        # Add user turn to conversation
        user_turn = ConversationTurn(role="user", content=original_input)
        self.state.conversation.append(user_turn)

        try:
            # Build task for orchestrator
            task = AgentTask(
                type="query",
                params={
                    "query": user_input,
                    "session_id": self.state.session_id,
                    "sources": getattr(self, "_research_sources", ["arxiv", "github", "huggingface", "web"]),
                },
                user_id=self.user_id,
            )

            # Execute with ThinkingOrchestrator
            result = await self.thinking_orchestrator.execute(task)

            if not result.success:
                return f"I encountered an error: {result.error}"

            data = result.data
            response = data.get("response", "")
            deliberation = data.get("deliberation", {})
            action_result = data.get("action_result")

            # Show deliberation reasoning if thinking is enabled
            if self.show_thinking:
                self._print_thought("Understanding", deliberation.get("reasoning", ""))
                tools_used = deliberation.get("tools_used", [])
                if tools_used:
                    self._print_thought("Tools", f"Used: {', '.join(tools_used)}")
                else:
                    self._print_thought("Tools", "Answered from context (no tools needed)")

            # Update conversation with assistant response
            thought_chain = [
                ThoughtStep("deliberation", deliberation.get("reasoning", "")),
            ]
            if deliberation.get("tools_used"):
                thought_chain.append(ThoughtStep("action", f"Tools: {deliberation['tools_used']}"))

            assistant_turn = ConversationTurn(
                role="assistant",
                content=response,
                thought_chain=thought_chain,
                action_taken=deliberation.get("action"),
                action_result=action_result,
            )
            self.state.conversation.append(assistant_turn)

            # Update context tracking
            self.state.context["last_query"] = user_input
            self.state.context["last_action"] = deliberation.get("action")
            self.state.context["turn_count"] = len(self.state.conversation) // 2

            # Track topic and entities
            self._update_topic_tracking(user_input, action_result, response)
            if action_result:
                self._extract_entities(action_result)
            self._track_pending_suggestions(response)

            execution_time = time.time() - start_time
            if self.verbose:
                self._print_thought("Time", f"{execution_time:.2f}s")

            return response

        except Exception as e:
            logger.exception("thinking_orchestrator_error", error=str(e))
            return f"I encountered an error: {str(e)}. Please try rephrasing your request."

    async def _llm_complete(
        self,
        prompt: str,
        task_type: Any,
        temperature: float = 0.5,
        max_tokens: int = 5000,
    ) -> str:
        """Helper to call LLM with proper request format."""
        from dova.config.providers import LLMRequest

        request = LLMRequest(
            task_type=task_type,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        response = await self.llm_router.complete(request)
        return response.content

    def _is_short_followup(self, user_input: str) -> bool:
        """Check if input is a short follow-up response."""
        followup_patterns = [
            "yes", "yeah", "yep", "sure", "ok", "okay", "please", "go ahead",
            "do it", "sounds good", "that works", "perfect", "great", "thanks",
            "no", "nope", "not really", "actually", "wait", "but",
        ]
        normalized = user_input.lower().strip().rstrip("!.,?")
        return (
            len(normalized.split()) <= 4 and
            (normalized in followup_patterns or
             any(normalized.startswith(p) for p in followup_patterns))
        )

    async def _expand_followup(self, user_input: str) -> str:
        """Expand short follow-up into full context-aware query."""
        from dova.config.providers import TaskType

        normalized = user_input.lower().strip()
        is_affirmative = any(w in normalized for w in ["yes", "yeah", "ok", "sure", "please", "go"])
        is_negative = any(w in normalized for w in ["no", "nope", "not"])

        # Get the last assistant message with suggestions
        last_assistant_content = ""
        last_user_query = ""
        for turn in reversed(self.state.conversation[:-1]):  # Exclude current input
            if turn.role == "assistant" and not last_assistant_content:
                last_assistant_content = turn.content
            elif turn.role == "user" and not last_user_query:
                last_user_query = turn.content
            if last_assistant_content and last_user_query:
                break

        # Use pending suggestions if available
        if self.state.pending_suggestions and is_affirmative:
            suggestion = self.state.pending_suggestions[0]
            return f"Please proceed with: {suggestion}"

        # Use last question asked if there was one
        if self.state.last_question_to_user and is_affirmative:
            return f"Yes, {self.state.last_question_to_user}"

        # For negative responses, ask for clarification based on context
        if is_negative:
            return f"The user said '{user_input}' in response to the previous discussion about {self.state.last_topic or 'the topic'}. What would they like instead?"

        # Ask LLM to expand based on context
        prompt = f"""The user gave a short follow-up response. Expand it into a clear request.

Previous user query: {last_user_query[:300]}
Previous assistant response (excerpt): {last_assistant_content[:500]}
Topic being discussed: {self.state.last_topic or 'unknown'}

User's follow-up: "{user_input}"

What is the user asking for? Expand into a specific request:"""

        expanded = await self._llm_complete(
            prompt=prompt,
            task_type=TaskType.CLASSIFICATION,
            temperature=0.3,
            max_tokens=1500,
        )
        return expanded.strip()

    def _get_conversation_context(self, num_turns: int = 6) -> str:
        """Get formatted conversation context."""
        if not self.state or len(self.state.conversation) < 2:
            return ""

        context_lines = []
        recent = self.state.conversation[-(num_turns * 2):]
        for turn in recent:
            role = "USER" if turn.role == "user" else "ASSISTANT"
            content = turn.content
            # Keep more content for better context
            if len(content) > 500:
                content = content[:500] + "..."
            context_lines.append(f"{role}: {content}")
        return "\n".join(context_lines)

    async def _observe_direct(self, user_input: str, original_input: str = "") -> str:
        """Observe and understand the user input (expansion already done)."""
        from dova.config.providers import TaskType

        if not original_input:
            original_input = user_input

        # Build richer context from conversation history
        history_context = self._get_conversation_context(num_turns=4)

        # Include topic context with better entity formatting
        topic_context = ""
        if self.state.last_topic:
            topic_context = f"Current discussion topic: {self.state.last_topic}"
        if self.state.entities_discussed:
            entity_summaries = []
            for k, v in list(self.state.entities_discussed.items())[:5]:
                if isinstance(v, dict):
                    name = v.get('title') or v.get('name') or v.get('id') or k
                    entity_summaries.append(f"{v.get('type', 'item')}: {name}")
                else:
                    entity_summaries.append(f"{k}: {v}")
            topic_context += f"\nEntities in context: {', '.join(entity_summaries)}"

        prompt = f"""Analyze this user input and provide a brief observation.

User Input: "{user_input}"
{f"(Expanded from: '{original_input}')" if user_input != original_input else ""}

{f"Conversation History:{chr(10)}{history_context}" if history_context else ""}
{topic_context}

Provide a 2-3 sentence observation covering:
1. What the user is asking for (be specific)
2. How this relates to the previous conversation (if applicable)
3. The type of request (research, follow-up question, clarification, etc.)

Observation:"""

        response = await self._llm_complete(
            prompt=prompt,
            task_type=TaskType.CLASSIFICATION,
            temperature=0.3,
            max_tokens=2000,
        )
        return response.strip()

    async def _recall(self, user_input: str) -> list[dict[str, Any]]:
        """Retrieve relevant memories."""
        try:
            from dova.services.memory_enhanced import MemoryType

            # Also search using topic context if available
            search_queries = [user_input]
            if self.state.last_topic and self.state.last_topic not in user_input.lower():
                search_queries.append(f"{user_input} {self.state.last_topic}")

            all_memories = []
            seen_ids = set()

            for query in search_queries:
                results = await self.memory_service.search_semantic(
                    query=query,
                    user_id=self.user_id,
                    top_k=5,
                    use_mmr=True,
                    memory_types=[MemoryType.LONG_TERM, MemoryType.SHORT_TERM],
                )

                for mem in results:
                    # Handle both dict and dataclass/object results
                    if hasattr(mem, '__dict__'):
                        mem_dict = {k: v for k, v in mem.__dict__.items() if not k.startswith('_')}
                    elif isinstance(mem, dict):
                        mem_dict = mem
                    else:
                        mem_dict = {"content": str(mem)}

                    # Deduplicate
                    mem_id = mem_dict.get("id") or mem_dict.get("entry_id") or str(hash(str(mem_dict.get("content", ""))))
                    if mem_id in seen_ids:
                        continue
                    seen_ids.add(mem_id)

                    # Extract and normalize content
                    content = mem_dict.get("content", {})
                    if isinstance(content, dict):
                        # Flatten nested content
                        mem_dict["text"] = content.get("text") or content.get("response") or content.get("query") or json.dumps(content)[:200]
                    else:
                        mem_dict["text"] = str(content)[:200]

                    all_memories.append(mem_dict)

                    # Track memory references
                    if mem_id and isinstance(mem_id, str):
                        self.state.memory_refs.append(mem_id)

            return all_memories[:7]  # Return top memories
        except Exception as e:
            logger.warning("memory_recall_failed", error=str(e))
            return []

    async def _reason(
        self,
        user_input: str,
        observation: str,
        memories: list[dict[str, Any]],
    ) -> str:
        """Chain-of-thought reasoning about the request."""
        from dova.config.providers import TaskType

        # Format memories with full content extraction
        memory_context = ""
        if memories:
            memory_lines = []
            for m in memories[:5]:
                content = m.get('content', '')
                # Handle nested content dict
                if isinstance(content, dict):
                    text = content.get('text') or content.get('query') or content.get('response') or str(content)
                else:
                    text = str(content)
                if text:
                    memory_lines.append(f"- {text[:300]}")
            memory_context = "\n".join(memory_lines)

        # Get full conversation context
        conv_context = self._get_conversation_context(num_turns=4)

        # Include entities and topic
        topic_info = ""
        if self.state.last_topic:
            topic_info = f"Main topic: {self.state.last_topic}"
        if self.state.entities_discussed:
            topic_info += f"\nKey entities: {json.dumps(self.state.entities_discussed, default=str)[:200]}"

        prompt = f"""Think step-by-step about how to best address this request.

User Request: "{user_input}"
Observation: {observation}

{f"Relevant Memories:{chr(10)}{memory_context}" if memory_context else ""}

{f"Conversation History:{chr(10)}{conv_context}" if conv_context else ""}

{topic_info}

Chain of Thought:
1. What is the core need?
2. How does this relate to the conversation so far?
3. What information from context should be used?
4. What action is needed (if any)?

Reasoning:"""

        response = await self._llm_complete(
            prompt=prompt,
            task_type=TaskType.REASONING,
            temperature=0.4,
            max_tokens=4000,
        )
        return response.strip()

    async def _plan(self, user_input: str, reasoning: str) -> dict[str, Any]:
        """Plan the best action to take."""
        from dova.config.providers import TaskType

        # Check if collaborative or deep mode forces debate
        reasoning_mode = getattr(self, "_reasoning_mode", "standard")
        if reasoning_mode in ("collaborative", "deep"):
            return {
                "action": "debate",
                "params": {
                    "topic": user_input,
                    "query": user_input,
                    "context": {
                        "last_topic": self.state.last_topic,
                        "entities": list(self.state.entities_discussed.keys())[:5],
                    },
                    "reasoning_mode": reasoning_mode,
                    "enable_two_pass": getattr(self, "_enable_two_pass", True),
                },
                "rationale": f"Using {reasoning_mode} mode with debate analysis",
            }

        # Auto-detect evaluative queries that should use debate
        if self._is_evaluative_query(user_input):
            return {
                "action": "debate",
                "params": {
                    "topic": user_input,
                    "query": user_input,
                    "context": {
                        "last_topic": self.state.last_topic,
                        "entities": list(self.state.entities_discussed.keys())[:5],
                    },
                },
                "rationale": "Auto-detected evaluative query requiring debate analysis",
            }

        # Build context summary
        context_summary = ""
        if self.state.last_topic:
            context_summary += f"Current topic: {self.state.last_topic}\n"
        if self.state.entities_discussed:
            context_summary += f"Entities discussed: {json.dumps(self.state.entities_discussed, default=str)[:300]}\n"

        # Check if this is a follow-up about something already discussed
        is_followup = len(self.state.conversation) > 2

        from datetime import datetime
        current_date = datetime.now().strftime("%B %d, %Y")
        current_year = datetime.now().year

        prompt = f"""Based on this analysis, determine the best action.

User Input: "{user_input}"
Reasoning: {reasoning}
Current Date: {current_date}

{context_summary}

Available Actions:
- research: Search for papers, repos, models (for NEW topics or when needing fresh data)
- debate: Run Bull vs Bear analysis (for evaluation/decision queries)
- synthesize: Combine and summarize previous information (for multi-turn synthesis)
- image: Generate images/pictures/illustrations/artwork (for creative image creation requests)
- respond: Direct response using existing context (for follow-up questions about already-discussed topics)

IMPORTANT:
- If the user is asking a follow-up question about something already discussed (like "who are the authors" after discussing a paper), use "respond" with existing context
- Only use "research" when NEW information is needed
- If information exists in the entities_discussed, use "respond"
- For recent/news queries, if adding years to search query, use CURRENT year ({current_year}) and recent past - NEVER outdated years
- For IMAGE GENERATION requests (create, generate, draw, make an image/picture/illustration/artwork), ALWAYS use "image" action - NOT research

{"This appears to be a follow-up question. Check if the answer is in the context before doing new research." if is_followup else ""}

Respond in JSON:
{{
    "action": "<research|debate|synthesize|image|respond>",
    "params": {{
        "query": "<search query if research>",
        "topic": "<debate topic if debate>",
        "prompt": "<image description if image action>",
        "sources": ["arxiv", "github", "huggingface", "web"],
        "use_context": true/false
    }},
    "rationale": "<brief explanation>"
}}

JSON:"""

        response = await self._llm_complete(
            prompt=prompt,
            task_type=TaskType.REASONING,
            temperature=0.2,
            max_tokens=3000,
        )

        # Parse JSON response
        try:
            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
            return {"action": "respond", "params": {}, "rationale": "Default response"}
        except json.JSONDecodeError:
            return {"action": "respond", "params": {}, "rationale": "Parse error"}

    async def _act(self, action_plan: dict[str, Any]) -> dict[str, Any]:
        """Execute the planned action."""
        action = action_plan.get("action", "respond")
        params = action_plan.get("params", {})

        try:
            if action == "research":
                return await self._action_research(params)
            elif action == "debate":
                return await self._action_debate(params)
            elif action == "synthesize":
                return await self._action_synthesize(params)
            elif action == "image":
                return await self._action_image(params)
            else:
                return {"status": "skipped", "reason": "No action required"}
        except Exception as e:
            logger.exception("action_execution_error", action=action, error=str(e))
            return {"status": "error", "error": str(e)}

    async def _action_research(self, params: dict[str, Any]) -> dict[str, Any]:
        """Execute research action."""
        from dova.agents.base import AgentTask

        query = params.get("query", "")
        sources = params.get("sources", ["arxiv", "github", "huggingface"])

        task = AgentTask(
            type="research",
            params={
                "query": query,
                "sources": sources,
                "max_results": 10,
            },
            user_id=self.user_id,
        )

        result = await self.research_agent.execute(task)

        if result.success:
            # Handle both dict and dataclass results
            data = result.data
            if hasattr(data, '__dict__'):
                # It's a dataclass - convert to dict-like access
                papers = getattr(data, 'papers', []) or []
                repos = getattr(data, 'repositories', []) or []
                models = getattr(data, 'models', []) or []
                web_results = getattr(data, 'web_results', []) or []
                summary = getattr(data, 'summary', '') or ''
                answer = getattr(data, 'answer', '') or ''
            else:
                # It's already a dict
                papers = data.get("papers", []) or []
                repos = data.get("repositories", []) or []
                models = data.get("models", []) or []
                web_results = data.get("web_results", []) or []
                summary = data.get("summary", "") or ""
                answer = data.get("answer", "") or ""

            # Convert dataclass items to dicts for JSON serialization
            def to_dict(item):
                if hasattr(item, '__dict__'):
                    d = {k: v for k, v in item.__dict__.items() if not k.startswith('_')}
                    # Ensure authors are properly extracted
                    if 'authors' in d and isinstance(d['authors'], (list, tuple)):
                        d['authors'] = [str(a) for a in d['authors']]
                    return d
                return item

            paper_dicts = [to_dict(p) for p in papers[:5]]
            repo_dicts = [to_dict(r) for r in repos[:5]]
            model_dicts = [to_dict(m) for m in models[:5]]
            web_dicts = [to_dict(w) for w in web_results[:5]]

            # Also extract images from the result (for image generation)
            if hasattr(data, '__dict__'):
                images = getattr(data, 'images', []) or []
            else:
                images = data.get("images", []) or []
            image_dicts = [to_dict(img) for img in images]

            return {
                "status": "success",
                "papers": paper_dicts,
                "repositories": repo_dicts,
                "models": model_dicts,
                "web_results": web_dicts,
                "images": image_dicts,
                "summary": summary,
                "answer": answer,
            }
        else:
            return {"status": "error", "error": result.error}

    async def _action_debate(self, params: dict[str, Any]) -> dict[str, Any]:
        """Execute debate action."""
        from dova.agents.base import AgentTask

        topic = params.get("topic", params.get("query", ""))
        context = params.get("context", {})

        # In collaborative/deep mode, run research first so the debate
        # has real data instead of relying on LLM training knowledge.
        reasoning_mode = params.get("reasoning_mode", "standard")
        research_data = {}
        if reasoning_mode in ("collaborative", "deep"):
            sources = getattr(self, "_research_sources", None) or ["arxiv", "github", "huggingface", "web"]
            research_result = await self._action_research({
                "query": topic,
                "sources": sources,
            })
            if research_result.get("status") == "success":
                research_data = research_result
                # Provide a summary of findings to the debate agent
                context["research_summary"] = research_result.get("summary", "")
                context["research_answer"] = research_result.get("answer", "")
                context["papers_found"] = len(research_result.get("papers", []))
                context["repos_found"] = len(research_result.get("repositories", []))
                context["models_found"] = len(research_result.get("models", []))
                context["web_results_found"] = len(research_result.get("web_results", []))

        task = AgentTask(
            type="debate",
            params={
                "topic": topic,
                "context": context,
            },
            user_id=self.user_id,
        )

        result = await self.debate_agent.execute(task)

        if result.success:
            debate_result = {
                "status": "success",
                "summary": result.data.get("summary", ""),
                "bull_strengths": result.data.get("bull_strengths", []),
                "bear_concerns": result.data.get("bear_concerns", []),
                "recommendation": result.data.get("recommendation", ""),
                "confidence": result.data.get("confidence_score", 0),
            }
            # Merge research data so the frontend can display papers/repos/models
            if research_data:
                debate_result["papers"] = research_data.get("papers", [])
                debate_result["repositories"] = research_data.get("repositories", [])
                debate_result["models"] = research_data.get("models", [])
                debate_result["web_results"] = research_data.get("web_results", [])
                debate_result["answer"] = research_data.get("answer", "")
            return debate_result
        else:
            return {"status": "error", "error": result.error}

    async def _action_synthesize(self, params: dict[str, Any]) -> dict[str, Any]:
        """Execute synthesis action using conversation context."""
        from dova.config.providers import TaskType

        # Gather context from conversation
        context_items = []
        for turn in self.state.conversation[-6:]:
            if turn.action_result and turn.action_result.get("status") == "success":
                context_items.append(turn.action_result)

        if not context_items:
            return {"status": "skipped", "reason": "No context to synthesize"}

        prompt = f"""Synthesize the following information:

{json.dumps(context_items, indent=2, default=str)[:2000]}

Provide:
1. Key findings (3-5 points)
2. Connections and patterns
3. Recommendations

Synthesis:"""

        response = await self._llm_complete(
            prompt=prompt,
            task_type=TaskType.SUMMARIZATION,
            temperature=0.4,
            max_tokens=5000,
        )

        return {"status": "success", "synthesis": response}

    async def _action_image(self, params: dict[str, Any]) -> dict[str, Any]:
        """Execute image generation action using HuggingFace Z-Image-Turbo."""
        from dova.config.providers import TaskType

        prompt = params.get("prompt", params.get("query", ""))
        if not prompt:
            return {"status": "error", "error": "No prompt provided for image generation"}

        logger.info("image_generation_starting", prompt=prompt[:100])

        # Step 1: Enhance the prompt for better image generation
        enhancement_prompt = f"""Rephrase this image request into an optimal prompt for AI image generation.

User request: {prompt}

Create a detailed, descriptive prompt that:
1. Describes the main subject clearly
2. Includes style descriptors (e.g., photorealistic, digital art, watercolor)
3. Specifies lighting, mood, and atmosphere
4. Adds quality keywords (high quality, detailed, 4k)

Return ONLY the enhanced prompt, nothing else."""

        enhanced_prompt = await self._llm_complete(
            prompt=enhancement_prompt,
            task_type=TaskType.REASONING,
            temperature=0.7,
            max_tokens=500,
        )
        enhanced_prompt = enhanced_prompt.strip()
        logger.info("image_prompt_enhanced", original=prompt[:50], enhanced=enhanced_prompt[:100])

        # Step 2: Call HuggingFace Z-Image-Turbo via MCP
        if not self._mcp_client:
            return {"status": "error", "error": "MCP client not available for image generation"}

        try:
            result = await self._mcp_client.invoke(
                "hugging-face",
                "gr1_z_image_turbo_generate",
                {
                    "prompt": enhanced_prompt,
                    "resolution": "1024x1024 ( 1:1 )",
                    "steps": 8,
                    "random_seed": True,
                },
            )

            logger.info(
                "image_generation_complete",
                result_type=type(result).__name__,
                result_preview=str(result)[:500] if result else "None",
            )

            # Parse the result - Z-Image-Turbo returns (gallery_images, seed_str, seed_int)
            images = []

            # Handle string result - might be JSON or direct URL
            if isinstance(result, str):
                result = result.strip()
                # Try to parse as JSON
                try:
                    import json
                    parsed = json.loads(result)
                    if isinstance(parsed, list):
                        result = parsed
                    elif isinstance(parsed, dict):
                        result = parsed
                except json.JSONDecodeError:
                    # Might be a direct URL or error message
                    if result.startswith(("http://", "https://", "data:")):
                        images.append({
                            "url": result,
                            "prompt": enhanced_prompt,
                            "resolution": "1024x1024",
                            "seed": 0,
                        })
                    else:
                        logger.warning("image_generation_unexpected_string", result=result[:200])
                        return {"status": "error", "error": f"Unexpected response: {result[:200]}"}

            if isinstance(result, (list, tuple)) and len(result) >= 1:
                gallery = result[0] if isinstance(result[0], list) else [result[0]]
                seed = result[2] if len(result) > 2 else 0
                for img in gallery:
                    if img:
                        url = img.get("url", img) if isinstance(img, dict) else str(img)
                        images.append({
                            "url": url,
                            "prompt": enhanced_prompt,
                            "resolution": "1024x1024",
                            "seed": seed,
                        })
            elif isinstance(result, dict):
                images.append({
                    "url": result.get("url", ""),
                    "prompt": enhanced_prompt,
                    "resolution": "1024x1024",
                    "seed": result.get("seed", 0),
                })

            if images:
                return {
                    "status": "success",
                    "images": images,
                    "summary": f"Generated {len(images)} image(s) based on your request.",
                }
            else:
                return {"status": "error", "error": "No images generated"}

        except Exception as e:
            logger.exception("image_generation_error", error=str(e))
            return {"status": "error", "error": f"Image generation failed: {str(e)}"}

    async def _reflect(
        self,
        user_input: str,
        action_plan: dict[str, Any],
        action_result: dict[str, Any] | None,
    ) -> str:
        """Reflect on the action and result."""
        if not action_result or action_result.get("status") == "skipped":
            return "No action was needed for this request."

        if action_result.get("status") == "error":
            return f"Action failed: {action_result.get('error', 'unknown error')}"

        action = action_plan.get("action", "unknown")
        return f"Successfully completed {action}. Results available for response."

    async def _respond(
        self,
        user_input: str,
        reasoning: str,
        action_plan: dict[str, Any],
        action_result: dict[str, Any] | None,
        memories: list[dict[str, Any]],
    ) -> str:
        """Generate the final response to the user."""
        from dova.config.providers import TaskType

        # Build context for response generation
        context_parts = [f"User Question: {user_input}"]

        # Add conversation history for context
        conv_context = self._get_conversation_context(num_turns=3)
        if conv_context:
            context_parts.append(f"Conversation History:\n{conv_context}")

        # Add entities discussed
        if self.state.entities_discussed:
            entities_text = json.dumps(self.state.entities_discussed, indent=2, default=str)[:500]
            context_parts.append(f"Previously Discussed Entities:\n{entities_text}")

        if memories:
            memory_lines = []
            for m in memories[:3]:
                content = m.get('content', '')
                if isinstance(content, dict):
                    text = content.get('text') or content.get('response') or str(content)
                else:
                    text = str(content)
                memory_lines.append(f"- {text[:200]}")
            context_parts.append(f"Relevant Prior Knowledge:\n" + "\n".join(memory_lines))

        if action_result and action_result.get("status") == "success":
            # Format action results based on type
            action = action_plan.get("action", "")

            if action == "research":
                result_summary = []
                if action_result.get("answer"):
                    result_summary.append(f"Answer: {action_result['answer']}")
                if action_result.get("papers"):
                    papers = action_result["papers"][:3]
                    result_summary.append(f"Found {len(action_result.get('papers', []))} papers")
                    for p in papers:
                        result_summary.append(f"  - {p.get('title', 'Unknown')}")
                if action_result.get("repositories"):
                    repos = action_result["repositories"][:3]
                    result_summary.append(f"Found {len(action_result.get('repositories', []))} repositories")
                    for r in repos:
                        result_summary.append(f"  - {r.get('name', 'Unknown')}: {r.get('description', '')[:50]}")
                if action_result.get("models"):
                    models = action_result["models"][:3]
                    result_summary.append(f"Found {len(action_result.get('models', []))} models")
                    for m in models:
                        result_summary.append(f"  - {m.get('id', 'Unknown')}")
                if action_result.get("web_results"):
                    web_results = action_result["web_results"][:5]
                    result_summary.append(f"Found {len(action_result.get('web_results', []))} web sources:")
                    for w in web_results:
                        title = w.get('title', 'Unknown')
                        desc = w.get('description', '')[:200]
                        url = w.get('url', '')
                        result_summary.append(f"  - **{title}**: {desc}")
                        if url:
                            result_summary.append(f"    Source: {url}")
                context_parts.append(f"Research Results:\n" + "\n".join(result_summary))

            elif action == "debate":
                debate_summary = [
                    f"Topic Analysis: {action_result.get('summary', '')}",
                    f"Bull (Pro) Points: {', '.join(action_result.get('bull_strengths', [])[:2])}",
                    f"Bear (Con) Points: {', '.join(action_result.get('bear_concerns', [])[:2])}",
                    f"Recommendation: {action_result.get('recommendation', '')}",
                ]
                context_parts.append(f"Debate Analysis:\n" + "\n".join(debate_summary))

                # Include research data when debate ran with research (collaborative/deep mode)
                if action_result.get("papers") or action_result.get("repositories") or action_result.get("web_results"):
                    research_lines = []
                    for p in (action_result.get("papers") or [])[:5]:
                        title = p.get("title", "Unknown")
                        arxiv_id = p.get("metadata", {}).get("arxiv_id", p.get("arxiv_id", ""))
                        authors = p.get("metadata", {}).get("authors", p.get("authors", []))
                        author_str = ", ".join(authors[:3]) if authors else ""
                        research_lines.append(f"  - Paper: {title} [{arxiv_id}] by {author_str}")
                    for r in (action_result.get("repositories") or [])[:5]:
                        name = r.get("name") or r.get("full_name", "Unknown")
                        desc = (r.get("description") or "")[:80]
                        research_lines.append(f"  - Repo: {name} - {desc}")
                    for m in (action_result.get("models") or [])[:3]:
                        research_lines.append(f"  - Model: {m.get('id', 'Unknown')}")
                    for w in (action_result.get("web_results") or [])[:5]:
                        title = w.get("title", "Unknown")
                        desc = (w.get("description") or "")[:200]
                        url = w.get("url", "")
                        research_lines.append(f"  - Web: **{title}**: {desc}")
                        if url:
                            research_lines.append(f"    Source: {url}")
                    if research_lines:
                        context_parts.append(f"Research Results (ONLY reference these — do NOT invent titles):\n" + "\n".join(research_lines))

            elif action == "synthesize":
                context_parts.append(f"Synthesis:\n{action_result.get('synthesis', '')}")

            elif action == "image":
                images = action_result.get("images", [])
                if images:
                    image_summary = [f"Generated {len(images)} image(s):"]
                    for img in images:
                        image_summary.append(f"  - Prompt: {img.get('prompt', '')[:100]}...")
                        image_summary.append(f"  - Resolution: {img.get('resolution', '1024x1024')}")
                    context_parts.append(f"Image Generation Results:\n" + "\n".join(image_summary))

        prompt = f"""Generate a helpful, informative response based on this context:

{chr(10).join(context_parts)}

Guidelines:
- Be direct and informative
- ONLY reference papers, repositories, and models that appear in the Research Results above — NEVER invent or hallucinate titles
- If no Research Results are provided, do not cite specific paper titles
- Acknowledge limitations if applicable
- Suggest next steps if relevant

Response:"""

        response = await self._llm_complete(
            prompt=prompt,
            task_type=TaskType.CHAT,
            temperature=0.5,
            max_tokens=8000,
        )

        return response.strip()

    async def _remember(
        self,
        user_input: str,
        response: str,
        action_result: dict[str, Any] | None,
    ) -> None:
        """Store interaction in memory for future reference."""
        try:
            from dova.services.memory_enhanced import MemoryType

            # Store in short-term memory
            memory_content = {
                "query": user_input[:200],
                "response": response[:300],
                "session_id": self.state.session_id,
                "timestamp": time.time(),
                "had_action": action_result is not None,
            }

            await self.memory_service.store(
                memory_type=MemoryType.SHORT_TERM,
                content=memory_content,
                user_id=self.user_id,
                tags=["interactive_session"],
            )

            # Store significant results in long-term memory
            if action_result and action_result.get("status") == "success":
                if action_result.get("answer") or action_result.get("recommendation"):
                    long_term_content = {
                        "content": action_result.get("answer") or action_result.get("recommendation"),
                        "source": "interactive_session",
                        "query": user_input[:100],
                    }
                    await self.memory_service.store(
                        memory_type=MemoryType.LONG_TERM,
                        content=long_term_content,
                        user_id=self.user_id,
                        importance=0.8,
                        tags=["research_result"],
                    )
        except Exception as e:
            logger.warning("memory_store_failed", error=str(e))

    def _update_topic_tracking(
        self,
        user_input: str,
        action_result: dict[str, Any] | None,
        response: str,
    ) -> None:
        """Update the current topic being discussed."""
        # Extract topic from user input (first significant phrase)
        words = user_input.lower().split()
        # Filter common words
        stopwords = {"the", "a", "an", "is", "what", "how", "why", "who", "find", "search", "get", "show", "tell", "me", "about", "of", "for", "on", "in", "out"}
        significant = [w for w in words if w not in stopwords and len(w) > 2]
        if significant:
            self.state.last_topic = " ".join(significant[:5])

    def _extract_entities(self, action_result: dict[str, Any] | None) -> None:
        """Extract and store entities from action results for follow-up queries."""
        if not action_result or action_result.get("status") != "success":
            return

        # Extract papers
        papers = action_result.get("papers", [])
        if papers:
            for idx, p in enumerate(papers[:5], 1):
                title = p.get("title", "")
                metadata = p.get("metadata", {})
                entity_data = {
                    "type": "paper",
                    "title": title,
                    "authors": metadata.get("authors", []) or p.get("authors", []),
                    "id": metadata.get("arxiv_id", "") or p.get("id", "") or p.get("arxiv_id", ""),
                    "abstract": (p.get("description", "") or p.get("abstract", ""))[:300],
                    "url": p.get("url", ""),
                }
                # Index key for "paper 1" lookups
                self.state.entities_discussed[f"paper_{idx}"] = entity_data
                # Title key for name-based lookups
                if title:
                    self.state.entities_discussed[f"paper:{title[:50]}"] = entity_data

        # Extract repositories
        repos = action_result.get("repositories", [])
        if repos:
            for r in repos[:3]:
                name = r.get("name", "")
                if name:
                    self.state.entities_discussed[f"repo:{name}"] = {
                        "type": "repository",
                        "name": name,
                        "description": r.get("description", ""),
                        "url": r.get("url", ""),
                        "stars": r.get("stars", 0),
                    }

        # Extract models
        models = action_result.get("models", [])
        if models:
            for m in models[:3]:
                model_id = m.get("id", "")
                if model_id:
                    self.state.entities_discussed[f"model:{model_id}"] = {
                        "type": "model",
                        "id": model_id,
                        "downloads": m.get("downloads", 0),
                    }

        # Extract debate results
        if action_result.get("summary"):
            self.state.entities_discussed["last_debate"] = {
                "type": "debate",
                "summary": action_result.get("summary", ""),
                "recommendation": action_result.get("recommendation", ""),
            }

    def _track_pending_suggestions(self, response: str) -> None:
        """Track suggestions/questions offered to user for follow-up handling."""
        self.state.pending_suggestions = []
        self.state.last_question_to_user = ""

        # Look for questions asked to the user
        question_patterns = [
            r"Would you like (?:me to |to )?(.+?)\?",
            r"Do you want (?:me to |to )?(.+?)\?",
            r"Should I (.+?)\?",
            r"Can I help you (.+?)\?",
        ]
        for pattern in question_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                self.state.last_question_to_user = match.group(1).strip()
                self.state.pending_suggestions.append(match.group(1).strip())
                break

        # Look for numbered suggestions
        suggestion_matches = re.findall(r'\d+\.\s*\*?\*?([^*\n]+)', response)
        for suggestion in suggestion_matches[:3]:
            cleaned = suggestion.strip()
            if len(cleaned) > 10 and cleaned not in self.state.pending_suggestions:
                self.state.pending_suggestions.append(cleaned)

    def _print_thought(self, label: str, content: str) -> None:
        """Print a thought step with formatting."""
        # Use dim color for thinking
        # Show more content in verbose mode
        max_len = 500 if self.verbose else 200
        display = content[:max_len] + "..." if len(content) > max_len else content
        print(f"\033[2m[{label}] {display}\033[0m")

    def get_session_summary(self) -> dict[str, Any]:
        """Get summary of current session."""
        if not self.state:
            return {"status": "no_session"}

        return {
            "session_id": self.state.session_id,
            "user_id": self.state.user_id,
            "started_at": datetime.fromtimestamp(self.state.started_at).isoformat(),
            "turns": len(self.state.conversation) // 2,
            "memory_refs": len(self.state.memory_refs),
            "context": self.state.context,
        }


async def run_interactive_loop(
    show_thinking: bool = True,
    verbose: bool = False,
    orchestrator_type: str = "standard",
) -> None:
    """Run the interactive CLI loop.

    Args:
        show_thinking: Show reasoning steps
        verbose: Show timing and debug info
        orchestrator_type: "standard" (task-graph) or "thinking" (deliberation-first)
    """
    session = InteractiveSession(
        show_thinking=show_thinking,
        verbose=verbose,
        orchestrator_type=orchestrator_type,
    )

    # Print welcome message
    mode_desc = "Thinking (deliberation-first)" if orchestrator_type == "thinking" else "Standard (task-graph)"
    print("\n" + "=" * 60)
    print("  DOVA Interactive Mode")
    print(f"  Orchestrator: {mode_desc}")
    print("  Type your questions or commands. Type 'exit' to quit.")
    print("  Commands: /status, /clear, /thinking on|off, /orchestrator, /help")
    print("=" * 60 + "\n")

    session.start_session()
    print(f"Session started: {session.state.session_id}\n")

    while True:
        try:
            # Get user input
            user_input = input("\033[1m> \033[0m").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() == "exit":
                print("\nGoodbye!")
                break

            if user_input.startswith("/"):
                await handle_command(session, user_input)
                continue

            # Process input
            print()  # Add spacing
            response = await session.process_input(user_input)
            print(f"\n\033[1mDOVA:\033[0m {response}\n")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'exit' to quit.")
        except EOFError:
            print("\nGoodbye!")
            break


async def handle_command(session: InteractiveSession, command: str) -> None:
    """Handle CLI commands."""
    cmd = command.lower().strip()

    if cmd == "/help":
        print("""
Commands:
  /status     - Show session status
  /clear      - Clear conversation history
  /thinking on|off - Toggle thinking display
  /orchestrator standard|thinking - Switch orchestrator mode
  /history    - Show conversation history
  /memory     - Show memory references
  /help       - Show this help
  exit        - Exit interactive mode

Orchestrator modes:
  standard - Task-graph based (searches all relevant sources)
  thinking - Deliberation-first (reasons before deciding on tools)
""")

    elif cmd == "/status":
        summary = session.get_session_summary()
        print(f"\nSession: {summary.get('session_id', 'none')}")
        print(f"Turns: {summary.get('turns', 0)}")
        print(f"Memory refs: {summary.get('memory_refs', 0)}")
        print()

    elif cmd == "/clear":
        if session.state:
            session.state.conversation = []
            session.state.context = {}
            print("Conversation cleared.\n")

    elif cmd.startswith("/thinking"):
        parts = cmd.split()
        if len(parts) > 1:
            session.show_thinking = parts[1] == "on"
            print(f"Thinking display: {'on' if session.show_thinking else 'off'}\n")
        else:
            print(f"Thinking display is currently: {'on' if session.show_thinking else 'off'}\n")

    elif cmd == "/history":
        if session.state and session.state.conversation:
            print("\nConversation History:")
            for i, turn in enumerate(session.state.conversation):
                role = turn.role.upper()
                content = turn.content[:100] + "..." if len(turn.content) > 100 else turn.content
                print(f"  [{role}] {content}")
            print()
        else:
            print("No conversation history.\n")

    elif cmd == "/memory":
        if session.state and session.state.memory_refs:
            print(f"\nMemory references: {len(session.state.memory_refs)}")
            for ref in session.state.memory_refs[:10]:
                print(f"  - {ref}")
            print()
        else:
            print("No memory references.\n")

    elif cmd.startswith("/orchestrator"):
        parts = cmd.split()
        if len(parts) > 1:
            new_type = parts[1]
            if new_type in ("standard", "thinking"):
                session.orchestrator_type = new_type
                # Reset the thinking orchestrator to pick up new type
                session._thinking_orchestrator = None
                mode_desc = "Thinking (deliberation-first)" if new_type == "thinking" else "Standard (task-graph)"
                print(f"Orchestrator switched to: {mode_desc}\n")
            else:
                print(f"Unknown orchestrator type: {new_type}. Use 'standard' or 'thinking'.\n")
        else:
            mode_desc = "Thinking (deliberation-first)" if session.orchestrator_type == "thinking" else "Standard (task-graph)"
            print(f"Current orchestrator: {mode_desc}\n")

    else:
        print(f"Unknown command: {command}\nType /help for available commands.\n")
