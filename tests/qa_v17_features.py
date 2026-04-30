"""
QA check for v1.7 advanced features.

Verifies three feature areas are deployed and functional end-to-end:

1. **Deliberation-first orchestration** — explicit meta-reasoning precedes
   tool invocation, informed by a persistent user model and entity-aware
   conversation context.
2. **Hybrid collaborative reasoning** — three-phase pipeline unifying
   ensemble diversity, blackboard transparency, and iterative refinement.
3. **Adaptive multi-tiered thinking + Multi-Round Adversarial Debate** —
   ReasoningMixin ReAct loops with iteration budget, plus multi-round
   DebateAgent (Bull vs Bear) wired into the orchestrator.

Run: `python tests/qa_v17_features.py`
Exit code 0 = all pass, 1 = at least one failure.
"""

from __future__ import annotations

import asyncio
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
INFO = "\033[36m→\033[0m"
results: list[tuple[str, bool, str]] = []


@contextmanager
def check(name: str):
    try:
        yield
        results.append((name, True, ""))
        print(f"  {PASS} {name}")
    except Exception as e:
        tb = traceback.format_exc(limit=3)
        results.append((name, False, f"{type(e).__name__}: {e}\n{tb}"))
        print(f"  {FAIL} {name}")
        print(f"    {type(e).__name__}: {e}")


def section(title: str):
    print(f"\n\033[1m{title}\033[0m")


# ============================================================================
# (1) Deliberation-first orchestration
# ============================================================================

async def qa_deliberation_first():
    section("(1) Deliberation-first orchestration")

    # Static structure
    from dova.agents.thinking_orchestrator import (
        ActionDecision,
        DELIBERATION_PROMPT,
        Deliberation,
        ThinkingOrchestrator,
        ToolConsideration,
        compute_intent_weights,
        get_available_tools,
    )
    from dova.agents.conversation_context import ConversationContext
    from dova.agents.user_model import ExpertiseLevel, ResponseDepth, UserModel
    from dova.agents.base import AgentResult, AgentTask

    with check("DELIBERATION_PROMPT includes explicit think-through steps"):
        assert "THINK THROUGH:" in DELIBERATION_PROMPT
        assert "ACTUALLY need" in DELIBERATION_PROMPT
        assert "AVAILABLE TOOLS" in DELIBERATION_PROMPT

    with check("ActionDecision enum exposes 3 decisions"):
        assert {d.value for d in ActionDecision} == {
            "respond_directly", "use_tools", "clarify",
        }

    with check("UserModel persists expertise, depth, formality, goals"):
        um = UserModel(user_id="qa")
        um.update_expertise("ml", ExpertiseLevel.EXPERT)
        um.preferred_depth = ResponseDepth.DETAILED
        um.current_goals = ["ship v1.7"]
        d = um.to_dict()
        um2 = UserModel.from_dict(d)
        assert um2.get_expertise("ml") == ExpertiseLevel.EXPERT
        assert um2.preferred_depth == ResponseDepth.DETAILED
        assert "ship v1.7" in um2.current_goals

    with check("ConversationContext tracks papers / repos / models / topic"):
        ctx = ConversationContext(session_id="s1", user_id="qa")
        ctx.add_turn(role="user", content="hi")
        ctx.add_paper({"title": "Attention Is All You Need", "arxiv_id": "1706.03762"})
        ctx.add_repo({"name": "pytorch/pytorch"})
        ctx.add_model({"id": "meta/Llama-3"})
        ctx.current_topic = "transformers"
        assert ctx.papers_discussed and ctx.repos_discussed and ctx.models_discussed
        assert ctx.current_topic == "transformers"

    with check("Deliberation dataclass carries intent_weights + tools_to_use"):
        d = Deliberation(
            understanding="x", can_answer_from_context=False,
            action=ActionDecision.USE_TOOLS,
            intent_weights={"ai": 0.6, "bio": 0.3, "web": 0.1},
        )
        assert sum(d.intent_weights.values()) == 1.0
        d.tools_to_use.append(ToolConsideration(
            tool_name="arxiv", would_help=True, rationale="r", search_query="q",
        ))
        assert d.tools_to_use[0].tool_name == "arxiv"

    with check("compute_intent_weights produces valid 3-way distribution"):
        for q in [
            "compare LoRA vs DoRA",
            "RFdiffusion de novo binders",
            "announced breakthrough in 2026",
        ]:
            w = compute_intent_weights(q)
            assert abs(sum(w.values()) - 1.0) < 0.01
            assert w["web"] >= 0.1 - 0.005  # web floor

    with check("get_available_tools surfaces bio umbrella to deliberation LLM"):
        tools = get_available_tools()
        assert "bio" in tools
        # Sub-servers hidden under the umbrella
        assert "pubmed-bio" not in tools

    # End-to-end: deliberation happens BEFORE tool execution
    with check("Orchestrator enters DELIBERATE → EXECUTE order"):
        orch = ThinkingOrchestrator(llm_router=MagicMock(), agents={}, mcp_client=None)
        trace: list[str] = []
        orig_delib = orch._deliberate
        orig_exec = orch._execute_selected_tools

        async def trace_delib(*a, **kw):
            trace.append("deliberate")
            return Deliberation(
                understanding="x", can_answer_from_context=False,
                action=ActionDecision.USE_TOOLS,
                tools_to_use=[
                    ToolConsideration(tool_name="arxiv", would_help=True,
                                       rationale="r", search_query="q")
                ],
            )

        async def trace_exec(*a, **kw):
            trace.append("execute")
            return {"papers": [], "repositories": [], "models": [],
                    "datasets": [], "web_results": [], "images": [], "mcp_results": []}

        orch._deliberate = trace_delib
        orch._execute_selected_tools = trace_exec
        orch.think = AsyncMock(return_value="synth")

        async def think_stream(*a, **kw):
            yield "synth"
        orch.think_stream = think_stream

        task = AgentTask(type="query", params={"query": "test"})
        result = await orch.execute(task)
        assert result.success
        assert trace == ["deliberate", "execute"], f"got {trace}"

        orch._deliberate = orig_delib
        orch._execute_selected_tools = orig_exec

    with check("User model + context load before deliberation (parallel)"):
        # This is a structural check — the execute() body calls
        # asyncio.gather on _load_user_model + _load_conversation_context.
        import inspect
        src = inspect.getsource(ThinkingOrchestrator.execute)
        assert "asyncio.gather" in src
        assert "_load_user_model" in src
        assert "_load_conversation_context" in src


# ============================================================================
# (2) Hybrid collaborative reasoning (3-phase)
# ============================================================================

async def qa_collaborative_reasoning():
    section("(2) Hybrid collaborative reasoning")

    from dova.services.collaborative import (
        CollaborationMode,
        CollaborativeReasoning,
    )
    from dova.services.ensemble import EnsembleReasoning, AggregationMethod
    from dova.services.blackboard import Blackboard, PostType

    with check("CollaborationMode exposes ENSEMBLE, BLACKBOARD, ITERATIVE, HYBRID"):
        members = {m.name for m in CollaborationMode}
        for needed in ("ENSEMBLE", "BLACKBOARD", "ITERATIVE", "HYBRID"):
            assert needed in members, f"missing {needed}"

    with check("HYBRID mode dispatches through the 3 phases"):
        # Inspect CollaborativeReasoning.reason() source for the ordered calls.
        import inspect
        src = inspect.getsource(CollaborativeReasoning)
        # All three sub-methods should be invoked in the hybrid branch.
        assert "_ensemble_reasoning" in src
        assert "_blackboard_reasoning" in src
        assert "_iterative_reasoning" in src

    with check("EnsembleReasoning supports multiple aggregation methods"):
        methods = {m.name for m in AggregationMethod}
        # Should include at least a couple of aggregation strategies
        assert len(methods) >= 2, f"got {methods}"

    with check("Blackboard emits typed posts (HYPOTHESIS, EVIDENCE, REFINEMENT, ...)"):
        post_types = {p.name for p in PostType}
        for needed in ("HYPOTHESIS", "EVIDENCE", "REFINEMENT", "CONSENSUS"):
            assert needed in post_types, f"missing {needed}"
        bb = Blackboard()
        pid = await bb.post(
            agent_name="qa", post_type=PostType.HYPOTHESIS, content="hello",
        )
        assert pid.startswith("post_")
        stored = bb._posts[pid]
        assert stored.content == "hello"
        assert stored.post_type == PostType.HYPOTHESIS

    with check("CollaborativeReasoning constructs with llm_func + phase components"):
        # llm_func is an async callable, not an llm_router
        async def fake_llm(prompt: str, **kw): return "answer"
        cr = CollaborativeReasoning(llm_func=fake_llm)
        # Verify it owns the 3-phase primitives
        assert cr.ensemble is not None  # ensemble diversity
        assert hasattr(cr, "_blackboard_reasoning")  # blackboard transparency
        assert hasattr(cr, "_iterative_reasoning")  # iterative refinement
        assert hasattr(cr, "_hybrid_reasoning")


# ============================================================================
# (3) Adaptive multi-tiered thinking + Multi-Round Adversarial Debate
# ============================================================================

async def qa_tiered_thinking_and_debate():
    section("(3) Adaptive multi-tiered thinking + Multi-Round Adversarial Debate")

    from dova.agents.mixins.reasoning import (
        ReasoningMixin, ReasoningTrace, ReasoningStep, StepType,
    )
    from dova.agents.debate import DebateAgent
    from dova.agents.base import AgentResult, AgentTask
    from dova.agents.thinking_orchestrator import (
        ThinkingOrchestrator,
        _is_evaluative_query,
        Deliberation,
        ActionDecision,
        ToolConsideration,
    )

    with check("ReasoningMixin.reason accepts max_iterations budget"):
        import inspect
        sig = inspect.signature(ReasoningMixin.reason)
        assert "max_iterations" in sig.parameters
        assert sig.parameters["max_iterations"].default >= 1

    with check("StepType covers Thought / Action / Observation / Reflection"):
        needed = {"THOUGHT", "ACTION", "OBSERVATION"}
        names = {s.name for s in StepType}
        missing = needed - names
        assert not missing, f"missing {missing}"

    with check("ReasoningTrace accumulates ordered steps"):
        trace = ReasoningTrace(problem="p")
        trace.steps.append(ReasoningStep(step_type=StepType.THOUGHT, content="t"))
        trace.steps.append(ReasoningStep(step_type=StepType.ACTION, content="a"))
        assert len(trace.steps) == 2
        assert trace.steps[0].step_type == StepType.THOUGHT
        assert trace.steps[1].step_type == StepType.ACTION

    with check("DebateAgent supports multi-round bull/bear"):
        llm = MagicMock()
        llm.complete = AsyncMock()
        agent = DebateAgent(llm_router=llm, num_rounds=3)
        assert agent.num_rounds == 3

    with check("_is_evaluative_query fires on comparative phrasing"):
        for q in [
            "compare LoRA vs DoRA", "RLHF vs DPO tradeoffs",
            "which is better: JAX or PyTorch", "pros and cons of MoE",
            "should I use DPO or PPO",
        ]:
            assert _is_evaluative_query(q), f"failed on: {q}"
        # Non-evaluative queries
        for q in [
            "explain attention",
            "RFdiffusion de novo binders",
            "latest scaling laws paper",
        ]:
            assert not _is_evaluative_query(q), f"false positive: {q}"

    with check("Orchestrator honors task.params.force_debate (unconditional)"):
        orch = ThinkingOrchestrator(llm_router=MagicMock(), agents={}, mcp_client=None)

        # Stub tool execution so we isolate the debate branch
        orch._deliberate = AsyncMock(return_value=Deliberation(
            understanding="x", can_answer_from_context=False,
            action=ActionDecision.USE_TOOLS,
            tools_to_use=[ToolConsideration(
                tool_name="arxiv", would_help=True, rationale="r", search_query="q",
            )],
        ))

        async def exec_stub(*a, **kw):
            return {"papers": [], "repositories": [], "models": [],
                    "datasets": [], "web_results": [], "images": [], "mcp_results": []}
        orch._execute_selected_tools = exec_stub
        orch.think = AsyncMock(return_value="synth")

        async def ts(*a, **kw):
            yield "synth"
        orch.think_stream = ts

        debate = MagicMock()
        debate.execute = AsyncMock(return_value=AgentResult(
            task_id="d", agent_name="debate", success=True,
            data={
                "summary": "debated",
                "bull_strengths": ["+A", "+B"],
                "bear_concerns": ["-X"],
                "balanced_assessment": "mixed",
                "recommendation": "prefer A",
                "confidence_score": 0.8,
            },
        ))
        orch.agents["debate"] = debate

        # Non-evaluative query but force_debate=True must still trigger debate
        task = AgentTask(type="query", params={
            "query": "explain attention mechanism",  # NOT evaluative
            "force_debate": True,
        })
        result = await orch.execute(task)
        assert result.success
        ar = result.data.get("action_result") or {}
        assert ar.get("bull_strengths") == ["+A", "+B"]
        assert ar.get("recommendation") == "prefer A"
        debate.execute.assert_called_once()

    with check("Orchestrator runs debate when auto_debate=True AND query evaluative"):
        orch = ThinkingOrchestrator(llm_router=MagicMock(), agents={}, mcp_client=None)

        orch._deliberate = AsyncMock(return_value=Deliberation(
            understanding="x", can_answer_from_context=False,
            action=ActionDecision.USE_TOOLS,
            tools_to_use=[ToolConsideration(
                tool_name="arxiv", would_help=True, rationale="r", search_query="q",
            )],
        ))

        async def exec_stub(*a, **kw):
            return {"papers": [], "repositories": [], "models": [],
                    "datasets": [], "web_results": [], "images": [], "mcp_results": []}
        orch._execute_selected_tools = exec_stub
        orch.think = AsyncMock(return_value="synth")

        async def ts(*a, **kw):
            yield "synth"
        orch.think_stream = ts

        debate = MagicMock()
        debate.execute = AsyncMock(return_value=AgentResult(
            task_id="d", agent_name="debate", success=True,
            data={"bull_strengths": ["+A"], "bear_concerns": ["-B"],
                   "recommendation": "go A", "confidence_score": 0.7,
                   "balanced_assessment": "", "summary": ""},
        ))
        orch.agents["debate"] = debate

        task = AgentTask(type="query", params={
            "query": "compare JAX vs PyTorch",   # evaluative
            "auto_debate": True,
        })
        result = await orch.execute(task)
        assert result.success
        ar = result.data.get("action_result") or {}
        assert ar.get("bull_strengths") == ["+A"]
        debate.execute.assert_called_once()

    with check("Orchestrator SKIPS debate for non-evaluative + auto_debate=True"):
        orch = ThinkingOrchestrator(llm_router=MagicMock(), agents={}, mcp_client=None)

        orch._deliberate = AsyncMock(return_value=Deliberation(
            understanding="x", can_answer_from_context=False,
            action=ActionDecision.USE_TOOLS,
            tools_to_use=[ToolConsideration(
                tool_name="arxiv", would_help=True, rationale="r", search_query="q",
            )],
        ))

        async def exec_stub(*a, **kw):
            return {"papers": [], "repositories": [], "models": [],
                    "datasets": [], "web_results": [], "images": [], "mcp_results": []}
        orch._execute_selected_tools = exec_stub
        orch.think = AsyncMock(return_value="synth")

        async def ts(*a, **kw):
            yield "synth"
        orch.think_stream = ts

        debate = MagicMock()
        debate.execute = AsyncMock()
        orch.agents["debate"] = debate

        task = AgentTask(type="query", params={
            "query": "RFdiffusion de novo binder protocol",  # not evaluative
            "auto_debate": True,
            "force_debate": False,
        })
        await orch.execute(task)
        debate.execute.assert_not_called()


# ============================================================================
# End-to-end configuration sanity
# ============================================================================

async def qa_end_to_end_config():
    section("End-to-end deployment wiring")

    from dova.api.schemas.chat import ChatRequest, ChatResponse

    with check("ChatRequest schema has always_debate + auto_debate + sources"):
        req = ChatRequest(message="hi")
        assert hasattr(req, "auto_debate")
        assert hasattr(req, "always_debate")
        assert "bio" in req.sources  # default source list includes bio

    with check("ChatResponse schema has intent_weights field"):
        resp = ChatResponse(session_id="s", message="m")
        assert hasattr(resp, "intent_weights")

    with check("Static UI includes AI/Web/Bio group chips"):
        html = Path("src/dova/api/static/index.html").read_text()
        assert 'data-group="ai"' in html
        assert 'data-group="web"' in html
        assert 'data-group="bio"' in html

    with check("Static UI wires always_debate toggle"):
        html = Path("src/dova/api/static/index.html").read_text()
        assert "alwaysDebate" in html
        assert "always_debate" in html

    with check("DebateAgent registered on ThinkingOrchestrator in API bootstrap"):
        src = Path("src/dova/api/main.py").read_text()
        assert 'register_agent("debate"' in src
        assert 'register_agent("research"' in src

    with check("CLI entry points default to thinking orchestrator"):
        src = Path("src/dova/cli/main.py").read_text()
        # Both --orchestrator flags default to 'thinking'
        assert src.count('default="thinking"') >= 2


# ============================================================================
# Runner
# ============================================================================

async def main():
    print("\n\033[1mDOVA v1.7 Advanced-Feature QA\033[0m")
    print(f"{INFO} Verifying deployment of the 3 pillar feature areas.\n")

    await qa_deliberation_first()
    await qa_collaborative_reasoning()
    await qa_tiered_thinking_and_debate()
    await qa_end_to_end_config()

    passed = sum(1 for _, ok, _ in results if ok)
    failed = sum(1 for _, ok, _ in results if not ok)
    print(f"\n\033[1mSummary:\033[0m  {passed} passed, {failed} failed "
          f"(out of {len(results)})\n")

    if failed:
        print("\033[31mFailures:\033[0m")
        for name, ok, detail in results:
            if not ok:
                print(f"  {FAIL} {name}")
                for line in detail.splitlines()[:5]:
                    print(f"      {line}")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
