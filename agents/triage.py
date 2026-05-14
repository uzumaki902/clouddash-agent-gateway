"""
agents/triage.py

Triage Agent — the Supervisor node in the LangGraph graph.

Responsibilities:
  1. Read the latest user message from ConversationState.
  2. Ask the LLM (via structured output) to classify intent and produce a
     HandoverPayload that names the target specialist agent.
  3. Apply hard-coded safety overrides (escalation keywords, low confidence,
     max-turn ceiling) BEFORE the LLM call whenever possible.
  4. Return a dict that updates ConversationState via LangGraph's reducer.

This module MUST NOT contain any RAG retrieval or API route logic.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config.loader import get_agent_prompt, get_routing_config
from core.state import (
    AgentRole,
    ConversationState,
    ConversationStatus,
    HandoverPayload,
    Message,
    MessageRole,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level LLM client
# Instantiated once; using .with_structured_output forces the model to always
# return a payload that Pydantic can directly validate as HandoverPayload.
# ---------------------------------------------------------------------------

_llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o"),
    temperature=0,          # Zero temp → deterministic routing decisions
    api_key=os.getenv("OPENAI_API_KEY"),
).with_structured_output(HandoverPayload)


# ---------------------------------------------------------------------------
# Pre-LLM safety checks (no API call needed)
# ---------------------------------------------------------------------------


def _check_escalation_keywords(text: str, keywords: list[str]) -> str | None:
    """
    Return the matched keyword if `text` triggers an instant-escalate rule,
    otherwise return None.

    Comparison is case-insensitive and checks whole-word substrings.
    """
    lowered = text.lower()
    for kw in keywords:
        if kw.lower() in lowered:
            return kw
    return None


def _build_fallback_handover(
    state: ConversationState,
    reason: str,
    trace_id: str,
) -> HandoverPayload:
    """
    Construct an ESCALATION HandoverPayload without calling the LLM.
    Used when pre-flight rules force an escalation path.
    """
    return HandoverPayload(
        trace_id=trace_id,
        from_agent=AgentRole.TRIAGE,
        to_agent=AgentRole.ESCALATION,
        intent="forced_escalation",
        confidence=1.0,
        summary=reason,
        rag_required=False,
        escalation_reason=reason,
    )


# ---------------------------------------------------------------------------
# Main node function
# ---------------------------------------------------------------------------


async def triage_node(state: ConversationState) -> dict:
    """
    LangGraph node — classifies the user's intent and routes to a specialist.

    Args:
        state: The current, fully-typed ConversationState snapshot passed in
               by the LangGraph executor.

    Returns:
        A dict of state field updates consumed by LangGraph's state reducer.
        Fields returned:
          - handover          : HandoverPayload  (latest routing decision)
          - handover_history  : list[HandoverPayload]  (full audit trail)
          - current_agent     : AgentRole  (who handles next turn)
          - status            : ConversationStatus  (escalation flag if needed)
          - error             : str | None  (populated on graceful fault)

    Raises:
        Nothing — all exceptions are caught and surfaced as a state.error
        with an ESCALATION fallback so the graph never deadlocks.
    """
    trace_id = state.trace_id
    log_ctx = {"trace_id": trace_id, "session_id": state.session_id}

    logger.info(
        "[triage_node] Invoked. turn_count=%d session=%s trace=%s",
        state.turn_count,
        state.session_id,
        trace_id,
    )

    # ------------------------------------------------------------------
    # 1. Guard: no user message → cannot route
    # ------------------------------------------------------------------
    latest_msg: Message | None = state.latest_user_message()
    if latest_msg is None:
        logger.warning(
            "[triage_node] No user message found in history. "
            "trace=%s session=%s",
            trace_id,
            state.session_id,
        )
        return {
            "error": "Triage aborted: no user message present in conversation history.",
            "current_agent": AgentRole.TRIAGE,
        }

    user_text = latest_msg.content

    # ------------------------------------------------------------------
    # 2. Pre-flight: max-turn ceiling
    # ------------------------------------------------------------------
    routing_cfg = get_routing_config()

    if state.turn_count >= routing_cfg.max_turns_before_escalation:
        reason = (
            f"Conversation exceeded the maximum turn limit "
            f"({routing_cfg.max_turns_before_escalation} turns)."
        )
        logger.warning("[triage_node] Max-turn ceiling hit. %s", log_ctx)
        payload = _build_fallback_handover(state, reason, trace_id)
        return _build_state_update(state, payload, force_escalation_status=True)

    # ------------------------------------------------------------------
    # 3. Pre-flight: hard escalation keywords (no LLM call needed)
    # ------------------------------------------------------------------
    matched_kw = _check_escalation_keywords(
        user_text, routing_cfg.escalation_keywords
    )
    if matched_kw:
        reason = f"Escalation keyword detected: '{matched_kw}'."
        logger.warning(
            "[triage_node] Keyword escalation triggered. keyword='%s' %s",
            matched_kw,
            log_ctx,
        )
        payload = _build_fallback_handover(state, reason, trace_id)
        return _build_state_update(state, payload, force_escalation_status=True)

    # ------------------------------------------------------------------
    # 4. Load triage system prompt from config (never hardcoded here)
    # ------------------------------------------------------------------
    agent_cfg = get_agent_prompt("triage")
    system_prompt = agent_cfg.system_prompt

    # ------------------------------------------------------------------
    # 5. Build the message list for the LLM
    #    We pass the full conversation history as context so the LLM can
    #    produce an accurate summary in the HandoverPayload.
    # ------------------------------------------------------------------
    history_text = "\n".join(
        f"[{msg.role.value.upper()}]: {msg.content}"
        for msg in state.history
    )

    llm_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=(
                f"Conversation history:\n{history_text}\n\n"
                f"Latest user message:\n{user_text}\n\n"
                f"Session trace_id: {trace_id}\n"
                "Classify and route. Respond only with the required JSON."
            )
        ),
    ]

    # ------------------------------------------------------------------
    # 6. Invoke LLM with structured output → HandoverPayload
    # ------------------------------------------------------------------
    try:
        logger.info(
            "[triage_node] Invoking LLM for intent classification. %s", log_ctx
        )

        # _llm already has .with_structured_output(HandoverPayload) applied;
        # the return type here is guaranteed to be HandoverPayload.
        payload: HandoverPayload = await _llm.ainvoke(llm_messages)  # type: ignore[assignment]

        # Inject the propagated trace_id — the LLM generates its own UUID
        # for trace_id, but we override it with the request-scoped one.
        payload = payload.model_copy(
            update={
                "trace_id": trace_id,
                "from_agent": AgentRole.TRIAGE,
            }
        )

        logger.info(
            "[triage_node] Routing decision: to_agent=%s intent=%s confidence=%.2f trace=%s",
            payload.to_agent.value,
            payload.intent,
            payload.confidence,
            trace_id,
        )

    except Exception as exc:  # noqa: BLE001  — intentional broad catch for graph safety
        logger.error(
            "[triage_node] LLM invocation failed. error=%r trace=%s session=%s",
            exc,
            trace_id,
            state.session_id,
            exc_info=True,
        )
        # Graceful degradation: escalate rather than crash the graph
        reason = f"Triage LLM failure — escalating for manual review. Error: {exc!r}"
        payload = _build_fallback_handover(state, reason, trace_id)
        return _build_state_update(
            state, payload, force_escalation_status=True, error=str(exc)
        )

    # ------------------------------------------------------------------
    # 7. Post-LLM: confidence threshold override
    # ------------------------------------------------------------------
    if payload.confidence < routing_cfg.confidence_threshold:
        original_intent = payload.intent
        logger.warning(
            "[triage_node] Confidence %.2f below threshold %.2f. "
            "Overriding to ESCALATION. original_intent=%s trace=%s",
            payload.confidence,
            routing_cfg.confidence_threshold,
            original_intent,
            trace_id,
        )
        reason = (
            f"Low classification confidence ({payload.confidence:.2f} < "
            f"{routing_cfg.confidence_threshold}). Original intent: {original_intent}."
        )
        payload = _build_fallback_handover(state, reason, trace_id)
        return _build_state_update(state, payload, force_escalation_status=True)

    # ------------------------------------------------------------------
    # 8. Happy path — return state update dict
    # ------------------------------------------------------------------
    return _build_state_update(state, payload)


# ---------------------------------------------------------------------------
# State update builder (keeps the node return logic DRY)
# ---------------------------------------------------------------------------


def _build_state_update(
    state: ConversationState,
    payload: HandoverPayload,
    *,
    force_escalation_status: bool = False,
    error: str | None = None,
) -> dict:
    """
    Build the LangGraph reducer-compatible dict from a finalized HandoverPayload.

    LangGraph merges this dict into the current state snapshot; we only
    include fields we actually want to change.
    """
    new_status = state.status

    if force_escalation_status:
        new_status = ConversationStatus.PENDING_ESCALATION

    return {
        # The single `handover` shortcut (latest)
        "handover": payload,
        # Full routing audit trail — append to existing list
        "handover_history": [*state.handover_history, payload],
        # Advance the active agent pointer
        "current_agent": payload.to_agent,
        # Lifecycle status update (only changes if escalating)
        "status": new_status,
        # Clear or propagate error annotation
        "error": error,
        # Stamp the update time
        "updated_at": datetime.now(timezone.utc),
    }
