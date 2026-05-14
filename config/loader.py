"""
config/loader.py

Loads config/prompts.yaml exactly once at import time and exposes
typed accessors for agent prompts and routing rules.

Usage:
    from config.loader import get_agent_prompt, get_routing_config, get_rag_config
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Resolve the YAML path relative to this file so it works regardless of cwd.
_PROMPTS_PATH = Path(__file__).parent / "prompts.yaml"


# ---------------------------------------------------------------------------
# Typed config models
# ---------------------------------------------------------------------------


class AgentPromptConfig(BaseModel):
    """Configuration block for a single agent."""

    role: str
    system_prompt: str


class RoutingConfig(BaseModel):
    """Routing rules read from the YAML; consumed by the LangGraph edge resolver."""

    confidence_threshold: float = Field(ge=0.0, le=1.0)
    max_turns_before_escalation: int = Field(ge=1)
    escalation_keywords: list[str] = Field(default_factory=list)


class RAGConfig(BaseModel):
    """RAG retrieval parameters; consumed by rag/retriever.py."""

    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    similarity_threshold: float
    collection_name: str


class PromptsConfig(BaseModel):
    """Root config model — mirrors the top-level structure of prompts.yaml."""

    agents: dict[str, AgentPromptConfig]
    routing: RoutingConfig
    rag: RAGConfig


# ---------------------------------------------------------------------------
# Internal loader (cached)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_config() -> PromptsConfig:
    """
    Parse prompts.yaml into a validated PromptsConfig.

    Called once; subsequent calls return the cached instance.
    Raises RuntimeError if the file is missing or invalid.
    """
    if not _PROMPTS_PATH.exists():
        raise RuntimeError(
            f"[config/loader] prompts.yaml not found at {_PROMPTS_PATH}. "
            "Ensure the file exists before starting the application."
        )

    try:
        raw: dict[str, Any] = yaml.safe_load(_PROMPTS_PATH.read_text(encoding="utf-8"))
        config = PromptsConfig.model_validate(raw)
        logger.info("[config/loader] prompts.yaml loaded and validated successfully.")
        return config
    except yaml.YAMLError as exc:
        raise RuntimeError(
            f"[config/loader] Failed to parse prompts.yaml: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Public accessors
# ---------------------------------------------------------------------------


def get_agent_prompt(agent_name: str) -> AgentPromptConfig:
    """
    Return the prompt config for a named agent.

    Args:
        agent_name: Key as it appears under `agents:` in prompts.yaml
                    (e.g. "triage", "tech_support").

    Raises:
        KeyError: If `agent_name` is not defined in prompts.yaml.
    """
    config = _load_config()
    if agent_name not in config.agents:
        raise KeyError(
            f"[config/loader] Agent '{agent_name}' not found in prompts.yaml. "
            f"Available agents: {list(config.agents.keys())}"
        )
    return config.agents[agent_name]


def get_routing_config() -> RoutingConfig:
    """Return the routing rules block from prompts.yaml."""
    return _load_config().routing


def get_rag_config() -> RAGConfig:
    """Return the RAG configuration block from prompts.yaml."""
    return _load_config().rag
