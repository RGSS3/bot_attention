from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TriggerRule(BaseModel):
    rule_id: str
    enabled: bool = True
    priority: int = 0
    source: Literal["manual", "mcp_memory", "llm_accept", "system"] = "manual"
    group_id: str = "*"
    user_id: str = "*"
    pattern: str
    match_mode: Literal["keyword", "regex"] = "keyword"
    case_fold: bool = True
    term_type: str | None = None
    starts_at: int | None = None
    expires_at: int | None = None
    time_window: str | None = None
    status: Literal["active", "expired", "ended"] = "active"
    probability: float = Field(ge=0.0, le=1.0, default=0.2)
    cooldown_sec: int = 0
    max_hits_per_hour: int = 0
    trigger_reason: str | None = None
    response_hint: str | None = None
    reason_visibility: Literal["llm_only", "debug_and_llm"] = "llm_only"
    created_at: int | None = None
    updated_at: int | None = None


class MatchedRuleOut(BaseModel):
    rule_id: str
    matched_text: str | None
    trigger_reason: str | None
    response_hint: str | None
    probability: float
    priority: int


class EvaluateAttentionResult(BaseModel):
    should_consider: bool
    effective_probability: float
    matched_rules: list[MatchedRuleOut]
    baseline_used: bool
    sample_roll: float
    debug_notes: list[str] = Field(default_factory=list)
