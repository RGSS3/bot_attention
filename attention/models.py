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
    time_distribution: Literal["normal", "linear", "poisson"] = "normal"


class MatchedRuleOut(BaseModel):
    rule_id: str
    matched_text: str | None
    trigger_reason: str | None
    response_hint: str | None
    probability: float
    adjusted_probability: float = Field(
        description="Probability after time_distribution (linear remainder vs exponential-in-time poisson); used for max-merge and sampling gate.",
    )
    priority: int
    time_distribution: Literal["normal", "linear", "poisson"] = "normal"
    expires_at: int | None = Field(default=None, description="Unix expiry if non-permanent; null if open-ended.")
    ttl_remaining_sec: int | None = Field(
        default=None,
        description="Seconds until expires_at; null if no expiry.",
    )
    ttl_prompt_hint: str | None = Field(
        default=None,
        description="Short Chinese note for the persona (renewal, remaining time); null if permanent.",
    )


class EvaluateAttentionResult(BaseModel):
    should_consider: bool
    effective_probability: float = Field(
        description="max(adjusted_probability) over matched rules; linear/poisson decay within starts_at..expires_at.",
    )
    matched_rules: list[MatchedRuleOut]
    baseline_used: bool
    sample_roll: float
    debug_notes: list[str] = Field(default_factory=list)
