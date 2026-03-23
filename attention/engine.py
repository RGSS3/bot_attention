from __future__ import annotations

import math
import random
import re
import time
from collections.abc import Callable

from attention.models import EvaluateAttentionResult, MatchedRuleOut, TriggerRule

MAX_REGEX_LEN = 512
# Poisson mode: we treat "attention / worth replying" loosely like events in a Poisson-like process
# over the TTL window—i.e. how likely a fresh "engagement opportunity" still feels as time passes.
# Interarrival intuition → exponential damping in elapsed fraction: exp(-λ·u) at u=elapsed/window.
# λ larger ⇒ steeper decay toward window end (factor exp(-λ) when u=1).
POISSON_DECAY_LAMBDA = 2.5


def _parse_time_window(tw: str | None, now_local: time.struct_time) -> bool:
    if not tw or not tw.strip():
        return True
    parts = tw.strip().split("-", 1)
    if len(parts) != 2:
        return True
    try:
        sh, sm = map(int, parts[0].strip().split(":", 1))
        eh, em = map(int, parts[1].strip().split(":", 1))
    except ValueError:
        return True
    cur = now_local.tm_hour * 60 + now_local.tm_min
    start = sh * 60 + sm
    end = eh * 60 + em
    if start <= end:
        return start <= cur <= end
    return cur >= start or cur <= end


def _compile_regex(pattern: str, case_fold: bool) -> re.Pattern[str]:
    if len(pattern) > MAX_REGEX_LEN:
        raise ValueError(f"regex longer than {MAX_REGEX_LEN}")
    flags = re.IGNORECASE if case_fold else 0
    return re.compile(pattern, flags)


def _match_rule(text: str, rule: TriggerRule) -> str | None:
    if rule.match_mode == "keyword":
        hay = text.lower() if rule.case_fold else text
        needle = rule.pattern.lower() if rule.case_fold else rule.pattern
        if needle in hay:
            return needle
        return None
    rx = _compile_regex(rule.pattern, rule.case_fold)
    m = rx.search(text)
    return m.group(0) if m else None


def _scope_ok(rule: TriggerRule, group_id: str, user_id: str) -> bool:
    if rule.group_id not in ("*", group_id):
        return False
    if rule.user_id not in ("*", user_id):
        return False
    return True


def _time_ok(rule: TriggerRule, now_ts: int, now_local: time.struct_time) -> bool:
    if rule.status != "active":
        return False
    if rule.starts_at is not None and now_ts < rule.starts_at:
        return False
    if rule.expires_at is not None and now_ts > rule.expires_at:
        return False
    return _parse_time_window(rule.time_window, now_local)


def _specificity(rule: TriggerRule) -> tuple[int, int]:
    g = 1 if rule.group_id != "*" else 0
    u = 1 if rule.user_id != "*" else 0
    return (g, u)


def _state_key(rule_id: str, group_id: str, user_id: str) -> str:
    return f"{rule_id}|{group_id}|{user_id}"


def _cooldown_ok(last_fire: float, cooldown_sec: int, now: float) -> bool:
    if cooldown_sec <= 0:
        return True
    if last_fire <= 0:
        return True
    return now - last_fire >= cooldown_sec


def _hourly_ok(count: int, window_start: float, cap: int, now: float) -> bool:
    if cap <= 0:
        return True
    if now - window_start >= 3600:
        return True
    return count < cap


def _time_window_decay_factor(rule: TriggerRule, now_ts: int) -> float:
    """Within [starts_at, expires_at]:
    linear — remaining fraction of the window (uniform shrink of effective p).
    poisson — exponential in elapsed fraction (see POISSON_DECAY_LAMBDA: Poisson / no-memory style
    modeling of "chance of a response-relevant beat" over the interval, not linear left-over time).
    """
    if rule.time_distribution == "normal":
        return 1.0
    if rule.expires_at is None:
        return 1.0
    if rule.starts_at is None or rule.expires_at <= rule.starts_at:
        return 1.0
    total = float(rule.expires_at - rule.starts_at)
    rem = float(rule.expires_at - now_ts)
    elapsed = total - rem
    u = max(0.0, min(1.0, elapsed / total))
    if rule.time_distribution == "linear":
        return max(0.0, min(1.0, rem / total))
    if rule.time_distribution == "poisson":
        return max(0.0, min(1.0, math.exp(-POISSON_DECAY_LAMBDA * u)))
    return 1.0


def _adjusted_probability(rule: TriggerRule, now_ts: int) -> float:
    return max(0.0, min(1.0, rule.probability * _time_window_decay_factor(rule, now_ts)))


def _ttl_aux(rule: TriggerRule, now_ts: int) -> tuple[int | None, str | None]:
    """Remaining seconds and a short hint for non-permanent rules (prompt injection)."""
    if rule.expires_at is None:
        return None, None
    rem = int(rule.expires_at - now_ts)
    if rem <= 0:
        return rem, "非永久规则：已到截止时间（若仍命中请核对消息时间与规则 expires_at）。"
    if rem < 60:
        hint = f"非永久规则：还剩约 {rem} 秒；窗口将结束，可按需续期。"
    elif rem < 3600:
        m = max(1, rem // 60)
        hint = f"非永久规则：还剩约 {m} 分钟；如需保持关注可续期。"
    else:
        h, rest = divmod(rem, 3600)
        m = rest // 60
        if m >= 30:
            hint = f"非永久规则：还剩约 {h + 1} 小时；可按需续期。"
        elif h > 0:
            hint = f"非永久规则：还剩约 {h} 小时{m} 分；可按需续期。"
        else:
            hint = f"非永久规则：还剩约 {max(1, m)} 分钟；可按需续期。"
    return rem, hint


def evaluate_attention(
    *,
    message_text: str,
    group_id: str,
    user_id: str,
    now_ts: int | None,
    rules: list[TriggerRule],
    fire_get: Callable[[str], tuple[float, int, float]],
    fire_record: Callable[[list[str], float], None] | None,
    global_baseline_probability: float,
    extra_literal_triggers: list[str] | None,
    sample_roll: float | None,
) -> EvaluateAttentionResult:
    notes: list[str] = []
    now = time.time()
    ts = int(now_ts if now_ts is not None else now)
    now_local = time.localtime(ts)

    roll = sample_roll if sample_roll is not None else random.random()
    matched: list[tuple[TriggerRule, str | None]] = []

    fire_cache: dict[str, tuple[float, int, float]] = {}

    def _cached_fire(key: str) -> tuple[float, int, float]:
        if key not in fire_cache:
            fire_cache[key] = fire_get(key)
        return fire_cache[key]

    for rule in rules:
        if not rule.enabled:
            continue
        if not _scope_ok(rule, group_id, user_id):
            continue
        if not _time_ok(rule, ts, now_local):
            continue
        key = _state_key(rule.rule_id, group_id, user_id)
        last_fire, hour_count, window_start = _cached_fire(key)
        if not _cooldown_ok(last_fire, rule.cooldown_sec, now):
            notes.append(f"skipped_by_cooldown:{rule.rule_id}")
            continue
        if not _hourly_ok(hour_count, window_start, rule.max_hits_per_hour, now):
            notes.append(f"skipped_by_hour_cap:{rule.rule_id}")
            continue
        try:
            hit = _match_rule(message_text, rule)
        except re.error as e:
            notes.append(f"regex_error:{rule.rule_id}:{e}")
            continue
        if hit is not None:
            matched.append((rule, hit))

    literal_hits: list[str] = []
    if extra_literal_triggers:
        lower = message_text.lower()
        for lit in extra_literal_triggers:
            if lit and lit.lower() in lower:
                literal_hits.append(lit)

    baseline_used = not matched and not literal_hits
    gate_rule_id: str | None = None
    if matched:
        matched.sort(
            key=lambda t: (
                -_specificity(t[0])[0],
                -_specificity(t[0])[1],
                -t[0].priority,
                t[0].rule_id,
            )
        )
        win_rule, _ = matched[0]
        gate_rule_id = win_rule.rule_id
        effective = _adjusted_probability(win_rule, ts)
    elif literal_hits:
        effective = max(global_baseline_probability, 0.35)
        notes.append("literal_self_or_wake_mention")
    else:
        effective = global_baseline_probability

    should = roll < effective

    out_rules: list[MatchedRuleOut] = []
    for rule, hit in matched:
        adj = _adjusted_probability(rule, ts)
        ttl_sec, ttl_hint = _ttl_aux(rule, ts)
        out_rules.append(
            MatchedRuleOut(
                rule_id=rule.rule_id,
                matched_text=hit,
                trigger_reason=rule.trigger_reason,
                response_hint=rule.response_hint,
                probability=rule.probability,
                adjusted_probability=adj,
                priority=rule.priority,
                time_distribution=rule.time_distribution,
                expires_at=rule.expires_at,
                ttl_remaining_sec=ttl_sec,
                ttl_prompt_hint=ttl_hint,
            )
        )

    if should and matched and fire_record is not None:
        win_rule, _ = matched[0]
        fire_record([_state_key(win_rule.rule_id, group_id, user_id)], now)

    return EvaluateAttentionResult(
        should_consider=should,
        effective_probability=effective,
        gate_rule_id=gate_rule_id,
        matched_rules=out_rules,
        baseline_used=baseline_used,
        sample_roll=roll,
        debug_notes=notes,
    )
