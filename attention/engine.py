from __future__ import annotations

import random
import re
import time
from collections.abc import Callable

from attention.models import EvaluateAttentionResult, MatchedRuleOut, TriggerRule

MAX_REGEX_LEN = 512


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
    if matched:
        matched.sort(
            key=lambda t: (
                -_specificity(t[0])[0],
                -_specificity(t[0])[1],
                -t[0].priority,
            )
        )
        effective = max(r.probability for r, _ in matched)
    elif literal_hits:
        effective = max(global_baseline_probability, 0.35)
        notes.append("literal_self_or_wake_mention")
    else:
        effective = global_baseline_probability

    should = roll < effective

    out_rules: list[MatchedRuleOut] = []
    for rule, hit in matched:
        out_rules.append(
            MatchedRuleOut(
                rule_id=rule.rule_id,
                matched_text=hit,
                trigger_reason=rule.trigger_reason,
                response_hint=rule.response_hint,
                probability=rule.probability,
                priority=rule.priority,
            )
        )

    if should and matched and fire_record is not None:
        keys = [_state_key(rule.rule_id, group_id, user_id) for rule, _ in matched]
        fire_record(keys, now)

    return EvaluateAttentionResult(
        should_consider=should,
        effective_probability=effective,
        matched_rules=out_rules,
        baseline_used=baseline_used,
        sample_roll=roll,
        debug_notes=notes,
    )
