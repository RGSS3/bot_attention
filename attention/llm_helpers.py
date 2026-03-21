from __future__ import annotations

import hashlib
import time
from attention.models import TriggerRule

_MAX_DURATION_MIN = 60 * 24 * 7


def _norm_group(group_id: str) -> str:
    g = (group_id or "").strip()
    return g if g else "*"


def _now_ts(now_ts: int | None) -> int:
    return int(now_ts if now_ts is not None else time.time())


def _clamp_duration_minutes(duration_minutes: float) -> float:
    if duration_minutes <= 0:
        return 5.0
    return min(float(duration_minutes), _MAX_DURATION_MIN)


def build_nickname_rule(
    *,
    user_id: str,
    nickname: str,
    group_id: str = "",
    trigger_reason: str = "",
    response_hint: str = "",
    probability: float = 0.55,
    now_ts: int | None = None,
) -> TriggerRule:
    uid = user_id.strip()
    nick = nickname.strip()
    if not uid or not nick:
        raise ValueError("user_id and nickname must be non-empty")
    gid = _norm_group(group_id)
    rule_id = f"addr:{gid}:{uid}"
    reason = trigger_reason.strip() or (
        "用户在当前作用域里用这个称呼指向我；命中只表示值得考虑接话，仍可能是在叫别人，需要结合上下文决定是否回复。"
    )
    return TriggerRule(
        rule_id=rule_id,
        enabled=True,
        priority=55,
        source="llm_accept",
        group_id=gid,
        user_id=uid,
        pattern=nick,
        match_mode="keyword",
        case_fold=True,
        term_type="nickname",
        starts_at=_now_ts(now_ts),
        expires_at=None,
        status="active",
        probability=max(0.0, min(1.0, probability)),
        cooldown_sec=0,
        max_hits_per_hour=0,
        trigger_reason=reason,
        response_hint=response_hint.strip() or None,
        reason_visibility="llm_only",
    )


def build_user_focus_rule(
    *,
    user_id: str,
    group_id: str = "",
    duration_minutes: float = 5.0,
    probability: float = 1.0,
    trigger_reason: str = "",
    response_hint: str = "",
    now_ts: int | None = None,
) -> TriggerRule:
    uid = user_id.strip()
    if not uid:
        raise ValueError("user_id must be non-empty")
    gid = _norm_group(group_id)
    dm = _clamp_duration_minutes(duration_minutes)
    ts = _now_ts(now_ts)
    rule_id = f"focus_user:{gid}:{uid}"
    reason = trigger_reason.strip() or (
        "短时“听这个人说话”窗口：该用户任意文本都可能进入高关注通道；是否真回复仍由你在生成阶段决定，可输出不回复。"
    )
    return TriggerRule(
        rule_id=rule_id,
        enabled=True,
        priority=82,
        source="llm_accept",
        group_id=gid,
        user_id=uid,
        pattern=".*",
        match_mode="regex",
        case_fold=True,
        term_type="user_focus",
        starts_at=ts,
        expires_at=ts + int(dm * 60),
        status="active",
        probability=max(0.0, min(1.0, probability)),
        cooldown_sec=0,
        max_hits_per_hour=0,
        trigger_reason=reason,
        response_hint=response_hint.strip() or None,
        reason_visibility="llm_only",
        time_distribution="poisson",
    )


def _topic_rule_id(gid: str, topic: str) -> str:
    h = hashlib.sha256(f"{gid}\n{topic}".encode("utf-8")).hexdigest()[:20]
    return f"topic_kw:{gid}:h{h}"


def build_topic_keyword_rule(
    *,
    topic: str,
    group_id: str = "",
    duration_minutes: float = 5.0,
    probability: float = 1.0,
    trigger_reason: str = "",
    response_hint: str = "",
    now_ts: int | None = None,
) -> TriggerRule:
    t = topic.strip()
    if not t:
        raise ValueError("topic must be non-empty")
    gid = _norm_group(group_id)
    dm = _clamp_duration_minutes(duration_minutes)
    ts = _now_ts(now_ts)
    rule_id = _topic_rule_id(gid, t)
    reason = trigger_reason.strip() or (
        "临时话题词窗口：群里正在聊这个主题时提高被提醒概率；仅字符串命中，不做语义理解；到期自动失效，可反复调用本工具续期。"
    )
    return TriggerRule(
        rule_id=rule_id,
        enabled=True,
        priority=48,
        source="llm_accept",
        group_id=gid,
        user_id="*",
        pattern=t,
        match_mode="keyword",
        case_fold=True,
        term_type="topic",
        starts_at=ts,
        expires_at=ts + int(dm * 60),
        status="active",
        probability=max(0.0, min(1.0, probability)),
        cooldown_sec=0,
        max_hits_per_hour=0,
        trigger_reason=reason,
        response_hint=response_hint.strip() or None,
        reason_visibility="llm_only",
        time_distribution="poisson",
    )
