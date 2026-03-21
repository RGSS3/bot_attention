from __future__ import annotations

import json
import os
import time
from pathlib import Path

from fastmcp import FastMCP
from pydantic import BaseModel, Field

from attention.engine import evaluate_attention as run_evaluate
from attention.llm_helpers import (
    build_nickname_rule,
    build_topic_keyword_rule,
    build_user_focus_rule,
)
from attention.models import EvaluateAttentionResult, TriggerRule
from attention.store import (
    connect,
    end_rule_status,
    fetch_rules,
    fetch_rules_for_eval,
    get_fire_row,
    init_schema,
    maybe_seed_from_json,
    record_fires,
    upsert_rule_row,
)


def _mcp_role() -> str:
    v = (os.environ.get("BOT_ATTENTION_MCP_ROLE") or "full").strip().lower()
    if v not in ("full", "gate", "admin"):
        return "full"
    return v


def _mcp_name() -> str:
    return {
        "gate": "bot-attention-gate",
        "admin": "bot-attention-admin",
        "full": "bot-attention",
    }[_mcp_role()]


def _db_path() -> Path:
    return Path(os.environ.get("BOT_ATTENTION_DB_PATH", Path("data") / "attention.db")).resolve()


def _rules_json_seed_path() -> Path:
    return Path(os.environ.get("BOT_ATTENTION_RULES_PATH", Path("data") / "rules.json")).resolve()


_role = _mcp_role()
mcp = FastMCP(
    _mcp_name(),
    instructions=(
        "Low-cost attention gate backed by SQLite (WAL). "
        "Use the same BOT_ATTENTION_DB_PATH for every MCP process (e.g. Astrbot gate + LLM admin) so rules and cooldowns stay consistent. "
        "Role gate: evaluate only. Role admin: rule writes + persona helper tools. Role full: both."
    ),
)


def _open_db():
    conn = connect(_db_path())
    init_schema(conn)
    maybe_seed_from_json(conn, _rules_json_seed_path())
    return conn


if _role in ("full", "gate"):

    @mcp.tool
    def evaluate_attention(
        message_text: str,
        group_id: str = "",
        user_id: str = "",
        now_ts: int | None = None,
        global_baseline_probability: float = 0.0,
        extra_literal_triggers: list[str] | None = None,
        sample_roll: float | None = None,
        rules_override_json: str | None = None,
    ) -> EvaluateAttentionResult:
        """Decide whether the QQ persona should *consider* replying (no NLP/LLM).

        Astrbot can pass message_str, group_id, sender id, timestamp.
        Rules load from SQLite per call: only enabled, active, in starts/expires range, and scope matching group_id/user_id (daily time_window still filtered in-engine).
        rules_override_json: if set, replaces DB rules for this call only; no fire-state writes.
        extra_literal_triggers: e.g. self_id substrings for cheap recall.
        sample_roll: optional fixed roll in [0,1) for tests.
        """
        gid = group_id or "*"
        uid = user_id or "*"
        ts = int(now_ts if now_ts is not None else time.time())
        conn = _open_db()
        try:
            if rules_override_json:
                rules = [TriggerRule.model_validate(x) for x in json.loads(rules_override_json)]

                def fire_get(_key: str) -> tuple[float, int, float]:
                    return (0.0, 0, 0.0)

                fire_record = None
            else:
                rules = fetch_rules_for_eval(conn, group_id=gid, user_id=uid, now_ts=ts)

                def fire_get(key: str) -> tuple[float, int, float]:
                    return get_fire_row(conn, key)

                def fire_record(keys: list[str], now: float) -> None:
                    record_fires(conn, keys, now)

            return run_evaluate(
                message_text=message_text,
                group_id=gid,
                user_id=uid,
                now_ts=ts,
                rules=rules,
                fire_get=fire_get,
                fire_record=fire_record,
                global_baseline_probability=global_baseline_probability,
                extra_literal_triggers=extra_literal_triggers,
                sample_roll=sample_roll,
            )
        finally:
            conn.close()


class UpsertRuleResult(BaseModel):
    ok: bool = True
    rule_id: str


class EndRuleResult(BaseModel):
    ok: bool
    rule_id: str


class ListRulesResult(BaseModel):
    rules: list[TriggerRule]
    count: int = Field(description="Number of rules returned")


class PersonaHelperResult(BaseModel):
    ok: bool = True
    rule_id: str
    kind: str = Field(description="nickname | user_focus | topic_keyword")
    expires_at: int | None = Field(default=None, description="Unix ts when TTL ends; null for long-lived nickname")
    summary: str = Field(description="One-line Chinese recap for logs / self-check")


if _role in ("full", "admin"):

    @mcp.tool
    def upsert_rule(rule_json: str) -> UpsertRuleResult:
        """Insert or replace one trigger rule by rule_id (JSON object matching TriggerRule fields)."""
        rule = TriggerRule.model_validate(json.loads(rule_json))
        conn = _open_db()
        try:
            upsert_rule_row(conn, rule)
        finally:
            conn.close()
        return UpsertRuleResult(rule_id=rule.rule_id)

    @mcp.tool
    def list_rules() -> ListRulesResult:
        """Return all rules from SQLite (same DB as gate)."""
        conn = _open_db()
        try:
            rules = fetch_rules(conn)
        finally:
            conn.close()
        return ListRulesResult(rules=rules, count=len(rules))

    @mcp.tool
    def end_rule(rule_id: str) -> EndRuleResult:
        """Mark a rule as ended (no longer participates in matching)."""
        conn = _open_db()
        try:
            ok = end_rule_status(conn, rule_id)
        finally:
            conn.close()
        return EndRuleResult(ok=ok, rule_id=rule_id)

    @mcp.tool
    def remember_user_addressing_me(
        user_id: str,
        nickname: str,
        group_id: str = "",
        trigger_reason: str = "",
        response_hint: str = "",
        probability: float = 0.55,
        now_ts: int | None = None,
    ) -> PersonaHelperResult:
        """人格侧：记住“某个用户（可选某个群）用什么称呼叫我”，以便之后更容易注意到相关消息。

        对应 design 里称呼规则：用 keyword 命中称呼词；命中只表示“值得考虑接话”，你可以在最后仍选择不回复。
        group_id 留空表示全群通用（group_id=*）；填具体群号则只在那个群里对该用户生效。
        若称呼可能也在叫别人，请写 trigger_reason 说明歧义；response_hint 只影响生成语气，不参与是否命中。
        同一 (group_id,user_id) 反复调用会覆盖同一条规则（稳定 rule_id），用于改名或改概率。
        """
        rule = build_nickname_rule(
            user_id=user_id,
            nickname=nickname,
            group_id=group_id,
            trigger_reason=trigger_reason,
            response_hint=response_hint,
            probability=probability,
            now_ts=now_ts,
        )
        conn = _open_db()
        try:
            upsert_rule_row(conn, rule)
        finally:
            conn.close()
        gid = rule.group_id or "*"
        return PersonaHelperResult(
            rule_id=rule.rule_id,
            kind="nickname",
            expires_at=rule.expires_at,
            summary=f"已记录称呼「{nickname}」← 用户 {user_id}，作用域 group={gid}，keyword 命中后提高关注概率。",
        )

    @mcp.tool
    def renew_short_user_focus(
        user_id: str,
        group_id: str = "",
        duration_minutes: float = 5.0,
        probability: float = 1.0,
        trigger_reason: str = "",
        response_hint: str = "",
        now_ts: int | None = None,
    ) -> PersonaHelperResult:
        """人格侧：接下来一小段时间里，更留意某个用户在当前作用域里说的任何话（不限具体话题）。

        对应 design 的“短时人物关注窗口”：内部用 regex `.*` + user 维度实现；expires_at 从当前时间向后推 duration_minutes。
        默认 probability=1.0 且 time_distribution=poisson：窗口内有效概率随时间指数衰减（泊松式“间隔内回应机会”建模），可在参数里改 probability 或后续 upsert_rule 改分布。
        适合对话里判断“这个人现在特别值得听”；每次调用都会把同一条规则续期到新的截止时间（稳定 rule_id）。
        这不等于必须回复：门控仍可能放行，最终是否说话由你决定。
        """
        rule = build_user_focus_rule(
            user_id=user_id,
            group_id=group_id,
            duration_minutes=duration_minutes,
            probability=probability,
            trigger_reason=trigger_reason,
            response_hint=response_hint,
            now_ts=now_ts,
        )
        conn = _open_db()
        try:
            upsert_rule_row(conn, rule)
        finally:
            conn.close()
        gid = rule.group_id or "*"
        return PersonaHelperResult(
            rule_id=rule.rule_id,
            kind="user_focus",
            expires_at=rule.expires_at,
            summary=(
                f"已为用户 {user_id} 续期约 {duration_minutes:g} 分钟的全文关注窗口（group={gid}），"
                f"截止 unix_ts={rule.expires_at}。"
            ),
        )

    @mcp.tool
    def renew_temporary_topic(
        topic: str,
        group_id: str,
        duration_minutes: float = 5.0,
        probability: float = 1.0,
        trigger_reason: str = "",
        response_hint: str = "",
        now_ts: int | None = None,
    ) -> PersonaHelperResult:
        """人格侧：临时提高某个“话题词”（如「烧烤」）在**当前群**里的被关注概率，并在到期前可反复续期。

        默认 probability=1.0 且 time_distribution=poisson（窗口内随时间指数衰减）；可改 probability 或 upsert_rule 调整。
        对应 design 的临时话题：仅 keyword 子串匹配，不做 NLP。
        本工具是 helper：必须从当前消息上下文给出**明确的群号** group_id；不接受留空、不接受通配 *（全群话题请用 upsert_rule 自行建模）。
        rule_id 对 (group_id, topic) 做了稳定哈希，同一群同一话题字串重复调用即续期；返回含 rule_id，便于需要时 end_rule。
        """
        gid = group_id.strip()
        if not gid or gid == "*":
            raise ValueError(
                "renew_temporary_topic 需要明确的 group_id（当前群号），不接受留空或 *。全群通配请改用 upsert_rule。"
            )
        rule = build_topic_keyword_rule(
            topic=topic,
            group_id=gid,
            duration_minutes=duration_minutes,
            probability=probability,
            trigger_reason=trigger_reason,
            response_hint=response_hint,
            now_ts=now_ts,
        )
        conn = _open_db()
        try:
            upsert_rule_row(conn, rule)
        finally:
            conn.close()
        return PersonaHelperResult(
            rule_id=rule.rule_id,
            kind="topic_keyword",
            expires_at=rule.expires_at,
            summary=(
                f"已为话题词「{topic.strip()}」续期约 {duration_minutes:g} 分钟（group={rule.group_id}），"
                f"截止 unix_ts={rule.expires_at}，rule_id={rule.rule_id}。"
            ),
        )


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
