from __future__ import annotations

import json
import os
import time
from pathlib import Path

from fastmcp import FastMCP
from pydantic import BaseModel, Field

from attention.engine import evaluate_attention as run_evaluate
from attention.llm_helpers import (
    build_group_focus_rule,
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
        """门控：是否进入「考虑回复」。常用：message_text、group_id、user_id、now_ts（消息时间）；extra_literal_triggers 可传 bot 名/id 子串；rules_override_json 仅测试用；sample_roll 固定随机数调试用。"""
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
    kind: str = Field(description="nickname | user_focus | group_focus | topic_keyword")
    expires_at: int | None = Field(default=None, description="Unix ts when TTL ends; null for long-lived nickname")
    summary: str = Field(description="One-line Chinese recap for logs / self-check")


if _role in ("full", "admin"):

    @mcp.tool
    def upsert_rule(rule_json: str) -> UpsertRuleResult:
        """按 rule_id 写入或覆盖一条规则（JSON 与 TriggerRule 字段一致）。"""
        rule = TriggerRule.model_validate(json.loads(rule_json))
        conn = _open_db()
        try:
            upsert_rule_row(conn, rule)
        finally:
            conn.close()
        return UpsertRuleResult(rule_id=rule.rule_id)

    @mcp.tool
    def list_rules() -> ListRulesResult:
        """列出库里全部规则。"""
        conn = _open_db()
        try:
            rules = fetch_rules(conn)
        finally:
            conn.close()
        return ListRulesResult(rules=rules, count=len(rules))

    @mcp.tool
    def end_rule(rule_id: str) -> EndRuleResult:
        """将 rule_id 标为 ended，不再参与匹配。"""
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
        """记录称呼：user_id + nickname；group_id 空=全群。可填 trigger_reason / response_hint。重复调用同作用域会覆盖。"""
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
        """短时盯某个用户在本作用域内的所有发言（默认约 5 分钟，可调 duration_minutes）。probability=0 可短时压低关注；更细调用 upsert_rule。"""
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
    def renew_short_group_focus(
        group_id: str,
        duration_minutes: float = 5.0,
        probability: float = 1.0,
        trigger_reason: str = "",
        response_hint: str = "",
        now_ts: int | None = None,
    ) -> PersonaHelperResult:
        """短时盯整个群（必填 group_id）。例如有人 at 了还没说话想盯后续。可调 duration_minutes、probability。"""
        gid = group_id.strip()
        if not gid or gid == "*":
            raise ValueError(
                "renew_short_group_focus 需要明确的 group_id（当前群号），不接受留空或 *。"
            )
        rule = build_group_focus_rule(
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
            kind="group_focus",
            expires_at=rule.expires_at,
            summary=(
                f"已为群 {gid} 续期约 {duration_minutes:g} 分钟的全员消息关注窗口，"
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
        """当前群临时话题词（必填 group_id + topic），keyword 子串匹配；同词重复调用即续期。返回 rule_id 可 end_rule。"""
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
