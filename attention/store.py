from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from attention.models import TriggerRule

DEFAULT_BUSY_MS = 5000


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=DEFAULT_BUSY_MS / 1000.0)
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout = {DEFAULT_BUSY_MS}")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS trigger_rules (
            rule_id TEXT PRIMARY KEY,
            enabled INTEGER NOT NULL DEFAULT 1,
            priority INTEGER NOT NULL DEFAULT 0,
            source TEXT NOT NULL DEFAULT 'manual',
            group_id TEXT NOT NULL DEFAULT '*',
            user_id TEXT NOT NULL DEFAULT '*',
            pattern TEXT NOT NULL,
            match_mode TEXT NOT NULL DEFAULT 'keyword',
            case_fold INTEGER NOT NULL DEFAULT 1,
            term_type TEXT,
            starts_at INTEGER,
            expires_at INTEGER,
            time_window TEXT,
            status TEXT NOT NULL DEFAULT 'active',
            probability REAL NOT NULL DEFAULT 0.2,
            cooldown_sec INTEGER NOT NULL DEFAULT 0,
            max_hits_per_hour INTEGER NOT NULL DEFAULT 0,
            trigger_reason TEXT,
            response_hint TEXT,
            reason_visibility TEXT NOT NULL DEFAULT 'llm_only',
            created_at INTEGER,
            updated_at INTEGER
        );

        CREATE TABLE IF NOT EXISTS attention_fire (
            state_key TEXT PRIMARY KEY,
            last_fire REAL NOT NULL DEFAULT 0,
            hour_count INTEGER NOT NULL DEFAULT 0,
            hour_window_start REAL NOT NULL DEFAULT 0
        );
        """
    )
    conn.commit()


def _row_to_rule(row: sqlite3.Row) -> TriggerRule:
    d = dict(row)
    d["enabled"] = bool(d["enabled"])
    d["case_fold"] = bool(d["case_fold"])
    return TriggerRule.model_validate(d)


def fetch_rules(conn: sqlite3.Connection) -> list[TriggerRule]:
    """All rows (admin / audit). Includes disabled, ended, expired-by-status, etc."""
    cur = conn.execute("SELECT * FROM trigger_rules ORDER BY priority DESC, rule_id ASC")
    return [_row_to_rule(r) for r in cur.fetchall()]


def fetch_rules_for_eval(
    conn: sqlite3.Connection,
    *,
    group_id: str,
    user_id: str,
    now_ts: int,
) -> list[TriggerRule]:
    """Candidate rules for one evaluate call: enabled, status=active, in time bounds, scope matches.

    Daily ``time_window`` is still enforced in the engine (not expressed in SQL).
    """
    cur = conn.execute(
        """
        SELECT * FROM trigger_rules
        WHERE enabled = 1
          AND status = 'active'
          AND (starts_at IS NULL OR starts_at <= ?)
          AND (expires_at IS NULL OR expires_at >= ?)
          AND (group_id = '*' OR group_id = ?)
          AND (user_id = '*' OR user_id = ?)
        ORDER BY priority DESC, rule_id ASC
        """,
        (now_ts, now_ts, group_id, user_id),
    )
    return [_row_to_rule(r) for r in cur.fetchall()]


def get_fire_row(conn: sqlite3.Connection, state_key: str) -> tuple[float, int, float]:
    cur = conn.execute(
        "SELECT last_fire, hour_count, hour_window_start FROM attention_fire WHERE state_key = ?",
        (state_key,),
    )
    row = cur.fetchone()
    if row is None:
        return (0.0, 0, 0.0)
    return (float(row["last_fire"]), int(row["hour_count"]), float(row["hour_window_start"]))


def record_fires(conn: sqlite3.Connection, keys: list[str], now: float) -> None:
    if not keys:
        return
    conn.execute("BEGIN IMMEDIATE")
    try:
        for key in keys:
            cur = conn.execute(
                "SELECT hour_count, hour_window_start FROM attention_fire WHERE state_key = ?",
                (key,),
            )
            row = cur.fetchone()
            if row is None or now - float(row["hour_window_start"]) >= 3600:
                hc, ws = 1, now
            else:
                hc, ws = int(row["hour_count"]) + 1, float(row["hour_window_start"])
            conn.execute(
                """
                INSERT INTO attention_fire (state_key, last_fire, hour_count, hour_window_start)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(state_key) DO UPDATE SET
                    last_fire = excluded.last_fire,
                    hour_count = excluded.hour_count,
                    hour_window_start = excluded.hour_window_start
                """,
                (key, now, hc, ws),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def upsert_rule_row(conn: sqlite3.Connection, rule: TriggerRule) -> None:
    ts = int(time.time())
    cur = conn.execute("SELECT created_at FROM trigger_rules WHERE rule_id = ?", (rule.rule_id,))
    prev = cur.fetchone()
    created_at = int(prev["created_at"]) if prev and prev["created_at"] is not None else ts
    data = rule.model_dump()
    data["created_at"] = created_at
    data["updated_at"] = ts
    data["enabled"] = 1 if data["enabled"] else 0
    data["case_fold"] = 1 if data["case_fold"] else 0
    cols = [
        "rule_id",
        "enabled",
        "priority",
        "source",
        "group_id",
        "user_id",
        "pattern",
        "match_mode",
        "case_fold",
        "term_type",
        "starts_at",
        "expires_at",
        "time_window",
        "status",
        "probability",
        "cooldown_sec",
        "max_hits_per_hour",
        "trigger_reason",
        "response_hint",
        "reason_visibility",
        "created_at",
        "updated_at",
    ]
    placeholders = ", ".join("?" * len(cols))
    conn.execute(
        f"""
        INSERT INTO trigger_rules ({", ".join(cols)})
        VALUES ({placeholders})
        ON CONFLICT(rule_id) DO UPDATE SET
            enabled = excluded.enabled,
            priority = excluded.priority,
            source = excluded.source,
            group_id = excluded.group_id,
            user_id = excluded.user_id,
            pattern = excluded.pattern,
            match_mode = excluded.match_mode,
            case_fold = excluded.case_fold,
            term_type = excluded.term_type,
            starts_at = excluded.starts_at,
            expires_at = excluded.expires_at,
            time_window = excluded.time_window,
            status = excluded.status,
            probability = excluded.probability,
            cooldown_sec = excluded.cooldown_sec,
            max_hits_per_hour = excluded.max_hits_per_hour,
            trigger_reason = excluded.trigger_reason,
            response_hint = excluded.response_hint,
            reason_visibility = excluded.reason_visibility,
            updated_at = excluded.updated_at
        """,
        tuple(data[c] for c in cols),
    )
    conn.commit()


def end_rule_status(conn: sqlite3.Connection, rule_id: str) -> bool:
    cur = conn.execute(
        "UPDATE trigger_rules SET status = 'ended', updated_at = ? WHERE rule_id = ?",
        (int(time.time()), rule_id),
    )
    conn.commit()
    return cur.rowcount > 0


def maybe_seed_from_json(conn: sqlite3.Connection, json_path: Path) -> int:
    if not json_path.exists():
        return 0
    cur = conn.execute("SELECT COUNT(*) AS c FROM trigger_rules")
    if int(cur.fetchone()["c"]) > 0:
        return 0
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    if not raw:
        return 0
    n = 0
    for item in raw:
        upsert_rule_row(conn, TriggerRule.model_validate(item))
        n += 1
    return n
