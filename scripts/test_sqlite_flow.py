from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ["BOT_ATTENTION_DB_PATH"] = str(ROOT / "data" / "test_flow.db")

from attention.engine import evaluate_attention as ev
from attention.models import TriggerRule
from attention.store import connect, fetch_rules, get_fire_row, init_schema, record_fires, upsert_rule_row


def main() -> None:
    p = Path(os.environ["BOT_ATTENTION_DB_PATH"])
    if p.exists():
        p.unlink()
    conn = connect(p)
    init_schema(conn)
    upsert_rule_row(conn, TriggerRule(rule_id="r1", pattern="foo", probability=1.0))
    conn.close()

    conn = connect(p)
    rules = fetch_rules(conn)
    r = ev(
        message_text="foo bar",
        group_id="g",
        user_id="u",
        now_ts=1,
        rules=rules,
        fire_get=lambda k: get_fire_row(conn, k),
        fire_record=lambda keys, now: record_fires(conn, keys, now),
        global_baseline_probability=0.0,
        extra_literal_triggers=None,
        sample_roll=0.01,
    )
    assert r.should_consider and r.matched_rules[0].rule_id == "r1"
    conn.close()
    print("ok")


if __name__ == "__main__":
    main()
