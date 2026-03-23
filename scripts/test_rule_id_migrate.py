"""One-off local test for rule_id v3 migration; not run in CI."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from attention.store import connect, init_schema


def main() -> None:
    p = ROOT / "data" / "migrate_id_test.db"
    if p.exists():
        p.unlink()
    os.environ["BOT_ATTENTION_DB_PATH"] = str(p.resolve())
    conn = connect(p)
    init_schema(conn)
    conn.execute(
        """INSERT INTO trigger_rules (
            rule_id,enabled,priority,source,group_id,user_id,pattern,match_mode,case_fold,
            status,probability,cooldown_sec,max_hits_per_hour,reason_visibility,time_distribution
        ) VALUES ('addr:*:u1',1,40,'manual','*','u1','nick','keyword',1,'active',0.5,0,0,'llm_only','normal')"""
    )
    conn.commit()
    conn.close()

    c2 = connect(p)
    init_schema(c2)
    rid = c2.execute("SELECT rule_id FROM trigger_rules").fetchone()[0]
    assert rid == "long:nickname:*:u1", rid
    c2.close()
    p.unlink(missing_ok=True)
    print("migrate ok")


if __name__ == "__main__":
    main()
