# bot-attention (FastMCP + SQLite)

Implements the **recall layer** described in the Astrbot attention design draft: keyword/regex + scope + time + probability + cooldown. No NLP/LLM inside the gate. (`design.md` is intentionally **not** tracked; keep your own local copy or symlink outside the repo.)

Persistence is **SQLite** with **WAL** and a busy timeout so several MCP worker processes can share one file safely. Astrbot can attach a **gate** process (evaluate only) and Cursor / LLM tooling can attach an **admin** process (rule writes); both must use the **same** `BOT_ATTENTION_DB_PATH`.

## Run (stdio)

From repo root (deps in `.venv`):

```bash
.venv\Scripts\python.exe -m attention.server
```

If the `fastmcp` CLI is on your PATH:

```bash
fastmcp run attention/server.py
```

## Environment

| Variable | Purpose |
|----------|---------|
| `BOT_ATTENTION_DB_PATH` | SQLite file (default `data/attention.db`) |
| `BOT_ATTENTION_MCP_ROLE` | `full` (default), `gate`, or `admin` |
| `BOT_ATTENTION_RULES_PATH` | Optional legacy JSON seed when the DB has zero rules |

### Two MCP entries (recommended)

Same command, different env:

1. **Gate (Astrbot)** — `BOT_ATTENTION_MCP_ROLE=gate` → tool `evaluate_attention` only.
2. **Admin (LLM / ops)** — `BOT_ATTENTION_MCP_ROLE=admin` → `upsert_rule`, `list_rules`, `end_rule`.

Point both at the same `BOT_ATTENTION_DB_PATH` so rules and cooldown counters stay aligned.

## Tools

- `evaluate_attention` — `message_str`, `group_id`, `user_id`, `timestamp`, optional `extra_literal_triggers`. **Gate merge:** among hits, **one winner** by (specificity → **priority** → `rule_id`), **not** `max(probability)`; result includes **`gate_rule_id`**. `matched_rules` is sorted the same way — **first entry is the winner**. Cooldown write applies to the winner only when `should_consider`.
- Each matched rule in the result includes **`ttl_remaining_sec` / `ttl_prompt_hint`** when `expires_at` is set (non-permanent), and **`adjusted_probability`** (after `time_distribution`).
- Rule field **`time_distribution`**: `normal` (fixed `probability`); **`linear`** (factor = remaining TTL / window length); **`poisson`** — models attention roughly as **events over the TTL interval** (how much "one more worth-engaging beat" remains plausible as time passes); implementation uses **exponential damping** `exp(-λ·elapsed/window)` (Poisson / memoryless flavor), λ≈2.5. Persona short-term helpers (`renew_short_user_focus`, `renew_temporary_topic`) default to **`poisson`** with **`probability=1.0`** (decay is entirely from the exponential time factor). Legacy rows that stored linear decay under the old `poisson` label are migrated to `linear` once (`PRAGMA user_version`).
- `upsert_rule` — JSON object for one `TriggerRule` row (`rule_id` is primary key).
- `list_rules` — dump all rules from SQLite.
- `end_rule` — set `status=ended` for a `rule_id`.

### Persona helpers (admin / full)

参数以平铺为主；工具 docstring 只写常见用法（原理见仓库内 `engine` / README 门控合并说明）。

默认 **priority**：长期称呼 **40**；短期话题 **62**、短期群关注 **72**、短期盯人 **82**（短期整体高于长期，合并时易压过 `remember_*`）。

Persona 写入的 **`rule_id` 带类型前缀**，长期与短期不可能撞 key：`long:nickname:{group}:{user}`、`short:user_focus:{group}:{user}`、`short:group_focus:{group}`、`short:topic_kw:{group}:h…`。旧库中带 `addr:` / `focus_*` / `topic_kw:` 的行在每次连接时会**幂等**迁到新前缀（并同步 `attention_fire` 的 `state_key`）。

| Tool | Intent |
|------|--------|
| `remember_user_addressing_me` | 长期称呼 keyword。 |
| `renew_short_user_focus` | 短时盯某用户全文；可调 `probability`、`duration_minutes`。 |
| `renew_short_group_focus` | 短时盯全群；**必填群号**。 |
| `renew_temporary_topic` | 当前群话题词 TTL；**必填 group_id + topic**。 |

## Dev check

```bash
.venv\Scripts\python.exe scripts\test_sqlite_flow.py
```


## Config Template

Cursor 里常见写法是项目根目录的 `.cursor/mcp.json`，或 **Settings → MCP** 里同结构的 JSON。顶层键名是 **`mcpServers`**（注意拼写）。

把 `YOUR_REPO_ROOT` 换成本仓库绝对路径（Windows 下 JSON 里反斜杠要写成 `\\`）。两个进程共用同一个 `BOT_ATTENTION_DB_PATH`，只有 `BOT_ATTENTION_MCP_ROLE` 不同。

```json
{
  "mcpServers": {
    "bot-attention-gate": {
      "command": "YOUR_REPO_ROOT\\.venv\\Scripts\\python.exe",
      "args": ["-m", "attention.server"],
      "cwd": "YOUR_REPO_ROOT",
      "env": {
        "BOT_ATTENTION_DB_PATH": "YOUR_REPO_ROOT\\data\\attention.db",
        "BOT_ATTENTION_MCP_ROLE": "gate"
      }
    },
    "bot-attention-admin": {
      "command": "YOUR_REPO_ROOT\\.venv\\Scripts\\python.exe",
      "args": ["-m", "attention.server"],
      "cwd": "YOUR_REPO_ROOT",
      "env": {
        "BOT_ATTENTION_DB_PATH": "YOUR_REPO_ROOT\\data\\attention.db",
        "BOT_ATTENTION_MCP_ROLE": "admin"
      }
    }
  }
}
```

本地开发若只想开一个进程，可只配一条并把 `BOT_ATTENTION_MCP_ROLE` 设为 `full`（或省略该变量，默认即为 `full`）。