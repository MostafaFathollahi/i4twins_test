# code_assistant_conversations

Durable Markdown archive of AI coding-assistant chats — a reviewable log of
queries, answers, and decisions that survives context compaction.

## Layout

```
code_assistant_conversations/
└── <assistant>/                # one folder per assistant (e.g. claude)
    └── <slug>_<YYMMDD>_<id>.md  # one file per session
```

- **slug** — from the session's AI-generated title (or first prompt).
- **YYMMDD** — session start date (local time).
- **id** — first 6 chars of the session id; guarantees one file per session
  even if two sessions share a title/date.

## How files are generated

A [Stop / SessionEnd / PreCompact hook](../.claude/settings.json) runs
[`.claude/hooks/export-conversation.mjs`](../.claude/hooks/export-conversation.mjs),
which reads the assistant's on-disk transcript and regenerates the file. It is:

- **Automatic** — fires every turn, at session end, and before compaction.
- **Idempotent** — rewrites the whole file each run; no append/dedup logic.
- **Source-of-truth safe** — reads the append-only transcript, never the
  model's compacted in-memory context, so nothing is ever lost.

## File contents

YAML front matter (title, assistant/model, session id, branch, timestamps),
then the conversation as `🧑 You` / `🤖 Claude` turns. Thinking and tool
calls/results are preserved but collapsed inside `<details>` so it reads like a
clean chat. Detail level (`full` | `clean` | `expanded`) is set via the hook's
`--detail` flag.

## Adding another assistant

Write a new adapter in `export-conversation.mjs` that parses that assistant's
transcript into the same normalized `{ meta, turns }` shape — the Markdown
renderer and naming are shared. Output lands in `<assistant>/` automatically.

## Manual / backfill run

```bash
node .claude/hooks/export-conversation.mjs \
  --transcript <path-to-transcript.jsonl> \
  --root . --assistant claude
```
