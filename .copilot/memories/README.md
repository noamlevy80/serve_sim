# Copilot repo memory (portable copy)

These files mirror the agent's **repository memory** (`/memories/repo/`), which is
otherwise stored under VS Code's per-machine `workspaceStorage` and does NOT travel
with a `git clone`. Committing them here makes the knowledge portable.

Files:
- `overview.md` — codebase orientation / change playbooks.
- `arbiter.md` — arbiter internals + debugging history (bandwidth, KV offload, PDD, perf).

## On a new system

Open this repo in VS Code, then ask Copilot to reload the repo memory. It can copy
each file back into `/memories/repo/` (via the memory tool), e.g.:

> "Import `.copilot/memories/overview.md` and `.copilot/memories/arbiter.md` into repo memory."

When you make notable changes, keep these files in sync with the live repo memory.
