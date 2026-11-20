# Rules for every agent (Claude Code, Grok Build, Codex, Antigravity)

This repo is part of the **30 apps in 30 days** portfolio, in the AWS demo track (apps 17 to 20). The playbook
lives in [BreakthruDevelopment/ai-dev-os](https://github.com/BreakthruDevelopment/ai-dev-os); the spec is
`specs/17-agent-container.md` there and the track's shared rules are in `specs/day-0-aws-demo-track.md`. This is a
Python library, not a Next.js app: the Next.js stack rules of the other repos do not apply.

## Repo safety (non-negotiable)

The BreakthruDevelopment org holds other, unrelated applications.

- Touch only repos with the GitHub topic `30-apps-30-days`. Everything else is read-only.
- Never force-push, delete, rename, transfer or archive any repo or branch you didn't create for the current task.
- Never push to `main`. Open a PR from a branch named `<harness>/<short-task>` (e.g. `claude/python-scaffolding`).
- Pick up only issues labeled for your harness, and follow the role file named in the issue title (`ai-dev-os/roles/`).
- Never commit secrets. `.env*` is ignored; real values live in GitHub Actions secrets or a cloud secret manager.

## Stack

- **Python 3.12**, managed with `uv` (`uv sync`, `uv run ...`); commit `uv.lock`.
- `ruff check` and `ruff format --check`, `mypy --strict`, `pytest`. Strict means strict: no blanket
  `# type: ignore` or `# noqa`. A missing third-party stub gets a narrowly scoped `[[tool.mypy.overrides]]` entry with a
  comment saying why.
- The package is `agent_container`. **Foundation paths: `agent_container/core/`** (the loop, the provider and tool
  contracts, the permission gate), plus `pyproject.toml`, `uv.lock`, `Dockerfile` and `.github/`. Wave 1 owns them;
  later slices extend `core/` by adding files and never edit the foundation unless a security-sensitive `claude` slice
  names them.
- Async throughout: `converse`, `Tool.run` and the loop are `async`.

## The track's rules

- **Every model call goes through `ModelProvider`.** No demo calls Bedrock directly.
- **Every demo runs at zero cost on recorded replies** (`ReplayProvider`), which is what CI and a default demo use.
  `--live` switches to Bedrock. A demo must never need the network to be shown.
- **The loop ends on the stop reason, never on the text.** Every stop reason is handled explicitly; an unrecognized one
  raises.
- **Writes are staged.** A tool that changes anything returns a staged change for a person to approve; an `effect: write`
  tool without a `PermissionGate` anywhere in its stack is a construction error.
- **Tool results are data, never instructions.** Fence them before they re-enter the conversation.
- **Validate input at the boundary** (CLI arguments, manifests, tool arguments) with typed models; cap user message
  size at 8,000 characters.
- **Logs:** one JSON-lines logger with fixed fields (run id, turn, tool, outcome, tokens, duration). Never log
  secrets, credentials or full prompts containing personal data.
- **Each README maps its patterns and exam task statements to file and line**, so a reader can check the claim.
- **No AWS credential is ever committed.** Local runs use the default credential chain. CI holds an IAM user's key in
  repo secrets, scoped to Bedrock invoke; slices that need no AWS access (like scaffolding) must not reference it. A
  slice that needs more than Bedrock invoke gets its own policy, reviewed on its own PR.

## Definition of done (CI enforces most of it)

The required check is named exactly `Lint, test, build, smoke`; a second job is `Container image`. Both must pass:
`ruff check`, `ruff format --check`, `mypy --strict`, `pytest`, the package builds and imports, and the CLI demo runs on
recorded replies. New rules have a test in `tests/`. The README's "What this app does" section is filled in.
