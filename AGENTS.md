# Codex agent workflow

This repository uses OpenAI Codex. Its project agents are defined in `.codex/agents/*.toml`; do not use Claude Code or `.claude/agents` for this workflow. `CLAUDE.md` is a legacy-named project context document that may be consulted for factual project state and prior decisions. Its name or Claude Code wording is not an instruction to use Claude Code or Claude tools.

Before planning, editing, or auditing this project, read and apply [GUARDRAILS.md](GUARDRAILS.md) for research integrity, data safety, system boundaries, and completion criteria.

For ordinary implementation tasks, agree on scope and checkable completion criteria first. Use one `worker` subagent to implement the task and run the relevant checks, then send its diff to a separate `reviewer` subagent.

When the user asks to find or fix bugs or security vulnerabilities across the system, start with the read-only `system_auditor` subagent. Have it inspect application code, service code, infrastructure and CI configuration, and tests; never ask it to expose secret values or modify files. For a user-named bug, send that symptom and relevant code scope to `system_auditor` first. For audit, find, report, explain, or investigate requests without authorization to fix, return findings without edits. When fixing is already authorized, including applicable authorization from earlier in the conversation, send the evidence-backed findings to one `worker` subagent to fix, then use `reviewer` to inspect the fixes; do not ask for permission again. An explicit audit-only or no-edits instruction limits that task to findings even if earlier work authorized fixes.

The auditor reports only evidence-backed defects, including security vulnerabilities and smaller correctness issues, with severity, file and line, concrete cause and impact, and a safe verification path where possible. It does not edit files or report style preferences as defects. The reviewer reports every evidence-backed defect, including small and low-severity ones, with file, location, and a concrete explanation. It does not edit files or report style preferences as defects. It returns `NEEDS_FIX` if any defect remains and `PASS` only when none remain.

When the reviewer returns `NEEDS_FIX`, send the full report back to the same `worker` subagent. The worker fixes every in-scope finding, reruns relevant checks, and returns the updated diff. Have the reviewer inspect that revision and verify previous findings are resolved. Repeat this cycle without a fixed retry limit until `PASS`. Do not report completion while findings remain. If a finding requires a scope change or an unresolved decision, bring that specific decision to the user.

Preserve unrelated working-tree changes. Do not stage, commit, or publish unless the user asks.
