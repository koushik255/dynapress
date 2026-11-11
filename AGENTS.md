# Agent guidance

This is a personal research project, and the repository is **public**
(`github.com/koushik255/dynapress`). Move fast, and never publish secrets.

## Commit and push without asking

When a piece of work is finished and its tests pass (`uv run --frozen pytest -q`),
commit and push it to `main` yourself. You do not need to ask first. Follow these steps:

1. Stage only the files your task changed, using `git add <paths>`. Do not use `git add -A`
   or stage whole directories: several threads may share this checkout, and other
   threads' unfinished work must stay out of your commit.
2. Run this secret check on what you staged. If it prints anything, unstage the
   finding, keep the value out of the repository (read it from an environment variable),
   and tell the user. Do not push.

   ```bash
   { git diff --cached -U0 | grep -nE "^\+.*(ak-[A-Za-z0-9]{16,}|as-[A-Za-z0-9]{16,}|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16}|BEGIN [A-Z ]*PRIVATE KEY)"; git diff --cached --name-only | grep -iE "(^|/)(\.env.*|\.modal\.toml|.*\.pem|.*\.key|id_rsa.*)$"; } || true
   ```

   Also read the staged diff for anything the pattern misses, such as passwords,
   personal data, or private file paths.
3. Commit, run `git pull --rebase origin main`, and push. If the rebase conflicts with
   another thread's work, stop and ask the user instead of resolving it by force.

Credentials come from the environment and are never written to files: Modal reads
`MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` (Amp secrets in orbs, `~/.modal.toml` locally).
Large data and results (`data/lora/`, `results/lora/`) are gitignored; keep them that way.
