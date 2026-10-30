# Anonymization record

This repository is the authors' research repository, rewritten for **MLSys 2027 double-blind
review**. It keeps the **full commit history with the original commit timestamps**, so the
order of pre-registration freezes and analyses can be checked with `git log`. Everything below
was applied identically to **every commit**.

## Commit metadata
- Author and committer of every commit set to `Anonymous Author <anonymous@example.com>`.
- Author and committer timestamps unchanged to the second. Timezones normalised to UTC.
- Commit hashes therefore differ from the private repository. References to earlier commits
  inside files and commit messages (e.g. "frozen at `abc1234`") were rewritten to the
  corresponding anonymized hash, so every such reference resolves in this repository.

## Removed from all history
- `*.log` files: wandb console output containing account names, institution and absolute
  paths. Results are preserved in `experiments/*.md` and `*.json`.
- Internal planning documents not needed to reproduce or audit the research (paper outline,
  venue planning, the submission-deadline checker and its systemd units).

## Rewritten in all history
- Author names → "the first author" / "the second author"; account names, emails,
  institution and home-directory paths → neutral placeholders.
- `experiments/phase7_meta_analysis.md`: a conflict-of-interest paragraph present in the
  2026-08 versions (later replaced by the current authorship-neutrality statement) and an
  internal note about that change are redacted, because they identified an author's
  circumstances. The audit's scores and rules are unchanged.
- Three commit messages discussing an author's personal circumstances keep their subject
  line; the body is redacted and says so.

Nothing else was changed. Code, data, pre-registrations, declarations and results are as
committed. Third-party author names in `experiments/phase7_coverage/*.json` are literature
search results, not the authors.
