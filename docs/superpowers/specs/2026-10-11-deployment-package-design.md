# Deployment Package Decomposition (Design)

Date: 2026-10-11
Status: Approved by Codex review (agent log, "Codex closure review and
deployment decomposition discussion"); follows decision D-015.

## 1. Why

`scripts/deploy_decision_policy.py` is 1,726 lines. It grew from 613 when model
promotion, restore and live verification were added, and its last four
correctness findings all sat at seams between operations that build or read
deployment records their own way. The latest was a model identity written under
`served_model` by one operation and read only from `model` by another.

The goal is boundaries that make that class of defect hard to write: one shared
definition of identity and records, one installer, and operations that compose
them.

## 2. Target structure

A package in the product tree, `app/deployment/`, leaving
`scripts/deploy_decision_policy.py` as a thin CLI. Master standard §1 separates
CLI helpers from core logic, and `app/` is already a legitimate root
(`0_coding_standards.md`).

| Module | Owns | Must not |
| --- | --- | --- |
| `errors.py` / `paths.py` | `DeployError`, `display`, `sha256`, `now_utc`, file-name constants | — |
| `identity.py` | Manifest interpretation, checkpoint fit, class-order and temperature validation, policy validation, adapters to the backend readers, producer-evidence comparison | Decide which model and policy belong together |
| `records.py` | The record schema: building, validating, and normalising historical records on read | Touch the target directory or load a model |
| `install.py` | Staging, backup, atomic `os.replace` install, post-install callback, rollback | Know what a model or policy is |
| `policy.py`, `promote.py`, `restore.py` | One operation each: choose what is installed, compose `identity`, `records` and `install` | Re-implement validation or record handling |
| `verify_live.py` | Live checks against a running API | Import any write operation |

Dependency direction: operations → `install`, `records`, `identity` → `paths`,
`errors`. The backend may share pure identity helpers but must never import
deployment orchestration.

## 3. Two phases, kept separate

**Phase 1 — mechanical extraction. Behaviour unchanged.** CLI flags, exit codes,
stdout/stderr text, JSON record contents, backup layout, restore behaviour and
legacy-record reading all stay identical. Evidence: fixture runs of every
operation captured before the move, and compared after it.

**Phase 2 — schema on write. A deliberate behaviour change, stated as one.**
Every new record carries `schema_version` and an `operation` discriminator, a
required common model identity, calibration and routing fields, and
operation-specific evidence and backup fields. The base record is validated
before the backup is created. The finalised record (with backup metadata) is
validated before any target file is replaced. A missing model identity fails
there instead of producing a record a reader would silently skip. Reading keeps
a separate normalisation path for historical records, with explicit limits for
fields they never captured. Old evidence is never invented from today's target.

## 4. Guardrails

- **Failure injection must still bite.** Tests patch `build_model`,
  `verify_model_files`, `verify_through_backend` and `datetime` on the CLI
  module. After extraction, a re-export does not make those patches reach calls
  inside the owning module, so the test would pass while injecting nothing.
  Patch the owning module, or keep an explicit injection seam. Every
  failure-injection test must also assert that its injected fault was actually
  reached (for example, a call counter). Green tests whose injection became a
  no-op are not acceptable.
- Behavioural assertions keep their meaning. Imports and patch targets may
  change; what is asserted may not.
- The real `app/artifacts/` is never modified. Rehearsals use copies.
- The current deployment must still pass `--verify-live` after each phase.
