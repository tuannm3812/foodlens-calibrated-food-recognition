# 8. Runtime Contract

## `/runtime/status`

Response shape:

- `classifier`
  - `status`: `"ready"` when both the checkpoint `model.json` names
    (`resnet50_ft_v2_best.pth` without a manifest) and `class_names.json` are
    present under the resolved artifact directory, otherwise
    `"missing_artifacts"`. A present but invalid `model.json` counts as
    present, so the failure surfaces as `classifier_load_error` rather than
    hiding behind `missing_artifacts`.
  - `artifact_status`: `"ready"` or `"mock"` mirror of classifier readiness.
  - `artifact_dir`: resolved directory path used for runtime.
  - `artifacts`: per-file status (`exists`, `size_bytes`, `path`) for:
    - `checkpoint` (`path: null` when `model.json` is invalid)
    - `model_manifest`
    - `class_names`
    - `calibration`
    - `decision_policy`
    - `hard_classes`
    - `confusion_pairs`
- `detector`
  - `status`: `"ready"` when `ultralytics` import is available, otherwise `"missing_dependency"`.
  - `dependency`: fixed string (`"ultralytics"`).
  - `dependency_available`: boolean import check result.
  - `weights_path`: resolved detector path.
    - explicit override via `FOODLENS_DETECTOR_WEIGHTS`
    - repo/root discovery fallback
    - otherwise package default path string
  - `weights_found`: whether `weights_path` exists on disk.
  - `weights_source`: `"environment"`, `"auto_discovered"`, or `"ultralytics_default"`.
  - `label_filter`:
    - `mode`: `"all"`, `"configured"`, or `"default"`
    - `labels`: configured labels when mode is `"configured"`
- `multi_food`
  - `mode`:
    - `"live_yolo_classifier"` when classifier and detector dependency are both available.
    - `"detector_only_classifier_fallback"` when detector is available but classifier artifacts are not.
    - `"demo_fallback"` when detector dependency is unavailable.
  - `detector_status`:
    - `"live_yolo"` for full classifier+detector path
    - `"live_yolo_classifier_fallback"` for detector-only path with classifier fallback labels
    - `"fallback_demo"` when both paths are unavailable
- `decision_layer` — the decision layer this process routes with. Reading it
  never loads the model.
  - `source`: `"loaded_runtime"` once `load_runtime()` has run (the first real
    prediction), reporting what the process has **cached**; `"artifact_files"`
    before that, reporting what the artifact files would load. A loaded process
    keeps its cached values when the files change, until it restarts.
  - `policy`: `auto_confidence`, `suggest_confidence`, `margin_threshold`.
  - `hard_class_count`, `confusion_pair_count`.
  - `fingerprint`: SHA-256 of the canonical decision layer
    (`app/backend/policy_fingerprint.py`): the three thresholds as floats, the
    sorted hard classes and the sorted `[actual, predicted]` pairs. Equal
    fingerprints mean identical routing.
  - `error`: present instead of the fields above when `source` is
    `"artifact_files"` and the files cannot be read.
- `model` — the classifier this process serves and the calibration it applies.
  Reading it never loads the model and never hashes the checkpoint or the
  class names.
  - `source`: `"loaded_runtime"` once `load_runtime()` has run, reporting the
    identity **cached at load time**; `"artifact_files"` before that,
    reporting what the artifact files would load. As with `decision_layer`, a
    loaded process keeps reporting the model it loaded until it restarts.
  - `manifest`: `"model.json"`, or `"legacy_default"` when there is none.
  - `architecture`, `checkpoint`, `model_name` (and `model_run` when the
    manifest records one).
  - `checkpoint_sha256`: SHA-256 of the exact bytes `load_runtime()` loaded,
    computed once at load. `null` before load, with a `note` saying so.
  - `temperature`: once loaded, the temperature the runtime **caches** and
    scales every prediction's logits by — the same value a prediction
    response reports as `temperature`. Before load, the temperature
    `calibration.json` would give (the built-in default when it is missing
    or unreadable), and the `note` says so. A loaded process keeps its cached
    temperature after `calibration.json` changes, until it restarts.
  - `class_names_sha256`: SHA-256 of the ordered class-name list the runtime
    maps output indices to, computed once at load (`null` before load). The
    canonical form is the list's compact, ASCII-escaped JSON in UTF-8
    (`class_names_sha256()` in `app/backend/artifacts.py`), so a reordered
    list hashes differently.
  - `error`: present instead of the identity fields when `source` is
    `"artifact_files"` and `model.json` is invalid.

## Model manifest

`app/artifacts/model.json` is optional and names the classifier to serve:

```json
{
  "architecture": "convnext_tiny",
  "checkpoint": "convnext_tiny_continued_best.pth",
  "model_name": "a3b_convnext_tiny",
  "model_run": "results/accuracy_phase1/a3b_convnext_tiny_continued_224"
}
```

- `architecture`: `"resnet50"` (head in `fc`) or `"convnext_tiny"` (head in
  `classifier[2]`); both use the project head and the same eval transform.
- `checkpoint`: a file name inside the artifact directory.
- `model_name`: what live responses report (`model_name` on
  `/predict/image`, `model` on multi-food responses).
- `model_run` (optional): the training run the checkpoint came from. The
  deploy script writes it on promotion and uses it to keep a later
  policy-only deploy from pairing this model with another model's policy.

**Absent manifest:** exactly the pre-manifest behaviour — `resnet50`, from
`resnet50_ft_v2_best.pth`, reported as `resnet50_ft_v2`.

**Invalid manifest** (unreadable JSON, not an object, a missing or unknown
key, an empty value, an unsupported architecture, or a checkpoint that is a
path rather than a file name): `load_runtime()` raises `ModelManifestError`
and requests fall back with `fallback_reason: classifier_load_error`, like an
unreadable `class_names.json`. It never falls back to ResNet50.

Demo and mock responses (`artifact_status: "mock"`, or any `fallback_reason`)
keep reporting `resnet50_ft_v2`: their fixed predictions are ResNet-era demo
data, not output of the configured model, and `fallback_reason` marks them.

## Multi-food response fields

`/predict/multi-food/*` responses use `MultiFoodPredictionResponse` with:

- `detector_status`:
  - `"fallback_demo"` for deterministic mock data
  - `"live_yolo_classifier_fallback"` when detector proposals exist but classification is unavailable
  - `"live_yolo"` for full live path
  - `"live_yolo_whole_image_fallback"` when no detections pass filters and full image is used as one region
- `artifact_status`: `"ready"` or `"mock"` to indicate classifier-artifact dependence
- `fallback_reason`: present when `artifact_status` is `"mock"` or detection path is constrained

Common fallback reasons:

- `"missing_artifacts"`
- `"missing_classifier_artifacts"`
- `"detector_runtime_unavailable"`
- `"detector_inference_error"`
- `"invalid_image"`
- `"no_detector_regions"`
- `"classifier_load_error"`
- `"classifier_inference_error"`
- `"video_mock"`

## Tuning-artifact resilience

`calibration.json`, `decision_policy.json`, `hard_classes.json`, and
`confusion_pairs.json` each have a documented default. If one of them is
missing, or present but unreadable (truncated write, invalid JSON, undecodable
bytes), the runtime degrades to that default rather than failing the request.
Degradation is logged at warning level, naming the file and the read error —
serving default thresholds while an operator believes their tuned policy is
live is a failure mode of its own, so it is never silent.

`class_names.json` is the exception: it has no safe default, since serving it
empty would build a classifier head with zero classes. A missing or unreadable
`class_names.json` still fails `load_runtime()` outright, which callers catch
and turn into the `"classifier_load_error"` demo fallback above. A present
`model.json` is strict in the same way (see "Model manifest").

## URL ingestion behavior

- Image URL endpoint requires public direct image URLs and validates host/IP safety.
- YouTube URL endpoint requires public URL and local optional dependencies:
  - `yt-dlp`
  - `ffmpeg`
- URL input errors return `400`.
- Missing media dependencies return `503`.
- Ingestion failures return `400` for user-facing URL/media issues.

## Detector weights

Live detection needs `yolo11n.pt` — the YOLO11-nano checkpoint from
Ultralytics, 5.6 MB. It is gitignored by the `*.pt` rule, so a fresh clone
does not have it, and `weights_found` reports `false` until it is present.

Resolution order, reflected in `weights_source`:

1. `"environment"` — the `FOODLENS_DETECTOR_WEIGHTS` environment variable.
2. `"auto_discovered"` — `yolo11n.pt` found at the repo root.
3. `"ultralytics_default"` — the package default path. Ultralytics downloads
   the checkpoint on first use when it resolves here.

With no weights and no `ultralytics` install, `/predict/multi-food/*` serves
the deterministic demo fallback (`detector_status: "fallback_demo"`). That is
the expected state in CI, which does not install the detector extra.

## Prediction evidence

A decision policy is fitted to one model's temperature-scaled confidences, so
the prediction CSVs it is fitted on mean something only together with what
produced them. `kaggle/a3b_rescore/rescore_predictions.py` therefore writes,
next to each `<output>.csv`, a sidecar **`<output>.csv.evidence.json`**
(schema `foodlens.prediction_evidence`, version 1) recording:

- `checkpoint`: its path and the **SHA-256 of the exact bytes loaded**;
- `architecture`;
- `class_names`: the SHA-256 of the **ordered** class list used for the
  output indices (same canonical form as the status block's);
- `temperature`: the effective value and its source;
- `preprocessing`: the eval transform — `Resize((224, 224))` with bilinear
  interpolation, `ToTensor`, then normalisation with mean
  `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]` — the transform
  `load_runtime()` also applies;
- `producer`: the rescorer's path and SHA-256;
- `self_check`: passed, or skipped (stated, with no recorded value), and the
  accuracies compared;
- `predictions`: the output CSV's file name, row count, split and
  **SHA-256**, which binds the sidecar to exactly that file.

The sidecar is written under the CSV's own rule: atomically, and only once the
self-check passes. An existing sidecar is refused without `--overwrite`.

`scripts/recalibrate_decision_layer.py` verifies each sidecar's CSV hash
against the bytes it read and rejects a mismatch; rejects a `temperature`
column that holds more than one value or differs from the sidecar's; and
requires the fit and eval evidence to agree on checkpoint, architecture,
class order, temperature and preprocessing. Evidence on one split only is
rejected. The verified sidecars are recorded in `derivation_provenance.json`
(schema version 3) under `predictions.fit.evidence` and
`predictions.eval.evidence`. A run from CSVs with no sidecar still completes,
but records no evidence, and the deploy script refuses its policy.
`scripts/compare_provenance.py` treats the model-describing evidence
(checkpoint, architecture, class names, temperature, self-check) as an
expected difference between two models' runs; the evidence schema,
preprocessing and producer hash must match.

The consumer side, and the checks below, live in
`scripts/prediction_evidence.py`. The rescorer keeps its own copy of the
writer because it is a Kaggle `code_file` and must stay self-contained; tests
pin the two together.

### Regenerating evidence for older runs

Prediction CSVs written before 2026-10-10 have no sidecar, and policies
recalibrated from them record no evidence; the deploy script refuses them.
Evidence is never written by hand: re-score with the rescorer, under a **new**
output name so nothing existing is overwritten, then recalibrate from the new
CSVs:

```bash
RUN=results/accuracy_phase1/a3b_convnext_tiny_continued_224
for SPLIT in val test; do
  python kaggle/a3b_rescore/rescore_predictions.py --results-dir $RUN \
    --arch convnext_tiny --split $SPLIT --data-dir data/food41/images \
    --output $RUN/${SPLIT}_predictions_rescored_<date>.csv
done
python scripts/recalibrate_decision_layer.py --results-dir $RUN \
  --fit-split val --eval-split test \
  --fit-predictions-file $RUN/val_predictions_rescored_<date>.csv \
  --eval-predictions-file $RUN/test_predictions_rescored_<date>.csv \
  --output-dir $RUN/decision_layer_closure_<date>
```

For a checkpoint outside the run directory, pass `--checkpoint` as an
absolute path (a relative one is resolved inside `--results-dir`), and
`--expected-top1`/`--expected-top5` when the run has no `<split>_metrics.csv`.

This was done for both models on 2026-10-10, into
`*_predictions_rescored_2026-10-11.csv` and `decision_layer_closure_2026-10-11/`.
All four regenerated CSVs are byte-identical to the
`*_predictions_rescored.csv` files the 2026-10-10 policies were fitted on, and
every regenerated `decision_policy.json`, `hard_classes.json`,
`confusion_pairs.json` and band-metric table is identical to the
`decision_layer_closure_2026-10-10/` one. The policy deployed today is
therefore the A3b `decision_layer_closure_2026-10-11/` policy byte for byte,
now with evidence bound to the deployed checkpoint and calibration. ResNet50's
val self-check compared against its earlier re-score (77.90% / 92.36%), as
that split has no independently recorded figure.

## Deployed decision policy

`app/artifacts/` is gitignored, so the policy the backend serves is not in
version control. This section records which one is deployed and how to verify
it.

**Current state (promoted 2026-10-10 UTC):** champion **A3b ConvNeXt-Tiny**
(`model_name: a3b_convnext_tiny`) with its own corrected decision policy from
`results/accuracy_phase1/a3b_convnext_tiny_continued_224/decision_layer_closure_2026-10-10/`.
Model and policy were deployed together in one promotion (decision D-013).

| Field | Value |
| --- | --- |
| Architecture / checkpoint | `convnext_tiny` / `convnext_tiny_continued_best.pth` |
| Checkpoint SHA-256 | `01d98554244d022d7a86d6158715dbf41364c51ee8ec8c2c7da325f5f852c8af` |
| Temperature (`calibration.json`) | 0.88435298204422 |
| `auto_confidence` / `suggest_confidence` / `margin_threshold` | 0.70 / 0.35 / 0.05 |
| Hard classes / confusion pairs | 11 / 40, derived from A3b's validation predictions |
| Source provenance SHA-256 | `63d8befae476f430441406b1bf09970f48ea2993732c13b28c57a020f0521a52` |
| `model.json` SHA-256 | `dbbb04e6201e1a1c7854a1271e943c1c7347871f2c96c0807e7f5c912dc2210c` |
| `calibration.json` SHA-256 | `5560c41b7b821f7d5acbd04f24ee08feee3c6b22878161c4bfd74ab79f121df3` |
| `class_names.json` SHA-256 (unchanged) | `90ae276a0af258b6f9f9d0fcd0d9e1b59119cb91e428abd1607c13f5c8ac5884` |
| `decision_policy.json` SHA-256 | `377a98891a443faa3f2e76c3b9a1fbb9f963ec93447f693be6397b8a1673e465` |
| `hard_classes.json` SHA-256 | `f383e78fd5d7d427dc3df2a32197ca6bd5550c517d08cf1de658d51aa16ec101` |
| `confusion_pairs.json` SHA-256 | `642099c2c7bdd5a82b580bbfc43c5411241b23e6a312c8a88e76b2b826205427` |
| Decision-layer fingerprint | `39672f7878c13f211ecb9b252c9991aa307ec07cca10174c5fca437b4d97f7e4` |

`decision_policy.json` hashes the same as ResNet50's because both validation
searches selected the same thresholds; the hard classes and confusion pairs
differ by model.

Measured on the held-out test split: auto-accept 66.63% at 96.66% top-1, review
2.19%; test ECE 0.0556. **Verification of the deployment:** after a restart,
`--verify-live` passed, reporting `convnext_tiny`, `a3b_convnext_tiny` and the
checkpoint hash above; and 400 test images posted over HTTP to `/predict/image`
(as PNG) matched the offline closure run on top-1 class and decision band for
400 of 400, with no fallbacks.

**Rolling back to ResNet50 FT-V2.** The promotion backed up the complete
previous state — ResNet50 checkpoint, calibration, class names, policy files and
provenance — in `app/artifacts/replaced_20261010T015403Z_alsevy57/`. One command
restores it, then restart and verify:

```bash
python scripts/deploy_decision_policy.py --target app/artifacts \
  --restore app/artifacts/replaced_20261010T015403Z_alsevy57
```

The restore path was rehearsed on a copy before the real promotion: the
restored copy matched the pre-promotion state byte for byte, and `--verify-live`
passed for ResNet50 after a restart.

**Previous state (2026-10-09 to 2026-10-10):** ResNet50 FT-V2 with its corrected
policy — decision-layer fingerprint `ed6e970d2b12640f0648c0810ed58f0f759cce0cf2b11c33cc95d3403a54fcd9`,
auto-accept 61.20% at 94.58% on test, review 2.54%, temperature 0.958111.

The replaced legacy policy (margin 0.40, 15 hard classes, 30 pairs, written
2026-05-30 by Notebook 5 with the withdrawn methodology) is preserved in
`app/artifacts/replaced_20261009T224436Z/`. Measured the corrected way, it
auto-accepted 59.02% at 94.83% and routed 11.52% of images to review — not the
58.02% / 96.47% it had been credited with.

**Deploying a policy.** Three steps, in order:

1. `scripts/deploy_decision_policy.py --source <run dir>` (add `--dry-run` to
   validate only).
2. Stop or restart **every** API process that serves `app/artifacts/`.
3. `scripts/deploy_decision_policy.py --verify-live <API base URL>`.

A policy-only deploy requires the policy's producer evidence (see "Prediction
evidence") to match the **served** model: the architecture `model.json` names
(the legacy ResNet50 default when there is none), the SHA-256 of the actual
checkpoint file it names, the target's class order and the temperature in the
target's `calibration.json`, compared exactly. A target without
`calibration.json` is refused, since the backend would silently serve a
built-in default. When `model.json` records a `model_run`, the predictions
must also live inside it. The record names the served model — under `model`,
the same key promotion and restore use — and the hashes of its checkpoint,
`calibration.json` and `class_names.json` (`served_model_files_sha256`), so
`--verify-live` can detect later drift.

Never copy recalibration outputs by hand: the recalibration script writes
`decision_policy.json` as a one-element list, while `read_policy()` expects a
dict. A verbatim copy makes `load_runtime()`
raise, and its `except Exception` turns every request into a demo fallback with
`fallback_reason: classifier_load_error` — while `artifact_status` still
reports `ready`. The deploy script converts the format, validates labels against
the served model's class names, stages and verifies the files before touching
the target, backs up what it replaces (including the previous
`deployment_provenance.json`) in a collision-free `replaced_<stamp>_<suffix>/`
directory, installs each file atomically, reads the result back through the
backend's own readers, and writes `app/artifacts/deployment_provenance.json`
with the decision-layer fingerprint. Any failure after installation starts is
rolled back: the target and its provenance are left byte-for-byte as they were,
and the new backup directory is removed.

**Why the restart.** `load_runtime()` caches the model, policy, hard classes
and confusion pairs in the process on first use, so a running API keeps
serving the old model, decision layer and temperature after the files change.
The deploy record's `artifact_files_verified` therefore covers the files only.

`--verify-live` first checks the target's files against
`deployment_provenance.json`, where it records them: the decision-layer
fingerprint; the model's architecture, checkpoint name and checkpoint hash
(from the record's `model` entry, which every kind of deployment writes — older
policy-only records are read from `served_model`); every recorded checkpoint,
`calibration.json` and `class_names.json` hash (`deployed_files_sha256` or
`served_model_files_sha256`); and the recorded temperature. A file changed
after deployment fails here, including a checkpoint swapped for different
weights of the same architecture after a policy-only deploy,
before the service is probed, even if a restarted service would agree with
it. It then sends one synthetic image to `/predict/image` (any
`fallback_reason` fails the check) and reads `/runtime/status`. Three checks
are reported **independently**, each as `passed` or `FAILED` with every
problem found:

- **routing** — `decision_layer`, as `loaded_runtime`, with a fingerprint
  equal to the target files';
- **model** — the `model` block, as `loaded_runtime`, with the target's
  `architecture`, `model_name` and `checkpoint_sha256` (the target's
  checkpoint is hashed by the script; the service reports the hash it
  computed at load);
- **calibration** — the probe response's `temperature` equals the `model`
  block's cached `temperature`, both equal the target's validated
  `calibration.json` exactly, and the block's `class_names_sha256` equals the
  target's. Temperature is in neither the model identity nor the routing
  fingerprint, so a service with the right model and policy but a stale
  temperature fails only here.

The command exits 1 on any failure, naming both values of every mismatch. A
service running code from before the `model` block, or before it reported
`temperature`, fails the check: restart it on the current code.

**Promoting a model.** A decision policy is fitted to one model's
confidences, so a model and its policy deploy together, never apart:

1. `scripts/deploy_decision_policy.py --source <policy run> --model-run <training run>
   --architecture <resnet50|convnext_tiny> --model-name <name>` (add
   `--checkpoint <file>` when the run holds more than one `.pth`, and
   `--dry-run` to validate only).
2. Stop or restart **every** API process that serves the target.
3. `scripts/deploy_decision_policy.py --verify-live <API base URL>`.

Step 1 installs the checkpoint, `calibration.json`, `class_names.json`,
`model.json` and the three policy files as one atomic unit, with the same
staging, backup, post-install check and rollback as a policy deploy. Before
the target is touched it refuses:

- a policy whose `derivation_provenance.json` names fit or eval predictions
  outside `--model-run`, or whose prediction files no longer match their
  recorded hashes (for example, the champion's policy with A3b's checkpoint);
- a policy whose producer evidence does not match what is being installed:
  the checkpoint's SHA-256, `--architecture`, the class order and the
  temperature in the `calibration.json` being installed, compared exactly
  (tolerance zero; JSON round-trips the value bit for bit). Directory
  membership is not evidence, so different valid weights of the same
  architecture, or a changed temperature, are refused even when the fitted
  CSVs are unchanged. A policy with **no** evidence is refused with
  instructions to regenerate it (see "Regenerating evidence for older
  runs"); no flag skips the check. The comparison runs right after the
  checkpoint is shown to fit the architecture, so a checkpoint that does not
  load reports that instead;
- a `class_names.json` that differs from the target's in content **or order**
  — a reordered list would mislabel every prediction;
- a checkpoint that does not load into `--architecture` with zero missing and
  zero unexpected keys (checked on the staged copy).

The backup holds every file the promotion replaces, the previously served
checkpoint, the previous `deployment_provenance.json`, and a
`backup_record.json` that also records which files did not exist (for
example, no `model.json` before the first promotion). After a promotion,
policy-only deploys check that the policy was fitted on the predictions of the
run `model.json` names, and that its evidence matches the served checkpoint
and calibration.

**Rolling back: `--restore`.**
`scripts/deploy_decision_policy.py --restore <target>/replaced_<stamp>_<suffix>`
puts back the state that backup recorded: each backed-up file is restored and
each file recorded as absent (a promoted checkpoint, `model.json`) is
removed. It runs through the same machinery: it checks every backup copy
against its recorded hash and refuses a damaged backup, checks that the
restored checkpoint fits its architecture, backs up the state it replaces (so
a restore can itself be undone), installs atomically, re-reads the result and
rolls back on any failure. `deployment_provenance.json` then records the
restore, embedding the restored provenance as `restored_deployment_provenance`;
every other file is byte-identical to the backed-up state. Backups made before
`backup_record.json` existed (such as `replaced_20261009T224436Z/`) are
refused, since the files that did not exist then are unknown. A restore puts
back a recorded state exactly and does not re-check producer evidence; a
policy-only deploy afterwards does. Rehearsed on a copy on 2026-10-10: after
restoring the ResNet50 backup, ResNet50's `decision_layer_closure_2026-10-11/`
policy validated against it and A3b's was refused. Restart every API process
and run `--verify-live` after a restore, as after a deploy.

**Verifying the deployed state.** Compare `deployment_provenance.json` against
the hashes above, then run `--verify-live` against the running API. The
2026-10-09 record's `backend_verified: true` (the field's name before it became
`artifact_files_verified`) meant the files read back correctly in a fresh
process, not that a running service had been checked, and that record carries
no fingerprint, so `--verify-live` compares against the files alone. On
2026-10-10 a freshly started API passed `--verify-live`, reporting the
decision-layer fingerprint in the table above as `loaded_runtime`.

The 2026-10-09 deployment was also verified end to end: 400 test images posted
to `/predict/image` (sent as PNG to avoid re-encoding) agreed with the closure
run's offline routing on top-1 class and decision band for 400 of 400, with no
fallbacks.
