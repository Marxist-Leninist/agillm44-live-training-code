# GitHub-bound single-file trainer

Canonical development source: `trainer/agillm44.py`. Historical files in the repository root remain snapshots; do not deploy them by filename/date.

## Active path

`main` commit -> GitHub-hosted Trainer contracts -> SG pull controller -> SHA-pinned GB10 release -> checkpoint-safe apply -> verified deployment receipt.

GitHub never receives the SSH key, dataset tokens, runtime environment, or checkpoint tensors. Pull requests run CPU tests on disposable GitHub-hosted runners, not on the GPU server. The SG controller accepts only the current `main` SHA with a successful push-event run of `trainer-ci.yml` for this exact repository. A newer failed CI attempt supersedes an older success. Third-party actions are pinned by full commit SHA.

## How an agent ships a trainer change

1. Edit `trainer/agillm44.py` in a branch, not a production dated copy. Preserve AR, learned dynamic SAT-var, NAT, Direct56, and exact checkpoint resume.
2. In `release.json`, set `parent_source_sha256` to the currently deployed source hash and `source_sha256` to the edited file hash. Keep the relative source path and capability/resume contract unchanged.
3. Run `python3 -m unittest discover -s tests -v`. Open/merge the tested change to `main` (or push an authorised change); successful `main` CI is the deployment trigger.
4. Read the actual SG/GB10 deployment receipt. A green CI job is not a claim that the new trainer is running. `BOUND_IDENTICAL_NO_RESTART` means the exact source bytes already running were bound without a restart; `DEPLOYED` means an exact-state restart passed three advancing finite heartbeat checks.

The pull timer checks once a minute. A process/file identity mismatch blocks that release instead of reverting a newer agent's live trainer. Reconcile and rebase `parent_source_sha256` on the actual live source. Same-byte documentation/control commits do not restart training.

## Changed-source deployment

The controller parses the candidate CLI on GB10 without launching training. It preserves the existing command line, hot config, private environment, optimizer, LR clock, and save directory. It addresses the current process using a Linux pidfd after checking its PID incarnation, command line, file hash, and installed graceful SIGTERM handler. No force-kill fallback exists.

The old trainer saves and exits. Every checkpoint/sidecar/shard listed by the training pointer is checked for safe paths, size, and SHA-256, including AR/SAT/NAT heads and optimizer. The pointer must be newer than the stop request and not older than the observed training step. A hardlinked pre-trial checkpoint is retained locally against checkpoint rotation. The new code resumes this exact state under the existing single-trainer start lock. A failure in the trial triggers graceful stop and rollback to the retained pre-trial code/checkpoint; trial updates are discarded and the commit is marked rejected to avoid a retry loop. A failed clean stop or changed PID is reported, not force-killed.

The worker/control plane is installed separately and does not silently self-update from the trainer repository. Controller changes require tests and an explicit controller deployment. Transaction state and private environment stay on GB10 with mode 0600. Hardlink rollback pins require local retention management and are not offsite backups.

## What these gates do and do not prove

CPU checks cover top-level and folded-module syntax, capability markers, the strict-JSON nonfinite regression, release hashes, runtime mode/reset guards, checkpoint integrity, CI event selection, process identity protection, and no-restart adoption. Capability-marker tests are not a proof that arbitrary edits preserve learning behaviour; inspect actual changes and add behavioural tests.

Runtime acceptance covers clean exact-state resume and finite advancing training heartbeats. It does **not** benchmark throughput or establish improved held-out loss. Architecture/PC-ALM changes must include their own matched-compute GPU experiment before merging. Do not call the legacy heartbeat counter verified physical-work throughput when the runtime monitor marks its accounting unverified.

## Host locations

SG service/timer: `agillm44-github-cd.service` / `.timer`.
SG configuration: `/etc/agillm44-github-cd.json` (host-local; never commit credentials).
SG status/receipts: `/var/lib/agillm44-github-cd/`.
GB10 controller/status/releases: `/workspace/agillm44-github-cd/`.

Useful checks: `systemctl status agillm44-github-cd.timer`; `journalctl -u agillm44-github-cd.service -n 30`; read the SG `status.json` and the GB10 `status.json`.

Pausing the deployment timer does not stop training. Checkpoints remain outside Git; existing offsite checkpoint rescue/migration is independent of this pipeline.

Operational waiting receipts are explicit: `LIVE_SOURCE_DRIFT` preserves a separately launched experimental trainer; `WAITING_FOR_EXISTING_TRAINER` does not invent a cold start while another training lane is between runs. Neither means that the GitHub release is running. `BLOCKED_OR_RECOVERY_REQUIRED` carries the local reason and must not be presented as a successful deployment.
