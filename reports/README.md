# Snapshot validation on 2026-10-04

- `tests-complete-2026-10-04.xml`: **261 passed, 0 failed, 0 skipped**.
  The two warnings are Torch sparse-tensor API notices. The earlier
  `tests-2026-10-04.xml` had one missing optional CRC dependency; that dependency
  was installed before the complete run.
- `model-verification-2026-10-04.json`: actual full-graph CUDA model load,
  checksum-locked trained parameters and optimizer memory, three inference
  scenes, and two continuation windows from a saved nonzero neural state.
  All checks passed. Integer pulses/actions/timing are exact; floating state
  uses the saved tolerance of 1e-6.

Environment: Windows, Python 3.12.13, Torch 2.11.0+cu128, CUDA 12.8,
NumPy 2.3.3. These checks establish correct packaging and reproducible model
execution, **not successful Snake learning**. Ordinary-game learning remains
unvalidated; the recorded frozen 24-game evaluation ate zero food.

GitHub Actions also passed **261 tests without skips**, recorded in
`tests-github-ci-2026-10-04.xml`. A fresh GitHub clone passed the same full suite
(`tests-fresh-github-clone-2026-10-04.xml`). The complete graph was downloaded
from the private Release and loaded with the cloned source: actual inference
and nonzero-state continuation passed again in
`model-from-github-verification-2026-10-04.json`.

`complete-graph-integrity-2026-10-04.json` verifies all 30 graph arrays and their
actual counts. `github-publication-2026-10-04.json` records the tested source
commit, remote Release asset sizes/digests, completed CI and download/replay.

GitHub Actions does not load the CUDA checkpoint
or prove full-graph game behavior. See the portable
[model audit](../docs/current-model-audit.md) for the behavioral results.
