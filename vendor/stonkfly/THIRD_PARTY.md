# Neural-only vendoring and provenance

Only the neural implementation of [Stonkfly](https://github.com/nftechie/stonkfly)
at commit `78ef3e05ab0fa086032098558d893667068944a0` is vendored here.
Copyright notices and the upstream MIT license are preserved in `LICENSE`;
the original attribution document is retained as `THIRD_PARTY_UPSTREAM.md`.
Stonkfly's neural code derives from DOOMFLY, copyright 2026 nftechie and
DOOMFLY contributors. No exchange SDK, trading application, game artwork or
market-execution code is included or needed.

Local changes are limited to the data directory default, Windows DLL export,
and use of the project-local Zig C++ compiler. Neural equations, integration
scheduling, connectivity inclusion policy, retinal mapping, transmitter
assumptions and candidate memory rule remain the upstream implementation.
The snake's fixed three-action readout and feedback protocol live outside this
vendored package and are explicitly engineering choices.

MaleCNS v1.0 is downloaded separately from the official project:
https://male-cns.janelia.org/download/
Data license: Creative Commons Attribution 4.0 International,
https://creativecommons.org/licenses/by/4.0/ .
Credit: MaleCNS collaboration / HHMI Janelia FlyEM, University of Cambridge,
MRC Laboratory of Molecular Biology and Google Research, and their upstream
contributors. Cite Berg et al., *Sexual dimorphism in the complete connectome
of the Drosophila male central nervous system*, Cell (2026),
https://doi.org/10.1016/j.cell.2026.08.015 .

`neural/sources.lock.json` preserves the upstream source SHA-256 locks.
Preparation verifies every downloaded byte stream and derived graph arrays,
and writes a separate local preparation receipt. These checks establish data
provenance and numerical implementation, not biological validity or learning.
