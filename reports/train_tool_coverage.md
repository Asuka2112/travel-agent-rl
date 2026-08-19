# Train Search production coverage

## Root cause and remediation

The 200-query pilot used a three-row China/2026 demonstration inventory while the frozen
training domain is US/2022, and Teacher v2 described flight/drive choices but not train
selection. The production tool now uses a versioned US intercity simulator with recurring
date semantics, USD prices, explicit `simulator: true` provenance, deterministic sorting,
and a closed JSON schema. Teacher v3+ considers rail only when requested or plausible and
must disclose simulated evidence; it is not instructed to call train for quota purposes.

## Coverage result

- Coverage-only queries: 24 (frozen-train/eval overlap: 0)
- Successful trajectories: 24/24
- `train_search` calls: 24
- Successful train observations: 24
- Illegal arguments / unknown tool observations: 0
- Final answers explicitly disclosing simulation: 24/24
- Train observations followed by a completed plan: 24/24
- Inventory source: `travel-agent-rl/train-simulator-us@v1`
- Coverage manifest SHA-256: `51c32cdbd635b1f5fadcdfef470f86fcb03404e61d8c0e2a93cb72950df49185`

Two first-attempt coverage trajectories exposed a concurrent generic-tool-cache temporary
file collision. The cache writer now uses per-write unique temporary files and quarantines
legacy malformed entries; resume then completed both records with no unknown exception.
