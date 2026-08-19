# Contamination report

- SFT/RL split algorithm: `fixed-hash-stratified-v1`, seed `20261022`
- SFT trajectories: 881
- RL queries: 200
- Synthetic engineering dev queries: 16 (coverage-only, excluded from train/eval)
- SFT ↔ RL query-ID overlap: 0
- Dev ↔ SFT/RL query-ID overlap: 0
- SFT and RL are strict subsets of frozen `train_queries_1835`: yes
- Parent frozen data SHA-256: `cf2f694419f7a96e868d07507a0a26d648cfd885ce69cda7daae36b3d3d82e6e`
- Parent TravelPlanner/frozen-evaluation contamination result: 0 overlaps
- Parent protected comparisons: 2495600

The split does not introduce new queries. Its contamination guarantee follows from strict
query-ID lineage to the immutable, previously exact/constraint/fuzzy-checked 1,835-query
parent manifest. No Judge threshold was changed to construct these counts.
