# Permutation-test-based-on-age-matching-constraint
This tool calculates intraclass correlation coefficients (ICC) for neuroimaging data, comparing twin pairs against age‑matched random pairs using permutation testing. It is designed for task‑based fMRI data (COPEs) to assess the heritability or familial consistency of regional brain activation.

## Features
- Automatically identifies twin pairs and non‑twin subjects from the dataset.
- Computes ICC(3,1) (absolute agreement, single measurement) for each brain region within twin pairs.
- Generates age‑matched random pairs (with adjustable age tolerance) and repeatedly calculates ICC to build a null distribution.
- Performs permutation testing to obtain a p‑value indicating whether the twin ICC is significantly higher than random ICC.
- Reports Cohen's d effect size and significance flags.
- Supports multiple COPEs and aggregates results into a summary table.
- Handles missing data gracefully, skipping regions with insufficient observations.

## Dependencies
Install the required Python libraries before running the script:
```bash
pip install pandas numpy
