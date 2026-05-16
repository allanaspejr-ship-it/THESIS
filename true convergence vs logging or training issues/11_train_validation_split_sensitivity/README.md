# 11 Train Validation Split Sensitivity

Place results from different train-validation splits here.

## What to Include

- Results using different random seeds.
- Results using different split ratios, if appropriate.
- Time-aware split results if the dataset is chronological.

## What This Verifies

This checks whether the RBFNN result is stable across data partitions.

## Good Sign

Performance remains reasonably stable, with some expected variation.

## Warning Sign

Performance changes drastically across splits or remains exactly identical under all splits.
