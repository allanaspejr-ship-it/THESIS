# 10 RBF Center Count Sensitivity

Place experiments using different numbers of RBF centers here.

## What to Include

- Results for multiple center counts, such as 5, 10, 20, 30, or other suitable values.
- Metrics for each configuration.
- Loss plots for each configuration if available.

## What This Verifies

This checks whether model performance responds to RBFNN capacity.

## Good Sign

Changing the number of centers changes performance or model behavior in a reasonable way.

## Warning Sign

All configurations produce exactly the same loss, metrics, and predictions, which may suggest a coding or logging issue.
