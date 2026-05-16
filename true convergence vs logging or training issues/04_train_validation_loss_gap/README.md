# 04 Train Validation Loss Gap

Place plots or tables comparing training loss and validation loss here.

## What to Include

- Training loss per epoch.
- Validation loss per epoch.
- Difference between validation loss and training loss.

## What This Verifies

This checks whether the validation behavior is consistent with generalization.

## Good Sign

Validation loss is slightly higher than training loss but remains stable and close to it.

## Warning Sign

Validation loss increases while training loss decreases, which may indicate overfitting. Exactly identical losses across all epochs may indicate a logging issue.
