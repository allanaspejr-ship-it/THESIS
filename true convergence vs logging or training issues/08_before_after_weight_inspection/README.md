# 08 Before After Weight Inspection

Use this folder if the RBFNN implementation uses iterative optimization.

## What to Include

- Initial output weights before training.
- Final output weights after training.
- Difference or norm of weight change.

## What This Verifies

This checks whether parameters changed when the implementation is expected to update them iteratively.

## Good Sign

Weights changed during training, or the documentation clearly shows that closed-form optimization was used instead.

## Warning Sign

Weights remain identical before and after training even though iterative learning was expected.
