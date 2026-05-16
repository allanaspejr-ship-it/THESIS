# 07 Output Weights Check

Place output-layer weight inspection results here.

## What to Include

- Output weight values or summary statistics.
- Confirmation that weights were solved analytically or updated during training.
- Notes on the method used: least squares, ridge regression, pseudoinverse, or optimizer-based update.

## What This Verifies

This checks whether the RBFNN output layer learned a mapping from RBF activations to hydropower generation.

## Good Sign

Output weights are finite, non-empty, and consistent with the chosen training method.

## Warning Sign

Weights are all zero, NaN, infinite, unchanged when they should update, or not saved/computed.
