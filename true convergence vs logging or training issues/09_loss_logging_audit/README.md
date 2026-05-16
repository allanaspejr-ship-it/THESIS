# 09 Loss Logging Audit

Place logging and plotting checks here.

## What to Include

- Raw epoch-by-epoch training loss values.
- Raw epoch-by-epoch validation loss values.
- Screenshot or note showing where loss is recorded.
- Confirmation that each epoch appends a newly computed loss.

## What This Verifies

This checks whether the flat curve is caused by repeated logging of the same value.

## Good Sign

Loss values are recomputed each epoch and small changes are visible in raw values, even if the plot appears flat.

## Warning Sign

The exact same loss value is copied repeatedly, or the validation loss is accidentally assigned from the training loss variable.
