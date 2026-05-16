# 01 Actual vs Predicted Plots

Place plots here comparing actual hydropower generation values against RBFNN forecasts.

## What to Include

- Line plot of actual vs predicted values over time.
- Scatter plot of actual values vs predicted values.
- Residual plot if available.

## What This Verifies

This checks whether the model is producing meaningful forecasts even though the loss curve is flat.

## Good Sign

Predicted values follow the general trend and magnitude of the actual hydropower generation values.

## Warning Sign

Predictions are almost horizontal, shifted far away from actual values, or fail to follow the hydropower generation pattern.
