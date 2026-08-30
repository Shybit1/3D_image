# Mathematical Method

## Depth → Height Proxy

Given RGB image `I`, the depth network produces inverse relative depth
`D = f_θ(I)`. Normalized to `[0,1]`:

```
h(x,y) = (D(x,y) - min(D)) / (max(D) - min(D))
```

## Relative DSM

```
Z_rel(x,y) = h(x,y) * V
```
where `V` is a chosen vertical display range (unitless).

## Absolute Calibration (Theil–Sen robust linear fit)

Given paired samples `{(h_i, z_i)}_{i=1..N}` (height proxy value, reference
elevation in meters) at co-registered pixel locations:

```
Z_abs = scale * h + offset
```

`scale` and `offset` are estimated as the **median** of all pairwise
slopes/intercepts between sample pairs (Theil–Sen estimator), which has a
breakdown point of ~29% (tolerates up to ~29% of samples being arbitrary
outliers before the estimate is corrupted) — substantially more robust
than ordinary least squares for noisy DEM/GCP reference data.

Fit quality is reported via:

```
residual_i = (scale * h_i + offset) - z_i
RMSE = sqrt(mean(residual_i^2))
MAE = mean(|residual_i|)
r = corr(h, z)
```

## Validation Metrics (predicted vs. reference elevation)

```
MAE  = mean(|Z_pred - Z_ref|)
RMSE = sqrt(mean((Z_pred - Z_ref)^2))
r    = corr(Z_pred, Z_ref)          (Pearson correlation coefficient)
```

Percentile errors (median, 90th, 95th percentile of `|Z_pred - Z_ref|`)
are reported to characterize the error distribution's tail, since RMSE
alone can be dominated by a small number of large errors.

## Optional Slope Error

Terrain/surface slope in degrees at each pixel, via finite-difference
gradient:

```
slope(x,y) = arctan( sqrt( (dZ/dx)^2 + (dZ/dy)^2 ) )   [converted to degrees]
```

computed with real pixel spacing (`pixel_size_m`) when available. Slope
RMSE compares predicted vs. reference slope fields the same way as
elevation RMSE.

## Uncertainty (Multi-Scale Ensemble Standard Deviation)

For `K` independent forward passes `{h_1, ..., h_K}` of the same model at
different input resolutions and with/without horizontal flip:

```
mean(x,y)   = (1/K) * sum_k h_k(x,y)
std(x,y)    = sqrt( (1/K) * sum_k (h_k(x,y) - mean(x,y))^2 )
uncertainty = normalize(std)   -> [0,1] for display
```

This is a real per-pixel statistic computed from `K` genuinely independent
forward passes (K=8 in the default config: 4 scales × 2 flip states) — not
a fixed or estimated confidence value.
