# Spatio-Temporal Modal Interval Regression (STMIR) in MATLAB

This repository provides a MATLAB implementation of a **Spatio-Temporal Modal Interval Regression (STMIR)** method.
The workflow is script-based: **run a single main script** and adjust parameters inside it.

## Contents

* **Main script**: runs the full STMIR pipeline (the method entry point).
* **Three core functions**:

  1. **Conditional distribution estimation**.
  2. **Conditional MI estimation** (extract the conditional modal interval from the estimated conditional distribution).
  3. **ADMM prox for quantile loss** (proximal operator used in the ADMM solver for the quantile-loss formulation).

## Files

* `main_stmir.m` — main script (run this)
* `cdfe_b.m` — conditional distribution estimator
* `cmie_b.m` — conditional MI estimator
* `per_prox.m` — prox operator for quantile loss used in ADMM

## Requirements

* MATLAB
* Require **Statistics and Machine Learning Toolbox**

## How to Use

You only need to modify the parameters in `main_stmir.m` and run the script.

Before running, provide the input data vectors:

* `X` — training set covariate vector for the first spatial covariate
* `Y` — training set covariate vector for the second spatial covariate
* `T` — training set temporal covariate vector, scaled to the range [0, 1]
* `Z` — training set response vector
* `test_X` — test set covariate vector for the first spatial covariate
* `test_Y` — test set covariate vector for the second spatial covariate
* `test_T` — test set temporal covariate vector, scaled to the range [0, 1]
* `test_Z` — test set response vector

Typical parameters you may want to adjust include:

* **Coverage level (`alpha`)**: the MI coverage level (e.g., `alpha = 0.5` for the 50% MI).
* **Number of polynomial pieces in the X-direction (`J`)**: the number of polynomial segments over the domain of the first spatial covariate.
* **Number of polynomial pieces in the Y-direction (`K`)**: the number of polynomial segments over the domain of the second spatial covariate.
* **Domain endpoints in the X-direction (`start_value_x`, `end_value_x`)**: the left and right endpoints of the first spatial covariate domain.
* **Domain endpoints in the Y-direction (`start_value_y`, `end_value_y`)**: the left and right endpoints of the second spatial covariate domain.
* **Spatial polynomial degree (`d2`)**: the polynomial degree used for the spatial spline over the X- and Y-directions.
* **Temporal polynomial degree (`d1`)**: the polynomial degree used for the temporal direction.
* **Spline smoothness (`rho`)**: the smoothness/continuity level imposed on the spline.
* **Smoothing parameter (`lambda`)**: the regularization strength (larger values typically yield smoother estimates).
* **ADMM iterations (`iter`)**: the number of ADMM iterations (maximum iteration count) used by the solver.
* **mCWC penalty parameter (`eta`)**: the penalty strength parameter in mCWC.
* **Number of data points for KDE (`n_kde`)**: the maximum number of data points used for KDE-based conditional distribution estimation when estimating the quantile levels. This parameter is used to reduce computation time. If computation speed is not a concern, it can be set to `inf`.
* **Colormaps (`map_width`, `map_upper`, `map_lower`)**: the colormaps used for the interval width, upper bound, and lower bound videos. You can change them by writing the names of the colormaps, such as `'turbo'`, `'parula'`, or `'jet'`.

## Notes on Spline Knots (Important)

In this implementation, spline knots are selected **uniformly**.

⚠️ **To avoid a singular matrix**, when choosing knots, make sure that:

* **Between any two neighboring knots in the X-direction, there are at least two data points.**
* **Between any two neighboring knots in the Y-direction, there are at least two data points.**

## Output

Running `main_stmir.m` produces three MP4 videos showing the estimated interval width, upper bound, and lower bound over time, saves the fitted coefficient vector `c_star` to `stmir_coefficient.mat`, and reports the interval quality metric **mCWC** value.
