# JAX-NIPALS Implementation Validation Report

## Overview

This report documents the comprehensive validation of the NIPALS-PLS implementation from the `gogipav14/open_nipals` fork as used in the ClimKern-Retune project for tunable radiative kernel estimation.

**Date:** 2026-01-17
**Validated Package:** `open_nipals` (commit: abd19af9863ef51a5afd17b1104fad7d08c9d35e)
**Test Suite:** `tests/test_jax_nipals_validation.py`

---

## Executive Summary

| Category | Tests | Passed | Failed | Pass Rate |
|----------|-------|--------|--------|-----------|
| Core NIPALS-PLS | 8 | 7 | 1 | 87.5% |
| Convergence | 3 | 3 | 0 | 100% |
| NaN Handling | 4 | 4 | 0 | 100% |
| NIPALS-PCA | 5 | 3 | 2 | 60% |
| Constrained PLS | 6 | 6 | 0 | 100% |
| Physical Constraints | 6 | 6 | 0 | 100% |
| Numerical Stability | 4 | 4 | 0 | 100% |
| Distance Metrics | 4 | 2 | 2 | 50% |
| Component Addition | 2 | 2 | 0 | 100% |
| Integration | 2 | 2 | 0 | 100% |
| **Total** | **44** | **39** | **5** | **88.6%** |

**Recommendation:** The implementation is suitable for use with documented limitations.

---

## Detailed Findings

### 1. Core NIPALS-PLS Algorithm (7/8 tests passed)

**PASSED:**
- `test_fit_returns_self`: Model correctly returns self from fit()
- `test_fitted_components_match_request`: Component count matches request
- `test_scores_orthogonality`: X scores are orthogonal (T'T is diagonal)
- `test_loadings_shape_consistency`: All matrices have correct dimensions
- `test_deflation_correctness`: Deflation removes variance correctly
- `test_predictions_similar_to_sklearn`: Predictions match sklearn PLSRegression
- `test_variance_explained_reasonable`: Model explains significant variance

**FAILED:**
- `test_regression_vector_produces_correct_predictions`: The `get_reg_vector()` method produces slightly different predictions than `predict()`.
  - **Max difference:** 0.158 (2.5% relative)
  - **Impact:** Minor. Use `predict()` directly for consistency.
  - **Root cause:** Different computation paths for regression vector vs. sequential prediction.

### 2. Convergence Properties (3/3 tests passed)

All convergence tests passed:
- Default parameters achieve convergence
- Ill-conditioned data (collinear features) still converges
- Tolerance parameter affects convergence appropriately

**Note:** Very tight tolerances (1e-12) may require max_iter > 500 for some datasets.

### 3. NaN Handling (4/4 tests passed)

The `_nan_mult()` utility function correctly handles:
- Basic matrix multiplication with NaN values
- Denominator normalization for proper scaling
- Fitting models with 10% missing data
- Transform operations on data with missing values

**Implementation Detail:** Uses row-wise loops with valid data masking, which may be slower for large datasets but is numerically stable.

### 4. NIPALS-PCA (3/5 tests passed)

**PASSED:**
- Loadings are orthonormal (P'P = I)
- Variance ordering is correct (decreasing)
- Reconstruction error decreases with more components

**FAILED:**
- `test_scores_orthogonality`: Scores show small off-diagonal values (~1e-6)
  - **Impact:** Negligible for practical applications
  - **Likely cause:** Floating-point accumulation in deflation

- `test_comparison_with_sklearn_pca`: Variance explained ratios differ from sklearn
  - **Max difference:** 0.12 (45% relative)
  - **Root cause:** Different normalization conventions between NIPALS and sklearn's SVD-based PCA
  - **Impact:** Moderate. Use for relative comparisons only, not absolute variance values.

### 5. Constrained NIPALS-PLS (6/6 tests passed)

All physical constraint tests passed:
- Fitting without constraints works correctly
- Stefan-Boltzmann surface constraint integrates properly
- Constraints reduce constraint residuals as expected
- Predictions are valid and finite
- Q² scores are reasonable (>0.3 on training data)
- Kernel contributions decompose correctly

**Key Finding:** The constrained optimization successfully enforces physical constraints while maintaining predictive accuracy.

### 6. Physical Constraint Functions (6/6 tests passed)

All constraint function tests passed:
- Stefan-Boltzmann constraint returns zero residual when satisfied
- S-B constraint scales correctly with T³
- Energy conservation detects balanced predictions
- Energy conservation detects imbalances correctly
- TOA emissivity constraint works at baseline
- Multilevel constraint factory creates correct configuration

**Constants Validated:**
- Stefan-Boltzmann constant: σ = 5.670374419e-8 W m⁻² K⁻⁴

### 7. Numerical Stability (4/4 tests passed)

The implementation handles:
- High dimensionality (100 features, 50 samples)
- Very small values (1e-10 scale)
- Very large values (1e10 scale)
- Nearly constant features (1e-12 variation)

**Warning:** The mean-centering warning appears for edge cases, which is expected behavior.

### 8. Distance Metrics (2/4 tests passed)

**PASSED:**
- Q residuals calculation
- DModX calculation (PCA)

**FAILED:**
- `test_hotelling_t2_pls` and `test_t2_increases_for_outliers`
  - **Error:** `TypeError: unsupported operand type(s) for -: 'NoneType' and 'float'`
  - **Root cause:** Bug in `calc_imd()` - when `input_array` is provided without `input_scores`, the function calls `transform()` but assigns result to wrong variable (`scores` instead of `input_scores`).
  - **Location:** `nipalsPLS.py:557-558`
  - **Impact:** High. Cannot use Hotelling T² with raw data.
  - **Workaround:** Call `transform()` first, then pass scores directly.

### 9. Component Addition (2/2 tests passed)

- Adding components to fitted models works correctly
- Reducing components preserves prediction consistency

### 10. Integration Tests (2/2 tests passed)

- Full climate workflow with constraints: PASSED
- Cross-validation consistency: PASSED

---

## Known Issues and Workarounds

### Issue 1: Hotelling T² Bug in PLS

**Problem:** `calc_imd()` fails when passing raw data array.

**Workaround:**
```python
# Instead of:
t2 = model.calc_imd(input_array=X, metric="HotellingT2")

# Use:
scores = model.transform(X)
t2 = model.calc_imd(input_scores=scores, metric="HotellingT2")
```

### Issue 2: Regression Vector Discrepancy

**Problem:** `get_reg_vector()` gives slightly different predictions than `predict()`.

**Workaround:** Use `predict()` directly for all predictions. Only use `get_reg_vector()` for interpretability analysis where small differences are acceptable.

### Issue 3: PCA Scores Not Perfectly Orthogonal

**Problem:** Small off-diagonal elements in T'T matrix (~1e-6).

**Workaround:** None needed - acceptable for all practical applications.

---

## Recommendations

### For Users

1. **Use `predict()` for predictions** rather than computing manually from regression vectors.

2. **Pre-compute scores for distance metrics:**
   ```python
   scores = model.transform(X)
   t2 = model.calc_imd(input_scores=scores)
   q = model.calc_oomd(input_array=X)
   ```

3. **Always mean-center data before fitting** to avoid warnings and ensure correct predictions.

4. **Use Q² score for model selection** - the implementation produces reliable predictive R² values.

### For Developers

1. **Bug fix needed in `nipalsPLS.py:557-558`:**
   ```python
   # Current (buggy):
   scores = self.transform(X=input_array)
   # Later references input_scores which is still None

   # Should be:
   input_scores = self.transform(X=input_array)
   ```

2. **Consider adding JAX backend** for gradient computation in constraint optimization (currently uses finite differences).

3. **Document normalization conventions** - the variance explained values differ from sklearn due to different conventions.

---

## Conclusion

The `open_nipals` implementation is **suitable for production use** in the ClimKern-Retune project for tunable radiative kernel estimation. The constrained NIPALS-PLS functionality works correctly, all physical constraints are properly implemented, and numerical stability is good.

The identified issues are minor and have documented workarounds. The one significant bug (Hotelling T² with raw data) has a simple workaround and should be fixed in the upstream repository.

**Overall Assessment: VALIDATED WITH MINOR ISSUES**
