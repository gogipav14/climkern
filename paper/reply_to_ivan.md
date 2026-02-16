# Reply to Ivan

Hi Ivan,

Thanks for the detailed feedback -- you were right on several counts and it led to important corrections.

## Addressing your concerns

**Q² < 0 for individual kernels:** You were correct to flag this as suspicious. The negative Q² values were caused by implementation bugs in how we applied the kernels to real data -- specifically, missing pressure-thickness weighting (dp/10000), no Clausius-Clapeyron humidity normalization, no tropopause masking, and a sign convention mismatch between kernel "feedback" convention and CERES "observational" convention. There was also a units bug in HadGEM3-GA7.1 (pressure levels stored in Pa rather than hPa). After fixing these to match ClimKern's own frontend methodology, individual kernels achieve Q² = 0.48 (CloudSat) to 0.64 (HadGEM2) -- physically reasonable and consistent with what you'd expect from the kernel method.

**Overfitting concerns with kernel weighting:** Also a fair point. With the corrected kernel predictions, the PLS improvement over individual kernels is modest: best individual kernel Q² = 0.640, global PLS Q² = 0.652, best retune Q² = 0.655. The simple mean of all 11 kernels gets Q² = 0.626. So the kernels largely agree once properly applied, and PLS is doing variance reduction rather than the dramatic bias correction we originally claimed. The 76% RMSE reduction was an artifact of the buggy kernel predictions -- the real figure is ~6%.

**ML-based kernels from observational data:** This is exactly what Step 2 does, and I think you and Amal are on the right track with this direction. The approach we use -- PLS regression with SIMCA regime classification -- is an established chemometrics method (Wold et al., 2001) for exactly this kind of problem: learning latent structure from multivariate observations. It's essentially a constrained form of the machine learning approach you suggested, with the advantage that PLS provides interpretable latent variables (the scores map to physical modes like Planck response, water vapor feedback) and built-in diagnostics (VIP scores, Hotelling's T², DModX) for assessing model quality. Step 2 achieves Q² = 0.704 on real holdout data using 27 atmospheric state features, with physically correct feedback signs. The remaining ~30% of unexplained variance is primarily cloud radiative effects and dynamical contributions absent from the linear framework.

## What changed in the paper

The write-up I sent was definitely not scrutinized as I normally would -- I appreciate you catching the issues. Key corrections:

- Individual kernel Q² updated from "< 0" to "0.48--0.64" throughout
- RMSE reduction from "76%" to "~6%"
- Rewrote the discussion section on kernel performance (the "Why Q² < 0" section is now "Why does PLS improve over individual kernels?")
- All tables and figures regenerated with corrected kernel predictions
- Added explicit mention of the kernel application methodology (dp weighting, CC normalization, tropopause masking) as essential for proper results

The corrected paper and updated figures are on the branch. The Step 2 results (Q² = 0.704) are unchanged since they use raw atmospheric features, not the kernel prediction pipeline.

Happy to discuss the ML-kernel direction further -- I think there's a natural connection between PLS/SIMCA and the more modern ML approaches Amal is exploring.

Best,
Gorgi
