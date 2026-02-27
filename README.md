# Speeding Up Sequential MCMC for Large-Scale Sensor Data — MATLAB Reference Code

This repository accompanies the paper:

> **Speeding Up Sequential Markov Chain Monte Carlo Methods in the Context of Large Volumes of Data from Distributed Sensor Networks**  

published in the International Journal of Advanced Sensor Networks.

The full-text is available: https://doi.org/10.1155/dsn/6527524

### Abstract
Advances in digital sensors, digital data storage and communications have resulted in systems being capable of accumulating large collections of data. In the light of dealing with the challenges that large volumes of data present, this work proposes solutions to inference and filtering problems within the Bayesian framework. Two novel sequential Markov chain Monte Carlo (SMCMC) frameworks are proposed for non-linear and non-Gaussian state space models, able to deal with large volumes of data (or observations). These are SMCMC frameworks relying on two key ideas: (1) a divide-and-conquer type approach computing local filtering distributions each using a subset of the data, and (2) subsample the large data and utilize a smaller subset for filtering and inference. Simulation results highlight the large computational savings, that can reach 90% by the proposed algorithms when compared with a state-of-the-art SMCMC approach.

---

## Included files

- `Run.m` - Driver script: simulates measurements, partitions them into shards, and runs KF + all SMCMC variants
- `KF.m` - Baseline Kalman filter
- `SMCMC.m` - Generic sequential MCMC
- `SMCMC_AS.m` - Adaptive Subsampling Sequential MCMC (AS-SMCMC)
- `SMCMC_EP.m` - Expectation Propagation Sequential MCMC (EP-SMCMC)
- `MHstepG.m` - MH log-acceptance using the full measurement set
- `MHstepG_dist.m` - MH log-acceptance tailored to the EP-SMCMC.
- `MHSubLHD.m` - Subsampled-likelihood MH step for the AS-SMCMC.
- `ComputationLikelihoodG.m` - Utility to evaluate Gaussian likelihoods.
- `Gtraj.mat` - pre-defined data and parameters

## Requirements

- MATLAB R2020b or newer 
- Statistics and Machine Learning Toolbox (for `mvnrnd`, `mvnpdf`)
- Parallel Computing Toolbox (for `parfor`)

## License

Distributed under the **MIT License** (see `LICENSE`).
