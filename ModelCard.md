# Model Card: NN-UCB Bayesian Black-Box Optimizer

This model card follows the framework (adapted from Mitchell et al., 2019), emphasizing transparency for optimization approaches in ML benchmarking. 
It documents the NN-UCB approach for black-box optimization (BBO), tested on standard benchmarks.

## Overview
Name: NN-UCB Surrogate Optimizer
Type: Neural network-based surrogate model with Upper Confidence Bound (UCB) acquisition for sequential BBO.
Version: v1.0 (developed over 10 rounds; includes hyperparameter tuning via random search and Monte Carlo Dropout for uncertainty).

## Intended Use
Suitable tasks: Low-dimensional (2-8D) hyperparameter tuning or simulation optimization where evaluations are expensive/noisy (e.g., ML model selection, engineering design). 
Ideal for sparse-data regimes needing balanced exploration-exploitation.
Use cases to avoid: High-dimensional (>20D) problems due to sampling inefficiency; real-time optimization (e.g., robotics control) given 5-10 min/round compute; non-stationary functions without retraining hooks.

## Details
The strategy is a hybrid BO pipeline: initial random exploration builds data, followed by surrogate-guided exploitation. 
Techniques include data normalization (z-score for stability), NN surrogate (2 hidden layers, ReLU+dropout), MCD for epistemic uncertainty (50 samples), and UCB acquisition (μ + κσ) on 10k uniform candidates.

**Evolution across 10 rounds**:
Rounds 1-4 (Exploration): Uniform random sampling in [0,1]^d to cover space (~15% coverage); no surrogate.
Rounds 5-6 (Transition): Fit initial NN (50 epochs, κ=3.0 for high exploration); add MCD; n_candidates=5k.
Rounds 7-9 (Refinement): Full training (1k epochs); random search tuning (20 trials on hidden_dim/lr/dropout); κ=2.0; increase to 10k candidates for robust argmax.
Round 10 (Convergence): Considered κ=1.5 for exploitation; adaptive re-normalization per append; focus on gaps via PCA viz.
Each round appends the prior evaluation, retrains, and selects via UCB argmax—adapting to patterns like negative correlations in params.

## Performance
Tested on 8 standard BBO functions (2D: Branin, Hartmann6, Eggholder; 6D: Levy, Ackley, Rosenbrock, Griewank, Sphere; bounds [0,1]^d normalized). 
Ran 10 queries per function (post-initial 5 random points).

## Assumptions and Limitations
Assumptions: Black-box is stationary/smooth (low-frequency, NN-approximable); uniform priors valid for sampling; evaluations i.i.d. (no temporal drift).
Constraints/Failure modes: Compute-bound (MCD scales O(n_candidates × samples) → limits to 10k pts/round, missing narrow peaks); small-data overfitting (mitigated by dropout but risky <20 pts); biased toward clustered data, under-exploring voids if early points unlucky.

Strengths: Transparent (seeded, traceable); adaptive (evolves κ/candidates); uncertainty-aware for noisy funcs.
Limitations: Not scalable to high-D without dimensionality reduction; surrogate errors propagate in multimodal landscapes.

## Ethical Considerations
Transparency (e.g., seeded code, diagnostic plots, per-round logs) enables reproducibility—researchers can re-run to verify regrets or adapt (e.g., swap to EI acquisition). 
In real-world adaptation, it supports auditing biases (e.g., sampling gaps undervaluing diverse regions), promoting fairer optimization in equitable apps like resource allocation. 
Open-sourcing mitigates misuse by flagging avoidance cases.

Decision-making: Approach selects via UCB argmax on surrogate preds: normalize candidates → MCD for μ/σ → compute acq → pick max (traceable via prints). 
Strengths: Balances greediness with novelty; limitations: Relies on NN calibration, potentially conservative in flat landscapes.
