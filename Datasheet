## Datasheet for BO Evaluation Dataset

## Motivation
Creation purpose: Benchmark NN surrogate for BO on black-box function (e.g., low-reward landscape like tiny accuracies). Supports hyperparameter tuning research and exploration-exploitation analysis.
Supported task: Surrogate training/eval in BO; regret/coverage metrics.
Inappropriate uses: High-stakes apps without validation; generalizing to high-D/multimodal functions.

## Composition
Contents: X (2D to 8D params), y (scalar outputs to maximize, near-zero).
Size/Format: 8+ points
Split: 80/20 for surrogate tuning (seeded).
Gaps: Sparse (8+ pts, ~10% coverage); bias to mid-high quadrant; negative-skewed y.

## Collection Process
Query generation: Iterative BO—early: uniform random; later: NN-UCB on 5k-10k candidates (κ=2.0).
Strategy evolution: Append eval, retrain NN, normalize; started random, added tuning/MCD from round 5.
Time frame: 13 weeks
Transformations: Z-score norm on X/y; denorm for preds.

## Distribution
Availability: In script; export .npy; GitHub repo potential.

## Model Details
Name/Type/Version: NN-UCB Surrogate; FFNN with MCD; v1.0 (tuned: hidden=50, lr=0.01, dropout=0.2).
I/O: Input: 2D-8D vector; Output: 1D
Arch: 2→50(ReLU+Drop)→50(ReLU+Drop)→1; ~2.6k params.

## Intended Use
Suitable tasks: Low-D (≤8D) BO maximization; sparse-data uncertainty opt.
Avoid: High-D; noisy/non-stationary; real-time (5-10 min/round).

## Strategy & Techniques
Description/Evolution: Hybrid BO: Early random; later NN fit (1k epochs, MCD=50 samples) + UCB. 
Evolved: High κ=3.0 early →2.0; candidates↑ to 10k; tuning via 20-trial random search.

## Considerations
Constraints/Failures: Compute limits n_candidates=10k (misses peaks); small data overfit risk; non-general to 8D+.
Ethics: Sampling bias undervalues regions; ensure audits.

## Transparency
Repro/Adapt: Seeded code (42); prints/plots trace (e.g., argmax from MCD→UCB). 
