Justification for Bayesian Black-Box Optimisation (BBO) Strategy

1. Why Neural Networks Were Chosen

Neural networks (NNs) were selected as the surrogate model due to their flexibility in modelling complex, high-dimensional, and non-linear objective functions. Unlike Gaussian Processes (GPs), which scale poorly beyond thousands of samples and struggle in high-dimensional scenarios, neural networks maintain computational tractability with mini-batch training and GPU acceleration. Their ability to approximate arbitrary functions (Universal Approximation Theorem) makes them well-suited for learning unknown objective landscapes and predicting promising regions for exploration.

In addition, neural networks enable feature learning rather than relying on predefined kernels, which is particularly beneficial when the underlying objective surface does not exhibit smoothness or stationarity. This adaptability complements a broader class of BBO problems where structural assumptions cannot be guaranteed.

2. Why Monte Carlo Sampling Was Used

Monte Carlo (MC) sampling was employed to approximate acquisition function behaviour and generate the next query point. Many acquisition functions such as Expected Improvement (EI), Probability of Improvement (PI), or Thompson Sampling require integrating over uncertainty. Neural networks do not naturally provide calibrated uncertainty the way GPs do, so MC sampling over network outputs introduces a practical and scalable method to estimate uncertainty.

MC-based exploration encourages a more diverse search, reducing the risk of premature convergence to suboptimal regions. This aligns well with high-dimensional, noisy, or non-smooth optimisation problems. It also integrates seamlessly with bootstrapped ensemble NNs or dropout-based Bayesian approximations.

3. Links to Academic Literature
	
  • Bergstra, J., Bardenet, R., Bengio, Y., & Kégl, B. (2011). Algorithms for Hyper-Parameter Optimization. Advances in Neural Information Processing Systems.
	
  •	Snoek, J., Larochelle, H., & Adams, R. (2012). Practical Bayesian Optimization of Machine Learning Algorithms. NIPS.
	
  •	Shahriari, B., Swersky, K., Wang, Z., Adams, R. P., & de Freitas, N. (2016). Taking the Human Out of the Loop: A Review of Bayesian Optimization. Proceedings of the IEEE.
	
  •	Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning. ICML.
	
  •	Rasmussen, C. E., & Williams, C. K. (2006). Gaussian Processes for Machine Learning. MIT Press.

These papers support the use of neural networks as scalable alternatives to GPs, highlight MC-based uncertainty estimation, and situate the approach within the broader Bayesian optimisation literature.

4. Comparison with Alternatives

Gaussian Processes (GPs)

Pros: well-calibrated uncertainty, mathematically elegant, strong performance in low dimensions.
Cons: poor scalability (O(n^3)), kernel limitations, difficulty in high-dimensional spaces.
NN Advantage: scalable, flexible, no strong smoothness assumptions.

Random Search

Pros: simple, parallelisable, no modelling assumptions.
Cons: extremely inefficient in complex objective landscapes.
NN/MC Advantage: guided exploration improves sample efficiency.

Evolutionary Algorithms (EAs)

Pros: robust, global search, no gradient requirements.
Cons: computationally expensive, slow convergence, many function evaluations required.
NN/MC Advantage: learns a surrogate to reduce evaluations, supports principled acquisition.

5. Full Citation List (APA Style)

Bergstra, J., Bardenet, R., Bengio, Y., & Kégl, B. (2011). Algorithms for hyper-parameter optimization. Advances in Neural Information Processing Systems, 24, 2546–2554.

Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation: Representing model uncertainty in deep learning. Proceedings of the 33rd International Conference on Machine Learning, 1050–1059.

Rasmussen, C. E., & Williams, C. K. (2006). Gaussian Processes for Machine Learning. MIT Press.

Shahriari, B., Swersky, K., Wang, Z., Adams, R. P., & de Freitas, N. (2016). Taking the human out of the loop: A review of Bayesian optimization. Proceedings of the IEEE, 104(1), 148–175.

Snoek, J., Larochelle, H., & Adams, R. (2012). Practical Bayesian optimization of machine learning algorithms. Advances in Neural Information Processing Systems, 25, 2951–2959.
