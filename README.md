# Capstone
Black-Box Optimization (BBO) Capstone Project

Section 1: Project Overview

The Black-Box Optimization (BBO) Capstone Project explores how to efficiently optimize an unknown and expensive-to-evaluate function when its analytical form is unavailable. This setting mirrors many real-world machine learning challenges, such as hyperparameter tuning, engineering design, and experimental optimization — where each query is costly, and insights must be drawn from limited data.

The core objective of this project is to maximize the output value of a hidden function using as few evaluations (queries) as possible. Since the function’s internal mechanism is unknown, we must rely entirely on observed inputs and outputs to guide the search for optimal regions.

This project builds practical skills in Bayesian Optimization (BO), Gaussian Processes (GP), and Support Vector Machines (SVMs) for surrogate modelling and decision-making. It also emphasizes balancing exploration (sampling uncertain regions) and exploitation (focusing on high-performing areas), a critical trade-off in real-world optimization.

From a career perspective, this project strengthens my expertise in data-driven decision-making under uncertainty, a capability valuable for roles in data science, applied machine learning, and AI research. It also enhances my ability to structure experiments, evaluate models iteratively, and communicate technical reasoning clearly through visualizations and documentation.

Section 2: Inputs and Outputs

The project operates on two core data components:
	•	Inputs: The model receives a NumPy array (initial_inputs.npy) representing query points in a multi-dimensional continuous space.
	•	Format: n x d (where n is the number of query points and d is the number of input dimensions).
	•	Each input vector defines a point at which the unknown function is evaluated.
	•	Constraints: Queries must be selected efficiently, as each evaluation is assumed to be computationally or financially costly.
	•	Outputs: Corresponding to each input, the model observes a scalar output value stored in initial_outputs.npy.
	•	Format: n x 1
	•	The output represents the performance score or objective function value at each input.
	•	The target of optimization is to maximize this output value.

Example: 
inputs.shape   # (50, 5)
outputs.shape  # (50, 1)

The surrogate model (e.g., Gaussian Process) learns a mapping from these input–output pairs to predict performance at new, untested query points.

Section 3: Challenge Objectives

The primary objective is to maximize an unknown, black-box function using as few queries as possible.
Key challenges include:
	•	The true function is unknown, non-linear, and possibly multi-modal.
	•	Only noisy observations are available.
	•	Each query carries a cost, limiting the number of allowable evaluations.
	•	There may be high-dimensional input spaces, which complicate the search landscape.

Thus, the project aims to develop a sample-efficient optimization strategy that identifies promising regions quickly and robustly. Performance is evaluated not only by the highest achieved value but also by how intelligently new points are selected.

Section 4: Technical Approach

1. Initial Strategy (Week 1–2): Gaussian Process-Based Bayesian Optimization

The initial approach relied entirely on Bayesian Optimization (BO) with a Gaussian Process (GP) as the surrogate model. The GP modelled the mean and uncertainty of the response surface, while the Expected Improvement (EI) acquisition function guided the next query selection.
Key parameters such as kernel type (Matern kernel), length scales, and noise levels were tuned heuristically based on warning diagnostics (e.g., expanding length_scale_bounds to avoid underfitting).
This stage focused on understanding the function landscape and achieving a balance between exploration and exploitation through acquisition tuning.

2. Intermediate Strategy (Week 3): Controlled Exploitation and Model Refinement

After several rounds of data collection, the model predictions became more stable, allowing greater emphasis on exploitation. The acquisition function was biased toward regions of previously high performance, though moderate exploration continued via non-zero acquisition noise.
Model interpretability and parameter diagnostics were improved by visualizing mean predictions and variance maps, leading to more controlled and confident decision-making.

3. Advanced Strategy (Week 4): Integrating Support Vector Machines (SVMs) for Region Classification

In the most recent iteration, I introduced an SVM classifier to enhance interpretability and reduce the number of required evaluations.
	•	The SVM model classifies input regions into “high-performing” and “low-performing” zones based on prior outputs.
	•	A soft-margin SVM allows tolerance for uncertainty near the decision boundary.
	•	A kernel SVM (RBF) captures non-linear separations, useful when the performance surface is non-linear.

This hybrid approach combines Bayesian Optimization’s uncertainty quantification with SVM-based regional classification, filtering potential queries before evaluation. The outcome is a more interpretable and efficient query process that maintains accuracy while reducing redundant sampling.

4. Exploration–Exploitation Balance

The combined model dynamically balances exploration and exploitation:
	•	Exploration: Guided by GP uncertainty and acquisition function variance.
	•	Exploitation: Reinforced by SVM filtering, which focuses searches on regions predicted to yield higher performance.

This design introduces human interpretability into a traditionally opaque optimization process, offering greater transparency and control in how decisions are made.

5. Current Optimisation Strategy: Neural Network Surrogate + Monte Carlo Acquisition

The latest and most effective approach in this project uses a neural network (NN) surrogate model to approximate the hidden function.
Key motivations:
	
	•	NNs handle non-linear surfaces effectively
	
	•	They scale better than Gaussian Processes in higher dimensions
	
	•	They allow smooth interpolation even when the function values are extremely small (common in this dataset)

The surrogate is a small fully-connected network: 

Input → Linear(64) → ReLU → Linear(32) → ReLU → Output(1)

This neural architecture provides sufficient capacity while avoiding overfitting under small datasets.

Acquisition Strategy: Monte Carlo Candidate Sampling

To propose the next query point, the system uses:
	
	1.	Random Monte Carlo sampling in the normalised search space
	
	2.	Surrogate predictions over these samples
	
	3.	Selection of the highest predicted value (“greedy exploitation”)

This works especially well when:
	•	Function evaluations are extremely sparse
	•	The response surface is smooth
	•	The model is updated every iteration

The general procedure:
	
	1.	Sample 5,000–10,000 random candidate points
	2.	Normalise them using the fitted StandardScaler
	3.	Predict outputs using the NN surrogate	
	4.	Select the candidate with the highest predicted value
	5.	De-normalise to obtain the real input value

This approach merges the principles of acquisition functions (e.g., Expected Improvement) with the flexibility of high-volume Monte Carlo sampling.

6. Future Directions and Planned extensions:

Ensemble neural surrogates to estimate predictive uncertainty

Trust-region–based BBO (e.g., TuRBO)

Bayesian neural networks for uncertainty-aware optimisation

Dimensionality reduction to isolate key influencing variables

Benchmark comparisons using Optuna, BoTorch, or Nevergrad


7. Future Structural Improvements and Detailed BBO Plan

The next stage of this Black Box Optimisation project focuses on transforming the current pipeline into a more scalable, uncertainty-aware and modular optimisation framework. The core priority is enhancing the surrogate modelling approach and making the acquisition process more intelligent and robust. The most significant structural improvement involves shifting from a single neural network surrogate to ensemble-based models, which provide better uncertainty estimates and more reliable guidance for choosing future query points. This aligns with state-of-the-art BBO research, where uncertainty estimation is essential for balancing exploration and exploitation.

Another major advancement is integrating trust-region strategies, such as TuRBO, which dynamically restrict the search to areas where the surrogate is reliable. This prevents wasted evaluations in poorly modelled regions and encourages systematic refinement around high-potential areas. The acquisition module will also be redesigned to support multiple strategies—Expected Improvement, Upper Confidence Bound, Thompson Sampling and enhanced Monte Carlo methods—allowing the optimiser to adapt as more data becomes available.

Future enhancements also include improved candidate-generation techniques such as Sobol sequences and Latin Hypercube Sampling, which provide better coverage of the search space than random sampling. Dimensionality reduction methods (e.g., PCA, autoencoders) will be introduced to identify and eliminate weak features, improving model stability and interpretability.

The longer-term plan follows a structured optimisation loop: preprocess data → train surrogate ensembles → compute acquisition scores → generate candidate points → select the next best query → update the model → repeat. Visualisations, including heatmaps, PCA projections and uncertainty maps, will make the process transparent and easier to evaluate.

Finally, benchmarking against established frameworks such as BoTorch, Optuna and Nevergrad will ensure the approach is competitive, reliable and grounded in modern optimisation standards. Together, these improvements create a clear roadmap for evolving the current system into a more powerful and research-aligned BBO engine.

