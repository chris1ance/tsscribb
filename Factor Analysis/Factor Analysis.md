# From PCA to Factor Analysis

Factor Analysis can be viewed as an extension (or modification) of Principal Components Analysis (PCA). While PCA tries to reduce dimensionality by explaining the variance of observed features with a small number of components, Factor Analysis introduces a more explicit notion of "noise" or "measurement error" and focuses on reproducing correlations among features.

In PCA, we often write each observed vector $\mathbf{x}_t \in \mathbb{R}^p$ (for $t = 1, \ldots, T$) as a combination of a small number $q$ of principal components. However, PCA does not explicitly model noise per feature, and it may try to explain even the random fluctuations.

By contrast, Factor Analysis posits that each observed data vector $\mathbf{x}_t \in \mathbb{R}^p$ is generated from a small set of $q$ latent factors $\mathbf{f}_t$ plus additive feature-specific noise. Hence the model captures shared structure (through a few factors) and also acknowledges individual "measurement error" or idiosyncratic variation for each feature.

# Definitions

**Static Factor Models**

In a static factor model, the relationship between observed variables and latent factors is contemporaneous, with no lag structure.

Let $\mathbf{x}_t \in \mathbb{R}^p$ be a vector of $p$ observed variables at time $t$. A static factor model represents $\mathbf{x}_t$ as:

$$\mathbf{x}_t = \mathbf{W} \mathbf{f}_t + \boldsymbol{\epsilon}_t$$

where:
- $\mathbf{f}_t \in \mathbb{R}^q$ is a vector of $q$ latent factors $(q < p)$
- $\mathbf{W} \in \mathbb{R}^{p \times q}$ is the matrix of factor loadings
- $\boldsymbol{\epsilon}_t \in \mathbb{R}^p$ is a vector of idiosyncratic disturbances

**Dynamic Factor Models**

Dynamic factor models extend static factor models by incorporating time-dependency structures allowing factors to affect variables across different time periods. Mathematical Definition:

$$\mathbf{x}_t = \mathbf{W}_0 \mathbf{f}_t + \mathbf{W}_1 \mathbf{f}_{t-1} + \ldots + \mathbf{W}_s \mathbf{f}_{t-s} + \boldsymbol{\epsilon}_t$$

The factors $\mathbf{f}_t$ often follow a time series process, typically a VAR:

$$\mathbf{f}_t = \mathbf{\Phi}_1 \mathbf{f}_{t-1} + \mathbf{\Phi}_2 \mathbf{f}_{t-2} + \ldots + \mathbf{\Phi}_r \mathbf{f}_{t-r} + \mathbf{u}_t$$

The idiosyncratic components $\boldsymbol{\epsilon}_t$ may also follow time series processes.

**Exact Factor Models**

In exact factor models, the idiosyncratic components are assumed to be uncorrelated across variables.

Starting with the basic factor model:

$$\mathbf{x}_t = \mathbf{W} \mathbf{f}_t + \boldsymbol{\epsilon}_t$$

An exact factor model specifically requires:

$$\text{Cov}(\boldsymbol{\epsilon}_t) = \mathbf{\Psi}$$ 

to be a diagonal matrix. All cross-sectional correlation in $\mathbf{x}_t$ is captured by the common factors. This means:

$$\mathbb{E}[\epsilon_{it}\epsilon_{jt}] = 0 \quad \text{for all } i \neq j$$

**Approximate Factor Models**

Approximate factor models relax the strict diagonality assumption of exact factor models, allowing for weak correlation among idiosyncratic components. Using the same baseline representation:

$$\mathbf{x}_t = \mathbf{W} \mathbf{f}_t + \boldsymbol{\epsilon}_t$$

An approximate factor model allows:

$$\mathbf{\Psi} = \text{Cov}(\boldsymbol{\epsilon}_t)$$ 

to be non-diagonal.

# 2. The Factor Analysis Model

## 2.1 Model Setup

Assume:

- $T$ observations
- $p$ observed features for each observation
- $q$ latent factors, with $q < p$

- $\mathbf{x}_t \in \mathbb{R}^p$ as the (centered) observed feature vector for the $t$-th observation.

- $\mathbf{f}_t \in \mathbb{R}^{q \times 1}$ as the latent factor scores for observation $t$.

- $\mathbf{W} \in \mathbb{R}^{p \times q}$ as the matrix of factor loadings. The entry $W_{jr}$ describes how strongly the $r$-th factor influences the $j$-th observed feature.

- $\boldsymbol{\epsilon}_t \in \mathbb{R}^p$ is the noise term. 

- Factor Analysis assumes:     
    - $\mathbb{E}[\boldsymbol{\epsilon}_t] = \mathbf{0}$
    - $\text{Cov}(\boldsymbol{\epsilon}_t) = \mathbf{\Psi} \in \mathbb{R}^{p \times p}$ and $\mathbf{\Psi}$ is diagonal with entries $\psi_j > 0$
    - $\text{Cov}(\boldsymbol{\epsilon}_t, \mathbf{f}_t) = \mathbf{0}$

The Factor Analysis model is:

$$\mathbf{x}_t = \mathbf{W} \mathbf{f}_t + \boldsymbol{\epsilon}_t$$

When you collect all $T$ observations, you can write:

$$\mathbf{X} = \mathbf{F} \mathbf{W}^T + \boldsymbol{\epsilon}$$

where
- $\mathbf{F} \in \mathbb{R}^{T \times q}$ (the factor scores for all $T$ observations)
- $\boldsymbol{\epsilon} \in \mathbb{R}^{T \times p}$

From the assumptions, the data covariance matrix $\mathbf{V} \in \mathbb{R}^{p \times p}$ (i.e. the sample covariance among all features) satisfies:

$$\text{Cov}(\mathbf{x}_t) \equiv \mathbf{V} = \mathbf{\Psi} + \mathbf{W} \mathbf{W}^\top$$

- $\mathbf{\Psi}$ is $p \times p$ diagonal, containing the specific variances $\psi_j$.
- $\mathbf{W} \mathbf{W}^\top$ is also $p \times p$, rank at most $q$.

Thus Factor Analysis attempts to decompose the covariance into a low-rank part ($\mathbf{W} \mathbf{W}^\top$) plus a diagonal part ($\mathbf{\Psi}$).

# 3. Maximum Likelihood Estimation (MLE)

A more statistically rigorous method is Maximum Likelihood Estimation, which typically assumes:

- $\mathbf{f}_t \sim N(\mathbf{0},\mathbf{I}_p)$
- $\boldsymbol{\epsilon}_t \sim N(\mathbf{0},\mathbf{\Psi})$

Then the observed data $\mathbf{x}_t$ follow a $p$-dimensional Gaussian distribution with covariance $\mathbf{W} \mathbf{W}^\top + \mathbf{\Psi}$. The log-likelihood of all $T$ observations is proportional to

$$L = -\frac{T}{2} \log|\mathbf{\Psi} + \mathbf{W} \mathbf{W}^\top| - \frac{T}{2} \text{tr}\left[(\mathbf{\Psi} + \mathbf{W} \mathbf{W}^\top)^{-1} \mathbf{V}\right],$$

where $\mathbf{V}$ is the sample covariance matrix. One can do:

1. Start from an initial $\mathbf{\Psi}$.
2. Optimize $\mathbf{W}$ given $\mathbf{\Psi}$.
3. Optimize $\mathbf{\Psi}$ given $\mathbf{W}$.
4. Iterate until convergence.

After finding $\mathbf{W}$ and $\mathbf{\Psi}$, one often wants the factor scores $\mathbf{f}_t$ for each observation $t$. A common method is the "regression method" (also called Thomson's method):

$$\hat{f}_{tr} = \sum_{j=1}^{p} x_{tj} b_{jr},$$

choosing $\mathbf{b} = (b_{jr})$ to minimize mean squared error $E[(f_{tr} - \hat{f}_{tr})^2]$. In practice, once $\mathbf{W}$ and $\mathbf{\Psi}$ are known, $\mathbf{b}$ can be derived by straightforward linear algebra.

# References

- https://www.stat.cmu.edu/~cshalizi/350/lectures/12/lecture-12.pdf
- https://scikit-learn.org/stable/modules/decomposition.html#fa
- https://d-nb.info/1014099986/34
- http://www.econweb.umd.edu/~chao/Teaching/Econ721/Econ721_Lecture_on_Factor_Models.pdf
- https://www.cemla.org/PDF/webinars/2017-05-MARCELLINOMASSIMILIANO.pdf