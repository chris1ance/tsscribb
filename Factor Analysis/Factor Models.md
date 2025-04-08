# From PCA to Factor Analysis

Factor Analysis can be viewed as an extension (or modification) of Principal Components Analysis (PCA). While PCA tries to reduce dimensionality by explaining the variance of observed features with a small number of components, Factor Analysis introduces a more explicit notion of "noise" or "measurement error" and focuses on reproducing correlations among features.

In PCA, we often write each observed vector $\mathbf{x}_t \in \mathbb{R}^p$ (for $t = 1, \ldots, T$) as a combination of a small number $q$ of principal components. However, PCA does not explicitly model noise per feature, and it may try to explain even the random fluctuations.

By contrast, Factor Analysis posits that each observed data vector $\mathbf{x}_t \in \mathbb{R}^p$ is generated from a small set of $q$ latent factors $\mathbf{f}_t$ plus additive feature-specific noise. Hence the model captures shared structure (through a few factors) and also acknowledges individual "measurement error" or idiosyncratic variation for each feature.

# Notation

- **Observables**: $\mathbf{X} \in \mathbb{R}^{T \times N}$

    - Element: $x_{ti} \in \mathbb{R}$ where $t=1,...,T$ and $i=1,...,N$

    - Columns: $\mathbf{x}_{\cdot i} \in \mathbb{R}^{T \times 1}$

    - Rows: $\mathbf{x}_{t \cdot} \in \mathbb{R}^{N \times 1}$

    - Covariance Matrix: $\mathbb{C}(\mathbf{x}_{t \cdot}) = \boldsymbol{\Sigma}_x \in \mathbb{R}^{N \times N}$

- **Factors**: $\mathbf{F} \in \mathbb{R}^{T \times q}$

    - Element: $f_{tr} \in \mathbb{R}$ where $r=1,...,q$

    - Columns: $\mathbf{f}_{\cdot r} \in \mathbb{R}^{T \times 1}$

    - Rows: $\mathbf{f}_{t \cdot} \in \mathbb{R}^{q \times 1}$

    - Covariance Matrix: $\mathbb{C}(\mathbf{f}_{t \cdot}) = \boldsymbol{\Sigma}_f \in \mathbb{R}^{q \times q}$

- **State Transition Matrix**: $\boldsymbol{\Phi} \in \mathbb{R}^{q \times q}$

- **Loadings (or measurement matrix, or observation matrix)**: $\boldsymbol{\Lambda} \in \mathbb{R}^{N \times q}$ represents a matrix of weights.

    - Element: $\lambda_{ir} \in \mathbb{R}$

    - Columns: $\boldsymbol{\lambda}_{\cdot r} \in \mathbb{R}^{N \times 1}$

    - Rows: $\boldsymbol{\lambda}_{i \cdot} \in \mathbb{R}^{q \times 1}$

- **Idiosyncratic errors/shocks/component**: $\mathbf{E} \in \mathbb{R}^{T \times N}$

    - Element: $e_{ti} \in \mathbb{R}$

    - Columns: $\mathbf{e}_{\cdot i} \in \mathbb{R}^{T \times 1}$

    - Rows: $\mathbf{e}_{t \cdot} \in \mathbb{R}^{N \times 1}$

    - Covariance Matrix: $\mathbb{C}(\mathbf{e}_{t \cdot}) = \boldsymbol{\Sigma}_e \in \mathbb{R}^{N \times N}$

- **Factor/State errors/shocks**: $\mathbf{U} \in \mathbb{R}^{T \times q}$

    - Element: $u_{tr} \in \mathbb{R}$

    - Columns: $\mathbf{u}_{\cdot r} \in \mathbb{R}^{T \times 1}$

    - Rows: $\mathbf{u}_{t \cdot} \in \mathbb{R}^{q \times 1}$

    - Covariance Matrix: $\mathbb{C}(\mathbf{u}_{t \cdot}) = \boldsymbol{\Sigma}_u \in \mathbb{R}^{q \times q}$

# Definitions

- **Common Component**: $$\boldsymbol{\Lambda} \mathbf{f}_{t \cdot}$$

- **Common Component for variable $i$**: $$\mathbf{f}_{t \cdot}^T \boldsymbol{\lambda}_{i \cdot}$$

- **Exact/Strict Factor Model**: Assumes $\boldsymbol{\Sigma}_e$ is a diagonal matrix.
    - I.I.D. hypotheses and hypotheses on the diagonality of $\boldsymbol{\Sigma}_e$, which prohibit the cross correlation, are often too strong for economic data. This can result in a risk of misspecification.

- **Approximate Factor Model**: Allows $\boldsymbol{\Sigma}_e$ to be non-diagonal matrix, allowing for weak correlation among idiosyncratic components.
    - Modeling idiosyncratic dynamics might improve forecasts for two reasons: first, we could forecast the idiosyncratic component; second, we could improve the efficiency of the common factor estimates in small samples or in real-time applications in which the cross-sections at the end of the sample are incomplete.
    - Idiosyncratic components can both be weakly mutually correlated and show little heteroskedasticity.
    - It is possible to have a weak correlation between the factors and the idiosyncratic components.
    - In contrast to strict factor models, approximate Factor Models are appropriate for large $N$. 

- **Static Factor Model**: In a static factor model, the relationship between observed variables and latent factors is contemporaneous, with no lag structure in the observation equation.

- **Dynamic Factor Model**: Allow lags of the factors to affect the observed variables in the observation equation.

# Static Factor Model

## Observation/Measurement Equation

**Scalar Form:**

$$
x_{ti} = \mathbf{f}_{t \cdot}^T \boldsymbol{\lambda}_{i \cdot} + e_{ti} 
\tag{1.1}
$$

**Vector Form:**

$$
\mathbf{x}_{t \cdot} = \boldsymbol{\Lambda} \mathbf{f}_{t \cdot} + \mathbf{e}_{t \cdot} 
\tag{1.2}
$$

**Matrix Form:**

$$
\mathbf{X} = \mathbf{F} \boldsymbol{\Lambda}^T + \mathbf{E} 
\tag{1.3}
$$

## State Equation

$$
\mathbf{f}_{t \cdot} = \boldsymbol{\Phi} \mathbf{f}_{t-1, \cdot} + \mathbf{u}_{t \cdot} 
\tag{1.4}
$$

# Dynamic Factor Model

$$
\mathbf{x}_{t \cdot} = 
\sum_{j'=1}^{p'} \boldsymbol{\Lambda}_{j'} \mathbf{f}_{t-j', \cdot} + \mathbf{e}_{t \cdot} 
\tag{2.1}
$$

$$
\mathbf{f}_{t \cdot} = 
\sum_{j=1}^p \boldsymbol{\Phi}_j \mathbf{f}_{t-j, \cdot} + \mathbf{u}_{t \cdot}
\tag{2.2}
$$

# Writing the Dynamic Factor Model as a Static Factor Model

$$
\begin{array}{lr}

\mathbf{x}_{t} = \tilde{\mathbf{\Lambda}} \tilde{\mathbf{f}}_{t} + \mathbf{e}_{t}, & \mathbf{e}_{t} \sim N(\mathbf{0}, \boldsymbol{\Sigma}_e) \\

\tilde{\mathbf{f}}_{t} = \tilde{\boldsymbol{\Phi}} \tilde{\mathbf{f}}_{t-1} + \mathbf{u}_{t}, & \mathbf{u}_{t} \sim N(\mathbf{0}, \tilde{\boldsymbol{\Sigma}}_u) \tag{3}

\end{array}
$$

where $\mathbf{x}_{t}, \mathbf{e}_{t}$ and $\boldsymbol{\Sigma}_e$ are as in (1) and (2), and the other matrices are

$$
\begin{align*}

\tilde{\mathbf{f}}_{t} & =
[\mathbf{f}_{t}^{\top}, \mathbf{f}_{t-1}^{\top}, \ldots, \mathbf{f}_{t-p}^{\top}]^{\top} \in \mathbb{R}^{qp}\\

\tilde{\mathbf{\Lambda}} & =
[\mathbf{\Lambda}, \mathbf{0}, \ldots, \mathbf{0}] \in \mathbb{R}^{N \times qp}, \text { where } \mathbf{0} \text { are } N \times q \text { matrices of zeros for each factor lag }  \\

\tilde{\boldsymbol{\Phi}} & =
\left(\begin{array}{ccccc}
\boldsymbol{\Phi}_{1} & \boldsymbol{\Phi}_{2} & \cdots & \boldsymbol{\Phi}_{p-1} & \boldsymbol{\Phi}_{p} \\
\mathbf{I}_{1} & \mathbf{0} & \cdots & \mathbf{0} & \mathbf{0} \\
\mathbf{0} & \mathbf{I}_{2} & \cdots & \mathbf{0} & \mathbf{0} \\
\vdots & \vdots & \ddots & \vdots & \vdots \\
\mathbf{0} & \mathbf{0} & \cdots & \mathbf{I}_{p-1} & \mathbf{0}
\end{array}\right) \in \mathbb{R}^{qp \times qp} \text {, where } \mathbf{0} / \mathbf{I} \text { are } r \times r \text { zero } / \text { identity matrices }  \\

\tilde{\mathbf{u}}_{t} & =
[\mathbf{u}_{t}^{\top}, \mathbf{0}^{\top}, \ldots, \mathbf{0}^{\top}]^{\top} \in \mathbb{R}^{qp}, \text { with } \mathbf{0} \text { a } q \times 1 \text { vector of zeros }  \\

\tilde{\boldsymbol{\Sigma}}_u & =
\left(\begin{array}{cccc}
\boldsymbol{\Sigma}_U & \mathbf{0} & \cdots & \mathbf{0} \\
\mathbf{0} & \mathbf{0} & \cdots & \mathbf{0} \\
\vdots & \vdots & \ddots & \vdots \\
\mathbf{0} & \mathbf{0} & \cdots & \mathbf{0}
\end{array}\right) \in \mathbb{R}^{qp \times qp}, \text { where } \mathbf{0} \text { are } q \times q \text { zero matrices}

\end{align*}
$$

# Common Assumptions

1. $\mathbf{x}_{t \cdot}$ is a stationary vector process standardized to mean 0 and unit variance.
2. Where $\mathbf{f}_{t \cdot}$ follows a VAR(p), e.g. as in (2.2), the process is assumed to be stationary.

# Estimation

## BANBURA and MODUGNO (2014)

**Model**

$$\mathbf{x}_{t \cdot} = \boldsymbol{\Lambda} \mathbf{f}_{t \cdot} + \mathbf{e}_{t \cdot}$$

$$\mathbf{f}_{t \cdot} = 
\sum_{j=1}^p \boldsymbol{\Phi}_j \mathbf{f}_{t-j, \cdot} + \mathbf{u}_{t \cdot}$$ 

**Overview**

BANBURA and MODUGNO (2014) EM algorithm: Write the likelihood as if the data were complete and to iterate between two steps: in the expectation step we 'fill in' the missing data in the likelihood, while in the maximization step we re-optimize this expectation.

Let us denote the joint log-likelihood = by $l(\mathbf{X}, \mathbf{F} ; \theta)$. Given the available data $\Omega_{T} \subseteq \mathbf{X}$ for the model given by equations the EM algorithm proceeds in a sequence of two alternating steps:

1. E-step: The expectation of the log-likelihood conditional on the data is calculated using the estimates from the previous iteration, $\theta(j)$ :

$$
L(\theta, \theta(j))=\mathbb{E}_{\theta(j)}\left[l(\mathbf{X}, \mathbf{F} ; \theta) \mid \Omega_{T}\right]
$$

2. M-step: The parameters are re-estimated through the maximization of the expected log-likelihood with respect to $\theta$:

$$
\theta(j+1)=\arg \max _{\theta} L(\theta, \theta(j)) \tag{3}
$$

**Case 1: Serially uncorrelated idiosyncratic errors**

We set for simplicity $p=1$, i.e. $\boldsymbol{\Phi}=\boldsymbol{\Phi}_{1}$; the modification to the case of $p>1$ is straightforward.

We first consider the case of serially uncorrelated $\mathbf{e}_{t}$ :

$$
\mathbf{e}_{t} \sim \text { i.i.d. } \mathcal{N}(0, \boldsymbol{\Sigma}_e) \tag{4}
$$


where $\boldsymbol{\Sigma}_e$ is a diagonal matrix. In that case $\theta=\{\boldsymbol{\Lambda}, \boldsymbol{\Phi}, \boldsymbol{\Sigma}_e, \boldsymbol{\Sigma}_u\}$ and the maximization of equation (3) results in the following expressions for $\theta(j+1)$ :


$$
\begin{align*}
& \boldsymbol{\Lambda}(j+1)=\left(\sum_{t=1}^{T} \mathbb{E}_{\theta(j)}\left[\mathbf{x}_{t} \mathbf{f}_{t}^{\prime} \mid \Omega_{T}\right]\right)\left(\sum_{t=1}^{T} \mathbb{E}_{\theta(j)}\left[\mathbf{f}_{t} \mathbf{f}_{t}^{\prime} \mid \Omega_{T}\right]\right)^{-1}  \tag{5}\\
& \boldsymbol{\Phi}(j+1)=\left(\sum_{t=1}^{T} \mathbb{E}_{\theta(j)}\left[\mathbf{f}_{t} \mathbf{f}_{t-1}^{\prime} \mid \Omega_{T}\right]\right)\left(\sum_{t=1}^{T} \mathbb{E}_{\theta(j)}\left[\mathbf{f}_{t-1} \mathbf{f}_{t-1}^{\prime} \mid \Omega_{T}\right]\right)^{-1} \tag{6}
\end{align*}
$$

$$
\begin{align*}
\boldsymbol{\Sigma}_e(j+1) & =\operatorname{diag}\left(\frac{1}{T} \sum_{t=1}^{T} \mathbb{E}_{\theta(j)}\left[\left(\mathbf{x}_{t}-\boldsymbol{\Lambda}(j+1) \mathbf{f}_{t}\right)\left(\mathbf{x}_{t}-\boldsymbol{\Lambda}(j+1) \mathbf{f}_{t}\right)^{\prime} \mid \Omega_{T}\right]\right)  \tag{7}\\
& =\operatorname{diag}\left(\frac{1}{T}\left(\sum_{t=1}^{T} \mathbb{E}_{\theta(j)}\left[\mathbf{x}_{t} \mathbf{x}_{t}^{\prime} \mid \Omega_{T}\right]-\boldsymbol{\Lambda}(j+1) \sum_{t=1}^{T} \mathbb{E}_{\theta(j)}\left[\mathbf{f}_{t} \mathbf{x}_{t}^{\prime} \mid \Omega_{T}\right]\right)\right)
\end{align*}
$$


$$
\boldsymbol{\Sigma}_u(j+1)=\frac{1}{T}\left(\sum_{t=1}^{T} \mathbb{E}_{\theta(j)}\left[\mathbf{f}_{t} \mathbf{f}_{t}^{\prime} \mid \Omega_{T}\right]-\boldsymbol{\Phi}(j+1) \sum_{t=1}^{T} \mathbb{E}_{\theta(j)}\left[\mathbf{f}_{t-1} \mathbf{f}_{t}^{\prime} \mid \Omega_{T}\right]\right) \tag{8}
$$


When $\mathbf{x}_{t}$ does not contain missing data, we have that


$$
\mathbb{E}_{\theta(j)}\left[\mathbf{x}_{t} \mathbf{x}_{t}^{\prime} \mid \Omega_{T}\right]=\mathbf{x}_{t} \mathbf{x}_{t}^{\prime} \quad \text { and } \quad \mathbb{E}_{\theta(j)}\left[\mathbf{x}_{t} \mathbf{f}_{t}^{\prime} \mid \Omega_{T}\right]=\mathbf{x}_{t} \mathbb{E}_{\theta(j)}\left[\mathbf{f}_{t}^{\prime} \mid \Omega_{T}\right] \tag{9}
$$


Finally, the conditional moments of the latent factors, $\mathbb{E}_{\theta(j)}\left[\mathbf{f}_{t} \mid \Omega_{T}\right], \mathbb{E}_{\theta(j)}\left[\mathbf{f}_{t} \mathbf{f}_{t}^{\prime} \mid \Omega_{T}\right], \mathbb{E}_{\theta(j)}\left[\mathbf{f}_{t-1} \mathbf{f}_{t-1}^{\prime} \mid \Omega_{T}\right]$ and $\mathbb{E}_{\theta(j)}\left[\mathbf{f}_{t} \mathbf{f}_{t-1}^{\prime} \mid \Omega_{T}\right]$, can be obtained through the Kalman smoother for the state space representation:

$$
\begin{array}{ll}
\mathbf{x}_{t}=\boldsymbol{\Lambda}(j) \mathbf{f}_{t}+\mathbf{e}_{t}, & \mathbf{e}_{t} \sim \text { i.i.d. } \mathcal{N}(0, \boldsymbol{\Sigma}_e(j)) \\
\mathbf{f}_{t}=\boldsymbol{\Phi}(j) \mathbf{f}_{t-1}+\mathbf{u}_{t}, & \mathbf{u}_{t} \sim \text { i.i.d. } \mathcal{N}(0, \boldsymbol{\Sigma}_u(j)) \tag{10}
\end{array}
$$

However, when $\mathbf{x}_{t}$ contains missing values we can no longer use equation (9) when developing the expressions (5) and (7). Let $\mathbf{W}_{t}$ be a diagonal matrix of size $N$ with $i$ th diagonal element equal to 0 if $x_{i,t}$ is missing and equal to 1 otherwise. $\boldsymbol{\Lambda}(j+1)$ can be obtained as

$$
\operatorname{vec}(\boldsymbol{\Lambda}(j+1))=\left(\sum_{t=1}^{T} \mathbb{E}_{\theta(j)}\left[\mathbf{f}_{t} \mathbf{f}_{t}^{\prime} \mid \Omega_{T}\right] \otimes \mathbf{W}_{t}\right)^{-1} \operatorname{vec}\left(\sum_{t=1}^{T} \mathbf{W}_{t} \mathbf{x}_{t} \mathbb{E}_{\theta(j)}\left[\mathbf{f}_{t}^{\prime} \mid \Omega_{T}\right]\right) \tag{11}
$$


Intuitively, $\mathbf{W}_{t}$ works as a selection matrix, so that only the available data are used in the calculations. Analogously, the expression (7) becomes

$$
\begin{align*}
\boldsymbol{\Sigma}_e(j+1)=\operatorname{diag} & \left(\frac { 1 } { T } \sum _ { t = 1 } ^ { T } \left(\mathbf{W}_{t} \mathbf{x}_{t} \mathbf{x}_{t}^{\prime} \mathbf{W}_{t}^{\prime}-\mathbf{W}_{t} \mathbf{x}_{t} \mathbb{E}_{\theta(j)}\left[\mathbf{f}_{t}^{\prime} \mid \Omega_{T}\right] \boldsymbol{\Lambda}(j+1)^{\prime} \mathbf{W}_{t}-\mathbf{W}_{t} \boldsymbol{\Lambda}(j+1) \mathbb{E}_{\theta(j)}\left[\mathbf{f}_{t} \mid \Omega_{T}\right] \mathbf{x}_{t}^{\prime} \mathbf{W}_{t}\right.\right. \\
& \left.\left.+\mathbf{W}_{t} \boldsymbol{\Lambda}(j+1) \mathbb{E}_{\theta(j)}\left[\mathbf{f}_{t} \mathbf{f}_{t}^{\prime} \mid \Omega_{T}\right] \boldsymbol{\Lambda}(j+1)^{\prime} \mathbf{W}_{t}+\left(I-\mathbf{W}_{t}\right) \boldsymbol{\Sigma}_e(j)\left(I-\mathbf{W}_{t}\right)\right)\right) \tag{12}
\end{align*}
$$

Again, only the available data update the estimate. $I-\mathbf{W}_{t}$ in the last term 'selects' the entries of $\boldsymbol{\Sigma}_e(j)$ corresponding to the missing observations. For example, when for some $t$ all the observations in $\mathbf{x}_{t}$ are missing, the period $t$ contribution to $\boldsymbol{\Sigma}_e(j+1)$ would be $\boldsymbol{\Sigma}_e(j) / T$.

When applying the Kalman filter on the state space representation (10), in case some of the observations in $\mathbf{x}_{t}$ are missing, the corresponding rows in $\mathbf{x}_{t}$ and $\boldsymbol{\Lambda}(j)$ (and the corresponding rows and columns in $\boldsymbol{\Sigma}_e(j)$ ) are skipped.

With $\mathbf{W}_{t} \equiv I$, (11) and (12) coincide with the 'complete data' expressions obtained by plugging (9) into (5) and (7).

**Restrictions on the Parameters**

We can modify the M-step to impose restrictions of the form $H_{\Lambda} \operatorname{vec}(\boldsymbol{\Lambda})=\kappa_{\Lambda}$ for the model given by (1)-(2). Restricted estimates are given by

$$
\begin{align*}
\operatorname{vec}\left(\boldsymbol{\Lambda}_{r}(j+1)\right)= & \operatorname{vec}\left(\boldsymbol{\Lambda}_{u}(j+1)\right)+\left(\sum_{t=1}^{T} \mathbb{E}_{\theta(j)}\left[\mathbf{f}_{t} \mathbf{f}_{t}^{\prime} \mid \Omega_{T}\right] \otimes \boldsymbol{\Sigma}_e(j)\right) H_{\Lambda}^{\prime}  \tag{13}\\
& \times\left(H_{\Lambda}\left(\sum_{t=1}^{T} \mathbb{E}_{\theta(j)}\left[\mathbf{f}_{t} \mathbf{f}_{t}^{\prime} \mid \Omega_{T}\right] \otimes \boldsymbol{\Sigma}_e(j)\right) H_{\Lambda}^{\prime}\right)^{-1}\left(\kappa_{\Lambda}-H_{\Lambda} \operatorname{vec}\left(\boldsymbol{\Lambda}_{u}(j+1)\right)\right)
\end{align*}
$$

where $\boldsymbol{\Lambda}_{u}(j+1)$ is the unrestricted estimate given by expression (11). Restrictions on the parameters in the transition equation $H_{\Phi} \operatorname{vec}(\boldsymbol{\Phi})=\kappa_{\Phi}$ can be imposed in an analogous manner (see Bork, 2009).

The methodology here can be applied to estimate these types of models in the presence of missing data.

**Case 2: Modelling the Serial Correlation in the Idiosyncratic Component**

We represent the idiosyncratic component by an $\operatorname{AR}(1)$ process and to add it to the state vector. More precisely, we assume that $e_{i,t}, i=1, \ldots, N$ in (1) can be decomposed as

$$
\begin{array}{ll}
e_{i,t}=\tilde{e}_{i,t}+\xi_{i,t}, & \xi_{i,t} \sim \text { i.i.d. } \mathcal{N}(0, \kappa) \\
\tilde{e}_{i,t}=\alpha_{i} \tilde{e}_{i,t-1}+\varepsilon_{i,t}, & \varepsilon_{i,t} \sim \text { i.i.d. } \mathcal{N}\left(0, \sigma_{i}^{2}\right) \tag{14}
\end{array}
$$

where both $\xi_{t}=\left[\xi_{1,t}, \ldots, \xi_{N,t}\right]^{\prime}$ and $\tilde{\mathbf{e}}_{t}=\left[\tilde{e}_{1,t}, \ldots, \tilde{e}_{N,t}\right]^{\prime}$ are cross-sectionally uncorrelated and $\kappa$ is a very small number. Combining (1), (2) and (14) results in the new state space representation:

$$
\begin{align*}

\mathbf{x}_{t}=
\tilde{\boldsymbol{\Lambda}} \tilde{\mathbf{f}}_{t}+\xi_{t}, \ \ \ & \xi_{t} \sim \mathcal{N}(0, \kappa \mathbf{I}) \\

\tilde{\mathbf{f}}_{t}=
\tilde{\boldsymbol{\Phi}} \tilde{\mathbf{f}}_{t-1}+\tilde{\mathbf{u}}_{t}, \ \ \ & \tilde{\mathbf{u}}_{t} \sim \mathcal{N}(0, \tilde{\boldsymbol{\Sigma}}_u) 

\tag{15}
\end{align*}
$$

where

$$
\tilde{\mathbf{f}}_{t}=\left[
\begin{array}{l}
\mathbf{f}_{t} \\
\tilde{\mathbf{e}}_{t}
\end{array}
\right]
$$

$$
\tilde{\mathbf{u}}_{t}=\left[
\begin{array}{l}
\mathbf{u}_{t} \\
\varepsilon_{t}
\end{array}
\right]
$$

$$
\tilde{\boldsymbol{\Lambda}}=\left[
\begin{array}{ll}
\boldsymbol{\Lambda} & I
\end{array}
\right]
$$

$$
\tilde{\boldsymbol{\Phi}}=\left[
\begin{array}{cc}
\boldsymbol{\Phi} & 0 \\
0 & \operatorname{diag}\left(\alpha_{1}, \cdots, \alpha_{N}\right)
\end{array}
\right]
$$

$$
\tilde{\boldsymbol{\Sigma}}_u=\left[
\begin{array}{cc}
\boldsymbol{\Sigma}_u & 0 \\
0 & \operatorname{diag}\left(\sigma_{1}^{2}, \cdots, \sigma_{N}^{2}\right)
\end{array}
\right]
$$

$$\varepsilon_{t}=\left[\varepsilon_{1,t}, \ldots, \varepsilon_{N,t}\right]^{\prime}$$

# References

- Darné, Olivier, Karim Barhoumi, and Laurent Ferrara. "Dynamic Factor Models: A Review of the Literature," 2018.
- Bańbura, Marta, and Michele Modugno. "Maximum Likelihood Estimation of Factor Models on Datasets with Arbitrary Pattern of Missing Data." Journal of Applied Econometrics 29, no. 1 (2014): 133–60. https://doi.org/10.1002/jae.2306.