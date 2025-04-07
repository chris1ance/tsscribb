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

- **Approximate Factor Model**: Allows $\boldsymbol{\Sigma}_e$ to be non-diagonal matrix, allowing for weak correlation among idiosyncratic components.

- **Static Factor Model**: In a static factor model, the relationship between observed variables and latent factors is contemporaneous, with no lag structure in the observation equation.

- **Dynamic Factor Model**: Allow lags of the factors to affect the observed variables in the observation equation.

# Static Factor Model

## Observation/Measurement Equation

**Scalar Form:**

$$
x_{ti} = \mathbf{f}_{t \cdot}^T \boldsymbol{\lambda}_{i \cdot} + e_{ti} \tag{1.1}
$$

**Vector Form:**

$$
\mathbf{x}_{t \cdot} = \boldsymbol{\Lambda} \mathbf{f}_{t \cdot} + \mathbf{e}_{t \cdot} \tag{1.2}
$$

**Matrix Form:**

$$
\mathbf{X} = \mathbf{F} \boldsymbol{\Lambda}^T + \mathbf{E} \tag{1.3}
$$

## State Equation

$$
\mathbf{f}_{t \cdot} = \boldsymbol{\Phi} \mathbf{f}_{t-1, \cdot} + \mathbf{u}_{t \cdot} \tag{1.4}
$$

# Dynamic Factor Model

$$
\mathbf{x}_{t \cdot} = \sum_{j'=1}^{p'} \boldsymbol{\Lambda}_{j'} \mathbf{f}_{t-j', \cdot} + \mathbf{e}_{t \cdot} \tag{2.1}
$$

$$
\mathbf{f}_{t \cdot} = \sum_{j=1}^p \boldsymbol{\Phi}_j \mathbf{f}_{t-j, \cdot} + \mathbf{u}_{t \cdot} \tag{2.2}
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

# Types of Factor Models

## Static factor models (SFM)

- Application: Small $N$, with a single factor can generally explain most of the variance, i.e. $Q=1$. 

- The series are assumed to be stationary, to have finite variance, and to be standardized. 

- Assumptions:
    - (SH1) The factors $\left(F_{t}\right)$ are centered, $E\left(F_{t}\right)=0$, and are mutually orthogonal for all $t$, i.e. : $\forall t, E\left(F_{j t} F_{j^{\prime} t}\right)=0$ for $j \neq j^{\prime}$. Consequently, the variance-covariance matrix of $\left(F_{t}\right), \Sigma_{F}=E\left(F_{t} F_{t}^{\prime}\right)$, is a diagonal matrix.
    - (SH2) The idiosyncratic processes $\left(\xi_{i t}\right)$ and $\left(\xi_{i^{\prime} t}\right)$ are mutually orthogonal for all $i \neq i^{\prime}$, with $E\left(\xi_{t}\right)=0$. Consequently the variance-covariance matrix $\left(\xi_{t}\right)$ is a diagonal matrix: $\Sigma_{\xi}=E\left(\xi_{t} \xi_{t}^{\prime}\right)=\operatorname{diag}\left(\sigma_{1}^{2}, \ldots, \sigma_{N}^{2}\right)$;
    - (SH3) The factors $\left(F_{t}\right)$ and idiosyncratic noise $\left(\xi_{i t}\right)_{i=1, \ldots, N}$ are not correlated, i.e. : $\forall i, j, t, t^{\prime}$ we have: $E\left(F_{j t} \xi_{i t^{\prime}}\right)=0$;
    - (SH4) The variables are assumed to be independent and identically distributed over time (IID), so that for $t \neq t^{\prime}, E\left(F_{j t} F_{j t^{\prime}}\right)=0$ and $E\left(\xi_{i t} \xi_{i t^{\prime}}\right)=0$.

- SFM: The factors $\left(F_{t}\right)$ do not possess their own dynamic and the relationship between the factors and variables is linear with constant weights over time. This model can be estimated either by assuming that the variables are IID (hypothesis SH4), or by assuming that there is a time dynamic within the variables (SH4 is abandoned).

- Assuming that $\left(F_{t}\right)$ and $\left(\xi_{t}\right)$ are not correlated and are zero mean, then the variance-covariance matrix for the static factor model, denoted $\Sigma_{X}=E\left(X_{t} X_{t}^{\prime}\right)$, is given by:


    $$
    \Sigma_{X}=\Lambda \Sigma_{F} \Lambda^{\prime}+\Sigma_{\xi} \tag{10.3}
    $$


    By normalizing the variance-covariance matrices of $\left(F_{t}\right), \Sigma_{F}=I_{r}$, and by assuming that the diagonal elements of the variance-covariance matrix $\Sigma_{\xi}$ of $\left(\xi_{t}\right)$ are bounded, we obtain:


    $$
    \Sigma_{X}=\Lambda \Lambda^{\prime}+\Sigma_{\xi} \tag{10.4}
    $$


    The static factor model can, thus, be identified and estimated. The factorial analysis method is used for the static estimation of the factors. The weighting matrix $\Lambda$ can be estimated by minimizing the sum of the squared residuals as follows:


    $$
    \sum_{t=1}^{T}\left(X_{t}-\Lambda F_{t}\right)^{\prime}\left(X_{t}-\Lambda F_{t}\right) \tag{10.5}
    $$


    subject to the constraint $\Lambda^{\prime} \Lambda=I_{r}$.

- Doz and Lenglart (1999) show that this method produces convergent estimators even when the data used are autocorrelated, as is the case with time series. Moreover, they show empirically that this method provides a very good approximation of the dynamic method.

## Exact or strict dynamic factor models (DFM)

- Application: Small $N$

- Static factor models (SFM) are different from exact or strict dynamic factor models (DFM) in the sense that the latter incorporate a time dynamic. Thus, in the DFM, the common component can be seen as a sum of common shocks, whether contemporaneous or lagged. The model is, then, defined as follows:


- Assumptions: 
    - (DH1) The factors $\left(F_{j t}\right)$ and $\left(F_{j^{\prime} t}\right)$ are mutually orthogonal, but the factors $\left(F_{j t}\right)$ can be autocorrelated and are variance-covariance stationary,i.e. : $\forall j \neq j^{\prime}, \tau \neq 0, E\left(F_{j, t}\right)=0, \operatorname{cov}\left(F_{j, t}, F_{j^{\prime}, t-\tau}\right)=0$ and $\operatorname{cov}\left(F_{j, t}, F_{j, t-\tau}\right)$ depends only on $\tau$.
    - (DH2) The idiosyncratic processes $\left(\xi_{i t}\right)$ and $\left(\xi_{i^{\prime} t}\right)$ are mutually orthogonal, but the processes $\left(\xi_{i t}\right)$ can be autocorrelated and covariance-stationary, i.e. : $\forall i \neq i^{\prime}, \tau \neq 0, E\left(\xi_{i t}\right)=0, \operatorname{cov}\left(\xi_{i, t}, \xi_{i^{\prime}, t-\tau}\right)=0$ and $\operatorname{cov}\left(\xi_{i, t}, \xi_{i, t-\tau}\right)$ depends only on $\tau$.
    - (DH3) The factors $\left(F_{j t}\right)$ and the idiosyncratic processes $\left(\xi_{i t}\right)$ are orthogonal for all $i, j$.

- Based on these hypotheses, we can, then, attempt to estimate a dynamic factor model by likelihood maximization in the time domain, with the additional hypothesis of Normality for the model residuals. The maximum likelihood estimator is calculated by, first, placing the model in a space-state form and, then, using a Kalman-type recursive filter.

- The DFM can be written in a space-state form, assuming that the common factors follow a VAR process of order $p$:


    $$
    \Phi(L) F_{t}=\varepsilon_{t} \quad \Leftrightarrow \quad F_{t}=\sum_{\tau=1}^{p} \Phi_{\tau} F_{t-\tau}+\varepsilon_{t} \tag{10.12}
    $$


    and, for a given index $i$, the idiosyncratic process $\left(\xi_{i t}\right)$ follows an AR process of order $p^{\prime}$ in the following form:


    $$
    \psi_{i}(L) \xi_{i t}=\eta_{i t} \quad \Leftrightarrow \quad \xi_{i t}=\sum_{\tau=1}^{p^{\prime}} \psi_{i \tau} \xi_{i, t-\tau}+\eta_{i t} \tag{10.13}
    $$


- In practice, the orders $p$ and $p^{\prime}$ of the lag polynomials must be selected prior to the estimation stage. This selection is generally done by minimizing an AIC-type information criterion or a BIC-type information criterion or by using the Doz and Lenglart (1999) test. In empirical studies, $p=2$ and $p^{\prime}=1$ are often shown to be sufficient to whiten the residuals.

- This type of model shown by equations $(1)-(3)$, $10.12$ and $10.13$ allows a space-state representation as follows:

    $$
    X_{t}=c_{t} \beta_{t}+m_{t} Z_{t}+w_{t} \tag{10.14}
    $$


    where $\left(Z_{t}\right)$ is a vector of $n$ explanatory variables, for example, the lagged values of the observed variables $\left(X_{t}\right)$, and where:


    $$
    \beta_{t}=a_{t} \beta_{t-1}+v_{t} \tag{10.15}
    $$


    Equation 10.14 is the measure equation, which describes the relations between the unobservable states, of dimension $r$, , and the observable variables, of dimension $n$, where $\beta_{t}$ represents the state vector:

    $$
    \beta_{t}=\left[\begin{array}{c}
    F_{t} \\
    \vdots \\
    F_{t-p+1} \\
    \xi_{t} \\
    \vdots \\
    \xi_{t-q+1}
    \end{array}\right]
    $$

    Equation 10.15 represents the state or transition equation, which describes the development of unobservable states. We see that $a_{t}, c_{t}$ and $m_{t}$ are matrices that can depend on time, of the dimensions $((p \times r+q \times N) \times(p \times r+q \times N)),(N \times(p \times r+q \times N))$ and $(N \times n)$, respectively, and where $v_{t}$ is a Gaussian white noise vector of dimension $(p \times r+q \times N)$, $w_{t}$ is a Gaussian white noise vector of dimension $N$, of the variance-covariance matrices $Q_{t}$ and $R_{t}$ respectively. In practice, the system is generally assumed to be invariant over time, i.e. $a_{t}, c_{t}$ and $m_{t}$ are constant. It is also assumed that for all $t, t^{\prime} \neq t$, $E\left(v_{t} w_{t}^{\prime}\right)=0$.

- The model in its space-state form can then be estimated by maximum likelihood using a filtering method such as the Kalman filter. MLE can take a great deal of time as it requires inversion of a large dimensional matrix, even when $N$ is small. In general, in the case of numerical optimization, the EM algorithm is used.

## Approximate factor models (large $N$)

Although the concept of factor models is attractive, the traditional approach presented in the previous section has a number of limitations that are both theoretical and practical in nature.

- $(N)$ is often larger than $(T)$ in economic data series.
- Asymptotic convergence of estimators is assured when $T$ tends to infinity and $N$ is fixed, but not when $N$ also tends to infinity;
- I.I.D. hypotheses and hypotheses on the diagonality of the variance-covariance matrix of the idiosyncratic component $\Sigma_{\xi}$, which prohibit the cross correlation, are often too strong for economic data. This can result in a risk of misspecification;
- MLE is generally considered unachievable for factor models of large dimensions because the number of parameters to be estimated is too large;
- The traditional approach makes it possible to consistently estimate the coefficients of the weighting factors $\left(\lambda_{i}\right)$ by MLE when $T$ is large, but not the common factors $\left(F_{t}\right)$, for which only the estimated value can be obtained. Meanwhile, in most economic problems, it is these common factors that are of greatest interest since they represent the common shocks, the diffusion indices, etc., for example.

To respond to a number of these limitations, the idea of factor models was generalized to allow for the manipulation of less strict hypotheses on the variance-covariance matrix of the idiosyncratic components by proposing an approximate factor structure.

### Approximate static factor models (SFM)

- Chamberlain and Rothschild (1983) are the first to introduce the so-called "approximate" factor structure concept by allowing for idiosyncratic errors to be weakly correlated.

### Approximate dynamic factor models (DFM)

- Forni and Lippi (1997), Forni and Reichlin (1998) and Forni et al. $(2000,2004)$ extend approximate factor models by considering dynamic factor models of large dimensions and introduce different methods for the estimation of this type of model. These models are referred to as "generalized" because they combine both dynamic and approximate structures, i.e., they generalize exact dynamic factor models by assuming that the number of variables $N$ tends to infinity and by allowing idiosyncratic processes to be mutually correlated.

- It has been shown that the principal components are convergent estimators of factors, both in the static context (Bai and Ng, 2002; Stock and Watson, 2002; Bai, 2003) and in the dynamic context (Forni et al., 2000, 2004).

- Approximate factor models have several advantages over strict models. They are flexible and appropriate under general hypotheses on measurement errors and, usually, on the cross-correlation of idiosyncratic components. The misspecification error resulting from the approximate structure of the idiosyncratic component disappears when $N$ and $T$ are large, as long as the cross-correlation of the idiosyncratic processes is relatively small and that of the common components increases across the transverse dimension when $N$ increases. 

- In short, approximate factor models have two important advantages over traditional factor models:
    - The idiosyncratic components can both be weakly mutually correlated and show little heteroskedasticity. This can reflect the condition in which all the eigenvalues of the idiosyncratic variance-covariance matrix $\Sigma_{\xi}=E\left(\xi_{t} \xi_{t}^{\prime}\right)$ are bounded. Thus, the absolute mean of the covariances is bounded;
    - In this type of model, it is possible to have a weak correlation between the factors $\left(F_{t}\right)$ and the idiosyncratic components $\left(\xi_{t}\right)$.


# Estimation of factor models for large $N$

In this section, we present the main estimation methods of factor models, whether static or dynamic, when the number of variables is high (large $N$ ). In this case, the usual methods based on maximizing likelihood run into the problem of the dimension of the parameter to be estimated.

## Static factor models: the Stock and Watson (2002) approach

One of the first approximate factor models is the one proposed by Stock and Watson (SW) (2002), which is based on a static PCA. The PCA is used since it allows for the estimation of both the parameters and the factors of the model given by equation 10.1) by maximizing the variance explained by the initial variables, for a small number $r$ of static factors $\left(F_{t}\right)$. The main aim of the SW approach is to approximate the factors by a linear combination of the data $\widehat{F}_{j, t}=\widehat{W}_{j}^{\prime} X_{t}$, for $j=1, \ldots, r$, that maximizes the variance of the estimated factors $\widehat{W}_{j}^{\prime} \widehat{\Sigma}_{x} \widehat{W}_{j}$, where $\widehat{\Sigma}_{x}=(1 / T) \sum_{t=1}^{T} X_{t} X_{t}^{\prime}$ is the empirical variance-covariance matrix of the vector of the initial standardized data $X_{t}$.

Under the following normalization assumption: $\widehat{W}_{j}^{\prime} \widehat{W}_{j^{\prime}}=1$ for $j=j^{\prime}$ and $\widehat{W}_{j}^{\prime} \widehat{W}_{j^{\prime}}=0$ for $j \neq j^{\prime}$, the maximization problem can, then, be transformed into the solution of a eigenvalues problem:


$$
\widehat{\Sigma}_{x} \widehat{W}_{j}=\widehat{\mu}_{j} \widehat{W}_{j}, \tag{10.28}
$$


where $\widehat{\mu}_{j}$ is the $j$-th eigenvalue and $\widehat{W}_{j}$ is the associated eigenvector of dimension ( $N \times 1$ ). Once they have been calculated, the highest $N$ eigenvalues are classified in decreasing order. Then, the eigenvectors are, in turn, classified in decreasing order with respect to the highest $r$ eigenvalues. The factors proposed by SW are, then, written as follows:


$$
F_{t}^{S W}=\widehat{W}^{\prime} X_{t} \tag{10.29}
$$


where $\widehat{W}$ is the matrix of dimension $(N \times r)$ of the stacked eigenvectors $\widehat{W}=\left(\widehat{W}_{1}, \ldots, \widehat{W}_{r}\right)$

Because it is easy to use, the Stock and Watson (2002) approach is naturally attractive, and the empirical results, particularly in a forecasting context, show that the results yielded by this approach are not significantly poorer than the other approaches in terms of forecasting error (on this point, see D'Agostino and Giannone, 2012, or Barhoumi, Darné and Ferrara, 2013).

## Dynamic factor models

The Stock and Watson approach does not allow for use of the different dynamics that may exist between the variables used. To take account of this dynamic structure in factor models, several alternatives to the static factor model have been proposed in the literature. Specifically, there are two main types of dynamic factor models or approaches. Developed by Doz, Giannone and Reichlin (2011, 2012), the first one is based on a space-state representation of models in the time domain. Proposed by Forni, Hallin, Lippi and Reichlin (2004, 2005), the second one is based on the spectral domain.

### Time domain approach
Doz, Giannone and Reichlin (DGR) $(2011,2012)$ propose a dynamic factor model that can be represented in a space-state form. Specifically, DGR $(2011,2012)$ estimate their dynamic factor model using two different approaches. The first one is the so-called two-step approach (DGR, 2011). The second one is based on the quasi maximum likelihood (DGR, 2012).

According to DGR (2011), for a number $r$ of factors and $q$, of dynamic shocks, the estimation is carried out in two steps. In the first step:

- $\widehat{F}_{t}$ is estimated using a PCA, as an initial estimate;
- Then, equations 10.6 and 10.11 are estimated using the estimated factor from the previous step, $\widehat{F}_{t}$, to obtain both $\widehat{\lambda}_{i}^{*}(L)$ and the variance-covariance matrix of the residuals $\widehat{\xi}^{*}$, denoted $\widehat{\Sigma}_{\xi^{*}}$. To obtain an estimate of $C(L)$, appearing in equation 10.10, DGR (2011) apply a decomposition of eigenvalues to the matrix $\widehat{\Sigma}_{\xi^{*}}$ by taking into account the number of dynamic shocks $q$. Let us introduce the matrix $M$ of dimension $(r \times q)$ corresponding to the largest $q$ eigenvalues and the matrix $P$ of dimension ( $q \times q$ ) containing the largest $q$ eigenvalues in its diagonal and zeros elsewhere. The estimate of $C(L)$ is, then, obtained by $\widehat{C}(L)=M \times P^{-1 / 2}$.

In a second step, the coefficients and parameters of the system described by equations 10.6 and 10.11 are considered to be known and provided by the first step. The model is, then, written in a space-state form and the Kalman filter is applied to obtain new estimates of the factors.

In their alternative approach, DGR (2012) estimate an approximate dynamic factor model using the quasi maximum likelihood method. The main aim of this approach is to consider the strict factor model as a misspecification of the approximate factor model and to analyze the properties of the maximum likelihood indicator of the factors under this misspecification. By analyzing the properties of the maximum likelihood estimator under several sources of misspecifications, such as an omitted serial correlation of the observations or a cross-sectional correlation of the idiosyncratic components, DGR (2012) show that these misspecifications do not affect the robustness of the common factors, particularly for fairly large $N$ and $T$. More specifically, this estimator is a valid parametric alternative for the estimator resulting from a PCA. The model defined by means of equations 10.6 and 10.11 can be put in a space-state form, with a number of states equal to the number of common factors $r$. It is noteworthy that the estimation of the parameters of the model, particularly the common factors, by the quasi maximum likelihood can be approximated by their anticipated values, using the Kalman filter.

These dynamic factor models have also been called restricted dynamic factor models, since the $r$ static factors are caused by a number $q$ of dynamic factors, with $q \leq r$ (Forni et al., 2005 ; Hallin and Liska, 2007).

### Frequency domain approach

In a series of articles, Forni, Hallin, Lippi and Reichlin (2000, 2003, 2004, 2005) (FHLR) propose a dynamic PCA in the frequency domain, also called a generalized dynamic factor model, to estimate dynamic factors. The purpose of their model is to identify the dynamic structure of a factor model. The dynamic factor model is given by equations 10.6 and 10.7. The method proposed by FHLR makes it possible to estimate dynamic factors in a first step and, then, obtain the static factors from the estimated dynamic factors in a second step. The approach proposed by FHLR aims to estimate both the dynamic factors and their covariances. This estimation is performed to maximize the variance of the common component under certain orthogonality restrictions. 

# Selection of the number of factors

## Selection of the number of factors for static factor models

To specify the number of factors, Bai and Ng (2002) suggest using information criteria to select the optimal number of static factors $r$, when $N$ and $T$ tend to infinity. Bai and Ng (2002) propose information criteria based on the quality of adjustment of the model to the data measured by the variance $V(j, F)$ such that :


$$
V(j, F)=(N T)^{-1} \sum_{t=1}^{T}\left(X_{t}-\widehat{\Lambda} \widehat{F}_{t}\right)^{2} \tag{10.38}
$$


where $j$ is a given number of factors such as $\widehat{F}_{t}=\left(\widehat{F}_{1 t}, \ldots, \widehat{F}_{j t}\right)^{\prime}$. Thus, if the number of factors $j$ increases, the variance of the factors increases mechanically and the sum of the squares of the residuals decreases in turn. Bai and Ng (2002), then, suggest introducing a penalty function in the criterion to be optimized and propose the following three criteria, corresponding to different penalty functions:

$$
\begin{gather*}
I C_{1}(j)=\ln (V(j, F))+j \cdot\left(\frac{N+T}{N T}\right) \ln \left(\frac{N T}{N+T}\right)  \tag{10.39}\\
I C_{2}(j)=\ln (V(j, F))+j \cdot\left(\frac{N+T}{N T}\right)  \tag{10.40}\\
I C_{3}(j)=\ln (V(j, F))+j \cdot\left(\ln C_{N T}^{2} / C_{N T}^{2}\right) \tag{10.41}
\end{gather*}
$$

where $C_{N T}=\min \{\sqrt{N}, \sqrt{T}\}$ and $\ln$ denotes the natural logarithm. The estimation of the number of factors $r$ is obtained by minimization of the information criteria for $j=0, j=r_{\max }$, where $r_{\max }$ is the maximum number of static factors. These criteria reflect the trade-off between the quality of the adjustment and the risk of overadjustmen ${ }^{13}$. Bai and $\mathrm{Ng}(2002)$ show that their criteria are robust to the presence of a heteroskedastic component in the time and cross-section dimensions between variables, but also in the presence of weak serial and cross-section dependence.

Subsequently, Alessi et al. (2010) extend this criterion by modifying the strength of the penalty function that appears in the preceding three criteria given by equations 10.39 - $10.40-10.41$. Alessi et al. (2010) propose an alternative to the criteria proposed by Bai and $\mathrm{Ng}(2002)$ by multiplying the penalty function by a positive constant $c$ suggested originally by Hallin and Liska (2007), representing the strength of the penalty function. The authors, thus, propose the following two criteria:

$$
\begin{align*}
I C_{1}^{*}(j) & =\ln (V(j, F))+c . j .\left(\frac{N+T}{N T}\right) \ln \left(\frac{N T}{N+T}\right)  \tag{10.42}\\
I C_{2}^{*}(j) & =\ln (V(j, F))+c . j \cdot\left(\frac{N+T}{N T},\right) \tag{10.43}
\end{align*}
$$

where $V(j, F)$ is given by equation 10.38 . The estimation of the number of factors $r$ is obtained by minimization of the information criteria $I C_{1}^{*}$ and $I C_{2}^{*}$ for $j=0, j=r_{\max }$, where $r_{\max }$ is the maximum number of static factors.

The procedure for the selection of the number of static factors depends both on the variance of the number of estimated factors $V_{c}(r)$ (for $N$ and $T$ tending to infinity) and on the constant $c \in\left[0, c_{\text {max }}\right]$. Alessi et al. (2010) suggest estimating this variance $V_{c}(r)$ by reiterating the procedure for estimating $r$ for a finite number of subsets of the initial $N$ variables, also making the number of observations $T$ vary.

Kapetanios (2010) proposes a concurrent method to the information criterion to estimate the number of static factors, based on the random matrix theory. His approach is based on a series of tests on the largest eigenvalues of the variance-covariance matrix of the initial data, which we have denoted $\Sigma_{X}$. Other procedures have been suggested by Yao and Pan (2008) and Onatski (2010).

## Selection of the number of factors for dynamic factor models

### The Bai and Ng (2007) criterion

In the context of dynamic factor models, the number of dynamic shocks $q$ (for the estimation of factors in dynamic principal components and their space-state form) can be determined using the Bai and Ng (2007) information criterion. This criterion is obtained by considering the $r$ estimated static factors as given and, then, estimating a VAR model of order $p$ on these factors, where the order $p$ is selected using the BIC criterion. Next, a spectral decomposition of the variance-covariance matrix of the estimated residuals of the VAR model, denoted $\widehat{\Sigma}_{\varepsilon}$ of dimension $(r \times r)$, is calculated. Then, the $j$-th ordered eigenvalue $\widehat{c}_{j}$, where $\widehat{c}_{1}>\widehat{c}_{2} \geq \ldots \widehat{c}_{r} \geq$ 0 is recovered. Finally, for $l=1, \ldots, r-1$, Bai et Ng (2007) propose the following two quantities:

$$
\begin{aligned}
& \widehat{D}_{1, l}=\left(\frac{\widehat{c}_{l+1}}{\sum_{j=1}^{r} \widehat{c}_{j}}\right)^{1 / 2} \\
& \widehat{D}_{2, l}=\left(\frac{\sum_{j=l+1}^{r} \widehat{c}_{j}}{\sum_{j=1}^{r} \widehat{c}_{j}}\right)^{1 / 2}
\end{aligned}
$$

where $\widehat{D}_{1, l}$ represents a measure of the marginal contribution of the $l+1^{\text {me }}$ eigenvalue and $\widehat{D}_{2, l}$ represents a measure of the cumulative contribution of the eigenvalues, under the hypotheses that $\widehat{\Sigma}_{\varepsilon}$ is the unit matrix of dimension $(r \times r)$ and that $c_{l}=0$ for $l>q^{14}$.

Thus, according to the selected marginal contribution measure, the number of dynamic factors $q$ is obtained by minimizing:

$$
\left\{l \text { tel que }: \widehat{D}_{1, l} \leq \frac{c}{\min \left[n^{\frac{2}{5}}, T^{\frac{2}{5}}\right]}\right\},
$$

or:

$$
\left\{l \text { tel que }: \widehat{D}_{2, l} \leq \frac{c}{\min \left[n^{\frac{2}{5}}, T^{\frac{2}{5}}\right]}\right\} .
$$

Bai and Ng (2007) suggest using $c=1$ based on Monte Carlo simulations.

In practice, these different criteria are used at three stages:

- First, one of the Bai and $\operatorname{Ng}(2002)$ criteria is used to determine the optimal number of factors $r \in$ $\left\{1, \ldots, r_{\max }\right\}$ in a static contex ${ }^{[15}$;
- Then, $\operatorname{a\operatorname {VAR}}(p)$ is estimated on these $r$ estimated factors and the order $p$ of the VAR is selected to minimize the BIC criterion;
- Finally, the Bai and $\mathrm{Ng}(2007)$ criteria are applied to the variance-covariance matrix or correlation matrix of the residuals $\left(\varepsilon_{t}\right)$ of the $\operatorname{VAR}(p)$ to obtain the optimal number of dynamic factors $q$.


### Other Criteria

- Stock and Watson (2005a) and Amengual and Watson (2007) show that the Bai and Ng (2002) estimator can be used to estimate the number of dynamic factors.
- Breitung and Pigorsch (2013) propose two information criteria to select the number of dynamic factors.
- Hallin and Liska (2007) develop an information criterion for generalized dynamic factor (GDF) models.

# References

- Darné, Olivier, Karim Barhoumi, and Laurent Ferrara. “Dynamic Factor Models: A Review of the Literature,” 2018.