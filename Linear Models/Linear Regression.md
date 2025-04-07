# Linear Regression

## Notation

- Number of observations: $T$
    - Indexed by: $t=1,...,T$
- Data Matrix: $ \mathbf{X} \in \mathbb{R}^{T \times d_x} $
    - Row: $\mathbf{x}_t \in \mathbb{R}^{d_x \times 1}$
    - Element: $x_{j,t}$
- Dependent variable: $ \mathbf{y} \in \mathbb{R}^{T \times 1} $
    - Element: $y_{t}$
- Error vector: $\boldsymbol{\epsilon}$
    - Element: $\epsilon_t$
- Coefficient Vectors: $\boldsymbol{\beta}$, $\boldsymbol{\phi}$

## Models

### Autoregression (AR) Model

The AR model expresses the dependent variable as a function of its own past values:

$$y_t = \phi_0 + \phi_1 y_{t-1} + \phi_2 y_{t-2} + \ldots + \phi_p y_{t-p} + \epsilon_t$$

Or in matrix notation:

$$\mathbf{y} = \mathbf{X}\boldsymbol{\phi} + \boldsymbol{\epsilon}$$

where 

$$
\mathbf{X} = \begin{bmatrix} 
1 & y_{p} & y_{p-1} & \cdots & y_{1} \\ 
1 & y_{p+1} & y_{p} & \cdots & y_{2} \\ 
\vdots & \vdots & \vdots & \ddots & \vdots \\ 
1 & y_{T-1} & y_{T-2} & \cdots & y_{T-p} 
\end{bmatrix}
\ \ \ \text{and} \ \ \
\boldsymbol{\phi} = \begin{bmatrix} \phi_0 \\ \phi_1 \\ \phi_2 \\ \vdots \\ \phi_p \end{bmatrix}
$$ 

### Static Model

The static model specifies a contemporaneous relationship between the dependent variable and explanatory variables:

$$y_t = \beta_0 + \beta_1 x_{1,t} + \ldots + \beta_{d_x} x_{d_x, t} + \epsilon_t$$

Or in matrix notation:

$$\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}$$

The design matrix $\mathbf{X}$ for the static model is:

$$\mathbf{X} = \begin{bmatrix} 
1 & x_{1,1} & x_{2,1} & \cdots & x_{d_x,1} \\ 
1 & x_{1,2} & x_{2,2} & \cdots & x_{d_x,2} \\ 
\vdots & \vdots & \vdots & \ddots & \vdots \\ 
1 & x_{1,T} & x_{2,T} & \cdots & x_{d_x,T} 
\end{bmatrix}$$

### Finite Distributed Lag (FDL) Model

The FDL model incorporates lagged effects of explanatory variables:

$$y_t = \alpha + \sum_{j=1}^{d_x} \sum_{\ell=0}^L \beta_{\ell,j} x_{j,t-\ell} + \epsilon_t$$

Or in matrix notation:

$$\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}$$

where

$$
\mathbf{X} = \begin{bmatrix} 
1 & x_{1,L+1} & x_{1,L} & \cdots & x_{1,1} & \cdots & x_{d_x,L+1} & \cdots & x_{d_x,1} \\ 
1 & x_{1,L+2} & x_{1,L+1} & \cdots & x_{1,2} & \cdots & x_{d_x,L+2} & \cdots & x_{d_x,2} \\ 
\vdots & \vdots & \vdots & \ddots & \vdots & \ddots & \vdots & \ddots & \vdots \\ 
1 & x_{1,T} & x_{1,T-1} & \cdots & x_{1,T-L} & \cdots & x_{d_x,T} & \cdots & x_{d_x,T-L} 
\end{bmatrix}
$$

$$
\boldsymbol{\beta} = \begin{bmatrix} 
\alpha \\ 
\beta_{0,1} \\ 
\beta_{1,1} \\ 
\vdots \\ 
\beta_{L,1} \\ 
\vdots \\ 
\beta_{0,d_x} \\ 
\vdots \\ 
\beta_{L,d_x}
\end{bmatrix}
$$

### Autoregressive Distributed Lag (ARDL) Model

The ARDL model includes both lagged dependent variables and lagged explanatory variables:

$$y_t = \alpha + \sum_{j=1}^{d_x} \sum_{\ell=0}^L \beta_{\ell,j} x_{j,t-\ell} + \sum_{p=1}^P \phi_p y_{t-p} + \epsilon_t$$

Or in matrix notation:

$$\mathbf{y} = \mathbf{X}\boldsymbol{\theta} + \boldsymbol{\epsilon}$$

where

$$
\mathbf{X} = \begin{bmatrix} 
1 & y_{p+L} & \cdots & y_{L+1} & x_{1,p+L+1} & \cdots & x_{1,L+1} & \cdots & x_{d_x,p+L+1} & \cdots & x_{d_x,L+1} \\ 
1 & y_{p+L+1} & \cdots & y_{L+2} & x_{1,p+L+2} & \cdots & x_{1,L+2} & \cdots & x_{d_x,p+L+2} & \cdots & x_{d_x,L+2} \\ 
\vdots & \vdots & \ddots & \vdots & \vdots & \ddots & \vdots & \ddots & \vdots & \ddots & \vdots \\ 
1 & y_{T-1} & \cdots & y_{T-p} & x_{1,T} & \cdots & x_{1,T-L} & \cdots & x_{d_x,T} & \cdots & x_{d_x,T-L} 
\end{bmatrix}
$$

$$
\boldsymbol{\theta} = \begin{bmatrix} 
\alpha \\ 
\phi_1 \\ 
\vdots \\ 
\phi_P \\ 
\beta_{0,1} \\ 
\vdots \\ 
\beta_{L,1} \\ 
\vdots \\ 
\beta_{0,d_x} \\ 
\vdots \\ 
\beta_{L,d_x}
\end{bmatrix}
$$

Note: The first observation available for estimation is at $t=K$ where $K = \max(P,L)$ since earlier observations are used as lags.