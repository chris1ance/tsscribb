# Data

- $\mathbf{y} \in \mathbb{R}^T$ with row $y_t \in \mathbb{R}$

- $\mathbf{X} \in \mathbb{R}^{T \times d}$ with row $\mathbf{x}_t \in \mathbb{R}^d$

# Assumptions

- $\mathbf{x}_t \in \mathbb{R}^d$ does not contain any elements from $\mathbf{y}$
- At time $t$, $\mathbf{x}_t$ is known and observable, but $\mathbf{y}_t$ is not.
- $\mathbf{y}$ and $\mathbf{X}$ are indexed by $t \in \{1, ..., T\}$ and concatenation occurs along the time axis and drops any missing values.

# Parameters

- $L^y \in \mathbb{N}^{+} \cup \{0\}$, lags of $y_t$ 
- $L^x \in \mathbb{N}^{+} \cup \{0\}$, lags of $\mathbf{x}_t$
- $h \in \mathbb{N}^{+}$, forecast horizon
- $T_c \in (\max(L^x, L^y)+(h-1)+1, ..., T)$, cutoff date for model training

#  Direct Forecasting Algorithm 

0. Define $\bar{L} = \max(L^x, L^y)$

1. If $L^x > 0$, define the lagged matrices $\mathbf{X}^{(\ell)}$ for each lag $\ell \in {1, ..., L^x}$, where $\mathbf{X}^{(\ell)}$ is $\mathbf{X}$ shifted downward by $\ell$ rows, removing the first $\ell$ rows. Concatenate all lagged versions to form:

    $$
    \mathbf{X}_{L} =
    \begin{bmatrix}
    \mathbf{X} & \mathbf{X}^{(1)} & \mathbf{X}^{(2)} & \dots & \mathbf{X}^{(L^x)}
    \end{bmatrix}
    \in \mathbb{R}^{(T - \bar{L}) \times (d (L^x + 1))}
    $$

    Else: 

    $$
    \mathbf{X}_L = \mathbf{X} \in \mathbb{R}^{(T - \bar{L}) \times (d (L^x + 1))}
    $$

2. If $L^y > 0$, define lagged vectors $\mathbf{y}^{(\ell)}$ for each lag $\ell \in {1, ..., L^y}$, where $\mathbf{y}^{(\ell)}$ is $\mathbf{y}$ shifted downward by $\ell$ rows, removing the first $\ell$ rows. Concatenate all lagged versions to form:

    $$
    \mathbf{Y}_L =
    \begin{bmatrix}
    \mathbf{y}^{(1)} & \mathbf{y}^{(2)} & \dots & \mathbf{y}^{(L^y)}
    \end{bmatrix}
    \in \mathbb{R}^{(T - \bar{L}) \times L^y}
    $$

3. If $L^y > 0$, concatenate $\mathbf{X}_L$ and $\mathbf{Y}_L$ to form:

    $$
    \tilde{\mathbf{Z}} = 
    \begin{bmatrix}
    \mathbf{X}_L & \mathbf{Y}_L
    \end{bmatrix}
    \in \mathbb{R}^{(T - \bar{L}) \times d_{train}}, \ \ \ \ \text{where} \ \ \ \ d_{train} = d(L^x + 1) + L^y
    $$

    Else: 

    $$
    \tilde{\mathbf{Z}} = \mathbf{X}_L \in \mathbb{R}^{(T - \bar{L}) \times d_{train}}
    $$

4. Form $$\mathbf{Z} \in \mathbb{R}^{(T - \bar{L} - (h-1)) \times d_{train}}$$ by shifting $\tilde{\mathbf{Z}}$ down by $h-1$ rows and removing the first $h-1$ rows.

5. Define $$T_{train} = T_c - \bar{L} - (h-1)$$ as the number of training observations with index $t \in \{\bar{L}+(h-1)+1, ..., T_c\}$

6. Form $\mathbf{y}_{train} \in \mathbb{R}^{T_{train}}$ by dropping from $\mathbf{y}$ observations:
    - $1$ through $\bar{L}+(h-1)$; and
    - $T_c+1$ through $T$

7. Form $\mathbf{Z}_{train} \in \mathbb{R}^{T_{train} \times d_{train}}$ by dropping from $\mathbf{Z}$ observations $T_c+1$ through $T$

8. Extract $\mathbf{z}_{test}$ as row $(T_c+1)$ of $\mathbf{Z}$

9. If $T_c+h \leq T$, extract $y_{test}$ as row $(T_c+h)$ of $\mathbf{y}$

10. Fit a forecasting model $f(y_t|\mathbf{z}_t)$ to $\mathbf{Z}_{train}$ and $\mathbf{y}_{train}$, yielding a fit model $\hat{f}(y_t|\mathbf{z}_t)$

11. Forecast $\mathbf{y}_{test}$ using $\hat{f}(\cdot|\mathbf{z}_{test})$, yielding $\hat{y}_{test}$

