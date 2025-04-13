# Inputs

- Data:
    - $\mathbf{y} = (y_1, y_2, \dots, y_T)^\top \in \mathbb{R}^{T}$ with scalar observation $y_t \in \mathbb{R}$ at time $t$
    - $\mathbf{X} \in \mathbb{R}^{T \times d}$ with row $\mathbf{x}_t \in \mathbb{R}^d$

- $\mathbf{L}^x = \{\ell^x_1,\ell^x_2,\dots,\ell^x_K\}$, where each element $\ell^x_k \in \mathbb{N}^{+} \cup \{0\}$ is a lag of $\mathbf{X}$ to include as predictors.

    - $\mathbf{L}^x$ may be empty

- $L^y \in \mathbb{N}^{+} \cup \{0\}$ if $\mathbf{L}^x$ is non-empty, else $L^y \in \mathbb{N}^{+}$. Lags of $y_t$ to include as predictors.

- $h \in \mathbb{N}^{+}$, forecast horizon

- $T_{cutoff}$: Cutoff date for model training, where $$T_{cutoff} \in (\bar{L}+(h-1)+1, ..., T) = (\bar{L}+h, ..., T)$$ for $\bar{L} = \max(\ell^x_1,\ell^x_2,\dots,\ell^x_K, L^y)$

# Assumptions

- $\mathbf{x}_t \in \mathbb{R}^d$ does not contain any elements from $\mathbf{y}$
- At time $t$, $\mathbf{x}_t$ is known and observable, but $\mathbf{y}_t$ is not.
- Data vectors and matrices are indexed by $t$ and concatenation occurs along the time axis and drops any rows with any missing values ($NaN$).

#  Direct Forecasting Algorithm 

1. If $L^y > 0$, for each lag $\ell \in {1, ..., L^y}$, define $\mathbf{y}^{(\ell)}$ as $\mathbf{y}$ shifted downward by $\ell$ rows (with the first $\ell$ rows filled by NaN). Concatenate all $\mathbf{y}^{(\ell)}$ to form:

    $$
    \mathbf{Y}_L =
    \begin{bmatrix}
    \mathbf{y}^{(1)} & \mathbf{y}^{(2)} & \dots & \mathbf{y}^{(L^y)}
    \end{bmatrix}
    \in \mathbb{R}^{(T - \bar{L}) \times L^y}
    $$

2. If $\mathbf{L}^x$ is **not** empty, for each $\ell^x_k \in \mathbf{L}^x$, define $\mathbf{X}^{(\ell^x_k)}$ as $\mathbf{X}$ shifted downward by $\ell^x_i$ rows (with the first $\ell^x_i$ rows filled by NaN). Then form

    $$
    \mathbf{X}_L
    =
    \begin{bmatrix}
    \mathbf{X}^{(\ell^x_1)} & \mathbf{X}^{(\ell^x_2)} & \dots & \mathbf{X}^{(\ell^x_K)}
    \end{bmatrix}
    \in \mathbb{R}^{(T-\bar{L})\times dK}
    $$

    where if $\mathbf{L}^x$ includes 0, then $\mathbf{X}^{(0)} = \mathbf{X}$. 

3. Concatenate $\mathbf{X}_L$ and $\mathbf{Y}_L$ to form:

    $$
    \tilde{\mathbf{Z}} = 
    \begin{bmatrix}
    \mathbf{Y}_L & \mathbf{X}_L
    \end{bmatrix}
    \in \mathbb{R}^{(T - \bar{L}) \times d_{train}}, \ \ \ \text{where} \ \ \ d_{train} = dK + L^y
    $$

4. Form $$\mathbf{Z} \in \mathbb{R}^{(T - \bar{L} - (h-1)) \times d_{train}}$$ by shifting $\tilde{\mathbf{Z}}$ downward by $(h-1)$ rows, dropping the top $(h-1)$ rows to align predictors at time $t$ with the target at time $t+h$.

5. Define $$T_{train} = T_{cutoff} - \bar{L} - (h-1)$$ corresponding to observations indexed by $$t \in \{\bar{L}+(h-1)+1, ..., T_{cutoff}\} = \{\bar{L}+h, ..., T_{cutoff}\}$$

6. Form $\mathbf{y}_{train} \in \mathbb{R}^{T_{train}}$ by dropping from $\mathbf{y}$ observations in [$1$, $\bar{L}+(h-1)$] and [$T_{cutoff}+1$, $T$]

7. Form $\mathbf{Z}_{train} \in \mathbb{R}^{T_{train} \times d_{train}}$ by dropping from $\mathbf{Z}$ observations in [$T_{cutoff}+1$, $T$]

8. Extract $\mathbf{z}_{test}$ as row $(T_{cutoff}+1)$ of $\mathbf{Z}$

9. If $T_{cutoff}+h \leq T$, extract $y_{test}$ as row $(T_{cutoff}+h)$ of $\mathbf{y}$

10. Fit a forecasting model $f(y_t|\mathbf{z}_t)$ to $\mathbf{Z}_{train}$ and $\mathbf{y}_{train}$, yielding a fit model $\hat{f}(y_t|\mathbf{z}_t)$

11. Forecast $\mathbf{y}_{test}$ using $\hat{f}(\cdot|\mathbf{z}_{test})$, yielding $\hat{y}_{test}$