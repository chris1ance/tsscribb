# Inputs

- Data:
    - $\mathbf{y} = (y_1, y_2, \dots, y_T)^\top \in \mathbb{R}^{T}$ with scalar observation $y_t \in \mathbb{R}$ at time $t$
    - $\mathbf{X} \in \mathbb{R}^{T \times d}$ with row $\mathbf{x}_t^{\top} \in \mathbb{R}^{d \times 1}$

- $\mathbf{L}^x = \{\ell^x_1,\ell^x_2,\dots,\ell^x_K\}$, where each element $\ell^x_k \in \mathbb{N}^{+} \cup \{0\}$ is a lag of $\mathbf{X}$ to include as predictors.

    - $\mathbf{L}^x$ may be empty

    - Denote $\bar{L}^x \equiv \text{max}\{\ell^x_1,\ell^x_2,\dots,\ell^x_K\}$, where we set $\bar{L}^x = 0$ if $\mathbf{L}^x = \emptyset$.

- $L^y \in \mathbb{N}^{+} \cup \{0\}$ if $\mathbf{L}^x$ is non-empty, else $L^y \in \mathbb{N}^{+}$. Lags of $y_t$ to include as predictors.

- $h \in \mathbb{N}^{+} \cup \{0\}$, forecast horizon

- $T_{cutoff}$: Cutoff date for model training

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
    \in \mathbb{R}^{(T - L^y) \times L^y}
    $$

    More explicitly, we first form:

    <div align="center">

    | $t$           | $\mathbf{y}^{(1)}_t$ | $\mathbf{y}^{(2)}_t$ | $\dots$ | $\mathbf{y}^{(L^y)}_t$ |
    |:-----------:|:--------------------:|:--------------------:|:-------:|:---------------------:|
    | 1           | NaN                  | NaN                  | ...     | NaN                  |
    | 2           | $y_1$                | NaN                  | ...     | NaN                  |
    | 3           | $y_2$                | $y_1$                | ...     | NaN                  |
    | 4           | $y_3$                | $y_2$                | ...     | NaN                  |
    | $\dots$     | $\dots$              | $\dots$              | $\dots$ | $\dots$              |
    | $L^y\!-\!1$ | $y_{L^y-2}$          | $y_{L^y-3}$          | $\dots$ | NaN                  |
    | $L^y$       | $y_{L^y-1}$          | $y_{L^y-2}$          | $\dots$ | NaN                  |
    | $L^y\!+\!1$ | $y_{L^y}$            | $y_{L^y-1}$          | $\dots$ | $y_1$                |
    | $L^y\!+\!2$ | $y_{L^y+1}$          | $y_{L^y}$            | $\dots$ | $y_2$                |
    | $\dots$     | $\dots$              | $\dots$              | $\dots$ | $\dots$              |
    | $T$         | $y_{T-1}$            | $y_{T-2}$            | $\dots$ | $y_{T-L^y}$          |

    </div>

    Finally, to form $\mathbf{Y}_L$, drop rows with any $NaN$, such that $\mathbf{Y}_L$ is then equal to:


    <div align="center">

    | $t$           | $\mathbf{y}^{(1)}_t$ | $\mathbf{y}^{(2)}_t$ | $\dots$ | $\mathbf{y}^{(L^y)}_t$ |
    |:-----------:|:--------------------:|:--------------------:|:-------:|:---------------------:|
    | $L^y\!+\!1$ | $y_{L^y}$            | $y_{L^y-1}$          | $\dots$ | $y_1$                |
    | $L^y\!+\!2$ | $y_{L^y+1}$          | $y_{L^y}$            | $\dots$ | $y_2$                |
    | $\dots$     | $\dots$              | $\dots$              | $\dots$ | $\dots$              |
    | $T$         | $y_{T-1}$            | $y_{T-2}$            | $\dots$ | $y_{T-L^y}$          |

    </div>


2. If $\mathbf{L}^x$ is **not** empty, for each $\ell^x_k \in \mathbf{L}^x$, define $\mathbf{X}^{(\ell^x_k)}$ as $\mathbf{X}$ shifted downward by $\ell^x_i$ rows (with the first $\ell^x_i$ rows filled by NaN). Then form

    $$
    \mathbf{X}_L
    =
    \begin{bmatrix}
    \mathbf{X}^{(\ell^x_1)} & \mathbf{X}^{(\ell^x_2)} & \dots & \mathbf{X}^{(\ell^x_K)}
    \end{bmatrix}
    \in \mathbb{R}^{(T-\bar{L}^x)\times dK}
    $$

    where if $\mathbf{L}^x$ includes 0, then $\mathbf{X}^{(0)} = \mathbf{X}$. 

    $\mathbf{X}_L$ is then equal to:

    <div align="center">

    | $t$   | $\mathbf{X}^{(\ell^x_1)}$ | $\mathbf{X}^{(\ell^x_2)}$ | $\dots$ | $\mathbf{X}^{(\ell^x_K)}$ |
    |:-----------:|:--------------------:|:--------------------:|:-------:|:---------------------:|
    | $\bar{L}^x\!+\!1$ | $\mathbf{x}_{\bar{L}^x}$    | $\mathbf{x}_{\bar{L}^x-1}$    | $\dots$ | $\mathbf{x}_1$     |
    | $\bar{L}^x\!+\!2$ | $\mathbf{x}_{\bar{L}^x+1}$   | $\mathbf{x}_{\bar{L}^x}$    | $\dots$ | $\mathbf{x}_2$      |
    | $\dots$     | $\dots$     | $\dots$  | $\dots$ | $\dots$              |
    | $T$         | $\mathbf{x}_{T-1}$    | $\mathbf{x}_{T-2}$     | $\dots$ | $\mathbf{x}_{T-\bar{L}^x}$     |

    </div>

    Where we have assumed, without loss of generality, that $\ell^x_K = \bar{L}^x$.


3. Concatenate $\mathbf{X}_L$ and $\mathbf{Y}_L$ to form:

    $$
    \tilde{\mathbf{Z}} = 
    \begin{bmatrix}
    \mathbf{Y}_L & \mathbf{X}_L
    \end{bmatrix}
    \in \mathbb{R}^{(T - \bar{L}+1) \times d_{train}}, \ \ \ \text{where} \ \ \ d_{train} = dK + L^y
    $$

    where $\bar{L} = \max\{ \bar{L}^x+1, L^y+1 \}$. 
    
    Then $\tilde{\mathbf{Z}}$ is equal to:

    <div align="center">

    | $t$   | $\tilde{\mathbf{Z}}$ |
    |:-----------:|:--------------------:|
    | $\bar{L}$ | $\tilde{\mathbf{z}}_{\bar{L}}$   | 
    | $\dots$     | $\dots$     |
    | $h+(\bar{L}+1)$ | $\tilde{\mathbf{z}}_{h+(\bar{L}+1)}$   | 
    | $\dots$     | $\dots$     |
    | $T$         | $\tilde{\mathbf{z}}_{T}$    | 

    </div>    


4. Form $$\tilde{\mathbf{y}}_{train} \in \mathbb{R}^{T-(\bar{L}+h)}$$ by dropping from $\mathbf{y}$ the first $\bar{L}+h$ observations, such that $\tilde{\mathbf{y}}_{train}$ is equal to:

    <div align="center">

    | $t$           | $\tilde{\mathbf{y}}_{train}$ |
    |:-----------:|:--------------------:|
    | $(\bar{L}+h)+1 = (\bar{L}+1) + h$   | $y_{(\bar{L}+1) + h}$           |
    | $\dots$     | $\dots$             |
    | $T$         | $y_{T}$           |

    </div>


5. Form $$\mathbf{Z} \in \mathbb{R}^{(T - (\bar{L}+h)) \times d_{train}}$$ by shifting $\tilde{\mathbf{Z}}$ by $h$ steps. Then $\mathbf{Z}$ is equal to:

    <div align="center">

    | $t$   | $\mathbf{Z}$ |
    |:-----------:|:--------------------:|
    | $h+(\bar{L}+1)$ | $\mathbf{z}_{(\bar{L}+1)}$   | 
    | $\dots$     | $\dots$     |
    | $T$         | $\mathbf{z}_{T-h}$    | 

    </div>  

6. Define $$T_{train} = T_{cutoff} - \bar{L} - h$$ corresponding to observations indexed by $$t \in \{\bar{L}+h+1, ..., T_{cutoff}\}$$ 

    Form $\mathbf{y}_{train} \in \mathbb{R}^{T_{train}}$ by dropping from $\tilde{\mathbf{y}}_{train}$ observations in [$T_{cutoff}+1$, $T$]. 
    
    Form $\mathbf{Z}_{train} \in \mathbb{R}^{T_{train} \times d_{train}}$ by dropping from $\mathbf{Z}$ observations in [$T_{cutoff}+1$, $T$].


7. Extract $\mathbf{z}_{test}$ as row $(T_{cutoff}+1)$ of $\mathbf{Z}$


8. If $T_{cutoff}+1+h \leq T$, extract $y_{test}$ as row $(T_{cutoff}+1+h)$ of $\mathbf{y}$


9. Fit a forecasting model $f(y_{t}|\mathbf{z}_{t-h})$ to $\mathbf{y}_{train}$ and $\mathbf{Z}_{train}$, yielding a fit model $\hat{f}(\mathbf{z}_{t-h})$


10. Forecast $\mathbf{y}_{test}$ using $\hat{f}(\mathbf{z}_{test})$, yielding $\hat{y}_{test}$