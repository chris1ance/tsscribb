# Dropping Rows

Given a matrix indexed by $t=1,...,T$, after dropping the first $N$ rows, the resulting matrix has $T - N$ rows.

**Proof:**

The rows of the original matrix are indexed by 

$$ \{1, 2, 3, \dots, T\} $$ 

This set has cardinality $T$. 

We remove the rows with indices

$$ \{1, 2, 3, \dots, N\} $$

After removing those $N$ rows, the set of indices left is

$$ \{N + 1, N + 2, \dots, T\} $$

The remaining set has size

$$ (T - (N + 1) + 1) = T - N $$

Hence, exactly $T - N$ rows remain in the matrix after dropping the first $N$ rows.

# Combining Matrices

Given $\mathbf{A}_{t':T}$ and $\mathbf{B}_{t'':T}$ for $t', t'' \in \{1,...,T\}$, concatenating $\mathbf{A}_{t':T}$ and $\mathbf{B}_{t'':T}$ and dropping and rows with indices that are not in both matrices results in a matrix $\mathbf{C}_{max(t',t''):T}$ with a number of rows equal to 

$$T-\text{max}(t',t'')+1$$

This follows because in forming $\mathbf{C}_{max(t',t''):T}$, we have removed rows with indices

$$ \{1, 2, \dots, \text{max}(t',t'')-1\} $$


# Lagging

# One Lag

Given $\mathbf{X} \in \mathbb{R}^{T \times d}$ with row $\mathbf{x}_t^{\top} \in \mathbb{R}^{d \times 1}$, for any lag $L$, the shifting is given by:

<div align="center">

| $t$           |  |
|:-----------:|:--------------------:|
| 1           | $\mathbf{x}_{1-L}^{\top}$           |
| 2           | $\mathbf{x}_{2-L}^{\top}$           |
| 3           | $\mathbf{x}_{3-L}^{\top}$           |
| 4           | $\mathbf{x}_{4-L}^{\top}$           |
| $\dots$     | $\dots$             |
| $L\!-\!1$  | $\mathbf{x}_{(L-1)-L = -1}^{\top}$            |
| $L$         | $\mathbf{x}_{L-L = 0}^{\top}$             |
| $L\!+\!1$  | $\mathbf{x}_{(L+1)-L = 1}^{\top}$             |
| $L\!+\!2$  | $\mathbf{x}_{(L+2)-L =2}^{\top}$             |
| $\dots$     | $\dots$             |
| $T$         | $\mathbf{x}_{T-L}^{\top}$           |

</div>

To form $\mathbf{X}_L$, drop rows with $t-L \leq 0$. Equivalently, whenever $t \leq L$, the computed index $t - L$ is non-positive, therefore we mark that cell as NaN since $\mathbf{x}_0$, $\mathbf{x}_{-1}$, etc. do not exist:

<div align="center">

| $t$           |  |
|:-----------:|:--------------------:|
| 1           | $NaN$           |
| 2           | $NaN$           |
| 3           | $NaN$           |
| 4           | $NaN$           |
| $\dots$     | $\dots$             |
| $L\!-\!1$  | $NaN$            |
| $L$         | $NaN$             |
| $L\!+\!1$  | $\mathbf{x}_{1}^{\top}$             |
| $L\!+\!2$  | $\mathbf{x}_{2}^{\top}$             |
| $\dots$     | $\dots$             |
| $T$         | $\mathbf{x}_{T-L}^{\top}$           |

</div>

This yields:

<div align="center">

| $t$           | $\mathbf{X}^{(L)}$ |
|:-----------:|:--------------------:|
| $L\!+\!1$  | $\mathbf{x}_{(L+1)-L = 1}^{\top}$             |
| $L\!+\!2$  | $\mathbf{x}_{(L+2)-L =2}^{\top}$             |
| $\dots$     | $\dots$             |
| $T$         | $\mathbf{x}_{T-L}^{\top}$           |

</div>

Since we have dropped the first $L$ rows, the resulting matrix has $T-L$ rows.
