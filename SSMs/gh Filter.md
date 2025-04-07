# 1. Overview

The g-h filter is a simple, linear, predictive filter used for smoothing and tracking time series data. It is particularly useful for estimating a quantity (often position) and its rate of change (often velocity) in the presence of noisy measurements. 

- $g$ (also called $\alpha$) is the smoothing factor (gain) used to update the position estimate.
- $h$ (also called $\beta$) is the smoothing factor (gain) used to update the velocity (or first derivative) estimate.

# 2. Basic Assumptions and Notation

Let:

- $ x_k $ be the true (but unknown) position at discrete time step $ k $.
- $ \dot{x}_k $ be the true (but unknown) velocity at time step $ k $.
- $ z_k $ be the noisy measurement of the position at time step $ k $.
- $ \Delta t $ be the time interval between consecutive steps (assumed constant for simplicity).
- $ \hat{x}_{k|k} $ denote the estimate of position at time $ k $ given all measurements up to (and including) time $ k $.
- $ \hat{\dot{x}}_{k|k} $ denote the estimate of velocity at time $ k $ given all measurements up to (and including) time $ k $.

In the g-h filter, we typically assume a constant-velocity model, i.e.,

$$ x_{k+1} = x_k + \dot{x}_k \Delta t \quad \text{and} \quad \dot{x}_{k+1} = \dot{x}_k. $$

Under noise-free conditions, that model would hold exactly. In the presence of noisy measurements, we apply a correction step to the prediction using the gains $g$ and $h$.

# 3. Predict Step

1. **Predict the next position** based on the previous state estimates:

$$ \hat{x}_{k|k-1} = \hat{x}_{k-1|k-1} + \hat{\dot{x}}_{k-1|k-1} \Delta t $$

2. **Predict the next velocity** under the constant-velocity assumption (it remains unchanged):

$$ \hat{\dot{x}}_{k|k-1} = \hat{\dot{x}}_{k-1|k-1} $$

Here:
- $\hat{x}_{k|k-1}$ means "our estimate of position at time $k$ given measurements up to $k-1$." 

- $\hat{\dot{x}}_{k|k-1}$ is the velocity estimate at time $k$ before incorporating the measurement at $k$.

# 4. Update Step

When a new measurement $z_k$ arrives at time $k$, we incorporate this measurement to update our predicted estimates.

1. **Update the position** estimate by mixing the prediction with the measurement:

$$ \hat{x}_{k|k} = \hat{x}_{k|k-1} + g \left( z_k - \hat{x}_{k|k-1} \right) $$

where $g$ is a constant gain ($0 < g < 1$) determining how heavily to weight the new measurement versus the previous prediction.

2. **Update the velocity** estimate using the same measurement residual:

$$ \hat{\dot{x}}_{k|k} = \hat{\dot{x}}_{k|k-1} + h \left( \frac{z_k - \hat{x}_{k|k-1}}{\Delta t} \right) $$

where $h$ (often $\beta$) is another constant gain for updating the velocity. Hence, the filter "trusts" the predicted velocity to remain the same unless the new measurement suggests that the velocity needs to be adjusted.

Constants $g$ and $h$ (or $\alpha$ and $\beta$) govern how the filter responds to new measurements:

- **Large $g$**: The position estimate heavily follows the new measurement (less smoothing).
- **Small $g$**: The position estimate is smoothed heavily by previous estimates (more smoothing).
- **Large $h$**: The velocity is updated quickly in response to new measurements; can be noisy.
- **Small $h$**: The velocity changes slowly over time; more stable but slower to respond to actual changes.

# 5. Complete Algorithm

1. **Initialization (at $k = 0$)**:

   - $\hat{x}_0 \leftarrow$ initial position estimate
   - $\hat{\dot{x}}_0 \leftarrow$ initial velocity estimate
   - $g, h \leftarrow$ chosen gains (constants)
   - $\Delta t \leftarrow$ sampling period

2. **Iteration (for $k = 1, 2, 3, \ldots$)**:

   1. **Predict step**:
      $$
      \hat{x}_{k|k-1} = \hat{x}_{k-1} + \hat{\dot{x}}_{k-1} \Delta t,
      $$
      $$
      \hat{\dot{x}}_{k|k-1} = \hat{\dot{x}}_{k-1}
      $$

   2. **Obtain measurement:** $z_k$

   3. **Compute residual**:
      $$
      \epsilon = z_k - \hat{x}_{k|k-1}
      $$

   4. **Update step**:
      $$
      \hat{x}_k = \hat{x}_{k|k-1} + g \epsilon
      $$
      $$
      \hat{\dot{x}}_k = \hat{\dot{x}}_{k|k-1} + h \left[ \frac{\epsilon}{\Delta t} \right]
      $$

