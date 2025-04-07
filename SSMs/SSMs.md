# SSMs

A state-space model (SSM) is a partially observed Markov model, in which the hidden state, $ \mathbf{z}_t $, evolves over time according to a Markov process, and each hidden state generates  observation $ \mathbf{y}_t $ at each time step. The main goal is to infer the hidden states given the observations. However, we may also be interested in using the model to predict future observations (e.g., for time-series forecasting).

An SSM can be represented as a stochastic discrete time nonlinear dynamical system of the form

$$
\mathbf{z}_t = f(\mathbf{z}_{t-1}, \mathbf{u}_t, \mathbf{q}_t)
$$

$$
\mathbf{y}_t = h(\mathbf{z}_t, \mathbf{u}_t, \mathbf{y}_{1:t-1}, \mathbf{r}_t)
$$

where $ \mathbf{z}_t \in \mathbb{R}^{d_z} $ are the hidden states, $ \mathbf{u}_t \in \mathbb{R}^{d_u} $ are optional observed inputs, $ \mathbf{y}_t \in \mathbb{R}^{d_y} $ are observed outputs, $ f $ is the transition function, $ \mathbf{q}_t $ is the process noise, $ h $ is the observation function, and $ \mathbf{r}_t $ is the observation noise.

Rather than writing this as a deterministic function of random noise, we can represent it as a probabilistic model:

$$
p(\mathbf{z}_t|\mathbf{z}_{t-1}, \mathbf{u}_t) = p(\mathbf{z}_t|f(\mathbf{z}_{t-1}, \mathbf{u}_t))
$$

$$
p(\mathbf{y}_t|\mathbf{z}_t, \mathbf{u}_t, \mathbf{y}_{1:t-1}) = p(\mathbf{y}_t|h(\mathbf{z}_t, \mathbf{u}_t, \mathbf{y}_{1:t-1}))
$$

where $ p(\mathbf{z}_t|\mathbf{z}_{t-1}, \mathbf{u}_t) $ is the transition model, and $ p(\mathbf{y}_t|\mathbf{z}_t, \mathbf{u}_t, \mathbf{y}_{1:t-1}) $ is the observation model.

Unrolling over time, we get the following joint distribution:

$$
p(\mathbf{y}_{1:T}, \mathbf{z}_{1:T} | \mathbf{u}_{1:T}) = \left[ p(\mathbf{z}_1|\mathbf{u}_1) \prod_{t=2}^{T} p(\mathbf{z}_t|\mathbf{z}_{t-1}, \mathbf{u}_t) \right] \left[ \prod_{t=1}^{T} p(\mathbf{y}_t|\mathbf{z}_t, \mathbf{u}_t, \mathbf{y}_{1:t-1}) \right]
$$

It is common to assume that the observations are conditionally independent of each other given the hidden state. In this case the joint simplifies to

$$
p(\mathbf{y}_{1:T}, \mathbf{z}_{1:T} | \mathbf{u}_{1:T}) = \left[ p(\mathbf{z}_1|\mathbf{u}_1) \prod_{t=2}^{T} p(\mathbf{z}_t|\mathbf{z}_{t-1}, \mathbf{u}_t) \right] \left[ \prod_{t=1}^{T} p(\mathbf{y}_t|\mathbf{z}_t) \right]
$$

Sometimes there are no external inputs, so the model further simplifies to the following unconditional generative model:

$$
p(\mathbf{y}_{1:T}, \mathbf{z}_{1:T}) = \left[ p(\mathbf{z}_1) \prod_{t=2}^{T} p(\mathbf{z}_t|\mathbf{z}_{t-1}) \right] \left[ \prod_{t=1}^{T} p(\mathbf{y}_t|\mathbf{z}_t) \right]
$$
