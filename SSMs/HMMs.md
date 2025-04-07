# Markov Model (a.k.a. Markov Chain)

A Markov model is a stochastic model used to represent a sequence of possible events where the probability of each event depends only on the state attained in the previous event. This property is known as the Markov property, which states that the future state depends only on the current state and not on the sequence of events that preceded it.

Mathematically, for a discrete-time Markov chain, the Markov property can be expressed as:

$$
P(Y_{t+1} = y \mid Y_t = y_t, Y_{t-1} = y_{t-1}, \dots, Y_0 = y_0) = P(Y_{t+1} = y \mid Y_t = y_t)
$$

where:
- $ Y_t $ denotes the state of the system at time $ t $.
- The conditional probability indicates that the next state depends only on the current state, not on the entire sequence of past states.

Using the rules of independence, we can calculate the joint probability of the sequence as: 

$$
P(y_1,...,y_T) = P(y_1)P(y_2|y_1)...P(y_{T}|y_{T-1}) = P(y_1) \prod_{t=2}^T P(y_t|y_{t-1})
$$

Markov chains assume that the conditional probability $P(y_t|y_{t-1})$ does not vary with time. Therefore, we can fully specify a Markov chain using three parameters:

- Number of States $(M)$: We generally assume that the observation can take one of $M$ states.

- Transition Matrix $(\mathbf{A})$: The transition matrix stores the probability of transition between the state $i$ to state $j$. Thus, the transition matrix can be represented as a $M \times M$ matrix where the entry $\mathbf{A}_{ij}$ is given by $\mathbf{A}_{ij} = P(y_t = j \mid y_{t-1} = i)$ where $i, j \in \{1, 2, \ldots, M\}$.

- Prior Probability $({\boldsymbol{\pi}})$: The probability of starting from one of the available states, denoted by ${\boldsymbol{\pi}}_i = P(y_1 = i)$ where $i \in \{1, 2, \ldots, M\}$.

# HMMs

## Assumptions

- Measurements: $(y_1,...,y_t)$
- Dynamic Hidden States: $(x_1,...,x_t)$
- **The Markov Property (Markovianity):** 
    $$p(x_t \mid x_{1:t-1}, y_{1:t-1}) = p(x_t \mid x_{t-1})$$
- **Conditional Independence of Measurements:** 
    $$p(y_t \mid x_{1:t}, y_{1:t-1}) = p(y_t \mid x_t)$$

Given only the measurements, we try to back out what the state was. As we move forward in time, the state evolves:

$$
X_0 \rightarrow X_1 \rightarrow \cdots \rightarrow X_t
$$

# Derivation: Predict Step

In the predict step, we will make an estimate about the current value of our state, given only the prior measurements. Suppose we start with our posterior from the previous timestep, which incorporates all measurements $ y_1, \ldots, y_{t-1} $. Via marginalization, we can rewrite the expression as:

$$
p(x_t \mid y_{1:t-1}) = \int_{x_{t-1}} p(x_t, x_{t-1} \mid y_{1:t-1})dx_{t-1}
$$

By the chain rule, we can factor:

$$
p(x_t \mid y_{1:t-1}) = \int_{x_{t-1}} p(x_t \mid x_{t-1}, y_{1:t-1})p(x_{t-1} \mid y_{1:t-1})dx_{t-1}
$$

By Markovianity, we can simplify the expression above, giving us the Predict Step:

$$
p(x_t \mid y_{1:t-1}) = \int_{x_{t-1}} p(x_t \mid x_{t-1})p(x_{t-1} \mid y_{1:t-1})dx_{t-1}
$$

# Derivation: Update Step

In what we call the update step, we use the current measurement to estimate the current state. We simply express the posterior using Bayes' Rule. Once again, we compute the denominator only by marginalization:

$$
p(x_t \mid y_{1:t}) = \frac{p(y_{1:t} \mid x_t)p(x_t)}{p(y_{1:t})} = \frac{p(y_{1:t} \mid x_t)p(x_t)}{\int_{x_t} p(y_{1:t} \mid x_t)p(x_t)dx_t}
$$

We can now factor the numerator with the chain rule:

$$
p(x_t \mid y_{1:t}) = \frac{p(y_t \mid x_t, y_{1:t-1})p(y_{1:t-1} \mid x_t)p(x_t)}{\int_{x_t} p(y_t \mid x_t, y_{1:t-1})p(y_{1:t-1} \mid x_t)p(x_t)dx_t}
$$

By the conditional independence of measurements:

$$
p(x_t \mid y_{1:t}) = \frac{p(y_t \mid x_t)p(y_{1:t-1} \mid x_t)p(x_t)}{\int_{x_t} p(y_t \mid x_t)p(y_{1:t-1} \mid x_t)p(x_t)dx_t}
$$

Interestingly enough, we can see above in the right hand side two terms from Bayes' Rule. We'll be able to collapse them into a single term. Using these two terms, the left side of Bayes' Rule would be:

$$
p(x_t \mid y_{1:t-1}) = \frac{p(y_{1:t-1} \mid x_t)p(x_t)}{\int_{x_t} p(y_{1:t-1} \mid x_t)p(x_t)dx_t}
$$

Multiplying by the denominator, we obtain:

$$
p(x_t \mid y_{1:t-1}) \int_{x_t} p(y_{1:t-1} \mid x_t)p(x_t)dx_t = p(y_{1:t-1} \mid x_t)p(x_t)
$$

Since by marginalization $ \int_{x_t} p(y_{1:t-1} \mid x_t)p(x_t)dx_t = p(y_{1:t-1}) $, we can simplify the line above to:

$$
p(x_t \mid y_{1:t-1})p(y_{1:t-1}) = p(y_{1:t-1} \mid x_t)p(x_t)
$$

We now plug this substitution into the numerator and in the denominator of the posterior:

$$
p(x_t \mid y_{1:t}) = \frac{p(y_t \mid x_t)p(x_t \mid y_{1:t-1})p(y_{1:t-1})}{\int_{x_t} p(y_t \mid x_t)p(x_t \mid y_{1:t-1})p(y_{1:t-1})dx_t}
$$

Since $ p(y_{1:t-1}) $ does not depend upon $ x $, we can pull it out of the integral, and the term cancels in the top and bottom:

$$
p(x_t \mid y_{1:t}) = \frac{p(y_t \mid x_t)p(x_t \mid y_{1:t-1})}{\int_{x_t} p(y_t \mid x_t)p(x_t \mid y_{1:t-1})dx_t}
$$

This is the closed form expression for the Update Step of the Bayes' Filter. Note that we use the result of our prediction to make the update:

The algorithm consists of repeatedly applying two steps: (1) the predict step, where we move forward the time step, and (2) the update step, where we incorporate the measurement.

It turns out that we can write out these integrals analytically for a very special family of distributions: Gaussian distributed random variables. This will be the Kalman Filter.

# Discrete Hidden Markov Model (HMM)

In a discrete Hidden Markov Model (HMM), discrete observations $y_t$ are generated based on discrete hidden states $x_t$. 

Formally, an HMM is defined by the following components:

1. **Hidden States**: A finite set of $ K $ hidden states $ S = \{x_1, x_2, \ldots, x_K\} $.

2. **Transition Matrix $ (\mathbf{A}) $**: An $ K \times K $ matrix where each entry $ \mathbf{A}_{ij} = P(x_j \mid x_i) $ represents the probability of transitioning from state $ x_i $ to state $ x_j $.

3. **Emission Matrix $ (\mathbf{B}) $**: An $ K \times M $ matrix where each entry $ \mathbf{B}_{ij} = P(y_j \mid x_i) $ represents the probability of emitting observation $ y_j $ from state $ x_i $, and $ M $ is the number of possible observations.

4. **Initial State Distribution $ ({\boldsymbol{\pi}}) $**: A vector $ {\boldsymbol{\pi}} $ where each entry $ {\boldsymbol{\pi}}_i = P(x_i) $ represents the probability of the system starting in state $ x_i $.

The joint probability of a sequence of hidden states $X = \{x_1, x_2, \ldots, x_T\} $ and observations $ Y = \{y_1, y_2, \ldots, y_T\} $ is given by:

$$
P(X,Y) = {\boldsymbol{\pi}}_{x_1} \prod_{t=2}^T \mathbf{A}_{x_{t-1}x_t} \prod_{t=1}^T \mathbf{B}_{x_t y_t}
$$
