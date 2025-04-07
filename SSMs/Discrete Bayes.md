# 1. Discrete Bayesian Inference

Bayesian inference is a method for updating our belief (or probability distribution) about some unknown variable $Z$ given observations (or measurements) $Y$. When both $Z$ and $Y$ take on discrete values (i.e., come from finite or countably infinite sets), we speak of Discrete Bayes.

Essentially, we begin with a prior distribution over the possible states of $Z$. Once we gather evidence $Y = y$, we update our belief about $Z$ using the likelihood of observing $y$ given each possible state $z$. The result is the posterior distribution.

Formally, if $z \in \mathcal{Z} = \{z_1, z_2, \ldots, z_K\}$ are the discrete possible states of $Z$, then:

1. Prior:

   $$P(Z = z_k) \quad \text{for} \quad k = 1,2,\ldots,K$$

2. Likelihood:

   $$P(Y = y \mid Z = z_k)$$

3. Posterior (the updated distribution after observing $Y = y$):

   $$P(Z = z_k \mid Y = y) = \frac{P(Z = z_k)P(Y = y \mid Z = z_k)}{\sum_{j=1}^K[P(Z = z_j)P(Y = y \mid Z = z_j)]}$$

The denominator

$$\sum_{j=1}^K P(Z = z_j)P(Y = y \mid Z = z_j)$$

acts as a normalization constant to ensure that the posterior probabilities sum up to 1.

# 2. Mathematical Details

## 2.1. Bayesian Formula for Discrete Variables

For discrete variables, the Bayesian update formula is typically written as:

$$P(z_k \mid y) = \frac{P(y \mid z_k)P(z_k)}{\sum_{j=1}^K P(y \mid z_j)P(z_j)}$$

Here:

- $P(z_k)$ is the prior probability of state $z_k$.
- $P(y \mid z_k)$ is the likelihood of the observation $y$ given $z_k$.
- $\sum_j P(y \mid z_j)P(z_j)$ is the evidence or normalizing constant.

The evidence ensures that the posterior probabilities across all $k$ sum to 1:

$$\sum_{k=1}^K P(z_k \mid y) = 1$$

## 2.2. Sequential Updates

If you receive a sequence of observations $y_1, y_2, \ldots, y_t$ over time and want to update beliefs incrementally, you can do this iteratively:

1. Let $P_t(z_k)$ denote the belief (posterior) at time $t$.

2. At time $t + 1$, you observe a new measurement $y_{t+1}$. Then update:

   $$P_{t+1}(z_k) = \frac{P(y_{t+1} \mid z_k)P_t(z_k)}{\sum_{j=1}^K P(y_{t+1} \mid z_j)P_t(z_j)}$$

### 2.2.1 Equivalence of Sequential and Batch Updates

To show that sequential updates are equivalent to computing the posterior with all observations at once, let's consider two observations $y_1$ and $y_2$ (the proof extends to any number of observations).

**Sequential Update Approach:**

1. First update with $y_1$:
   $$P(z_k \mid y_1) = \frac{P(y_1 \mid z_k)P(z_k)}{\sum_{j=1}^K P(y_1 \mid z_j)P(z_j)}$$

2. Then use this as prior and update with $y_2$:
   $$P(z_k \mid y_1, y_2) = \frac{P(y_2 \mid z_k)P(z_k \mid y_1)}{\sum_{j=1}^K P(y_2 \mid z_j)P(z_j \mid y_1)}$$

**Batch Update Approach:**

Computing posterior with both observations at once:
$$P(z_k \mid y_1, y_2) = \frac{P(y_1, y_2 \mid z_k)P(z_k)}{\sum_{j=1}^K P(y_1, y_2 \mid z_j)P(z_j)}$$

**These approaches are equivalent because:**

Assuming conditional independence of observations given the state:

$$P(y_1, y_2 \mid z_k) = P(y_1 \mid z_k)P(y_2 \mid z_k)$$

Substituting the Sequential Update Step 1 into the Sequential Update Step 2:

   $$P(z_k \mid y_1, y_2) = \frac{P(y_2 \mid z_k)\frac{P(y_1 \mid z_k)P(z_k)}{\sum_{j=1}^K P(y_1 \mid z_j)P(z_j)}}{\sum_{m=1}^K P(y_2 \mid z_m)\frac{P(y_1 \mid z_m)P(z_m)}{\sum_{j=1}^K P(y_1 \mid z_j)P(z_j)}}$$

To simplify, call $C_1 = \sum_{j=1}^K P(y_1 \mid z_j)P(z_j)$ to make the equation clearer:
   
$$P(z_k \mid y_1, y_2) = \frac{P(y_2 \mid z_k)\frac{P(y_1 \mid z_k)P(z_k)}{C_1}}{\sum_{m=1}^K P(y_2 \mid z_m)\frac{P(y_1 \mid z_m)P(z_m)}{C_1}}$$
   
The $C_1$ terms cancel out in numerator and denominator, and yield the batch update formula:
   
$$P(z_k \mid y_1, y_2) = 
\frac{P(y_2 \mid z_k)P(y_1 \mid z_k)P(z_k)}{\sum_{m=1}^K P(y_2 \mid z_m)P(y_1 \mid z_m)P(z_m)} = 
\frac{P(y_1, y_2 \mid z_k)P(z_k)}{\sum_{j=1}^K P(y_1, y_2 \mid z_j)P(z_j)} $$

# 3. Algorithmic Steps

Below is a typical algorithmic outline for doing a single Bayesian update with discrete variables:

1. **Define the prior distribution**  
   You have an array or list of probabilities, $\text{prior}[k]$, summing to 1, where each index $k$ corresponds to a possible discrete state $z_k$.

2. **Define or compute the likelihood for each state**  
   You have (or compute) $\text{likelihood}[k] = P(y \mid z_k)$. This depends on your model of how likely observation $y$ is if the true state is $z_k$.

3. **Multiply prior by likelihood**  
   Create an unnormalized posterior:
   
   $$\text{unnorm\_posterior}[k] = \text{prior}[k] \times \text{likelihood}[k].$$

4. **Normalize**  
   Sum all elements of the unnormalized posterior to get the normalization constant:
   
   $$C = \sum_k \text{unnorm\_posterior}[k].$$
   
   Then divide each element in the unnormalized posterior by $C$:
   
   $$\text{posterior}[k] = \frac{\text{unnorm\_posterior}[k]}{C}.$$

5. **Use the posterior as the new prior** (if you have multiple observations sequentially)  
   If another observation arrives later, repeat the above steps, but start with the newly computed posterior as your next prior.

# Derivation

Suppose we have observed a measurement $ y $ for $ t - 1 $ timesteps. In the previous time step we had posterior:

$$
p(x \mid y_{1:t-1}) = \frac{p(y_{1:t-1} \mid x)p(x)}{\int_x p(y_{1:t-1} \mid x)p(x)dx}
$$

We want to compute: 

$$
p(x \mid y_{1:t}) = \frac{p(y_{1:t} \mid x)p(x)}{\int_x p(y_{1:t} \mid x)p(x)dx}
$$

By the Chain Rule:

$$
p(x \mid y_{1:t}) =

\frac{p(y_{1:t} \mid x)p(x)}{\int_x p(x , y_{1:t})dx} =

\frac{p(y_t \mid x, y_{1:t-1})p(y_{1:t-1} \mid x)p(x)}{\int_x p(y_{1:t} \mid x)p(x)dx}
$$

We assume that conditioned upon $ x $, all $ y_t $ are conditionally independent. Thus: 

$$ p(y_t \mid x, y_{1:t-1}) = p(y_t \mid x) $$

Simplifying:

$$
p(x \mid y_{1:t}) = \frac{p(y_t \mid x)p(y_{1:t-1} \mid x)p(x)}{\int_x p(y_t \mid x)p(y_{1:t-1} \mid x)p(x)dx}
$$

At this point, we may divide the numerator and denominator by any quantity of our choice, without changing the value of the expression. Suppose we choose (conveniently, as we'll see) to divide both by:

$$
\int_x p(y_{1:t-1} \mid x)p(x)dx
$$

Then:

$$
p(x | y_{1:t}) = 

\frac{
      p(y_t | x)p(y_{1:t-1} | x)p(x)
   }{
      \int_x p(y_t | x)p(y_{1:t-1} | x)p(x)dx
   } = 

\frac{
      p(y_t | x)\frac{p(y_{1:t-1}|x)p(x)}{\int_x p(y_{1:t-1}|x)p(x)dx}
   }{
      \frac{\int_x p(y_t|x)p(y_{1:t-1}|x)p(x)dx}{\int_x p(y_{1:t-1} \mid x)p(x)dx}
   }
$$

Now we will do a bit of simplification, which will allow us to rewrite this expression in terms of the posterior from the previous time step, i.e. $p(x | y_{1:t-1})$:

$$p(x | y_{1:t}) = \frac{p(y_t | x)p(x | y_{1:t-1})}{\int_x p(y_t|x)p(y_{1:t-1}|x)p(x)dx \over \int_x p(y_{1:t-1}|x)p(x)dx}$$

We can now combine these integrals into a single integral (since constant w.r.t. integral):
$$p(x | y_{1:t}) = \frac{p(y_t | x)p(x | y_{1:t-1})}{\int_x \frac{p(y_t|x)p(y_{1:t-1}|x)p(x)}{\int_x p(y_{1:t-1}|x)p(x)dx} dx} = \frac{p(y_t | x)p(x | y_{1:t-1})}{\int_x p(y_t | x)\frac{p(y_{1:t-1}|x)p(x)}{\int_x p(y_{1:t-1}|x)p(x)dx} dx}$$

When grouped differently, as shown on the far right, we see the previous posterior lurking in its unsimplified form in the denominator. We can use its simplified form to show:

$$p(x | y_{1:t}) = \frac{p(y_t | x)p(x | y_{1:t-1})}{\int_x p(y_t | x)p(x | y_{1:t-1})dx} = f\left(y_t, p(x | y_{1:t-1})\right)$$

In the recursive Bayes Estimator, the prior is just the posterior from the previous time step. Thus, we have the following loop: Measure->Estimate->Measure->Estimate.