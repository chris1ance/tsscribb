# Principal Components Analysis (PCA)

* Dataset: $ \mathbf{X} \in \mathbb{R}^{n \times p} $

* When faced with a large set of correlated variables, principal components summarize this set with a smaller number of representative variables that collectively explain most of the variability in the original set.

* Each of the dimensions found by PCA is a linear combination of $p$ features. 

* The first principal component direction of the data is that along which the observations vary the most. There is also another interpretation: the first principal component vector defines the line in $p$-dimensional space that is as close as possible to the data (using average squared Euclidean distance as a measure of closeness).
    * The notion of principal components as the dimensions that are closest to the n observations extends beyond just the first principal component. For instance, the first two principal components of a data set span the plane that is closest to the $n$ observations, in terms of average squared Euclidean distance. The frst three principal components of a data set span the three-dimensional hyperplane that is closest to the $n$ observations, and so forth.

* PCA produces __up to__ $p$ principal components $(\mathbf{z}_1,...,\mathbf{z}_p)$, where $\mathbf{z}_j \in \mathbb{R}^n$ for each $j=1,...,p$, such that:
    * Each $\mathbf{z}_j$ is a linear combination of the original predictors
    * Each $\mathbf{z}_j$ is orthonormal (pairwise orthogonal with unit length)
    * The $\mathbf{z}_j$'s are ordered based on the amount of variance explained.

* While PCA mathematically produces $p$ components, we deliberately choose a subset $M < p$ for analysis, modeling, and to achieve dimensionality reduction.

* Interpretation:
    * Principal component loading vectors are the directions in feature space along which the data vary the most, and the principal component scores are projections along these directions
    * Principal components provide low-dimensional linear surfaces that are closest to the observations. PCA projects data onto the hyperplane (i.e., a  linear subspace) that minimizes the distance between the original data and the projected data

* PCA Pros:
1. Can help avoid the curse of dimensionality and overfitting
2. Easy to visualize how predictive your features are of the response variable
3. Reduces multicollinearity, and so can improve the computational time when fitting models.

* PCA Cons:
1. __PCA knows nothing about the response variable__. Component vectors as predictors might not be ordered from best to worst
2. Direct interpretation of coefficients in PCR is lost
3. PCA can often fail to improve the predictive power of a model

## PCA Computation

Given an $ n \times p $ data set $ \mathbf{X} $, how do we compute the first principal component? Since we are only interested in variance, we assume that each of the variables in $ \mathbf{X} $ has been centered to have mean zero (that is, the column means of $ \mathbf{X} $ are zero). Denote this centered matrix as $\tilde{\mathbf{X}}$ We then look for the linear combination of the sample feature values of the form:

$$ z_{i1} = \phi_{11}\tilde{x}_{i1} + \phi_{21}\tilde{x}_{i2} + \ldots + \phi_{p1}\tilde{x}_{ip} \quad (1)$$

Or, more compactly:

$$ \mathbf{z}_1 = \tilde{\mathbf{X}} \mathbf{\phi}_1 \quad (1.1)$$

where $\mathbf{z}_1 \in \mathbb{R}^n$, that has the largest sample variance, subject to the constraint that $ \sum_{j=1}^{p} \phi_{j1}^2 = 1 $. In other words, the first principal component loading vector $\mathbf{\phi}_1$ solves the optimization problem:

$$ \underset{\mathbf{\phi}_1}{argmax} \left\{ \frac{1}{n} \sum_{i=1}^{n} \left( \sum_{j=1}^{p} \phi_{j1}x_{ij} \right)^2 \right\} \text{ subject to } \sum_{j=1}^{p} \phi_{j1}^2 = 1. \quad (2) $$

Using (1), we can rewrite (2) as:

$$ \underset{\mathbf{\phi}_1}{argmax} \left\{ \frac{1}{n} \sum_{i=1}^{n} z_{i1} ^2 \right\} \text{ subject to } \sum_{j=1}^{p} \phi_{j1}^2 = 1. \quad (3) $$

Since $\frac{1}{n} \sum_{i=1}^n x_{ij} = 0$, the average of the elements of $\mathbf{z}_1$ will be zero as well, hence the objective we are maximizing is just the sample variance of $\mathbf{z}_1$. 

We refer to $\mathbf{z}_1$ as the __scores__ of the first principal component and $\mathbf{\phi}_1$ is the first principal component __loading vector__.

After the first principal component $\mathbf{z}_1$ of the features has been determined, the second principal component $\mathbf{z}_2$ is defined as the linear combination $\tilde{\mathbf{X}} \mathbf{\phi}_2$ that has maximal variance out of all linear combinations that are uncorrelated with $\mathbf{z}_1$. It turns out that constraining $\mathbf{z}_1$ and $\mathbf{z}_2$ to be uncorrelated is equivalent to constraining the direction of $\mathbf{\phi}_2$ to be orthogonal to the direction of $\mathbf{\phi}_1$. To find $\mathbf{\phi}_2$, we solve a problem similar to (3) with $\mathbf{\phi}_2$ replacing $\mathbf{\phi}_1$ and with the additional constraing that $\mathbf{\phi}_2$ is orthogonal to $\mathbf{\phi}_1$. 

Once we have computed the principal components, we can plot them against each other in order to produce low-dimensional views of the data. For instance, we can plot the score vector $\mathbf{z}_1$ against $\mathbf{z}_2$, $\mathbf{z}_1$ against $\mathbf{z}_3$, $\mathbf{z}_2$ against $\mathbf{z}_3$, and so forth.

## Scaling the Variables 

We have already mentioned that before PCA is performed, the variables should be centered to have mean zero. Furthermore, the results obtained when we perform PCA will also depend on whether the variables have been individually scaled (each multiplied by a diferent constant). 

When performing PCA, it's often recommended to scale (standardize) the variables, especially when they are measured on different scales or have different units. Here are the reasons why:

* Equal Importance: Without scaling, variables with larger scales (e.g., salary measured in thousands) could dominate those with smaller scales (e.g., age). This dominance is not due to any inherent importance of these variables but purely because of their scale. Standardizing ensures that all variables have equal weight and importance.

* Interpretability: PCA identifies the directions of maximum variance. If variables aren't standardized, the principal components could be driven mainly by the variables with higher variances, which are influenced by their scales. Scaling ensures that the identified components represent the directions of maximum variance in a standardized space, making them more interpretable.

* Consistent Variance: The variance of standardized data is consistent (equal to 1 for each variable), which helps when calculating covariance or correlation matrices, commonly used in PCA.

* Numerical Stability: Standardizing variables can help in achieving better numerical stability. Certain numerical procedures might be sensitive to the scale of the data, and standardizing can help in reducing potential numerical errors.

However, if all variables are measured in the same units and are of roughly equal magnitude, scaling might not be necessary.

## PCA Computation with Matrices

__Step 1: Center the Data__

Given a dataset $\mathbf{X} \in \mathbb{R}^{n \times p}$ subtract the mean of each feature from the dataset to center the data around the origin.

$$ \tilde{\mathbf{X}} = \mathbf{X} - \mathbf{\mu} $$

Where $\mathbf{\mu} \in \mathbb{R}^{1 \times p}$ is a vector of means for each feature.

__Step 2: Compute the Covariance Matrix__

The covariance matrix $\mathbf{C} \in \mathbb{R}^{p \times p}$ of the centered data can be calculated as:

$$ \mathbf{C} = \frac{1}{n-1} \tilde{\mathbf{X}}^T \tilde{\mathbf{X}} $$

__Step 3: Compute the Eigenvalues and Eigenvectors of the Covariance Matrix__

Find the $p$ eigenvectors $\mathbf{v}_j \in \mathbb{R}^{p} $ and eigenvalues $\lambda_j \in \mathbb{R}$ of $\mathbf{C}$. The eigenvectors represent the directions of maximum variance in the data, and the eigenvalues represent the magnitude of the variance in those directions.

__Step 4: Sort Eigenvalues and Corresponding Eigenvectors__

Sort the eigenvalues in decreasing order. The eigenvector associated with the largest eigenvalue is the loadings of the first principal component. The eigenvector associated with the second largest eigenvalue is the loadings of the second principal component, and so on.

__Step 5: Select Principal Components__

Choose the first $M$ eigenvectors, where $M$ is the number of principal components you want to retain. This will form a $p \times M$ loadings matrix $\mathbf{W}$.

__Step 6: Project Data onto Lower-Dimensional Space__

The transformed data $\mathbf{Z} \in \mathbb{R}^{n \times M} $ (i.e. the principal component scores) can be obtained by projecting the centered data onto the selected principal components:

$$ \mathbf{Z} = \tilde{\mathbf{X}} \mathbf{W} $$

## PCA Computation with SVD Decomposition

__Step 1: Center the Data__

Given a dataset $\mathbf{X} \in \mathbb{R}^{n \times p}$ subtract the mean of each feature from the dataset to center the data around the origin.

$$ \tilde{\mathbf{X}} = \mathbf{X} - \mathbf{\mu} $$

Where $\mathbf{\mu} \in \mathbb{R}^{1 \times p}$ is a vector of means for each feature.

__Step 2: Perform SVD on Centered Data__

SVD decomposition:

$$ \tilde{\mathbf{X}} = U \Sigma V^T $$

Where:
- $U \in \mathbb{R}^{n \times n}$ is an orthogonal matrix whose columns are the left singular vectors
- $\Sigma \in \mathbb{R}^{n \times p}$ is a rectangular diagonal matrix containing the singular values $\sigma_1 \geq \sigma_2 \geq ... \geq \sigma_p \geq 0$
- $V \in \mathbb{R}^{p \times p}$ is an orthogonal matrix whose columns are the right singular vectors

__Step 3: Interpret SVD Components in Terms of PCA__

The right singular vectors $V$ from the SVD are equivalent to the eigenvectors of the covariance matrix $\mathbf{C}$. These are the principal component loading vectors (i.e., $V = \mathbf{W}$).

The singular values in $\Sigma$ are related to the eigenvalues $\lambda_j$ of the covariance matrix by:

$$ \lambda_j = \frac{\sigma_j^2}{n-1} $$

__Step 4: Extract the Principal Components__

Select the first $M$ columns of $V$ to form the loadings matrix $\mathbf{W} \in \mathbb{R}^{p \times M}$:

$$ \mathbf{W} = V_{:,1:M} $$

__Step 5: Compute the Principal Component Scores__

The principal component scores can be computed directly from the SVD:

$$ \mathbf{Z} = U_{:,1:M} \Sigma_{1:M,1:M} = \tilde{\mathbf{X}} V_{:,1:M} = \tilde{\mathbf{X}} \mathbf{W} $$

## Proportion of Variance Explain (PVE)

How much of the information in a given data set is lost by projecting the observations onto the frst few principal components? That is, how much of the variance in the data is not contained in the first few principal components? More generally, we are interested in knowing the proportion of variance explained (PVE) by each principal component. The total variance present in a data set (assuming that the variables have been centered to have mean zero) is defined as:

$$
\text{Var}(X_j) = \sum_{j=1}^p \frac{1}{n} \sum_{i=1}^{n} x_{ij}^2
$$

The variance explained by the $ m^{th} $ principal component is:

$$
\frac{1}{n} \sum_{i=1}^{n} z_{im}^2 = \frac{1}{n} \sum_{i=1}^{n} \left( \sum_{j=1}^{p} \phi_{jm}x_{ij} \right)^2
$$

Therefore, the PVE of the $ m^{th} $ principal component is given by:

$$
PVE_m = \frac{\sum_{i=1}^{n} z_{im}^2}{\sum_{j=1}^{p} \sum_{i=1}^{n} z_{ij}^2}
$$

To compute the cumulative PVE of the first $ M $ principal components, we simply sum the individual PVEs of the first $M$ PCs.

## Principal Components Regression (PCR)

* The principal components regression (PCR) approach involves constructing the first $M$ principal components, $(\mathbf{z}_1,...,\mathbf{z}_M)$ and then using these components as the predictors in a linear regression model that is fit using least square.

* The key idea is that often a small number of principal components sufce to explain most of the variability in the data, as well as the relationship with the response.  In other words, we assume that the directions in which $\mathbf{x}_{\boldsymbol{\cdot} 1},...,\mathbf{x}_{\boldsymbol{\cdot} p}$ show the most variation are the directions that are associated with $\mathbf{y}$. While this assumption is not guaranteed to be true, it often turns out to be a reasonable enough approximation to give good results. If the assumption underlying PCR holds, then ftting a least squares
model to $(\mathbf{z}_1,...,\mathbf{z}_M)$ will lead to better results than ftting a least squares model to $\mathbf{x}_{\boldsymbol{\cdot} 1},...,\mathbf{x}_{\boldsymbol{\cdot} p}$ since most or all of the information in the data that relates to the response is contained in $(\mathbf{z}_1,...,\mathbf{z}_M)$, and by estimating only $M << p$ coefcients we can mitigate overfitting

* As more principal components $M$ are used in the regression model, the bias decreases, but the variance increases. In PCR, the number of principal components, M, is typically chosen by cross-validation.

* When performing PCR, we generally recommend standardizing each predictor (dividing each predictor by its standard deviation), prior to generating the principal components. This standardization ensures that all variables are on the same scale. In the absence of standardization, the high-variance variables will tend to play a larger role in the principal components obtained, and the scale on which the variables are measured will ultimately have an efect on the fnal PCR model. However, if the variables are all measured in the same units (say, kilograms, or inches), then one might choose not to standardize them.

* You can always put PCT coefficients in terms of the original predictors:

$$ \mathbf{y} = \mathbf{Z} \beta_Z = (\tilde{\mathbf{X}} \mathbf{W}) \beta_Z = \tilde{\mathbf{X}} (\mathbf{W} \beta_Z) = \tilde{\mathbf{X}} \beta_X $$

where:
* $\tilde{\mathbf{X}}$ is the original dataset, centered so that each feature has mean zero
* $\mathbf{W}$ is the matrix whose columns are the first $M$ eigenvectors of $\mathbf{C} = \frac{1}{n-1} \tilde{\mathbf{X}}^T \tilde{\mathbf{X}}$
* $\mathbf{Z}$ is the transformed dataset created by projecting $\tilde{\mathbf{X}}$ onto the principal components contained in $\mathbf{W}$
* $\beta_Z$ are the coefficients for the PCR model
* $\beta_X$ are the coefficients in terms of the original predictors.

* When $M = p$, the recovered coefficients are identical to those from a model fit on the original predictors.