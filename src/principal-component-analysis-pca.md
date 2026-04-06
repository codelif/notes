# Principal Component Analysis (PCA)
PCA is a **dimensionality reduction** algorithm.
It creates new features called **principal components** that keep as much information as possible from the original dataset.
The main idea is:
- find directions where the data varies the most
- rank those directions
- keep only the most important ones

If the original data has $d$ features, PCA can produce up to $d$ principal components.

---
## Why PCA is Needed
High-dimensional data causes problems such as:
- harder visualization
- more computation
- more difficult learning
- curse of dimensionality

PCA solves this by projecting the data onto fewer dimensions while preserving maximum variance.

---
## Core Idea
Suppose the data matrix is:
$$
X\in \mathbb{R}^{n\times d}
$$
where:
- $n$ = number of samples
- $d$ = number of features

PCA finds a new set of orthogonal directions:
$$
u_1,u_2,\dots,u_d
$$
such that:
- $u_1$ captures the maximum variance
- $u_2$ captures the next maximum variance
- and so on

These directions are the **eigenvectors** of the covariance matrix.
Their importance is given by the **eigenvalues**.

---
## Main Equations
### 1. Standardization
Each feature is standardized using Z-score:
$$
x'=\frac{x-\mu}{\sigma}
$$

### 2. Covariance matrix
For centered data:
$$
C=\frac{1}{n}X^TX
$$

### 3. Eigen decomposition
$$
Cu=\lambda u
$$
where:
- $u$ = eigenvector
- $\lambda$ = eigenvalue

### 4. Projection
If $W_k$ contains the top $k$ principal components, then:
$$
X_{\text{proj}}=XW_k
$$

---
## Intuition
Each principal component is a direction in feature space.
If data spreads a lot along a direction, that direction contains a lot of information.
So PCA keeps the directions with the **largest variance**.

That is why PCA chooses eigenvectors with the **largest eigenvalues**.

---
## Short and Clean Code
```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA as SklearnPCA

class PCAFromScratch:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean_ = None
        self.std_ = None
        self.components_ = None
        self.eigenvalues_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)

        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        Xs = (X - self.mean_) / self.std_

        C = (Xs.T @ Xs) / len(Xs)
        eigenvalues, eigenvectors = np.linalg.eigh(C)

        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        self.eigenvalues_ = eigenvalues[:self.n_components]
        self.components_ = eigenvectors[:, :self.n_components]
        self.explained_variance_ratio_ = eigenvalues / eigenvalues.sum()
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        Xs = (X - self.mean_) / self.std_
        return Xs @ self.components_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

iris = load_iris()
X = iris.data

pca = PCAFromScratch(n_components=2)
X_proj = pca.fit_transform(X)

print("Top eigenvalues:", np.round(pca.eigenvalues_, 4))
print("Principal components:\n", np.round(pca.components_, 4))
print("Explained variance ratio:", np.round(pca.explained_variance_ratio_, 4))

sk_pca = SklearnPCA(n_components=2)
X_sk = sk_pca.fit_transform((X - X.mean(axis=0)) / X.std(axis=0))

print("Sklearn components:\n", np.round(sk_pca.components_.T, 4))
```

---
## What This Code Does
This code:
1. loads the Iris dataset
2. standardizes all features
3. computes the covariance matrix
4. finds eigenvalues and eigenvectors
5. sorts them from largest to smallest
6. selects the top `n_components`
7. projects the original data onto those components

So 4-dimensional Iris data becomes 2-dimensional.

---
## Dataset
The Iris dataset has:
- 150 samples
- 4 features
- 3 flower classes

The four features are:
- sepal length
- sepal width
- petal length
- petal width

PCA uses only the feature matrix:
$$
X\in \mathbb{R}^{150\times 4}
$$

Since PCA is unsupervised, labels are not needed.

---
## Step-by-Step Algorithm
## Step 1: Standardize the dataset
PCA is sensitive to scale.
If one feature has larger values, it can dominate the variance.

So each column is standardized:
$$
x'=\frac{x-\mu}{\sigma}
$$

Code:
```python
self.mean_ = X.mean(axis=0)
self.std_ = X.std(axis=0)
Xs = (X - self.mean_) / self.std_
```

Concept:
- subtract column mean
- divide by column standard deviation
- now every feature has roughly comparable scale

Why important:
- PCA is variance-based
- variance depends on feature scale
- standardization ensures fairness among features

---
## Step 2: Compute covariance matrix
The covariance matrix measures how features vary together.

Equation:
$$
C=\frac{1}{n}X_s^TX_s
$$

Code:
```python
C = (Xs.T @ Xs) / len(Xs)
```

Concept:
- diagonal entries = variance of each standardized feature
- off-diagonal entries = covariance between pairs of features

For Iris:
$$
C\in \mathbb{R}^{4\times 4}
$$

Because there are 4 original features.

---
## Step 3: Find eigenvalues and eigenvectors
PCA solves:
$$
Cu=\lambda u
$$

Code:
```python
eigenvalues, eigenvectors = np.linalg.eigh(C)
```

Concept:
- each eigenvector gives a direction
- each eigenvalue tells how much variance is captured in that direction

Why `eigh` and not `eig`:
- covariance matrix is symmetric
- `np.linalg.eigh` is better for symmetric matrices

Interpretation:
- large eigenvalue $\Rightarrow$ important component
- small eigenvalue $\Rightarrow$ less important component

---
## Step 4: Sort eigenvalues and eigenvectors
The most useful components are the ones with the largest eigenvalues.

Code:
```python
order = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[order]
eigenvectors = eigenvectors[:, order]
```

Concept:
- `argsort` sorts indices
- `[::-1]` reverses order to descending
- now the first column of `eigenvectors` is the first principal component

---
## Step 5: Select top principal components
If we want only $k$ components:
$$
W_k=[u_1\ u_2\ \dots\ u_k]
$$

Code:
```python
self.eigenvalues_ = eigenvalues[:self.n_components]
self.components_ = eigenvectors[:, :self.n_components]
```

Concept:
- keep only the dominant directions
- discard weaker directions
- dimension reduces from $d$ to $k$

For example:
- original data: 4 features
- choose 2 components
- reduced data: 2 features

---
## Step 6: Project data onto the new space
Projection formula:
$$
X_{\text{proj}}=X_sW_k
$$

Code:
```python
return Xs @ self.components_
```

Concept:
- each sample is re-expressed in terms of principal components
- this gives lower-dimensional data
- information loss is minimized as much as possible for the chosen number of components

If:
$$
X_s\in\mathbb{R}^{150\times 4},\quad W_2\in\mathbb{R}^{4\times 2}
$$
then:
$$
X_{\text{proj}}\in\mathbb{R}^{150\times 2}
$$

---
## Concept -> Equation -> Code Mapping
## 1. Equalize feature scales
Concept:
All features should contribute fairly.

Equation:
$$
x'=\frac{x-\mu}{\sigma}
$$

Code:
```python
self.mean_ = X.mean(axis=0)
self.std_ = X.std(axis=0)
Xs = (X - self.mean_) / self.std_
```

---
## 2. Measure variance structure
Concept:
We need a matrix that summarizes how features vary together.

Equation:
$$
C=\frac{1}{n}X_s^TX_s
$$

Code:
```python
C = (Xs.T @ Xs) / len(Xs)
```

---
## 3. Find important directions
Concept:
The best projection directions are the eigenvectors of the covariance matrix.

Equation:
$$
Cu=\lambda u
$$

Code:
```python
eigenvalues, eigenvectors = np.linalg.eigh(C)
```

---
## 4. Rank the directions
Concept:
Directions with larger variance are more useful.

Equation:
$$
\lambda_1\ge \lambda_2\ge \cdots \ge \lambda_d
$$

Code:
```python
order = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[order]
eigenvectors = eigenvectors[:, order]
```

---
## 5. Keep only the top directions
Concept:
Dimensionality reduction means retaining only the most informative directions.

Equation:
$$
W_k=[u_1\ u_2\ \dots\ u_k]
$$

Code:
```python
self.components_ = eigenvectors[:, :self.n_components]
```

---
## 6. Project data
Concept:
Convert old features into principal component coordinates.

Equation:
$$
X_{\text{proj}}=X_sW_k
$$

Code:
```python
return Xs @ self.components_
```

---
## Why Eigenvectors and Eigenvalues Appear
PCA wants to maximize the variance of projected data.
If we project onto a unit vector $u$, projected variance is:
$$
\text{Var}(Xu)=u^TCu
$$
subject to:
$$
u^Tu=1
$$

This is an optimization problem.
Using Lagrange multipliers:
$$
\mathcal{L}(u,\lambda)=u^TCu-\lambda(u^Tu-1)
$$
Taking derivative and setting to zero gives:
$$
Cu=\lambda u
$$

So:
- the best directions are eigenvectors of $C$
- the amount of retained variance is given by eigenvalues

This is the mathematical reason behind PCA.

---
## Why the Largest Eigenvalue Matters
The projected variance along direction $u$ is:
$$
u^TCu
$$
For an eigenvector:
$$
Cu=\lambda u
$$
So:
$$
u^TCu=u^T(\lambda u)=\lambda u^Tu
$$
Since:
$$
u^Tu=1
$$
we get:
$$
u^TCu=\lambda
$$

Therefore:
- projected variance equals the eigenvalue
- maximizing variance means choosing the **largest eigenvalue**

That is why PCA selects top eigenvalues first.

---
## Explained Variance Ratio
A useful quantity is:
$$
\text{Explained Variance Ratio}_i=\frac{\lambda_i}{\sum_j \lambda_j}
$$

Code:
```python
self.explained_variance_ratio_ = eigenvalues / eigenvalues.sum()
```

Concept:
This tells how much total information each principal component retains.

Example interpretation:
- PC1 = 72%
- PC2 = 23%
- then first two PCs keep 95% of the total variance

This helps decide how many components to keep.

---
## Worked Mini Example
Suppose after covariance computation, eigenvalues are:
$$
[2.91,\ 0.92,\ 0.15,\ 0.02]
$$

Then:
- first principal component captures the most variance
- second captures the next most
- third and fourth contribute little

Total variance:
$$
2.91+0.92+0.15+0.02=4.00
$$

Explained variance ratios:
$$
\frac{2.91}{4}=0.7275
$$
$$
\frac{0.92}{4}=0.23
$$
$$
\frac{0.15}{4}=0.0375
$$
$$
\frac{0.02}{4}=0.005
$$

So:
- PC1 keeps about 72.75%
- PC2 keeps about 23%
- first two together keep about 95.75%

This means reducing from 4D to 2D is very reasonable.

---
## Understanding the Components Matrix
If the selected components are:
$$
W_2=
\begin{bmatrix}
0.52 & -0.27\\
-0.26 & -0.92\\
0.58 & -0.03\\
0.56 & -0.27
\end{bmatrix}
$$
then:
- first column = first principal component
- second column = second principal component

Each column shows how the original features combine to form the new axis.

For example, first component:
$$
PC_1=0.52x_1-0.26x_2+0.58x_3+0.56x_4
$$

So a principal component is a **linear combination** of original features.

---
## Comparing with Scikit-Learn
Code:
```python
sk_pca = SklearnPCA(n_components=2)
X_sk = sk_pca.fit_transform((X - X.mean(axis=0)) / X.std(axis=0))
print(np.round(sk_pca.components_.T, 4))
```

Concept:
This checks whether the scratch implementation gives similar principal components.

Important note:
Principal components may differ by sign.
That means if one library gives:
$$
u
$$
another may give:
$$
-u
$$
This is still correct because both represent the same direction.

So when comparing PCA outputs, sign flips are normal.

---
## Why PCA Works
PCA works because:
1. it identifies directions of maximum spread
2. those directions preserve the most information
3. the directions are orthogonal, so they do not duplicate information
4. low-variance directions can often be removed with minimal information loss

So PCA compresses data while keeping the most useful structure.

---
## Limitations
## 1. PCA is linear
PCA only finds **linear** combinations of features.
If structure is highly non-linear, PCA may not capture it well.

## 2. PCA is sensitive to scale
Without standardization, large-scale features dominate.

## 3. Components may be hard to interpret
The new axes are combinations of original features, so they may not be as interpretable as raw columns.

## 4. Variance does not always mean usefulness
PCA keeps directions with high variance, but high variance does not always mean high predictive importance for a target variable.

---
## Exam-Oriented Summary
## Definition
PCA is an unsupervised dimensionality reduction technique that transforms correlated features into orthogonal principal components.

## Goal
Reduce the number of features while retaining maximum variance.

## Steps
1. standardize data
2. compute covariance matrix
3. find eigenvalues and eigenvectors
4. sort them in descending order
5. select top $k$ components
6. project data onto them

## Important Equations
Standardization:
$$
x'=\frac{x-\mu}{\sigma}
$$
Covariance:
$$
C=\frac{1}{n}X_s^TX_s
$$
Eigen equation:
$$
Cu=\lambda u
$$
Projection:
$$
X_{\text{proj}}=X_sW_k
$$
Explained variance ratio:
$$
\frac{\lambda_i}{\sum_j \lambda_j}
$$

## Interpretation
- eigenvectors = directions of principal components
- eigenvalues = variance captured by those directions
- larger eigenvalue = more important component

---
## Very Short Revision
PCA reduces dimensions by:
- standardizing data
- computing covariance matrix
- finding eigenvectors/eigenvalues
- sorting by largest eigenvalue
- keeping top components
- projecting data onto them

Main idea:
$$
\text{keep directions with maximum variance}
$$

---
## Final Takeaway
PCA transforms high-dimensional data into a lower-dimensional form by finding the most informative orthogonal directions.
These directions are the eigenvectors of the covariance matrix, and their importance is measured by eigenvalues.
The larger the eigenvalue, the more variance that principal component preserves, so the better it is for dimensionality reduction.
