# Linear Regression
Simple Linear Regression predicts a continuous value using **one input feature** by fitting a straight line:
$$
\hat y=b_0+b_1x
$$
where:
- $b_0$ = intercept
- $b_1$ = slope
- $\hat y$ = predicted output

The goal is to find the line that minimizes the **sum of squared errors**:
$$
J(b_0,b_1)=\sum_{i=1}^{n}(y_i-\hat y_i)^2
$$

---
## Intuition
We want the "best" straight line through the data.
- If $b_1>0$, the line goes upward
- If $b_1<0$, the line goes downward
- $b_0$ tells where the line cuts the $y$-axis

For a dataset with one feature:
$$
X=\begin{bmatrix}x_1\\x_2\\ \vdots \\x_n\end{bmatrix},\quad
y=\begin{bmatrix}y_1\\y_2\\ \vdots \\y_n\end{bmatrix}
$$
we add a bias column of 1s:
$$
X_b=
\begin{bmatrix}
1 & x_1\\
1 & x_2\\
\vdots & \vdots\\
1 & x_n
\end{bmatrix}
$$

Then the parameters are computed using the **Normal Equation**:
$$
\theta=(X_b^TX_b)^{-1}X_b^Ty
$$
where:
$$
\theta=
\begin{bmatrix}
b_0\\
b_1
\end{bmatrix}
$$

---
## Short and Clean Code
```python
import numpy as np
import matplotlib.pyplot as plt

class SimpleLinearRegression:
    def __init__(self):
        self.intercept_ = 0.0
        self.coef_ = 0.0
        self.r2_ = 0.0

    def fit(self, X, y):
        X = np.asarray(X).reshape(-1, 1)
        y = np.asarray(y).reshape(-1, 1)

        Xb = np.c_[np.ones((len(X), 1)), X]
        theta = np.linalg.inv(Xb.T @ Xb) @ Xb.T @ y

        self.intercept_ = theta[0, 0]
        self.coef_ = theta[1, 0]

        y_pred = Xb @ theta
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        self.r2_ = 1 - ss_res / ss_tot
        return self

    def predict(self, X):
        X = np.asarray(X).reshape(-1, 1)
        return self.intercept_ + self.coef_ * X

X = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([2,4,5,4,5,7,8,9,10,12])

model = SimpleLinearRegression().fit(X, y)
y_pred = model.predict(X)

print("Intercept:", round(model.intercept_, 4))
print("Slope:", round(model.coef_, 4))
print("R^2:", round(model.r2_, 4))

plt.scatter(X, y, label="Data")
plt.plot(X, y_pred, label="Regression line")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Simple Linear Regression")
plt.legend()
plt.show()
```

---
## Output Meaning
From this code, the fitted line is approximately:
$$
\hat y=1.0667+1.0182x
$$

So:
- intercept $\approx 1.0667$
- slope $\approx 1.0182$

This means:
- when $x=0$, predicted $y\approx 1.0667$
- for every increase of 1 in $x$, $y$ increases by about $1.0182$

The $R^2$ score is approximately:
$$
R^2\approx 0.9525
$$
which means the model explains about **95.25\%** of the variance in the target.

---
## Step-by-Step Algorithm
## Step 1: Prepare the data
We start with:
```python
X = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([2,4,5,4,5,7,8,9,10,12])
```

Concept:
- $X$ is the input feature
- $y$ is the actual output

Dataset pairs:
$$
(1,2),(2,4),(3,5),(4,4),(5,5),(6,7),(7,8),(8,9),(9,10),(10,12)
$$

---
## Step 2: Add the bias column
To learn both intercept and slope together, we transform $X$ into:
$$
X_b=
\begin{bmatrix}
1 & 1\\
1 & 2\\
1 & 3\\
1 & 4\\
1 & 5\\
1 & 6\\
1 & 7\\
1 & 8\\
1 & 9\\
1 & 10
\end{bmatrix}
$$

Code:
```python
Xb = np.c_[np.ones((len(X), 1)), X.reshape(-1, 1)]
```

Concept:
- first column of 1s handles the intercept
- second column stores the feature values

---
## Step 3: Compute parameters using the Normal Equation
Equation:
$$
\theta=(X_b^TX_b)^{-1}X_b^Ty
$$

Code:
```python
theta = np.linalg.inv(Xb.T @ Xb) @ Xb.T @ y.reshape(-1, 1)
```

This gives:
$$
\theta=
\begin{bmatrix}
1.0667\\
1.0182
\end{bmatrix}
$$

So:
$$
b_0=1.0667,\quad b_1=1.0182
$$

Meaning:
$$
\hat y=1.0667+1.0182x
$$

---
## Step 4: Predict values
For each input, substitute into:
$$
\hat y=b_0+b_1x
$$

Code:
```python
y_pred = self.intercept_ + self.coef_ * X.reshape(-1, 1)
```

Let us compute a few predictions manually.

For $x=1$:
$$
\hat y=1.0667+1.0182(1)=2.0849
$$

For $x=2$:
$$
\hat y=1.0667+1.0182(2)=3.1030
$$

For $x=5$:
$$
\hat y=1.0667+1.0182(5)=6.1576
$$

For $x=10$:
$$
\hat y=1.0667+1.0182(10)=11.2485
$$

So the model predictions are approximately:
$$
[2.0849,3.1030,4.1212,5.1394,6.1576,7.1758,8.1939,9.2121,10.2303,11.2485]
$$

---
## Step 5: Measure performance using $R^2$
The coefficient of determination is:
$$
R^2=1-\frac{\sum (y-\hat y)^2}{\sum (y-\bar y)^2}
$$

Where:
- $\sum (y-\hat y)^2$ = residual sum of squares
- $\sum (y-\bar y)^2$ = total sum of squares
- $\bar y$ = mean of actual $y$

Code:
```python
ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - y.mean()) ** 2)
self.r2_ = 1 - ss_res / ss_tot
```

For this dataset:
$$
R^2\approx 0.9525
$$

Interpretation:
- close to 1 means strong fit
- close to 0 means poor fit

---
## Code Explanation: Concept -> Equation -> Code
## 1. Store model parameters
Concept:
We need to remember the learned intercept, slope, and model score.

Code:
```python
class SimpleLinearRegression:
    def __init__(self):
        self.intercept_ = 0.0
        self.coef_ = 0.0
        self.r2_ = 0.0
```

Meaning:
- `intercept_` stores $b_0$
- `coef_` stores $b_1$
- `r2_` stores $R^2$

---
## 2. Convert input into column form
Concept:
Matrix equations require $X$ and $y$ in proper shapes.

Code:
```python
X = np.asarray(X).reshape(-1, 1)
y = np.asarray(y).reshape(-1, 1)
```

Meaning:
- `reshape(-1, 1)` makes data a column vector

Example:
$$
X=
\begin{bmatrix}
1\\2\\3\\ \vdots \\10
\end{bmatrix}
$$

---
## 3. Add the bias term
Concept:
The intercept must be part of the matrix multiplication.

Equation:
$$
\hat y=X_b\theta
$$

Code:
```python
Xb = np.c_[np.ones((len(X), 1)), X]
```

This builds:
$$
X_b=
\begin{bmatrix}
1 & x_1\\
1 & x_2\\
\vdots & \vdots\\
1 & x_n
\end{bmatrix}
$$

---
## 4. Learn the best-fit line
Concept:
Choose parameters that minimize squared error.

Equation:
$$
\theta=(X_b^TX_b)^{-1}X_b^Ty
$$

Code:
```python
theta = np.linalg.inv(Xb.T @ Xb) @ Xb.T @ y
```

This is the heart of the algorithm.

Then:
```python
self.intercept_ = theta[0, 0]
self.coef_ = theta[1, 0]
```

This maps:
$$
\theta=
\begin{bmatrix}
b_0\\
b_1
\end{bmatrix}
$$

---
## 5. Predict outputs
Concept:
Once parameters are known, plug them into the line equation.

Equation:
$$
\hat y=b_0+b_1x
$$

Code:
```python
y_pred = Xb @ theta
```
or in `predict()`:
```python
return self.intercept_ + self.coef_ * X
```

Both do the same thing.

---
## 6. Evaluate model fit
Concept:
Compare predictions with actual values.

Equation:
$$
SS_{res}=\sum (y-\hat y)^2
$$
$$
SS_{tot}=\sum (y-\bar y)^2
$$
$$
R^2=1-\frac{SS_{res}}{SS_{tot}}
$$

Code:
```python
ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - y.mean()) ** 2)
self.r2_ = 1 - ss_res / ss_tot
```

---
## Worked Example
Using:
$$
x=6
$$
Prediction:
$$
\hat y=1.0667+1.0182(6)=7.1758
$$

Actual value in dataset:
$$
y=7
$$

Error:
$$
y-\hat y=7-7.1758=-0.1758
$$

Squared error:
$$
(-0.1758)^2\approx 0.0309
$$

This is how the model measures how far prediction is from truth.

---
## Why This Algorithm Works
Simple Linear Regression assumes:
1. the relationship is approximately linear
2. one feature is enough to explain the target
3. the best line is the one with minimum squared error

By minimizing squared error, the algorithm finds a line that stays as close as possible to all points overall.

---
## Final Summary
Simple Linear Regression learns:
$$
\hat y=b_0+b_1x
$$
using:
$$
\theta=(X_b^TX_b)^{-1}X_b^Ty
$$

For this dataset, the learned model is:
$$
\hat y=1.0667+1.0182x
$$
with:
$$
R^2\approx 0.9525
$$

So the model fits the data well and captures a strong positive linear relationship.

---
## Exam-Oriented Points
- Used for predicting a **continuous** value
- Works with **one input feature**
- Equation of model:
$$
\hat y=b_0+b_1x
$$
- Parameters are found using the **Normal Equation**
- Performance is commonly measured with **$R^2$**
- Best when the relationship between feature and target is approximately linear

---
## Very Short Revision
- Add bias column
- Compute parameters using Normal Equation
- Form regression line
- Predict output
- Evaluate using $R^2$

Formula:
$$
\theta=(X_b^TX_b)^{-1}X_b^Ty
$$
Model:
$$
\hat y=b_0+b_1x
$$
