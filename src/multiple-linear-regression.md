# Multiple Linear Regression
Multiple Linear Regression predicts a continuous target using **two or more input features**.
It assumes a linear relationship between inputs and output:
$$
\hat y=b_0+b_1x_1+b_2x_2+\cdots+b_nx_n
$$
where:
- $b_0$ = intercept
- $b_1,b_2,\dots,b_n$ = coefficients
- $\hat y$ = predicted value

For this example with two features:
$$
\hat y=b_0+b_1x_1+b_2x_2
$$

The model learns the best coefficients by minimizing the **sum of squared errors**:
$$
J=\sum_{i=1}^{m}(y_i-\hat y_i)^2
$$

---
## Intuition
Simple Linear Regression fits a **line**.
Multiple Linear Regression with two features fits a **plane**.
With more than two features, it fits a **hyperplane**.

So here:
- $x_1$ affects $y$
- $x_2$ affects $y$
- the model combines both effects into one equation

---
## Main Equation
If the dataset has two input features:
$$
X=
\begin{bmatrix}
x_{11} & x_{12}\\
x_{21} & x_{22}\\
\vdots & \vdots\\
x_{m1} & x_{m2}
\end{bmatrix}
$$
then after adding the bias column:
$$
X_b=
\begin{bmatrix}
1 & x_{11} & x_{12}\\
1 & x_{21} & x_{22}\\
\vdots & \vdots & \vdots\\
1 & x_{m1} & x_{m2}
\end{bmatrix}
$$

The Normal Equation is:
$$
\theta=(X_b^TX_b)^{-1}X_b^Ty
$$
where:
$$
\theta=
\begin{bmatrix}
b_0\\
b_1\\
b_2
\end{bmatrix}
$$

---
## Short and Clean Code
```python
import numpy as np
import matplotlib.pyplot as plt

class MultipleLinearRegression:
    def __init__(self):
        self.intercept_ = 0.0
        self.coef_ = None
        self.r2_ = 0.0

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)

        Xb = np.c_[np.ones((len(X), 1)), X]
        theta = np.linalg.inv(Xb.T @ Xb) @ Xb.T @ y

        self.intercept_ = theta[0, 0]
        self.coef_ = theta[1:, 0]

        y_pred = Xb @ theta
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        self.r2_ = 1 - ss_res / ss_tot
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return self.intercept_ + X @ self.coef_

np.random.seed(0)
X1 = np.random.randint(1, 11, 15)
X2 = np.random.randint(1, 11, 15)
X = np.column_stack((X1, X2))
y = 1 + 2 * X1 + 3 * X2 + np.random.randn(15) * 2

model = MultipleLinearRegression().fit(X, y)
y_pred = model.predict(X)

print("Intercept:", round(model.intercept_, 4))
print("Coefficients:", np.round(model.coef_, 4))
print("R^2:", round(model.r2_, 4))

fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(X[:, 0], X[:, 1], y, label="Data")

x1_grid, x2_grid = np.meshgrid(
    np.linspace(X[:, 0].min(), X[:, 0].max(), 20),
    np.linspace(X[:, 1].min(), X[:, 1].max(), 20)
)
grid_points = np.c_[x1_grid.ravel(), x2_grid.ravel()]
z_grid = model.predict(grid_points).reshape(x1_grid.shape)

ax.plot_surface(x1_grid, x2_grid, z_grid, alpha=0.5)
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("y")
ax.set_title("Multiple Linear Regression Plane")
plt.show()
```

---
## Dataset Used
The target was generated using:
$$
y=1+2x_1+3x_2+\text{noise}
$$

Code:
```python
y = 1 + 2 * X1 + 3 * X2 + np.random.randn(15) * 2
```

Concept:
- true intercept is near 1
- true coefficient of $x_1$ is near 2
- true coefficient of $x_2$ is near 3
- random noise is added to make it realistic

So the model should learn values close to:
$$
b_0\approx 1,\quad b_1\approx 2,\quad b_2\approx 3
$$

---
## Step-by-Step Algorithm
## Step 1: Prepare the data
We create two input features:
```python
X1 = np.random.randint(1, 11, 15)
X2 = np.random.randint(1, 11, 15)
X = np.column_stack((X1, X2))
```

Concept:
Each sample now has two features:
$$
x=(x_1,x_2)
$$

So a row may look like:
$$
(6,8)
$$
meaning:
- first feature = 6
- second feature = 8

---
## Step 2: Add the bias column
To learn the intercept together with slopes, add a column of ones:
$$
X_b=
\begin{bmatrix}
1 & x_{11} & x_{12}\\
1 & x_{21} & x_{22}\\
\vdots & \vdots & \vdots
\end{bmatrix}
$$

Code:
```python
Xb = np.c_[np.ones((len(X), 1)), X]
```

Concept:
- first column handles the intercept
- remaining columns are the actual features

---
## Step 3: Compute coefficients using the Normal Equation
Equation:
$$
\theta=(X_b^TX_b)^{-1}X_b^Ty
$$

Code:
```python
theta = np.linalg.inv(Xb.T @ Xb) @ Xb.T @ y
```

Concept:
This directly computes the best-fit coefficients without iterative optimization.

The parameter vector is:
$$
\theta=
\begin{bmatrix}
b_0\\
b_1\\
b_2
\end{bmatrix}
$$

Code mapping:
```python
self.intercept_ = theta[0, 0]
self.coef_ = theta[1:, 0]
```

So:
- `intercept_` stores $b_0$
- `coef_[0]` stores $b_1$
- `coef_[1]` stores $b_2$

---
## Step 4: Form the regression equation
Once the parameters are learned, the model becomes:
$$
\hat y=b_0+b_1x_1+b_2x_2
$$

Code:
```python
return self.intercept_ + X @ self.coef_
```

Concept:
For each new row:
- multiply each feature by its coefficient
- add the intercept
- get the predicted output

---
## Step 5: Compute predictions
Predictions for training data:
$$
\hat y=X_b\theta
$$

Code:
```python
y_pred = Xb @ theta
```

Concept:
This gives the fitted values of the regression plane on the training samples.

---
## Step 6: Evaluate with $R^2$
The coefficient of determination is:
$$
R^2=1-\frac{\sum(y-\hat y)^2}{\sum(y-\bar y)^2}
$$

Code:
```python
ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - y.mean()) ** 2)
self.r2_ = 1 - ss_res / ss_tot
```

Meaning:
- $SS_{res}$ = residual sum of squares
- $SS_{tot}$ = total sum of squares

Interpretation:
- $R^2=1$ means perfect fit
- larger $R^2$ means better fit
- close to 0 means poor explanatory power

---
## Code Explanation: Concept -> Equation -> Code
## 1. Store model parameters
Concept:
The model needs to store intercept, coefficients, and score.

Code:
```python
def __init__(self):
    self.intercept_ = 0.0
    self.coef_ = None
    self.r2_ = 0.0
```

Meaning:
- `intercept_` = $b_0$
- `coef_` = slopes
- `r2_` = model accuracy measure

---
## 2. Convert input shapes properly
Concept:
Matrix operations need correctly shaped arrays.

Code:
```python
X = np.asarray(X, dtype=float)
y = np.asarray(y, dtype=float).reshape(-1, 1)
```

Meaning:
- `X` becomes a 2D matrix
- `y` becomes a column vector

---
## 3. Add intercept term
Concept:
We include the intercept in matrix multiplication by adding a bias column.

Equation:
$$
X_b=
\begin{bmatrix}
1 & x_1 & x_2
\end{bmatrix}
$$

Code:
```python
Xb = np.c_[np.ones((len(X), 1)), X]
```

---
## 4. Learn best coefficients
Concept:
Find coefficients that minimize squared error.

Equation:
$$
\theta=(X_b^TX_b)^{-1}X_b^Ty
$$

Code:
```python
theta = np.linalg.inv(Xb.T @ Xb) @ Xb.T @ y
```

---
## 5. Separate intercept and slopes
Concept:
The first value of $\theta$ is intercept, the rest are feature coefficients.

Code:
```python
self.intercept_ = theta[0, 0]
self.coef_ = theta[1:, 0]
```

If:
$$
\theta=
\begin{bmatrix}
1.2\\
1.9\\
3.1
\end{bmatrix}
$$
then the model is:
$$
\hat y=1.2+1.9x_1+3.1x_2
$$

---
## 6. Predict outputs
Concept:
Use the learned plane to estimate new values.

Equation:
$$
\hat y=b_0+b_1x_1+b_2x_2
$$

Code:
```python
return self.intercept_ + X @ self.coef_
```

---
## 7. Evaluate goodness of fit
Concept:
Check how much variance in $y$ is explained by the model.

Equation:
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
Suppose the learned model is approximately:
$$
\hat y=1.3+1.9x_1+3.0x_2
$$

Take one sample:
$$
x_1=4,\quad x_2=5
$$

Prediction:
$$
\hat y=1.3+1.9(4)+3.0(5)
$$
$$
\hat y=1.3+7.6+15
$$
$$
\hat y=23.9
$$

So for that point, the model predicts:
$$
23.9
$$

This is exactly what the `predict()` method is doing for every row.

---
## Why This Algorithm Works
Multiple Linear Regression assumes:
1. the target depends linearly on the features
2. each feature contributes additively
3. the best model is the one with minimum squared error

So it finds a plane:
$$
\hat y=b_0+b_1x_1+b_2x_2
$$
that stays as close as possible to the observed data points.

---
## Understanding the 3D Plot
With two features, the model can be visualized in 3D:
- horizontal axis 1 = $x_1$
- horizontal axis 2 = $x_2$
- vertical axis = $y$

The blue points are actual data.
The surface is the regression plane.

Code:
```python
ax.scatter(X[:, 0], X[:, 1], y, label="Data")
ax.plot_surface(x1_grid, x2_grid, z_grid, alpha=0.5)
```

Concept:
- scatter points show real observations
- plane shows model predictions
- a good model has points lying near the plane

---
## Practical Notes
## 1. Multiple features
Unlike simple linear regression, multiple linear regression uses:
$$
x_1,x_2,\dots,x_n
$$
So it can capture the effect of several variables together.

## 2. Coefficient meaning
If:
$$
\hat y=b_0+b_1x_1+b_2x_2
$$
then:
- $b_1$ = change in $y$ for 1 unit increase in $x_1$, keeping $x_2$ fixed
- $b_2$ = change in $y$ for 1 unit increase in $x_2$, keeping $x_1$ fixed

## 3. Multicollinearity
If input features are highly correlated, coefficient estimates can become unstable.

## 4. Linear assumption
If the true relationship is non-linear, this model may underfit.

---
## Exam-Oriented Summary
## Definition
Multiple Linear Regression predicts a continuous output using two or more input features.

## Model Equation
$$
\hat y=b_0+b_1x_1+b_2x_2+\cdots+b_nx_n
$$

## Normal Equation
$$
\theta=(X_b^TX_b)^{-1}X_b^Ty
$$

## Performance Metric
$$
R^2=1-\frac{\sum(y-\hat y)^2}{\sum(y-\bar y)^2}
$$

## Steps
1. arrange input matrix
2. add bias column
3. compute coefficients using Normal Equation
4. predict outputs
5. evaluate using $R^2$

---
## Very Short Revision
- add bias column
- apply Normal Equation
- get intercept and coefficients
- form regression plane
- predict target values
- evaluate using $R^2$

Main formulas:
$$
\hat y=b_0+b_1x_1+b_2x_2
$$
$$
\theta=(X_b^TX_b)^{-1}X_b^Ty
$$

---
## Final Takeaway
Multiple Linear Regression extends simple linear regression to multiple input features.
Instead of fitting a line, it fits a plane or hyperplane.
It learns coefficients using the Normal Equation and predicts continuous values using:
$$
\hat y=b_0+b_1x_1+b_2x_2+\cdots+b_nx_n
$$
For your sample dataset, it should learn values close to the true generating rule:
$$
y=1+2x_1+3x_2+\text{noise}
$$
so the fitted coefficients should be close to:
$$
b_0\approx 1,\quad b_1\approx 2,\quad b_2\approx 3
$$
