# Logistic Regression
Logistic Regression is a **binary classification** algorithm used when the output belongs to one of two classes:
$$
y\in\{0,1\}
$$
It does **not** predict the class directly using a line. Instead, it predicts a **probability** using the sigmoid function, then converts that probability into class `0` or `1`.
The model is:
$$
z=w_1x_1+w_2x_2+\cdots+w_nx_n+b
$$
$$
\hat y=\sigma(z)=\frac{1}{1+e^{-z}}
$$
where:
- $z$ = linear score
- $\sigma(z)$ = sigmoid function
- $\hat y$ = predicted probability that class is 1

If:
$$
\hat y\ge 0.5 \Rightarrow \text{predict }1
$$
$$
\hat y<0.5 \Rightarrow \text{predict }0
$$

---
## Main Idea
Linear Regression gives any real number as output, but classification needs a value between 0 and 1.
So Logistic Regression first computes a linear combination:
$$
z=Xw+b
$$
then applies the sigmoid:
$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$
This maps any real number into:
$$
(0,1)
$$
So the output can be interpreted as a probability.

---
## Short and Clean Code
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class LogisticRegressionScratch:
    def __init__(self, lr=0.1, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = 0.0
        self.loss_history = []

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def _loss(self, y, p):
        eps = 1e-9
        p = np.clip(p, eps, 1 - eps)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        m, n = X.shape
        self.w = np.zeros(n)

        for _ in range(self.epochs):
            z = X @ self.w + self.b
            p = self._sigmoid(z)

            dw = (X.T @ (p - y)) / m
            db = np.mean(p - y)

            self.w -= self.lr * dw
            self.b -= self.lr * db

            self.loss_history.append(self._loss(y, p))
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        return self._sigmoid(X @ self.w + self.b)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

np.random.seed(42)
X = np.random.rand(200, 2) * 10
y = (X[:, 0] + X[:, 1] > 10).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegressionScratch(lr=0.1, epochs=1000)
model.fit(X_train, y_train)

pred = model.predict(X_test)
acc = np.mean(pred == y_test)

print("Weights:", np.round(model.w, 4))
print("Bias:", round(model.b, 4))
print("Accuracy:", round(acc, 4))

plt.plot(model.loss_history)
plt.title("Loss Convergence")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.grid(True)
plt.show()
```

---
## What This Code Does
This example creates 2D points:
$$
X=[x_1,x_2]
$$
and labels them using:
$$
y=
\begin{cases}
1,&x_1+x_2>10\\
0,&x_1+x_2\le 10
\end{cases}
$$
So the true decision boundary is:
$$
x_1+x_2=10
$$
This is a binary classification problem.

---
## Step-by-Step Algorithm
## Step 1: Create the dataset
Code:
```python
np.random.seed(42)
X = np.random.rand(200, 2) * 10
y = (X[:, 0] + X[:, 1] > 10).astype(int)
```

Concept:
- `X` contains 200 samples
- each sample has 2 features: $x_1,x_2$
- class label depends on whether the sum is greater than 10

Equation:
$$
y=
\begin{cases}
1,&x_1+x_2>10\\
0,&\text{otherwise}
\end{cases}
$$

Meaning:
- points above the line belong to class 1
- points below the line belong to class 0

---
## Step 2: Split into train and test sets
Code:
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

Concept:
- training data is used to learn parameters
- test data is used to check performance on unseen data

Here:
- 80% for training
- 20% for testing

---
## Step 3: Standardize features
Code:
```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

Concept:
Gradient descent works better when features are on similar scales.

Standardization formula:
$$
x_{\text{scaled}}=\frac{x-\mu}{\sigma}
$$
where:
- $\mu$ = mean of the feature
- $\sigma$ = standard deviation

Why it helps:
- faster convergence
- more stable updates
- one feature does not dominate another due to large magnitude

---
## Step 4: Compute the linear score
For each sample, Logistic Regression first computes:
$$
z=w_1x_1+w_2x_2+\cdots+w_nx_n+b
$$
In vector form:
$$
z=Xw+b
$$

Code:
```python
z = X @ self.w + self.b
```

Concept:
This is the same linear part used in linear models.
But here it is **not the final output**.
It is only the input to the sigmoid function.

---
## Step 5: Apply sigmoid to get probability
Equation:
$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

Code:
```python
p = self._sigmoid(z)
```

Concept:
The sigmoid compresses any real number into a value between 0 and 1.

Examples:
- if $z=0$:
$$
\sigma(0)=\frac{1}{1+e^0}=0.5
$$
- if $z$ is large positive, probability is close to 1
- if $z$ is large negative, probability is close to 0

So:
$$
p=P(y=1\mid X)
$$

---
## Step 6: Measure error using cross-entropy loss
For Logistic Regression, we do **not** use mean squared error.
We use **cross-entropy loss**:
$$
J(w,b)=-\frac{1}{m}\sum_{i=1}^{m}\left[y_i\log(\hat y_i)+(1-y_i)\log(1-\hat y_i)\right]
$$

Code:
```python
def _loss(self, y, p):
    eps = 1e-9
    p = np.clip(p, eps, 1 - eps)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
```

Concept:
- if actual class is 1, we want $\hat y$ close to 1
- if actual class is 0, we want $\hat y$ close to 0
- wrong confident predictions are penalized heavily

Why clipping is used:
- `log(0)` is undefined
- so probabilities are clipped slightly away from 0 and 1

---
## Step 7: Compute gradients
To reduce loss, we update weights and bias using gradient descent.

Gradient formulas:
$$
\frac{\partial J}{\partial w}=\frac{1}{m}X^T(\hat y-y)
$$
$$
\frac{\partial J}{\partial b}=\frac{1}{m}\sum(\hat y-y)
$$

Code:
```python
dw = (X.T @ (p - y)) / m
db = np.mean(p - y)
```

Concept:
- `dw` tells how weights should change
- `db` tells how bias should change
- if prediction is too large, parameters are pushed downward
- if prediction is too small, parameters are pushed upward

---
## Step 8: Update parameters
Gradient descent update rule:
$$
w:=w-\alpha \frac{\partial J}{\partial w}
$$
$$
b:=b-\alpha \frac{\partial J}{\partial b}
$$
where $\alpha$ is the learning rate.

Code:
```python
self.w -= self.lr * dw
self.b -= self.lr * db
```

Concept:
- move parameters in the direction that reduces loss
- repeat many times until learning stabilizes

---
## Step 9: Convert probabilities to classes
After training, predicted probability is converted to class label.

Rule:
$$
\hat y=
\begin{cases}
1,&p\ge 0.5\\
0,&p<0.5
\end{cases}
$$

Code:
```python
return (self.predict_proba(X) >= 0.5).astype(int)
```

Concept:
- probabilities are continuous
- classification needs discrete labels

---
## Step 10: Measure accuracy
Code:
```python
pred = model.predict(X_test)
acc = np.mean(pred == y_test)
```

Equation:
$$
\text{Accuracy}=\frac{\text{Number of correct predictions}}{\text{Total predictions}}
$$

Concept:
Accuracy tells what fraction of test samples were classified correctly.

---
## Concept -> Equation -> Code Mapping
## 1. Model parameters
Concept:
The model must learn weights and bias.

Equation:
$$
z=Xw+b
$$

Code:
```python
self.w = np.zeros(n)
self.b = 0.0
```

Meaning:
- start with all weights as 0
- start with bias as 0

---
## 2. Probability model
Concept:
Turn linear score into probability.

Equation:
$$
\hat y=\sigma(z)=\frac{1}{1+e^{-z}}
$$

Code:
```python
def _sigmoid(self, z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))
```

Why `clip`:
- avoids overflow for very large positive or negative values

---
## 3. Forward pass
Concept:
Compute predictions from current parameters.

Equation:
$$
z=Xw+b
$$
$$
p=\sigma(z)
$$

Code:
```python
z = X @ self.w + self.b
p = self._sigmoid(z)
```

Meaning:
- `z` = raw score
- `p` = predicted probability

---
## 4. Loss calculation
Concept:
See how wrong the predictions are.

Equation:
$$
J(w,b)=-\frac{1}{m}\sum \left[y\log(p)+(1-y)\log(1-p)\right]
$$

Code:
```python
self.loss_history.append(self._loss(y, p))
```

Meaning:
- each iteration stores loss
- useful for checking whether training is improving

---
## 5. Backward pass
Concept:
Find how parameters affect the loss.

Equation:
$$
\frac{\partial J}{\partial w}=\frac{1}{m}X^T(p-y)
$$
$$
\frac{\partial J}{\partial b}=\frac{1}{m}\sum(p-y)
$$

Code:
```python
dw = (X.T @ (p - y)) / m
db = np.mean(p - y)
```

Meaning:
These gradients guide the update step.

---
## 6. Learning step
Concept:
Improve the model gradually.

Equation:
$$
w:=w-\alpha dw
$$
$$
b:=b-\alpha db
$$

Code:
```python
self.w -= self.lr * dw
self.b -= self.lr * db
```

Meaning:
Repeated updates make the model better at classification.

---
## Worked Example on One Sample
Suppose after some training, for one sample:
$$
x=[0.8,1.2]
$$
and the model has:
$$
w=[1.5,1.0],\quad b=-0.4
$$

### Step 1: Compute score
$$
z=(1.5)(0.8)+(1.0)(1.2)-0.4
$$
$$
z=1.2+1.2-0.4=2.0
$$

### Step 2: Apply sigmoid
$$
\hat y=\sigma(2)=\frac{1}{1+e^{-2}}
$$
$$
\hat y\approx 0.8808
$$

### Step 3: Classify
Since:
$$
0.8808>0.5
$$
prediction is:
$$
1
$$

So this sample is classified as class 1.

---
## Why Logistic Regression Works
The model learns a boundary where the probability changes from class 0 to class 1.
For two features, the decision boundary is:
$$
w_1x_1+w_2x_2+b=0
$$
Because:
$$
\sigma(z)=0.5 \text{ when } z=0
$$
So:
- if $z>0$, class tends toward 1
- if $z<0$, class tends toward 0

This creates a **linear decision boundary**.

For your dataset, true labels come from:
$$
x_1+x_2>10
$$
So Logistic Regression is a good fit because the classes are separable by a line.

---
## What the Loss Curve Means
Code:
```python
plt.plot(model.loss_history)
plt.title("Loss Convergence")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.grid(True)
plt.show()
```

Concept:
- at the beginning, loss is high
- during learning, loss should decrease
- a downward curve means gradient descent is working

If the curve:
- decreases smoothly -> learning is stable
- oscillates wildly -> learning rate may be too high
- decreases very slowly -> learning rate may be too low

---
## Practical Notes
## 1. Feature scaling matters
Because Logistic Regression uses gradient descent, features with large values can slow down convergence.
That is why:
$$
x_{\text{scaled}}=\frac{x-\mu}{\sigma}
$$
is important.

## 2. Learning rate matters
If learning rate $\alpha$ is:
- too high -> training may diverge
- too low -> training becomes very slow

Typical starting values:
$$
0.01 \text{ to } 0.1
$$

## 3. Linear boundary limitation
Logistic Regression assumes the boundary is linear:
$$
w_1x_1+w_2x_2+\cdots+w_nx_n+b=0
$$
If data is non-linear, you may need:
- feature engineering
- polynomial features
- another model

## 4. Correlated features can cause issues
If features are strongly correlated, learning may become unstable.
This is called multicollinearity.

---
## Exam-Oriented Summary
## Definition
Logistic Regression is a supervised learning algorithm used for **binary classification**.

## Model
$$
z=Xw+b
$$
$$
\hat y=\frac{1}{1+e^{-z}}
$$

## Decision Rule
$$
\hat y\ge 0.5 \Rightarrow 1
$$
$$
\hat y<0.5 \Rightarrow 0
$$

## Loss Function
$$
J(w,b)=-\frac{1}{m}\sum\left[y\log(\hat y)+(1-y)\log(1-\hat y)\right]
$$

## Gradient Descent Updates
$$
w:=w-\alpha\frac{1}{m}X^T(\hat y-y)
$$
$$
b:=b-\alpha\frac{1}{m}\sum(\hat y-y)
$$

## Uses
- spam detection
- disease prediction
- pass/fail prediction
- yes/no classification tasks

---
## Very Short Revision
- Compute linear score:
$$
z=Xw+b
$$
- Apply sigmoid:
$$
\hat y=\frac{1}{1+e^{-z}}
$$
- Compute cross-entropy loss
- Update weights and bias using gradient descent
- Convert probability to class using threshold 0.5

---
## Final Takeaway
Logistic Regression is a simple but powerful classification algorithm.
It learns a linear decision boundary, uses sigmoid to output probabilities, and improves itself by minimizing cross-entropy loss using gradient descent.
For your example, it works well because the classes are generated by a rule that is approximately linearly separable.
