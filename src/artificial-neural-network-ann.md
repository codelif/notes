# Artificial Neural Network (ANN)
An Artificial Neural Network is a model made of layers of neurons.
A basic ANN has:
- input layer
- hidden layer(s)
- output layer
- weights and biases
- activation functions

A neuron computes:
$$
z=Wx+b
$$
and then applies an activation function:
$$
a=g(z)
$$

For this example, we build a **2-layer neural network** for **binary classification**:
- hidden layer uses **ReLU**
- output layer uses **Sigmoid**

The final output is a probability:
$$
\hat y\in(0,1)
$$
and prediction is:
$$
\hat y\ge 0.5 \Rightarrow 1,\quad \hat y<0.5 \Rightarrow 0
$$

---
## What This Network Learns
For input $X$, the network computes:
$$
Z_1=W_1X+b_1
$$
$$
A_1=\text{ReLU}(Z_1)
$$
$$
Z_2=W_2A_1+b_2
$$
$$
A_2=\sigma(Z_2)=\frac{1}{1+e^{-Z_2}}
$$

Where:
- $Z_1,Z_2$ = linear outputs
- $A_1$ = hidden layer activations
- $A_2$ = final predicted probability

---
## Short and Clean Code
```python
import numpy as np

class SimpleANN:
    def __init__(self, input_size, hidden_size, lr=0.1, epochs=10000):
        np.random.seed(42)
        self.lr = lr
        self.epochs = epochs
        self.W1 = np.random.randn(hidden_size, input_size) * 0.1
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = np.random.randn(1, hidden_size) * 0.1
        self.b2 = np.zeros((1, 1))
        self.costs = []

    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_deriv(self, Z):
        return (Z > 0).astype(float)

    def sigmoid(self, Z):
        Z = np.clip(Z, -500, 500)
        return 1 / (1 + np.exp(-Z))

    def forward(self, X):
        Z1 = self.W1 @ X + self.b1
        A1 = self.relu(Z1)
        Z2 = self.W2 @ A1 + self.b2
        A2 = self.sigmoid(Z2)
        cache = (X, Z1, A1, Z2, A2)
        return A2, cache

    def compute_cost(self, Y, A2):
        eps = 1e-9
        A2 = np.clip(A2, eps, 1 - eps)
        return -np.mean(Y * np.log(A2) + (1 - Y) * np.log(1 - A2))

    def backward(self, Y, cache):
        X, Z1, A1, Z2, A2 = cache
        m = X.shape[1]

        dZ2 = A2 - Y
        dW2 = (dZ2 @ A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m

        dZ1 = (self.W2.T @ dZ2) * self.relu_deriv(Z1)
        dW1 = (dZ1 @ X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        return dW1, db1, dW2, db2

    def fit(self, X, Y):
        for i in range(self.epochs):
            A2, cache = self.forward(X)
            cost = self.compute_cost(Y, A2)
            dW1, db1, dW2, db2 = self.backward(Y, cache)

            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2

            self.costs.append(cost)
            if i % 1000 == 0:
                print(f"Epoch {i}: Cost = {cost:.6f}")

    def predict(self, X):
        A2, _ = self.forward(X)
        return (A2 >= 0.5).astype(int)

X = np.array([[0, 0, 1, 1],
              [0, 1, 0, 1]], dtype=float)

Y = np.array([[0, 0, 0, 1]], dtype=float)

model = SimpleANN(input_size=2, hidden_size=4, lr=0.1, epochs=10000)
model.fit(X, Y)

pred = model.predict(X)
print("Predictions:", pred)
```

---
## Dataset Used: AND Gate
The network is trained on the AND truth table:
$$
\begin{array}{c c|c}
x_1 & x_2 & y\\
\hline
0 & 0 & 0\\
0 & 1 & 0\\
1 & 0 & 0\\
1 & 1 & 1
\end{array}
$$

Input matrix:
```python
X = np.array([[0, 0, 1, 1],
              [0, 1, 0, 1]], dtype=float)
```

Target matrix:
```python
Y = np.array([[0, 0, 0, 1]], dtype=float)
```

Shape meaning:
- $X$ has shape $(2,4)$
- 2 input features
- 4 training examples

So each column is one example:
$$
X=
\begin{bmatrix}
0 & 0 & 1 & 1\\
0 & 1 & 0 & 1
\end{bmatrix}
$$

---
## Network Architecture
This ANN has:
- 2 input neurons
- 4 hidden neurons
- 1 output neuron

So parameter shapes are:
$$
W_1\in\mathbb{R}^{4\times 2},\quad b_1\in\mathbb{R}^{4\times 1}
$$
$$
W_2\in\mathbb{R}^{1\times 4},\quad b_2\in\mathbb{R}^{1\times 1}
$$

---
## Step-by-Step Algorithm
## Step 1: Initialize weights and biases
We begin with small random weights and zero biases.

Code:
```python
self.W1 = np.random.randn(hidden_size, input_size) * 0.1
self.b1 = np.zeros((hidden_size, 1))
self.W2 = np.random.randn(1, hidden_size) * 0.1
self.b2 = np.zeros((1, 1))
```

Concept:
- weights decide how strongly neurons influence the next layer
- biases shift the activation
- small random values break symmetry
- if all weights start the same, neurons learn the same thing

Why not large random weights:
- large values can make training unstable
- small values help smoother learning at the start

---
## Step 2: Hidden layer linear transformation
Each hidden neuron computes:
$$
Z_1=W_1X+b_1
$$

Code:
```python
Z1 = self.W1 @ X + self.b1
```

Concept:
This is the weighted sum of inputs plus bias.

For one hidden neuron:
$$
z=w_1x_1+w_2x_2+b
$$

Since there are 4 hidden neurons, this is done 4 times in parallel.

---
## Step 3: Apply ReLU activation
ReLU function is:
$$
\text{ReLU}(z)=\max(0,z)
$$

Code:
```python
A1 = self.relu(Z1)
```

and:
```python
def relu(self, Z):
    return np.maximum(0, Z)
```

Concept:
- negative values become 0
- positive values remain unchanged

Why ReLU:
- introduces non-linearity
- lets the network learn more complex patterns
- simple and efficient

Without activation, multiple layers would collapse into just one linear transformation.

---
## Step 4: Output layer linear transformation
Now hidden activations are passed to the output neuron:
$$
Z_2=W_2A_1+b_2
$$

Code:
```python
Z2 = self.W2 @ A1 + self.b2
```

Concept:
This combines the hidden-layer outputs into one final score.

---
## Step 5: Apply Sigmoid to get probability
Sigmoid function:
$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

Code:
```python
A2 = self.sigmoid(Z2)
```

and:
```python
def sigmoid(self, Z):
    Z = np.clip(Z, -500, 500)
    return 1 / (1 + np.exp(-Z))
```

Concept:
- converts raw score into probability
- output is between 0 and 1
- suitable for binary classification

Meaning:
$$
A_2=P(y=1\mid X)
$$

Why `clip` is used:
- prevents overflow in `exp`
- improves numerical stability

---
## Step 6: Compute the cost
For binary classification, we use **binary cross-entropy loss**:
$$
J=-\frac{1}{m}\sum\left[Y\log(A_2)+(1-Y)\log(1-A_2)\right]
$$

Code:
```python
def compute_cost(self, Y, A2):
    eps = 1e-9
    A2 = np.clip(A2, eps, 1 - eps)
    return -np.mean(Y * np.log(A2) + (1 - Y) * np.log(1 - A2))
```

Concept:
- if actual label is 1, we want output close to 1
- if actual label is 0, we want output close to 0
- confident wrong predictions get heavily penalized

Why clip again:
- avoids $\log(0)$ which is undefined

---
## Step 7: Backpropagation for output layer
The error at the output layer is:
$$
dZ_2=A_2-Y
$$

Code:
```python
dZ2 = A2 - Y
dW2 = (dZ2 @ A1.T) / m
db2 = np.sum(dZ2, axis=1, keepdims=True) / m
```

Equations:
$$
dW_2=\frac{1}{m}dZ_2A_1^T
$$
$$
db_2=\frac{1}{m}\sum dZ_2
$$

Concept:
This tells how much the output weights and bias contributed to the error.

---
## Step 8: Backpropagation for hidden layer
The hidden layer error is:
$$
dZ_1=(W_2^TdZ_2)\odot \text{ReLU}'(Z_1)
$$

Code:
```python
dZ1 = (self.W2.T @ dZ2) * self.relu_deriv(Z1)
dW1 = (dZ1 @ X.T) / m
db1 = np.sum(dZ1, axis=1, keepdims=True) / m
```

Equations:
$$
dW_1=\frac{1}{m}dZ_1X^T
$$
$$
db_1=\frac{1}{m}\sum dZ_1
$$

ReLU derivative:
$$
\text{ReLU}'(z)=
\begin{cases}
1,&z>0\\
0,&z\le 0
\end{cases}
$$

Code:
```python
def relu_deriv(self, Z):
    return (Z > 0).astype(float)
```

Concept:
- output error is sent backward into the hidden layer
- only active ReLU neurons pass gradient
- this is how the network learns internal representations

---
## Step 9: Update parameters
Gradient descent update rule:
$$
W:=W-\alpha dW
$$
$$
b:=b-\alpha db
$$

Code:
```python
self.W1 -= self.lr * dW1
self.b1 -= self.lr * db1
self.W2 -= self.lr * dW2
self.b2 -= self.lr * db2
```

Concept:
- move parameters in the direction that reduces loss
- repeat this many times
- gradually improve predictions

Here:
- $\alpha$ is the learning rate
- a higher learning rate updates faster, but may overshoot
- a lower learning rate is safer, but slower

---
## Step 10: Make predictions
After training, the network outputs probabilities.
Convert them into classes using threshold 0.5:
$$
\hat y=
\begin{cases}
1,&A_2\ge 0.5\\
0,&A_2<0.5
\end{cases}
$$

Code:
```python
def predict(self, X):
    A2, _ = self.forward(X)
    return (A2 >= 0.5).astype(int)
```

---
## Concept -> Equation -> Code Mapping
## 1. Weighted input
Concept:
Each neuron forms a weighted sum of inputs.

Equation:
$$
z=Wx+b
$$

Code:
```python
Z1 = self.W1 @ X + self.b1
Z2 = self.W2 @ A1 + self.b2
```

---
## 2. Non-linearity
Concept:
Activation functions allow the network to learn beyond straight-line relationships.

Equations:
$$
A_1=\text{ReLU}(Z_1)
$$
$$
A_2=\sigma(Z_2)
$$

Code:
```python
A1 = self.relu(Z1)
A2 = self.sigmoid(Z2)
```

---
## 3. Forward propagation
Concept:
Data flows from input to hidden to output.

Equations:
$$
Z_1=W_1X+b_1
$$
$$
A_1=\text{ReLU}(Z_1)
$$
$$
Z_2=W_2A_1+b_2
$$
$$
A_2=\sigma(Z_2)
$$

Code:
```python
def forward(self, X):
    Z1 = self.W1 @ X + self.b1
    A1 = self.relu(Z1)
    Z2 = self.W2 @ A1 + self.b2
    A2 = self.sigmoid(Z2)
```

---
## 4. Loss measurement
Concept:
We need to measure how wrong predictions are.

Equation:
$$
J=-\frac{1}{m}\sum\left[Y\log(A_2)+(1-Y)\log(1-A_2)\right]
$$

Code:
```python
cost = self.compute_cost(Y, A2)
```

---
## 5. Error propagation backward
Concept:
The network computes gradients layer by layer from output back to input.

Equations:
$$
dZ_2=A_2-Y
$$
$$
dW_2=\frac{1}{m}dZ_2A_1^T
$$
$$
dZ_1=(W_2^TdZ_2)\odot \text{ReLU}'(Z_1)
$$
$$
dW_1=\frac{1}{m}dZ_1X^T
$$

Code:
```python
dZ2 = A2 - Y
dW2 = (dZ2 @ A1.T) / m
dZ1 = (self.W2.T @ dZ2) * self.relu_deriv(Z1)
dW1 = (dZ1 @ X.T) / m
```

---
## 6. Learning
Concept:
Use gradients to improve parameters.

Equation:
$$
\theta:=\theta-\alpha\nabla J
$$

Code:
```python
self.W1 -= self.lr * dW1
self.b1 -= self.lr * db1
self.W2 -= self.lr * dW2
self.b2 -= self.lr * db2
```

---
## Solving the AND Gate Example
The AND gate outputs 1 only when both inputs are 1:
$$
(0,0)\to 0
$$
$$
(0,1)\to 0
$$
$$
(1,0)\to 0
$$
$$
(1,1)\to 1
$$

During training:
- the network starts with random weights
- predictions are poor at first
- after many epochs, weights and biases adjust
- the cost decreases
- final outputs approach the correct AND values

Expected final prediction:
$$
[0,0,0,1]
$$

Code:
```python
pred = model.predict(X)
print("Predictions:", pred)
```

If training succeeds, output becomes:
```python
Predictions: [[0 0 0 1]]
```

---
## One Forward Pass Example
Suppose for one sample:
$$
x=
\begin{bmatrix}
1\\
1
\end{bmatrix}
$$

Assume one hidden neuron has:
$$
w=
\begin{bmatrix}
0.6 & 0.4
\end{bmatrix},\quad b=-0.2
$$

Then:
$$
z=0.6(1)+0.4(1)-0.2=0.8
$$

Apply ReLU:
$$
a=\max(0,0.8)=0.8
$$

Then output neuron may combine hidden activations and pass through sigmoid.
If final output score is:
$$
z_2=2.1
$$
then:
$$
A_2=\sigma(2.1)=\frac{1}{1+e^{-2.1}}\approx 0.8909
$$

Since:
$$
0.8909>0.5
$$
prediction is:
$$
1
$$

This is how the ANN converts inputs into a class decision.

---
## Why ANN Works
A neural network works because:
1. weights learn which inputs matter
2. biases shift decision boundaries
3. activation functions add non-linearity
4. backpropagation tells each parameter how it contributed to the error
5. gradient descent improves the parameters repeatedly

So the network gradually learns a function that maps input to output.

---
## Why Hidden Layers Matter
A single linear model can only learn a linear boundary.
A hidden layer with activation allows:
- combinations of features
- piecewise linear transformations
- more expressive decision boundaries

Even though AND is simple, this example demonstrates the full learning pipeline of an ANN.

---
## Cost Curve Meaning
The printed cost every 1000 epochs tells whether learning is working.

If cost decreases:
- predictions are improving
- gradients are useful
- parameter updates are moving in the correct direction

If cost does not decrease:
- learning rate may be wrong
- architecture may be unsuitable
- initialization may be poor

---
## Practical Notes
## 1. Initialization matters
Bad initialization can slow or break learning.

## 2. Learning rate matters
If learning rate is:
- too high -> unstable training
- too low -> very slow training

## 3. Activation choice matters
- ReLU is common in hidden layers
- Sigmoid is common for binary output

## 4. More layers increase capacity
Deeper networks can learn more complex patterns, but are also harder to train.

---
## Exam-Oriented Summary
## Definition
An ANN is a layered network of neurons that learns by adjusting weights and biases using backpropagation and gradient descent.

## Architecture Used
- 2 input neurons
- 1 hidden layer with 4 neurons
- 1 output neuron

## Important Equations
Hidden layer:
$$
Z_1=W_1X+b_1,\quad A_1=\text{ReLU}(Z_1)
$$
Output layer:
$$
Z_2=W_2A_1+b_2,\quad A_2=\sigma(Z_2)
$$
Sigmoid:
$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$
ReLU:
$$
\text{ReLU}(z)=\max(0,z)
$$
Loss:
$$
J=-\frac{1}{m}\sum\left[Y\log(A_2)+(1-Y)\log(1-A_2)\right]
$$
Gradient descent:
$$
W:=W-\alpha dW,\quad b:=b-\alpha db
$$

## Training Steps
1. initialize parameters
2. perform forward propagation
3. compute loss
4. perform backpropagation
5. update parameters
6. repeat for many epochs

---
## Very Short Revision
- input passes through weights and bias
- ReLU activates hidden layer
- sigmoid gives output probability
- cross-entropy measures error
- backpropagation computes gradients
- gradient descent updates weights
- repeat until cost decreases and predictions improve

---
## Final Takeaway
This ANN from scratch shows the complete neural-network learning process:
- forward propagation computes predictions
- loss measures error
- backpropagation computes gradients
- gradient descent updates parameters

For the AND gate dataset, the network learns the correct truth table:
$$
[0,0,0,1]
$$
which shows that it has successfully learned the mapping from inputs to outputs.
