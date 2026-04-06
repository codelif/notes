# K-Nearest Neighbor
KNN is an **instance-based** or **lazy learning** algorithm.
It does not learn an explicit equation during training.
Instead, it:
- stores the training data
- waits until a new test point comes
- finds the nearest stored examples
- predicts using those neighbors

There are two common versions:
- **KNN Classification** -> predict the **most common class**
- **KNN Regression** -> predict the **mean of neighbor target values**

---
## Main Idea
Each sample is a point in $n$-dimensional space:
$$
x=\langle x_1,x_2,\dots,x_n\rangle
$$
To predict for a new point, KNN measures the distance from that point to all training points.

The most common distance used is **Euclidean distance**:
$$
d(x^{(a)},x^{(b)})=\sqrt{\sum_{j=1}^{n}(x_j^{(a)}-x_j^{(b)})^2}
$$

Then:
- in **classification**, choose the majority class among the nearest $K$ neighbors
- in **regression**, choose the average target among the nearest $K$ neighbors

---
# Part A: KNN Classification
## What It Does
Given a new test point:
1. compute its distance to every training point
2. sort distances
3. pick the nearest $K$
4. return the majority class

So the prediction rule is:
$$
\hat y=\text{mode of the }K\text{ nearest labels}
$$

---
## Short and Clean KNN Classification Code
```python
import numpy as np
from collections import Counter

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)

    def _distance(self, a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    def _predict_one(self, x):
        distances = [self._distance(x, x_train) for x_train in self.X_train]
        k_idx = np.argsort(distances)[:self.k]
        k_labels = self.y_train[k_idx]
        return Counter(k_labels).most_common(1)[0][0]

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(x) for x in X])

X_train = np.array([
    [1, 1],
    [2, 1],
    [4, 3],
    [5, 4],
    [6, 5]
], dtype=float)

y_train = np.array([0, 0, 1, 1, 1])

X_test = np.array([
    [3, 2],
    [5, 5]
], dtype=float)

clf = KNNClassifier(k=3)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)

print("Predictions:", pred)
```

---
## Small Classification Example
Training data:
$$
(1,1)\to 0,\quad (2,1)\to 0,\quad (4,3)\to 1,\quad (5,4)\to 1,\quad (6,5)\to 1
$$

Test point:
$$
x_t=(3,2)
$$

Take:
$$
K=3
$$

---
## Solving the Classification Example Step by Step
We compute distance from $(3,2)$ to every training point.

### Distance to $(1,1)$
$$
d=\sqrt{(3-1)^2+(2-1)^2}=\sqrt{4+1}=\sqrt{5}\approx 2.236
$$

### Distance to $(2,1)$
$$
d=\sqrt{(3-2)^2+(2-1)^2}=\sqrt{1+1}=\sqrt{2}\approx 1.414
$$

### Distance to $(4,3)$
$$
d=\sqrt{(3-4)^2+(2-3)^2}=\sqrt{1+1}=\sqrt{2}\approx 1.414
$$

### Distance to $(5,4)$
$$
d=\sqrt{(3-5)^2+(2-4)^2}=\sqrt{4+4}=\sqrt{8}\approx 2.828
$$

### Distance to $(6,5)$
$$
d=\sqrt{(3-6)^2+(2-5)^2}=\sqrt{9+9}=\sqrt{18}\approx 4.243
$$

Now sort the distances:

| Point | Label | Distance |
|---|---:|---:|
| $(2,1)$ | 0 | 1.414 |
| $(4,3)$ | 1 | 1.414 |
| $(1,1)$ | 0 | 2.236 |
| $(5,4)$ | 1 | 2.828 |
| $(6,5)$ | 1 | 4.243 |

Nearest 3 labels:
$$
[0,1,0]
$$

Majority class:
$$
0
$$

So:
$$
\hat y=0
$$

That is exactly how KNN classification works.

---
## KNN Classification Algorithm
### Step 1: Store training data
Unlike linear models, KNN does not compute weights during training.

Code:
```python
def fit(self, X, y):
    self.X_train = np.asarray(X, dtype=float)
    self.y_train = np.asarray(y)
```

Concept:
Training in KNN simply means **memorizing** the dataset.

---
### Step 2: Compute distance
Equation:
$$
d(x,x_i)=\sqrt{\sum_{j=1}^{n}(x_j-x_{ij})^2}
$$

Code:
```python
def _distance(self, a, b):
    return np.sqrt(np.sum((a - b) ** 2))
```

Concept:
This measures how close a training sample is to the test sample.

Smaller distance means more similar.

---
### Step 3: Rank neighbors
Code:
```python
distances = [self._distance(x, x_train) for x_train in self.X_train]
k_idx = np.argsort(distances)[:self.k]
```

Concept:
- compute all distances
- sort them
- keep the indices of the nearest $K$

---
### Step 4: Vote by majority
Code:
```python
k_labels = self.y_train[k_idx]
return Counter(k_labels).most_common(1)[0][0]
```

Concept:
Among the nearest neighbors, whichever class occurs most is chosen.

Equation:
$$
\hat y=\text{mode}(y_{(1)},y_{(2)},\dots,y_{(K)})
$$

---
## Concept -> Equation -> Code Mapping for Classification
## 1. Represent samples as points
Concept:
Each row is a point in feature space.

Equation:
$$
x=\langle x_1,x_2,\dots,x_n\rangle
$$

Code:
```python
self.X_train = np.asarray(X, dtype=float)
```

---
## 2. Measure closeness
Concept:
Similarity is measured using Euclidean distance.

Equation:
$$
d(x^{(a)},x^{(b)})=\sqrt{\sum_{j=1}^{n}(x_j^{(a)}-x_j^{(b)})^2}
$$

Code:
```python
def _distance(self, a, b):
    return np.sqrt(np.sum((a - b) ** 2))
```

---
## 3. Select nearest neighbors
Concept:
Prediction depends only on nearby training samples.

Code:
```python
k_idx = np.argsort(distances)[:self.k]
```

---
## 4. Use majority vote
Concept:
Classification chooses the most frequent class.

Equation:
$$
\hat y=\operatorname{mode}(k\text{ nearest labels})
$$

Code:
```python
return Counter(k_labels).most_common(1)[0][0]
```

---
## Why KNN Classification Works
The assumption is:
> points that are close in feature space tend to have the same class

So instead of learning a global rule, KNN makes a **local decision** around the test point.

That is why it is called **instance-based learning**.

---
# Part B: KNN Regression
## What It Does
KNN regression follows the same steps as classification:
1. compute distance to all training points
2. choose the nearest $K$
3. average their target values

Prediction rule:
$$
\hat y=\frac{1}{K}\sum_{i=1}^{K} y_{(i)}
$$
where $y_{(i)}$ are the target values of the nearest $K$ neighbors.

---
## Short and Clean KNN Regression Code
```python
import numpy as np

class KNNRegressor:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y, dtype=float)

    def _distance(self, a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    def _predict_one(self, x):
        distances = [self._distance(x, x_train) for x_train in self.X_train]
        k_idx = np.argsort(distances)[:self.k]
        k_values = self.y_train[k_idx]
        return np.mean(k_values)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(x) for x in X])

X_train = np.array([[1], [2], [3], [6], [7]], dtype=float)
y_train = np.array([30000, 35000, 40000, 70000, 75000], dtype=float)

X_test = np.array([[4]], dtype=float)

reg = KNNRegressor(k=3)
reg.fit(X_train, y_train)
pred = reg.predict(X_test)

print("Prediction:", pred)
```

---
## Small Regression Example
Suppose training data is:

| YearsExperience | Salary |
|---:|---:|
| 1 | 30000 |
| 2 | 35000 |
| 3 | 40000 |
| 6 | 70000 |
| 7 | 75000 |

Test point:
$$
x_t=4
$$

Take:
$$
K=3
$$

---
## Solving the Regression Example Step by Step
We compute the distance from 4 to each training point.

### Distance to 1
$$
d=|4-1|=3
$$

### Distance to 2
$$
d=|4-2|=2
$$

### Distance to 3
$$
d=|4-3|=1
$$

### Distance to 6
$$
d=|4-6|=2
$$

### Distance to 7
$$
d=|4-7|=3
$$

Sorted distances:

| Point | Target | Distance |
|---:|---:|---:|
| 3 | 40000 | 1 |
| 2 | 35000 | 2 |
| 6 | 70000 | 2 |
| 1 | 30000 | 3 |
| 7 | 75000 | 3 |

Nearest 3 target values:
$$
[40000,35000,70000]
$$

Prediction is their mean:
$$
\hat y=\frac{40000+35000+70000}{3}
$$
$$
\hat y=\frac{145000}{3}\approx 48333.33
$$

So the predicted salary is:
$$
48333.33
$$

---
## KNN Regression Algorithm
### Step 1: Store training data
Code:
```python
def fit(self, X, y):
    self.X_train = np.asarray(X, dtype=float)
    self.y_train = np.asarray(y, dtype=float)
```

Concept:
Just store examples and target values.

---
### Step 2: Compute all distances
Equation:
$$
d(x,x_i)=\sqrt{\sum_{j=1}^{n}(x_j-x_{ij})^2}
$$

Code:
```python
distances = [self._distance(x, x_train) for x_train in self.X_train]
```

Concept:
Measure how close the new point is to all training samples.

---
### Step 3: Pick nearest $K$
Code:
```python
k_idx = np.argsort(distances)[:self.k]
```

Concept:
Keep the closest $K$ neighbors only.

---
### Step 4: Average their target values
Equation:
$$
\hat y=\frac{1}{K}\sum_{i=1}^{K} y_{(i)}
$$

Code:
```python
k_values = self.y_train[k_idx]
return np.mean(k_values)
```

Concept:
Regression prediction is the average of nearby outputs.

---
## Concept -> Equation -> Code Mapping for Regression
## 1. Use the same distance idea
Concept:
Near points should have similar target values.

Equation:
$$
d(x^{(a)},x^{(b)})=\sqrt{\sum_{j=1}^{n}(x_j^{(a)}-x_j^{(b)})^2}
$$

Code:
```python
def _distance(self, a, b):
    return np.sqrt(np.sum((a - b) ** 2))
```

---
## 2. Find local neighborhood
Concept:
Prediction uses local samples rather than a global fitted line.

Code:
```python
k_idx = np.argsort(distances)[:self.k]
```

---
## 3. Average local outputs
Concept:
Regression uses the mean of neighbor target values.

Equation:
$$
\hat y=\frac{1}{K}\sum_{i=1}^{K} y_{(i)}
$$

Code:
```python
return np.mean(k_values)
```

---
# KNN Classification vs KNN Regression
| Aspect | KNN Classification | KNN Regression |
|---|---|---|
| Output type | class label | continuous value |
| Final rule | majority vote | mean of neighbors |
| Formula | mode of nearest labels | average of nearest targets |

---
## Unified Intuition
KNN does not learn a formula like:
$$
y=mx+b
$$
or:
$$
\hat y=\sigma(wx+b)
$$

Instead, it says:
> for this new point, let me look around nearby training points and decide locally

So every test sample gets its own local prediction rule.

That is why KNN is called:
- **lazy learning**
- **instance-based learning**

---
## Choosing the Value of $K$
The value of $K$ controls smoothness.

### Small $K$
- sensitive to noise
- more flexible
- may overfit

### Large $K$
- smoother decision
- may underfit
- local details may be lost

A common rough idea:
$$
K\approx \sqrt{m}
$$
where $m$ is the number of training samples

But in practice, $K$ is usually chosen using validation.

---
## Why Feature Scaling Matters
KNN depends entirely on distance.
So features with larger numeric ranges dominate the distance.

Example:
- Age may range from 20 to 80
- Glucose may range from 0 to 200

Then Glucose can dominate Euclidean distance.

So scaling is often important:
$$
x'=\frac{x-\mu}{\sigma}
$$

Without scaling, nearest neighbors may be misleading.

---
## Time Cost of KNN
KNN is cheap during training but expensive during prediction.

### Training
- just store the data

### Prediction
For every test point:
- compute distance to all training points
- sort them
- then predict

So prediction can be costly when the training set is large.

This is one of the main disadvantages of KNN.

---
## Advantages of KNN
- very simple
- no training optimization needed
- easy to understand
- works for both classification and regression
- naturally handles multi-class classification

---
## Limitations of KNN
- prediction is slow on large datasets
- sensitive to feature scale
- sensitive to irrelevant features
- choice of $K$ matters a lot
- can perform poorly in very high dimensions

This last issue is related to the **curse of dimensionality**.

---
## Exam-Oriented Summary
## Definition
KNN is an instance-based supervised learning algorithm that predicts a new sample using the nearest stored training samples.

## Distance Formula
$$
d(x^{(a)},x^{(b)})=\sqrt{\sum_{j=1}^{n}(x_j^{(a)}-x_j^{(b)})^2}
$$

## Classification Rule
$$
\hat y=\text{mode of }K\text{ nearest labels}
$$

## Regression Rule
$$
\hat y=\frac{1}{K}\sum_{i=1}^{K} y_{(i)}
$$

## Steps
1. store training data
2. compute distance from test point to all training points
3. sort by distance
4. select nearest $K$
5. classify by majority vote or regress by mean

---
## Very Short Revision
### KNN Classification
- find nearest $K$
- take majority class
- output class label

### KNN Regression
- find nearest $K$
- take mean of target values
- output continuous value

Main formula:
$$
d(x^{(a)},x^{(b)})=\sqrt{\sum_{j=1}^{n}(x_j^{(a)}-x_j^{(b)})^2}
$$

---
## Final Takeaway
KNN is one of the simplest machine learning algorithms.
It does not build a model during training, but predicts by comparing a new point to stored training examples.
For **classification**, it uses the **majority class** of the nearest neighbors.
For **regression**, it uses the **mean target value** of the nearest neighbors.
Its simplicity makes it excellent for understanding local learning, but its prediction cost and sensitivity to scaling are important limitations.
