# Decision Tree (ID3 & C4.5)
Both **ID3** and **C4.5** are decision tree algorithms used for **classification**.
They build a tree top-down by repeatedly choosing the best feature to split the dataset.

A decision tree has:
- **root/internal nodes** = feature tests
- **branches** = test outcomes
- **leaf nodes** = final class labels

The key difference is:
- **ID3** chooses splits using **Information Gain**
- **C4.5** improves ID3 by using **Gain Ratio** and can handle **continuous features** better

---
## 1. Decision Tree Idea
At each step, we want to ask the **best question** about the data.
A good split should make the child groups more pure.

If a node contains both classes mixed together, it is impure.
If a node contains only one class, it is pure.

Decision trees reduce impurity step by step until leaves are formed.

---
# Part A: ID3
## What ID3 Does
ID3 = **Iterative Dichotomiser 3**
It builds the tree recursively:
1. compute entropy of current dataset
2. compute information gain of every feature
3. choose the feature with the highest information gain
4. split the data on that feature
5. repeat for each subset

ID3 mainly works best with **categorical features**.

---
## ID3 Core Equations
### Entropy
Entropy measures uncertainty in a dataset:
$$
H(S)=-\sum_i p_i\log_2 p_i
$$
where:
- $S$ = current dataset
- $p_i$ = proportion of class $i$

For binary classification:
$$
H(S)= -p_0\log_2 p_0 - p_1\log_2 p_1
$$

Meaning:
- entropy = 0 $\Rightarrow$ perfectly pure
- entropy is high $\Rightarrow$ classes are mixed

### Information Gain
If we split dataset $S$ using feature $A$:
$$
IG(S,A)=H(S)-\sum_{v\in values(A)}\frac{|S_v|}{|S|}H(S_v)
$$
where:
- $S_v$ = subset where feature $A$ takes value $v$

Meaning:
- higher information gain = better split
- ID3 chooses the feature with **maximum** information gain

---
## Short and Clean ID3 Code
```python
import pandas as pd
import numpy as np

class ID3:
    def __init__(self):
        self.tree = None
        self.default_class = None

    def entropy(self, y):
        values, counts = np.unique(y, return_counts=True)
        p = counts / counts.sum()
        return -np.sum(p * np.log2(p + 1e-12))

    def information_gain(self, X, y, feature):
        parent_entropy = self.entropy(y)
        values, counts = np.unique(X[feature], return_counts=True)

        child_entropy = 0
        for v, c in zip(values, counts):
            y_sub = y[X[feature] == v]
            child_entropy += (c / len(X)) * self.entropy(y_sub)

        return parent_entropy - child_entropy

    def best_feature(self, X, y):
        gains = {f: self.information_gain(X, y, f) for f in X.columns}
        return max(gains, key=gains.get)

    def build(self, X, y):
        if len(np.unique(y)) == 1:
            return y.iloc[0]

        if X.empty:
            return y.mode()[0]

        best = self.best_feature(X, y)
        tree = {best: {}}

        for v in X[best].unique():
            mask = X[best] == v
            X_sub = X.loc[mask].drop(columns=[best])
            y_sub = y.loc[mask]

            if len(X_sub) == 0:
                tree[best][v] = y.mode()[0]
            else:
                tree[best][v] = self.build(X_sub, y_sub)

        return tree

    def fit(self, X, y):
        self.default_class = y.mode()[0]
        self.tree = self.build(X, y)
        return self

    def _predict_one(self, x, tree):
        if not isinstance(tree, dict):
            return tree

        feature = next(iter(tree))
        value = x.get(feature)

        if value not in tree[feature]:
            return self.default_class

        return self._predict_one(x, tree[feature][value])

    def predict(self, X):
        return X.apply(lambda row: self._predict_one(row, self.tree), axis=1)
```

---
## Small Example for ID3
We use a tiny categorical dataset:
```python
data = pd.DataFrame({
    "Weather": ["Sunny", "Sunny", "Overcast", "Rain", "Rain", "Rain", "Overcast", "Sunny"],
    "Wind":    ["Weak", "Strong", "Weak", "Weak", "Weak", "Strong", "Strong", "Weak"],
    "Play":    ["No", "No", "Yes", "Yes", "Yes", "No", "Yes", "No"]
})

X = data[["Weather", "Wind"]]
y = data["Play"]

model = ID3().fit(X, y)
print("ID3 Tree:", model.tree)
print("Predictions:", model.predict(X).tolist())
```

---
## Solving the ID3 Example Step by Step
Dataset:
$$
S=\{8\text{ samples}\}
$$
Target classes:
- `Yes` = 4
- `No` = 4

So root entropy:
$$
H(S)= -\frac{4}{8}\log_2\frac{4}{8}-\frac{4}{8}\log_2\frac{4}{8}=1
$$

So the root is maximally mixed.

### Step 1: Try splitting on `Weather`
Possible values:
- Sunny
- Overcast
- Rain

Subsets:
- `Sunny` $\to$ [No, No, No]
- `Overcast` $\to$ [Yes, Yes]
- `Rain` $\to$ [Yes, Yes, No]

Entropies:
#### Sunny subset
All are `No`, so:
$$
H(Sunny)=0
$$

#### Overcast subset
All are `Yes`, so:
$$
H(Overcast)=0
$$

#### Rain subset
2 Yes, 1 No:
$$
H(Rain)= -\frac{2}{3}\log_2\frac{2}{3}-\frac{1}{3}\log_2\frac{1}{3}\approx 0.9183
$$

Weighted entropy after splitting on Weather:
$$
\frac{3}{8}(0)+\frac{2}{8}(0)+\frac{3}{8}(0.9183)=0.3444
$$

Information gain:
$$
IG(S,\text{Weather})=1-0.3444=0.6556
$$

### Step 2: Try splitting on `Wind`
Possible values:
- Weak
- Strong

Subsets:
- `Weak` $\to$ [No, Yes, Yes, Yes, No]
- `Strong` $\to$ [No, No, Yes]

Entropies:
#### Weak
3 Yes, 2 No:
$$
H(Weak)= -\frac{3}{5}\log_2\frac{3}{5}-\frac{2}{5}\log_2\frac{2}{5}\approx 0.9710
$$

#### Strong
1 Yes, 2 No:
$$
H(Strong)= -\frac{1}{3}\log_2\frac{1}{3}-\frac{2}{3}\log_2\frac{2}{3}\approx 0.9183
$$

Weighted entropy:
$$
\frac{5}{8}(0.9710)+\frac{3}{8}(0.9183)\approx 0.9512
$$

Information gain:
$$
IG(S,\text{Wind})=1-0.9512=0.0488
$$

### Step 3: Choose the best feature
Since:
$$
IG(\text{Weather}) > IG(\text{Wind})
$$
ID3 chooses:
$$
\text{Weather}
$$
as the root feature.

So the first tree becomes:
```text
Weather
├── Sunny     -> No
├── Overcast  -> Yes
└── Rain      -> split again
```

### Step 4: Recurse on the `Rain` subset
Rain subset:
- Weak -> Yes
- Weak -> Yes
- Strong -> No

Now only one feature remains: `Wind`

If split on `Wind`:
- `Weak` -> all Yes
- `Strong` -> all No

So final tree:
```text
Weather
├── Sunny     -> No
├── Overcast  -> Yes
└── Rain
    ├── Weak  -> Yes
    └── Strong -> No
```

This is exactly how ID3 builds the tree recursively.

---
## ID3 Concept -> Equation -> Code
## 1. Measure impurity
Concept:
We first measure how mixed the labels are.

Equation:
$$
H(S)=-\sum_i p_i\log_2 p_i
$$

Code:
```python
def entropy(self, y):
    values, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return -np.sum(p * np.log2(p + 1e-12))
```

---
## 2. Score each feature
Concept:
Check how much uncertainty reduces after splitting on a feature.

Equation:
$$
IG(S,A)=H(S)-\sum_{v}\frac{|S_v|}{|S|}H(S_v)
$$

Code:
```python
def information_gain(self, X, y, feature):
    parent_entropy = self.entropy(y)
    values, counts = np.unique(X[feature], return_counts=True)

    child_entropy = 0
    for v, c in zip(values, counts):
        y_sub = y[X[feature] == v]
        child_entropy += (c / len(X)) * self.entropy(y_sub)

    return parent_entropy - child_entropy
```

---
## 3. Pick the best feature
Concept:
Choose the feature with maximum information gain.

Code:
```python
def best_feature(self, X, y):
    gains = {f: self.information_gain(X, y, f) for f in X.columns}
    return max(gains, key=gains.get)
```

---
## 4. Build tree recursively
Concept:
After choosing the best feature, split data and build smaller trees.

Code:
```python
def build(self, X, y):
    if len(np.unique(y)) == 1:
        return y.iloc[0]

    if X.empty:
        return y.mode()[0]

    best = self.best_feature(X, y)
    tree = {best: {}}

    for v in X[best].unique():
        mask = X[best] == v
        X_sub = X.loc[mask].drop(columns=[best])
        y_sub = y.loc[mask]
        tree[best][v] = self.build(X_sub, y_sub)

    return tree
```

Meaning:
- if node is pure -> make leaf
- if no features left -> return majority class
- else split and recurse

---
## ID3 Pseudocode
```python
ID3(D, features):
    if all labels same:
        return leaf
    if no features left:
        return majority class
    best = feature with max information gain
    create node(best)
    for each value v of best:
        recurse on subset where best = v
```

---
## ID3 Advantages
- simple and easy to understand
- tree rules are interpretable
- good for categorical data
- useful in exam explanations because steps are very clear

## ID3 Limitations
- biased toward features with many distinct values
- not naturally suited for continuous features
- can overfit
- sensitive to noise

---
# Part B: C4.5
## What C4.5 Does
C4.5 is an improved version of ID3.
It fixes major weaknesses of ID3.

Main improvements:
1. uses **Gain Ratio** instead of pure Information Gain
2. handles **continuous features**
3. can handle missing values better
4. usually produces more practical trees

So:
- ID3 = older, simpler
- C4.5 = smarter extension of ID3

---
## Why ID3 Needs Improvement
Information Gain can favor a feature with many distinct values.

Example:
- a feature like `StudentID` may split every row separately
- this gives very high information gain
- but it does not generalize

So C4.5 divides Information Gain by a quantity called **Split Information**.

---
## C4.5 Core Equations
### Entropy
Same as ID3:
$$
H(S)=-\sum_i p_i\log_2 p_i
$$

### Information Gain
Same as ID3:
$$
IG(S,A)=H(S)-\sum_v\frac{|S_v|}{|S|}H(S_v)
$$

### Split Information
$$
SI(S,A)=-\sum_v\frac{|S_v|}{|S|}\log_2\frac{|S_v|}{|S|}
$$

### Gain Ratio
$$
GR(S,A)=\frac{IG(S,A)}{SI(S,A)}
$$

C4.5 chooses the feature with the **highest gain ratio**.

---
## Short and Clean C4.5 Code
```python
import pandas as pd
import numpy as np

class C45:
    def __init__(self):
        self.tree = None
        self.default_class = None

    def entropy(self, y):
        values, counts = np.unique(y, return_counts=True)
        p = counts / counts.sum()
        return -np.sum(p * np.log2(p + 1e-12))

    def information_gain(self, X, y, feature):
        parent_entropy = self.entropy(y)
        values, counts = np.unique(X[feature], return_counts=True)

        child_entropy = 0
        for v, c in zip(values, counts):
            y_sub = y[X[feature] == v]
            child_entropy += (c / len(X)) * self.entropy(y_sub)

        return parent_entropy - child_entropy

    def split_info(self, X, feature):
        values, counts = np.unique(X[feature], return_counts=True)
        p = counts / counts.sum()
        return -np.sum(p * np.log2(p + 1e-12))

    def gain_ratio(self, X, y, feature):
        ig = self.information_gain(X, y, feature)
        si = self.split_info(X, feature)
        return 0 if si == 0 else ig / si

    def best_feature(self, X, y):
        ratios = {f: self.gain_ratio(X, y, f) for f in X.columns}
        return max(ratios, key=ratios.get)

    def build(self, X, y):
        if len(np.unique(y)) == 1:
            return y.iloc[0]

        if X.empty:
            return y.mode()[0]

        best = self.best_feature(X, y)
        tree = {best: {}}

        for v in X[best].unique():
            mask = X[best] == v
            X_sub = X.loc[mask].drop(columns=[best])
            y_sub = y.loc[mask]

            if len(X_sub) == 0:
                tree[best][v] = y.mode()[0]
            else:
                tree[best][v] = self.build(X_sub, y_sub)

        return tree

    def fit(self, X, y):
        self.default_class = y.mode()[0]
        self.tree = self.build(X, y)
        return self

    def _predict_one(self, x, tree):
        if not isinstance(tree, dict):
            return tree

        feature = next(iter(tree))
        value = x.get(feature)

        if value not in tree[feature]:
            return self.default_class

        return self._predict_one(x, tree[feature][value])

    def predict(self, X):
        return X.apply(lambda row: self._predict_one(row, self.tree), axis=1)
```

---
## Small Example for C4.5
We use a dataset with a high-cardinality feature to see why Gain Ratio helps:
```python
data = pd.DataFrame({
    "ID":      ["S1", "S2", "S3", "S4", "S5", "S6"],
    "Weather": ["Sunny", "Sunny", "Rain", "Rain", "Overcast", "Overcast"],
    "Play":    ["No", "No", "Yes", "Yes", "Yes", "Yes"]
})

X = data[["ID", "Weather"]]
y = data["Play"]

model = C45().fit(X, y)
print("C4.5 Tree:", model.tree)
print("Predictions:", model.predict(X).tolist())
```

---
## Solving the C4.5 Example Step by Step
Root labels:
- `Yes` = 4
- `No` = 2

Entropy:
$$
H(S)= -\frac{4}{6}\log_2\frac{4}{6}-\frac{2}{6}\log_2\frac{2}{6}\approx 0.9183
$$

### Feature 1: `ID`
Every ID is unique.
So each split contains exactly 1 sample.
That means every child subset is pure.

Weighted child entropy:
$$
0
$$
Thus:
$$
IG(S,\text{ID})=0.9183
$$

This looks perfect for ID3.
But it is misleading because `ID` is just a unique label.

Now compute split information.
Since there are 6 equally-sized branches:
$$
SI(S,\text{ID})=-6\left(\frac{1}{6}\log_2\frac{1}{6}\right)=\log_2 6\approx 2.585
$$

So gain ratio:
$$
GR(S,\text{ID})=\frac{0.9183}{2.585}\approx 0.355
$$

### Feature 2: `Weather`
Values:
- Sunny -> [No, No]
- Rain -> [Yes, Yes]
- Overcast -> [Yes, Yes]

All child subsets are pure, so:
$$
IG(S,\text{Weather})=0.9183
$$

Split information:
There are 3 equally-sized groups of size 2:
$$
SI(S,\text{Weather})=-3\left(\frac{2}{6}\log_2\frac{2}{6}\right)=1.585
$$

Gain ratio:
$$
GR(S,\text{Weather})=\frac{0.9183}{1.585}\approx 0.579
$$

### Choose the best feature
Although both features have equal information gain, gain ratio is larger for `Weather`:
$$
GR(\text{Weather}) > GR(\text{ID})
$$

So C4.5 correctly chooses:
$$
\text{Weather}
$$
instead of `ID`.

This is the key improvement over ID3.

---
## C4.5 Concept -> Equation -> Code
## 1. Compute entropy
Concept:
Measure uncertainty at the current node.

Equation:
$$
H(S)=-\sum_i p_i\log_2 p_i
$$

Code:
```python
def entropy(self, y):
    values, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return -np.sum(p * np.log2(p + 1e-12))
```

---
## 2. Compute information gain
Concept:
Measure reduction in entropy after split.

Equation:
$$
IG(S,A)=H(S)-\sum_v\frac{|S_v|}{|S|}H(S_v)
$$

Code:
```python
def information_gain(self, X, y, feature):
    parent_entropy = self.entropy(y)
    ...
    return parent_entropy - child_entropy
```

---
## 3. Compute split information
Concept:
Measure how broadly the split divides the data.

Equation:
$$
SI(S,A)=-\sum_v\frac{|S_v|}{|S|}\log_2\frac{|S_v|}{|S|}
$$

Code:
```python
def split_info(self, X, feature):
    values, counts = np.unique(X[feature], return_counts=True)
    p = counts / counts.sum()
    return -np.sum(p * np.log2(p + 1e-12))
```

---
## 4. Compute gain ratio
Concept:
Normalize information gain so features with too many distinct values are not unfairly favored.

Equation:
$$
GR(S,A)=\frac{IG(S,A)}{SI(S,A)}
$$

Code:
```python
def gain_ratio(self, X, y, feature):
    ig = self.information_gain(X, y, feature)
    si = self.split_info(X, feature)
    return 0 if si == 0 else ig / si
```

---
## 5. Choose best feature and recurse
Concept:
Build the decision tree exactly like ID3, but use Gain Ratio instead of Information Gain.

Code:
```python
def best_feature(self, X, y):
    ratios = {f: self.gain_ratio(X, y, f) for f in X.columns}
    return max(ratios, key=ratios.get)
```

---
## Handling Continuous Features in C4.5
A major improvement of C4.5 is continuous-value handling.
For a numeric feature, C4.5 tries thresholds:
$$
A \le t \quad \text{vs} \quad A > t
$$
and chooses the threshold with the best gain ratio.

Example split:
$$
\text{Glucose} \le 120 \ ? 
$$

So unlike ID3, C4.5 can naturally work with continuous attributes by converting them into binary threshold splits.

A simplified threshold idea in code would be:
```python
threshold = 120
left = X[feature] <= threshold
right = X[feature] > threshold
```

Then entropy, IG, and GR are computed for that split.

---
# ID3 vs C4.5
## Main Difference
ID3 chooses:
$$
\max IG
$$
C4.5 chooses:
$$
\max GR
$$

## Comparison Table
| Aspect | ID3 | C4.5 |
|---|---|---|
| Split criterion | Information Gain | Gain Ratio |
| Handles categorical features | Yes | Yes |
| Handles continuous features | Poorly / not directly | Yes |
| Bias toward many-valued features | High | Reduced |
| Missing values | Weak | Better |
| Complexity | Simpler | Slightly more advanced |

---
## Full Combined Example
```python
import pandas as pd

data = pd.DataFrame({
    "Weather": ["Sunny", "Sunny", "Overcast", "Rain", "Rain", "Rain", "Overcast", "Sunny"],
    "Wind":    ["Weak", "Strong", "Weak", "Weak", "Weak", "Strong", "Strong", "Weak"],
    "Play":    ["No", "No", "Yes", "Yes", "Yes", "No", "Yes", "No"]
})

X = data[["Weather", "Wind"]]
y = data["Play"]

id3_model = ID3().fit(X, y)
c45_model = C45().fit(X, y)

print("ID3 Tree:", id3_model.tree)
print("C4.5 Tree:", c45_model.tree)
print("ID3 Predictions:", id3_model.predict(X).tolist())
print("C4.5 Predictions:", c45_model.predict(X).tolist())
```

---
## How the Tree Predicts
Suppose tree is:
```text
Weather
├── Sunny     -> No
├── Overcast  -> Yes
└── Rain
    ├── Weak   -> Yes
    └── Strong -> No
```

For a new row:
```text
Weather = Rain, Wind = Weak
```
Path:
1. check `Weather`
2. move to `Rain`
3. check `Wind`
4. `Weak` -> leaf = `Yes`

So predicted class is:
$$
\text{Yes}
$$

---
## Why These Algorithms Work
Both algorithms work by repeatedly reducing uncertainty.
They ask:
> which feature makes the class labels as pure as possible after splitting?

ID3 answers this using **information gain**.
C4.5 answers this using **gain ratio**.

The recursion stops when:
- node becomes pure
- no features remain
- or no useful split is possible

So a large classification problem becomes a sequence of small rule-based decisions.

---
## Advantages of ID3 and C4.5
- easy to interpret
- produces human-readable rules
- no heavy math during prediction
- useful for categorical classification tasks
- good for exam answers because the logic is visual and recursive

---
## Limitations
## ID3
- biased toward attributes with many values
- struggles with continuous data
- can overfit
- sensitive to noise

## C4.5
- more complex than ID3
- deeper trees can still overfit without pruning
- threshold search for continuous features adds computation

---
## Exam-Oriented Summary
## ID3 Definition
ID3 is a top-down greedy decision tree algorithm that chooses the feature with the highest information gain at each step.

## C4.5 Definition
C4.5 is an improved decision tree algorithm that extends ID3 by using gain ratio and supporting continuous features.

## ID3 Formula
Entropy:
$$
H(S)=-\sum_i p_i\log_2 p_i
$$
Information Gain:
$$
IG(S,A)=H(S)-\sum_v\frac{|S_v|}{|S|}H(S_v)
$$

## C4.5 Formula
Split Information:
$$
SI(S,A)=-\sum_v\frac{|S_v|}{|S|}\log_2\frac{|S_v|}{|S|}
$$
Gain Ratio:
$$
GR(S,A)=\frac{IG(S,A)}{SI(S,A)}
$$

## Common Steps
1. compute impurity of current dataset
2. score every feature
3. choose best feature
4. split dataset
5. repeat recursively
6. stop when node is pure or features are exhausted

---
## Very Short Revision
### ID3
- uses entropy
- computes information gain
- picks feature with max gain
- recursive tree construction

### C4.5
- starts like ID3
- adds split information
- uses gain ratio
- handles continuous features better

---
## Final Takeaway
ID3 and C4.5 both build decision trees by recursively selecting the best splitting feature.
ID3 uses **Information Gain**, while C4.5 improves it using **Gain Ratio** to avoid unfair preference for features with many distinct values.
So:
- use **ID3** to understand the core decision tree idea
- use **C4.5** as the more practical and improved version
