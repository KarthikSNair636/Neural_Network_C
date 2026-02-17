# 🧠 Neural Network

------------------------------------------------------------------------

## 📌 1. What is a Neural Network?

A Neural Network is a system that learns patterns by adjusting numbers
called **weights**.

Think of it as:

Input → Hidden Thinking → Final Decision

------------------------------------------------------------------------

## 🖼 Structure of a Neural Network

![Neural Network Structure](Md_IMG/n1.png)
image source:Geeks for Geeks

-   Input Layer → receives data\
-   Hidden Layer → processes data\
-   Output Layer → gives prediction

------------------------------------------------------------------------

## 📌 2. Analogy (Real Life)

Imagine preparing tea ☕

-   Ingredients → Inputs\
-   Quantity adjustments → Weights\
-   Extra sugar → Bias\
-   Taste test → Activation\
-   Final flavor → Output

If taste is bad → adjust ingredients → try again

That adjustment process = learning.

------------------------------------------------------------------------

## 📌 3. Forward Propagation (Python Formulas)

### Step 1: Hidden Layer

``` python
z1 = W1 @ x + b1
a1 = relu(z1)
```

### Step 2: Output Layer

``` python
z2 = W2 @ a1 + b2
y_hat = sigmoid(z2)
```

------------------------------------------------------------------------

## 🖼 Forward Flow

![Forward Flow](Md_IMG/w2.png)

------------------------------------------------------------------------

## 📌 4. Activation Functions (Python)

``` python
def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

------------------------------------------------------------------------

## 📌 5. Loss Function

``` python
loss = 0.5 * (y - y_hat)**2
```

------------------------------------------------------------------------

## 📌 6. Backpropagation (Core Python Formulas)

``` python
# Output layer gradient
dL_dyhat = -(y - y_hat)
dyhat_dz2 = y_hat * (1 - y_hat)

delta2 = dL_dyhat * dyhat_dz2
dW2 = delta2 @ a1.T
db2 = delta2

# Hidden layer gradient
delta1 = (W2.T @ delta2) * (a1 > 0)
dW1 = delta1 @ x.T
db1 = delta1
```

------------------------------------------------------------------------

## 📌 7. Weight Update Rule

``` python
W1 -= lr * dW1
b1 -= lr * db1
W2 -= lr * dW2
b2 -= lr * db2
```

------------------------------------------------------------------------

## 🎯 Final Idea

Neural Network =

Linear math,then\
+ Activation,then\
+ Error calculation,then\
+ Small improvements repeated many times,until

Eventually good predictions or outcomes occur then we say that the model is trained 🎯

------------------------------------------------------------------------

**Author: Karthik S Nair**