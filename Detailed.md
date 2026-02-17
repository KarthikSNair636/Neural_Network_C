<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Neural Network – Mathematical Explanation</title>
<style>
    body {
        font-family: "Segoe UI", Arial, sans-serif;
        margin: 40px;
        line-height: 1.6;
        background-color: #010101ff;
        color: #222;
    }
    h1, h2, h3 {
        color: #1a237e;
    }
    h1 {
        text-align: center;
        margin-bottom: 5px;
    }
    hr {
        margin: 30px 0;
    }
    pre {
        background-color: #000000ff;
        padding: 15px;
        border-radius: 8px;
        overflow-x: auto;
    }
    .center {
        text-align: center;
        font-weight: bold;
    }
    .box {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin-bottom: 25px;
    }
</style>
</head>

<body>

<h1>🧠 Neural Network – Mathematical Explanation</h1>
<hr>

<div class="box">
<h2>1️⃣ Definition</h2>

A neural network is a parameterized function:

<p class="center">f(x; θ)</p>

that maps an input vector

<p class="center">x ∈ ℝⁿ</p>

to an output vector

<p class="center">y ∈ ℝᵐ</p>

where the parameters are

<p class="center">
θ = {W<sup>(1)</sup>, b<sup>(1)</sup>, ..., W<sup>(L)</sup>, b<sup>(L)</sup>}
</p>

Training solves the optimization problem:

<p class="center">θ* = argmin<sub>θ</sub> L(y, ŷ)</p>
</div>

<hr>

<div class="box">
<h2>2️⃣ Layer-wise Computation</h2>

For a network with L layers:

<ul>
<li><b>W<sup>(l)</sup></b> = weight matrix of layer l</li>
<li><b>b<sup>(l)</sup></b> = bias vector of layer l</li>
<li><b>φ</b> = activation function</li>
</ul>

For each layer l:

<p class="center">
z<sup>(l)</sup> = W<sup>(l)</sup> x<sup>(l−1)</sup> + b<sup>(l)</sup>
</p>

<p class="center">
x<sup>(l)</sup> = φ(z<sup>(l)</sup>)
</p>

Where:
<ul>
<li>z<sup>(l)</sup> = weighted sum (pre-activation)</li>
<li>x<sup>(l)</sup> = activation/output of layer l</li>
</ul>

Input layer:

<p class="center">x<sup>(0)</sup> = input vector</p>

Final prediction:

<p class="center">ŷ = x<sup>(L)</sup></p>
</div>

<hr>

<div class="box">
<h2>3️⃣ Activation Functions</h2>

<h3>Sigmoid</h3>
<p class="center">
σ(z) = 1 / (1 + e<sup>−z</sup>)
</p>
<p>Range: (0,1) — commonly used for probabilities.</p>

<h3>ReLU</h3>
<p class="center">
ReLU(z) = max(0, z)
</p>
<p>Introduces non-linearity and speeds up deep network training.</p>
</div>

<hr>

<div class="box">
<h2>4️⃣ Loss Function (Mean Squared Error)</h2>

For one training sample:

<p class="center">
L = (1/2)(y − ŷ)²
</p>

The factor 1/2 simplifies derivative calculations during backpropagation.
</div>

<hr>

<div class="box">
<h2>5️⃣ Numerical Example</h2>

<h3>Network Architecture</h3>
<ul>
<li>2 Input neurons</li>
<li>1 Hidden neuron</li>
<li>1 Output neuron</li>
<li>Sigmoid activation</li>
</ul>

<h3>Step 1: Input</h3>
<pre>
x^(0) = [1, 2]
</pre>

<h3>Step 2: Hidden Layer</h3>

<pre>
W^(1) = [0.5   -1.0]
b^(1) = 0
</pre>

<pre>
z^(1) = (0.5 × 1) + (-1.0 × 2)
       = -1.5
</pre>

<pre>
x^(1) = σ(-1.5)
       = 1 / (1 + e^(1.5))
       ≈ 0.182
</pre>

<h3>Step 3: Output Layer</h3>

<pre>
W^(2) = [2.0]
b^(2) = 0
</pre>

<pre>
z^(2) = 2.0 × 0.182
       = 0.364
</pre>

<pre>
ŷ = σ(0.364)
   ≈ 0.590
</pre>

<h3>Step 4: Loss Computation</h3>

<pre>
y = 1
L = (1/2)(1 − 0.590)²
  ≈ 0.084
</pre>
</div>

<hr>

<div class="box">
<h2>6️⃣ General Neuron Formula (Index Notation)</h2>

<p class="center">
z<sub>j</sub><sup>(l)</sup> = Σ<sub>i</sub> w<sub>ji</sub><sup>(l)</sup> x<sub>i</sub><sup>(l−1)</sup> + b<sub>j</sub><sup>(l)</sup>
</p>

<p class="center">
x<sub>j</sub><sup>(l)</sup> = φ(z<sub>j</sub><sup>(l)</sup>)
</p>
</div>

<hr>

<div class="box">
<h2>7️⃣ Functional Composition View</h2>

A neural network is a composition of affine and nonlinear functions:

<p class="center">
f(x) = φ<sub>L</sub>(W<sup>(L)</sup> φ<sub>L−1</sub>( ... φ<sub>1</sub>(W<sup>(1)</sup>x + b<sup>(1)</sup>) ... ))
</p>
</div>

<hr>

<div class="box">
<h2>🔎 Key Insights</h2>

<ul>
<li>Each layer performs a linear transformation</li>
<li>Then applies a nonlinear activation</li>
<li>The network is a composition of affine + nonlinear mappings</li>
<li>Training adjusts parameters to minimize loss</li>
</ul>
</div>

<hr>

<p style="text-align:center;">
<b>Author:</b> Karthik S Nair<br>
CSE, CET Trivandrum
</p>

</body>
</html>
