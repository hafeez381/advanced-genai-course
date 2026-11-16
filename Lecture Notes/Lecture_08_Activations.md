---
type: lecture
course: '[[_CS5326 Advanced GenAI|CS5326 Advanced GenAI]]'
source:
- '[[005-Activations.pdf]]'
- '[[005-ReLU-Boundary-Explorer.html]]'
tags:
- course/cs5326
---

%% nav:start %%
[[_CS5326 Advanced GenAI|CS5326 Advanced GenAI]] · ← [[Lecture_07_Residuals_and_Normalization]] · [[Lecture_09_FFNs_Dense_and_MoE]] →
Concepts: [[Activation Function]], [[ReLU]], [[GELU]], [[SwiGLU]], [[Gated Linear Unit]]
%% nav:end %%

# Activation Functions: From Sigmoid to SwiGLU (CS5326, Lecture 8)

*Follows [[Lecture_07_Residuals_and_Normalization|Lecture 7]], which looked at the residual stream and LayerNorm/RMSNorm around each Transformer sublayer. This lecture opens the remaining box inside the FFN (feed-forward network) sublayer: the activation function. It asks why any nonlinearity is needed, what a good one must provide, and how the field moved from sigmoid to ReLU, to smooth self-gated units (GELU, SiLU), and finally to the separately gated FFNs (SwiGLU) used in modern LLMs.*

## Topic Inventory

- 1. The Case for Non-Linearity
  - 1.1 What a neuron computes
  - 1.2 Affine layers collapse
  - 1.3 What the collapse costs
  - 1.4 A concrete failure: XOR
  - 1.5 How a nonlinear hidden layer solves XOR
  - 1.6 Why not put nonlinearity only at the output?
- 2. What a Useful Activation Needs
  - 2.1 Requirements and preferences
  - 2.2 What an activation does not need
  - 2.3 The activation derivative scales the learning signal
  - 2.4 Vanishing, exploding and blocked gradients
- 3. Bounded Activations: Sigmoid and Tanh
  - 3.1 Sigmoid and its derivative
  - 3.2 Odds, log-odds, and why sigmoid converts one to the other
  - 3.3 Sigmoid's main limitation: saturation
  - 3.4 Why sigmoid still works well at the output
  - 3.5 Tanh: bounded but signed
- 4. ReLU
  - 4.1 Definition and derivative
  - 4.2 Why discard negative values?
  - 4.3 Recovering the negative side with two ReLUs
  - 4.4 How a piecewise-linear function makes a nonlinear network
  - 4.5 Blocked learning and dead ReLUs
  - 4.6 Dead or deadish, and the controls
- 5. Fixing the Negative Side: Leaky ReLU, PReLU, ELU, SELU, Softplus
  - 5.1 Leaky ReLU
  - 5.2 PReLU and the $\alpha = 1$ problem
  - 5.3 ELU
  - 5.4 SELU and self-normalization
  - 5.5 Why SELU cannot replace Transformer normalization
  - 5.6 Softplus
- 6. Self-Regulating Activations: GELU, SiLU, Swish
  - 6.1 GELU: ReLU's gate made gradual
  - 6.2 The probabilistic interpretation
  - 6.3 GELU's derivative, non-monotonicity and limitations
  - 6.4 SiLU and self-gating
  - 6.5 Swish
  - 6.6 Smooth is not monotonic: the negative dip
  - 6.7 Why gradients can still vanish
- 7. Separately Regulated Activations: GLU and Gated FFNs
  - 7.1 From one neuron to a gated pair
  - 7.2 Why separate content from control
  - 7.3 Extending to a whole layer
  - 7.4 Why sigmoid returns: gradients through the two branches
  - 7.5 Conventional vs gated FFN
  - 7.6 ReGLU, GEGLU and SwiGLU
  - 7.7 The SwiGLU parameter budget
- 8. Choosing an Activation
  - 8.1 Hidden vs output activations
  - 8.2 Practical starting points
  - 8.3 Summary of activation functions
- 9. Numerical Problems
- 10. Formula Summary Sheet
- 11. Cheat Sheet & Recall Aids

---

## 1. The Case for Non-Linearity

### 1.1 What a neuron computes

A neuron performs two operations: a weighted sum, then an activation.

$$
z = \sum_{i=1}^{d} w_i x_i + b, \qquad a = \phi(z)
$$

| Symbol | Meaning |
|---|---|
| $x_i$ | input feature $i$ (of $d$ inputs) |
| $w_i$ | its learned weight |
| $b$ | a learned bias |
| $z$ | the **preactivation**, or learned score |
| $\phi$ | the **activation function** |
| $a$ | the neuron's output (its **activation**) |

The weighted sum detects a learned combination of input features; the activation decides how the neuron responds to that combination. For example, a neuron could respond strongly only when two features jointly provide enough evidence for a pattern. In short: **the weights decide what the neuron looks for; the activation decides how it responds.**

### 1.2 Affine layers collapse

Suppose we stack two layers with no activation between them, where $x$ is a row vector and $W_1, W_2$ are weight matrices:

$$
h = xW_1 + b_1, \qquad y = hW_2 + b_2 .
$$

Substituting the first into the second and distributing,

$$
y = (xW_1 + b_1)W_2 + b_2 = x(W_1W_2) + (b_1W_2 + b_2).
$$

Defining $W_{\text{effective}} = W_1W_2$ and $b_{\text{effective}} = b_1W_2 + b_2$ gives

$$
\boxed{y = xW_{\text{effective}} + b_{\text{effective}}}
$$

The two layers collapse into one **affine** transformation (a linear map plus a bias; neural-network discussions often loosely call both "linear"). The same substitution applies to any number of affine layers, by induction: collapse the first two, then collapse the result with the third, and so on.

### 1.3 What the collapse costs

Because the whole stack equals one affine map, more layers cannot create a nonlinear relationship between input and output.

- Increasing hidden width does not remove this restriction either.
- Depth can change the *optimization* and *parameterization* of a linear model (how training moves through weight space), but not what the model can represent. A hundred affine layers can rearrange a linear transformation; they cannot turn it into a nonlinear one.

Adding nonlinear activations (such as sigmoid or ReLU) changes the picture. Compare three cases:

| Network | Decision boundary |
|---|---|
| One affine neuron + sigmoid | The probability output is nonlinear in $x$, but thresholding it gives a **linear** boundary (§1.6) |
| Several affine layers, no activations | Collapse to one affine map, so the boundary stays **linear** |
| A hidden layer with nonlinear activations + a layer combining its outputs | Builds nonlinear features, so the boundary can be **nonlinear** |

For example, different hidden ReLU neurons can introduce bends at different locations, and the next layer combines those responses into a more complex boundary. The key is the combination: layers compose transformations, and nonlinear activations stop that composition from collapsing into one affine map. One sufficiently wide nonlinear hidden layer can already create complex boundaries *(expanding on this: this is the universal approximation theorem: one hidden layer with enough units can approximate any continuous function on a bounded domain)*; extra depth makes some patterns more efficient to represent, needing far fewer units.

### 1.4 A concrete failure: XOR

XOR outputs 1 when its two binary inputs differ:

| $x_1$ | $x_2$ | Required output |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

![[L08_xor_input_space.png|380]]

*XOR in input space. The positive examples (01, 10, blue squares) sit on opposite corners of the square, the negatives (00, 11, orange circles) on the other two. The dashed line shows one attempt; any single line leaves at least one point on the wrong side.*

No single straight line separates the positives from both negatives. The limitation is **representational**: no amount of training can make an affine model (or an affine model with a thresholded output) represent this boundary on the raw inputs. *(Why, in one line: a linear separator would need $b < 0$ for 00, $w_1 + b > 0$ and $w_2 + b > 0$ for 01/10, and $w_1 + w_2 + b < 0$ for 11; adding the middle two gives $w_1 + w_2 + 2b > 0$, so $w_1 + w_2 + b > -b > 0$, a contradiction.)*

### 1.5 How a nonlinear hidden layer solves XOR

Define ReLU (covered properly in §4) simply as $\text{ReLU}(z) = \max(0, z)$. Build two hidden neurons and a linear output:

$$
h_1 = \text{ReLU}(x_1 + x_2), \qquad h_2 = \text{ReLU}(x_1 + x_2 - 1), \qquad y = h_1 - 2h_2 .
$$

| Inputs | $h_1$ | $h_2$ | $y = h_1 - 2h_2$ |
|---|---|---|---|
| (0, 0) | 0 | 0 | 0 |
| (0, 1) | 1 | 0 | 1 |
| (1, 0) | 1 | 0 | 1 |
| (1, 1) | 2 | 1 | 0 |

![[L08_xor_hidden_space.png|380]]

*The same four points after the hidden layer. 01 and 10 both map to $(h_1, h_2) = (1, 0)$; 00 maps to $(0,0)$ and 11 to $(2,1)$. Now one straight line separates the classes.*

The output layer is still linear, but it receives nonlinear features of the original inputs. The hidden layer transforms the representation so that a simple output rule becomes sufficient. $h_2$ only switches on when *both* inputs are 1; it acts as an "AND" detector that the output subtracts twice.

### 1.6 Why not put nonlinearity only at the output?

Suppose all hidden layers collapse into one scalar affine score $z = xw + b$, and we apply a sigmoid, $p = \sigma(z)$. Although $p$ is nonlinear in $x$, sigmoid is strictly increasing with $\sigma(0) = 0.5$, so classifying at $p > 0.5$ is equivalent to

$$
z > 0 .
$$

The classification boundary is still the hyperplane $xw + b = 0$. A nonlinear conversion of the final score does not necessarily produce a nonlinear decision boundary; the network needs **nonlinear feature construction inside it**.

More generally, nonlinearity can also come from other operations: products, attention, or normalization, not only named scalar activation functions. The collapse argument assumes the network is made entirely of affine operations.

---

## 2. What a Useful Activation Needs

Section 1 showed that *some* nonlinearity is mandatory. Almost any nonlinear function would do for expressiveness, so the real question is which ones also make the network trainable.

### 2.1 Requirements and preferences

Separate the mathematical purpose from practical preferences:

| Property | Why it matters | Is it mandatory? |
|---|---|---|
| Nonlinearity | Prevents an otherwise affine stack from collapsing | Necessary somewhere for nonlinear modeling |
| Usable derivatives | Enables ordinary gradient-based learning | Practically important; isolated nondifferentiable points are acceptable |
| Useful gradient magnitudes | Helps learning signals reach earlier layers | Desirable, not guaranteed everywhere |
| Manageable activation magnitudes | Reduces numerical and optimization problems | Desirable |
| Efficient computation | Matters when applied billions of times | Desirable |
| Appropriate output range | Matches the variable being predicted or controlled | Essential for some output/gating roles |
| Sufficient expressiveness | Allows useful learned representations | Must be assessed with the whole architecture |

### 2.2 What an activation does not need

An activation does **not** have to be:

| Not required | Counterexample in common use |
|---|---|
| Bounded | ReLU is unbounded above |
| Monotonic | GELU and SiLU are not monotonic everywhere (§6.6) |
| Smooth everywhere | ReLU has a corner at 0 |
| Nonnegative | tanh and GELU can be negative |
| Zero-centered | sigmoid and ReLU are not symmetric around zero |

A hard step function, $\mathbb{1}[z > 0]$, is nonlinear, but its derivative is zero almost everywhere (and undefined at 0). Gradients through it are always zero, so it is unsuitable for straightforward backpropagation without special training methods *(e.g. straight-through estimators, which pretend the derivative is 1 in the backward pass)*. **Being nonlinear is enough to escape affine collapse; it is not enough to make a network easy to train.**

### 2.3 The activation derivative scales the learning signal

For one neuron with $z = \sum_j w_j x_j + b$, $a = \phi(z)$ and loss $L$, changing $w_i$ changes $z$, which changes $a$, which can change the loss:

$$
w_i \longrightarrow z \longrightarrow a \longrightarrow L .
$$

The chain rule follows this path backward:

$$
\frac{\partial L}{\partial w_i} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w_i},
$$

read as: how much the loss changes with the activation, times how much the activation changes with its input, times how much that input changes with the weight. The two local derivatives are $\partial a / \partial z = \phi'(z)$ (from $a = \phi(z)$) and $\partial z / \partial w_i = x_i$ (from the weighted sum). Therefore

$$
\boxed{\frac{\partial L}{\partial w_i} = \underbrace{\frac{\partial L}{\partial a}}_{\text{incoming error signal}} \cdot \underbrace{\phi'(z)}_{\text{activation sensitivity}} \cdot \underbrace{x_i}_{\text{input to this weight}}}
$$

The activation's derivative scales the learning signal that reaches the weight. If $\phi'(z) = 0$, this neuron contributes no gradient to its incoming weights for this example.

### 2.4 Vanishing, exploding and blocked gradients

Across many layers, backpropagation repeatedly multiplies by activation derivatives and weight matrices:

- Repeatedly **small** factors can produce **vanishing gradients** (early layers barely learn).
- Repeatedly **large** factors can produce **exploding gradients** (updates blow up).
- A **zero** factor can block a particular path entirely.

The full behaviour depends on the weights, architecture, initialization and normalization, not only the activation. A useful summary: **the forward function decides what information passes; its derivative decides how readily that decision can change during learning.**

---

## 3. Bounded Activations: Sigmoid and Tanh

With the two criteria from §2 in hand (nonlinearity and a usable derivative), the historical starting point is the sigmoid.

### 3.1 Sigmoid and its derivative

![[L08_sigmoid.png|380]]

$$
\sigma(z) = g(z) = \frac{1}{1 + e^{-z}}
$$

- **Range:** $(0, 1)$, with mean output around 0.5.
- **Gradient → 0** for large $|z|$ (both tails).

Differentiating with the chain rule, $\frac{d}{dz}(1 + e^{-z})^{-1} = \frac{e^{-z}}{(1+e^{-z})^2} = \frac{1}{1+e^{-z}} \cdot \frac{e^{-z}}{1+e^{-z}}$, and since $\frac{e^{-z}}{1+e^{-z}} = 1 - \frac{1}{1+e^{-z}}$,

$$
\sigma'(z) = \frac{1}{1+e^{-z}}\left(1 - \frac{1}{1+e^{-z}}\right) = \boxed{\sigma(z)\big(1 - \sigma(z)\big)} .
$$

So if $a = \sigma(z)$, then $\sigma'(z) = a(1 - a)$: the derivative is computable from the forward output alone.

| $z$ | $\sigma(z)$ | $\sigma'(z)$ |
|---|---|---|
| $-10$ | $\approx 0$ | $\approx 0$ |
| $-5$ | 0.007 | 0.0066 |
| $0$ | 0.5 | $\tfrac12(1 - \tfrac12) = \mathbf{0.25}$ |
| $5$ | 0.993 | 0.0066 |
| $10$ | $\approx 1$ | $\approx 0$ |

**Motivation:** replace an abrupt binary threshold with a smooth transition. Sigmoid turns an unrestricted score into a bounded response, and it has a useful probability interpretation: if $p = \sigma(z)$, then $z = \log \frac{p}{1-p}$. **Sigmoid converts log-odds into a probability.**

### 3.2 Odds, log-odds, and why sigmoid converts one to the other

A probability compares an outcome with all possibilities. **Odds** compare it with the alternative:

$$
\text{odds} = \frac{p}{1-p} .
$$

If the probability of rain is $p = 0.8$, the probability of no rain is 0.2, so the odds of rain are $0.8/0.2 = 4$, or "4 to 1 in favour of rain". **Log-odds** (also called the **logit**) take the natural log of that ratio, $\ln \frac{p}{1-p}$:

| Probability $p$ | Odds $\frac{p}{1-p}$ | Log-odds |
|---|---|---|
| 0.2 | 1/4 | $-1.386$ |
| 0.5 | 1 | 0 |
| 0.8 | 4 | $+1.386$ |

Negative log-odds favour the negative class, zero means equal chances, and positive log-odds favour the positive class. Log-odds range over all of $\mathbb{R}$, which is exactly the range of a neuron's score.

To see that sigmoid is the conversion, interpret the score as log-odds, $z = \ln\frac{p}{1-p}$, and undo the logarithm: $e^z = \frac{p}{1-p}$. Multiplying out, $e^z(1-p) = p$, so $e^z = p(1 + e^z)$, and isolating $p$,

$$
p = \frac{e^z}{1 + e^z} = \frac{1}{1 + e^{-z}} = \sigma(z).
$$

The neuron can produce any real score, and sigmoid turns it into a value in $(0,1)$ used as the predicted probability. Whether that prediction matches real-world frequencies depends on how well the model is trained and **calibrated** (a model is calibrated if, among the cases where it predicts 0.8, about 80% are actually positive).

### 3.3 Sigmoid's main limitation: saturation

**Strengths:** smooth, bounded, differentiable, and suitable for binary probabilities or gates. **Main limitation: saturation**, meaning the output flattens at both ends, so the derivative $\sigma'(z) = \sigma(z)(1-\sigma(z))$ is at most 0.25 (at $z = 0$) and approaches zero at either extreme.

Ignoring weights just to illustrate the multiplication across ten sigmoid layers, even the *best case* gives

$$
0.25^{10} \approx 9.54 \times 10^{-7} .
$$

Sigmoid's outputs are also always positive, which makes hidden representations less conveniently centered *(expanding on this: if every input $x_i$ to the next neuron is positive, then all its weight gradients $\frac{\partial L}{\partial a}\phi'(z)x_i$ share the sign of $\frac{\partial L}{\partial a}$, so the weights can only all increase or all decrease together in one step, which produces zig-zag updates)*. Saturation and initialization were central issues in early analyses of deep-network training (Glorot & Bengio, 2010).

**When useful:** binary-classification outputs and bounded gates. It is a less attractive default for many stacked hidden layers.

### 3.4 Why sigmoid still works well at the output

For binary classification with $p = \sigma(z)$ and label $y \in \{0, 1\}$, the **binary cross-entropy (BCE)** loss is

$$
L = -y \log p - (1-y)\log(1-p).
$$

Differentiating with respect to $p$ and combining over a common denominator,

$$
\frac{\partial L}{\partial p} = -\frac{y}{p} + \frac{1-y}{1-p} = \frac{p - y}{p(1-p)} .
$$

Since $\partial p / \partial z = p(1-p)$, the chain rule gives

$$
\frac{\partial L}{\partial z} = \frac{p-y}{p(1-p)} \cdot p(1-p) = \boxed{p - y} .
$$

The sigmoid's small derivative cancels exactly against the loss's $1/(p(1-p))$ term. For a confidently wrong prediction, $p = 0.001$ when $y = 1$, the logit gets $\partial L / \partial z = -0.999$: a strong corrective gradient, even though sigmoid is deeply saturated there. **An activation must be judged together with its loss**: sigmoid plus cross-entropy behaves very differently from sigmoid buried inside many hidden layers.

### 3.5 Tanh: bounded but signed

![[L08_tanh.png|380]]

$$
\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} = 2\sigma(2z) - 1
$$

The identity shows tanh is a sigmoid that has been scaled horizontally (by 2), stretched vertically (by 2) and shifted down (by 1).

> [!warning] Possible source inconsistency
> The Andrew Ng slide calls tanh a "shifted version of the sigmoid". It is shifted *and* rescaled, per the identity on the next slide, $\tanh z = 2\sigma(2z) - 1$.

- **Range:** $(-1, 1)$, with $\tanh(0) = 0$.
- **Derivative:** $\tanh'(z) = 1 - \tanh^2(z)$, so if $a = \tanh(z)$, then $g'(z) = 1 - a^2$.
- Examples: $z = 10$ gives $g \approx 1, g' \approx 0$; $z = -10$ gives $g \approx -1, g' \approx 0$; $z = 0$ gives $g = 0, g' = 1$.

**Motivation:** represent positive and negative evidence symmetrically. For symmetrically distributed inputs it gives zero-mean outputs; it does not force zero mean for arbitrary inputs. The derivative at zero is 1 (four times sigmoid's peak), but it approaches zero at large positive or negative inputs.

**Strengths:** smooth, signed, bounded, and sensitive near zero. **Limitation:** it still saturates in both tails; centering improves one aspect of the representation but does not remove vanishing-gradient risks. **When useful:** bounded signed outputs and architectures that specifically use tanh, including many recurrent-state constructions (e.g. the LSTM cell). Tanh improves the symmetry of the response; it does not remove the cost of squeezing large magnitudes into a bounded range.

---

## 4. ReLU

Both bounded activations die in their tails. The fix that made deep networks practical was to stop bounding the positive side at all.

### 4.1 Definition and derivative

![[L08_relu.png|380]]

The **Rectified Linear Unit (ReLU)** is

$$
\text{ReLU}(z) = \max(0, z), \qquad
\text{ReLU}'(z) = \begin{cases} 0 & z < 0 \\ 1 & z > 0 \\ \text{undefined} & z = 0 \end{cases}
$$

- **Range:** $[0, \infty)$.
- Removes the diminished gradients for large positive $z$: the derivative is exactly 1 however large $z$ gets.
- Gradient is 0 for $z < 0$ and undefined at $z = 0$; in practice a convention is chosen for $z = 0$.

> [!warning] Possible source inconsistency
> The Ng slide writes ReLU's range as $(0, \infty)$; since $\text{ReLU}(z) = 0$ for all $z \le 0$, it is $[0, \infty)$. It also uses the convention $g'(0) \approx 1$ (grouping $z \ge 0$), while a later slide says implementations commonly use $g'(0) = 0$ (PyTorch does). Either convention is fine: $z$ is exactly 0 with negligible probability.

**Motivation:** keep nonlinearity using an extremely cheap operation, while letting positive signals grow without saturating.

### 4.2 Why discard negative values?

Remember that $z = xw + b$ is a *learned score*. ReLU says: "activate this feature when its evidence crosses a threshold." The activation condition is

$$
xw > -b ,
$$

so the weights decide which direction counts as evidence, and the bias sets the threshold. A negative *input* is not necessarily discarded: with $x = -5$, $w = -1$, $b = 0$, the score is $z = 5$ and passes. Nor does a nonnegative activation prevent negative contributions later, because the following layer can use negative weights.

**Strengths:** cheap, selective, exactly zero for inactive features (**sparse** activations), and derivative 1 on the positive side; these properties supported successful training of deep rectifier networks (Glorot, Bordes & Bengio, 2011). **When useful:** a simple and efficient hidden-layer baseline. ReLU's negative cutoff defines a learned feature detector; it does not declare negative information worthless.

### 4.3 Recovering the negative side with two ReLUs

One ReLU does lose information about negative inputs: $z = -2$ and $z = -7$ both give 0. But two neurons can capture complementary information, $\text{ReLU}(z)$ for the positive part and $\text{ReLU}(-z)$ for the magnitude of the negative part:

| Input $z$ | $\text{ReLU}(z)$ | $\text{ReLU}(-z)$ | First minus second |
|---|---|---|---|
| $+3$ | 3 | 0 | $+3$ |
| $-3$ | 0 | 3 | $-3$ |
| 0 | 0 | 0 | 0 |

$$
\boxed{z = \text{ReLU}(z) - \text{ReLU}(-z)}
$$

A network can implement this when $z = \mathbf{w}^\top\mathbf{x} + b$ by giving two hidden neurons opposite weights and biases, $h_1 = \text{ReLU}(\mathbf{w}^\top\mathbf{x} + b)$ and $h_2 = \text{ReLU}(-\mathbf{w}^\top\mathbf{x} - b)$, then a linear neuron with weights $+1, -1$ and zero bias: $y = h_1 - h_2 = z$. The sign is preserved by *which* neuron activates, the magnitude by *how strongly*. This shows a ReLU network *can* preserve negative information; it does not mean training automatically learns such pairs.

### 4.4 How a piecewise-linear function makes a nonlinear network

ReLU is made of straight segments, but it is not one global straight line. A weighted ReLU $a = \max(0, wx + b)$ changes behaviour at the threshold $wx + b = 0$. Combining several such units creates multiple regions with different slopes; in higher dimensions, each combination of active neurons (an **activation pattern**) applies a different affine transformation in its own region of input space.

**Within a fixed activation pattern, a ReLU network is affine; across patterns, it is nonlinear.** Nonlinearity does not require a smoothly curved graph: changing *which* linear rule applies is already nonlinear. Greater depth composes these changes to build increasingly complex representations (the number of regions can grow exponentially with depth). The companion page [[005-ReLU-Boundary-Explorer.html|ReLU Boundary Explorer]] lets you place ReLU units and watch the boundary bend.

### 4.5 Blocked learning and dead ReLUs

ReLU's limitation: when a neuron outputs zero, it also blocks the gradient that could adjust its incoming weights. Whether this is **temporary** or **persistent** is the key distinction.

For one neuron with $z = \sum_i w_i x_i + b$ and $a = \max(0, z)$, the chain rule from §2.3 is

$$
\frac{\partial L}{\partial w_i} = \underbrace{\frac{\partial L}{\partial a}}_{\text{output} \to \text{loss}} \; \underbrace{\frac{\partial a}{\partial z}}_{\text{preact} \to \text{output}} \; \underbrace{\frac{\partial z}{\partial w_i}}_{\text{weight} \to \text{preact}} .
$$

For $z < 0$, small changes to $z$ leave the output at zero (moving from $z = -3$ to $z = -2.9$ still gives $a = 0$), so $\partial a / \partial z = 0$ and

$$
\frac{\partial L}{\partial w_i} = \frac{\partial L}{\partial a} \times 0 \times x_i = 0, \qquad \frac{\partial L}{\partial b} = \frac{\partial L}{\partial a} \times 0 \times 1 = 0 .
$$

Even if changing this neuron's output would reduce the loss, the zero derivative stops that signal from reaching its weights and bias.

**Dead ReLU.** Suppose all relevant inputs satisfy $0 \le x \le 1$, but the neuron has $z = x - 2$. Then every input gives $-2 \le z \le -1$: the neuron outputs zero on the entire dataset, every example gives zero gradient, and gradient descent with learning rate $\eta$ leaves the parameters unchanged:

$$
w_{\text{new}} = w - \eta \cdot 0 = w, \qquad b_{\text{new}} = b - \eta \cdot 0 = b .
$$

The neuron would need to cross into $z > 0$ to receive a gradient, but its gradients provide no update that would move it there. This can come from an unlucky initialization or from one large update that pushes a previously useful neuron into the negative region for all relevant inputs.

### 4.6 Dead or deadish, and the controls

Is a dead neuron necessarily dead forever? Not always.

- In a hidden layer, **earlier layers keep learning** and can change the inputs reaching this neuron, possibly making $z > 0$ again.
- **Optimizer momentum** or other parameter updates (e.g. weight decay) can also move it.

"Dead" describes persistent inactivity and lack of task gradient through the ReLU, not a guarantee that recovery is impossible.

**Controls:** appropriate initialization (e.g. **He initialization**, weight variance $2/n$ for $n$ inputs, which compensates for ReLU zeroing about half its inputs), sensible learning rates, monitoring persistently inactive units, and using a nonzero negative-side slope (Leaky ReLU, §5.1). The corner at zero is easily handled by a derivative convention; the much larger issue is the **whole negative half-line with zero derivative**.

---

## 5. Fixing the Negative Side: Leaky ReLU, PReLU, ELU, SELU, Softplus

Every variant in this section keeps ReLU's identity-like positive branch and changes only what happens for $z < 0$, trading off sparsity, sign information and gradient flow differently.

### 5.1 Leaky ReLU

![[L08_leaky_relu.png|380]]

Leaky ReLU keeps a small slope on the negative side:

$$
\text{LeakyReLU}(z) = \begin{cases} z & z \ge 0 \\ \alpha z & z < 0 \end{cases} \;=\; \max(\alpha z, z) \text{ for } 0 < \alpha < 1, \qquad \text{e.g. } \alpha = 0.01 .
$$

- **Range:** $(-\infty, \infty)$. Derivative: $\alpha$ for $z < 0$, 1 for $z > 0$, undefined at 0 (conventionally 1 or $\alpha$).
- Example with $\alpha = 0.01$: $\text{LeakyReLU}(-3) = -0.03$.

For a negative preactivation the weight gradient becomes

$$
\frac{\partial L}{\partial w_i} = \frac{\partial L}{\partial a} \times \alpha \times x_i ,
$$

which is reduced but no longer forced to zero, so the incoming parameters keep learning. **Motivation:** a suppressed feature should still receive some corrective signal. **Strengths:** cheap, preserves signed outputs, no completely flat negative half-line. **Limitations:** a small slope can still give weak gradients; exact sparsity disappears; the slope is a manually chosen trade-off. **When useful:** a minimal fix when dead ReLUs are a concern.

### 5.2 PReLU and the $\alpha = 1$ problem

![[L08_prelu.svg|380]]

**PReLU** (Parametric ReLU; He et al., 2015) uses the same form but *learns* $\alpha$ during training, often one $\alpha$ per channel. <mark style="background: #FFF3A3A6;">Its advantage: features can learn how strongly to respond on the negative side.</mark> Its limitation:<mark style="background: #ABF7F7A6;"> extra flexibility does not guarantee better generalization, and unless constrained, the learned slope need not stay small and positive</mark>.

In particular, if $\alpha = 1$, both branches follow the same rule:

$$
\text{PReLU}(z) = \begin{cases} z & z \ge 0 \\ z & z < 0 \end{cases} \;\Rightarrow\; \text{PReLU}(z) = z .
$$

The bend disappears and the activation simply passes its input. If every activation between two successive layers becomes the identity, those layers collapse into one affine map (§1.2), and the activation adds no nonlinear power. A single neuron learning $\alpha = 1$ does not make the whole network linear if other nonlinearities remain. **Leaky ReLU preserves a small negative response; PReLU learns that response.**

### 5.3 ELU

The **Exponential Linear Unit (ELU)** (Clevert et al., 2016) is

$$
\text{ELU}(z) = \begin{cases} z & z > 0 \\ \alpha(e^z - 1) & z \le 0 \end{cases}, \qquad \alpha > 0 .
$$

**Motivation:** keep ReLU's positive branch but replace the negative side with a curve that gradually flattens. As $z \to -\infty$, $e^z \to 0$, so the output approaches $-\alpha$: $\alpha$ sets the **negative saturation level**. Unlike PReLU, ELU does not become linear at $\alpha = 1$, because the negative branch still contains $e^z$. That is the sense in which "ELU fixes the $\alpha = 1$ linearity".

![[L08_elu.png|450]]

*ELU for $\alpha = 0.1, 0.2, 0.3, 1.0$. The positive branch is the identity for all of them; $\alpha$ only sets how far below zero the negative branch levels off ($-\alpha$).*

**Negative outputs move the mean toward zero.** ReLU outputs are all nonnegative, so their mean is positive whenever some outputs are positive; ELU's negative outputs can offset positive ones (a motivation from the original paper). For one neuron with preactivations $-1$ and $+1$ and $\alpha = 1$:

| Activation | Output for $-1$ | Output for $+1$ | Mean output |
|---|---|---|---|
| ReLU | 0 | 1 | 0.5 |
| ELU | $-0.632$ | 1 | 0.184 |

The mean moves closer to zero but does not become zero. ELU does not compute or subtract a mean; the result depends on the inputs and parameters, and if all preactivations are positive ELU leaves them unchanged.

**Negative saturation makes strongly inactive responses less sensitive.** For $\alpha = 1$ and $z < 0$, $\text{ELU}(z) = e^z - 1$ with slope $e^z$:

| $z$ | ELU output | Slope $e^z$ |
|---|---|---|
| $-1$ | $-0.6321$ | 0.3679 |
| $-3$ | $-0.9502$ | 0.0498 |
| $-5$ | $-0.9933$ | 0.0067 |
| $-10$ | $-0.99995$ | 0.000045 |

Moving from $-5$ to $-10$ barely changes the output; both are about $-1$. Intuitively, if the neuron detects a feature, a very negative score means "this feature is absent", and once it is clearly absent, further differences in that score matter little. This suppresses noise in strongly inactive responses. ("Inactive" here means in the negative, saturated region; the output is not zero.)

**Smoothness at zero.** Away from zero, $\text{ELU}'(z) = 1$ for $z > 0$ and $\alpha e^z$ for $z < 0$. Approaching zero, the slope is 1 from the right and $\alpha e^0 = \alpha$ from the left. When $\alpha = 1$ these match, so the curved negative branch joins the straight positive branch without a corner (the first derivative is continuous).

**Limitations.** The same small slope that reduces sensitivity also weakens learning:
$$
\frac{\partial L}{\partial w_i} = \frac{\partial L}{\partial a} \; \underbrace{\alpha e^z}_{\text{very small for very negative } z} \; x_i .
$$

- ELU's negative branch never has ReLU's *exactly* zero derivative at finite inputs, but the gradient can become so small that learning is extremely slow.
- The negative branch needs an exponential, which costs more than ReLU's comparison.
- Signed outputs and smooth saturation do not guarantee better predictions; validation performance decides whether they help a given model.

### 5.4 SELU and self-normalization

**SELU** (Scaled Exponential Linear Unit; Klambauer et al., 2017) aims to keep activations at a stable centre and spread as they pass through many layers:

$$
\text{SELU}(z) = \begin{cases} \lambda z & z > 0 \\ \lambda\alpha(e^z - 1) & z \le 0 \end{cases}, \qquad \alpha \approx 1.6733, \quad \lambda \approx 1.0507 .
$$

These constants target **mean 0 and variance 1** under the paper's assumptions.

![[L08_selu.png|400]]

*SELU: slope $\lambda \approx 1.05$ on the positive side, and a negative branch that saturates at $-\lambda\alpha \approx -1.758$.*

**What stable mean and variance means.** Watch one neuron's output across many inputs. Its **mean** is the average output (near zero means positive and negative responses balance), and its **variance** is how widely outputs spread around that average (near one gives a consistent scale). Across many layers we want neither statistic drifting toward huge values nor collapsing toward a constant. **Self-normalization** means that, under suitable conditions, small deviations from the target $(0, 1)$ shrink as signals pass through successive layers; the paper proves the mean–variance map has a stable, attracting fixed point. It is the whole layer, weighted sum followed by SELU, that does this.

**How the shape helps.** Three properties work together:

| Property | Intuitive role |
|---|---|
| Positive and negative outputs | Allow responses to balance around zero |
| Negative saturation | Compresses large negative variations, limiting excessive spread |
| Positive slope $\lambda > 1$ | Amplifies positive responses slightly, countering shrinking spread |

The constants balance these effects under the assumed input distribution. For example, $z = 2$ gives $\text{SELU}(2) \approx 2.1014$, and the negative branch approaches $-\lambda\alpha \approx -1.7581$. But a fixed activation cannot normalize *every* collection of inputs: if all inputs equal 2, all outputs equal 2.1014, and their variance stays zero.

**Conditions.** The intended setup needs standardized inputs, suitable feed-forward layers, and weights initialized with

$$
\mathbb{E}[w] = 0, \qquad \operatorname{Var}(w) = \frac{1}{n},
$$

where $n$ is the number of incoming connections (**LeCun normal** initialization). The analysis also assumes approximate independence and roughly Gaussian weighted sums. The initialization rationale: for $n$ independent, zero-mean, unit-variance inputs, the variance of a sum of independent terms is the sum of their variances, so

$$
z = \sum_{i=1}^n w_i x_i \;\Rightarrow\; \operatorname{Var}(z) \approx n \times \frac{1}{n} \times 1 = 1 .
$$

Much larger weights would blow up the variance before SELU ever sees it.

**Why AlphaDropout?** **Dropout** randomly zeroes activations during training to regularize the network. Ordinary **inverted dropout** keeps each activation with probability $q$ and divides survivors by $q$, <mark style="background: #ABF7F7A6;">which keeps the mean but, for zero-mean inputs, changes the variance</mark>:

$$
\operatorname{Var}(\text{dropout output}) = \frac{\operatorname{Var}(\text{input})}{q} .
$$

*(Why: the output is $x \cdot m / q$ with mask $m \sim \text{Bernoulli}(q)$, so $\mathbb{E}[(xm/q)^2] = \mathbb{E}[x^2] \cdot q / q^2 = \mathbb{E}[x^2]/q$.)* Keeping half the activations therefore doubles the variance and breaks SELU's statistics. **AlphaDropout** <mark style="background: #FFF3A3A6;">instead sets dropped values to SELU's negative saturation value</mark> $-\lambda\alpha$, then <mark style="background: #FFF3A3A6;">applies a scale and shift</mark> designed to restore mean 0 and variance 1.

**Key limitation:** SELU's self-normalization depends on the network design, initialization and assumptions. It is not a general replacement for normalization layers in arbitrary Transformers.

### 5.5 Why SELU cannot replace Transformer normalization

A Transformer contains other operations that change these statistics. A residual addition $y = x + F(x)$ gives

$$
\operatorname{Var}(y) = \operatorname{Var}(x) + \operatorname{Var}(F(x)) + 2\operatorname{Cov}(x, F(x)) .
$$

Even if both branches have variance 1, the sum has variance 2 when they are uncorrelated (and up to 4 if perfectly correlated). Attention also mixes representations with input-dependent weights. SELU's feed-forward argument does not automatically extend to these operations, so swapping the FFN activation for SELU does not show that LayerNorm or RMSNorm (the per-token normalization layers from the previous lecture) can be safely removed.

### 5.6 Softplus

$$
\text{Softplus}(z) = \log(1 + e^z), \qquad \text{Softplus}'(z) = \frac{e^z}{1+e^z} = \sigma(z) .
$$

For strongly positive $z$, $\log(1 + e^z) \approx \log e^z = z$; for strongly negative $z$, it approaches $\log 1 = 0$. So softplus is a smooth version of ReLU, and its derivative is exactly the sigmoid.

![[L08_softplus.png|450]]

*Softplus (purple) and its derivative (red). The derivative is the sigmoid: it rises smoothly from 0 to 1 instead of jumping, as ReLU's does.*

**Motivation:** a smooth rectifier without ReLU's hard corner or exactly flat negative half-line. **Strengths:** smooth and strictly positive for finite inputs (in exact arithmetic). **Limitations:** <mark style="background: #ABF7F7A6;">very negative inputs still have tiny derivatives; it is more expensive than ReLU; no exact zeros.</mark> **When useful:** predicting a quantity that must be positive, such as a scale or standard deviation; a small constant can be added when a positive floor is needed. Stable implementations avoid computing $e^z$ for large $z$ *(e.g. $\text{Softplus}(z) = \max(z, 0) + \log(1 + e^{-|z|})$)*. Softplus addresses smoothness; it does not preserve signed weak signals (it is never negative).

---

## 6. Self-Regulating Activations: GELU, SiLU, Swish

The section-5 variants each patch ReLU's negative side with a different hand-designed curve. GELU and SiLU take a different route: they rewrite ReLU as "input times gate" and make the gate smooth.

### 6.1 GELU: ReLU's gate made gradual

ReLU can be rewritten as the input multiplied by a binary gate:

$$
\text{ReLU}(z) = z \cdot \mathbb{1}[z > 0], \qquad \mathbb{1}[z > 0] = \begin{cases} 1 & z > 0 \text{ (retain the whole input)} \\ 0 & \text{otherwise (discard it)} \end{cases}
$$

The **Gaussian Error Linear Unit (GELU)** (Hendrycks & Gimpel, 2016) replaces that all-or-nothing gate with a gradual weight:

$$
\boxed{\text{GELU}(z) = z\,\Phi(z)}, \qquad \Phi(z) = P(U \le z), \quad U \sim \mathcal{N}(0, 1) .
$$

$\Phi$ is the **standard normal CDF** (cumulative distribution function): the area under the standard bell curve (mean 0, standard deviation 1) to the left of $z$. For example, $\Phi(0) = 0.5$, $\Phi(1) \approx 0.8413$ (about 84% lies below one), $\Phi(-1) \approx 0.1587$. The bell curve only supplies the weighting rule; GELU does not estimate the distribution of the current inputs or compute their percentiles.

**Motivation:** soften ReLU's all-or-nothing decision while letting strong positive signals pass almost unchanged; probability provides a principled way to build the gradual weighting. Larger scores get weights closer to one, smaller scores weights closer to zero.

![[L08_gelu.png|420]]

*GELU on $[-3, 3]$: almost the identity for large positive $z$, near zero for large negative $z$, with a smooth shallow dip just below zero.*

| Score $z$ | Weight $\Phi(z)$ | GELU $z\Phi(z)$ | ReLU |
|---|---|---|---|
| $-3$ | 0.00135 | $-0.00405$ | 0 |
| $-1$ | 0.1587 | $-0.1587$ | 0 |
| 0 | 0.5 | 0 | 0 |
| 1 | 0.8413 | 0.8413 | 1 |
| 3 | 0.99865 | 2.99595 | 3 |

So strongly positive scores are almost fully retained, moderately positive ones partially, moderately negative ones leave a small negative response, and strongly negative ones are almost fully suppressed. At $z = 0$ the weight is 0.5 but the output is $0 \times 0.5 = 0$. The *weight* lies in $(0,1)$, but the *output* need not; GELU is not producing a class probability.

*(Expanding on this: $\Phi$ has no closed form, so implementations often use the tanh approximation $\text{GELU}(z) \approx \tfrac12 z\big(1 + \tanh[\sqrt{2/\pi}(z + 0.044715z^3)]\big)$, from the GELU paper, or the cheaper $z\,\sigma(1.702z)$, which already looks like SiLU from §6.4.)*

### 6.2 The probabilistic interpretation

GELU outputs the average value you would get if you randomly kept or discarded the neuron's score with a particular probability, but it computes that average directly, with no randomness.

Imagine a switch that either keeps the score (output $z$) or discards it (output 0), keeping it with probability $\Phi(z)$, so larger scores are more likely kept. Fix $z = 1$, where $\Phi(1) \approx 0.8413$. Running the switch 10,000 times would give about 8,413 outputs of 1 and 1,587 outputs of 0, with average $\frac{8413(1) + 1587(0)}{10000} = 0.8413$. The exact expectation is

$$
\mathbb{E}[\text{output} \mid z = 1] = 1(0.8413) + 0(0.1587) = 0.8413,
$$

where "$\mid z = 1$" means the input is fixed at 1 and we average over switch outcomes, not over different inputs. GELU skips the repetitions and simply returns that expectation:

$$
\mathbb{E}[\text{hypothetical output} \mid z] = z\Phi(z), \qquad \text{GELU}(1) = 1 \times \Phi(1) = 0.8413 .
$$

| Hypothetical random switch | Actual GELU |
|---|---|
| Returns either 1 or 0 on each trial | Returns 0.8413 every time |
| Its expected output is 0.8413 | Directly computes that expected output |

The switch keeps $z$, not necessarily the number 1; the $z = 1$ example hides this. For $z = 2$ it would output 2 with probability $\Phi(2) \approx 0.9772$ and 0 otherwise, so $\text{GELU}(2) = 2(0.9772) \approx 1.9545$. **$\Phi(z)$ is the retention probability in the story; $z\Phi(z)$ is the activation value, which is not itself a probability.** *(Expanding on this: the GELU paper motivates the story by dropout-like stochastic regularizers such as zoneout, which multiply inputs by random 0/1 masks; GELU makes the mask probability depend on the input itself.)*

### 6.3 GELU's derivative, non-monotonicity and limitations

By the product rule, with $\phi(z) = \Phi'(z)$ the standard normal **density** (the bell curve's height),

$$
\frac{d}{dz}\big[z\Phi(z)\big] = \underbrace{\Phi(z)}_{\text{current weight}} + \underbrace{z\,\phi(z)}_{\text{effect of changing the weight}}, \qquad \text{GELU}'(0) = 0.5 .
$$

Near zero, the neuron receives gradients from either side, unlike ReLU. Far into the negative tail both terms approach zero, so learning can still become very slow.

**GELU is not monotonic.** In the table, moving from $-3$ to $-1$ makes the output *more* negative ($-0.004 \to -0.159$): the input's magnitude decreases, but the retained fraction increases enough to outweigh that. The derivative is negative in part of this region and zero at the minimum *(at $z \approx -0.75$, where $\text{GELU} \approx -0.170$)*.

**Practical limitations:** GELU produces many small nonzero values (it is nonzero for every finite $z \ne 0$), whereas ReLU gives exact zeros for every negative input; and computing $\Phi$ or an approximation costs more than ReLU's maximum.

| GELU | |
|---|---|
| Strengths | Smooth transition, small negative outputs, no exactly flat negative half-line |
| Limitations | Tiny negative-tail gradients, more involved computation, little exact sparsity |
| When useful | A smooth activation in a conventional two-matrix FFN (as in BERT and GPT-2). Its usefulness is empirical, not a consequence of smoothness alone |

### 6.4 SiLU and self-gating

**SiLU** (Sigmoid Linear Unit; Elfwing et al., 2017) follows the same idea as GELU with sigmoid as the multiplier:

$$
\text{GELU}(z) = z\Phi(z), \qquad \boxed{\text{SiLU}(z) = z\,\sigma(z)} .
$$

It is applied exactly like an ordinary activation: $z = \sum_i w_i x_i + b \longrightarrow a = z\sigma(z)$.

![[L08_silu.png|420]]

*SiLU on $[-10, 10]$: zero far left, a small dip just below zero, then nearly the identity on the right.*

**Self-gating.** The same scalar $z$ plays two roles:

$$
\text{SiLU}(z) = \underbrace{z}_{\text{value being passed}} \; \underbrace{\sigma(z)}_{\text{fraction retained}} .
$$

For $z = 2$ the multiplier is 0.881 and the output 1.762; for $z = -2$ the multiplier is 0.119 and the output $-0.238$. It is called **self-gating** because the value controls its own retention; there is no separate neuron generating the gate (contrast GLU, §7).

**Why multiply by $z$ instead of using sigmoid alone?** Sigmoid alone approaches a ceiling, $\sigma(5) \approx 0.993$, $\sigma(10) \approx 1$, so large positive scores become indistinguishable and the derivative vanishes. With SiLU, $5\sigma(5) \approx 4.967$ and $10\sigma(10) \approx 10$: although the multiplier saturates near one, the output keeps growing,

$$
z\sigma(z) \approx z \quad \text{for large positive } z,
$$

which keeps a nearly linear positive branch with derivative approaching one.

### 6.5 Swish

**Swish** (Ramachandran, Zoph & Le, 2017, found by an automated search over activation functions and evaluated empirically across several architectures) generalizes SiLU with a parameter $\beta$ (fixed or learned):

$$
\text{Swish}_\beta(z) = z\,\sigma(\beta z) .
$$

$\beta$ controls how abruptly the multiplier changes around zero:

| $\beta$ | Multiplier $\sigma(\beta z)$ | Result |
|---|---|---|
| 0 | $\sigma(0) = \tfrac12$ for every input | $\text{Swish}_0(z) = z/2$, **linear** |
| 1 | $\sigma(z)$ | SiLU |
| Larger positive | Switches more rapidly from near 0 to near 1 | Sharper bend |
| $\beta \to \infty$ | $\to 0$ for $z < 0$, $\to 1$ for $z > 0$ | Approaches **ReLU** (at $z = 0$ the output is already 0) |

![[L08_swish.png|420]]

*Swish for $\beta = 0.1$ (nearly the line $z/2$), $\beta = 1$ (SiLU, with its dip) and $\beta = 10$ (nearly ReLU).*

So Swish interpolates between a linear function and ReLU, with SiLU in between.

### 6.6 Smooth is not monotonic: the negative dip

Why can the output decrease while the input increases? Both factors change, and for negative inputs, opening the gate lets *more negative content* through:

$$
\text{SiLU}(-2) = (-2)(0.119) \approx -0.238, \qquad \text{SiLU}(-1) = (-1)(0.269) \approx -0.269 .
$$

Moving from $-2$ to $-1$, the input's magnitude halves but the retained fraction more than doubles, so the output becomes slightly more negative. Eventually, as $z \to 0$, the input itself goes to zero and the output rises back to zero, producing a small dip *(minimum at $z \approx -1.28$, value $\approx -0.278$)*.

> [!warning] Possible source inconsistency
> The slide calls SiLU "smooth … and monotonically increasing (increasing the input never decreases the output)". The dip just described, and the next slide's negative derivative $\text{SiLU}'(-2) \approx -0.091$, show SiLU is **not** monotonic. The slide's own point stands: a curve can be smooth while still having a dip.

The product rule gives

$$
\text{SiLU}'(z) = \underbrace{1 \cdot \sigma(z)}_{\text{change in the value}} + \underbrace{z \cdot \sigma(z)(1 - \sigma(z))}_{\text{change in the gate}}, \qquad \text{SiLU}'(0) = 0.5 .
$$

For negative $z$ the second term is negative and can outweigh the first:

| $z$ | $\text{SiLU}'(z)$ | Meaning |
|---|---|---|
| $-2$ | $-0.091$ | Increasing the input makes the output more negative |
| 0 | 0.5 | Gradients pass through the origin |
| 2 | 1.091 | Locally, the output changes slightly faster than the input |

A derivative above one is possible even though the multiplier is below one, because increasing $z$ increases both the value and the fraction retained.

All five activations side by side:

| $z$ | Sigmoid | Tanh | ReLU | GELU | SiLU |
|---|---|---|---|---|---|
| $-2$ | 0.119 | $-0.964$ | 0 | $-0.046$ | $-0.238$ |
| $-1$ | 0.269 | $-0.762$ | 0 | $-0.159$ | $-0.269$ |
| 0 | 0.500 | 0 | 0 | 0 | 0 |
| 1 | 0.731 | 0.762 | 1 | 0.841 | 0.731 |
| 2 | 0.881 | 0.964 | 2 | 1.954 | 1.762 |

For both GELU and SiLU, moving from $-2$ to $-1$ makes the output more negative: at very negative inputs the multiplier is almost closed, and as the input rises the multiplier opens and lets more negative content through, before the input finally reaches zero. These functions are smooth nonlinear modulation, not merely softened versions of a monotonic threshold.

### 6.7 Why gradients can still vanish

For very negative $z$, $\sigma(z) = \frac{1}{1 + e^{-z}} \approx e^z$ (since $e^{-z}$ dominates the denominator) and $1 - \sigma(z) \approx 1$, so

$$
\text{SiLU}'(z) \approx e^z + z e^z = (1 + z)e^z \to 0 .
$$

The gate is almost closed, so changes in the input barely affect the output. SiLU avoids ReLU's entire flat negative half-line but does not eliminate very small gradients. This applies to SiLU and Swish with any fixed $\beta > 0$; the special case $\beta = 0$ has constant derivative $1/2$.

---

## 7. Separately Regulated Activations: GLU and Gated FFNs

SiLU and GELU let a score gate *itself*. The structural change in this section is to compute the gate from a *different* learned projection of the input.

### 7.1 From one neuron to a gated pair

ReLU, GELU and SiLU each transform one weighted sum: an ordinary SiLU neuron computes $z = \sum_i w_i x_i + b$, $h = z\sigma(z)$, so one score supplies both the value and its multiplier. A **Gated Linear Unit (GLU)** (Dauphin et al., 2017) computes two weighted sums from the same input and combines them by multiplication:

$$
u = \sum_i w_{u,i}x_i + b_u, \qquad g = \sum_i w_{g,i}x_i + b_g, \qquad \boxed{h = u\,\sigma(g)}
$$

The two sums use different learned weights and biases:

- $u$ (the **content**) decides the value available to pass forward;
- $g$ (the **gate score**) decides how much of that value passes forward.

They are **parallel** computations on the same input vector, not two successive layers. This separation is the defining GLU mechanism. (The original GLU paper used it in gated convolutional language models.)

### 7.2 Why separate content from control

**Motivation:** the conditions under which a feature is useful may differ from the evidence that the feature is present. SiLU lets a feature regulate itself; GLU lets a separately learned signal regulate it. (These are useful intuitions, not guaranteed semantic roles.)

Imagine a feature for "evidence for the financial meaning of *bank*". A useful division of work might be: the content branch produces a candidate financial feature, and the gate branch looks at contextual evidence such as "deposit" versus "river" and controls its contribution. In a Transformer FFN both branches see the token's representation, which may already contain context gathered by attention. With content fixed at $u = 4$:

| Context-dependent gate score $g$ | Multiplier $\sigma(g)$ | Output $h = 4\sigma(g)$ |
|---|---|---|
| $-3$ | 0.047 | 0.190 |
| 0 | 0.500 | 2.000 |
| 3 | 0.953 | 3.810 |

The same content can contribute weakly or strongly depending on another learned score. This shows the architecture's flexibility; training does not guarantee that individual branches acquire such clean meanings.

### 7.3 Extending to a whole layer

Omitting biases, for a row-vector input $x \in \mathbb{R}^{1 \times d}$ and $m$ hidden units,

$$
\underset{1\times m}{u} = \underset{1\times d}{x}\,\underset{d\times m}{W_u}, \qquad \underset{1\times m}{g} = \underset{1\times d}{x}\,\underset{d\times m}{W_g}, \qquad h = u \odot \sigma(g),
$$

where $\odot$ is **elementwise multiplication**: $h_j = u_j\,\sigma(g_j)$ for $j = 1, \dots, m$, i.e. $h = [u_1\sigma(g_1), \dots, u_m\sigma(g_m)]$. Each branch is fully connected to all $d$ inputs; the multiplication only pairs corresponding outputs. A complete gated FFN then maps back to width $d$:

$$
\underset{1\times d}{y} = \underset{1\times m}{h}\,\underset{m\times d}{W_{\text{out}}} .
$$

### 7.4 Why sigmoid returns: gradients through the two branches

Section 3.3 dismissed sigmoid for hidden layers because of saturation. In GLU it controls a *separate linear* content branch, which changes the gradient picture. The two local derivatives of $h = u\sigma(g)$ are

$$
\frac{\partial h}{\partial u} = \sigma(g), \qquad \frac{\partial h}{\partial g} = u\,\sigma(g)\big(1 - \sigma(g)\big) .
$$

For one pair with $u = \sum_i w_{u,i}x_i$ and $g = \sum_i w_{g,i}x_i$, the chain rule gives, for the content and gate weights respectively,

$$
\frac{\partial L}{\partial w_{u,i}} = \frac{\partial L}{\partial h}\frac{\partial h}{\partial u}\frac{\partial u}{\partial w_{u,i}} = \frac{\partial L}{\partial h}\,\sigma(g)\,x_i, \qquad
\frac{\partial L}{\partial w_{g,i}} = \frac{\partial L}{\partial h}\frac{\partial h}{\partial g}\frac{\partial g}{\partial w_{g,i}} = \frac{\partial L}{\partial h}\,u\,\sigma(g)\big(1-\sigma(g)\big)\,x_i .
$$

**Only the gate-weight path contains the sigmoid derivative.** When adjusting the content weights we change $u$ while the separately computed $g$ stays fixed, so the fixed gate just scales the content gradient. With $u = 4$:

| Gate score | Gate value $\sigma(g)$ | Content sensitivity $\partial h/\partial u$ | Gate sensitivity $\partial h/\partial g$ |
|---|---|---|---|
| $g = 5$: nearly open | 0.9933 | 0.9933 | 0.0266 |
| $g = 0$: intermediate | 0.5 | 0.5 | 1 |
| $g = -5$: nearly closed | 0.0067 | 0.0067 | 0.0266 |

When the gate is nearly open, $h \approx u$: changing $u$ changes $h$ almost one-for-one, while changing an already large $g$ barely moves the gate. So a saturated-open gate is slow to adjust but the content passing through it keeps learning; this linear gradient path was an explicit motivation for GLU. Two caveats: a saturated-*closed* gate ($\sigma(g) \approx 0$) suppresses content learning too, and none of this guarantees a large overall gradient, since the downstream factor $\partial L / \partial h$ can itself be small.

### 7.5 Conventional vs gated FFN

A conventional FFN is

$$
x \longrightarrow \text{weighted sums} \longrightarrow \text{activation} \longrightarrow \text{output weighted sums}.
$$

A gated FFN replaces the first weighted sums and activation with two parallel branches:

$$
u = xW_u, \qquad g = xW_g, \qquad h = u \odot f(g), \qquad y = hW_{\text{out}},
$$

where $f$ is sigmoid, ReLU, GELU or SiLU depending on the variant. The last equation is just a layer of output neurons, each combining all gated hidden features: $y_k = \sum_{j=1}^m h_j (W_{\text{out}})_{jk}$. Biases are omitted throughout (modern LLMs usually drop them). Usually no activation follows this output projection in a Transformer FFN; its job is to turn the hidden representation back into a $d$-dimensional vector that can be added to the residual stream.

### 7.6 ReGLU, GEGLU and SwiGLU

Shazeer (2020) kept the two-branch structure and changed the activation on the gate score:

| Variant | Hidden representation | What the multiplier permits |
|---|---|---|
| GLU | $u \odot \sigma(g)$ | Attenuation or approximate preservation (multiplier in $(0,1)$) |
| ReGLU | $u \odot \text{ReLU}(g)$ | Suppression or amplification (multiplier in $[0, \infty)$) |
| GEGLU | $u \odot \text{GELU}(g)$ | Smooth suppression, amplification, or sign change |
| SwiGLU | $u \odot \text{SiLU}(g) = u \odot g \odot \sigma(g)$ | Smooth suppression, amplification, or sign change |

ReGLU offers simple multiplicative gating but keeps a zero-gradient region for negative gate scores. GEGLU replaces the hard cutoff with Gaussian-based smooth weighting. **SwiGLU** uses sigmoid-based smooth weighting,

$$
h = u \odot g \odot \sigma(g) .
$$

Its strength is the combination of a smooth activation with interactions between two learned projections (the product $u \cdot g$ is a second-order term in $x$). Its limitations: added computation, small gradients for strongly negative gate scores, and potentially large products. GEGLU and SwiGLU did best in Shazeer's controlled Transformer experiments; that is evidence of usefulness, not a universal ranking. SwiGLU is the FFN in LLaMA, PaLM and most current open LLMs.

![[L08_gated_ffns.png|650]]

*Three gated FFNs with the same two-branch structure. The input ($1 \times d$) feeds a content projection $u = xW_u$ and a gate projection $g = xW_g$ (both $d \times m$); only the gate activation differs (ReLU, GELU, SiLU). The $m$ hidden outputs are multiplied elementwise and projected back by $W_{\text{out}}$ ($m \times d$).*

**A numerical SwiGLU example.** Fix the content at $u = 4$:

| Gate score $g$ | GLU output $4\sigma(g)$ | SwiGLU output $4g\sigma(g)$ |
|---|---|---|
| $-2$ | 0.477 | $-0.954$ |
| 0 | 2.000 | 0 |
| 2 | 3.523 | 7.046 |

Three observations follow: at $g = 0$, GLU passes half the content while SwiGLU outputs zero; with sufficiently positive $g$, SwiGLU can **amplify** (7.046 > 4); and with negative $g$, SwiGLU can **reverse the content's sign**. So "gate" here means a learned multiplier, not necessarily an on/off switch. The content $u$ can itself be negative, so the final sign depends on both branches.

### 7.7 The SwiGLU parameter budget

A conventional FFN and a SwiGLU FFN are

$$
\underset{1\times d}{y} = \phi\big(\underset{1\times d}{x}\,\underset{d\times m}{W_1}\big)\,\underset{m\times d}{W_2}, \qquad
\underset{1\times d}{y} = \big(\underbrace{xW_u \odot \text{SiLU}(xW_g)}_{1\times m}\big)\,\underset{m\times d}{W_o} .
$$

The conventional version has two weight matrices; SwiGLU has three. Ignoring biases,

$$
P_{\text{conventional}} = 2d\,m_{\text{conventional}}, \qquad P_{\text{SwiGLU}} = 3d\,m_{\text{SwiGLU}} .
$$

Setting them equal gives

$$
\boxed{m_{\text{SwiGLU}} = \tfrac{2}{3}\,m_{\text{conventional}}} .
$$

If the conventional hidden width is $4d$, the matched SwiGLU width is about $8d/3$, usually rounded up for hardware efficiency; LLaMA uses this adjustment. At a matched budget, SwiGLU trades some width for richer feature interactions. It is therefore more than a scalar activation swap: the architecture, parameter count and intermediate activations all change.

---

## 8. Choosing an Activation

Sections 3–7 covered the options for hidden layers. The last question is how to choose among them, and output layers follow different rules.

### 8.1 Hidden vs output activations

A hidden activation should support useful representations and effective learning. An output activation must also match the meaning and allowed range of the prediction:

| Prediction task | Typical output transformation | Reason |
|---|---|---|
| Unrestricted real-valued regression | Identity | Allows any real value |
| Binary classification | Sigmoid | One probability |
| Multilabel classification | Separate sigmoids | Several labels may be true at once |
| Mutually exclusive multiclass classification | Softmax | Probabilities compete and sum to one |
| Strictly positive parameter | Softplus or exponential | Enforces positivity |
| Bounded signed quantity | Scaled tanh | Enforces a bounded interval |

These are common constructions, not the only possible modelling choices. (**Softmax** maps a score vector to $\text{softmax}(z)_k = e^{z_k}/\sum_j e^{z_j}$; with two classes it reduces to the sigmoid of the score difference.)

### 8.2 Practical starting points

| Situation | Sensible starting point | Main concern |
|---|---|---|
| Simple hidden-layer baseline | ReLU | Persistent inactivity |
| ReLU units frequently die | Leaky ReLU | Choosing the negative slope |
| A learnable negative response is useful | PReLU | Added flexibility vs generalization |
| Smooth conventional FFN | GELU or SiLU | Quality and runtime on the target architecture |
| More expressive gated FFN | GEGLU or SwiGLU | Matched width, memory and kernel efficiency |
| Architecture built for self-normalization | SELU with its intended setup | Assumptions and component compatibility |
| Positive output parameter | Softplus | Small gradients for very negative logits |
| Existing pretrained model | Keep its trained activation | A replacement changes the learned function |

Guidance for choosing:

- Compare choices on validation quality, training stability, memory and measured execution time, not on how the curve looks.
- Initialization, learning rate, normalization and residual connections all interact with the activation; no activation guarantees stable gradients on its own.
- Replacing an activation in a trained network is not a free upgrade: the weights were optimized for the original function. Moving from an ordinary FFN to SwiGLU also requires a new projection and training or adaptation.

### 8.3 Summary of activation functions

The deck closes with two reference tables (a large comparison table from a Medium article, and the infographic below). The one-line summaries from the closing slide, with three functions that appear only there defined:

| Activation | One-line summary |
|---|---|
| Sigmoid | Squashes input into $(0,1)$; historically popular, causes vanishing gradients in deep nets |
| Tanh | Like sigmoid but range $(-1, 1)$; zero-centered, which helps somewhat |
| ReLU | Activates when input exceeds a threshold; cheap, helps vanishing gradients, can create dead neurons |
| Leaky ReLU | Small negative slope stops neurons from dying |
| PReLU | Like Leaky ReLU, but the negative slope is learned |
| ELU | Pushes mean activations toward zero; negative outputs in $(-\alpha, 0)$ |
| SELU | Scaled ELU that self-normalizes (mean 0, variance 1) under specific conditions |
| Softplus | Smooth, always-positive approximation to ReLU |
| **Softsign** | $z / (1 + \lvert z\rvert)$: tanh-like shape in $(-1,1)$ but with polynomial rather than exponential tails; rarely used |
| **Hard Sigmoid** | Piecewise-linear sigmoid, e.g. $\max(0, \min(1, z/6 + 1/2))$ in PyTorch; cheaper than the exponential |
| Swish | Self-gated $z\sigma(\beta z)$ found by search at Google; often beats ReLU |
| **Mish** | $z \tanh(\text{Softplus}(z))$: smooth, non-monotonic, SiLU-like; reported to beat several older activations |

*(Softsign, Hard Sigmoid and Mish formulas are added from outside the deck; hard-sigmoid slopes vary by library, e.g. Keras historically used $0.2z + 0.5$.)*

> [!warning] Possible source inconsistency
> The closing summary slide gives sigmoid's range as $[0, 1]$ (the open interval $(0,1)$ is correct, §3.1) and says tanh's zero-centering "can help mitigate the vanishing gradient problem to some extent", while §3.5's slide says centering "does not remove vanishing-gradient risks". The two are compatible (helps a little, does not fix), but tanh still saturates. The ELU line's "slowing down the learning but producing a more robust model" is the summary author's claim; §5.3's slides say only validation can show whether ELU helps.

![[L08_activation_summary.png|420]]

*Shapes of the twelve activations above. The family resemblance is clear: almost all modern choices are near-identity for large positive inputs and differ only in how they treat the negative side and the region around zero.*

---

## 9. Numerical Problems

**Problem 1 (affine collapse).** A two-layer network with no activations has $W_1 = \begin{bmatrix} 1 & 2 \\ 0 & 1 \end{bmatrix}$, $b_1 = [1, -1]$, $W_2 = \begin{bmatrix} 2 \\ -1 \end{bmatrix}$, $b_2 = 3$. Find the equivalent single affine layer and check it on $x = [1, 1]$.

*Solution.* From §1.2, $W_{\text{eff}} = W_1W_2 = \begin{bmatrix} 1\cdot2 + 2\cdot(-1) \\ 0\cdot2 + 1\cdot(-1) \end{bmatrix} = \begin{bmatrix} 0 \\ -1 \end{bmatrix}$ and $b_{\text{eff}} = b_1W_2 + b_2 = (1)(2) + (-1)(-1) + 3 = 6$. So $y = -x_2 + 6$: the network ignores $x_1$ entirely, which is impossible to see from the separate layers. Checking directly, $h = xW_1 + b_1 = [1, 3] + [1, -1] = [2, 2]$ and $y = 2(2) + 2(-1) + 3 = 5$; the collapsed form gives $-1 + 6 = 5$.

$$
\boxed{y = -x_2 + 6, \quad y([1,1]) = 5}
$$

**Problem 2 (sigmoid, log-odds, BCE).** (a) What logit gives $p = 0.9$? (b) For $z = -2$ and $y = 1$, compute $p$, the BCE gradient $\partial L/\partial z$, and the hidden-layer sigmoid derivative $\sigma'(z)$.

*Solution.* (a) Using the log-odds of §3.2, $z = \ln\frac{0.9}{0.1} = \ln 9 = 2.197$. (b) $p = \sigma(-2) = 0.1192$. By §3.4, $\partial L / \partial z = p - y = -0.881$, a strong push upward. If the same sigmoid sat in a hidden layer, its local factor would be only $\sigma'(-2) = 0.1192 \times 0.8808 = 0.105$, an 8× weaker scaling; the BCE cancellation is what rescues the output case.

$$
\boxed{z = 2.197; \quad \partial L/\partial z = -0.881; \quad \sigma'(-2) = 0.105}
$$

**Problem 3 (vanishing across depth).** Ignoring weights, multiply the activation derivatives along a 10-layer path where every preactivation equals (a) 0 with sigmoid, (b) 1 with sigmoid, (c) 1 with tanh, (d) any positive value with ReLU.

*Solution.* (a) $0.25^{10} = 9.54 \times 10^{-7}$. (b) $\sigma'(1) = 0.731 \times 0.269 = 0.1966$, so $0.1966^{10} = 8.6 \times 10^{-8}$. (c) $\tanh'(1) = 1 - 0.762^2 = 0.420$, so $0.420^{10} = 1.7 \times 10^{-4}$, larger than sigmoid's but still vanishing. (d) $1^{10} = 1$.

$$
\boxed{9.5\times10^{-7}, \quad 8.6\times10^{-8}, \quad 1.7\times10^{-4}, \quad 1}
$$

**Problem 4 (dead ReLU vs Leaky ReLU).** A neuron $z = 3x - 5$ sees inputs $x \in [0, 1]$. (a) Is it dead? (b) At $x = 0.5$ with upstream gradient $\partial L/\partial a = 2$, compute $\partial L / \partial w$ and $\partial L / \partial b$ under ReLU and under Leaky ReLU with $\alpha = 0.01$.

*Solution.* (a) $z$ ranges over $[-5, -2]$, all negative, so a ReLU neuron outputs 0 on every input and is dead (§4.5). (b) At $x = 0.5$, $z = -3.5$. ReLU: $\partial a/\partial z = 0$, so both gradients are 0. Leaky ReLU: $\partial a/\partial z = 0.01$, giving $\partial L / \partial w = 2 \times 0.01 \times 0.5 = 0.01$ and $\partial L / \partial b = 2 \times 0.01 \times 1 = 0.02$. Small, but nonzero, so the bias can climb back.

$$
\boxed{\text{ReLU: } (0, 0); \quad \text{Leaky: } \partial L/\partial w = 0.01, \; \partial L/\partial b = 0.02}
$$

**Problem 5 (ELU mean and SELU values).** One neuron receives preactivations $\{-2, 1, 3\}$. (a) Compute the mean output under ReLU and under ELU ($\alpha = 1$). (b) Compute $\text{SELU}(-1)$ and $\text{SELU}(1)$.

*Solution.* (a) ReLU gives $\{0, 1, 3\}$, mean $4/3 = 1.333$. ELU gives $e^{-2} - 1 = -0.865$ for the negative input, so $\{-0.865, 1, 3\}$, mean $3.135/3 = 1.045$. The mean moves toward zero but, as §5.3 warns, ELU does not centre it. (b) $\text{SELU}(-1) = 1.0507 \times 1.6733 \times (e^{-1} - 1) = 1.7581 \times (-0.6321) = -1.111$; $\text{SELU}(1) = 1.0507$.

$$
\boxed{\bar a_{\text{ReLU}} = 1.333, \; \bar a_{\text{ELU}} = 1.045; \quad \text{SELU}(-1) = -1.111, \; \text{SELU}(1) = 1.051}
$$

**Problem 6 (GELU vs SiLU at $z = 1.5$).** Given $\Phi(1.5) = 0.9332$ and $\phi(1.5) = 0.1295$, compute GELU, SiLU and their derivatives.

*Solution.* $\text{GELU}(1.5) = 1.5 \times 0.9332 = 1.400$, and by §6.3, $\text{GELU}'(1.5) = 0.9332 + 1.5 \times 0.1295 = 1.127$. For SiLU, $\sigma(1.5) = 0.8176$, so $\text{SiLU}(1.5) = 1.226$ and $\text{SiLU}'(1.5) = 0.8176 + 1.5 \times 0.8176 \times 0.1824 = 1.041$. Both derivatives exceed 1, for the reason in §6.6: increasing $z$ raises both the value and the fraction kept. GELU's gate is steeper (the normal CDF rises faster than the sigmoid), so it retains more at the same $z$.

$$
\boxed{\text{GELU} = 1.400, \; \text{GELU}' = 1.127; \quad \text{SiLU} = 1.226, \; \text{SiLU}' = 1.041}
$$

**Problem 7 (GLU gradients and SwiGLU output).** One GLU pair has $u = -3$, $g = 1$, upstream gradient $\partial L/\partial h = 0.5$, and input $x_i = 2$ feeding both branches. Compute $\partial L / \partial w_{u,i}$ and $\partial L / \partial w_{g,i}$, then the SwiGLU output for the same $u, g$.

*Solution.* $\sigma(1) = 0.7311$ and $\sigma(1)(1 - \sigma(1)) = 0.1966$. From §7.4, the content gradient is $0.5 \times 0.7311 \times 2 = 0.731$ and the gate gradient is $0.5 \times (-3)(0.1966) \times 2 = -0.590$. The gate gradient carries the sigmoid derivative and the content's sign; the content gradient does not. SwiGLU gives $h = u \cdot g \cdot \sigma(g) = -3 \times 1 \times 0.7311 = -2.193$, whereas GLU would give $-3 \times 0.7311 = -2.193$ too, since $g = 1$ makes the extra factor $g$ equal to 1.

$$
\boxed{\partial L/\partial w_{u,i} = 0.731, \quad \partial L/\partial w_{g,i} = -0.590, \quad h_{\text{SwiGLU}} = -2.193}
$$

**Problem 8 (SwiGLU width matching).** A model has $d = 4096$ and a conventional FFN of width $4d$. (a) How many FFN weights per layer? (b) What SwiGLU width matches it? (c) LLaMA-7B uses $m = 11008$; what fraction of the conventional budget is that?

*Solution.* (a) $P = 2dm = 2 \times 4096 \times 16384 = 134{,}217{,}728$. (b) From §7.7, $m_{\text{SwiGLU}} = \tfrac23 \times 16384 = 10922.7$. (c) $11008$ is $10922.7$ rounded up to a multiple of 256, giving $P = 3 \times 4096 \times 11008 = 135{,}266{,}304$, which is $1.0078\times$ the conventional budget, i.e. within 1%.

$$
\boxed{134.2\text{M}; \quad m \approx 10923; \quad 135.3\text{M} \approx 1.008\times}
$$

**Problem 9 (dropout variance and SELU initialization).** (a) Inverted dropout with keep probability $q = 0.8$ acts on zero-mean, unit-variance activations. What is the output variance? (b) Under SELU's initialization, what standard deviation should weights have in a layer with $n = 512$ inputs?

*Solution.* (a) From §5.4, $\operatorname{Var} = 1/0.8 = 1.25$, a 25% inflation, which is why SELU networks use AlphaDropout. (b) $\operatorname{Var}(w) = 1/512$, so $\text{std} = 1/\sqrt{512} = 0.0442$.

$$
\boxed{\operatorname{Var} = 1.25; \quad \sigma_w = 0.0442}
$$

---

## 10. Formula Summary Sheet

| Concept | Formula |
|---|---|
| Neuron | $z = \sum_i w_i x_i + b, \; a = \phi(z)$ |
| Affine collapse | $W_{\text{eff}} = W_1W_2, \; b_{\text{eff}} = b_1W_2 + b_2$ |
| XOR network | $h_1 = \text{ReLU}(x_1 + x_2), \; h_2 = \text{ReLU}(x_1 + x_2 - 1), \; y = h_1 - 2h_2$ |
| Weight gradient | $\frac{\partial L}{\partial w_i} = \frac{\partial L}{\partial a}\,\phi'(z)\,x_i$ |
| Sigmoid | $\sigma(z) = \frac{1}{1+e^{-z}}, \; \sigma' = \sigma(1-\sigma) \le 0.25$ |
| Odds / log-odds | $\frac{p}{1-p}, \; z = \ln\frac{p}{1-p} \Leftrightarrow p = \sigma(z)$ |
| BCE through sigmoid | $\frac{\partial L}{\partial z} = p - y$ |
| Tanh | $\tanh z = 2\sigma(2z) - 1, \; \tanh' = 1 - \tanh^2$ |
| ReLU | $\max(0, z)$, derivative $\mathbb{1}[z > 0]$ |
| Sign-preserving ReLU pair | $z = \text{ReLU}(z) - \text{ReLU}(-z)$ |
| Leaky ReLU / PReLU | $z$ if $z \ge 0$, $\alpha z$ otherwise ($\alpha$ fixed / learned) |
| ELU | $z$ if $z > 0$, $\alpha(e^z - 1)$ otherwise; $\text{ELU}' = \alpha e^z$ for $z<0$ |
| SELU | $\lambda z$ / $\lambda\alpha(e^z - 1)$, $\alpha \approx 1.6733, \lambda \approx 1.0507$ |
| SELU init | $\mathbb{E}[w] = 0, \; \operatorname{Var}(w) = 1/n$ |
| Inverted dropout variance | $\operatorname{Var}(\text{out}) = \operatorname{Var}(\text{in})/q$ |
| Residual variance | $\operatorname{Var}(x + F) = \operatorname{Var}(x) + \operatorname{Var}(F) + 2\operatorname{Cov}(x, F)$ |
| Softplus | $\log(1 + e^z), \; \text{Softplus}' = \sigma(z)$ |
| GELU | $z\Phi(z), \; \text{GELU}' = \Phi(z) + z\phi(z), \; \text{GELU}'(0) = 0.5$ |
| SiLU | $z\sigma(z), \; \text{SiLU}' = \sigma(z) + z\sigma(z)(1 - \sigma(z))$ |
| SiLU negative tail | $\text{SiLU}'(z) \approx (1+z)e^z \to 0$ |
| Swish | $z\sigma(\beta z)$; $\beta = 0$: $z/2$; $\beta = 1$: SiLU; $\beta \to \infty$: ReLU |
| GLU | $h = u\,\sigma(g), \; u = xW_u, \; g = xW_g$ |
| GLU gradients | $\partial h/\partial u = \sigma(g), \; \partial h/\partial g = u\sigma(g)(1 - \sigma(g))$ |
| Gated FFN | $y = (xW_u \odot f(xW_g))W_{\text{out}}$ |
| SwiGLU | $h = u \odot g \odot \sigma(g)$ |
| Parameter match | $3dm_{\text{SwiGLU}} = 2dm_{\text{conv}} \Rightarrow m_{\text{SwiGLU}} = \tfrac23 m_{\text{conv}} \approx 8d/3$ |

---

## 11. Cheat Sheet & Recall Aids

**The story in one line per stage**
- **Need:** affine layers collapse; nonlinearity *inside* the network is required (XOR).
- **Sigmoid/tanh:** smooth but saturate both tails; $0.25^{10} \approx 10^{-6}$.
- **ReLU:** derivative 1 on the positive side, but a flat negative half-line kills neurons.
- **Leaky/PReLU/ELU/SELU/Softplus:** five ways to repair the negative side.
- **GELU/SiLU/Swish:** input × its own smooth gate.
- **GLU family:** input projection × a *separate* projection's gate; SwiGLU is the modern LLM default.

**Mnemonic for the evolution:** "**S**ome **T**rain **R**eally **L**ong, **E**ventually **G**ating **S**olves **G**radients" — Sigmoid, Tanh, ReLU, Leaky/PReLU, ELU/SELU, GELU, SiLU/Swish, GLU/SwiGLU.

**Gotchas**
- Sigmoid at the output is fine because BCE gives $\partial L/\partial z = p - y$; inside hidden layers it saturates.
- Nonlinear output ≠ nonlinear boundary: $\sigma(z) > 0.5 \iff z > 0$.
- PReLU with $\alpha = 1$ is the identity; ELU with $\alpha = 1$ is still curved (and $C^1$ at zero).
- ELU moves the mean *toward* zero; it does not subtract it. SELU self-normalizes only under its assumptions (LeCun init, AlphaDropout, no residual/attention).
- GELU's weight $\Phi(z)$ is a probability; the output $z\Phi(z)$ is not. There is no randomness at run time.
- GELU and SiLU are smooth but **not monotonic** (dip below zero); their derivatives can exceed 1.
- Self-gating (SiLU) = the gate is the same score; separate gating (GLU) = a different projection.
- In GLU, only the gate path pays the sigmoid derivative; a closed gate still blocks the content gradient.
- SwiGLU can amplify and flip signs; "gate" means learned multiplier, not on/off.
- SwiGLU's hidden width is $\tfrac23$ of the conventional one, because it has three matrices.
- ReLU' at 0 is a convention (PyTorch: 0); it does not matter in practice.

**Quick intuition**
- Weights decide *what* a neuron looks for; the activation decides *how* it responds.
- ReLU networks are piecewise affine: one affine map per activation pattern.
- Forward function decides what passes; its derivative decides how easily that can change.

**Recall checklist**
- [ ] Derive $W_{\text{eff}}, b_{\text{eff}}$ and the XOR ReLU construction.
- [ ] Derive $\sigma' = \sigma(1-\sigma)$, sigmoid from log-odds, and $\partial L/\partial z = p - y$.
- [ ] Explain dead ReLU with the chain rule, and three ways a neuron can recover.
- [ ] State ELU, SELU (constants), Softplus and their derivatives.
- [ ] Explain GELU's switch story and why it is non-monotonic.
- [ ] Compute $\text{SiLU}'(z)$ at $-2, 0, 2$.
- [ ] Write both GLU weight gradients and say which contains $\sigma'$.
- [ ] Derive $m_{\text{SwiGLU}} = \tfrac23 m$.
