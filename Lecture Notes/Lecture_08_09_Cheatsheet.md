---
type: cheatsheet
course: '[[_CS5326 Advanced GenAI|CS5326 Advanced GenAI]]'
source:
- '[[005-Activations.pdf]]'
- '[[006-FFN-part 1.pdf]]'
tags:
- course/cs5326
---

# Cheatsheet: Activations (L8) and FFNs, Dense and MoE (L9)

*Exam cheatsheet for [[Lecture_08_Activations]] and [[Lecture_09_FFNs_Dense_and_MoE]]; derivations live there.*

## Topic Inventory

- 1. Why Nonlinearity
  - 1.1 Neuron and affine collapse
  - 1.2 XOR
  - 1.3 Output-only nonlinearity
- 2. What an Activation Needs
  - 2.1 Required vs not required
  - 2.2 Gradient chain and vanishing/exploding/blocked
- 3. Sigmoid and Tanh
  - 3.1 Sigmoid
  - 3.2 Log-odds and BCE
  - 3.3 Tanh
- 4. ReLU
  - 4.1 Definition and threshold view
  - 4.2 Sign-preserving pair and piecewise-affine networks
  - 4.3 Dead ReLU and controls
- 5. Fixing the Negative Side
  - 5.1 Leaky ReLU and PReLU
  - 5.2 ELU
  - 5.3 SELU, dropout and residuals
  - 5.4 Softplus
- 6. Self-Gated: GELU, SiLU, Swish
  - 6.1 GELU
  - 6.2 SiLU and Swish
  - 6.3 Non-monotonic dip and vanishing tails
  - 6.4 Comparison table
- 7. Separately Gated: GLU Family
  - 7.1 GLU and its gradients
  - 7.2 ReGLU, GEGLU, SwiGLU
  - 7.3 SwiGLU width
- 8. Choosing an Activation
- 9. The Position-Wise FFN
- 10. Dense Models and MoE Motivation
  - 10.1 Dense limitation
  - 10.2 Learn better vs scale better
  - 10.3 Gradient interference
- 11. The |x| Two-Expert Example
- 12. MoE in LLMs
  - 12.1 Components and dense vs sparse
  - 12.2 What experts learn
  - 12.3 Routes through layers
- 13. Routing
- Formula Summary Sheet
- Cheat Sheet & Recall Aids

---

## 1. Why Nonlinearity

### 1.1 Neuron and affine collapse

$$
z = \sum_i w_i x_i + b, \qquad a = \phi(z)
$$

$$
(xW_1 + b_1)W_2 + b_2 = x\underbrace{W_1W_2}_{W_{\text{eff}}} + \underbrace{b_1W_2 + b_2}_{b_{\text{eff}}}
$$

Any stack of affine layers = one affine layer. Depth without $\phi$ buys nothing.

### 1.2 XOR

| $x_1, x_2$ | $h_1 = \text{ReLU}(x_1+x_2)$ | $h_2 = \text{ReLU}(x_1+x_2-1)$ | $y = h_1 - 2h_2$ |
|---|---|---|---|
| 0, 0 | 0 | 0 | 0 |
| 0, 1 | 1 | 0 | 1 |
| 1, 0 | 1 | 0 | 1 |
| 1, 1 | 2 | 1 | 0 |

| Input space                       | Hidden space                       |
| --------------------------------- | ---------------------------------- |
| ![[L08_xor_input_space.png\|260]] | ![[L08_xor_hidden_space.png\|260]] |

Not linearly separable in input space; separable after the ReLU layer.

### 1.3 Output-only nonlinearity

$\sigma(z) > 0.5 \iff z > 0$ → boundary still linear. Nonlinearity must be **inside**.

---

## 2. What an Activation Needs

### 2.1 Required vs not required

| Needed | Not needed (counterexample) |
|---|---|
| Nonlinearity (somewhere) | Bounded (ReLU) |
| Usable derivatives (isolated kinks OK) | Monotonic (GELU, SiLU) |
| Useful gradient magnitudes (desirable) | Smooth everywhere (ReLU) |
| Manageable activation magnitudes | Nonnegative (tanh, GELU) |
| Cheap compute | Zero-centred (sigmoid, ReLU) |
| Right output range (for outputs/gates) | |

Step $\mathbb{1}[z>0]$: nonlinear but derivative 0 a.e. → untrainable by plain backprop (needs straight-through estimator).

### 2.2 Gradient chain

$$
\frac{\partial L}{\partial w_i} = \underbrace{\frac{\partial L}{\partial a}}_{\text{incoming error}} \cdot \underbrace{\phi'(z)}_{\text{sensitivity}} \cdot \underbrace{x_i}_{\text{input}}
$$

| Product of factors across layers | Effect |
|---|---|
| repeatedly $< 1$ | **vanishing**: early layers barely learn |
| repeatedly $> 1$ | **exploding** |
| $= 0$ | **blocked** path |

Forward function decides what passes; derivative decides how easily that can change.

---

## 3. Sigmoid and Tanh

### 3.1 Sigmoid

![[L08_sigmoid.png|320]]

$$
\sigma(z) = \frac{1}{1+e^{-z}}, \qquad \sigma'(z) = \sigma(z)(1-\sigma(z)) \le 0.25 \;(\text{max at } z=0)
$$

| $z$ | $-5$ | 0 | 5 | $\pm10$ |
|---|---|---|---|---|
| $\sigma$ | 0.007 | 0.5 | 0.993 | ≈0 / ≈1, $\sigma' \approx 0$ |

- Range $(0,1)$, mean ≈ 0.5 (not zero-centred), saturates both tails.
- 10 layers at best case: $0.25^{10} \approx 9.54\times10^{-7}$.
- Use: binary outputs, gates. Poor default for deep hidden layers.

> [!warning] Source inconsistency
> Closing summary slide says sigmoid range $[0,1]$; correct is $(0,1)$.

### 3.2 Log-odds and BCE

$$
\text{odds} = \frac{p}{1-p}, \qquad z = \ln\frac{p}{1-p} \iff p = \sigma(z)
$$

$$
L = -[y\ln p + (1-y)\ln(1-p)] \;\Rightarrow\; \boxed{\frac{\partial L}{\partial z} = p - y}
$$

$\sigma'$ cancels → sigmoid output doesn't saturate learning. E.g. $z=-2$, $y=1$: $p=0.119$, $\partial L/\partial z = -0.881$, vs hidden $\sigma'(-2) = 0.105$.

### 3.3 Tanh

![[L08_tanh.png|320]]

$$
\tanh z = \frac{e^z - e^{-z}}{e^z + e^{-z}} = 2\sigma(2z) - 1, \qquad \tanh' = 1 - \tanh^2 \;(=1 \text{ at } 0)
$$

- Range $(-1,1)$, $\tanh(0)=0$, zero-centred for symmetric inputs.
- Still saturates both tails: $\tanh'(1)^{10} = 0.42^{10} \approx 1.7\times10^{-4}$.
- Use: bounded signed outputs, RNN state.

> [!warning] Source inconsistency
> Ng slide calls tanh a "shifted" sigmoid; it is scaled **and** shifted ($2\sigma(2z)-1$).

---

## 4. ReLU

### 4.1 Definition and threshold view

![[L08_relu.png|320]]

$$
\text{ReLU}(z) = \max(0,z), \qquad \text{ReLU}'(z) = \begin{cases} 0 & z<0 \\ 1 & z>0 \end{cases} \;(\text{at } 0: \text{convention, PyTorch } 0)
$$

- Range $[0,\infty)$ (slide says $(0,\infty)$: wrong at 0).
- Fires when $xw > -b$: weights = direction of evidence, bias = threshold. Negative **input** can still pass ($x=-5, w=-1 \Rightarrow z=5$).
- Strengths: cheap, sparse (exact zeros), derivative 1 on positive side, no positive saturation.

### 4.2 Sign-preserving pair and piecewise-affine networks

$$
z = \text{ReLU}(z) - \text{ReLU}(-z), \qquad |z| = \text{ReLU}(z) + \text{ReLU}(-z)
$$

- Fixed activation pattern → affine; across patterns → nonlinear. Regions can grow exponentially with depth. ([[005-ReLU-Boundary-Explorer.html|ReLU Boundary Explorer]])

### 4.3 Dead ReLU and controls

$$
z<0 \Rightarrow \frac{\partial L}{\partial w_i} = \frac{\partial L}{\partial a}\cdot 0 \cdot x_i = 0, \quad \frac{\partial L}{\partial b} = 0 \Rightarrow w_{\text{new}} = w
$$

- **Dead:** $z<0$ for every input (e.g. $z = x-2$, $x\in[0,1]$). Causes: bad init, one big update.
- **Deadish, not always forever:** earlier layers, momentum, weight decay can revive it.
- **Controls:** He init ($\text{Var}(w) = 2/n$), sensible LR, monitor inactive units, nonzero negative slope.
- The real problem is the **whole flat negative half-line**, not the kink at 0.

---

## 5. Fixing the Negative Side

### 5.1 Leaky ReLU and PReLU

| Leaky ReLU | PReLU |
|---|---|
| ![[L08_leaky_relu.png\|260]] | ![[L08_prelu.svg\|260]] |

$$
\text{LeakyReLU}(z) = \begin{cases} z & z \ge 0 \\ \alpha z & z<0 \end{cases}, \quad \alpha = 0.01 \text{ fixed}; \quad \text{LeakyReLU}(-3) = -0.03
$$

| | Leaky ReLU | PReLU (He et al., 2015) |
|---|---|---|
| $\alpha$ | fixed small | **learned**, often per channel |
| Plus | never fully silent; signed | learns negative response |
| Minus | weak gradients; no exact sparsity; hand-picked $\alpha$ | no generalisation guarantee; $\alpha$ may drift |
| $\alpha = 1$ | – | **identity** → layers collapse (only if all activations between them do) |

### 5.2 ELU

![[L08_elu.png|320]]

$$
\text{ELU}(z) = \begin{cases} z & z>0 \\ \alpha(e^z-1) & z \le 0 \end{cases}, \qquad \text{ELU}'(z) = \alpha e^z \;(z<0)
$$

- Negative saturation at $-\alpha$; $\alpha=1$ still curved (fixes PReLU's collapse) and $C^1$ at 0.
- Moves mean **toward** 0 (doesn't subtract it): inputs $\{-1,+1\}$ → ReLU mean 0.5, ELU mean 0.184.
- Very negative $z$: output ≈ $-1$, slope $e^z$ tiny ($e^{-5} = 0.0067$) → noise-robust but slow learning; costs an exp.

### 5.3 SELU, dropout and residuals

![[L08_selu.png|320]]

$$
\text{SELU}(z) = \lambda\begin{cases} z & z>0 \\ \alpha(e^z-1) & z \le 0 \end{cases}, \quad \alpha \approx 1.6733, \; \lambda \approx 1.0507
$$

- Self-normalising toward mean 0, var 1 **only with**: LeCun init $\mathbb{E}[w]=0, \text{Var}(w)=1/n$; standardised inputs; plain feed-forward stack; **AlphaDropout**.
- Inverted dropout: $\text{Var}(\text{out}) = \text{Var}(\text{in})/q$ ($q=0.8 \to 1.25$). AlphaDropout sets dropped units to $-\lambda\alpha$ then rescales/shifts.
- Residuals break it: $\text{Var}(x+F) = \text{Var}(x) + \text{Var}(F) + 2\text{Cov}(x,F)$ (= 2 if uncorrelated). Attention also mixes. → SELU can't replace LayerNorm/RMSNorm in Transformers.

### 5.4 Softplus

![[L08_softplus.png|351]]

$$
\text{Softplus}(z) = \log(1+e^z), \qquad \text{Softplus}'(z) = \sigma(z)
$$

Smooth ReLU, always positive; $z\gg0 \Rightarrow \approx z$, $z\ll0 \Rightarrow \approx 0$. Use: positive output parameters.

---

## 6. Self-Gated: GELU, SiLU, Swish

### 6.1 GELU

![[L08_gelu.png|331]]

$$
\text{GELU}(z) = z\,\Phi(z), \qquad \text{GELU}'(z) = \Phi(z) + z\,\phi(z), \qquad \text{GELU}'(0) = 0.5
$$

($\Phi$ = standard normal CDF, $\phi$ = density.) Approximations: $\tfrac12 z(1+\tanh[\sqrt{2/\pi}(z+0.044715z^3)])$, or $z\sigma(1.702z)$.

- **Switch story:** keep $z$ with prob $\Phi(z)$, else 0; GELU returns the **expectation** $z\Phi(z)$ deterministically. $\Phi$ is a probability; $z\Phi(z)$ is not. $\text{GELU}(1) = 0.841$, $\text{GELU}(2) = 1.954$.
- Min ≈ $-0.170$ at $z \approx -0.75$. Few exact zeros; costs more than ReLU. Used in BERT, GPT-2.

### 6.2 SiLU and Swish

| SiLU | Swish |
|---|---|
| ![[L08_silu.png\|260]] | ![[L08_swish.png\|260]] |

$$
\text{SiLU}(z) = z\,\sigma(z), \qquad \text{SiLU}'(z) = \sigma(z) + z\,\sigma(z)(1-\sigma(z))
$$

$$
\text{Swish}_\beta(z) = z\,\sigma(\beta z): \quad \beta=0 \to z/2, \quad \beta=1 \to \text{SiLU}, \quad \beta\to\infty \to \text{ReLU}
$$

- **Self-gating:** the same score is both value and gate. Why multiply by $z$: sigmoid alone saturates at 1 for large $z$; $z\sigma(z) \approx z$ keeps growing.

### 6.3 Non-monotonic dip and vanishing tails

| $z$ | $-2$ | 0 | 2 |
|---|---|---|---|
| $\text{SiLU}'(z)$ | $-0.091$ | 0.5 | 1.091 |

- $-2 \to -1$: SiLU $-0.238 \to -0.269$ (gate opens faster than magnitude shrinks). Min ≈ $-0.278$ at $z\approx-1.28$.
- Derivative can exceed 1: raising $z$ raises both value and gate.
- Tail: $\text{SiLU}'(z) \approx (1+z)e^z \to 0$ → small gradients remain (no flat half-line though).

> [!warning] Source inconsistency
> Slide calls SiLU "monotonically increasing"; the dip and $\text{SiLU}'(-2) < 0$ show it isn't.

### 6.4 Comparison table

| $z$ | Sigmoid | Tanh | ReLU | GELU | SiLU |
|---|---|---|---|---|---|
| $-2$ | 0.119 | $-0.964$ | 0 | $-0.046$ | $-0.238$ |
| $-1$ | 0.269 | $-0.762$ | 0 | $-0.159$ | $-0.269$ |
| 0 | 0.500 | 0 | 0 | 0 | 0 |
| 1 | 0.731 | 0.762 | 1 | 0.841 | 0.731 |
| 2 | 0.881 | 0.964 | 2 | 1.954 | 1.762 |

---

## 7. Separately Gated: GLU Family

### 7.1 GLU and its gradients

$$
u = xW_u, \quad g = xW_g \;(\text{parallel, same input}), \qquad \boxed{h = u\,\sigma(g)}
$$

$$
\frac{\partial h}{\partial u} = \sigma(g), \qquad \frac{\partial h}{\partial g} = u\,\sigma(g)(1-\sigma(g))
$$

| $g$ | $\sigma(g)$ = content sensitivity | gate sensitivity ($u=1$) |
|---|---|---|
| 5 (open) | 0.9933 | 0.0066 |
| 0 | 0.5 | 0.25 |
| −5 (closed) | 0.0067 | 0.0066 |

- Only the **gate** path pays $\sigma'$; content path is linear → why sigmoid is OK here. Closed gate still blocks content.
- Self-gating (SiLU) vs separate gating (GLU): own score vs a different projection.

### 7.2 ReGLU, GEGLU, SwiGLU (Shazeer, 2020)

$$
\text{Gated FFN: } y = \big(xW_u \odot f(xW_g)\big)W_{\text{out}} \quad (\text{no biases, no activation after } W_{\text{out}})
$$

![[L08_gated_ffns.png|560]]

| Variant | $h$ | Multiplier allows |
|---|---|---|
| GLU | $u\odot\sigma(g)$ | $(0,1)$: attenuate |
| ReGLU | $u\odot\text{ReLU}(g)$ | $[0,\infty)$: suppress/amplify; dead region |
| GEGLU | $u\odot\text{GELU}(g)$ | smooth; amplify; sign flip |
| **SwiGLU** | $u\odot g\odot\sigma(g)$ | smooth; amplify; sign flip |

SwiGLU with $u=4$: $g=-2 \to -0.954$ (flip), $g=0 \to 0$ (GLU gives 2), $g=2 \to 7.046$ (amplify). Used in LLaMA, PaLM.

### 7.3 SwiGLU width

$$
2d\,m_{\text{conv}} = 3d\,m_{\text{SwiGLU}} \;\Rightarrow\; \boxed{m_{\text{SwiGLU}} = \tfrac23 m_{\text{conv}} = \tfrac83 d}
$$

LLaMA-7B: $d = 4096$, $\tfrac23\cdot16384 = 10923 \to 11008$ (multiple of 256), params $1.008\times$ conventional.

---

## 8. Choosing an Activation

| Output task | Activation |
|---|---|
| Regression | Identity |
| Binary / multilabel | Sigmoid / separate sigmoids |
| Multiclass | Softmax $e^{z_k}/\sum_j e^{z_j}$ |
| Positive parameter | Softplus / exp |
| Bounded signed | Scaled tanh |

| Hidden situation | Start with |
|---|---|
| Baseline | ReLU |
| Dying units | Leaky ReLU |
| Learnable negative slope | PReLU |
| Smooth FFN | GELU / SiLU |
| Modern LLM FFN | SwiGLU / GEGLU |
| Self-normalising design | SELU + its setup |
| Pretrained model | keep its activation |

Extras in summary slide: Softsign $z/(1+|z|)$; Hard sigmoid $\max(0,\min(1, z/6+1/2))$; Mish $z\tanh(\text{Softplus}(z))$.

![[L08_activation_summary.png|420]]

---

## 9. The Position-Wise FFN

![[L09_ffn_block.png|150]]

$$
\text{FFN}(x) = \max(xW_1+b_1, 0)\,W_2 + b_2
$$

$$
\underbrace{x}_{1\times d} \xrightarrow{\text{expand}} \underbrace{h}_{1\times m} \xrightarrow{\phi} \underbrace{a}_{1\times m} \xrightarrow{\text{project}} \underbrace{y}_{1\times d}; \qquad h = \phi(xW_1+b_1), \; y = hW_2 + b_2
$$

| Choice | Why |
|---|---|
| Expand $d\to m$ ($m = 4d$) | more feature combinations |
| $\phi$ | else $W_1W_2$ collapse |
| Back to $d$ | residual add + next layer |

- **Attention: between tokens. FFN: within each token.** Same weights at every position; can't see other tokens directly.
- Params $2dm + m + d$; $d=512, m=2048 \Rightarrow 2{,}099{,}712$; FLOPs ≈ $4dm = 4.19$M/token.

---

## 10. Dense Models and MoE Motivation

### 10.1 Dense limitation

Every token uses every parameter. Plus: regular, hardware-friendly. Minus: bigger model = more capacity **and** more compute per token. → *Can we add parameters without every token using them?*

### 10.2 Learn better vs scale better

| Motivation | Mechanism |
|---|---|
| 1. **Learn better** (Jacobs et al., 1991) | separate experts → specialisation, less interference |
| 2. **Scale better** (sparse MoE) | many params, few active per token |

Dense FFN → **Dense MoE** (all experts run, mixed) → **Sparse MoE** (top-$k$ run).

### 10.3 Gradient interference

$$
\nabla_\theta L_{\text{math}}^{\top}\,\nabla_\theta L_{\text{trans}} < 0 \;\Rightarrow\; \Delta L_{\text{trans}} \approx -\eta\,\nabla L_{\text{trans}}^\top\nabla L_{\text{math}} > 0
$$

Different experts → less sharing → potentially less interference → better specialisation (possible, **not guaranteed**).

> [!warning] Source inconsistency
> Slide: "one is negative while the other is positive". It's the **dot product** of gradient vectors that is negative (angle > 90°).

Specialist analogy: consult the right specialist(s). Experts aren't assigned human subjects; specialisation is learned.

---

## 11. The |x| Two-Expert Example

| Approach | Model | Works? |
|---|---|---|
| Linear | $\hat y = ax+b$ | No: needs slopes +1 and −1 |
| Dense ReLU | $\text{ReLU}(x)+\text{ReLU}(-x)$ | Yes, exactly |
| MoE | $E_1 = x$, $E_2 = -x$, $p_1 = \mathbb{1}[x\ge0]$, $p_2 = 1-p_1$ | Yes |

$$
\boxed{y = p_1(x)E_1(x) + p_2(x)E_2(x)}
$$

- Dense is **not less expressive**; MoE's gain is structural: decomposition + learned selection, no interference, one expert runs.
- Gate and experts trained **jointly**; nobody labels the source.
- Routing helps only if **features reveal the source**.

---

## 12. MoE in LLMs

### 12.1 Components and dense vs sparse

- **Experts:** FFNs (usually whole FFNs), own set per layer. **Router / gate network:** picks experts per token. Expert ≠ an LLM.

![[L09_router_mechanism.png|220]]

| Dense FFN (512 → 2048 → 512) | Dense: all params active | Sparse: subset active |
|---|---|---|
| ![[L09_dense_ffnn_expansion.png\|260]] | ![[L09_dense_model.png\|200]] | ![[L09_sparse_model.png\|200]] |

### 12.2 What experts learn

Not domains; **specific tokens in specific contexts**.

![[L09_experts_syntax.png|420]]

| Model | Finding |
|---|---|
| Switch Transformer (encoder, top-1) | experts for punctuation, conjunctions/articles, verbs, visual descriptions, numbers |
| Mixtral 8x7B (decoder, 8 experts, top-2) | no domain specialism, but **consistent** per token type / syntax |

![[L09_expert_specialization_switch.png|420]]

![[L09_mixtral_expert_colouring.png|300]]

### 12.3 Routes through layers

![[L09_expert_architecture.png|460]]

![[L09_moe_routes.png|500]]

Different tokens → different paths; decoder block's single FFN replaced by $N$ FFNs.

---

## 13. Routing

1. Router scores experts from the token's representation.
2. Select a few (1–2).
3. Only those run.
4. Combine with routing weights.

$$
y = 0.7\,E_2(x) + 0.3\,E_5(x), \qquad p(x) = \text{softmax}(xW_r), \; W_r \in \mathbb{R}^{d\times N}
$$

- Output = expert × gate value (slide: $p = 0.45 \Rightarrow 0.45\,E_1(x)$); the multiplication gives the router gradient.
- Capacity grows with $N$, per-token compute with $k$. E.g. 8 × 2.1M experts, top-2 → 16.8M total, 4.2M active (25%), router $512\times8 = 4096$.
- Logits $(1.2, 0.3, -1.0, 0.8)$ → $p = (0.457, 0.186, 0.051, 0.306)$; top-2 renormalised → 0.599 / 0.401 *(renormalisation is Mixtral's rule, not on these slides)*.

---

## Formula Summary Sheet

| Concept | Formula |
|---|---|
| Neuron | $z = \sum_i w_ix_i + b$, $a = \phi(z)$ |
| Affine collapse | $W_{\text{eff}} = W_1W_2$, $b_{\text{eff}} = b_1W_2 + b_2$ |
| XOR net | $h_1 = \text{ReLU}(x_1+x_2)$, $h_2 = \text{ReLU}(x_1+x_2-1)$, $y = h_1 - 2h_2$ |
| Weight gradient | $\partial L/\partial w_i = (\partial L/\partial a)\,\phi'(z)\,x_i$ |
| Sigmoid | $1/(1+e^{-z})$; $\sigma' = \sigma(1-\sigma) \le 0.25$ |
| Log-odds | $z = \ln\frac{p}{1-p}$ |
| BCE gradient | $\partial L/\partial z = p - y$ |
| Tanh | $2\sigma(2z)-1$; $\tanh' = 1-\tanh^2$ |
| ReLU | $\max(0,z)$; $\mathbb{1}[z>0]$ |
| ReLU identities | $z = \text{ReLU}(z) - \text{ReLU}(-z)$; $\lvert z\rvert = \text{ReLU}(z) + \text{ReLU}(-z)$ |
| He init | $\text{Var}(w) = 2/n$ |
| Leaky / PReLU | $z$ or $\alpha z$ |
| ELU | $\alpha(e^z-1)$ for $z\le0$; slope $\alpha e^z$ |
| SELU | $\lambda\,\text{ELU}_\alpha$, $\alpha = 1.6733$, $\lambda = 1.0507$; init $\text{Var}(w) = 1/n$ |
| Dropout variance | $\text{Var}(\text{in})/q$ |
| Residual variance | $\text{Var}(x)+\text{Var}(F)+2\text{Cov}(x,F)$ |
| Softplus | $\log(1+e^z)$; derivative $\sigma(z)$ |
| GELU | $z\Phi(z)$; $\Phi(z) + z\phi(z)$ |
| SiLU | $z\sigma(z)$; $\sigma + z\sigma(1-\sigma)$; tail $(1+z)e^z$ |
| Swish | $z\sigma(\beta z)$ |
| GLU | $h = u\,\sigma(g)$; $\partial h/\partial u = \sigma(g)$, $\partial h/\partial g = u\sigma(g)(1-\sigma(g))$ |
| Gated FFN | $y = (xW_u \odot f(xW_g))W_{\text{out}}$ |
| SwiGLU | $u \odot g \odot \sigma(g)$ |
| SwiGLU width | $m' = \tfrac23 m = \tfrac83 d$ |
| FFN | $h = \phi(xW_1+b_1)$, $y = hW_2+b_2$ |
| FFN params | $2dm + m + d$ |
| Gradient conflict | $\nabla L_A^\top \nabla L_B < 0$ |
| Two-expert MoE | $y = p_1E_1 + p_2E_2$, $p_2 = 1-p_1$ |
| Sparse MoE | $y = \sum_{i\in\text{top-}k} p_i(x)E_i(x)$ |
| Router | $p = \text{softmax}(xW_r)$ |

---

## Cheat Sheet & Recall Aids

**Mnemonics**
- Activation evolution: "**S**ome **T**rain **R**eally **L**ong, **E**ventually **G**ating **S**olves **G**radients" = Sigmoid, Tanh, ReLU, Leaky/PReLU, ELU/SELU, GELU, SiLU/Swish, GLU/SwiGLU.
- MoE per-token recipe: "**R**oute, **S**elect, **R**un, **M**ix".
- SELU's four conditions: "**L**eCun **S**tandardised **F**eed-forward **A**lpha" (LeCun init, standardised inputs, feed-forward only, AlphaDropout).

**Gotchas**
- Sigmoid OK at output (BCE gives $p-y$), bad in hidden layers (saturates).
- Nonlinear output ≠ nonlinear boundary.
- Tanh is zero-centred but still saturates.
- ReLU range is $[0,\infty)$; the kink is harmless, the flat negative half-line is the problem.
- PReLU $\alpha=1$ → identity; ELU $\alpha=1$ → still curved, smooth at 0.
- ELU moves mean toward 0, doesn't centre it.
- GELU: $\Phi(z)$ is a probability, $z\Phi(z)$ isn't; no runtime randomness.
- GELU and SiLU are smooth but **non-monotonic**; derivatives can exceed 1.
- In GLU only the gate path carries $\sigma'$.
- SwiGLU "gate" can amplify and flip sign; width is $\tfrac23$ because 3 matrices.
- FFN: within token; attention: between tokens.
- Dense isn't less expressive than MoE.
- Gradient conflict = negative **dot product**.
- Experts ≠ topics; decoder experts track syntax/token type.
- Gate value multiplies output → router gets gradient.

**If you remember nothing else**
- Nonlinearity prevents collapse; its derivative controls learning.
- ReLU fixed saturation, created dead units; Leaky/ELU fix the negative side; GELU/SiLU smooth it; GLU separates the gate; SwiGLU is today's LLM FFN.
- MoE = many FFN experts + softmax router; capacity scales with $N$, compute with $k$.

**Recall checklist**
- [ ] Write $W_{\text{eff}}, b_{\text{eff}}$ and the XOR net.
- [ ] $\sigma'$, $\tanh'$, BCE $p-y$, $0.25^{10}$.
- [ ] Why a dead ReLU never updates; three controls.
- [ ] ELU, SELU constants, SELU's conditions, dropout $1/q$.
- [ ] GELU, SiLU, Swish formulas + derivatives at 0.
- [ ] GLU two gradients; SwiGLU $\tfrac23$.
- [ ] FFN shapes and params for $d=512$.
- [ ] Two MoE motivations; $|x|$ example; Switch vs Mixtral findings.
- [ ] Router softmax for given logits; total vs active params.
