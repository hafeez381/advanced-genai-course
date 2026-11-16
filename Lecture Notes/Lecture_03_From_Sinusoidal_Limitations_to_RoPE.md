---
type: lecture
course: '[[_CS5326 Advanced GenAI|CS5326 Advanced GenAI]]'
source:
- '[[001-PositionAndOrdering-3.pdf]]'
tags:
- course/cs5326
---

%% nav:start %%
[[_CS5326 Advanced GenAI|CS5326 Advanced GenAI]] · ← [[Lecture_02_Sinusoidal_Positional_Encoding]] · [[Lecture_04_Long_Context_Extension]] →
Concepts: [[Rotary Position Embedding]], [[Sinusoidal Positional Encoding]]
%% nav:end %%

# From Sinusoidal Limitations to RoPE (CS5326 — Advanced GenAI and Agents, Lecture 3)

*Deck slides 80-121. Lecture 2 built sinusoidal encoding in isolation. This lecture asks how it is actually used inside the Transformer; the answer exposes a problem that RoPE (rotary positional embedding) fixes.*

## Topic Inventory

- [[#1. How Sinusoidal Encoding Enters the Model|1. How Sinusoidal Encoding Enters the Model]]
- [[#2. The Entanglement Problem|2. The Entanglement Problem]]
- [[#3. Rotary Position Embeddings (RoPE)|3. Rotary Position Embeddings (RoPE)]]
  - [[#3.1 The design goal|3.1 The design goal]]
  - [[#3.2 The key insight: reorder the operations|3.2 The key insight: reorder the operations]]
  - [[#3.3 The rotation operation in two dimensions|3.3 The rotation operation in two dimensions]]
  - [[#3.4 Why rotation gives pure relative position|3.4 Why rotation gives pure relative position]]
  - [[#3.5 The same result through matrix algebra|3.5 The same result through matrix algebra]]
  - [[#3.6 How exactly is this better than sinusoidal?|3.6 How exactly is this better than sinusoidal?]]
- [[#4. Scaling RoPE to Full Dimensions|4. Scaling RoPE to Full Dimensions]]
- [[#5. Does Rotation Corrupt Semantic Meaning?|5. Does Rotation Corrupt Semantic Meaning?]]
- [[#6. Limitations of RoPE|6. Limitations of RoPE]]
  - [[#6.1 Aliasing|6.1 Aliasing]]
  - [[#6.2 Encoding a position is not the same as using it|6.2 Encoding a position is not the same as using it]]
  - [[#6.3 Attention dilution in long contexts|6.3 Attention dilution in long contexts]]
  - [[#6.4 What a context-extension method must fix|6.4 What a context-extension method must fix]]

---

## 1. How Sinusoidal Encoding Enters the Model

**Two separate properties of sin/cos.** Both need sine and cosine, but for different reasons:
- **Dot product property:** the dot product of two encodings is $\cos(\theta(m-n))$, which depends only on relative position. This matters inside attention, where $q \cdot k$ is computed.
- **Linear shift property:** a fixed $M_k$ maps $PE_{pos} \to PE_{pos+k}$ (Lecture 2). This matters for linear layers like $W_q, W_k$.

The sin/cos pair was designed in [[Lecture_02_Sinusoidal_Positional_Encoding#8. Why Both Sine and Cosine?|Lecture 2 §8]] for the linear shift property. The dot product property falls out of the same geometry, since both encodings are points on the same circle.

**Deriving the dot product property.** With unit vectors $\mathbf p_m = (\sin m\theta, \cos m\theta)$ and $\mathbf p_n = (\sin n\theta, \cos n\theta)$, applying $\cos(A-B) = \cos A\cos B + \sin A\sin B$:

$$\mathbf p_m^\top\mathbf p_n = \sin m\theta\sin n\theta + \cos m\theta\cos n\theta = \boxed{\cos(\theta(m-n))}$$

Since both have norm 1, this is also their cosine similarity. For the full encoding, $PE_m^\top PE_n = \sum_i\cos(\theta_i(m-n))$. The absolute positions vanish.

**Position is added to the embedding:**

$$\text{input}_m = \text{embed}_m + PE_m$$
$$Q_m = W_q\,\text{embed}_m + W_q\,PE_m,\qquad K_n = W_k\,\text{embed}_n + W_k\,PE_n$$

Each query and key now has a **semantic part** plus a **positional part**, and attention works on the sums.

---

## 2. The Entanglement Problem

**Four terms.** Multiplying the two sums:

$$
\begin{aligned}
Q_m\cdot K_n &= (W_q\,\text{embed}_m)\cdot(W_k\,\text{embed}_n) &&\text{Term 1: semantic × semantic}\\
&+ (W_q\,\text{embed}_m)\cdot(W_k\,PE_n) &&\text{Term 2: meaning of } m \text{ vs position of } n\\
&+ (W_q\,PE_m)\cdot(W_k\,\text{embed}_n) &&\text{Term 3: position of } m \text{ vs meaning of } n\\
&+ (W_q\,PE_m)\cdot(W_k\,PE_n) &&\text{Term 4: position × position}
\end{aligned}
$$

The clean $\cos(\theta(m-n))$ lives **only in Term 4**. Even that is not guaranteed after arbitrary $W_q, W_k$ projections, because the identity holds only for the unprojected dot product.

**Cross terms are a burden.** The model only sees the sum. Terms 2 and 3 are noise that mixes content with position, and nothing isolates Term 4, so the model must learn to use it on its own. $W_q, W_k$ end up doing two jobs: projecting meaning usefully and preserving positional structure.

**Absolute vs relative.** $PE_5$ always means "position 5". The model wants "these two are 9 apart", but relative position is never stored. It only appears as a byproduct in Term 4, after the dot product, so the model must learn it with no guarantee.

**Position is fused with meaning.** Once added, $\text{embed}_m + PE_m$ can't be separated: **addition removes the boundary**. Later layers that need only meaning or only position are stuck with the mixture.

| Weakness | What goes wrong |
|---|---|
| **Unstructured interaction** | Cross terms between content and absolute position |
| **No guaranteed relative-only score** | Relative relationships are expressible, but learned Q/K projections need not preserve them |
| **Additive fusion** | Content and position enter as a sum with no explicit separation |

**One root cause: position is added to the embedding.** What if we instead applied position as an **operation on Q and K**, just before attention?

---

## 3. Rotary Position Embeddings (RoPE)

RoPE (Su et al., 2021) is used in almost every modern LLM (LLaMA (Meta's family of open large language models), Mistral, Gemma, Phi).

### 3.1 The design goal

$$Q_m\cdot K_n = f(\text{semantics},\ m-n)$$

The score should depend on meaning and relative distance, and nothing else: no cross terms, no entanglement.

### 3.2 The key insight: reorder the operations

Sinusoidal: add position **first**, then multiply by $W_q, W_k$, which produces the cross terms. RoPE leaves the embedding alone: $W_q, W_k$ project first, then position is applied to the resulting Q and K. **Apply position after the projection, not before.** Now $W_q, W_k$ only handle meaning and the positional operation only handles position. The operation is **rotation**.

### 3.3 The rotation operation in two dimensions

For $Q = (q_0, q_1)$ at position $m$ with frequency $\theta$, rotate by $m\theta$:

$$\begin{bmatrix}q_0^{new}\\ q_1^{new}\end{bmatrix} = \begin{bmatrix}\cos m\theta & -\sin m\theta\\ \sin m\theta & \cos m\theta\end{bmatrix}\begin{bmatrix}q_0\\ q_1\end{bmatrix}$$

$$q_0^{new} = q_0\cos m\theta - q_1\sin m\theta,\qquad q_1^{new} = q_0\sin m\theta + q_1\cos m\theta$$

These are the same sin/cos values as before, but used to **rotate**, not add. Position 1 rotates by $\theta$, position 100 by $100\theta$. Length is unchanged; only direction encodes position. The key at position $n$ is rotated by $n\theta$ the same way.

### 3.4 Why rotation gives pure relative position

Let the query point at angle $\alpha$ and the key at angle $\beta$ (the semantic content produced by $W_q$, $W_k$). For unit vectors, $Q\cdot K = \cos(\alpha - \beta)$.

![[L03_rope_semantic_angle.png|500]]

*The dot product depends only on the angle between Q and K, $(\alpha - \beta)$: the "semantic angle".*

Rotation adds to the angles: $Q_m$ points at $\alpha + m\theta$ and $K_n$ at $\beta + n\theta$. So

$$Q_m\cdot K_n = \cos((\alpha + m\theta) - (\beta + n\theta)) = \boxed{\cos\big((\alpha-\beta) + (m-n)\theta\big)}$$

$(\alpha - \beta)$ is the untouched **semantic** relationship; $(m-n)\theta$ is a **purely relative** positional piece. This is exactly the goal in §3.1.

> [!note] Animations
> Deck slides 95 and 107 are embedded animations (black in the PDF) showing two vectors rotated by their positions with the angle between them fixed under a common shift.

### 3.5 The same result through matrix algebra

$$Q_m\cdot K_n = (R_mQ)^\top(R_nK) = Q^\top R_m^\top R_nK$$

Rotations are **orthogonal**, so $R_m^\top = R_m^{-1} = R_{-m}$, and angles add, so $R_{-m}R_n = R_{n-m}$:

$$\boxed{Q_m\cdot K_n = Q^\top R_{n-m}K}$$

Only $n - m$ survives.

### 3.6 How exactly is this better than sinusoidal?

The obvious answer is slightly wrong. **Mixing content and position is not the problem: attention needs both.** RoPE's improvement is that the positional dependence is **explicitly relative**:

$$\text{score} = \|q\|\|k\|\cos\big(\underbrace{(\alpha-\beta)}_{\text{content angle}} + \underbrace{(m-n)\theta}_{\text{relative positional angle}}\big)$$

Moving the same content from positions (5, 8) to (105, 108): under sinusoidal, the cross terms compare content against **absolute** position, so the score can change. Under RoPE, $(5-8)\theta = (105-108)\theta = -3\theta$, so the score is identical for fixed Q/K. This holds for every pair, and so for their sum.

| Additive sinusoidal | RoPE |
|---|---|
| Content interacts with absolute position encodings | Content interacts with relative rotations |
| A common shift can change the score | A common shift leaves the score unchanged (fixed content) |

RoPE is **explicitly relative**, gives **same content + same separation → same score**, and **preserves vector lengths** (position affects only alignment). It doesn't eliminate content-position interaction; it gives that interaction a **guaranteed relative structure**.

---

## 4. Scaling RoPE to Full Dimensions

Real heads have $d$ = 64 or 128. **Split into $d/2$ pairs** $(q_0,q_1),\dots,(q_{d-2},q_{d-1})$ and rotate each at its own frequency:

$$\theta_i = \frac{1}{10000^{2i/d}}$$

This is the same spectrum as sinusoidal PE (positional encoding): early pairs rotate fast, late pairs slowly. There is no mixing across pairs. The full rotation is **block diagonal**:

$$R_m = \text{diag}\big(R(m\theta_0), R(m\theta_1), \dots, R(m\theta_{d/2-1})\big)$$

For $d = 6$, with $c_i = \cos m\theta_i$ and $s_i = \sin m\theta_i$:

$$R_m = \begin{bmatrix}c_0&-s_0&0&0&0&0\\ s_0&c_0&0&0&0&0\\ 0&0&c_1&-s_1&0&0\\ 0&0&s_1&c_1&0&0\\ 0&0&0&0&c_2&-s_2\\ 0&0&0&0&s_2&c_2\end{bmatrix},\qquad R_mq = \begin{bmatrix}c_0q_0 - s_0q_1\\ s_0q_0 + c_0q_1\\ c_1q_2 - s_1q_3\\ s_1q_2 + c_1q_3\\ c_2q_4 - s_2q_5\\ s_2q_4 + c_2q_5\end{bmatrix}$$

$(q_0,q_1)$ rotates by $m\theta_0$, $(q_2,q_3)$ by $m\theta_1$, $(q_4,q_5)$ by $m\theta_2$. The pairs stay in their slots and are never summed.

---

## 5. Does Rotation Corrupt Semantic Meaning?

**Objection:** Q's direction $\alpha$ is meaning, and rotation changes direction, so rotation corrupts meaning.

**Flaw:** meaning in attention is **not the direction of one vector**. It is the angle **between** Q and K, $\alpha - \beta$, which still sits intact inside $\cos((\alpha-\beta) + (m-n)\theta)$. Position is combined with meaning, not substituted for it.

**Sanity checks.**
- **Same position** ($m = n$): the score is $\cos(\alpha - \beta)$, exactly the un-rotated value.
- **Length preserved:** $\|Rv\|^2 = v^\top R^\top Rv = v^\top v$, since $R^\top R = I$. Magnitude (which can carry learned information) is untouched.

**What $\alpha$, $\beta$ really are.** They are **not** embedding angles. They are the directions of $Q = W_q\,\text{embed}$ and $K = W_k\,\text{embed}$, **after** projection, in Q/K space. Why would $(\alpha - \beta)$ mean anything? **Training.** Gradient descent shapes $W_q, W_k$ so that tokens that should attend get a small angle (high cosine, high score) and tokens that shouldn't get a large one. The meaning is **learned**, and RoPE rotates within that learned space.

---

## 6. Limitations of RoPE

### 6.1 Aliasing

Cosine repeats every $2\pi$, so two gaps look identical in pair $i$ when

$$(g_1 - g_2)\,\theta_i = 2\pi k$$

This is **aliasing**.
- **Fast pairs** ($\theta \approx 1$) alias every ≈ 6 tokens (gap 1 ≈ gap 7). That's fine: they only serve short-range precision.
- **Slow pairs** ($\theta \approx 0.0001$) don't repeat until $2\pi\cdot10{,}000 \approx 62{,}832$ tokens, so they never alias in normal contexts.

**The spectrum saves us:** for two gaps to be truly indistinguishable, they'd have to alias in **every** pair at once (fast, medium, and slow periods together). The combination gives each gap a unique fingerprint, the same argument as unique positions in Lecture 2.

### 6.2 Encoding a position is not the same as using it

RoPE's angle is $m\omega_i$, and attention sees $\Delta\theta_i = (m-n)\omega_i$. A model trained on 4k tokens has only learned how the rotations behave over that range. Beyond it the formula still works, but the **learned patterns may fail**.

Example: one slow pair with matching content contributes $\cos\Delta\theta$ and only saw 0°-30° during training.

| Distance | Relative angle | Contribution |
|---|---|---|
| Max training distance | 30° | +0.866 |
| 4× that | 120° | −0.5 |

Content that always **supported** a match now **counts against** it (cosine is negative between 90° and 270°).

**Don't fast pairs change sign too?** Yes, but they completed many rotations during training, so those sign changes are familiar. Slow pairs only explored a small arc. Long distances also produce unfamiliar **combinations** across pairs. The real problem is **extrapolating learned behaviour past the distances it was learned on**. There is no universal "breaking angle", and one pair wrapping doesn't make the encoding repeat.

### 6.3 Attention dilution in long contexts

Even with perfect positions, softmax $P(j) = e^{z_j}/\sum_k e^{z_k}$ spreads mass over more tokens. With one relevant token at score $g$ and $N$ others at 0:

$$P(\text{relevant}) = \frac{e^g}{e^g + N}$$

$g = 2.5$ ($e^{2.5} = 12.18$): **80.2%** with 3 competitors, **28.2%** with 31.

| $N$ | Total tokens | Weight at $g = 2$ ($e^2 = 7.39$) |
|---|---|---|
| 3 | 4 | 71.1% |
| 9 | 10 | 45.1% |
| 99 | 100 | 6.9% |

The relevant token still beats each competitor, but together they take more of the mass. $P$ is the attention **weight**, not the probability of relevance. **RoPE alone doesn't solve this.**

### 6.4 What a context-extension method must fix

The goal: **extend positional range while preserving useful distinctions and effective attention.**

| Problem | Direction |
|---|---|
| Longer distances give unfamiliar rotations | **Position interpolation**: compress positions into the familiar range |
| Uniform compression also shrinks nearby distinctions | **Frequency-dependent scaling**: keep fast-frequency detail, slow other pairs |
| Attention concentration changes with length | **YaRN** (Yet another RoPE extensioN): selective frequency scaling + attention-score scaling |

These help, but reliable extension may still need extra training.

---

## Numerical Problems

**Problem 1 — Dot-product property.** $\theta = 0.2$, $m = 10$, $n = 4$.

$\mathbf p_{10} = (\sin 2, \cos 2) = (0.9093, -0.4161)$ and $\mathbf p_4 = (\sin 0.8, \cos 0.8) = (0.7174, 0.6967)$. The dot product is $0.6523 - 0.2899 = 0.3624$, and $\cos(0.2\cdot6) = \cos 1.2 = 0.3624$.

$$\boxed{0.3624}$$

(110, 104) would give the same value, since only the gap matters.

**Problem 2 — Shift invariance.** $\alpha = 0°$, $\beta = 60°$, $\theta = 10°$.

- $(5, 8)$: $\cos(50° - 140°) = \cos(-90°) = 0$.
- $(105, 108)$: $\cos(1050° - 1140°) = \cos(-90°) = 0$.
- Closed form: $\cos(-60° - 30°) = 0$.

$$\boxed{\text{score}(5,8) = \text{score}(105,108) = 0}$$

**Problem 3 — Aliasing distance.** $d = 64$, $i = 8$: $\theta_8 = 10000^{-16/64} = 10^{-1} = 0.1$. Setting $k = 1$ gives $g_1 - g_2 = 2\pi/0.1 \approx 62.8$.

$$\boxed{\text{aliases every} \approx 63 \text{ positions}}$$

The aliasing distance is the band's wavelength.

**Problem 4 — Dilution.** $g = 3$ ($e^3 = 20.086$).

- $N = 7$: $20.086/27.086 = 74.2\%$.
- $N = 127$: $20.086/147.086 = 13.7\%$.

$$\boxed{74.2\% \to 13.7\%,\ \approx 5.4\times\ \text{smaller}}$$

The score is unchanged; only the denominator grew.

---

## Formula Summary Sheet

| Concept | Formula |
|---|---|
| Pair dot product | $\mathbf p_m^\top\mathbf p_n = \cos(\theta(m-n))$; full: $\sum_i\cos(\theta_i(m-n))$ |
| Additive input | $\text{input}_m = \text{embed}_m + PE_m$ |
| Four-term expansion | $(W_qe_m)(W_ke_n) + (W_qe_m)(W_kPE_n) + (W_qPE_m)(W_ke_n) + (W_qPE_m)(W_kPE_n)$ |
| RoPE goal | $Q_m\cdot K_n = f(\text{semantics}, m-n)$ |
| 2D rotation | $q_0' = q_0\cos m\theta - q_1\sin m\theta$; $q_1' = q_0\sin m\theta + q_1\cos m\theta$ |
| RoPE score | $\|Q\|\|K\|\cos((\alpha-\beta) + (m-n)\theta)$ |
| Matrix form | $Q_m\cdot K_n = Q^\top R_{n-m}K$, $R_m^\top = R_{-m}$ |
| Length preservation | $\|Rv\|^2 = v^\top R^\top Rv = \|v\|^2$ |
| Pair frequency | $\theta_i = 1/10000^{2i/d}$ |
| Block-diagonal $R_m$ | $\text{diag}(R(m\theta_0),\dots,R(m\theta_{d/2-1}))$ |
| Aliasing | $(g_1-g_2)\theta_i = 2\pi k$ |
| Dilution | $P(\text{relevant}) = e^g/(e^g + N)$ |

---

## Cheat Sheet & Recall Aids

- **Story:** sinusoidal adds position **before** $W_q, W_k$, giving four terms with one usable. RoPE rotates Q, K **after** projection, so the score depends only on content and $m - n$.
- **Mnemonic "SS, SP, PS, PP":** the four terms. Only PP carries $\cos\theta(m-n)$; SP and PS are noise.
- **Three weaknesses, one cause:** unstructured interaction, no guaranteed relative score, additive fusion, all because position is **added**.
- **Precise claim:** RoPE doesn't remove content-position interaction; it guarantees the positional part depends only on $m - n$.
- **Invariance test:** shift both tokens by 100. Sinusoidal: the score may change. RoPE: it doesn't.
- **Rotation keeps meaning:** meaning = angle **between** Q and K; rotation is length-preserving.
- **$\alpha, \beta$** live in Q/K space, and their meaning is learned.
- **Two separate long-context failures:** unfamiliar rotations (30° → 120°, +0.866 → −0.5) and softmax dilution (which is not a PE problem).
- **Aliasing:** each band repeats at $2\pi/\theta_i$ (≈ 6 fast, ≈ 62,832 slow); all at once is essentially impossible.

**Self-test**
- [ ] Derive $\cos(\theta(m-n))$ from the cosine difference identity.
- [ ] Write the four terms and label them.
- [ ] Derive $Q^\top R_{n-m}K$.
- [ ] Rebut "rotation destroys semantics" with both arguments.
- [ ] Explain why fast-band aliasing is harmless but slow-band extrapolation isn't.
- [ ] Compute $e^g/(e^g+N)$ for two values of $N$.
