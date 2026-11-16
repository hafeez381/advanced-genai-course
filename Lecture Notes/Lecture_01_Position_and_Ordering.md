---
type: lecture
course: '[[_CS5326 Advanced GenAI|CS5326 Advanced GenAI]]'
source:
- '[[001-PositionAndOrdering-1.pdf]]'
tags:
- course/cs5326
---

%% nav:start %%
[[_CS5326 Advanced GenAI|CS5326 Advanced GenAI]] · [[Lecture_02_Sinusoidal_Positional_Encoding]] →
Concepts: [[Positional Encoding]]
%% nav:end %%

# Position and Ordering (CS5326 — Advanced GenAI and Agents, Fall 2026, Agha Ali Raza)

## Topic Inventory

- [[#1. Why Sequence Position Matters|1. Why Sequence Position Matters]]
- [[#2. Positional Information in Recurrent Architectures|2. Positional Information in Recurrent Architectures]]
  - [[#2.1 Order as an implicit property of recurrence|2.1 Order as an implicit property of recurrence]]
  - [[#2.2 The LSTM's memory mechanism|2.2 The LSTM's memory mechanism]]
  - [[#2.3 Ordering vs. absolute position|2.3 Ordering vs. absolute position]]
- [[#3. The Transformer's Positional Blind Spot|3. The Transformer's Positional Blind Spot]]
  - [[#3.1 Sequential computation vs. parallel computation: the RNN/Transformer trade-off|3.1 Sequential computation vs. parallel computation: the RNN/Transformer trade-off]]
  - [[#3.2 Self-attention and permutation equivariance|3.2 Self-attention and permutation equivariance]]
  - [[#3.3 Why order becomes invisible in raw self-attention|3.3 Why order becomes invisible in raw self-attention]]
  - [[#3.4 Permutation invariance vs. permutation equivariance|3.4 Permutation invariance vs. permutation equivariance]]
  - [[#3.5 Why equivariance without position still breaks language understanding|3.5 Why equivariance without position still breaks language understanding]]
- [[#4. Designing a Positional Encoding|4. Designing a Positional Encoding]]
  - [[#4.1 Attempt 1 — raw position integers|4.1 Attempt 1 — raw position integers]]
  - [[#4.2 Attempt 2 — normalizing positions into [0, 1]|4.2 Attempt 2 — normalizing positions into [0, 1]]]
  - [[#4.3 Attempt 3 — binary encoding vectors|4.3 Attempt 3 — binary encoding vectors]]
  - [[#4.4 The frequency structure hidden inside binary encoding|4.4 The frequency structure hidden inside binary encoding]]
  - [[#4.5 The discontinuity problem and the bridge to smooth encodings|4.5 The discontinuity problem and the bridge to smooth encodings]]

---

## 1. Why Sequence Position Matters

Take the sentence *"The dog hit the cat with a bat while favoring the rat"*. As a bare bag of words, a model knows **which** words occur but not **where**. Indexing makes order explicit: $\text{The}_1\ \text{dog}_2\ \text{hit}_3 \dots \text{rat}_{12}$.

The question driving the lecture: **why did position suddenly need explicit handling with Transformers, when RNNs (recurrent neural networks) and LSTMs (long short-term memory networks) never needed it?**

> **[In-class poll]** "Why didn't we discuss positional information in RNNs?" Answer: **RNNs encode position implicitly**, through recurrence (§2).

---

## 2. Positional Information in Recurrent Architectures

### 2.1 Order as an implicit property of recurrence

Order is built into the recurrence:

$$h_t = f(x_t, h_{t-1})$$

$x_t$ cannot be processed until $x_1,\dots,x_{t-1}$ have been. The hidden state is a running summary of everything so far, **in that order**.

![[L01_slide4.png|600]]

*The same weights ($W_{xh}, W_{hh}, W_{hy}$) are reused at every step. The slide's analogy: $h_t$ is a smoothie blended one ingredient at a time ("You", "played", "the", ... "home"). The newest flavour is strongest and older ones fade, which is why vanilla RNNs **forget** distant context.*

**Order changes the computation.** "dog → bites → man" gives $h_3 = f(x_{man}, f(x_{bites}, f(x_{dog}, h_0)))$, while "man → bites → dog" gives a different chain, so in general $h_3 \ne h_3'$.

**The same token at different positions gets different states.** If "cat" appears as $x_1$ and $x_{20}$, the embeddings are equal but $h_1 = f(x_{cat}, h_0)$ while $h_{20} = f(x_{cat}, h_{19})$ carries 19 steps of history. The RNN never stores "this is position 20"; position is baked into the nesting:

$$h_t = f\big(x_t,\ f(x_{t-1},\ f(x_{t-2}, \ldots f(x_1, h_0)\ldots))\big)$$

### 2.2 The LSTM's memory mechanism

Inputs at step $t$: $X_t$, $H_{t-1}$, $C_{t-1}$. $\sigma$ is the sigmoid and $\odot$ is elementwise product.

$$
\begin{aligned}
F_t &= \sigma(U_f H_{t-1} + W_f X_t + b_f) &&\text{forget gate: how much old cell to keep}\\
I_t &= \sigma(U_i H_{t-1} + W_i X_t + b_i) &&\text{input gate: how much new info to admit}\\
O_t &= \sigma(U_o H_{t-1} + W_o X_t + b_o) &&\text{output gate: how much cell to expose}\\
\tilde C_t &= \tanh(U_g H_{t-1} + W_g X_t + b_g) &&\text{candidate memory}\\
C_t &= F_t \odot C_{t-1} + I_t \odot \tilde C_t\\
H_t &= O_t \odot \tanh(C_t),\qquad \hat Y_t = \phi_o(W_{hy}H_t + b_y)
\end{aligned}
$$

Shapes (hidden size $d$, input size $V$): each gate is $[d\times1] = \sigma([d\times d][d\times1] + [d\times V][V\times1] + [d\times1])$; the output is $[V\times1] = [V\times d][d\times1] + [V\times1]$.

![[Pasted image 20260914125559.png|470]]
*LSTM cell (Olah, "Understanding LSTM Networks"). The horizontal cell state $C_t$ is a highway that information flows along, gated only where the network writes or erases.*

For position the LSTM is the same as the RNN: $C_t$ depends on $C_{t-1}$ and $H_t$ on $H_{t-1}$, so order comes for free. It just preserves long-range information better.

### 2.3 Ordering vs. absolute position

The RNN gets **ordering** ("I arrived here after these tokens"), not **absolute position** ("I am token 37"). It *could* learn a counter, e.g. one dimension with $h_{t,j} = h_{t-1,j} + 1$ giving $0,1,2,\dots$, but nothing requires it.

**In RNNs/LSTMs, order is intrinsic; absolute position is only implicit.**

---

## 3. The Transformer's Positional Blind Spot

### 3.1 Sequential computation vs. parallel computation: the RNN/Transformer trade-off

> "Did we just learn RNNs were better than Transformers, and spend two lectures solving what RNNs already solved?"

No. The RNN's free ordering has a price. $h_1 \to h_2 \to \dots \to h_T$ is strictly sequential ($h_{100}$ needs $h_1..h_{99}$), so training can't be parallelized, and RNNs forget distant context (LSTMs only partly fix this). A Transformer processes $x_1,\dots,x_T$ **in parallel**.

| | RNN | Transformer |
|---|---|---|
| Computation | Sequential | Parallel |
| Order | Comes for free | Disappears |
| Cost | Poor parallelism, weak long-range memory | **Must explicitly add position** |

### 3.2 Self-attention and permutation equivariance

Self-attention takes the stacked $X = [x_1; x_2; x_3]$ and computes

$$Q = XW_Q,\ K = XW_K,\ V = XW_V,\qquad \text{Attention}(X) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

Nothing says "$x_1$ came before $x_2$": permute the inputs and the outputs are permuted the same way (**permutation equivariance**, §3.4).

### 3.3 Why order becomes invisible in raw self-attention

"dog" has the same embedding at position 1 or 3, so

$$\text{score}(\text{dog}, \text{bites}) = (W_q e_{dog})\cdot(W_k e_{bites})$$

is identical in "Dog bites man" and "Man bites dog". Every pair behaves this way, so the two attention matrices are the same up to relabeling rows and columns:

| "Dog bites man" | Dog | bites | man |
|---|---|---|---|
| **Dog** | 0.34 | 0.41 | 0.25 |
| **bites** | 0.30 | 0.38 | 0.32 |
| **man** | 0.22 | 0.44 | 0.34 |

| "Man bites dog" | Man | bites | dog |
|---|---|---|---|
| **Man** | 0.34 | 0.44 | 0.22 |
| **bites** | 0.32 | 0.38 | 0.30 |
| **dog** | 0.25 | 0.41 | 0.34 |

Embeddings are looked up by **token identity, not position**. Without positional information, self-attention treats a sequence as a **set**: it knows what tokens are present and how they relate, not their order.

### 3.4 Permutation invariance vs. permutation equivariance

> **[In-class poll]** Same thing or different? **Different.**

| | Definition | Example |
|---|---|---|
| **Invariance** | $f(PX) = f(X)$: input reordered, output unchanged | Sum: $\{2,5,7\}$ and $\{7,2,5\}$ both give 14 |
| **Equivariance** | $f(PX) = Pf(X)$: output reordered the same way | $ABC \to abc$, so $CAB \to cab$ |

$P$ is a **permutation matrix**, which just reorders rows:

$$X = \begin{bmatrix}A\\B\\C\end{bmatrix},\quad P = \begin{bmatrix}0&0&1\\1&0&0\\0&1&0\end{bmatrix},\quad PX = \begin{bmatrix}C\\A\\B\end{bmatrix}$$

![[L01_slide18.png|520]]

*Equivariance: reorder the input and the output is reordered the same way. Invariance: reorder the input and one pooled output stays unchanged.*

**Vanilla self-attention without position is permutation-equivariant**, not invariant.

### 3.5 Why equivariance without position still breaks language understanding

Let $S_1$ = "dog bites man" with $f(S_1) = [h_{dog}, h_{bites}, h_{man}]$. Since $S_2 = PS_1$, equivariance gives $f(S_2) = [h_{man}, h_{bites}, h_{dog}]$. So $h_{dog}$ is **exactly the same vector** in both sentences, even though dog is the agent in one and the patient in the other. The only thing that changed was *where* the words sit, and attention can't see that.

It gets worse with a permutation-invariant pooling $g$ (sum/mean for classification): $g(f(S_1)) = g(f(S_2))$, so the two sentences are **indistinguishable**. Equivariance is not the flaw. The flaw is that **a token's representation cannot depend on its position without an explicit mechanism**. The fix is to inject position:

$$x_i \to x_i + PE_i$$

Then $x_{dog} + PE_1 \ne x_{dog} + PE_3$, so attention can learn "dog before bites" ≠ "dog after bites".

**Aside: a causal mask is not a positional encoding.**

![[L01_slide24.png|520]]

*Unmasked: every token sees every token. Causal: a token sees only its prefix. The mask adds an earlier/later **asymmetry**, but never encodes a distance like "7 tokens apart". The mask restricts **which** tokens are seen; the encoding tells the model **where** each token sits.*

---

## 4. Designing a Positional Encoding

Each attempt below fixes the previous one's failure.

### 4.1 Attempt 1 — raw position integers

$$e_i' = e_i + i$$

"Dog bites man" and "Man bites dog" now differ. Example with a shared embedding $e = [0.32, -0.71, 0.15, 0.58]$:

| Token (pos) | Position added | $e' = e + \text{pos}$ |
|---|---|---|
| This (0) | 0 | 0.32, −0.71, 0.15, 0.58 |
| quick (1) | 1 | 1.32, 0.29, 1.15, 1.58 |
| fox (100) | 100 | 100.32, 99.29, 100.15, 100.58 |
| dog (500) | 500 | 500.32, 499.29, 500.15, 500.58 |

![[L01_slide29.png|600]]

*Position 0 leaves the embedding intact. At position 500 every dimension becomes ≈ 500 (500.6, 499.6, ...), and the token's pattern of small ± values is buried.*

Three failures:
- **Scale:** embeddings live in about $[-1, 1]$, so adding 500 ==buries the semantic content==. Early and late tokens end up in completely different numeric ranges purely because of where they sit.
- **No upper bound:** ==a model trained on length 512 has only seen positions 0-511. At inference with 1024 tokens it meets values up to 1023==, an **out-of-distribution** problem.
- **Inconsistent distance:** ==1 → 2 is a 100% jump; 500 → 501 is 0.2%==, so "the next token" has no consistent meaning.

### 4.2 Attempt 2 — normalizing positions into [0, 1]

$$PE(\text{pos}) = \frac{\text{pos}}{L-1}$$

For $L = 512$: 0 → 0.0, 255 → 0.5, 511 → 1.0. Bounded, so scale is fixed. Two new problems:
- **Spacing:** ==$L = 10$ gives a gap of 0.11 between neighbours; $L = 1000$ gives 0.001==. "One token apart" means a 100× different distance depending on length.
- **Same position, different values:** ==position 5 is $5/9 = 0.556$ at $L = 10$ but $5/999 = 0.005$ at $L = 1000$.== A model trained on short sequences learns "0.5 ≈ middle of a sentence", which is wrong when 0.5 means position 500.

Root cause: ==the scheme is **relative to sequence length**. It encodes "how far through am I", not a fixed position identity==, so there is no consistent identity and no consistent spacing.

### 4.3 Attempt 3 — binary encoding vectors

Both attempts used **one number** per position. Instead, ==use a **vector**==: the binary representation of the position, with ==$\lceil\log_2 L\rceil$ bits== (9 bits for $L = 512$).

| Position | Binary vector |
|---|---|
| 0 | [0,0,0,0,0,0,0,0,0] |
| 1 | [0,0,0,0,0,0,0,0,1] |
| 2 | [0,0,0,0,0,0,0,1,0] |
| 5 | [0,0,0,0,0,0,1,0,1] |
| 255 | [0,1,1,1,1,1,1,1,1] |
| 511 | [1,1,1,1,1,1,1,1,1] |

It fixes everything so far:
- **Bounded:** entries are 0/1.
- **Unique per position.**
- **Length-independent:** ==position 5 is the same vector whatever the sequence length.==
- **Slowly growing dimension:** 10 bits cover 1024 positions, 20 bits cover 1,048,576.

### 4.4 The frequency structure hidden inside binary encoding

Read each bit column down positions 0-7:

| Pos | Bit 2 | Bit 1 | Bit 0 |
|---|---|---|---|
| 0 | 0 | 0 | 0 |
| 1 | 0 | 0 | 1 |
| 2 | 0 | 1 | 0 |
| 3 | 0 | 1 | 1 |
| 4 | 1 | 0 | 0 |
| 5 | 1 | 0 | 1 |
| 6 | 1 | 1 | 0 |
| 7 | 1 | 1 | 1 |

Bit 0 flips every position (period 2), bit 1 every 2 (period 4), bit 2 every 4 (period 8). In general:

$$\text{period of bit } i = 2^{\,i+1}$$

| Bit | Flips every | Period | Role |
|---|---|---|---|
| 0 (least significant bit, LSB) | 1 | 2 | Finest grain |
| 1 | 2 | 4 | |
| 2 | 4 | 8 | |
| 3 | 8 | 16 | |
| ⋮ | | | |
| 8 (most significant bit, MSB) | 256 | 512 | Coarsest grain |

![[L01_slide42.png|550]]

*Cycles per 512 positions: bit 0 = 256, bit 1 = 128, ..., bit 8 = 1. Each bit halves the frequency of the one before.*

![[Pasted image 20260914125749.png]]
*The bits as stacked square waves. Low bits oscillate fast (odd/even, fine position); high bits oscillate slowly (which half of the sequence, coarse position).*

This is the key observation: binary is a **multi-frequency encoding**. ==The lowest bit gives fine position, the highest bit gives coarse position.== This directly motivates sinusoidal encoding.

### 4.5 The discontinuity problem and the bridge to smooth encodings

==Positions 3 = [0,1,1] and 4 = [1,0,0] are adjacent, but every bit flips==, so they are far apart in vector space. Positions 2 = [0,1,0] and 3 = [0,1,1] differ in one bit. ==Adjacent positions have wildly inconsistent distances.== The cause is **carries**: when the lower bits are all 1, incrementing flips them all to 0 and flips the next bit.

Neural nets learn smooth functions best (nearby inputs → nearby outputs), so the model can't learn a smooth notion of "nearby positions".

| Keep | Fix |
|---|---|
| Multi-frequency structure (fast bits = fine, slow bits = coarse) | Discrete 0/1 jumps |
| Vector representation | |

**Fix:** keep the periods, replace the square waves with smooth **sine/cosine** waves.

![[Pasted image 20260914125845.png]]
*"Keep the periods, remove the jumps": binary square waves at periods 8, 16, 32, the motivation for sinusoidal encoding.*

The source deck is Part 1 of Naveen Reddy's *Positional Encoding Explained: From Position to Binary Encoding*. Sinusoidal encoding comes next lecture.

---

## Numerical Problems

**Problem 1 — Scale problem.** $e = (0.42, -0.31, 0.55, -0.18)$, raw addition at $i = 300$. Find $\|e'\|/\|e\|$.

$e' = (300.42, 299.69, 300.55, 299.82)$. $\|e\| = \sqrt{0.1764 + 0.0961 + 0.3025 + 0.0324} = \sqrt{0.6074} \approx 0.779$. $\|e'\|^2 \approx 90252.2 + 89814.1 + 90330.3 + 89892.0 = 360288.6$, so $\|e'\| \approx 600.2$.

$$\boxed{\|e'\|/\|e\| \approx 770}$$

The position term inflates the vector about 770×, burying the semantics.

**Problem 2 — Binary encoding.** $L = 512$ (9 bits). Encode position 37 and check bit 5 with the period formula.

$37 = 32 + 4 + 1 = 2^5 + 2^2 + 2^0$, so bits 5, 2, 0 are set. Check: bit 5 has period $2^6 = 64$ and is 1 for positions 32-63 (then 96-127, ...). 37 is in [32, 63]. ✓

$$\boxed{\text{PE}(37) = [0,0,0,1,0,0,1,0,1]}$$

---

## Formula Summary Sheet

| Concept | Formula |
|---|---|
| RNN recurrence | $h_t = f(x_t, h_{t-1})$ |
| Nested RNN computation | $h_t = f(x_t, f(x_{t-1}, \ldots f(x_1, h_0)))$ |
| LSTM gates | $F_t, I_t, O_t = \sigma(U H_{t-1} + W X_t + b)$ |
| LSTM candidate | $\tilde C_t = \tanh(U_g H_{t-1} + W_g X_t + b_g)$ |
| LSTM cell / hidden | $C_t = F_t \odot C_{t-1} + I_t \odot \tilde C_t$; $H_t = O_t \odot \tanh(C_t)$ |
| LSTM output | $\hat Y_t = \phi_o(W_{hy}H_t + b_y)$ |
| Self-attention | $\text{softmax}(QK^\top/\sqrt{d_k})V$, $Q = XW_Q$, $K = XW_K$, $V = XW_V$ |
| Equivariance / invariance | $f(PX) = Pf(X)$ / $f(PX) = f(X)$ |
| Position injection | $x_i \to x_i + PE_i$ |
| Attempt 1 | $e_i' = e_i + i$ |
| Attempt 2 | $PE(\text{pos}) = \text{pos}/(L-1)$ |
| Attempt 3 | $\lceil\log_2 L\rceil$ bits; period of bit $i$ = $2^{i+1}$ |

---

## Cheat Sheet & Recall Aids

- **RNN:** order for free, computation stuck sequential. **Transformer:** parallel, order added by hand.
- **Mask ≠ position:** the mask restricts *which* tokens are seen; the encoding says *where* they are.
- **Mnemonic for the three failed attempts: "Too Big, Too Relative, Too Jumpy."** Raw integers (unbounded, swamp the embedding), normalized (same index means different values at different lengths), binary (carries make neighbours far apart).
- **Carry forward:** multi-frequency structure. Low bits fast/fine, high bits slow/coarse. Sinusoidal PE makes it smooth.

**Self-test**
- [ ] Show $h_3 \ne h_3'$ for dog/man bites via the recurrence.
- [ ] Why is self-attention equivariant, not invariant?
- [ ] Reproduce the "dog bites man" identical-attention argument.
- [ ] List the three failures of raw integers (scale, no bound, inconsistent distance).
- [ ] Compute a bit's period with $2^{i+1}$.
