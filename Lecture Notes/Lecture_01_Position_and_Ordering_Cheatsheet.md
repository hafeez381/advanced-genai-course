---
type: study-aid
course: '[[_CS5326 Advanced GenAI|CS5326 Advanced GenAI]]'
tags:
- course/cs5326
---

# Position and Ordering — Exam Cheatsheet (CS5326)

## Topic Inventory

- [[Lecture_01_Position_and_Ordering#1. Why Sequence Position Matters|1. Why Sequence Position Matters]]
- [[Lecture_01_Position_and_Ordering#2. Positional Information in Recurrent Architectures|2. Positional Information in Recurrent Architectures]]
  - [[Lecture_01_Position_and_Ordering#2.1 Order as an implicit property of recurrence|2.1 Order via recurrence]]
  - [[Lecture_01_Position_and_Ordering#2.2 The LSTM's memory mechanism|2.2 LSTM]]
  - [[Lecture_01_Position_and_Ordering#2.3 Ordering vs. absolute position|2.3 Ordering vs. absolute position]]
- [[Lecture_01_Position_and_Ordering#3. The Transformer's Positional Blind Spot|3. The Transformer's Positional Blind Spot]]
  - [[Lecture_01_Position_and_Ordering#3.1 Sequential computation vs. parallel computation: the RNN/Transformer trade-off|3.1 Sequential vs. parallel]]
  - [[Lecture_01_Position_and_Ordering#3.2 Self-attention and permutation equivariance|3.2 Self-attention & permutation equivariance]]
  - [[Lecture_01_Position_and_Ordering#3.3 Why order becomes invisible in raw self-attention|3.3 Why order is invisible]]
  - [[Lecture_01_Position_and_Ordering#3.4 Permutation invariance vs. permutation equivariance|3.4 Invariance vs. equivariance]]
  - [[Lecture_01_Position_and_Ordering#3.5 Why equivariance without position still breaks language understanding|3.5 Why it breaks language]]
- [[Lecture_01_Position_and_Ordering#4. Designing a Positional Encoding|4. Designing a Positional Encoding]]
  - [[Lecture_01_Position_and_Ordering#4.1 Attempt 1 — raw position integers|4.1 Raw integers]]
  - [[Lecture_01_Position_and_Ordering#4.2 Attempt 2 — normalizing positions into [0, 1]|4.2 Normalized [0,1]]]
  - [[Lecture_01_Position_and_Ordering#4.3 Attempt 3 — binary encoding vectors|4.3 Binary vectors]]
  - [[Lecture_01_Position_and_Ordering#4.4 The frequency structure hidden inside binary encoding|4.4 Frequency structure]]
  - [[Lecture_01_Position_and_Ordering#4.5 The discontinuity problem and the bridge to smooth encodings|4.5 Discontinuity → bridge to sinusoidal]]

---

## 1–2. Recurrence Gives Order for Free

$$h_t = f(x_t, h_{t-1})$$

- $x_t$ cannot be processed before $x_1,\dots,x_{t-1}$ → order is intrinsic to computation.
- Same 3 words, different order → different final state: $h_3 \neq h_3'$ (dog-bites-man vs. man-bites-dog).
- Repeated token at different $t$ → same input, different hidden state: $x_1 = x_{20}$ but $h_1 \neq h_{20}$.
- Nested form: $h_t = f(x_t, f(x_{t-1}, \ldots f(x_1,h_0)\ldots))$ — position is implicit in computational depth, never an explicit index.
- **Order is intrinsic; absolute position is only implicit** (RNN *could* learn a counter $h_{t,j}=h_{t-1,j}+1$, but isn't required to).

**LSTM gates:**

$$F_t=\sigma(U_fH_{t-1}+W_fX_t+b_f)\quad I_t=\sigma(U_iH_{t-1}+W_iX_t+b_i)\quad O_t=\sigma(U_oH_{t-1}+W_oX_t+b_o)$$
$$\tilde{C}_t=\tanh(U_gH_{t-1}+W_gX_t+b_g)\qquad C_t=(F_t\odot C_{t-1})+(I_t\odot\tilde C_t)\qquad H_t=O_t\odot\tanh(C_t)$$
$$\hat Y_t=\phi_o(W_{hy}H_t+b_y)$$

- $C_t$ depends on $C_{t-1}$, $H_t$ depends on $H_{t-1}$ → order intrinsic here too; gates just fight forgetting better than vanilla RNN.

---

## 3. Transformers Lose Order

| | RNN | Transformer |
|---|---|---|
| Computation | Sequential ($h_1\to h_2\to\cdots$) | Parallel ($x_1,\dots,x_T$ at once) |
| Order | Free | Lost — must be added |
| Trade-off | Poor parallelism, weak long memory | Excellent parallelism, no position |

**Self-attention:**
$$Q=XW_Q,\ K=XW_K,\ V=XW_V \qquad \text{Attention}(X)=\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

- Score depends only on token identity: $\text{score}(\text{dog,bites}) = (W_q e_{\text{dog}})\cdot(W_k e_{\text{bites}})$ — identical in "dog bites man" and "man bites dog."
- Full attention matrices for both sentences are identical (up to relabeling) → **position doesn't exist in the computation**.

**Invariance vs. equivariance:**

| Property | Definition | Example |
|---|---|---|
| Equivariant | $f(PX)=Pf(X)$ | $ABC\to abc$, $CAB\to cab$ |
| Invariant | $f(PX)=f(X)$ | sum: $f(2,5,7)=f(7,2,5)=14$ |

- Raw self-attention = **permutation-equivariant**, NOT invariant.
- $P$ = permutation matrix, e.g. $P=\begin{bmatrix}0&0&1\\1&0&0\\0&1&0\end{bmatrix}$ reorders $[A,B,C]\to[C,A,B]$.

**Why it breaks language:** $S_2=PS_1 \Rightarrow f(S_2)=Pf(S_1)$, so $h_{\text{dog}}$, $h_{\text{man}}$ are identical across both sentences even though agent/patient roles flip. Under permutation-invariant pooling: $g(f(S_1))=g(f(S_2))$ → sentences indistinguishable.

**Fix:** $x_i \to x_i + PE_i$.

- **Causal mask ≠ positional encoding.** Mask = who can attend to whom (earlier/later only); does not encode "7 tokens apart."

---

## 4. Positional Encoding Attempts

| Attempt | Formula | Fixes | Breaks |
|---|---|---|---|
| 1. Raw integer | $e_i' = e_i + i$ | Order no longer invisible | **Scale** (500 drowns embedding), **no upper bound** (OOD beyond training length), **inconsistent distance** (1→2 is 100%, 500→501 is 0.2%) |
| 2. Normalized | $PE(\text{pos})=\dfrac{\text{pos}}{L-1}$ | Bounded to $[0,1]$ | **Spacing problem** (gap shrinks as $L$ grows), **same position ≠ same value** across lengths (pos 5: 0.556 for $L{=}10$ vs. 0.005 for $L{=}1000$) — relative to length, not absolute |
| 3. Binary vector | $d=\lceil\log_2 L\rceil$ bits | Bounded, unique, length-independent, fixed low dimensionality | **Discontinuity**: adjacent positions can flip all bits (pos 3 $[0,1,1]\to$ pos 4 $[1,0,0]$) due to carries |

**Binary encoding oscillation:** bit $i$ (0 = LSB) flips every $2^i$ positions, full period $2^{i+1}$.

| Bit | Flips every | Period | Role |
|---|---|---|---|
| 0 (LSB) | 1 | 2 | Finest grain |
| 1 | 2 | 4 | |
| 2 | 4 | 8 | |
| ⋮ | ⋮ | ⋮ | ⋮ |
| 8 (MSB, 9-bit code) | 256 | 512 | Coarsest grain |

- **Multi-frequency structure** = low bits fine-grained, high bits coarse-grained → carries forward into sinusoidal PE (sine/cosine replacing square waves, same periods, no jumps).
- Binary examples: pos 0 = `[0]*9`, pos 1 = `[0,0,0,0,0,0,0,0,1]`, pos 511 = `[1]*9`.

---

## Formula Summary Sheet

| Concept | Formula |
|---|---|
| RNN recurrence | $h_t = f(x_t, h_{t-1})$ |
| Nested RNN computation | $h_t = f\big(x_t, f(x_{t-1}, \ldots f(x_1, h_0)\ldots)\big)$ |
| LSTM gates | $F_t,I_t,O_t=\sigma(U_\bullet H_{t-1}+W_\bullet X_t+b_\bullet)$ |
| LSTM candidate | $\tilde C_t=\tanh(U_gH_{t-1}+W_gX_t+b_g)$ |
| LSTM cell/hidden update | $C_t=(F_t\odot C_{t-1})+(I_t\odot\tilde C_t)$, $H_t=O_t\odot\tanh(C_t)$ |
| Self-attention | $\text{Attention}(X)=\text{softmax}(QK^\top/\sqrt{d_k})V$ |
| Equivariance | $f(PX)=Pf(X)$ |
| Invariance | $f(PX)=f(X)$ |
| Position injection | $x_i\to x_i+PE_i$ |
| Raw position | $e_i'=e_i+i$ |
| Normalized position | $PE(\text{pos})=\text{pos}/(L-1)$ |
| Binary bit period | $\text{period}(i)=2^{i+1}$ |

---

## Cheat Sheet & Recall Aids

- **"Too Big, Too Relative, Too Jumpy"** — mnemonic for the 3 failed PE attempts (raw / normalized / binary).
- **RNN vs. Transformer in one line:** order free but sequential (RNN) ↔ parallel but order lost (Transformer).
- **Gotcha:** causal mask (who can see whom) ≠ positional encoding (where am I). Never confuse the two.
- **Gotcha:** equivariant ($f(PX)=Pf(X)$, output reorders with input) vs. invariant ($f(PX)=f(X)$, output unchanged) — self-attention w/o PE is equivariant, not invariant.
- **Gotcha:** normalized position encodes *relative* progress through the sequence, not absolute identity — same index, different value at different $L$.
- **Quick recall — why binary encoding almost works:** bounded + unique + length-independent + multi-frequency, but discontinuous (bit carries break smoothness) → motivates sine/cosine.
- **Self-test:**
  - [ ] Write $h_t=f(x_t,h_{t-1})$ from memory and explain why $h_3\neq h_3'$ under reordering.
  - [ ] State the attention formula and why the score is order-blind.
  - [ ] Give the equivariance and invariance equations from memory.
  - [ ] List all 3 PE attempts and their failure modes in order.
  - [ ] Compute $\text{period}(i)=2^{i+1}$ for $i=5$ and identify which positions have bit 5 = 1.
