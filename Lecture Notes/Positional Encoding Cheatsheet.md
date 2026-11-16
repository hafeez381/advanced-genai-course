---
type: study-aid
course: '[[_CS5326 Advanced GenAI|CS5326 Advanced GenAI]]'
tags:
- course/cs5326
---

# Positional Encoding — Sinusoidal Embeddings and RoPE (Cheatsheet)

## Topic Inventory

- [[Lecture_02_Sinusoidal_Positional_Encoding#3. The Sinusoidal Positional Encoding Formula|1. Sinusoidal Positional Encoding]]
  - [[Lecture_02_Sinusoidal_Positional_Encoding#4. Choosing the Frequencies|1.1 Frequency Formula]]
  - [[Lecture_02_Sinusoidal_Positional_Encoding#8. Why Both Sine and Cosine?|1.2 Sine/Cosine Pairing and Linear Transformability]]
  - [[Lecture_03_From_Sinusoidal_Limitations_to_RoPE#2. The Entanglement Problem|1.3 Entanglement Problem in Attention]]
- [[Lecture_03_From_Sinusoidal_Limitations_to_RoPE#3. Rotary Position Embeddings (RoPE)|2. Rotary Position Embeddings (RoPE)]]
  - [[Lecture_03_From_Sinusoidal_Limitations_to_RoPE#3.2 The key insight: reorder the operations|2.1 Core Idea: Rotate, Don't Add]]
  - [[Lecture_03_From_Sinusoidal_Limitations_to_RoPE#3.3 The rotation operation in two dimensions|2.2 2D Rotation and the Attention Score]]
  - [[Lecture_03_From_Sinusoidal_Limitations_to_RoPE#3.5 The same result through matrix algebra|2.3 Matrix Proof of Relative-Position Property]]
  - [[Lecture_03_From_Sinusoidal_Limitations_to_RoPE#5. Does Rotation Corrupt Semantic Meaning?|2.4 Semantic Meaning Is Preserved]]
  - [[Lecture_03_From_Sinusoidal_Limitations_to_RoPE#4. Scaling RoPE to Full Dimensions|2.5 Extension to Full Dimensions (Block-Diagonal Rotation)]]
  - [[Lecture_03_From_Sinusoidal_Limitations_to_RoPE#6. Limitations of RoPE|2.6 Long-Range Behavior, Aliasing, and Failure at Extreme Distances]]

---

## 1. Sinusoidal Positional Encoding

### 1.1 Frequency Formula

$$PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d}}\right), \qquad PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

- $\omega_i = 1/10000^{2i/d}$ — dimension $i=0$ → fastest wave; $i=d/2$ → slowest wave.
- Base $10000$: slowest period $\approx 2\pi \times 10000 \approx 62{,}832$ positions.
- Base too small (e.g. 100) → repetition, distant positions collide.
- Base too large (e.g. $10^8$) → most dims nearly flat/useless.

### 1.2 Sine/Cosine Pairing and Linear Transformability

$$PE(pos)=(\sin(pos\,\theta),\ \cos(pos\,\theta))$$
$$PE(pos+k) = M_k \cdot PE(pos), \qquad M_k=\begin{bmatrix}\cos(k\theta) & \sin(k\theta)\\ -\sin(k\theta) & \cos(k\theta)\end{bmatrix}$$

- $M_k$ depends only on shift $k$, not on $pos$.
- Sine alone → cannot recover $\cos(pos\,\theta)$ → no fixed linear shift matrix exists.
- Cosine is required to make the shift a pure rotation.

### 1.3 Entanglement Problem in Attention

$$input_m = embed_m + PE_m$$
$$Q_m\cdot K_n = \underbrace{(W_q e_m)\cdot(W_k e_n)}_{\text{semantic}} + \underbrace{(W_q e_m)\cdot(W_k PE_n)}_{\text{cross}} + \underbrace{(W_q PE_m)\cdot(W_k e_n)}_{\text{cross}} + \underbrace{(W_q PE_m)\cdot(W_k PE_n)}_{\cos(\theta(m-n))}$$

Three flaws:
- **Entanglement** — clean relative signal buried in Term 4, mixed with 2 cross terms.
- **No direct relative position** — only absolute $PE_m$ stored; relative position is a byproduct.
- **Fused representation** — $embed_m+PE_m$ can never be separated downstream.

---

## 2. Rotary Position Embeddings (RoPE)

### 2.1 Core Idea: Rotate, Don't Add

- Goal: $Q_m\cdot K_n = f(\text{semantics}, m-n)$ — single clean term, no cross terms.
- Order flip: project first ($Q=W_qe$, $K=W_ke$), **then** rotate by position — position never touches the embedding or $W_q,W_k$.

### 2.2 2D Rotation and the Attention Score

$$\begin{bmatrix}q_0^{new}\\q_1^{new}\end{bmatrix}=\begin{bmatrix}\cos(m\theta) & -\sin(m\theta)\\ \sin(m\theta)&\cos(m\theta)\end{bmatrix}\begin{bmatrix}q_0\\q_1\end{bmatrix}$$

$$\boxed{Q_m\cdot K_n=\cos\big((\alpha-\beta)+(m-n)\theta\big)}$$

- $\alpha,\beta$ = directions of $Q,K$ in **query/key space** (post-projection, learned — not embedding-space angles).
- $(\alpha-\beta)$: pure semantic term, untouched by rotation.
- $(m-n)\theta$: pure relative-position term.

### 2.3 Matrix Proof of Relative-Position Property

$$Q_m\cdot K_n=(R_mQ)^\top(R_nK)=Q^\top R_m^\top R_n K = Q^\top R_{-m}R_nK=Q^\top R_{n-m}K$$

- Uses orthogonality: $R_m^\top=R_m^{-1}=R_{-m}$, and $R_{-m}R_n=R_{n-m}$.
- Depends only on $n-m$; absolute $m,n$ vanish individually.

### 2.4 Semantic Meaning Is Preserved

- Meaning = angle **between** $Q,K$ (i.e. $\alpha-\beta$), not either vector's absolute direction.
- $m=n \Rightarrow$ score $=\cos(\alpha-\beta)$ exactly (no rotation effect).
- Rotation preserves length: $|Rv|^2=v^\top R^\top Rv=v^\top v=|v|^2$ — only direction turns, magnitude (often carries learned info) untouched.

### 2.5 Extension to Full Dimensions (Block-Diagonal Rotation)

- Split $d$-dim vector into $d/2$ pairs $(q_0,q_1),(q_2,q_3),\dots$
- Pair $i$ frequency: $\theta_i = 1/10000^{2i/d}$ (same formula as sinusoidal).
- Each pair rotated independently — no cross-pair mixing.

$$R(m)=\begin{bmatrix}R(m\theta_0)&0&\cdots&0\\0&R(m\theta_1)&\cdots&0\\ \vdots&\vdots&\ddots&\vdots\\0&0&\cdots&R(m\theta_{d/2-1})\end{bmatrix}$$

### 2.6 Long-Range Behavior, Aliasing, and Failure at Extreme Distances

- Score $=\cos((\alpha-\beta)+(m-n)\theta)$: semantic and positional terms **compete**.
- Gap 34 (slow pair, $\theta\approx0.0001$): extra angle $\approx0.0034$ rad $\approx0.2°$ — negligible; semantic term dominates. Multi-head attention lets some heads specialize in long-range via slow pairs.
- Gap 32000: angle $\approx3.2$ rad $\approx183°$ → **cosine flips sign** → positional term actively fights the correct semantic signal. Real, known RoPE limitation → motivates long-context extension research.
- **Aliasing**: cosine repeats every $2\pi$; two gaps $g_1,g_2$ alias in a pair if $(g_1-g_2)\theta_i=2\pi k$.
  - Fast pairs alias every ~6 tokens (fine, only used for local precision).
  - Slow pairs alias only after ~62,832 tokens (never within normal context).
  - Full spectrum together: gaps alias only if they alias in *every* band simultaneously → gives near-unique fingerprint per gap in practice.
  - **Real danger is not aliasing, but sign-flip past 180°** in slow pairs at extreme distances (see above) — this is the structural failure mode, not confusion between two distances.

---

## Formula Summary Sheet

| Concept | Formula |
|---|---|
| Sinusoidal frequency | $\omega_i = \dfrac{1}{10000^{2i/d}}$ |
| Sinusoidal encoding | $PE(pos,2i)=\sin\!\big(pos/10000^{2i/d}\big)$, $PE(pos,2i+1)=\cos\!\big(pos/10000^{2i/d}\big)$ |
| Shift matrix (sinusoidal) | $PE(pos+k)=M_k\cdot PE(pos)$, $M_k=\begin{bmatrix}\cos k\theta&\sin k\theta\\-\sin k\theta&\cos k\theta\end{bmatrix}$ |
| Attention score (sinusoidal, added PE) | $Q_m\cdot K_n=$ Term1(sem)+Term2(cross)+Term3(cross)+Term4$(\cos(\theta(m-n)))$ |
| RoPE 2D rotation | $\begin{bmatrix}q_0^{new}\\q_1^{new}\end{bmatrix}=\begin{bmatrix}\cos m\theta&-\sin m\theta\\ \sin m\theta&\cos m\theta\end{bmatrix}\begin{bmatrix}q_0\\q_1\end{bmatrix}$ |
| RoPE attention score | $\boxed{Q_m\cdot K_n=\cos\big((\alpha-\beta)+(m-n)\theta\big)}$ |
| RoPE matrix form | $Q_m\cdot K_n=Q^\top R_{n-m}K$ |
| Rotation length preservation | $\lvert Rv\rvert^2=v^\top v=\lvert v\rvert^2$ |
| Per-pair RoPE frequency | $\theta_i=1/10000^{2i/d}$ |
| Aliasing condition (per pair) | $(g_1-g_2)\theta_i=2\pi k,\ k\in\mathbb{Z}$ |

---

## Cheat Sheet & Recall Aids

**Mnemonic (evolution of PE):** *Raw → Normalized → Binary → Sinusoidal → RoPE* — each step fixes exactly one flaw of the previous (scale → consistency → smoothness → entanglement).

**One-line gotchas:**
- Sinusoidal *adds* position to embedding → 4-term attention expansion, 2 are noise (cross terms). RoPE *rotates* Q/K after projection → 1 clean term, no cross terms.
- Cosine is not redundant with sine: sine alone can't be linearly shifted (missing $\cos(pos\,\theta)$ term); pairing sine+cosine makes the shift a fixed rotation matrix $M_k$.
- $\alpha,\beta$ in RoPE are angles of $Q,K$ **after** projection — not the raw embedding's angle. $W_q,W_k$ are rectangular → they reshape, not just rotate, so embedding angle and query angle are unrelated. Meaning of $(\alpha-\beta)$ comes entirely from training, not from embedding geometry.
- Rotation doesn't corrupt meaning: attention meaning lives in the *angle between* Q and K, not either one's absolute direction; rotating both by their own position leaves $\alpha-\beta$ intact and only adds the relative term.
- RoPE breaks at extreme range not because of "aliasing confusion" but because slow-pair cosine **flips sign past 180°**, actively opposing the correct semantic score.

**Quick-reference intuition (one sentence each):**
- Sinusoidal encoding: smooth, bounded, unique per-position vector; relative position hides inside the dot product's Term 4 but gets mixed with semantic noise.
- RoPE: apply position as a rotation on Q/K after projection so the dot product depends purely on semantics + relative distance, nothing else.
- Base 10000: tunes frequency spread so fast dims resolve nearby positions and slow dims stay unique across the whole context without going flat.
- Block-diagonal rotation: full-dimension RoPE is just many independent 2D rotations, one per frequency pair, stacked with zero cross-pair mixing.

**Self-test checklist:**
- [ ] Can derive $\omega_i=1/10000^{2i/d}$ and explain why base 10000 (not too small/large).
- [ ] Can explain why cosine is necessary for linear transformability (sine-only failure argument).
- [ ] Can expand $Q_m\cdot K_n$ into 4 terms for sinusoidal PE and name each term's meaning.
- [ ] Can state RoPE's design goal and why swapping add→rotate achieves it.
- [ ] Can derive $Q_m\cdot K_n=\cos((\alpha-\beta)+(m-n)\theta)$ from the 2D rotation.
- [ ] Can prove via matrices that $Q_m\cdot K_n$ depends only on $n-m$ (orthogonality + $R_{-m}R_n=R_{n-m}$).
- [ ] Can explain why rotation preserves semantic meaning (angle-between argument + length preservation).
- [ ] Can explain the block-diagonal extension to full dimensions and per-pair frequency assignment.
- [ ] Can compute the rotation angle at gap=34 vs gap=32000 for a slow pair and explain why one is fine and one flips sign.
- [ ] Can distinguish "aliasing" from the "cosine sign-flip" failure mode and say which is the real long-context problem.
