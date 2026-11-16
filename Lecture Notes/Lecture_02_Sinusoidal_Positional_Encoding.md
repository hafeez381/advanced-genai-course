---
type: lecture
course: '[[_CS5326 Advanced GenAI|CS5326 Advanced GenAI]]'
source:
- '[[001-PositionAndOrdering-2.pdf]]'
tags:
- course/cs5326
---

%% nav:start %%
[[_CS5326 Advanced GenAI|CS5326 Advanced GenAI]] · ← [[Lecture_01_Position_and_Ordering]] · [[Lecture_03_From_Sinusoidal_Limitations_to_RoPE]] →
Concepts: [[Sinusoidal Positional Encoding]]
%% nav:end %%

# Sinusoidal Positional Encoding (CS5326 — Advanced GenAI and Agents, Lecture 2)

*Lecture 1 ended with binary encoding: right **structure** (stacked waves at geometrically decreasing frequencies, one vector per position), wrong **waveform** (square waves that jump at carries). This lecture keeps the structure and swaps the waveform.*

## Topic Inventory

- [[#1. From Square Waves to Sine Waves|1. From Square Waves to Sine Waves]]
- [[#2. Building a Multi-Frequency Sinusoidal Encoding|2. Building a Multi-Frequency Sinusoidal Encoding]]
- [[#3. The Sinusoidal Positional Encoding Formula|3. The Sinusoidal Positional Encoding Formula]]
  - [[#3.1 The formula and the sin/cos pair|3.1 The formula and the sin/cos pair]]
  - [[#3.2 What the denominator does|3.2 What the denominator does]]
  - [[#3.3 The encoding in the embedding pipeline|3.3 The encoding in the embedding pipeline]]
- [[#4. Choosing the Frequencies|4. Choosing the Frequencies]]
- [[#5. Why Is the Base 10,000?|5. Why Is the Base 10,000?]]
- [[#6. Visualizing the Encoding|6. Visualizing the Encoding]]
  - [[#6.1 The positional encoding heatmap|6.1 The positional encoding heatmap]]
  - [[#6.2 Wavelengths across dimension pairs|6.2 Wavelengths across dimension pairs]]
- [[#7. The Wave Vocabulary|7. The Wave Vocabulary]]
- [[#8. Why Both Sine and Cosine?|8. Why Both Sine and Cosine?]]
  - [[#8.1 Linear transformability|8.1 Linear transformability]]
  - [[#8.2 The rotation matrix|8.2 The rotation matrix]]
  - [[#8.3 Why sine alone cannot do this|8.3 Why sine alone cannot do this]]

---

## 1. From Square Waves to Sine Waves

The smoothest periodic function is the **sine wave** (cosine is the same wave at a different phase). It rises and falls continuously, so <mark style="background: #FFF3A3A6;font-weight: bold;">two positions that are near each other will always produce sine values that are near each other.</mark>

![[Pasted image 20260908154529.png]]
*Square vs. sine at the same frequency: same period, same rate. The only difference is how they get from low to high: the square wave jumps, the sine moves smoothly. The sine also spans $[-1, 1]$, matching embedding scale.*

---

## 2. Building a Multi-Frequency Sinusoidal Encoding

Stack sine waves at different frequencies, just as binary encoding stacked bits:

$$PE(pos, i) = \sin(pos\cdot\omega_i)$$

$\omega_i$ is the frequency of dimension $i$. Fast dimensions play the LSB (least significant bit) role; slow dimensions play the MSB (most significant bit) role. This leaves three questions: (1) how does the scheme work, (2) how do we choose $\omega_i$, (3) is sine enough, or do we need cosine?

**Background: Euler's formula.**

$$e^{ix} = \cos x + i\sin x$$

![[L02_slide3.png|600]]

*"The two shadows": $e^{ix}$ is where you are on the unit circle; $ie^{ix}$ is your velocity, a quarter turn ahead with length 1, so elapsed time $x$ equals the angle. Growth perpendicular to your position is circular. Cosine and sine are the **horizontal and vertical shadows of one rotating point**, not two independent features. §8 builds on this.*

---

## 3. The Sinusoidal Positional Encoding Formula

### 3.1 The formula and the sin/cos pair

$$PE_{(pos,\,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{model}}}\right),\qquad PE_{(pos,\,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$pos$ is the token position; $d_{model}$ is the embedding size (e.g. 512); $i$ indexes **pairs**, with even dims getting sine and odd dims cosine.

Dims $2i$ and $2i+1$ are **one point on the unit circle**, not two features. With $\theta_i = pos/10000^{2i/d_{model}}$, Euler gives

$$PE(pos, 2i+1) + j\,PE(pos, 2i) = e^{j\theta_i}$$

The order looks swapped only because the Transformer stores $(\sin, \cos)$ while Euler writes $(\cos, \sin)$.

### 3.2 What the denominator does

<mark style="background: #FFF3A3A6;">Low-index dimensions oscillate rapidly, capturing fine-grained local position differences; high-index dimensions oscillate slowly, capturing long-range position differences. The mechanism is the denominator </mark>$10000^{2i/d_{model}}$, <mark style="background: #FFF3A3A6;">which controls the wavelength (time period) of each sinusoid.</mark>

As $i$ grows the denominator grows exponentially, giving a smooth fast-to-slow spectrum and a unique fingerprint per position. The real payoff (proved in §8): $PE_{pos+k}$ is a **fixed linear (rotation) function of $PE_{pos}$**, which makes relative position learnable by attention.

### 3.3 The encoding in the embedding pipeline

![[Pasted image 20260914152754.png]]
*StatQuest (a statistics YouTube channel): one-hot token → Embeddings Lookup → add, per dimension, the value of a stack of alternating sin/cos waves of decreasing frequency, sampled at the token's position. As the position advances, the sample point slides right along every wave: fast waves change a lot, slow ones barely move. A token's PE is a vertical slice through the stack.*

---

## 4. Choosing the Frequencies

We need **high** frequencies to separate neighbours sharply and **low** ones to carry long-range information without repeating:

$$\omega_i = \frac{1}{10000^{2i/d}}$$

- $i = 0$: $\omega_0 = 1$, changes every position (the **fast bit**).
- $i = d/2$: $\omega = 1/10000$, barely moves over thousands of positions (the **slow bit**).
- In between, frequencies decrease **geometrically**, the continuous analogue of bit-period doubling.

Dividing $x$ by a constant stretches the wave by that factor:

| Wave | Cycles over [0, 100] | Behaviour |
|---|---|---|
| $\sin(x)$ | ≈ 15.9 | Baseline, cycle every $2\pi$ |
| $\sin(x/10)$ | ≈ 1.59 | Cycle ≈ 63 positions |
| $\sin(x/50)$ | ≈ 0.32 | Gentle curve |
| $\sin(x/100)$ | ≈ 0.16 | Under one cycle; neighbours look identical |

![[Pasted image 20260914153440.png|448]]
![[Pasted image 20260914153507.png|446]]
![[Pasted image 20260914153522.png|444]]
![[Pasted image 20260914153534.png|440]]
![[Pasted image 20260914153840.png|474]]
*The four waves individually and overlaid on [0, 100]: binary's stack of square waves, made smooth. Each wave tells the model something different about position, and the base 10000 controls the whole spectrum.*

---

## 5. Why Is the Base 10,000?

The slowest dimension is roughly $\sin(pos/B)$, which repeats when $pos/B = 2\pi$:

$$\boxed{pos \approx 2\pi B}$$

| Base $B$ | Slowest wave repeats after ≈ |
|---|---|
| 100 | 628 positions |
| 10,000 | 62,832 positions |
| 100 million | 628 million positions |

![[Pasted image 20260914154005.png]]
*$\sin(pos/B^{2i/256})$ for bases 100, 10,000, $10^8$ (rows) at dims 0, 64, 96, 127 (columns), positions 0-2000.*

| | dim 0 | dim 64 | dim 96 | dim 127 |
|---|---|---|---|---|
| **B = 100** | fast | many cycles | many cycles | several cycles in 1000 positions |
| **B = 10,000** | fast | ≈ 2 cycles / 1000 | < 1 cycle, smooth rise (separates 100 from 800) | nearly flat (0 → 0.1) |
| **B = $10^8$** | fast | nearly a straight line | flat at 0 | flat at 0 |

- dim 0 is **always fast**, whatever the base, since $B^0 = 1$.
- **Too small (100):** even the slowest dims oscillate, so far-apart positions can collide in every dimension.
- **Too large ($10^8$):** most dims are flat and carry no information.
- **10,000 is the Goldilocks choice:** no repetition within typical context lengths, and a spectrum that is still useful.

---

## 6. Visualizing the Encoding

### 6.1 The positional encoding heatmap

![[Pasted image 20260914154126.png]]
*Lilian Weng: rows = position (0-31), columns = dimension (0-127). Left: rapid checkerboard (fast waves, separate neighbours). Moving right: stripes widen (slow waves, long-range structure). Far right: near-uniform bars. Every **row** has a unique fingerprint; no single dimension identifies a position, but the combination of fast, medium and slow does.*

### 6.2 Wavelengths across dimension pairs

For $d_{model} = 512$: $\tau_i = 10000^{2i/512}$, $\lambda_i = 2\pi\tau_i$, $i = 0..255$.

| $i$ | $\tau_i$ | $\lambda_i$ (positions) | |
|---|---|---|---|
| 0 | 1.00 | 6.3 | "who is next to me" |
| 1 | 1.04 | 6.5 | |
| 2 | 1.07 | 6.8 | |
| 3 | 1.11 | 7.0 | early dims similar (exponential starts slowly) |
| 128 | 100 | 628 | coarse, long-range |
| 200 | 1,334 | 8,378 | |
| 253 | 8,977 | 56,405 | |
| 255 | 9,647 | 60,616 | ≈ linear over realistic lengths |

A geometric progression from $2\pi$ to almost $2\pi\cdot10000$.

> [!warning] Possible source inconsistency
> A later slide repeats this table with $i = 127$ as "halfway" ($\tau = 100$), $\tau_{200} = 1{,}445$, $\tau_{253} = 7{,}880$, and $\tau_{255}$ = "the full 10,000" ($\lambda \approx 62{,}832$). The formula gives $\tau_{128} = 100$ exactly and $\tau_{255} = 10000^{510/512} \approx 9{,}647$, so the table above is the consistent one. The conclusion (≈ 6 to ≈ 60,000 positions per cycle) is the same either way.

![[Pasted image 20260914155820.png|525]]
*$\sin(pos/\tau_i)$ for $i = 0, 1, 3$ (three bunched fast waves) and $i = 127$ (nearly linear).*

**$\tau_i$ is positions per radian, not the wavelength.** One step forward adds $1/\tau_i$ radians, and a cycle is $2\pi$ radians, so $\lambda_i = 2\pi\tau_i$ positions. Check: $\tau_0 = 1$ gives $\lambda_0 \approx 6.3$.

---

## 7. The Wave Vocabulary

For $\sin(\omega x + \phi)$, every quantity answers **how fast the angle turns as $x$ advances**. A **radian** is arc length / radius: a pure number, with a full circle $= 2\pi$. Radians (unlike the human-chosen 360°) make $\frac{d}{dx}\sin x = \cos x$ with no conversion constant.

| Symbol | Name | Units | Meaning |
|---|---|---|---|
| $\omega$ | angular frequency | rad per unit $x$ | radians turned per step |
| $f$ | frequency | cycles per unit $x$ | cycles per step |
| $T$ / $\lambda$ | period / wavelength | $x$ per cycle | distance to repeat (period for time, wavelength for space) |
| $\tau$ | $1/\omega$ | $x$ per radian | distance to turn one radian |

$$\omega = 2\pi f,\qquad f = 1/T,\qquad T = 2\pi/\omega$$

For the PE: $\omega_i = 1/\tau_i$, $f_i = 1/(2\pi\tau_i)$, $\lambda_i = 2\pi\tau_i$. At $i = 0$: 1 rad/token, 0.16 cycles/token, 6.3 tokens/cycle.

**Phase.** $\cos(\omega x) = \sin(\omega x + \pi/2)$. The sin/cos pair is one wave sampled a quarter cycle apart (a **quadrature pair**), which is what lets a fixed rotation carry $PE_{pos}$ to $PE_{pos+k}$.

---

## 8. Why Both Sine and Cosine?

Sine alone already gives a smooth, bounded, unique, multi-frequency encoding. So why does the paper put sine on even dims and cosine on odd ones? Because of **linear transformability**.

### 8.1 Linear transformability

We want $PE(pos+k)$ to be $PE(pos)$ times a matrix that **depends only on $k$**, not on $pos$. Attention's $W_q$ and $W_k$ are linear layers, so a fixed-matrix relationship is something attention can learn to use for **relative** position.

### 8.2 The rotation matrix

Take one frequency $\theta$ and $PE(pos) = (\sin(pos\,\theta), \cos(pos\,\theta))$. Using $\sin(A+B) = \sin A\cos B + \cos A\sin B$ and $\cos(A+B) = \cos A\cos B - \sin A\sin B$ with $A = pos\,\theta$, $B = k\theta$:

$$\sin((pos+k)\theta) = \sin(pos\,\theta)\cos(k\theta) + \cos(pos\,\theta)\sin(k\theta)$$
$$\cos((pos+k)\theta) = \cos(pos\,\theta)\cos(k\theta) - \sin(pos\,\theta)\sin(k\theta)$$

Both are linear in $(\sin(pos\,\theta), \cos(pos\,\theta))$ with constant coefficients:

$$\boxed{PE(pos+k) = M_k\,PE(pos),\qquad M_k = \begin{bmatrix}\cos k\theta & \sin k\theta\\ -\sin k\theta & \cos k\theta\end{bmatrix}}$$

$M_k$ depends only on $k$: it is a 2D rotation by $k\theta$. Shifting position rotates each pair's point on its circle by a fixed angle.

### 8.3 Why sine alone cannot do this

With only $\sin(pos\,\theta)$ stored, the expansion of $\sin((pos+k)\theta)$ needs $\cos(pos\,\theta)$, which we don't have. Could we recover it?

$$\sin^2\theta + \cos^2\theta = 1 \Rightarrow \boxed{\cos\theta = \pm\sqrt{1-\sin^2\theta}}$$

The $\pm$ is fatal: **one sine value maps to two phases**. $30°$ and $150°$ both give $\sin = 0.5$ (with $\cos = \pm0.87$), and they move in **opposite** directions when shifted. The recovery is also **nonlinear**, so $W_q, W_k$ couldn't absorb it anyway.

![[Pasted image 20260914160147.png|510]]
*Two points with the same height $\sin\theta = 0.5$ on opposite sides of the circle: one projection loses the phase, two coordinates recover it.*

**Corollary:** the encoding is $d/2$ **pairs** sharing one $\omega_i$ each, not $d$ independent frequencies. A pair is one feature (a point on a circle) stored so that it supports linear shifts.

So far the encoding has been studied in isolation. [[Lecture_03_From_Sinusoidal_Limitations_to_RoPE]] asks what happens once it is added to the embeddings and passed through $W_q, W_k$, and finds that this clean rotation structure gets tangled with content, which motivates RoPE (rotary positional embedding).

---

## Numerical Problems

**Problem 1 — PE by hand.** $d_{model} = 4$, $pos = 3$.

$\tau_0 = 1$, $\tau_1 = 10000^{0.5} = 100$. Pair 0: $\theta = 3$ gives $(\sin 3, \cos 3) = (0.1411, -0.9900)$. Pair 1: $\theta = 0.03$ gives $(0.0300, 0.9996)$.

$$\boxed{PE(3) \approx (0.1411, -0.9900, 0.0300, 0.9996)}$$

The fast pair is nearly halfway around its circle; the slow pair has barely moved.

**Problem 2 — Wavelength of pair $i = 64$** ($d_{model} = 512$).

$\tau_{64} = 10000^{128/512} = 10^{4\times0.25} = 10$, so $\omega = 0.1$ rad/token and $\lambda = 20\pi \approx 62.8$. Over 1000 tokens: $1000/62.8 \approx 15.9$ cycles.

$$\boxed{\tau = 10,\ \lambda \approx 62.8,\ \approx 15.9 \text{ cycles}}$$

This is exactly the $\sin(x/10)$ wave from §4.

**Problem 3 — Verify $M_3$.** $\theta = 1$: $PE(2) = (0.9093, -0.4161)$ and $M_3 = \begin{bmatrix}-0.9900 & 0.1411\\ -0.1411 & -0.9900\end{bmatrix}$.

$M_3PE(2) = (-0.9001 - 0.0587,\ -0.1283 + 0.4119) = (-0.9588, 0.2836)$, and $PE(5) = (\sin 5, \cos 5) = (-0.9589, 0.2837)$. ✓

$$\boxed{M_3\,PE(2) \approx PE(5)}$$

$M_3$ never used the starting position 2, so it would equally map $PE(10) \to PE(13)$.

**Problem 4 — Base 100 vs 10,000 at $i = 255$** ($d_{model} = 512$). The exponent is $510/512 = 0.9961$.

- $B = 100$: $\tau = 10^{1.992} \approx 98.2$, $\lambda \approx 617$, so $1000/617 \approx 1.6$ cycles in 1000 tokens. It wraps.
- $B = 10^4$: $\tau = 10^{3.984} \approx 9647$, $\lambda \approx 60{,}616$, so $0.016$ cycles. No wrap.

$$\boxed{B = 100 \text{ wraps within 1000 tokens};\ B = 10^4 \text{ does not}}$$

---

## Formula Summary Sheet

| Concept | Formula |
|---|---|
| General form | $PE(pos, i) = \sin(pos\,\omega_i)$ |
| Euler | $e^{ix} = \cos x + i\sin x$ |
| Sinusoidal PE | $PE_{(pos,2i)} = \sin(pos/10000^{2i/d_{model}})$, $PE_{(pos,2i+1)} = \cos(\cdot)$ |
| Pair as complex number | $PE(pos,2i+1) + j\,PE(pos,2i) = e^{j\theta_i}$, $\theta_i = pos/10000^{2i/d_{model}}$ |
| Frequency | $\omega_i = 1/10000^{2i/d}$ |
| Divisor / wavelength | $\tau_i = 10000^{2i/d}$ (positions/radian); $\lambda_i = 2\pi\tau_i$ |
| Slowest repeat | $pos \approx 2\pi B$ |
| Wave relations | $\omega = 2\pi f$, $f = 1/T$, $T = 2\pi/\omega$ |
| Quadrature | $\cos(\omega x) = \sin(\omega x + \pi/2)$ |
| Angle addition | $\sin(A+B) = \sin A\cos B + \cos A\sin B$; $\cos(A+B) = \cos A\cos B - \sin A\sin B$ |
| Shift matrix | $PE(pos+k) = M_kPE(pos)$, $M_k = \begin{bmatrix}\cos k\theta & \sin k\theta\\ -\sin k\theta & \cos k\theta\end{bmatrix}$ |
| Sign ambiguity | $\cos\theta = \pm\sqrt{1-\sin^2\theta}$ |

---

## Cheat Sheet & Recall Aids

- **Story:** keep binary's frequency spectrum, swap square waves for sin/cos, use base 10000 so the slowest wave doesn't repeat, and store sin **and** cos so a shift is a fixed rotation.
- **Mnemonic "SBP": Smooth, Base, Pair.** Smooth fixes discontinuity; Base fixes the frequency spread; Pair fixes linear transformability.
- **Gotcha:** $\tau_i$ is positions per **radian**; the wavelength is $2\pi\tau_i$.
- **Gotcha:** $i$ indexes **pairs** (0-255 for $d = 512$), occupying dims $2i, 2i+1$.
- **Base:** too small, slow dims wrap and positions collide; too big, dims are flat and useless; dim 0 is fast for any base.
- **Why cosine:** the shift needs $\cos(pos\,\theta)$, and $\pm\sqrt{1-\sin^2}$ is ambiguous and nonlinear.
- **Anchor numbers:** $\lambda_0 \approx 6.3$, $\lambda_{128} \approx 628$, $\lambda_{255} \approx 60{,}616$; slowest repeat ≈ $2\pi B$.

**Self-test**
- [ ] Write both PE formulas from memory.
- [ ] Convert between $\tau$, $\omega$, $f$, $\lambda$.
- [ ] Derive $M_k$ from the angle-addition identities.
- [ ] Explain why $M_k$ being independent of $pos$ matters for $W_q, W_k$.
- [ ] Give the $30°/150°$ counterexample.
