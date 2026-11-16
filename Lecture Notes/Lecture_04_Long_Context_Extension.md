---
type: lecture
course: '[[_CS5326 Advanced GenAI|CS5326 Advanced GenAI]]'
source:
- '[[002-Long Context Extension - Complete.pdf]]'
tags:
- course/cs5326
---

%% nav:start %%
[[_CS5326 Advanced GenAI|CS5326 Advanced GenAI]] · ← [[Lecture_03_From_Sinusoidal_Limitations_to_RoPE]] · [[Lecture_05_Attention_and_Multi_Head_Attention]] →
Concepts: [[Long-Context Extension]]
%% nav:end %%

# Long-Context Extension (CS5326 — Advanced GenAI and Agents, Lecture 4)

*The RoPE (rotary positional embedding) notes ended with three problems RoPE alone doesn't solve: unfamiliar rotations at long distances, lost local detail under uniform compression, and attention dilution. This lecture covers the fixes (PI (position interpolation) → NTK (neural tangent kernel)-aware → NTK-by-parts → YaRN (Yet another RoPE extensioN)) plus ALiBi (Attention with Linear Biases) as an alternative branch.*

## Topic Inventory

- 1. The Problem and the Setup
  - 1.1 The central question and readings
  - 1.2 Conventions: four rotating hands
  - 1.3 The running baseline
- 2. Why Extrapolation Fails, Precisely
  - 2.1 The unseen-angle diagnosis
  - 2.2 Turns completed in training, $r_i$
  - 2.3 Where the crossover lands
  - 2.4 Empirical evidence
- 3. Position Interpolation
- 4. NTK-Aware Scaling
  - 4.1 Idea and mechanism
  - 4.2 Choosing the new base
  - 4.3 Running example and flaw
  - 4.4 Dynamic scaling
- 5. NTK-by-Parts
- 6. YaRN
  - 6.1 Part 1: by-parts frequencies
  - 6.2 The second problem: attention got flatter
  - 6.3 Part 2: attention temperature
  - 6.4 What sharpening can and cannot restore
  - 6.5 The running example under all methods
- 7. ALiBi: A Different Branch
- 8. Comparing the Methods
- 9. The NTK Background Behind the Name

---

## 1. The Problem and the Setup

### 1.1 The central question and readings

> **How do we fit a longer sequence into a model's positional experience while preserving useful distinctions between nearby tokens?**

The two halves conflict. Making long distances familiar means compressing positions, and compressing positions squeezes neighbours together. Every method answers "compress **what**, by **how much**, and **where**".

| # | Topic | Reference |
|---|---|---|
| 1 | Sinusoidal PE (positional encoding) | Vaswani et al. 2017, [arXiv:1706.03762](https://arxiv.org/abs/1706.03762), §3.5 only |
| 2 | RoPE | Su et al. 2024 (RoFormer (the paper that introduced RoPE)), [arXiv:2104.09864](https://arxiv.org/abs/2104.09864), §3.2-3.4 incl. long-term decay |
| 3 | Position interpolation | Chen et al. 2023, [arXiv:2306.15595](https://arxiv.org/abs/2306.15595), §2 + bound in §3 |
| 4 | NTK-aware | bloc97, Reddit r/LocalLLaMA, June 2023 (cited by Code Llama, Qwen) |
| 5 | Dynamic NTK | emozilla, Reddit r/LocalLLaMA, June 2023 |
| 6 | NTK-by-parts & YaRN | Peng et al., ICLR 2024, [arXiv:2309.00071](https://arxiv.org/abs/2309.00071), all of §3 |
| 7 | ALiBi | Press, Smith, Lewis, ICLR 2022, [arXiv:2108.12409](https://arxiv.org/abs/2108.12409) |

Optional: Jacot et al. 2018 (NTK, [arXiv:1806.07572](https://arxiv.org/abs/1806.07572), §1, 2, 5) and Tancik et al. 2020 (Fourier features, [arXiv:2006.10739](https://arxiv.org/abs/2006.10739)). See §9.

**A formula can outgrow its training.** $\phi_i(p) = p\,\omega_i$ is perfectly defined at $p = 12{,}000$ even if training stopped at 4,000. The formula doesn't break; **what the model learned** about those angles doesn't generalize. Scenario: a model trained at 4k tokens needs to handle 12-16k at inference.

### 1.2 Conventions: four rotating hands

| Symbol | Meaning | Example |
|---|---|---|
| $L$ | training context | 4,000 |
| $L'$ | target context | 16,000 |
| $s = L'/L$ | extension factor | 4 |
| $d$ | Q/K head dim | 8 |
| $b$ | RoPE base | 10,000 |
| $i$ | pair index | 0-3 |

$$\omega_i = b^{-2i/d},\qquad \phi_i(p) = p\,\omega_i\ \ (\text{angle} = \text{position} \times \text{speed})$$

With $d = 8$, $b = 10^4$: $\omega_i = 10^{-i}$.

| Pair | $\omega_i$ (rad/token) | Angle at $p = 10$ |
|---|---|---|
| 0 | 1 | 10 |
| 1 | 0.1 | 1 |
| 2 | 0.01 | 0.1 |
| 3 | 0.001 | 0.01 |

**Clock picture:** pair 0 is the second hand (fast, fine detail) and pair 3 the hour hand (slow, far distances). Extending context means making the hour hand cover more ground **without ruining the second hand**.

### 1.3 The running baseline

This recaps [[Lecture_03_From_Sinusoidal_Limitations_to_RoPE#3.4 Why rotation gives pure relative position|Lecture 3 §3.4]]. For one pair with content angles $a, b$, the angle difference is $(m-n)\theta + (a-b)$, so

$$q^\top k = \|q\|\|k\|\cos\big((m-n)\theta + (a-b)\big)$$

In matrix form: $(R(p\omega)q)^\top(R(j\omega)k) = q^\top R(-p\omega)R(j\omega)k = q^\top R((j-p)\omega)k$ (transposing reverses a rotation; rotations add).

**Baseline tracked through every method:** $m = 10$, $n = 0$, pair 1 ($\theta = 0.1$), $s = 4$, $q = k = (1,0)$ (lengths 1, $a = b = 0$):

$$q^\top k = \cos(10\times0.1) = \cos 1 \approx \boxed{0.5403}$$

The rotation $(1,0) \mapsto (\cos\phi, \sin\phi)$ gives $(0.5403, 0.8415)$ at $\phi = 1$: the arrow turns and its length stays 1. Real attention sums all four pairs; we look at one for clarity.

---

## 2. Why Extrapolation Fails, Precisely

### 2.1 The unseen-angle diagnosis

With the setup fixed, the first question is why a formula that still computes fine produces a model that fails, and exactly which pairs are to blame. Every method in this lecture is built on this diagnosis.


In training, $(m-n)\theta_i$ ranged over $[0, L\theta_i]$.
- **Fast pairs** ($L\theta_i \gg 2\pi$): saw every angle, many times. Nothing is unfamiliar.
- **Slow pairs** ($L\theta_i < 2\pi$): only saw an arc, and $W_q, W_k$ shaped $\alpha_i, \beta_i$ assuming the offset stays in that arc.

A slow pair barely rotates, so the model repurposes it as a **near-absolute channel** ("this key is early in the document"). Past $L$ it reports an angle with no learned meaning and the score becomes arbitrary. That is why perplexity **blows up within a few hundred tokens past $L$**, not gradually: the channel leaves its domain rather than accumulating error.

### 2.2 Turns completed in training, $r_i$

$$\lambda_i = \frac{2\pi}{\theta_i}\ \text{(tokens per turn)},\qquad \boxed{r_i = \frac{L}{\lambda_i} = \frac{L\theta_i}{2\pi}}\ \text{(turns during training)}$$

$r_i \gg 1$ means the pair saw the whole circle; $r_i < 1$ means it only saw an arc. **The one number every method keys off.**

### 2.3 Where the crossover lands

LLaMA (Meta's family of open large language models)-style: $d = 128$ (64 pairs), $L = 4096$.
- Pair 0: $\theta = 1$, $\lambda \approx 6.3$, $r \approx 650$ turns.
- Pair 63: $\theta \approx 1.15\times10^{-4}$, $\lambda \approx 54{,}000$, $r \approx 0.075$.

**Crossover** ($\lambda_i = L$): $\theta_i = 2\pi/4096 = 1.534\times10^{-3}$. With $\theta_i = 10^{-i/16}$, $-i/16 = \log_{10}(1.534\times10^{-3}) = -2.814$, so

$$\boxed{i \approx 45}$$

The first ~45 pairs saw the full circle; **the last ~18 never completed a turn, and that is where extrapolation dies.** PI squeezes all 64 when only 18 need help; NTK-by-parts locates this boundary exactly.

### 2.4 Empirical evidence

![[Pasted image 20260917163224.png]]
*Chen et al. 2023, Fig. 2: attention scores fitted on [0, 2048] are fine inside the range, **explode past 8000** outside it (breaking attention), and stay smooth under interpolation. Via amaarora.github.io.*

Literature (via kaiokendev's analysis):
- **Anil et al. 2022:** fine-tuning approaches fail to fix length-generalization pathologies.
- **Press et al. 2022:** Transformers overfit to trained position embeddings, even with RoPE, memorizing position-token pairs. This motivated ALiBi (§7).
- **Liu et al. 2023:** small logit fluctuations cause catastrophic long-range failure.
- **Chi et al. 2022:** constraining the receptive field *improves* extrapolation.
- **Tao et al. 2023:** rear position embeddings get fewer updates in training.

---

## 3. Position Interpolation

**Never show the model an unseen angle.** "**Keep the tokens, change the ruler.**"

$$p' = \frac{p\,L_{train}}{L_{new}} = \frac{p}{s}$$

This is $f(x, m) \to f(x, m/s)$: rotate the token at $m$ as if it were at $m/s$.

| Actual $p$ | 0 | 1 | 4 | 12,000 | 15,999 |
|---|---|---|---|---|---|
| RoPE coordinate $p/4$ | 0 | 0.25 | 1 | 3,000 | 3,999.75 |

(With $L = 4096$, $s = 2$: position 8192 gets the rotation of 4096, the largest trained angle.)

![[L04_slide13.png|600]]

*Every token is kept; only the positional coordinates are compressed into the familiar 0-4000 range.*

![[L04_pi_vs_extrapolation.png|600]]

*Top: extrapolation samples angles in an unseen range (red). Bottom: PI, $f'(x, m) = f(x, m/2)$, samples more densely inside the pretrained range.*

**Scaling position = scaling frequency:** $\frac{p}{s}\omega_i = p\frac{\omega_i}{s}$. A fast pair's 20°/token becomes 5°; a slow pair's 1° becomes 0.25°.

> [!warning] Do one or the other, never both
> Doing both divides the angle by $s^2 = 16$. PI slows the **entire spectrum uniformly**.

**Why it helps:** the slowest pair ($\omega_3 = 0.001$) spans $4000\times0.001 = 4$ rad in training. Unadjusted at 16k it would span 16 rad (unseen), but with PI, $16000\times0.00025 = 4$ rad: the **same span as training**.

**What it costs:** the relative separation becomes $\frac{p-j}{s}\omega$. Ten tokens apart at $\omega = 0.1$: 1 rad → 0.25 rad, so the contribution goes from $\cos 1 = 0.5403$ to $\cos 0.25 = \boxed{0.9689}$. Tokens have come "closer".

> [!note]
> This doesn't mean PI always raises attention; that depends on Q/K content. What is reliable is that positional **differences shrink**.

![[L04_slide22.png|600]]

*After 4× PI, positions 0-4 sit at 0, 0.25, 0.5, 0.75, 1.0. Similarly 100, 101, 102, 103 → 25.0, 25.25, 25.5, 25.75: 100 tokens occupy 25 old units.*

**Resolution loss:** adjacent tokens differ by $\theta_i/s$ instead of $\theta_i$. Pair 0 with $s = 8$: gaps 1 and 2 go from 1 rad apart to 0.125 rad, which is what **1/8 of a token** used to look like. The fast pairs were the ruler for local order, so this becomes catastrophic at high ratios. **Large distances become familiar; small distances become hard to tell apart.**

**Score under PI:** $q^\top k = \|q\|\|k\|\cos\big(\frac{m-n}{s}\theta + (a-b)\big)$. The content term is untouched.

**When is PI applied?** To an already-trained RoPE model: change the mapping, optionally keep training, and **keep the mapping at inference**. The fine-tuning data is ordinary text with next-token prediction (e.g. the Pile, long windows): load the model → use $p/4$ → feed up to 16k-token sequences → predict next tokens → backprop. No labels are needed. The model relearns a ruler whose ticks moved; it learns no new knowledge.

**Limits:** Chen et al. bound how far the attention score can move between two neighbouring (trained) positions. Because interpolated positions always fall *between* trained ones, the score is pinned by nearby known values, and the bound comes out about **600× tighter** than for extrapolated positions, which have no trained neighbours on the far side. So PI is not just a heuristic. But it needs fine-tuning (to relearn the shrunken spacing) and **degrades beyond ~8×**, because by then the fast pairs' resolution loss is too large.

---

## 4. NTK-Aware Scaling

### 4.1 Idea and mechanism

Pairs 0-45 never needed help, yet PI squeezed all 64. **Why slow every hand equally?** Keep the second hand's detail and extend the hour hand's range.

![[L04_slide24.png|600]]

*$d = 64$: pair 0 at 1 rad/token (one turn per 6.28 tokens) vs pair 28 at 0.000316 (one turn per 19,869). Over the same 64 tokens, one is a blur and the other barely moves. An intuition about scale, not a rule on what pairs can learn.*

"**Frequency-dependent scaling**" is a family: NTK-aware and NTK-by-parts are members, and YaRN picks one member and adds a temperature. Theory (§9): networks are bad at learning high-frequency structure from compressed inputs, so **don't destroy high-frequency detail**.

**NTK = Neural Tangent Kernel**, but you never compute one. The name records the motivation only.

**Basic idea:** $\omega_{new} = \omega_{old}^{\,p}$ with $p > 1$. Since $1^p = 1$, the fastest pair is untouched and smaller frequencies slow proportionally more. With $p = 2$ (illustrative only): $1 \to 1$, $0.1 \to 0.01$ (10× slower), $0.01 \to 0.0001$ (100× slower).

**Implemented as a base change**, with target $\omega_i' = \omega_i/s_i$ and per-pair slowdown $s_i$:

$$\frac{\omega_i'}{\omega_i} = \left(\frac{b}{b'}\right)^{2i/d} = \frac{1}{k^{2i/d}},\qquad k = \frac{b'}{b} > 1$$

A larger $i$ means both a slower original frequency and a stronger slowdown, so one knob gives a graded schedule.

![[L04_slide25.png|600]]

*Speed retained $\omega'/\omega$ vs pair index: PI (dashed) is flat at 0.25; NTK-aware (solid) starts at 1 and falls to 0.25 at the slowest pair.*

> [!note]
> "Slower frequencies shrink more" means a larger **percentage** reduction, not a larger absolute one.

**PI** keeps the frequencies' ratios and changes the position indices; **NTK** keeps the indices and changes the frequencies unequally.

### 4.2 Choosing the new base

$$\boxed{b' = b\,s^{d/(d-2)}}$$

This leaves the fastest pair unchanged and slows the slowest by exactly $s$. It is a construction rule, not a guarantee of useful $s$-fold extension. **Why the exponent:** the last pair is $i = (d-2)/2$, and requiring $(b/b')^{(d-2)/d} = 1/s$ gives $b'/b = s^{d/(d-2)}$. It is a boundary condition.

Example ($d = 8$, $s = 4$): $b' = 10000\cdot4^{8/6} \approx 63{,}496$ ($k \approx 6.35$), so $\omega_i'/\omega_i = s^{-2i/(d-2)} = 4^{-i/3}$.

### 4.3 Running example and flaw

| Pair | $\omega_i$ | Retained $4^{-i/3}$ | $\omega_i'$ | Slowdown |
|---|---|---|---|---|
| 0 | 1 | 1 | 1 | 1× |
| 1 | 0.1 | 0.6300 | 0.06300 | 1.59× |
| 2 | 0.01 | 0.3969 | 0.003969 | 2.52× |
| 3 | 0.001 | 0.25 | 0.00025 | 4× |

Check: $(63496/10000)^{-1/4} = 0.6300$. ✓ **The fast end stays fast, the slow end gets the full extension, and the middle changes gradually.**

Pair 16 in a $d = 64$ head ($\omega = 0.01$):

| Method | Speed retained | New step |
|---|---|---|
| RoPE | 100% | 0.010000 |
| PI | 25.0% | 0.0025000 |
| NTK-aware | 48.9% | 0.0048894 |

Same target $s$, different per-pair slowdowns. **Baseline:** $\theta' = 0.063$, giving $\cos(0.63) = \boxed{0.8080}$, between RoPE's 0.5403 and PI's 0.9689.

**Flaw: the schedule follows index $i$, not coverage $r_i$.** Slow-but-not-slowest pairs get squeezed by less than $s$ while $\lambda > L$, so they still poke off-arc. Mid pairs with dozens of turns get squeezed for nothing. It works without fine-tuning only for modest $s$.

### 4.4 Dynamic scaling

Static PI/NTK force a choice upfront: a large factor hurts short sequences, a small one limits extension. **Dynamic scaling** uses exact positions up to $L$ and scales only as needed: positions 0-2048 use scale 1.0, 4096 uses 2.0, 8192 uses 4.0.

![[L04_bloc97_ntk_perplexity.png|550]]

*bloc97, LLaMA 7B, no fine-tuning: baseline (gray) blows up past 2k; linear interpolation at scale 4 (blue dashed) is better; NTK-aware with α = 8 (green) keeps perplexity low across much longer contexts.*

![[L04_dynamic_scaling_perplexity.png|550]]

*Dynamic variants: **Dynamic NTK (orange) is best overall**, low perplexity at all lengths, and all dynamic methods avoid the static methods' catastrophic failure.*

---

## 5. NTK-by-Parts

**Decide by turns, not by index: use $r_i$ directly.**
- Many turns (high frequency, local order): already seen the whole circle, so **leave it alone**.
- Under one turn (low frequency, long range): would go off-range, so **fully interpolate** by $s$.
- In between: **blend**.

High frequencies encode local patterns (interpolate little); low frequencies encode global ones (scale fully).

With $\alpha = 1$ and $\beta = 32$ turns:

$$\gamma_i = \begin{cases}0 & r_i < 1\\ \dfrac{r_i - 1}{31} & 1 \le r_i \le 32\\ 1 & r_i > 32\end{cases}\qquad \boxed{\theta_i' = (1-\gamma_i)\frac{\theta_i}{s} + \gamma_i\theta_i = \theta_i\Big[\frac{1-\gamma_i}{s} + \gamma_i\Big]}$$

$\gamma_i$ is **how much the pair keeps its original speed**. The multiplier lies in $[1/s, 1]$.
- **Below 1 turn:** the pair only saw an arc, so any extrapolation is off-arc and it must be fully interpolated.
- **Above 32 turns:** the pair is fully periodic, and slowing it only costs resolution.
- **The ramp** avoids a cliff between adjacent pairs. **32 was chosen empirically.**

$\gamma = 0$ everywhere recovers PI; YaRN is this rule plus a temperature.

![[L04_slide40.png|600]]

*Speed retained vs original revolutions $r$ ($s = 4$): 0.25 (full slowdown) for $r \le 1$, a linear ramp to 1 (unchanged) at $r = 32$.*

**YaRN config** ($d = 128$, $L = 4096$, $s = 16$, $L_{new} = 65536$): the boundaries fall at $r = 32 \Rightarrow i \approx 21$ and $r = 1 \Rightarrow i \approx 45$.

| Pair | $r_i$ | $\gamma_i$ | Multiplier | Effect |
|---|---|---|---|---|
| 0 | 652 | 1 | 1 | untouched |
| 20 | 37 | 1 | 1 | untouched |
| 30 | 8.7 | 0.25 | 0.30 (≈ 1/3.4) | partly slowed |
| 40 | 2.1 | 0.03 | 0.094 (≈ 1/10.6) | mostly slowed |
| 63 | 0.075 | 0 | 1/16 | fully slowed |

Checks: pair 63 at 65536 reaches $65536\,\theta_{63}/16 = 4096\,\theta_{63}$, exactly its training max. Pair 0 keeps 1 rad/token.

**One pair end to end** ($d = 64$, $L = 4000$, $s = 4$, pair 16):

$$\omega = 0.01 \ \to\ r = 4000\cdot\omega/(2\pi) = 6.3662 \ \to\ \gamma = (r-1)/31 = 0.1731 \ \to\ \omega' = \Big[\frac{1-\gamma}{4} + \gamma\Big]\omega = 0.0037983$$

**Buys:** avoids both PI's blur and NTK-aware's late blowup. **Leaves open:** empirical thresholds, blended pairs still partly off-distribution, and **nothing done about attention diluted over $s\times$ more keys.**

---

## 6. YaRN

**YaRN = NTK-by-parts (fixes geometry) + attention temperature (fixes concentration).** It changes **no weights and no attention kernel**. It only edits the RoPE cos/sin table (a per-pair multiplier plus one global scalar), then fine-tunes briefly (often < 1% of pretraining compute). It reaches 128K+ with minimal perplexity loss and keeps fine positional detail.

### 6.1 Part 1: by-parts frequencies

This is §5 unchanged: (1) $r_i = L\theta_i/2\pi$; (2) $\gamma_i$ with thresholds 1 and 32; (3) $\theta_i' = (1-\gamma_i)\theta_i/s + \gamma_i\theta_i$.

### 6.2 The second problem: attention got flatter

This happens **even with perfect positions**.

**Cause 1, more competitors:** $w = e^{z^*}/(e^{z^*} + \sum_{\text{others}} e^z)$. Toy example with the right key at logit 9 and all others at 0:
- 4096 keys: $e^9/(e^9 + 4095) \approx 66\%$.
- 65,536 keys: $\approx 11\%$.

Same logits, **6× less focus**, and every higher layer was trained on the 66% version.

**Cause 2, slowed pairs make keys look alike:** smaller angle differences mean clustered logits, which means a flatter softmax. This is **created by Part 1**, which is why the two parts go together.

Frequency scaling changes geometry; it doesn't set softmax concentration. Toy: one score of 2.5 against others at 0 gets **80.2%** with 3 competitors but **28.2%** with 31.

### 6.3 Part 2: attention temperature

$$\text{softmax}\Big(\frac{1}{t}\cdot\frac{q_m\cdot k_n}{\sqrt d}\Big),\qquad \frac1t > 1$$

The flattening is **order-preserving** (relevant scores are still higher, just bunched), so multiplying every score by more than 1 stretches the gaps and restores the preference that was already there. The later layers expect a certain peakiness, and how much to restore depends on $s$.

**Baked into the RoPE cache:** multiply the cos/sin table by $c = \sqrt{1/t}$. Both $q$ and $k$ pick up $c$:

$$(cR_mq)\cdot(cR_nk) = c^2(R_mq)\cdot(R_nk) = \frac1t\,q_m\cdot k_n$$

One constant, **no attention-code change**, works with fused kernels, **zero inference overhead**.

> [!note] Watch the squares
> $c$ multiplies $q$ and $k$; $c^2 = 1/t$ multiplies the score. Sources quote both.

**Choosing it** (YaRN fit over many $s$):

$$\boxed{\sqrt{1/t} = c = 0.1\ln s + 1}$$

Each doubling of $s$ adds $0.1\ln2 \approx 0.07$: sharpen a little, and more slowly as context grows.

| $s$ | $c$ (on $q$, $k$) | $c^2$ (on score) |
|---|---|---|
| 8 | 1.21 | 1.46 |
| 16 | 1.28 | 1.63 |
| 32 | 1.35 | 1.81 |

> [!warning] Column labels
> The slide leaves these columns unlabeled and uses "$1/t$" for both. Check: $0.1\ln8 + 1 = 1.208$ (column 1) and $1.208^2 = 1.459$ (column 2).

Example: 31 competitors with advantage 2.5, $c = 1 + 0.1\ln4 = 1.139$, $t = 1/c^2 = 0.771$: **28.2% → 45.2%**. **Equal scores stay equal.**

### 6.4 What sharpening can and cannot restore

![[L04_dilution_a.png|650]]
![[L04_dilution_b.png|650]]

*Three relevant tokens (logits 3.0, 2.5, 2.0) among distractors near 0. Left column: no scaling ($\kappa = 1$) at N = 10, 30, 60. Right column: sharpened.*

- **Order is preserved:** 3.0 > 2.5 > 2.0 in every panel. Dilution changes heights, not ranking.
- **Flattening is pure dilution:** the relevant share is 85% at N = 10, 59% at 30, 40% at 60. Twenty extra distractors at ≈ 1.5% each add up.
- **The restoring $\kappa$ grows slowly:** 1.51 brings N = 30 back to 85%; N = 60 needs 1.79. Doubling $s$ (3 → 6) adds only 0.28, which is the log behaviour behind $0.1\ln s + 1$.
- **Average-case, not exact:** the top token comes back at 50% / 54%, not the original 43%. Uniform sharpening restores the **winners' share**, not the distribution's **shape**. It helps focused heads, mildly hurts diffuse ones, and its constants were fitted on one model family, hence brief fine-tuning.

### 6.5 The running example under all methods

Pair 1, $d = 8$, $m = 10$, $n = 0$, $q = k = (1,0)$, $s = 4$:

| Method | $\theta'$ | Angle diff | Contribution |
|---|---|---|---|
| RoPE | 0.1 | 1 | $\cos 1 = 0.5403$ |
| PI | 0.025 | 0.25 | $\cos 0.25 = 0.9689$ |
| NTK-aware | 0.063 | 0.63 | $\cos 0.63 = 0.8080$ |
| YaRN | 0.1 | 1 | $\cos 1 = 0.5403$ → × $c^2$ = **0.7005** |

YaRN leaves this pair alone: it completed ≈ 63.66 revolutions (> 32), so $\gamma = 1$. PI and NTK slowed it needlessly. The temperature then multiplies the score by $c^2 = 1.29648$.

> [!warning] Which constant is 1.29648?
> The slide calls it "$c$", but $0.1\ln4 + 1 = 1.1386$ and $1.1386^2 = 1.2965$, so it is the score multiplier $c^2$.

Scope: this is one pair. Real attention sums all pairs, divides by $\sqrt d$, then applies softmax. Shifting both tokens by 100 ($p = 110$, $j = 100$) gives the same $\cos(-1)$: relative only.

---

## 7. ALiBi: A Different Branch

ALiBi isn't the next RoPE extension. It **drops rotation** and adds distance as a score bias.

| | RoPE family | ALiBi |
|---|---|---|
| Mechanism | Rotate projected Q, K | Bias the content score |
| Score | $q^\top R((j-p)\omega)k$ | $q^\top k/\sqrt d - m(p-j)$ |
| Effect | Position changes geometry | Distance adds a recency preference |

**Distance fee** with head slope $m = 0.10$ and equal content score 6.0:

| Distance | Content | Penalty | Final |
|---|---|---|---|
| 2 (near) | 6.0 | −0.20 | 5.80 |
| 20 (far) | 6.0 | −2.00 | 4.00 |

**Linear fee = exponential discount:**

$$\exp(u - mD) = \exp(u)\times\exp(-mD)$$

Unnormalized support = content support × distance discount.

![[L04_slide46.png|550]]

*$\exp(-mD)$ with $m = 0.1$ decays smoothly from 1 toward 0 over 100 tokens. It is a multiplier, not a probability.*

**A bias, not a ban:** the extra penalty is $0.1\times(20-2) = 1.8$. With equal content, near gets 85.8% and far 14.2%. A content advantage above 1.8 flips it.

**Trade-off:** extrapolates to any length for free (no procedure, no fine-tuning), but the fixed recency bias caps long-range retrieval, gives one monotone distance signal instead of a multi-frequency fingerprint, and loses RoPE's relative-position richness.

---

## 8. Comparing the Methods

| Method | Core idea / exact change | Downside |
|---|---|---|
| **RoPE** | Rotate Q, K by angle ∝ position; score depends on distance | Slow pairs hit unseen angles past $L$; collapse within a few hundred tokens |
| **PI** | Divide every position (frequency) by $s$ | Blurs fast pairs (local order); needs fine-tuning; degrades beyond ~8× |
| **NTK-aware** | Raise the base: fast pairs ≈ untouched, slow pairs compressed | Schedule by index, not coverage; some pairs still off-arc, mid pairs squeezed for nothing; zero-shot only for modest $s$ |
| **NTK-by-parts** | By turns: <1 compress, >32 keep, blend between | Empirical thresholds; blended pairs partly off-distribution; ignores dilution |
| **YaRN** | By-parts + logit sharpening $c = 0.1\ln s + 1$, brief fine-tune | One average temperature (helps focused, hurts diffuse heads); constants may not transfer; still needs fine-tuning |
| **ALiBi** | No rotation; subtract $m\times$ distance | Free extrapolation, but recency bias caps retrieval; no multi-frequency fingerprint |

**A longer coordinate range still needs learned behaviour and affordable compute.** These methods teach nothing new; they keep what was learned applicable. (The slide figures are illustrative mathematics, not benchmarks.)

---

## 9. The NTK Background Behind the Name

"NTK-aware" has been used since §4 without justifying the name. This section supplies the theory; nothing in it is needed to *apply* the methods, but it explains why preserving the fast pairs matters.


- **NTK (Jacot et al. 2018):** a very wide network trained by gradient descent behaves like a fixed kernel method. Some patterns are learned fast, others slowly.
- **Eigen-directions** are the basic patterns the network can learn; the **eigenvalue** is how fast it learns each one. The network learns the target one eigen-direction at a time. For a 1-D position input: slowly varying (low frequency) patterns have large eigenvalues and are learned fast; rapidly varying (high frequency) ones have small eigenvalues and are learned slowly. This is **spectral bias**, and it means a raw position index makes fine structure hard to learn.
- **Fourier features (Tancik et al. 2020):** feed $\gamma(x) = [\sin\omega_1x, \cos\omega_1x, \dots, \sin\omega_kx, \cos\omega_kx]$ instead of $x$. The kernel changes from $K(x, x')$ to $K(\gamma(x), \gamma(x'))$ because its inputs changed. $x = 100$ and $101$ are near-identical scalars, but their high-frequency features can differ a lot, so high-frequency patterns get larger effective eigenvalues. Analogy: make the important difference **visible** (red vs blue), not bigger. RoPE's multi-frequency rotations *are* such Fourier features.

**Painter analogy.** The network is a painter and its input features are brushes. It learns broad strokes fast and fine detail almost never **unless the fine detail is already in its inputs**. It can't invent a fine brush. Slow waves are broad brushes; fast waves are fine brushes. RoPE's fast pairs are the fine brushes that separate "the token right before me" from "two back". **PI shrinks every brush by $s$**, and fine-tuning can't fix that: *you can't train a painter to draw hairlines with a roller.*

So "**NTK-aware**" means **aware that resolution is capped by the finest brush you provide**. Keep the fine brushes and enlarge only the broad ones (which are what run out of canvas). NTK-aware does this with one base change; NTK-by-parts does it brush by brush, by turns completed. The name comes from the motivation; no NTK is ever computed.

All four lectures so far treated attention itself as a black box that consumes rotated $q$ and $k$. [[Lecture_05_Attention_and_Multi_Head_Attention]] opens that box, including the $\sqrt d$ scaling that YaRN's temperature adjusts.

---

## Numerical Problems

**Problem 1 — Extrapolation boundary.** $d = 64$, $b = 10^4$, $L = 2048$.

$\theta_i = 2\pi/2048 = 3.068\times10^{-3}$ and $\theta_i = 10^{-i/8}$, so $-i/8 = \log_{10}(3.068\times10^{-3}) = -2.513$ and $i \approx 20.1$.

$$\boxed{i \approx 20;\ \text{pairs } 20\text{-}31 \ (\approx 12 \text{ of } 32) \text{ at risk}}$$

Halving $L$ halves every $r_i$ and moves the boundary lower (compare 45/64 for $L = 4096$, $d = 128$).

**Problem 2 — PI resolution.** $L = 4096 \to 32{,}768$, so $s = 8$. Pair 0: $\theta_0' = 0.125$ rad/token.

$$\boxed{\text{one token now looks like } 1/8 \text{ of an old token}}$$

This sits right at PI's ~8× ceiling.

**Problem 3 — NTK-aware base.** $d = 64$, $s = 8$.

$b' = 10^4\cdot8^{64/62} = 10^4\cdot e^{1.0323\ln8} = 10^4\cdot8.559 \approx 85{,}590$, so $b/b' = 0.11683$.
- Pair 16: $0.11683^{0.5} = 0.3418$.
- Pair 31: $0.11683^{0.96875} = 0.1251$.

$$\boxed{b' \approx 85{,}590;\ \text{pair 16 keeps } 34.2\%;\ \text{pair 31 keeps } 12.5\% = 1/s}$$

Under PI, pair 16 would keep only 12.5%.

**Problem 4 — By-parts ramp.** $d = 128$, $L = 4096$, $s = 16$, $i = 35$.

1. $\theta_{35} = 10^{-4\cdot70/128} = 10^{-2.1875} = 6.494\times10^{-3}$.
2. $r = 4096\,\theta/2\pi = 4.233$.
3. $\gamma = 3.233/31 = 0.1043$.
4. Multiplier $= 0.8957/16 + 0.1043 = 0.1603$.

$$\boxed{r \approx 4.23,\ \gamma \approx 0.104,\ \times0.160\ (\approx 1/6.2),\ \theta' \approx 1.04\times10^{-3}}$$

This lies between pair 30 (≈ 1/3.4) and pair 40 (≈ 1/10.6). ✓

**Problem 5 — YaRN temperature.** $s = 32$: $c = 0.1\ln32 + 1 = 1.3466$ and $c^2 = 1.8133$. Relevant logit 3.0 against 200 competitors at 0:
- Before: $e^3/(e^3 + 200) = 20.09/220.09 = 9.13\%$.
- After: logit $3\times1.8133 = 5.44$, so $230.4/430.4 = 53.5\%$.

$$\boxed{c \approx 1.347,\ c^2 \approx 1.813;\ 9.1\% \to 53.5\%}$$

The winner doesn't change; only the gap widens.

---

## Formula Summary Sheet

| Concept | Formula |
|---|---|
| Pair frequency / angle | $\omega_i = b^{-2i/d}$; $\phi_i(p) = p\,\omega_i$ |
| RoPE score (one pair) | $\|q\|\|k\|\cos((m-n)\theta + (a-b))$; $q^\top R((j-p)\omega)k$ |
| Wavelength / turns | $\lambda_i = 2\pi/\theta_i$; $r_i = L/\lambda_i = L\theta_i/2\pi$ |
| Crossover | $\theta_i = 2\pi/L$ |
| Extension factor | $s = L'/L$ |
| PI | $p' = p/s \equiv \theta_i/s$ (not both) |
| PI score | $\cos(\frac{m-n}{s}\theta + (a-b))$ |
| NTK-aware | $\omega_{new} = \omega_{old}^p$; $b' = b\,s^{d/(d-2)}$; $\omega_i'/\omega_i = (b/b')^{2i/d} = s^{-2i/(d-2)}$ |
| By-parts ramp | $\gamma_i$: 0 if $r<1$; $(r-1)/31$ if $1\le r\le32$; 1 if $r>32$ |
| By-parts frequency | $\theta_i' = \theta_i[(1-\gamma_i)/s + \gamma_i]$ |
| Dilution | $e^{z^*}/(e^{z^*} + \sum e^z)$; toy: $e^g/(e^g+N)$ |
| YaRN temperature | $\text{softmax}(\frac1t\cdot\frac{q\cdot k}{\sqrt d})$; $c = \sqrt{1/t} = 0.1\ln s + 1$ |
| Cache trick | $(cR_mq)\cdot(cR_nk) = c^2(R_mq)\cdot(R_nk)$ |
| ALiBi | $q^\top k/\sqrt d - m(p-j)$; $\exp(u - mD) = e^u e^{-mD}$ |
| Fourier features | $\gamma(x) = [\sin\omega_jx, \cos\omega_jx]_j$ |

---

## Cheat Sheet & Recall Aids

- **Story:** slow RoPE pairs never completed a turn, so they break first past $L$. Each method is a rule for **which hands to slow and by how much**; YaRN adds a separate fix for softmax dilution.
- **Key number:** $r_i = L\theta_i/2\pi$. PI ignores it, NTK-aware approximates it by index, by-parts uses it directly.
- **Mnemonic "All, Index, Turns, Turns+Temp, None":** PI, NTK-aware, NTK-by-parts, YaRN, ALiBi.
- **Anchors:** $d = 128$, $L = 4096$ gives crossover $i \approx 45$; $r_0 \approx 650$, $r_{63} \approx 0.075$; thresholds 1 and 32; PI ≤ ~8×; $c = 0.1\ln s + 1$.
- **Gotcha:** dividing the position OR the frequency by $s$ is the same thing; both gives $s^2$.
- **Gotcha:** $c$ multiplies q, k; $c^2$ multiplies the score.
- **Abrupt failure:** a slow pair used as an "early in document" channel exits its domain.
- **Painter:** no hairlines with a roller, so fine-tuning can't recover PI's lost resolution.
- **Temperature:** restores the winners' share, not the distribution's shape; equal scores stay equal.
- **ALiBi:** a linear fee is an exponential discount; recency is a bias, not a ban.

**Self-test**
- [ ] Solve $\lambda_i = L$ for the crossover given $d$, $L$.
- [ ] Why $d/(d-2)$ in the NTK base?
- [ ] State the by-parts ramp and justify both thresholds.
- [ ] Two causes of flattening; which one Part 1 creates.
- [ ] Derive the $c^2$ score multiplier from scaling the cos/sin table.
- [ ] What each method does to the 0.5403 baseline.
