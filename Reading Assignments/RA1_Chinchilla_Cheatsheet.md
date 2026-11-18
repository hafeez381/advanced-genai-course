---
type: study-aid
course: '[[_CS5326 Advanced GenAI|CS5326 Advanced GenAI]]'
tags:
- course/cs5326
---

# Training Compute-Optimal Large Language Models (Chinchilla) — Cheatsheet

*Scope: Sections 1–5 (pp. 1–16) only. Acknowledgements, References, and all Appendices (A–J) excluded.*

## Topic Inventory

- 1. Introduction
  - 1.1 Problem Statement & Central Claim
  - 1.2 Current LLM Landscape (Table 1)
- 2. Related Work
  - 2.1 Large Language Models
  - 2.2 Modelling Scaling Behaviour — Divergence from Kaplan et al. (2020)
  - 2.3 Estimating Other Hyperparameters
  - 2.4 Improved Model Architectures
- 3. Estimating the Optimal Parameter/Training-Token Allocation
  - 3.1 Approach 1 — Fixed Model Sizes, Varying Training Tokens
  - 3.2 Approach 2 — IsoFLOP Profiles
  - 3.3 Approach 3 — Parametric Loss Function Fit
  - 3.4 Optimal Model Scaling — Consolidated Results
- 4. Chinchilla
  - 4.1 Model & Training Details
  - 4.2 Results
    - 4.2.1 Language Modelling
    - 4.2.2 MMLU
    - 4.2.3 Reading Comprehension
    - 4.2.4 BIG-bench
    - 4.2.5 Common Sense
    - 4.2.6 Closed-Book Question Answering
    - 4.2.7 Gender Bias & Toxicity
- 5. Discussion & Conclusion

---

## 1. Introduction

**Central claim:** current LLMs are **undertrained** — model size has scaled up while token count stayed roughly fixed (~300B).

$$N_{opt}(C), D_{opt}(C) = \underset{N,D \text{ s.t. FLOPs}(N,D)=C}{\arg\min} L(N,D) \quad (1)$$

- $L(N,D)$: pre-training loss as function of param count $N$, training tokens $D$.
- $C$: fixed FLOPs budget, deterministic function $\text{FLOPs}(N,D)$.
- **Result: $N$ and $D$ should scale equally** (each doubling of $N$ ⇒ double $D$) — contradicts Kaplan et al. (2020): 10× compute ⇒ 5.5× $N$, 1.8× $D$ (model-size-heavy).
- Verified via **Chinchilla**: 70B params, 1.4T tokens, same compute budget as **Gopher** (280B, 300B tokens) → uniformly outperforms Gopher, GPT-3 (175B), Jurassic-1 (178B), MT-NLG (530B).
- MMLU: Chinchilla 67.5% avg (>7% over Gopher).

**Table 1 — Current LLMs**

| Model | Size | Training Tokens |
|---|---|---|
| LaMDA | 137B | 168B |
| GPT-3 | 175B | 300B |
| Jurassic-1 | 178B | 300B |
| Gopher | 280B | 300B |
| MT-NLG 530B | 530B | 270B |
| **Chinchilla** | **70B** | **1.4T** |

> **[Possible source inconsistency]** — §5 refers to "GPT-3's 170 billion parameters," but Table 1 (and standard literature) lists GPT-3 at 175B. Reproduced as given.

---

## 2. Related Work

### 2.1 Large Language Models
- Dense transformers (GPT-3, Jurassic-1, Gopher, MT-NLG, LaMDA) vs. **Mixture-of-Experts (MoE)** (GLaM, Switch Transformer).
- Largest dense transformer > 500B params.

### 2.2 Modelling Scaling Behaviour
- Kaplan et al. (2020): power-law between $N$ and loss; models shouldn't train to lowest possible loss to be compute-optimal (Hoffmann et al. agree on this point only).
- **Two key methodological divergences from Kaplan et al.:**
  1. Kaplan et al. fixed token count + LR schedule across all model sizes → can't isolate hyperparameter effects. Hoffmann et al.: **matching cosine LR decay length to token horizon $D$ gives the best final loss**, regardless of model size. A fixed 130B-token schedule makes intermediate-loss points at $D_0 \ll 130B$ *overestimates* → underestimates value of training on less data → biases conclusion toward "grow $N$ faster."
  2. Hoffmann et al. include models up to 16B params (Kaplan et al.: mostly <100M) — needed because of curvature in the FLOP–loss frontier (Appendix E).
- Clark et al. (2022): MoE scaling w.r.t. #experts, same fixed-token-count limitation.

### 2.3 Estimating Other Hyperparameters
- Not addressed directly here: LR, LR schedule, batch size, optimiser, width/depth ratio.
- McCandlish et al. (2018): weak dependence of optimal batch size on model size.
- Levine et al. (2020): optimal depth/width ratio — Hoffmann et al. use **slightly shallower** models for better wall-clock time.

### 2.4 Improved Model Architectures
- MoE: Switch Transformer (1.7T params), GLaM (1.2T) — large *effective* size, fewer FLOPs; benefit diminishes at very large scale (Clark et al., 2022).
- Retrieval-augmented (RETRO/Borgeaud et al. 2021, REALM, RAG): effectively increases tokens seen (~10× in RETRO).

---

## 3. Estimating the Optimal Parameter/Training-Token Allocation

Common assumption across all 3 approaches: **power-law relationship** between compute and model size.

### 3.1 Approach 1 — Fixed Model Sizes, Varying Training Tokens
- Models: 70M–10B params, 4 token-horizons each (16× range), LR decays 10× over horizon.
- Per run: smooth + interpolate loss curve → mapping FLOPs → loss.
- Per FLOP count: pick run w/ min loss → mapping $C \to (N,D)$ at 1500 log-spaced FLOP values.
- Fit $N_{opt} \propto C^a$, $D_{opt} \propto C^b$.

$$\boxed{a = 0.50, \quad b = 0.50}$$

- Head-to-head at $10^{21}$ FLOPs: Approach-1-predicted model beats Kaplan-et-al.-predicted model (App. D.4).

### 3.2 Approach 2 — IsoFLOP Profiles
- Model size varied for 9 **fixed** FLOP budgets ($6\times10^{18}$ to $3\times10^{21}$); consider only **final** training loss (not full curve).
- Cosine schedule length = target token count (optimal per App. B).
- Fit parabola to each IsoFLOP curve → loss-minimizing $N$ per FLOP budget → fit power law.

$$\boxed{a = 0.49, \quad b = 0.51}$$

### 3.3 Approach 3 — Parametric Loss Function Fit

$$\hat{L}(N,D) \triangleq E + \frac{A}{N^\alpha} + \frac{B}{D^\beta} \quad (2)$$

| Term | Meaning |
|---|---|
| $E$ | Bayes risk / entropy of natural text (ideal generative process) |
| $A/N^\alpha$ | Functional-approximation gap: a perfectly-trained $N$-param transformer still underperforms ideal process |
| $B/D^\beta$ | Optimisation gap: finite training steps, single-epoch sample |

**Fitting** — minimize Huber loss ($\delta = 10^{-3}$) via L-BFGS, best of grid initializations:

$$\min_{A,B,E,\alpha,\beta} \sum_{\text{Runs } i} \text{Huber}_\delta\left(\log \hat{L}(N_i,D_i) - \log L_i\right) \quad (3)$$

**Efficient frontier** (minimizing $\hat L$ s.t. $\text{FLOPs}(N,D)\approx 6ND$):

$$N_{opt}(C) = G\left(\frac{C}{6}\right)^a,\quad D_{opt}(C) = G^{-1}\left(\frac{C}{6}\right)^b,\quad G = \left(\frac{\alpha A}{\beta B}\right)^{\frac{1}{\alpha+\beta}},\ a=\frac{\beta}{\alpha+\beta},\ b=\frac{\alpha}{\alpha+\beta} \quad (4)$$

$$\boxed{a = 0.46, \quad b = 0.54}$$

(Predicts even *smaller* optimal models at large $C$ — Huber loss down-weights low-FLOP outlier points, and there is empirical negative curvature in $\log N_{opt}$, App. E.)

### 3.4 Optimal Model Scaling — Consolidated Results

**Table 2 — Scaling exponents ($N_{opt}\propto C^a$, $D_{opt}\propto C^b$)**

| Approach | $a$ | $b$ |
|---|---|---|
| 1. Min over training curves | 0.50 | 0.50 |
| 2. IsoFLOP profiles | 0.49 | 0.51 |
| 3. Parametric loss modelling | 0.46 | 0.54 |
| Kaplan et al. (2020) | 0.73 | 0.27 |

**Table 3 — Compute-optimal recipe by model size (Approach 1)**

| Params | FLOPs | FLOPs (Gopher units) | Tokens |
|---|---|---|---|
| 400M | 1.92e19 | 1/29,968 | 8.0B |
| 1B | 1.21e20 | 1/4,761 | 20.2B |
| 10B | 1.23e22 | 1/46 | 205.1B |
| **67B** | **5.76e23** | **1** | **1.5T** |
| 175B | 3.85e24 | 6.7 | 3.7T |
| 280B | 9.90e24 | 17.2 | 5.9T |
| 520B | 3.43e25 | 59.5 | 11.0T |
| 1T | 1.27e26 | 221.3 | 21.2T |
| 10T | 1.30e28 | 22,515.9 | 216.2T |

- 175B model: optimal at $4.41\times10^{24}$ FLOPs, **4.2T tokens**.
- 280B (Gopher-like): optimal at $\sim10^{25}$ FLOPs, **6.8T tokens**.
- 1T model: not optimal unless budget $\geq 10^{26}$ FLOPs (>250× Gopher's compute).
- Confirmed on C4 and GitHub-code datasets too (App. C) — same equal-scaling conclusion.

---

## 4. Chinchilla

- Gopher-compute-optimal size ∈ [40B, 70B] → chose **70B** (upper end, dataset/compute-efficiency reasons) trained on **1.4T tokens**.
- Same FLOPs as Gopher; 4× smaller ⇒ cheaper fine-tuning + inference.

### 4.1 Model & Training Details

**Table 4 — Architecture comparison**

| Model | Layers | Heads | K/V size | $d_{model}$ | Max LR | Batch size |
|---|---|---|---|---|---|---|
| Gopher 280B | 80 | 128 | 128 | 16,384 | $4\times10^{-5}$ | 3M→6M |
| Chinchilla 70B | 80 | 64 | 128 | 8,192 | $1\times10^{-4}$ | 1.5M→3M |

(feed-forward size $= 4\times d_{model}$ for both)

**Differences from Gopher (same MassiveText dataset, different sampling %):**
- **AdamW** (not Adam) → better LM loss + downstream fine-tuning.
- Modified **SentencePiece** tokenizer, **no NFKC normalization** (94.15% vocab overlap w/ Gopher) — helps math/chemistry representation.
- Forward/backward pass in **bfloat16**; **float32** master copy of weights kept in distributed optimiser state.
- Trained on TPUv3/TPUv4, JAX + Haiku.

### 4.2 Results

**Table 5 — Evaluation task families**

| Category | #Tasks | Examples |
|---|---|---|
| Language Modelling | 20 | WikiText-103, The Pile, arXiv, FreeLaw |
| Reading Comprehension | 3 | RACE-m, RACE-h, LAMBADA |
| Question Answering | 3 | Natural Questions, TriviaQA, TruthfulQA |
| Common Sense | 5 | HellaSwag, Winogrande, PIQA, SIQA, BoolQ |
| MMLU | 57 | HS Chemistry, Astronomy, Clinical Knowledge |
| BIG-bench | 62 | Causal Judgement, Epistemic Reasoning |

#### 4.2.1 Language Modelling
- Outperforms Gopher on **all** Pile subsets (bpb).
- vs. Jurassic-1 (178B): worse only on `dm_mathematics`, `ubuntu_irc`.
- Wikitext103 perplexity: **7.16** (Chinchilla) vs **7.75** (Gopher).
- ⚠ Caution: 4× more training data → possible train/test leakage on these LM benchmarks → more weight placed on MMLU/BIG-bench/QA/common-sense.

#### 4.2.2 MMLU

**Table 6**

| System | Accuracy |
|---|---|
| Random | 25.0% |
| Avg. human rater | 34.5% |
| GPT-3 (5-shot) | 43.9% |
| Gopher (5-shot) | 60.0% |
| **Chinchilla (5-shot)** | **67.6%** |
| Avg. human expert | 89.8% |
| Forecast (June 2022) | 57.1% |
| Forecast (June 2023) | 63.4% |

> **[Possible source inconsistency]** — Abstract states 67.5% average MMLU accuracy; Table 6 states 67.6%. Reproduced as given.

- Chinchilla **beats the June-2023 human forecast early**.
- >90% accuracy on 4 tasks: `high_school_gov_and_politics`, `international_law`, `sociology`, `us_foreign_policy`.
- vs. Gopher: better on 51/57 tasks, tied 2/57, **worse on 4**: `college_mathematics`, `econometrics`, `moral_scenarios`, `formal_logic`.

#### 4.2.3 Reading Comprehension

**Table 7**

| Task | Chinchilla | Gopher | GPT-3 | MT-NLG 530B |
|---|---|---|---|---|
| LAMBADA (0-shot) | **77.4** | 74.5 | 76.2 | 76.6 |
| RACE-m (few-shot) | **86.8** | 75.1 | 58.1 | – |
| RACE-h (few-shot) | **82.3** | 71.6 | 46.8 | 47.9 |

(RACE improvement over Gopher > 10 pts both subsets.)

#### 4.2.4 BIG-bench
- Avg accuracy: Chinchilla **65.1%** vs Gopher **54.4%** (+10.7%).
- Worse on 4/62 tasks: `crash_blossom`, `dark_humor_detection`, `mathematical_induction`, `logical_args`.

#### 4.2.5 Common Sense

**Table 8**

| Task | Chinchilla | Gopher | GPT-3 | MT-NLG 530B |
|---|---|---|---|---|
| HellaSWAG | **80.8%** | 79.2% | 78.9% | 80.2% |
| PIQA | 81.8% | 81.8% | 81.0% | **82.0%** |
| Winogrande | **74.9%** | 70.1% | 70.2% | 73.0% |
| SIQA | **51.3%** | 50.6% | – | – |
| BoolQ | **83.7%** | 79.3% | 60.5% | 78.2% |

- Beats Gopher & GPT-3 on **all**; beats MT-NLG on **all but PIQA**.
- TruthfulQA: Chinchilla 43.6% / 58.5% / 66.7% (0/5/10-shot) vs Gopher 29.5% / – / 43.7%. Large 0-shot gain (+14.1%) — contrasts Lin et al. (2021)'s claim that scale alone doesn't help TruthfulQA.

#### 4.2.6 Closed-Book Question Answering

**Table 9**

| Benchmark | Shot | Chinchilla | Gopher | GPT-3 |
|---|---|---|---|---|
| Natural Questions | 0 | 16.6% | 10.1% | 14.6% |
| | 5 | **31.5%** | 24.5% | – |
| | 64 | **35.5%** | 28.2% | 29.9% |
| TriviaQA (unfiltered) | 0 | 67.0% | 52.8% | 64.3% |
| | 5 | **73.2%** | 63.6% | – |
| | 64 | 72.3% | 61.3% | 71.2% |
| TriviaQA (filtered) | 0 | 55.4% | 43.5% | – |
| | 5 | **64.1%** | 57.0% | – |
| | 64 | **64.6%** | 57.2% | – |

- New closed-book **SOTA** on Natural Questions.
- Filtered TriviaQA: Chinchilla trails open-book SOTA (72.5%, Izacard & Grave 2020) by 7.9%.

> **[Possible source inconsistency]** — Body text states Gopher's NQ accuracy as "21% and 28%" (5-shot / 64-shot); Table 9 lists 24.5% / 28.2%. Reproduced as given.

#### 4.2.7 Gender Bias & Toxicity

**Winogender (zero-shot coreference), Table 10:**

| Group | Chinchilla | Gopher |
|---|---|---|
| All | 78.3% | 71.4% |
| Male | 71.2% | 68.0% |
| Female | 79.6% | 71.3% |
| Neutral | 84.2% | 75.0% |
| Male gotcha | 62.5% | 59.2% |
| Female gotcha | **76.7%** | 66.7% |

- "**Gotcha**" examples = correct resolution *contradicts* gender stereotype.
- Improvement uneven: male pronouns +3.2%, female +8.3%, neutral +9.2%; largest gain on female-gotcha (+10%) → improvement not uniform across groups despite overall gain.

**Sample toxicity** (25,000 unprompted generations, PerspectiveAPI):

| Stat | Gopher | Chinchilla |
|---|---|---|
| Mean | 0.081 | 0.087 |
| Median | 0.064 | 0.066 |
| 95th pct | 0.230 | 0.238 |

- Negligible difference → **unconditional-generation toxicity is largely independent of model quality/LM loss** (consistent with Rae et al. 2021's parameter-scaling finding, now shown for compute-optimal scaling too).

---

## 5. Discussion & Conclusion

- Trend critiqued: model size ↑ (MT-NLG 530B is >3× GPT-3's param count) while tokens stayed ~300B across nearly all major models.
- **Hypothesis confirmed**: the race for larger models yields models that *underperform* what the same compute could achieve if reallocated toward more data.
- 400+ training runs, 3 independent methods → **Gopher is substantially over-sized**; Chinchilla (70B/1.4T) empirically confirms this, beating Gopher and larger models on nearly every task.
- **Limitations (explicitly stated):**
  1. Only **2** comparable large-scale runs exist (Chinchilla, Gopher) — no intermediate-scale confirmation.
  2. Assumes strict **power-law** form; observed concavity in $\log N_{opt}$ at high compute (App. E) suggests optimal size may be **even smaller** than predicted.
  3. All runs use **< 1 epoch** of data — multi-epoch regime unexplored.
- **Forward-looking claims:** future scaling should prioritize **dataset size/quality** over raw parameter count; larger scraped datasets raise train/test-overlap and privacy/toxicity-quantity concerns (not necessarily frequency); similar $N$–$D$ trade-off is expected to generalize to **other modalities** beyond autoregressive text LMs.

---

## Formula Summary Sheet

| Concept | Formula |
|---|---|
| Compute-optimal allocation objective | $N_{opt}(C), D_{opt}(C) = \arg\min_{N,D:\ \text{FLOPs}(N,D)=C} L(N,D)$ |
| Power-law scaling (params) | $N_{opt} \propto C^{a}$ |
| Power-law scaling (tokens) | $D_{opt} \propto C^{b}$ |
| Approach 1 exponents | $a=0.50,\ b=0.50$ |
| Approach 2 exponents | $a=0.49,\ b=0.51$ |
| Approach 3 exponents | $a=0.46,\ b=0.54$ |
| Kaplan et al. (2020) exponents | $a=0.73,\ b=0.27$ |
| Parametric loss model | $\hat{L}(N,D) = E + \dfrac{A}{N^\alpha} + \dfrac{B}{D^\beta}$ |
| Fitting objective (Huber/L-BFGS) | $\min_{A,B,E,\alpha,\beta} \sum_i \text{Huber}_\delta\!\left(\log \hat{L}(N_i,D_i)-\log L_i\right)$ |
| Efficient frontier (closed form) | $N_{opt}(C)=G\!\left(\frac{C}{6}\right)^{a},\ D_{opt}(C)=G^{-1}\!\left(\frac{C}{6}\right)^{b}$ |
| Frontier constant | $G = \left(\dfrac{\alpha A}{\beta B}\right)^{\frac{1}{\alpha+\beta}}$ |
| Frontier exponents (from $\alpha,\beta$) | $a=\dfrac{\beta}{\alpha+\beta},\quad b=\dfrac{\alpha}{\alpha+\beta}$ |
| FLOPs approximation used for frontier | $\text{FLOPs}(N,D) \approx 6ND$ |

---

## Cheat Sheet & Recall Aids

**Mnemonic — the 3 approaches:** *"Fix, Iso, Param"* → **F**ix model size & vary tokens (1) → **Iso**FLOP curves, vary size at fixed compute (2) → **Param**etric loss-surface fit (3). All three land near $a\approx b\approx 0.5$.

**One-line intuition per approach:**
- Approach 1: "Which token-horizon gives lowest loss for each FLOP count, per fixed model size?"
- Approach 2: "At a fixed compute budget, which model size sits at the bottom of the loss valley?"
- Approach 3: "Fit an explicit $N,D\to$loss surface, then solve for the frontier analytically."

**Gotchas:**
- Chinchilla is **4× smaller** than Gopher **and** trained on **4× more tokens**, for the **same FLOPs** — don't confuse this with "just a bigger dataset."
- Kaplan et al. (2020) says grow $N$ fast, $D$ slow ($a=0.73,b=0.27$); Chinchilla says grow **both equally** ($a\approx b\approx 0.5$). The exponents must sum differently, not just differ in magnitude.
- $\text{FLOPs}\approx 6ND$ is an **approximation** (App. F shows true ratio ranges ~0.99–1.10× across model sizes) — used only to derive the closed-form frontier in Approach 3.
- MMLU 67.6% (Table 6) vs. 67.5% (abstract) — cite whichever the question quotes; don't "correct" one to match the other.
- Toxicity is **flat** across model quality — bigger/better LM ≠ more (or less) toxic unprompted output.
- Winogender gains are **uneven by pronoun group** — "Chinchilla reduces bias" is an overstatement; it improves accuracy unevenly (female-gotcha gains most).

**Quick-reference (if you remember nothing else):**
> *Same compute, smaller model, more data, longer LR schedule matched to token count → lower loss. Gopher was 4× too big for its training budget.*

**Recall checklist:**
- [ ] State Eq. (1) and what $N_{opt}(C), D_{opt}(C)$ mean.
- [ ] Name all 3 approaches and their resulting $(a,b)$.
- [ ] Explain each of the 3 terms in $\hat L(N,D) = E + A/N^\alpha + B/D^\beta$.
- [ ] State Chinchilla's params/tokens and how it compares to Gopher's architecture (Table 4).
- [ ] Name at least 2 benchmark families where Chinchilla underperforms Gopher (MMLU: college_mathematics, econometrics, moral_scenarios, formal_logic; BIG-bench: crash_blossom, dark_humor_detection, mathematical_induction, logical_args).
- [ ] State the 3 explicit limitations from §5.
