---
type: study-aid
course: '[[_CS5326 Advanced GenAI|CS5326 Advanced GenAI]]'
source:
- '[[Rafailov2023_DPO.pdf]]'
tags:
- course/cs5326
---

%% nav:start %%
[[_CS5326 Advanced GenAI|CS5326 Advanced GenAI]] · Full notes: [[RA2_DPO]]
%% nav:end %%

# Direct Preference Optimization (DPO) — Cheatsheet

*Scope: Abstract and Sections 1–7 (pp. 1–10) only. References and Appendices are excluded.*

## Topic Inventory

- 1. Problem and Contribution
- 2. Related Work
- 3. RLHF Pipeline (Preliminaries)
  - 3.1 SFT
  - 3.2 Reward Modelling
  - 3.3 RL Fine-Tuning
- 4. DPO
  - 4.1 Derivation Chain
  - 4.2 Loss and Gradient
  - 4.3 Pipeline
- 5. Theory
  - 5.1 Equivalence Classes, Lemmas, Theorem 1
  - 5.2 Actor-Critic Instability
- 6. Experiments
  - 6.1 Setup
  - 6.2 Results
- 7. Limitations
- Formula Summary Sheet
- Cheat Sheet & Recall Aids
- Practice Questions (new, not in RAQ4)
  - MCQs (correct answers only)
  - Short Questions

---

## 1. Problem and Contribution

**Abbreviations used throughout:**
- LM = language model.
- RLHF / RLAIF = reinforcement learning from human / AI feedback.
- SFT = supervised fine-tuning.
- PPO = Proximal Policy Optimization.
- KL = Kullback–Leibler divergence.
- BT = Bradley–Terry.

**Problem.** Unsupervised LMs learn broad knowledge, but their behaviour is hard to control. We want to select the desired behaviours from that knowledge: understanding bad code but writing good code, and knowing a 50% misconception without asserting it.

**RLHF:**

$$\text{preferences} \xrightarrow{\text{MLE}} \text{reward model} \xrightarrow{\text{RL (PPO)}} \text{policy}$$

It is complex and unstable, and it samples from the LM inside the training loop.

**DPO:**

$$\text{preferences} \xrightarrow{\text{MLE / BCE}} \text{final LM}$$

- no explicit reward model and no RL;
- optimizes the **same** KL-constrained reward objective;
- trained with a **binary cross-entropy** loss (BCE = binary cross-entropy);
- a per-example importance weight prevents degeneration.

**Change of variables:** the preference loss is written over policies instead of rewards, and the policy *is* the implicit reward model.

**Claim:** DPO is ≥ PPO-RLHF on sentiment, summarization and dialogue, for models up to **6B** parameters.

![[RA2_DPO_fig1_rlhf_vs_dpo.png|650]]
*RLHF: reward model plus an RL loop. DPO: a single maximum-likelihood step to the final LM.*

---

## 2. Related Work

| Line of work | Key point |
|---|---|
| Zero-/few-shot LMs, instruction tuning | fine-tuning on instructions and human completions improves alignment and generalization |
| Preference fine-tuning (translation, summarization, story-telling, instruction-following) | relative judgments are easier to collect than demonstrations; fit a BT reward, then RL (REINFORCE / PPO) |
| RLAIF / synthetic preferences | an LLM labels data for safety or harmlessness, guided by a human text rubric |
| Contextual dueling bandits (CDB) | preferences instead of rewards; solution = **von Neumann winner** (≥ 50% win rate vs *any* policy); labels arrive **online** |
| Preference-based RL (PbRL) | binary preferences from an unknown scoring function; usually estimate a reward model first, then optimize |
| **DPO** | **single-stage**, offline, directly optimizes the policy |

---

## 3. RLHF Pipeline (Preliminaries)

### 3.1 SFT

Fine-tune a pre-trained LM on high-quality task data to get $\pi^{\text{SFT}}$.

### 3.2 Reward Modelling

- Sample $(y_1,y_2)\sim\pi^{\text{SFT}}(y\mid x)$; humans label $y_w \succ y_l \mid x$; the labels come from a latent $r^*$.
- BT model:

$$p^*(y_1\succ y_2\mid x) = \frac{e^{r^*(x,y_1)}}{e^{r^*(x,y_1)}+e^{r^*(x,y_2)}} = \sigma\big(r^*(x,y_1)-r^*(x,y_2)\big) \tag{1}$$

- Plackett–Luce generalizes BT to full rankings.
- Reward-model loss (BCE):

$$\mathcal{L}_R = -\mathbb{E}_{\mathcal{D}}\big[\log\sigma(r_\phi(x,y_w)-r_\phi(x,y_l))\big] \tag{2}$$

- $r_\phi$ = the SFT model plus a linear head on the final transformer layer, outputting a scalar.
- Normalized so that $\mathbb{E}[r_\phi(x,y)] = 0$ for every $x$ (lower variance).

### 3.3 RL Fine-Tuning

$$\max_{\pi_\theta}\mathbb{E}_{x\sim\mathcal{D},y\sim\pi_\theta}[r_\phi(x,y)] - \beta\,\mathbb{D}_{\text{KL}}[\pi_\theta\,\|\,\pi_{\text{ref}}] \tag{3}$$

- $\pi_{\text{ref}} = \pi^{\text{SFT}}$, and $\pi_\theta$ is initialized to it.
- **The KL term:**
  - keeps the policy where the reward model is accurate;
  - keeps generations diverse and prevents mode collapse.
- **Why RL:** the objective is not differentiable, because generation samples discrete tokens.
- **PPO reward:**
$$r = r_\phi - \beta(\log\pi_\theta - \log\pi_{\text{ref}})$$

---

## 4. DPO

### 4.1 Derivation Chain

**Step 1 — Eq. 4 (optimal policy):**

$$\pi_r(y\mid x) = \tfrac{1}{Z(x)}\pi_{\text{ref}}(y\mid x)\,e^{r(x,y)/\beta},\qquad Z(x)=\textstyle\sum_y\pi_{\text{ref}}(y\mid x)e^{r(x,y)/\beta}$$

$Z(x)$ sums over all completions, so it is intractable.

**Step 2 — take the log (Eq. 5):**

$$r(x,y) = \beta\log\frac{\pi_r(y\mid x)}{\pi_{\text{ref}}(y\mid x)} + \beta\log Z(x)$$

**Step 3 — substitute into BT (Eq. 6).** The term $\beta\log Z(x)$ cancels because both completions share the same $x$:

$$p^*(y_1\succ y_2\mid x) = \Big[1+\exp\Big(\beta\log\tfrac{\pi^*(y_2\mid x)}{\pi_{\text{ref}}(y_2\mid x)} - \beta\log\tfrac{\pi^*(y_1\mid x)}{\pi_{\text{ref}}(y_1\mid x)}\Big)\Big]^{-1}$$

**Step 4 — maximum likelihood for $\pi_\theta$ gives Eq. 7** (§4.2).

### 4.2 Loss and Gradient

**Loss:**

$$\boxed{\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{\mathcal{D}}\Big[\log\sigma\Big(\beta\log\tfrac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \beta\log\tfrac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\Big)\Big]} \tag{7}$$

**Implicit reward:**

$$\hat r_\theta(x,y) = \beta\log\dfrac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)}$$

This is a reparameterized BT fit, so it inherits BT's **consistency**.

**Gradient:**

$$\nabla_\theta\mathcal{L}_{\text{DPO}} = -\beta\,\mathbb{E}\big[\underbrace{\sigma(\hat r_\theta(x,y_l)-\hat r_\theta(x,y_w))}_{\text{large when wrong}}\,(\underbrace{\nabla\log\pi(y_w\mid x)}_{\uparrow\ y_w} - \underbrace{\nabla\log\pi(y_l\mid x)}_{\downarrow\ y_l})\big]$$

| Gradient part | Effect |
|---|---|
| $\nabla\log\pi(y_w\mid x)$ | raise the likelihood of the preferred response |
| $-\nabla\log\pi(y_l\mid x)$ | lower the likelihood of the dispreferred response |
| weight $\sigma(\hat r_l-\hat r_w)$ | larger the more wrongly the pair is ordered, scaled by $\beta$ |
| no weight (naive) | the model **degenerates** (App. Table 3) |

### 4.3 Pipeline

1. Sample $y_1,y_2\sim\pi_{\text{ref}}(\cdot\mid x)$, label them, and build the offline $\mathcal{D}$.
2. Minimize $\mathcal{L}_{\text{DPO}}$ for the given $\pi_{\text{ref}}$, $\mathcal{D}$ and $\beta$.

**Choosing $\pi_{\text{ref}}$:**
- $\pi_{\text{ref}} = \pi^{\text{SFT}}$ if it is available.
- Otherwise $\pi_{\text{ref}} = \arg\max_\pi\mathbb{E}_{x,y_w\sim\mathcal{D}}[\log\pi(y_w\mid x)]$. This is Preferred-FT, and it reduces distribution shift.

---

## 5. Theory

### 5.1 Equivalence Classes, Lemmas, Theorem 1

| Item | Statement |
|---|---|
| **Def. 1** | $r\sim r' \iff r(x,y)-r'(x,y) = f(x)$; this partitions rewards into classes |
| **Lemma 1** | same class ⇒ **same preference distribution** (under Plackett–Luce / BT); this is PL's under-specification, so maximum-likelihood estimates need identifiability constraints |
| **Lemma 2** | same class ⇒ **same optimal policy** under constrained RL, so recovering *any* member of the optimal class is enough |
| **Theorem 1** | every PL/BT-consistent reward class can be written as $r = \beta\log\frac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)}$, with no loss of generality |
| **Proof (Eq. 8)** | $f(r;\pi_{\text{ref}},\beta)(x,y) = r - \beta\log\sum_y\pi_{\text{ref}}e^{r/\beta} = \beta\log\frac{\pi_r}{\pi_{\text{ref}}}$; the subtracted term depends only on $x$, so $f(r)$ is in $r$'s class |
| **Eq. 9** | DPO picks the class member with $\sum_y\pi_{\text{ref}}(y\mid x)e^{r(x,y)/\beta} = 1$, i.e. a valid distribution with partition function = 1 |

**"Secretly a reward model":** $\beta\log(\pi_\theta/\pi_{\text{ref}})$ is itself a reward, and $\pi_\theta$ is its optimal policy.

### 5.2 Actor-Critic Instability

Minimizing $\mathbb{D}_{\text{KL}}[\pi_\theta\,\|\,\pi^*]$ (control as inference) gives

$$\max_{\pi_\theta}\mathbb{E}_{\pi_\theta}\Big[\underbrace{r_\phi - \beta\log\textstyle\sum_y\pi_{\text{ref}}e^{r_\phi/\beta}}_{f(r_\phi,\pi_{\text{ref}},\beta)} - \underbrace{\beta\log\tfrac{\pi_\theta}{\pi_{\text{ref}}}}_{\text{KL}}\Big] \tag{10}$$

- The normalizer is the **soft value function of $\pi_{\text{ref}}$**. It does not change the optimum, but without it the **policy gradient has high variance**.
- **PPO's fixes:**
  - a learned value function (critic), which is hard to optimize;
  - or a human-completion baseline, a one-sample Monte Carlo estimate of the normalizer.
- **DPO:** its reward already has $Z = 1$, so it needs **no baseline**.

> [!warning] Source cross-reference slip
> The paper says "$\pi^*$ … from Eq. 7"; this should be Eq. 4.

---

## 6. Experiments

### 6.1 Setup

| Task | Data | Preferences | SFT / $\pi_{\text{ref}}$ | Metric |
|---|---|---|---|---|
| Controlled sentiment | IMDb review prefixes | a **sentiment classifier** (ground-truth reward) | GPT-2-large fine-tuned on IMDb | **reward–KL frontier** |
| Summarization | Reddit **TL;DR** | humans (Stiennon et al.) | GPT-J SFT (TRLX) | GPT-4 win rate vs reference summaries |
| Single-turn dialogue | **Anthropic HH** (~170k) | humans | Pythia-2.8B + Preferred-FT | GPT-4 win rate vs the chosen response |

**Baselines:**
- GPT-J (zero-shot);
- Pythia-2.8B (2-shot);
- SFT;
- Preferred-FT;
- Unlikelihood ($\alpha\in[0,1]$);
- PPO;
- PPO-GT (oracle reward; TRL and tuned versions);
- Best of $N$ (impractical: $N$ samples per query at test time).

**Sweep (22 runs):**

| Algorithm             | Values swept      |
| --------------------- | ----------------- |
| PPO target KL         | 3, 6, 9, 12       |
| DPO $\beta$           | 0.05, 0.1, 1, 5   |
| Unlikelihood $\alpha$ | 0.05, 0.1, 0.5, 1 |
| Preferred-FT          | random seeds      |

Each policy is evaluated every 100 steps; KL is sequence-level (the sum of per-token KLs).

### 6.2 Results

![[RA2_DPO_fig2_frontier_tldr.png|650]]
*Left: DPO's reward–KL frontier dominates every method, PPO-GT included. Right: TL;DR win rate; DPO stays high across temperatures while PPO collapses at temperature 1.0.*

![[RA2_DPO_fig3_dialogue.png|650]]
*Left: on HH dialogue, DPO is the only efficient method above the chosen responses (≈ Best of 128). Right: DPO passes 50% within about 300 steps and stays stable.*

| Question | Result |
|---|---|
| RLHF objective (frontier) | DPO **strictly dominates PPO**, even **PPO-GT** |
| TL;DR summarization | DPO **≈ 61%** (temp 0) vs PPO **≈ 57%** (temp 0); DPO beats Best of $N$'s max; DPO is robust to temperature while PPO falls to GPT-J level; Preferred-FT ≈ SFT; $\beta$ not tuned |
| Human head-to-head | DPO (temp 0.25) preferred **58%** over PPO (temp 0) |
| HH dialogue | DPO is the only efficient method beating the chosen responses; ≈ or > Best of 128 (the PPO proxy, since the public PPO-HH model didn't beat base Pythia); converges fast |
| OOD (CNN/DailyMail) | DPO **0.36 / 0.31** vs PPO **0.26 / 0.23** (temp 0 / 0.25), without PPO's extra unlabeled prompts |

**Table 2 — GPT-4 vs human judgments (TL;DR; each column vs PPO at temp 0):**

| | DPO | SFT | PPO-1 |
|---|---|---|---|
| N | 272 | 122 | 199 |
| GPT-4 (S) win % | 47 | 27 | 13 |
| GPT-4 (C) win % | 54 | 32 | 12 |
| Human win % | 58 | 43 | 17 |
| GPT-4 (S)–H agree | 70 | 77 | 86 |
| GPT-4 (C)–H agree | 67 | 79 | 85 |
| H–H agree | 65 | – | 87 |

- **GPT-4 (S)** = simple prompt; it prefers longer, repetitive summaries.
- **GPT-4 (C)** = concise prompt; it is closer to humans, so it is used for the main results.
- GPT-4–human agreement ≈ human–human agreement.

---

## 7. Limitations

1. Out-of-distribution generalization vs an explicit reward model; can self-labeling use unlabeled prompts?
2. Reward over-optimization in DPO; is the late dip in Fig. 3 (right) an instance?
3. Scale beyond 6B.
4. GPT-4 win rates depend on the prompt.
5. Other modalities.

---

## Formula Summary Sheet

| Concept | Formula |
|---|---|
| BT (Eq. 1) | $p^*(y_1\succ y_2\mid x) = \sigma(r^*(x,y_1)-r^*(x,y_2))$ |
| Reward-model loss (Eq. 2) | $-\mathbb{E}\log\sigma(r_\phi(x,y_w)-r_\phi(x,y_l))$ |
| RLHF objective (Eq. 3) | $\max\mathbb{E}[r_\phi] - \beta\mathbb{D}_{\text{KL}}[\pi_\theta\|\pi_{\text{ref}}]$ |
| PPO shaped reward | $r_\phi - \beta(\log\pi_\theta-\log\pi_{\text{ref}})$ |
| Optimal policy (Eq. 4) | $\pi_r = \frac{1}{Z(x)}\pi_{\text{ref}}e^{r/\beta}$, $\ Z(x)=\sum_y\pi_{\text{ref}}e^{r/\beta}$ |
| Reward from policy (Eq. 5) | $r = \beta\log\frac{\pi_r}{\pi_{\text{ref}}} + \beta\log Z(x)$ |
| Preference via policy (Eq. 6) | $p^* = \sigma\big(\beta\log\frac{\pi^*(y_1)}{\pi_{\text{ref}}(y_1)} - \beta\log\frac{\pi^*(y_2)}{\pi_{\text{ref}}(y_2)}\big)$ |
| DPO loss (Eq. 7) | $-\mathbb{E}\log\sigma(\hat r_\theta(x,y_w)-\hat r_\theta(x,y_l))$ |
| Implicit reward | $\hat r_\theta = \beta\log\frac{\pi_\theta}{\pi_{\text{ref}}}$ |
| DPO gradient | $-\beta\mathbb{E}[\sigma(\hat r_l-\hat r_w)(\nabla\log\pi(y_w)-\nabla\log\pi(y_l))]$ |
| $\pi_{\text{ref}}$ without SFT | $\arg\max_\pi\mathbb{E}[\log\pi(y_w\mid x)]$ |
| Equivalence (Def. 1) | $r-r' = f(x)$ |
| Projection (Eq. 8) | $f(r) = r - \beta\log\sum_y\pi_{\text{ref}}e^{r/\beta}$ |
| DPO's selected reward (Eq. 9) | $\sum_y\pi_{\text{ref}}e^{r/\beta}=1$ |
| Actor-critic objective (Eq. 10) | $\mathbb{E}_{\pi_\theta}[r_\phi - \beta\log\sum_y\pi_{\text{ref}}e^{r_\phi/\beta} - \beta\log\frac{\pi_\theta}{\pi_{\text{ref}}}]$ |
| Useful identities | $\sigma(-z)=1-\sigma(z)$; $\ -\log\sigma(u)=\log(1+e^{-u})$; $\ \frac{d}{du}\log\sigma(u)=\sigma(-u)$ |

---

## Cheat Sheet & Recall Aids

**Derivation mnemonic: "O-I-C-L"** — **O**ptimum $\pi\propto\pi_{\text{ref}}e^{r/\beta}$ → **I**nvert for $r$ → **C**ancel $Z$ in BT → **L**ogistic loss on $\pi_\theta$.

**Pipeline mnemonic: "S-R-R → S-D"** — RLHF is **S**FT, **R**eward model, **R**L; DPO is **S**FT, then **D**PO.

**Gotchas:**
- DPO has an **implicit** reward, not *no* reward. It is free of a reward *model* and of RL.
- $Z$ cancels only because $y_w, y_l$ share the **same prompt**.
- $\beta$ is the KL strength, not a learning rate: large $\beta$ keeps the policy close to $\pi_{\text{ref}}$.
- The gradient weight $\to 1$ when wrong and $\to 0$ when already right. Without it the model degenerates.
- DPO is cheap because it does no LM sampling during training, and has no reward model and no critic.
- The source of PPO instability is the normalizer (soft value of $\pi_{\text{ref}}$). DPO has $Z=1$, so it needs no baseline.
- A reward–KL frontier exists only for **sentiment**, because only there is a ground-truth reward (a classifier) available.
- 61/57 is GPT-4 vs the references; **58%** is the human head-to-head DPO vs PPO.
- OOD: **0.36 vs 0.26**.

**If you remember nothing else:** RLHF's optimum is $\pi_{\text{ref}}e^{r/\beta}/Z$. Read backwards, any policy defines a reward $\beta\log(\pi/\pi_{\text{ref}})$, so fit BT on that reward with a logistic loss and the policy is trained directly.

**Recall checklist:**
- [ ] Eqs. 1–7 from memory, including the $Z$ cancellation.
- [ ] The three parts of the gradient.
- [ ] Def. 1, Lemmas 1–2, Theorem 1, Eq. 9.
- [ ] Why actor-critic is unstable (Eq. 10).
- [ ] The 3 tasks, their metrics, and the numbers 61/57, 58, 0.36/0.26.
- [ ] The 5 limitations.

---

## Practice Questions (new, not in RAQ4)

*RAQ4 (Foundations of GenAI) already asked about:*
- DPO's conceptual shift;
- the reward-model phase;
- the gradient when the model is wrong;
- BT;
- the purpose of the KL term;
- the closed-form $\pi_r$;
- the role of $\beta$;
- why DPO is lightweight;
- a BT numerical;
- an implicit-reward-difference numerical.

*None of the questions below repeat those.*

### MCQs (correct answers only)

1. **Why does the partition function $Z(x)$ disappear when Eq. 5 is substituted into the Bradley–Terry model?**
   → Both completions share the same prompt $x$, so $\beta\log Z(x)$ appears in both rewards, and BT depends only on their difference.
2. **Why can't Eq. 4 be used directly to obtain the optimal policy in practice?**
   → $Z(x)$ sums over every possible completion $y$, which is intractable even with a good reward estimate.
3. **Two reward functions are "equivalent" (Definition 1) when…**
   → their difference is a function of the prompt only, $r(x,y)-r'(x,y)=f(x)$.
4. **Lemma 1 states that two rewards in the same equivalence class…**
   → induce the same preference distribution under Plackett–Luce / Bradley–Terry (the under-specification problem).
5. **Lemma 2 states that two rewards in the same equivalence class…**
   → induce the same optimal policy under the KL-constrained RL problem.
6. **Which member of each reward equivalence class does the DPO reparameterization select (Eq. 9)?**
   → The one whose partition function equals 1, i.e. $\sum_y\pi_{\text{ref}}(y\mid x)e^{r(x,y)/\beta}=1$.
7. **What does Theorem 1 guarantee?**
   → Every reward class consistent with Plackett–Luce/BT can be represented as $\beta\log\frac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)}$, so the reparameterization loses no generality.
8. **According to §5.2, what makes actor-critic RLHF (PPO) unstable?**
   → The normalization term (the soft value function of $\pi_{\text{ref}}$); without it the policy gradient has high variance.
9. **Why is the RL fine-tuning objective (Eq. 3) typically optimized with RL rather than plain gradient descent?**
   → Language generation is discrete (sampled tokens), so the objective is not differentiable.
10. **How is the reward model $r_\phi$ usually built in RLHF?**
    → It is initialized from the SFT model, with a linear layer on the final transformer layer that outputs a scalar.
11. **Why do prior works normalize the reward so that $\mathbb{E}_{x,y}[r_\phi(x,y)]=0$?**
    → To reduce the variance of the reward function.
12. **When no SFT model is available, how does DPO initialize $\pi_{\text{ref}}$?**
    → By maximizing the likelihood of the preferred completions, $\arg\max_\pi\mathbb{E}[\log\pi(y_w\mid x)]$ (Preferred-FT).
13. **What happens if DPO is trained without the per-example weighting coefficient?**
    → The language model can degenerate.
14. **In the controlled sentiment task, how were the preference pairs generated?**
    → With a pre-trained sentiment classifier: $y_w$ has the higher $p(\text{positive}\mid x,y)$.
15. **Why was the reward–KL frontier evaluated only in the sentiment task?**
    → Only there is the ground-truth reward function (the sentiment classifier) available.
16. **What was the reward–KL frontier result?**
    → DPO strictly dominated PPO, even PPO-GT, which uses ground-truth rewards.
17. **What was DPO's TL;DR summarization win rate at temperature 0 compared with PPO's best?**
    → About 61% for DPO vs about 57% for PPO.
18. **How did the methods respond to higher sampling temperature on TL;DR?**
    → DPO stayed robust, while PPO degraded toward base GPT-J performance.
19. **On out-of-distribution CNN/DailyMail articles (temperature 0), what were the win rates?**
    → DPO 0.36 vs PPO 0.26.
20. **Why was the GPT-4 (C) "concise" prompt used for the main results?**
    → GPT-4 (S) preferred longer, repetitive summaries, and (C) win rates were more representative of humans.
21. **What did the human study (Table 2) conclude about GPT-4 as an evaluator?**
    → GPT-4 agrees with humans about as often as humans agree with each other.
22. **In the dialogue experiment, which baseline served as a rough proxy for PPO-level performance?**
    → Best of 128 Preferred-FT completions.
23. **Why is Best of $N$ considered impractical despite performing well?**
    → It samples $N$ completions for every query at test time.
24. **How is the sequence-level KL in the frontier experiment computed?**
    → As the sum of the per-timestep (per-token) KL divergences.
25. **What is the largest model size evaluated in the paper?**
    → 6B parameters.
26. **What objective does the Unlikelihood baseline optimize?**
    → Maximize $p(y_w)$ and minimize $p(y_l)$, with an optional coefficient $\alpha\in[0,1]$ on the unlikelihood term.
27. **In contextual dueling bandits, preference labels are given ___, while in learning from human preferences they come from ___.**
    → Online; a fixed offline batch.
28. **If the implicit reward model already ranks $y_w$ far above $y_l$, the DPO gradient for that pair is…**
    → Close to zero, because the weight $\sigma(\hat r_l-\hat r_w)\to0$.
29. **The quantity $\beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)}$ in DPO is called…**
    → The implicit reward defined by the language model and the reference model.
30. **Which step of the RLHF pipeline does DPO keep unchanged?**
    → Supervised fine-tuning (SFT); $\pi^{\text{SFT}}$ becomes $\pi_{\text{ref}}$.

### Short Questions

**Q1. Starting from Eq. 4, derive the expression for the reward (Eq. 5). Then show why $Z(x)$ does not appear in the DPO loss.** (3 marks)

*Answer.*
- Taking logs of Eq. 4 gives $\log\pi_r = \log\pi_{\text{ref}} + r/\beta - \log Z(x)$.
- Rearranging gives $r(x,y) = \beta\log\frac{\pi_r(y\mid x)}{\pi_{\text{ref}}(y\mid x)} + \beta\log Z(x)$.
- BT uses $\sigma(r(x,y_1)-r(x,y_2))$. Both rewards carry the same $+\beta\log Z(x)$ because they share $x$, so it cancels in the difference.
- What remains is $\sigma\big(\beta\log\frac{\pi(y_1)}{\pi_{\text{ref}}(y_1)} - \beta\log\frac{\pi(y_2)}{\pi_{\text{ref}}(y_2)}\big)$, which contains no $Z$.

**Q2. Let $\beta = 0.2$, $\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} = 1.0$ and $\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)} = -1.5$. Compute (a) the per-example DPO loss and (b) the gradient weight.** (3 marks)

*Answer.*
- The implicit rewards are $\hat r_w = 0.2$ and $\hat r_l = -0.3$, so the margin is $u = 0.5$.
- (a) $\mathcal{L} = -\log\sigma(0.5) = \log(1+e^{-0.5}) = \log 1.6065 \approx \mathbf{0.474}$.
- (b) The weight is $\sigma(\hat r_l - \hat r_w) = \sigma(-0.5) = 1/(1+e^{0.5}) = 1/2.6487 \approx \mathbf{0.378}$.

**Q3. A prompt has two possible completions. $\pi_{\text{ref}} = (0.5, 0.5)$, rewards $r = (0, 2)$ and $\beta = 1$. Compute $Z(x)$ and the optimal policy $\pi_r$ from Eq. 4.** (3 marks)

*Answer.*
- The weights are $0.5e^{0} = 0.5$ and $0.5e^{2} = 3.6945$, so $Z = 4.1945$.
- $\pi_r = (0.5/4.1945,\ 3.6945/4.1945) \approx \mathbf{(0.119,\ 0.881)}$.
- Check: with a uniform reference this equals $(\sigma(-2), \sigma(2))$.

**Q4. Using Eq. 4, explain what happens to the optimal policy as $\beta\to0$ and as $\beta\to\infty$.** (2 marks)

*Answer.*
- As $\beta\to0$, $e^{r/\beta}$ makes the highest-reward completion dominate exponentially. The policy collapses onto the argmax-reward answer: maximal reward, unbounded KL, mode collapse.
- As $\beta\to\infty$, $e^{r/\beta}\to1$ for every $y$, so $\pi_r\to\pi_{\text{ref}}$: the reward is ignored and the KL is zero.

**Q5. Show that adding $f(x)$ to a reward does not change the optimal policy (Lemma 2).** (2 marks)

*Answer.*
- With $r' = r + f(x)$, Eq. 4 gives $\pi_{r'} \propto \pi_{\text{ref}}e^{r/\beta}e^{f(x)/\beta}$.
- The factor $e^{f(x)/\beta}$ is the same for every $y$, so it also multiplies $Z(x)$ and cancels on normalization.
- Hence $\pi_{r'} = \pi_r$.

**Q6. Why does PPO need a learned value function (critic) or a baseline, while DPO does not? Answer with reference to Eq. 10.** (3 marks)

*Answer.*
- Rewriting the RLHF problem as minimizing $\mathbb{D}_{\text{KL}}[\pi_\theta\|\pi^*]$ gives an objective containing the normalizer $\beta\log\sum_y\pi_{\text{ref}}e^{r_\phi/\beta}$, the soft value function of $\pi_{\text{ref}}$.
- It does not change the optimum, because it depends only on $x$. But leaving it out makes policy-gradient estimates high-variance.
- PPO therefore estimates it, either with a critic (hard to optimize) or a human-completion baseline (a one-sample Monte Carlo estimate).
- DPO's reparameterized reward is chosen so that the partition function is 1 (Eq. 9), so the normalizer is zero and no baseline is needed.

**Q7. For one sample, $r_\phi(x,y) = 1.0$, $\beta = 0.2$, $\log\pi_\theta(y\mid x) = -30$ and $\log\pi_{\text{ref}}(y\mid x) = -28$. Compute the KL-shaped reward PPO maximizes and interpret its sign.** (2 marks)

*Answer.*
- $r = 1.0 - 0.2(-30 + 28) = 1.0 + 0.4 = \mathbf{1.4}$.
- The policy made $y$ *less* likely than the reference did, so the per-sample KL estimate is negative and adds a bonus.
- Only the *expected* KL is guaranteed to be non-negative.

**Q8. The DPO policy beat PPO on CNN/DailyMail even though it was trained on Reddit TL;DR. Why is this result notable?** (2 marks)

*Answer.*
- It is a distribution shift from forum posts to news articles, yet DPO won 0.36 vs 0.26 (temperature 0).
- PPO uses extra *unlabeled* TL;DR prompts during RL sampling and DPO does not.
- So DPO generalizes at least as well as PPO despite seeing less data. This is initial evidence only; the paper lists OOD generalization as future work.

**Q9. Why did the authors compare algorithms on a reward–KL frontier instead of reward alone?** (2 marks)

*Answer.*
- The RLHF objective trades reward against KL from $\pi_{\text{ref}}$.
- Slightly higher reward at much higher KL is not necessarily better: the policy may be exploiting the reward or losing diversity.
- A frontier shows the best reward reached at each KL level. DPO's frontier was highest at every KL.

**Q10. Name the three conditions under which the DPO gradient for a pair is large, small, and exactly 0.5 × (−β) times the log-likelihood difference.** (2 marks)

*Answer.*
- **Large** (weight → 1) when the implicit reward ranks $y_l$ well above $y_w$, i.e. the model is badly wrong.
- **Small** (weight → 0) when $y_w$ is already ranked well above $y_l$.
- **Weight exactly 0.5** when the implicit rewards are equal, e.g. at initialization when $\pi_\theta = \pi_{\text{ref}}$ and both log-ratios are 0.
