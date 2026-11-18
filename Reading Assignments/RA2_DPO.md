---
type: study-aid
course: '[[_CS5326 Advanced GenAI|CS5326 Advanced GenAI]]'
source:
- '[[Rafailov2023_DPO.pdf]]'
tags:
- course/cs5326
---

%% nav:start %%
[[_CS5326 Advanced GenAI|CS5326 Advanced GenAI]] · Cheatsheet: [[RA2_DPO_Cheatsheet]]
Related: [[ATML_Lecture_8 - LLM Alignment|ATML Lecture 8 — LLM Alignment (RLHF, PPO, GRPO, DPO)]]
%% nav:end %%

# Direct Preference Optimization: Your Language Model is Secretly a Reward Model (Rafailov et al., NeurIPS 2023)

*Scope: Abstract and Sections 1–7 (pp. 1–10) only. Acknowledgements, References and all Appendices (A–D, including the full derivations in A.1–A.6) are excluded. Where the main text says "see Appendix", these notes give only what the main text states plus short marked expansions.*

## Topic Inventory

- 1. Abstract and Introduction
  - 1.1 Why Language Models Need Steering
  - 1.2 The RLHF Pipeline and Its Cost
  - 1.3 The DPO Idea and Contribution
- 2. Related Work
  - 2.1 Instruction Tuning and Learning from Human Preferences
  - 2.2 Preferences Outside Language: Dueling Bandits and Preference-Based RL
- 3. Preliminaries: The RLHF Pipeline
  - 3.1 Supervised Fine-Tuning (SFT)
  - 3.2 Reward Modelling Phase
  - 3.3 RL Fine-Tuning Phase
- 4. Direct Preference Optimization
  - 4.1 The Closed-Form Optimal Policy
  - 4.2 Rewriting the Reward in Terms of the Policy
  - 4.3 Preferences in Terms of the Policy Alone
  - 4.4 The DPO Loss
  - 4.5 What Does the DPO Update Do? (Gradient)
  - 4.6 DPO Outline
- 5. Theoretical Analysis of DPO
  - 5.1 Your Language Model Is Secretly a Reward Model
  - 5.2 Instability of Actor-Critic Algorithms
- 6. Experiments
  - 6.1 Tasks, Evaluation and Baselines
  - 6.2 How Well Can DPO Optimize the RLHF Objective?
  - 6.3 Can DPO Scale to Real Preference Datasets?
  - 6.4 Generalization to a New Input Distribution
  - 6.5 Validating GPT-4 Judgments with Human Judgments
- 7. Discussion, Limitations and Future Work
- Numerical Problems
- Formula Summary Sheet
- Cheat Sheet & Recall Aids

---

## 1. Abstract and Introduction

### 1.1 Why Language Models Need Steering

Large unsupervised **language models (LMs)**, i.e. neural networks trained to predict the next token on huge text corpora, pick up broad world knowledge and some reasoning ability. Because that training is completely unsupervised, though, it is hard to control precisely *how* the model behaves. The data was written by people with many different goals, priorities and skill levels, and not all of it is worth imitating. The paper's two examples:

- A coding assistant should *understand* common programming mistakes (so it can fix them), but when it writes code we want it biased toward the rare high-quality code in its training data, not the common buggy code.
- A model should be *aware* of a misconception that 50% of people believe, but it certainly should not assert that misconception in 50% of its answers.

So the task is to select the model's **desired responses and behaviour** from its very wide knowledge and abilities. This is what makes an AI system safe, performant and controllable.

### 1.2 The RLHF Pipeline and Its Cost

Desired behaviour is usually instilled after pre-training, using curated **human preferences**: humans compare model outputs and say which one is better. The simplest approach would be **supervised fine-tuning (SFT)** on human-written high-quality demonstrations, but the most successful methods so far are **reinforcement learning from human (or AI) feedback (RLHF / RLAIF)**. In RLAIF the preference labels come from an AI model instead of humans.

RLHF works in two moves (Figure 1, left):

1. Fit a **reward model**, a network that outputs a scalar "how good is this response" score, to a dataset of human preferences.
2. Use **reinforcement learning (RL)** to optimize the LM policy so it produces high-reward responses without drifting too far from the original model.

This produces impressive conversational and coding models. However, the pipeline is far more complex than supervised learning:

- it trains several LMs (policy, reward model, and in PPO also a value model);
- it samples from the LM policy *inside* the training loop, which is computationally expensive;
- RL training is often unstable.

![[RA2_DPO_fig1_rlhf_vs_dpo.png|700]]
*Figure 1 — Left: RLHF first fits a reward model to preference pairs by maximum likelihood, then runs an RL loop in which the LM policy samples completions and the reward model labels them. Right: DPO goes straight from the preference data to the final LM with a single maximum-likelihood (classification) objective. DPO fits an **implicit** reward model whose optimal policy can be extracted in closed form.*

### 1.3 The DPO Idea and Contribution

The paper shows how to optimize an LM directly for human preferences **without an explicit reward model and without RL**. The algorithm is **Direct Preference Optimization (DPO)**. Its key properties:

- **Same objective as RLHF:** <mark style="background: #FFF3A3A6;">DPO implicitly optimizes the same goal as existing RLHF algorithms, namely reward maximization with a KL-divergence constraint</mark>. **KL divergence** (Kullback–Leibler divergence) measures how different one probability distribution is from another.
- **Simple:** it is trained with a simple **binary cross-entropy** (logistic classification) loss.
- **What the update does:** <mark style="background: #FFF3A3A6;">it increases the relative log probability of preferred over dispreferred responses</mark>. It also carries a dynamic, per-example **importance weight**, which prevents the model degeneration the authors observed with a naive probability-ratio objective.
- **Theoretical basis:** like RLHF, DPO relies on a **preference model** such as the Bradley–Terry model (§3.2), which measures how well a reward function agrees with the observed preferences.

The difference lies in how the preference model is used. RLHF uses it to define a loss for a reward model and then trains a policy against that reward model. DPO instead uses a **change of variables** to write the preference loss as a function of the *policy* directly. Training the policy with this loss is equivalent to fitting an *implicit* reward function to the preference data, and the policy is the optimal policy for that reward.

**Main claim:** DPO is at least as effective as existing methods, including PPO-based RLHF, on sentiment modulation, summarization and dialogue, using LMs of up to **6B parameters**. **PPO** (Proximal Policy Optimization) is the standard RL algorithm used in RLHF.

---

## 2. Related Work

The introduction placed DPO as an alternative to RLHF. The related-work section shows where RLHF itself came from and which non-language preference-learning ideas DPO draws on.

### 2.1 Instruction Tuning and Learning from Human Preferences

**Zero-shot and few-shot ability.** Larger self-supervised LMs can complete some tasks **zero-shot** (from the instruction alone) or with **few-shot prompts** (a handful of examples in the prompt).

**Instruction tuning.** Their downstream performance and alignment with user intent improve a lot when they are fine-tuned on datasets of instructions paired with human-written completions. This lets LLMs generalize to instructions outside the tuning set and generally makes them more usable.

**Learning from preferences.** Despite instruction tuning's success, *relative* human judgments of response quality ("A is better than B") are often easier to collect than expert demonstrations. Later works therefore fine-tuned LLMs on preference data. This improved:

- translation;
- summarization;
- story-telling;
- instruction-following.

These methods follow the same two steps:

1. Optimize a neural reward function to agree with the preference dataset under a preference model such as **Bradley–Terry**.
2. Fine-tune the LM to maximize that reward using RL, most often **REINFORCE** (the basic policy-gradient estimator) or **PPO** (or variants).

**AI feedback.** A closely related line uses LLMs that were themselves fine-tuned with human feedback to *generate* additional synthetic preference data for targeted attributes such as safety or harmlessness. Here human supervision is weak: just a text rubric for the LLM's annotations.

These methods sit where two bodies of work meet:

- training LMs with RL for various objectives;
- general methods for learning from human preferences.

The appeal of relative preferences is clear, but fine-tuning large LMs with RL remains a major practical challenge. The paper's contribution is a theoretically justified way to optimize relative preferences **without RL**.

### 2.2 Preferences Outside Language: Dueling Bandits and Preference-Based RL

Learning policies from preferences has also been studied in bandit and RL settings outside language.

**Contextual dueling bandits (CDB).** A **contextual bandit** is a one-step decision problem: observe a context, choose an action, get feedback. In the *dueling* version the feedback is a preference or ranking between two actions rather than a numerical reward.

- With no absolute rewards, CDB theory replaces "optimal policy" with the **von Neumann winner**: a policy whose expected win rate against *any* other policy is at least 50%.
- In CDBs, preference labels arrive **online** (interactively). In learning from human preferences, we instead learn from a **fixed offline batch** of preference-annotated action pairs.

**Preference-based RL (PbRL).** PbRL learns from binary preferences generated by an *unknown* "scoring" function rather than from rewards.

- Several PbRL algorithms exist, including some that can reuse off-policy preference data.
- They generally first explicitly estimate the latent scoring function (i.e. a reward model) and then optimize it.

DPO differs from both: it is a **single-stage** policy-learning approach that directly optimizes a policy to satisfy preferences.

---

## 3. Preliminaries: The RLHF Pipeline

Before deriving DPO, the paper sets out the standard RLHF pipeline (Ziegler et al. and later work) that DPO is meant to replace. It has three phases:

1. supervised fine-tuning (SFT);
2. preference sampling and reward learning;
3. RL optimization.

### 3.1 Supervised Fine-Tuning (SFT)

RLHF usually starts by fine-tuning a pre-trained LM with supervised learning on high-quality data for the downstream task(s) of interest, such as dialogue or summarization. The result is the model $\pi^{\text{SFT}}$. Here $\pi(y\mid x)$ denotes a **policy**: the LM's probability of producing completion $y$ given prompt $x$.

### 3.2 Reward Modelling Phase

**Collecting preference pairs.** The SFT model is prompted with prompts $x$ to produce pairs of answers $(y_1, y_2) \sim \pi^{\text{SFT}}(y \mid x)$. Human labelers see each pair and state which they prefer, written $y_w \succ y_l \mid x$:

- $y_w$ is the preferred ("winning") completion;
- $y_l$ is the dispreferred ("losing") completion.

The preferences are assumed to come from a **latent reward model** $r^*(y, x)$ that we cannot access.

**The Bradley–Terry model.** Several approaches exist for modelling preferences. The **Bradley–Terry (BT)** model is the popular one. The more general **Plackett–Luce** ranking models extend it from pairs to full rankings of several answers and are also compatible with the framework. BT says the human preference distribution $p^*$ is

$$
p^*(y_1 \succ y_2 \mid x) = \frac{\exp\left(r^*(x, y_1)\right)}{\exp\left(r^*(x, y_1)\right) + \exp\left(r^*(x, y_2)\right)} \tag{1}
$$

*(expanding on this)* Dividing the numerator and denominator by $\exp(r^*(x,y_1))$ gives

$$
p^*(y_1 \succ y_2 \mid x) = \frac{1}{1 + \exp\left(-(r^*(x,y_1) - r^*(x,y_2))\right)} = \sigma\left(r^*(x,y_1) - r^*(x,y_2)\right),
$$

where $\sigma(z) = 1/(1+e^{-z})$ is the **logistic (sigmoid) function**. So BT depends **only on the difference of rewards**. Adding the same constant to both rewards leaves the preference probability unchanged. This fact drives the whole derivation in §4.

**Fitting the reward model.** Assume a static dataset of comparisons $\mathcal{D} = \{x^{(i)}, y_w^{(i)}, y_l^{(i)}\}_{i=1}^{N}$ sampled from $p^*$. We parameterize a reward model $r_\phi(x, y)$ with parameters $\phi$ and estimate them by **maximum likelihood**. Framed as binary classification ("is $y_w$ the winner?"), the negative log-likelihood loss is

$$
\mathcal{L}_R(r_\phi, \mathcal{D}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\left[\log \sigma\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right)\right] \tag{2}
$$

The loss is small when $r_\phi(x,y_w) \gg r_\phi(x,y_l)$, because then $\sigma(\cdot) \to 1$ and $\log\sigma \to 0$.

Two practical details:

- **Initialization:** for LMs, $r_\phi(x, y)$ is often initialized from $\pi^{\text{SFT}}(y \mid x)$. A linear layer on top of the final transformer layer outputs a single scalar reward.
- **Normalization:** to lower the reward's variance, prior works normalize it so that $\mathbb{E}_{x, y \sim \mathcal{D}}[r_\phi(x, y)] = 0$ for all $x$.

### 3.3 RL Fine-Tuning Phase

In the RL phase, the learned reward gives feedback to the LM. Following prior work, the optimization is
$$
\max_{\pi_\theta}\; \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(y \mid x)}\left[r_\phi(x, y)\right] - \beta\, \mathbb{D}_{\text{KL}}\left[\pi_\theta(y \mid x) \,\|\, \pi_{\text{ref}}(y \mid x)\right] \tag{3}
$$

The symbols:

- $\pi_\theta$ is the LM policy being trained (parameters $\theta$). In practice it is initialized to $\pi^{\text{SFT}}$.
- $\pi_{\text{ref}}$ is the **reference policy**, namely the initial SFT model $\pi^{\text{SFT}}$.
- $\beta > 0$ controls how far $\pi_\theta$ may drift from $\pi_{\text{ref}}$:
  - large $\beta$ means a strong pull back toward $\pi_{\text{ref}}$;
  - small $\beta$ means the policy chases reward more freely.

**Why the KL constraint matters.** It does two jobs:

1. It keeps the model close to the distribution on which the reward model is accurate. The reward model was trained on outputs from $\pi^{\text{SFT}}$, so its scores are unreliable far away from them. A policy that drifts there can exploit reward-model errors ("reward hacking").
2. It keeps generations diverse and prevents **mode collapse**, where the model always outputs a single high-reward answer.

**Why RL is needed.** Language generation is **discrete**: the model samples tokens. You cannot backpropagate through the sampling of $y$, so this objective is not differentiable with respect to $\theta$ and is typically optimized with RL.

**The standard approach.** Construct the per-sample reward

$$
r(x, y) = r_\phi(x, y) - \beta\left(\log \pi_\theta(y \mid x) - \log \pi_{\text{ref}}(y \mid x)\right)
$$

and maximize it with PPO. The subtracted term is a sample estimate of the KL penalty: its expectation under $y\sim\pi_\theta$ is exactly $\beta\,\mathbb{D}_{\text{KL}}$.

---

## 4. Direct Preference Optimization

RL on large problems such as LM fine-tuning is hard, which motivates a simpler way to optimize a policy from preferences directly.

**DPO's key insight** is an **analytical mapping from reward functions to optimal policies**. With this mapping, a loss over *reward functions* can be turned into a loss over *policies*:

- Unlike prior RLHF methods, which learn a reward and then optimize it with RL, DPO picks a reward-model parameterization whose optimal policy can be extracted in **closed form**, with no RL training loop.
- The change of variables avoids fitting an explicit, standalone reward model, while still optimizing under the same preference models (e.g. Bradley–Terry).
- In essence, **the policy network represents both the LM and the (implicit) reward**.

### 4.1 The Closed-Form Optimal Policy

Start from the same RL objective as prior work (Eq. 3), but with a general reward function $r$. Following prior work, it is straightforward to show that the optimal solution to this KL-constrained reward maximization is

$$
\boxed{\pi_r(y \mid x) = \frac{1}{Z(x)}\, \pi_{\text{ref}}(y \mid x) \exp\left(\frac{1}{\beta} r(x, y)\right)} \tag{4}
$$

Here

$$
Z(x) = \sum_y \pi_{\text{ref}}(y \mid x) \exp\left(\frac{1}{\beta} r(x, y)\right)
$$

is the **partition function**: the normalizing constant that makes $\pi_r(\cdot\mid x)$ sum to 1 over all completions $y$. The full derivation is in Appendix A.1, which is out of scope here.

*(intuition, expanding on this)* The optimal policy takes the reference model and **re-weights** each completion by $e^{r/\beta}$:

- completions with higher reward get exponentially more probability;
- $\beta$ acts as a temperature:
  - $\beta \to 0$ concentrates all mass on the highest-reward answer;
  - $\beta \to \infty$ leaves $\pi_{\text{ref}}$ unchanged.

**The catch.** Even with the maximum-likelihood estimate $r_\phi$ of the true reward $r^*$, estimating $Z(x)$ is still expensive. It sums over *every possible completion* $y$, an astronomically large set of token sequences. So Eq. 4 cannot be used directly in practice.

### 4.2 Rewriting the Reward in Terms of the Policy

The trick is to rearrange Eq. 4 to express the *reward* in terms of the optimal policy $\pi_r$, the reference $\pi_{\text{ref}}$ and the unknown $Z(\cdot)$.

Taking the logarithm of both sides of Eq. 4 gives

$$
\log \pi_r(y\mid x) = \log \pi_{\text{ref}}(y\mid x) + \tfrac{1}{\beta} r(x,y) - \log Z(x).
$$

Isolating $r$ and multiplying through by $\beta$:

$$
\boxed{r(x, y) = \beta \log \frac{\pi_r(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)} \tag{5}
$$

In words, any reward equals $\beta$ times the log-ratio of its optimal policy to the reference, plus a term that depends **only on the prompt $x$**.

### 4.3 Preferences in Terms of the Policy Alone

Apply Eq. 5 to the ground-truth reward $r^*$ and its optimal policy $\pi^*$. Since BT depends only on the reward difference (§3.2),

$$
p^*(y_1 \succ y_2 \mid x) = \sigma\left(r^*(x, y_1) - r^*(x, y_2)\right).
$$

Substitute Eq. 5 for both rewards:

$$
r^*(x,y_1) - r^*(x,y_2) = \beta \log \frac{\pi^*(y_1\mid x)}{\pi_{\text{ref}}(y_1\mid x)} + \beta\log Z(x) - \beta \log \frac{\pi^*(y_2\mid x)}{\pi_{\text{ref}}(y_2\mid x)} - \beta\log Z(x).
$$

**The $\beta\log Z(x)$ terms cancel**, because both completions share the same prompt $x$ and hence the same $Z(x)$. The intractable partition function disappears, and the human preference probability can be written using only $\pi^*$ and $\pi_{\text{ref}}$. Under BT, the optimal RLHF policy $\pi^*$ satisfies

$$
p^*(y_1 \succ y_2 \mid x) = \frac{1}{1 + \exp\left(\beta \log \frac{\pi^*(y_2 \mid x)}{\pi_{\text{ref}}(y_2 \mid x)} - \beta \log \frac{\pi^*(y_1 \mid x)}{\pi_{\text{ref}}(y_1 \mid x)}\right)} \tag{6}
$$

This is just $\sigma(\text{difference})$ written out, with the sign flipped inside the exponent. The derivation is in Appendix A.2. Similar expressions hold under the more general Plackett–Luce models (Appendix A.3).

### 4.4 The DPO Loss

We now have the probability of the preference data in terms of the optimal *policy* rather than the reward model. That lets us write a maximum-likelihood objective for a parameterized policy $\pi_\theta$. <mark style="background: #FFF3A3A6;">It is the direct analogue of the reward-model loss </mark>in Eq. 2, with the reward difference replaced by the log-ratio difference:

$$
\boxed{\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)\right]} \tag{7}
$$

<mark style="background: #FFF3A3A6;">This way we fit an implicit reward through an alternative parameterization whose optimal policy is simply</mark> $\pi_\theta$.

<mark style="background: #ABF7F7A6;">Because the procedure is equivalent to fitting a reparameterized Bradley–Terry model, it inherits that model's theoretical properties</mark>, such as **consistency** under suitable assumptions on the preference-data distribution. <mark style="background: #ABF7F7A6;">A consistent estimator converges to the true parameters as data grows.</mark>

*(expanding on this)* Everything in Eq. 7 is computable with ordinary forward passes:

- $\log\pi_\theta(y\mid x)$ is the sum of the policy's token log-probabilities over $y$;
- $\pi_{\text{ref}}$ is a frozen copy of the SFT model.

No sampling, reward model or value network is needed. It is a plain logistic-regression loss on the margin $\hat r_\theta(x,y_w) - \hat r_\theta(x,y_l)$.

### 4.5 What Does the DPO Update Do? (Gradient)

For a mechanistic understanding, look at the gradient of $\mathcal{L}_{\text{DPO}}$ with respect to $\theta$:

$$
\nabla_\theta \mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\beta\, \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\Big[\underbrace{\sigma\left(\hat{r}_\theta(x, y_l) - \hat{r}_\theta(x, y_w)\right)}_{\text{higher weight when reward estimate is wrong}} \Big[\underbrace{\nabla_\theta \log \pi(y_w \mid x)}_{\text{increase likelihood of } y_w} - \underbrace{\nabla_\theta \log \pi(y_l \mid x)}_{\text{decrease likelihood of } y_l}\Big]\Big]
$$

where

$$
\boxed{\hat{r}_\theta(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}}
$$

is the **reward implicitly defined** by the LM $\pi_\theta$ and the reference $\pi_{\text{ref}}$.

*(the paper states the gradient directly; this fills in why)* Write $u = \hat r_\theta(x,y_w) - \hat r_\theta(x,y_l)$, so the per-example loss is $-\log\sigma(u)$. Using $\frac{d}{du}\log\sigma(u) = 1-\sigma(u) = \sigma(-u)$:

- $\nabla_\theta[-\log\sigma(u)] = -\sigma(-u)\,\nabla_\theta u$;
- $\nabla_\theta u = \beta\left(\nabla_\theta\log\pi_\theta(y_w\mid x) - \nabla_\theta\log\pi_\theta(y_l\mid x)\right)$, because $\pi_{\text{ref}}$ is frozen;
- $\sigma(-u) = \sigma(\hat r_\theta(x,y_l) - \hat r_\theta(x,y_w))$.

Putting these together gives the expression above.

**Reading the gradient.** A descent step on $\mathcal{L}_{\text{DPO}}$:

- **increases the likelihood of the preferred completion** $y_w$;
- **decreases the likelihood of the dispreferred completion** $y_l$.

Crucially, each example is **weighted by how much higher the implicit reward model rates the dispreferred completion**, scaled by $\beta$. In other words, <mark style="background: #FFF3A3A6;">the weight reflects how *incorrectly* the implicit reward model orders the pair</mark>, accounting for the strength of the KL constraint:

- if the model already ranks $y_w$ well above $y_l$, the weight $\sigma(\cdot)\to 0$ and the example barely moves the model;
- if it ranks them the wrong way round, the weight approaches 1 and the update is large.

The paper's experiments show this weighting matters. A **naive version without the weighting coefficient** (just push $\log\pi(y_w)$ up and $\log\pi(y_l)$ down equally for every pair) can cause the LM to **degenerate** (Appendix Table 3). *(expanding on this)* <mark style="background: #ABF7F7A6;">The unweighted objective keeps pushing</mark> $\pi_\theta(y_l)$ <mark style="background: #ABF7F7A6;">toward zero even for pairs that are already learned, and the easiest way to lower a likelihood is to wreck the model's general text distribution.</mark>

### 4.6 DPO Outline

The general DPO pipeline:

1. **Build the dataset.** Sample completions $y_1, y_2 \sim \pi_{\text{ref}}(\cdot \mid x)$ for every prompt $x$. Label them with human preferences to build the offline dataset $\mathcal{D} = \{x^{(i)}, y_w^{(i)}, y_l^{(i)}\}_{i=1}^{N}$.
2. **Train.** Optimize $\pi_\theta$ to minimize $\mathcal{L}_{\text{DPO}}$ for the given $\pi_{\text{ref}}$, $\mathcal{D}$ and chosen $\beta$.

**Choosing $\pi_{\text{ref}}$.** In practice one wants to reuse public preference datasets rather than generate samples and collect new labels. Those datasets were sampled using some $\pi^{\text{SFT}}$, so:

- **If $\pi^{\text{SFT}}$ is available:** set $\pi_{\text{ref}} = \pi^{\text{SFT}}$.
- **If it is not available:** initialize $\pi_{\text{ref}}$ by maximizing the likelihood of the preferred completions,
  $$
  \pi_{\text{ref}} = \arg\max_\pi \mathbb{E}_{x, y_w \sim \mathcal{D}}\left[\log \pi(y_w \mid x)\right].
  $$
  This mitigates the **distribution shift** between the true reference distribution (which is unavailable) and the $\pi_{\text{ref}}$ that DPO uses.

Implementation details and hyperparameters are in Appendix B.

---

## 5. Theoretical Analysis of DPO

§4 derived DPO by algebra. §5 asks whether the reparameterization loses anything (it does not), and uses the same framework to explain why PPO-style actor-critic training is unstable.

### 5.1 Your Language Model Is Secretly a Reward Model

<mark style="background: #FFF3A3A6;">DPO skips both fitting an explicit reward and running RL, using a single maximum-likelihood objective.</mark> Eq. 5 says the objective is equivalent to a Bradley–Terry model with reward parameterization
$$
r^*(x, y) = \beta \log \frac{\pi^*_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)},
$$

and we optimize the parametric model $\pi_\theta$, equivalently to the reward-model optimization in Eq. 2 under the change of variables. <mark style="background: #FFF3A3A6;">The question is whether this parameterization **restricts** which reward models can be learned. It does not, and it lets us recover the optimal policy exactly.</mark>

**Definition 1 (equivalent rewards).** Two reward functions $r(x, y)$ and $r'(x, y)$ are **equivalent** iff $r(x, y) - r'(x, y) = f(x)$ for some function $f$. In other words, they differ only by a prompt-dependent shift. This is an **equivalence relation**: reflexive, symmetric and transitive. It partitions the set of reward functions into **classes**.

**Lemma 1.** Under the Plackett–Luce (and in particular Bradley–Terry) preference framework, two reward functions from the same class induce the **same preference distribution**.

**Lemma 2.** Two reward functions from the same equivalence class induce the **same optimal policy** under the constrained RL problem.

The proofs are in Appendix A.5. *(intuition, expanding on this)*

- **Lemma 1:** BT sees only $r(x,y_1)-r(x,y_2)$, and $f(x)$ cancels in that difference.
- **Lemma 2:** in Eq. 4, $e^{f(x)/\beta}$ multiplies both the numerator and $Z(x)$, so it cancels.

**What the lemmas imply.**

- Lemma 1 is a well-known **under-specification** issue of Plackett–Luce models: preferences cannot tell apart rewards in the same class. As a result, extra **identifiability constraints** are usually needed for any guarantee on the maximum-likelihood estimates from Eq. 2.
- Lemma 2 says that every reward in a class gives the same optimal policy. So for the final objective we only need to recover **any one** reward function from the optimal class.

**Theorem 1.** Under mild assumptions, all reward classes consistent with the Plackett–Luce (and Bradley–Terry in particular) models can be represented with the reparameterization

$$
r(x, y) = \beta \log \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)}
$$

for some model $\pi(y \mid x)$ and a given reference model $\pi_{\text{ref}}(y \mid x)$.

**Proof sketch.** Take any reward function $r(x, y)$. It induces an optimal model $\pi_r(y \mid x)$ via Eq. 4. To show that some member of $r$'s equivalence class has the reparameterized form, define the **projection** $f$:

$$
f(r; \pi_{\text{ref}}, \beta)(x, y) = r(x, y) - \beta \log \sum_y \pi_{\text{ref}}(y \mid x) \exp\left(\frac{1}{\beta} r(x, y)\right) \tag{8}
$$

The steps:

- $f$ simply normalizes $r$ by subtracting $\beta$ times the log of $\pi_r$'s partition function, i.e. $f(r)(x,y) = r(x,y) - \beta\log Z(x)$.
- The subtracted term depends only on the prefix $x$, so $f(r; \pi_{\text{ref}}, \beta)(x, y)$ is in the **same equivalence class** as $r(x, y)$.
- Replace $r$ with the right-hand side of Eq. 5, which holds for any reward function. The $\beta\log Z(x)$ terms cancel, leaving
  $$
  f(r; \pi_{\text{ref}}, \beta)(x, y) = \beta \log \frac{\pi_r(y \mid x)}{\pi_{\text{ref}}(y \mid x)}.
  $$

So $f$ produces a member of $r$'s class with exactly the desired form, and the proposed reparameterization **loses no generality**. ∎

**Which reward DPO picks.** Theorem 1 can also be read as specifying exactly *which* reward within each equivalence class DPO selects: the one satisfying

$$
\sum_y \underbrace{\pi_{\text{ref}}(y \mid x) \exp\left(\frac{1}{\beta} r(x, y)\right)}_{=\pi(y \mid x),\ \text{using Thm. 1 reparam.}} = 1 \tag{9}
$$

The underbrace holds because substituting $r = \beta\log(\pi/\pi_{\text{ref}})$ gives $\pi_{\text{ref}}\cdot\pi/\pi_{\text{ref}} = \pi$. So Eq. 9 just says $\pi(y \mid x)$ is a valid distribution: probabilities are positive and sum to 1.

Comparing with Eq. 4, Eq. 9 is the **partition function** of the optimal policy induced by $r$. DPO therefore selects the member of the class whose partition function equals 1, i.e. $\log Z(x) = 0$.

**The key insight of DPO:** constraints can be placed on the under-constrained Plackett–Luce (and BT) family so that:

- the class of representable reward models is preserved;
- the optimal policy in Eq. 4 becomes **analytically tractable for every prompt $x$**.

That is the sense in which "your language model is secretly a reward model": $\beta\log(\pi_\theta/\pi_{\text{ref}})$ *is* a reward function, and $\pi_\theta$ is its optimal policy.

### 5.2 Instability of Actor-Critic Algorithms

The same framework diagnoses instabilities in the standard **actor-critic** algorithms used for RLHF, such as PPO. In an actor-critic method, the "actor" is the policy and the "critic" is a learned value function that estimates expected future reward, used as a baseline.

Consider the RL fine-tuning step of §3. This connects to the **control-as-inference** framework, which treats reward maximization as probabilistic inference over a policy, applied to the constrained RL problem. Assume a parameterized model $\pi_\theta(y \mid x)$ and minimize $\mathbb{D}_{\text{KL}}[\pi_\theta(y \mid x) \,\|\, \pi^*(y \mid x)]$, where $\pi^*$ is the optimal policy induced by the reward $r_\phi(y, x)$. With some algebra this gives the objective

$$
\max_{\pi_\theta} \mathbb{E}_{\pi_\theta(y \mid x)}\Big[\underbrace{r_\phi(x, y) - \beta \log \sum_y \pi_{\text{ref}}(y \mid x) \exp\left(\frac{1}{\beta} r_\phi(x, y)\right)}_{f(r_\phi, \pi_{\text{ref}}, \beta)} - \underbrace{\beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}}_{\text{KL}}\Big] \tag{10}
$$

> [!warning] Possible source inconsistency
> The main text says "$\pi^*$ is the optimal policy from Eq. 7". Eq. 7 is the DPO loss; the closed-form optimal policy is Eq. 4, which is what is meant here. Similarly, §5.1 opens with "the optimization objective Eq. 5 is equivalent to a Bradley–Terry model", which reads more naturally as Eq. 7 (the loss), with Eq. 5 supplying the parameterization. Both are cross-reference slips that do not change the content.

**Reading Eq. 10.**

- **Same objective as prior work.** This is the objective optimized in prior RLHF work, using the **DPO-equivalent reward** for the reward class of $r_\phi$, i.e. the normalized $f(r_\phi, \pi_{\text{ref}}, \beta)$.
- **The normalization term.** The term $\beta\log\sum_y\pi_{\text{ref}}\exp(r_\phi/\beta)$ in $f$ can be interpreted as the **soft value function** of the reference policy $\pi_{\text{ref}}$. A **soft value** is a log-sum-exp version of the expected reward: it measures how good prompt $x$ is overall.
- **Why it causes instability.** This term does **not** change the optimal solution, because it depends only on $x$. Without it, though, the **policy gradient can have high variance**, which makes learning unstable.
- **How prior methods cope.** The normalization can be accommodated with a learned value function (the PPO critic), but that can also be hard to optimize. Alternatively, prior works normalized rewards with a **human-completion baseline**, which is essentially a single-sample Monte Carlo estimate of the normalizing term.
- **DPO avoids the problem.** The DPO reparameterization yields a reward function that **needs no baseline**: by Eq. 9 its normalizer is already 1.

---

## 6. Experiments

Having argued in theory that DPO solves the same problem as RLHF more cleanly, the paper tests this empirically with two questions:

1. In a controlled setting, how efficiently does DPO trade off maximizing reward against minimizing KL to the reference, compared with PPO and other preference-learning methods?
2. Does DPO hold up on larger models and harder RLHF tasks (summarization, dialogue)?

**Headline result:** with almost no hyperparameter tuning, DPO performs as well as or better than strong baselines, including RLHF with PPO and **Best of $N$** (§6.1). Experimental details are in Appendix C.

### 6.1 Tasks, Evaluation and Baselines

**Tasks.** All experiments learn a policy from a preference dataset $\mathcal{D} = \{x^{(i)}, y_w^{(i)}, y_l^{(i)}\}_{i=1}^N$. There are three open-ended text-generation tasks:

| Task | Prompt $x$ | Desired output $y$ | Preference data | SFT model |
|---|---|---|---|---|
| **Controlled sentiment generation** | prefix of a movie review from **IMDb** | a continuation with **positive sentiment** | *generated* using a pre-trained **sentiment classifier**: $p(\text{positive}\mid x,y_w) > p(\text{positive}\mid x,y_l)$ | **GPT-2-large** fine-tuned to convergence on IMDb train reviews |
| **Summarization** | a Reddit forum post | a summary of the post's main points | Reddit **TL;DR** dataset with human preferences gathered by Stiennon et al. (on samples from a different, similarly trained SFT model) | an SFT model fine-tuned on human-written summaries, trained with the **TRLX** RLHF framework |
| **Single-turn dialogue** | a human query (anything from astrophysics questions to relationship advice) | an engaging, helpful response | **Anthropic Helpful and Harmless (HH)** dataset: ~170k human–assistant dialogues, each ending with two responses from a large (unknown) LM and a label for the human-preferred one | none available, so an off-the-shelf LM is fine-tuned on only the preferred completions |

Two terms from the table:

- **IMDb** is a movie-review dataset.
- **TL;DR** ("too long; didn't read") posts are Reddit posts that end with a short author-written summary.

**Evaluation.** Two approaches:

- **Reward–KL frontier (sentiment only).** The question is how well each algorithm optimizes the *constrained* reward-maximization objective. Each algorithm is scored by the frontier of achieved reward versus KL-divergence from the reference policy. This is computable only here, because the ground-truth reward (the sentiment classifier) is available.
- **Win rate (summarization and dialogue).** In the real world the ground-truth reward is unknown. Algorithms are scored by their **win rate** against a baseline policy, using **GPT-4** as a proxy for human evaluation of summary quality and response helpfulness. The win rate is the fraction of head-to-head comparisons a method's output wins.
  - Summarization baseline: the reference summaries in the test set.
  - Dialogue baseline: the preferred response in the test set.

Studies suggest LMs can be better automated evaluators than existing metrics. To justify using GPT-4, the authors also run a human study (§6.5). It finds GPT-4 judgments correlate strongly with humans: human–GPT-4 agreement is typically similar to or higher than inter-human agreement.

**Methods compared.**

| Method | What it is |
|---|---|
| **GPT-J** (zero-shot) | prompting a 6B open LM with no examples (summarization) |
| **Pythia-2.8B** (2-shot) | prompting a 2.8B open LM with two examples (dialogue) |
| **SFT** | the supervised fine-tuned model |
| **Preferred-FT** | supervised fine-tuning on only the chosen completions $y_w$, starting from the SFT model (sentiment, summarization) or a generic LM (dialogue) |
| **Unlikelihood** | maximize the probability of $y_w$ and *minimize* that of $y_l$, with an optional coefficient $\alpha \in [0, 1]$ on the "unlikelihood" term (pseudo-supervised) |
| **PPO** | RLHF with a reward function learned from the preference data |
| **PPO-GT** | an *oracle* PPO that learns from the **ground-truth** reward (sentiment setting only); two implementations, an off-the-shelf **TRL** one and a modified one that normalizes rewards and further tunes hyperparameters (the latter modifications are also used for "normal" PPO) |
| **Best of $N$** | sample $N$ responses from the SFT model (or Preferred-FT in dialogue) and return the one with the highest learned reward |

**Best of $N$** decouples the quality of the reward model from the PPO optimization and performs well. However, it is **computationally impractical** even for moderate $N$, because it samples $N$ completions for *every* query at test time.

### 6.2 How Well Can DPO Optimize the RLHF Objective?

The KL-constrained objective balances exploiting reward against staying close to the reference. So algorithms must be compared on **both** reward and KL: slightly higher reward at a much higher KL is not necessarily desirable.

**Setup.**

- Several training runs per algorithm, each with a different conservativeness hyperparameter:

| Algorithm             | Values swept      |
| --------------------- | ----------------- |
| PPO target KL         | 3, 6, 9, 12       |
| DPO $\beta$           | 0.05, 0.1, 1, 5   |
| Unlikelihood $\alpha$ | 0.05, 0.1, 0.5, 1 |
| Preferred-FT          | random seeds      |

  That is **22 runs** in total.
- Every 100 training steps until convergence, each policy is evaluated on a set of test prompts, computing:
  - the average reward under the **true** reward function;
  - the average **sequence-level KL** to the reference, $\text{KL}(\pi\|\pi_{\text{ref}})$, i.e. the sum of the per-timestep (per-token) KL divergences.

![[RA2_DPO_fig2_frontier_tldr.png|700]]
*Figure 2 — Left: reward vs $\text{KL}(\pi_\theta\|\pi_{\text{ref}})$ on IMDb sentiment. DPO's points (yellow) reach reward ≈ 1.0 at KL of only ≈ 3–5 and sit above every other method at every KL, so DPO gives the highest expected reward at all KL values. Right: TL;DR summarization win rate against human-written reference summaries (GPT-4 judge) vs sampling temperature. DPO is about 0.62 at temperatures 0–0.25 and degrades gently; PPO peaks at about 0.57 at temperature 0 and collapses to about 0.07 at temperature 1.0. The dashed line marks 50%.*

**Result.** DPO produces **by far the most efficient frontier**: the highest reward while keeping KL low. This is notable for two reasons:

1. DPO and PPO optimize the **same objective**, yet DPO is notably more efficient; its reward/KL tradeoff **strictly dominates** PPO's.
2. DPO achieves a better frontier than PPO **even when PPO can access ground-truth rewards** (PPO-GT).

### 6.3 Can DPO Scale to Real Preference Datasets?

The next step is real human preferences on summarization and single-turn dialogue.

**Summarization (TL;DR).**

- **Why GPT-4 win rates.** Automatic metrics such as **ROUGE** (n-gram overlap with a reference summary) can correlate poorly with human preferences, and prior work found that fine-tuning with PPO on human preferences gives more effective summaries.
- **Protocol.** Sample each method's completions on the TL;DR test split and compute the average win rate against the reference completions. Sampling temperatures range from 0.0 to 1.0 (Figure 2, right). **Sampling temperature** rescales the logits before sampling: 0 is greedy decoding, higher values give more random text.
- **Shared starting point.** DPO, PPO and Preferred-FT all fine-tune the same **GPT-J SFT model**.

Results:

| Finding | Numbers |
|---|---|
| DPO vs PPO at their best | DPO ≈ **61%** win rate at temperature 0.0; PPO ≈ **57%** at its optimal temperature 0.0 |
| vs Best of $N$ | DPO reaches a **higher maximum win rate** |
| Robustness | DPO is much more robust to sampling temperature; PPO can degrade to base GPT-J level at high temperatures |
| Preferred-FT | does not improve significantly over SFT |
| Human head-to-head (§6.5) | DPO samples at temperature 0.25 were preferred **58%** of the time over PPO samples at temperature 0 |

DPO's $\beta$ was **not meaningfully tuned**, so these results may *underestimate* its potential.

**Single-turn dialogue (Anthropic HH).**

- **Data.** The subset of the HH test split with one step of human–assistant interaction. GPT-4 evaluations use the preferred test completions as the reference for computing win rates.
- **Reference model.** There is no standard SFT model for this task, so the authors start from a pre-trained **Pythia-2.8B** and use Preferred-FT to train a reference model on the chosen completions. This keeps completions within the model's distribution. They then train with DPO.
- **Baselines.**
  - Best of 128 Preferred-FT completions; the Best of $N$ baseline plateaus at 128 for this task (Appendix Figure 4).
  - A 2-shot prompted Pythia-2.8B.
  - An RLHF model trained with PPO on HH from a well-known source. No prompt or temperature was found that beat the base Pythia-2.8B, so TL;DR evidence is used instead: there DPO beats PPO, and both methods optimize the same reward. Best of 128 is therefore treated as a rough proxy for PPO-level performance.

![[RA2_DPO_fig3_dialogue.png|700]]
*Figure 3 — Left: GPT-4-judged win rate against the chosen (preferred) responses in Anthropic-HH one-step dialogue, vs sampling temperature. DPO rises from about 0.37 at temperature 0.25 to about 0.6–0.63 at 0.75–1.0 and is the only method above 50% alongside Best of 128. Preferred-FT stays below 0.45 and 2-shot Pythia-2.8B stays below 0.25. Right: DPO's win rate over fine-tuning steps at temperatures 1.0 and 0.7. It climbs above 50% within about 300 steps and stays roughly flat at 0.55–0.64 afterwards, so improvement over the dataset labels is fast and stable.*

**Result.** DPO performs as well as or better than the best-performing temperature of each method. It is the **only computationally efficient method that improves over the preferred completions** in HH, and it is similar to or better than the computationally demanding Best of 128. Figure 3 (right) shows DPO converges to its best performance relatively quickly.

### 6.4 Generalization to a New Input Distribution

To compare PPO and DPO under **distribution shift**, the Reddit TL;DR policies are evaluated on a different distribution: news articles from the test split of **CNN/DailyMail**.

- **Settings:** the best sampling temperatures from TL;DR (0 and 0.25).
- **Metric:** GPT-4 win rate against the datasets' ground-truth summaries.
- **Prompt:** the same GPT-4 (C) prompt as Reddit TL;DR, with "forum post" replaced by "news article".

**Table 1 — GPT-4 win rates vs ground-truth summaries, out-of-distribution CNN/DailyMail**

| Alg. | Temp 0 | Temp 0.25 |
|---|---|---|
| DPO | **0.36** | **0.31** |
| PPO | 0.26 | 0.23 |

DPO keeps outperforming PPO by a significant margin. This is initial evidence that DPO policies generalize at least as well as PPO policies, **even though DPO does not use the additional unlabeled Reddit TL;DR prompts that PPO uses**. PPO samples fresh completions on extra prompts during RL.

### 6.5 Validating GPT-4 Judgments with Human Judgments

A human study checks whether GPT-4's judgments are reliable, using the TL;DR results and two GPT-4 prompts:

- **GPT-4 (S)** ("simple") asks which summary better summarizes the important information in the post.
- **GPT-4 (C)** ("concise") also asks which summary is more concise. It was evaluated because GPT-4 with the (S) prompt prefers **longer, more repetitive** summaries than humans do.

The full prompts are in Appendix C.2.

**Design.** Three comparisons, chosen to cover a range of sample qualities:

- the **highest**-performing method: DPO, temperature 0.25;
- the **lowest**: PPO, temperature 1.0 ("PPO-1");
- a **middle** one: SFT, temperature 0.25.

All three are compared against greedily sampled PPO (temperature 0), PPO's best-performing temperature.

**Table 2 — human vs GPT-4 win rates and per-judgment agreement on TL;DR (each column: method vs PPO at temp 0)**

| | DPO | SFT | PPO-1 |
|---|---|---|---|
| N respondents | 272 | 122 | 199 |
| GPT-4 (S) win % | 47 | 27 | 13 |
| GPT-4 (C) win % | 54 | 32 | 12 |
| Human win % | 58 | 43 | 17 |
| GPT-4 (S)–H agree | 70 | 77 | 86 |
| GPT-4 (C)–H agree | 67 | 79 | 85 |
| H–H agree | 65 | – | 87 |

**Findings.**

- With both prompts, GPT-4 agrees with humans about **as often as humans agree with each other**, so GPT-4 is a reasonable proxy for human evaluation.
- Because human raters were limited, multiple human judgments were collected only for the DPO and PPO-1 comparisons, hence the "–" for SFT's H–H agreement.
- Overall the **GPT-4 (C)** prompt gives win rates more representative of humans (54 vs human 58 for DPO, compared with 47 for (S)). It is therefore the prompt used for the main results in §6.3.

Details of the human study (web interface, list of volunteers) are in Appendix D.3.

---

## 7. Discussion, Limitations and Future Work

The experiments back up the theoretical claim, and the paper closes by restating the contribution and its open questions.

**Summary.** Learning from preferences is a powerful, scalable framework for training capable, aligned LMs.

- DPO is a simple training paradigm for learning from preferences **without RL**.
- Rather than force preference learning into a standard RL setting to use off-the-shelf RL algorithms, DPO identifies a **mapping between LM policies and reward functions**. This lets an LM be trained to satisfy human preferences *directly*, with a simple cross-entropy loss, without RL and without loss of generality.
- With virtually no hyperparameter tuning, DPO performs similarly to or better than existing RLHF algorithms, including PPO-based ones. It therefore meaningfully lowers the barrier to training more LMs from human preferences.

**Limitations and future work.**

1. **Out-of-distribution generalization.** How does a DPO policy generalize out of distribution compared with learning from an explicit reward function? Initial results (§6.4) suggest it is similar to PPO, but a more comprehensive study is needed. For example, can **self-labeling** from the DPO policy make effective use of unlabeled prompts, as PPO does?
2. **Reward over-optimization.** How does **reward over-optimization** show up in the direct preference setting? Over-optimization means the policy exploits the reward proxy until true quality drops. Is the slight decrease in performance in Figure 3 (right) an instance of it?
3. **Scale.** Models up to 6B parameters were evaluated; scaling DPO to state-of-the-art models that are orders of magnitude larger is future work.
4. **Evaluation.** GPT-4 win rates depend on the prompt. Future work could study how best to elicit high-quality judgments from automated systems.
5. **Other modalities.** DPO has many possible applications beyond LMs, including training generative models in other modalities from human preferences.

---

## Numerical Problems

These problems use only the formulas from the main paper (Eqs. 1–9 and the gradient in §4.5). Logarithms are natural logs.

**Problem 1 — Bradley–Terry probability and shift invariance.** A latent reward model gives $r^*(x,y_1) = 1.2$ and $r^*(x,y_2) = 0.4$. (a) Compute $p^*(y_1 \succ y_2 \mid x)$. (b) Add $f(x) = 5$ to both rewards and recompute.

*Solution.* Using the sigmoid form of Eq. 1, $p^* = \sigma(1.2 - 0.4) = \sigma(0.8) = 1/(1+e^{-0.8})$. Since $e^{-0.8} \approx 0.4493$, this is $1/1.4493 \approx 0.690$.

For (b), the rewards become 6.2 and 5.4. Their difference is still 0.8, so the probability is unchanged. This is Lemma 1 in action: rewards in the same equivalence class induce the same preferences.

$$
\boxed{p^*(y_1\succ y_2\mid x) \approx 0.690 \text{ in both cases}}
$$

**Problem 2 — Reward-model loss for a correctly and an incorrectly ranked pair.** A reward model outputs $r_\phi(x,y_w) = 0.5$, $r_\phi(x,y_l) = 1.5$ on one labeled pair. (a) Compute this pair's contribution to $\mathcal{L}_R$ (Eq. 2). (b) Recompute if the scores were swapped.

*Solution.* Use $-\log\sigma(u) = \log(1+e^{-u})$.

- In (a) the margin is $u = 0.5 - 1.5 = -1$, so the loss is $\log(1+e^{1}) = \log 3.7183 \approx 1.313$.
- In (b) the margin is $u = +1$, so the loss is $\log(1 + e^{-1}) = \log 1.3679 \approx 0.313$.

The wrongly ranked pair costs about four times as much, which is what drives the reward model to fix its ordering.

$$
\boxed{\text{(a) } 1.313 \qquad \text{(b) } 0.313}
$$

**Problem 3 — The closed-form optimal policy (Eq. 4) and the role of $\beta$.** Take a toy prompt with only two possible completions $a, b$. The reference policy is $\pi_{\text{ref}}(a) = 0.6$, $\pi_{\text{ref}}(b) = 0.4$, and the rewards are $r(x,a) = 1$, $r(x,b) = 3$. Compute $Z(x)$ and $\pi_r$ for (a) $\beta = 1$ and (b) $\beta = 2$.

*Solution.* Eq. 4 multiplies each reference probability by $e^{r/\beta}$ and renormalizes.

For $\beta = 1$:
- the unnormalized weights are $0.6e^{1} = 1.6310$ and $0.4e^{3} = 8.0342$;
- so $Z(x) = 9.6652$;
- and $\pi_r(a) = 1.6310/9.6652 \approx 0.169$, $\pi_r(b) \approx 0.831$.

For $\beta = 2$:
- the weights are $0.6e^{0.5} = 0.9892$ and $0.4e^{1.5} = 1.7927$;
- so $Z(x) = 2.7819$;
- and $\pi_r(a) \approx 0.356$, $\pi_r(b) \approx 0.644$.

Doubling $\beta$ keeps the policy much closer to $\pi_{\text{ref}} = (0.6, 0.4)$, which is the KL constraint's job.

$$
\boxed{\beta=1:\ (0.169,\ 0.831),\ Z=9.665 \qquad \beta=2:\ (0.356,\ 0.644),\ Z=2.782}
$$

**Problem 4 — Recovering the reward from the policy (Eq. 5) and the projection (Eqs. 8–9).** Using the $\beta = 1$ result of Problem 3: (a) verify Eq. 5 for completion $a$; (b) compute the DPO-selected reward $f(r)(x,y) = r(x,y) - \beta\log Z(x)$ for both completions; (c) check that it satisfies Eq. 9.

*Solution.*

- **(a)** Eq. 5 gives $r(x,a) = \log(0.16875/0.6) + \log 9.6652 = \log 0.28125 + 2.2685 = -1.2685 + 2.2685 = 1.000$, matching the original reward of 1.
- **(b)** Subtracting $\log Z = 2.2685$ gives $f(r)(x,a) = -1.2685$ and $f(r)(x,b) = 0.7315$. Each equals $\log(\pi_r/\pi_{\text{ref}})$, the Theorem 1 form.
- **(c)** Eq. 9 gives $0.6e^{-1.2685} + 0.4e^{0.7315} = 0.6(0.28125) + 0.4(2.0781) = 0.16875 + 0.83125 = 1.000$.

So the selected reward has partition function 1, as Eq. 9 requires.

$$
\boxed{f(r) = (-1.2685,\ 0.7315),\quad \textstyle\sum_y \pi_{\text{ref}}e^{f(r)/\beta} = 1}
$$

**Problem 5 — Computing the DPO loss and gradient weight on one pair.** Let $\beta = 0.1$ and the sequence log-probabilities be

$$
\log\pi_\theta(y_w\mid x) = -12,\quad \log\pi_{\text{ref}}(y_w\mid x) = -14,\quad \log\pi_\theta(y_l\mid x) = -15,\quad \log\pi_{\text{ref}}(y_l\mid x) = -13.
$$

Compute (a) the implicit rewards and margin, (b) the per-example DPO loss, (c) the gradient weight $\sigma(\hat r_\theta(y_l) - \hat r_\theta(y_w))$, and (d) the loss and weight at initialization, when $\pi_\theta = \pi_{\text{ref}}$.

*Solution.*

- **(a)** $\hat r_\theta(x,y_w) = 0.1(-12+14) = 0.2$ and $\hat r_\theta(x,y_l) = 0.1(-15+13) = -0.2$, so the margin is $u = 0.4$.
- **(b)** $\mathcal{L} = -\log\sigma(0.4) = \log(1 + e^{-0.4}) = \log 1.6703 \approx 0.513$.
- **(c)** The weight is $\sigma(-0.4) = 1/(1+e^{0.4}) = 1/2.4918 \approx 0.401$.
- **(d)** At initialization both log-ratios are 0, so $u = 0$. The loss is $\log 2 \approx 0.693$ and the weight is $\sigma(0) = 0.5$.

Training has already moved the policy in the right direction (the loss fell from 0.693 to 0.513), so this pair now gets a somewhat smaller weight.

$$
\boxed{u=0.4,\quad \mathcal{L}\approx 0.513,\quad \text{weight}\approx 0.401\quad(\text{init: } 0.693,\ 0.5)}
$$

**Problem 6 — Which pair dominates the update?** In a batch, pair A has implicit-reward margin $u_A = \hat r_\theta(y_w) - \hat r_\theta(y_l) = +3$ (already well ordered) and pair B has $u_B = -1$ (wrongly ordered). Compute each gradient weight and their ratio.

*Solution.* The weight is $\sigma(-u)$:

- pair A: $\sigma(-3) = 1/(1+e^{3}) = 1/21.086 \approx 0.047$;
- pair B: $\sigma(1) = 1/(1+e^{-1}) = 1/1.3679 \approx 0.731$.

Pair B's gradient is about $0.731/0.047 \approx 15.4$ times larger. This is the importance weighting of §4.5: the update focuses on pairs the implicit reward model gets wrong. The naive unweighted version would treat both pairs equally.

$$
\boxed{w_A \approx 0.047,\quad w_B \approx 0.731,\quad w_B/w_A \approx 15.4}
$$

**Problem 7 — The KL-shaped RLHF reward (§3.3).** For one sample, $r_\phi(x,y) = 2.0$, $\beta = 0.1$, $\log\pi_\theta(y\mid x) = -20$ and $\log\pi_{\text{ref}}(y\mid x) = -25$. Compute the reward PPO maximizes.

*Solution.* The shaped reward is

$$
r = r_\phi - \beta(\log\pi_\theta - \log\pi_{\text{ref}}) = 2.0 - 0.1(-20 + 25) = 2.0 - 0.5 = 1.5.
$$

The policy made $y$ $e^{5}\approx 148$ times more likely than the reference did, so it pays a KL penalty of 0.5.

$$
\boxed{r = 1.5}
$$

---

## Formula Summary Sheet

| Concept | Formula |
|---|---|
| Bradley–Terry preference model (Eq. 1) | $p^*(y_1\succ y_2\mid x) = \dfrac{\exp r^*(x,y_1)}{\exp r^*(x,y_1)+\exp r^*(x,y_2)} = \sigma\big(r^*(x,y_1)-r^*(x,y_2)\big)$ |
| Logistic function | $\sigma(z) = 1/(1+e^{-z})$, $\ \sigma(-z) = 1-\sigma(z)$ |
| Preference dataset | $\mathcal{D} = \{x^{(i)}, y_w^{(i)}, y_l^{(i)}\}_{i=1}^N$ |
| Reward-model loss (Eq. 2) | $\mathcal{L}_R(r_\phi,\mathcal{D}) = -\mathbb{E}_{\mathcal{D}}\left[\log\sigma\big(r_\phi(x,y_w)-r_\phi(x,y_l)\big)\right]$ |
| Reward normalization | $\mathbb{E}_{x,y\sim\mathcal{D}}[r_\phi(x,y)] = 0$ |
| KL-constrained RL objective (Eq. 3) | $\max_{\pi_\theta}\mathbb{E}_{x\sim\mathcal{D},y\sim\pi_\theta}[r_\phi(x,y)] - \beta\,\mathbb{D}_{\text{KL}}[\pi_\theta(y\mid x)\,\|\,\pi_{\text{ref}}(y\mid x)]$ |
| PPO shaped reward | $r(x,y) = r_\phi(x,y) - \beta\big(\log\pi_\theta(y\mid x)-\log\pi_{\text{ref}}(y\mid x)\big)$ |
| Optimal policy (Eq. 4) | $\pi_r(y\mid x) = \frac{1}{Z(x)}\pi_{\text{ref}}(y\mid x)\exp\big(\tfrac{1}{\beta}r(x,y)\big)$ |
| Partition function | $Z(x) = \sum_y \pi_{\text{ref}}(y\mid x)\exp\big(\tfrac1\beta r(x,y)\big)$ |
| Reward from policy (Eq. 5) | $r(x,y) = \beta\log\dfrac{\pi_r(y\mid x)}{\pi_{\text{ref}}(y\mid x)} + \beta\log Z(x)$ |
| Preference via policy (Eq. 6) | $p^*(y_1\succ y_2\mid x) = \Big[1+\exp\Big(\beta\log\frac{\pi^*(y_2\mid x)}{\pi_{\text{ref}}(y_2\mid x)} - \beta\log\frac{\pi^*(y_1\mid x)}{\pi_{\text{ref}}(y_1\mid x)}\Big)\Big]^{-1}$ |
| DPO loss (Eq. 7) | $\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{\mathcal{D}}\Big[\log\sigma\Big(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\Big)\Big]$ |
| Implicit reward | $\hat r_\theta(x,y) = \beta\log\dfrac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)}$ |
| DPO gradient | $\nabla_\theta\mathcal{L}_{\text{DPO}} = -\beta\,\mathbb{E}_{\mathcal{D}}\big[\sigma(\hat r_\theta(x,y_l)-\hat r_\theta(x,y_w))\,(\nabla_\theta\log\pi(y_w\mid x) - \nabla_\theta\log\pi(y_l\mid x))\big]$ |
| $\pi_{\text{ref}}$ when no SFT model | $\pi_{\text{ref}} = \arg\max_\pi\mathbb{E}_{x,y_w\sim\mathcal{D}}[\log\pi(y_w\mid x)]$ |
| Equivalent rewards (Def. 1) | $r \sim r' \iff r(x,y)-r'(x,y) = f(x)$ |
| Theorem 1 reparameterization | $r(x,y) = \beta\log\dfrac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)}$ |
| Projection (Eq. 8) | $f(r;\pi_{\text{ref}},\beta)(x,y) = r(x,y) - \beta\log\sum_y\pi_{\text{ref}}(y\mid x)\exp\big(\tfrac1\beta r(x,y)\big)$ |
| DPO's selected reward (Eq. 9) | $\sum_y \pi_{\text{ref}}(y\mid x)\exp\big(\tfrac1\beta r(x,y)\big) = 1$ |
| Actor-critic objective (Eq. 10) | $\max_{\pi_\theta}\mathbb{E}_{\pi_\theta}\Big[r_\phi(x,y) - \beta\log\sum_y\pi_{\text{ref}}e^{r_\phi/\beta} - \beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)}\Big]$ |

---

## Cheat Sheet & Recall Aids

**The one-line story.** The KL-constrained RLHF optimum is $\pi^* \propto \pi_{\text{ref}}e^{r/\beta}$. Invert it to get $r = \beta\log(\pi^*/\pi_{\text{ref}}) + \beta\log Z$, plug that into Bradley–Terry, $Z$ cancels, and you get a logistic loss on the policy.

**Mnemonic for the derivation: "O-I-C-L"** — **O**ptimum (Eq. 4) → **I**nvert for reward (Eq. 5) → **C**ancel $Z$ in BT (Eq. 6) → **L**oss (Eq. 7).

**Mnemonic for the RLHF pipeline: "S-R-R"** — **S**FT → **R**eward model → **R**L (PPO). DPO keeps the S and replaces "R-R" with one supervised loss.

**Gotchas.**
- DPO is **not** reward-free in theory: it fits an *implicit* reward $\hat r_\theta = \beta\log(\pi_\theta/\pi_{\text{ref}})$. It is reward-*model*-free and RL-free.
- $Z(x)$ cancels because both completions share the **same prompt** $x$. That is why DPO needs *pairs* for the same $x$.
- Large $\beta$ means the policy stays close to $\pi_{\text{ref}}$; small $\beta$ means it drifts more. $\beta$ is **not** a learning rate.
- The gradient weight $\sigma(\hat r_l - \hat r_w)$ is **large when the model is wrong**. It does not shrink to reduce noise; removing it makes the model degenerate.
- DPO is cheaper because it does **not sample from the LM during training**. It uses a fixed offline dataset, with no reward model and no value network.
- The PPO instability lives in the normalizer $\beta\log Z(x)$ (soft value of $\pi_{\text{ref}}$). It does not change the optimum but blows up gradient variance. DPO's reward has $Z = 1$ by construction (Eq. 9), so it needs no baseline.
- In the sentiment experiment, preferences come from a **sentiment classifier**, not humans. That is why a ground-truth reward, and hence a reward–KL frontier, exists only there.
- DPO beats PPO **even PPO-GT** (oracle reward) on the frontier.
- Two TL;DR comparisons are easy to mix up:
  - GPT-4: DPO 61% vs PPO 57% against the reference summaries;
  - humans: DPO preferred 58% over PPO head-to-head.

**Quick-reference intuitions.**
- *Eq. 4:* reweight the reference model by $e^{r/\beta}$.
- *Eq. 5:* a reward is a scaled log-ratio of policies, plus something that depends only on $x$.
- *Theorem 1:* any reward class contains a member of the form $\beta\log(\pi/\pi_{\text{ref}})$, so nothing is lost.
- *Lemma 1 / Lemma 2:* adding $f(x)$ to a reward changes neither the preferences nor the optimal policy.
- *Experiments:* better reward–KL frontier, higher win rates, more robust to temperature, better out of distribution (0.36 vs 0.26), and GPT-4 agrees with humans about as much as humans agree with each other.

**Recall checklist.**
- [ ] Write Eq. 1 in both exp-ratio and sigmoid form.
- [ ] Write Eq. 3 and give two reasons for the KL term.
- [ ] State Eq. 4, define $Z(x)$, and say why it is intractable.
- [ ] Derive Eq. 5 from Eq. 4 and show $Z$ cancelling to get Eq. 7.
- [ ] Write the DPO gradient and explain each of its three parts.
- [ ] State Definition 1, Lemmas 1–2 and Theorem 1, and what Eq. 9 selects.
- [ ] Explain the source of actor-critic instability (Eq. 10).
- [ ] Name the 3 tasks, their datasets and evaluation methods, and the headline numbers (61/57, 0.36/0.26, Table 2 agreement).
- [ ] List the limitations in §7.
