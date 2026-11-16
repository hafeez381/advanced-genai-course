---
type: lecture
course: "[[_CS5326 Advanced GenAI|CS5326 Advanced GenAI]]"
source:
  - "[[006-FFN-MOE.pdf]]"
  - "[[006-FFN-part 1.pdf]]"
tags:
  - course/cs5326
textfold: md-lecture-09-ffns-dense-and-moe-0305b5fc
---

%% nav:start %%
[[_CS5326 Advanced GenAI|CS5326 Advanced GenAI]] · ← [[Lecture_08_Activations]] · → [[Lecture_10_Neural_Scaling_Laws]]
Concepts: [[Feed-Forward Network]], [[Mixture of Experts]], [[Router]], [[Sparse Model]]
%% nav:end %%

# FFNs: Dense and Mixture-of-Experts (CS5326, Lecture 9)

*Follows [[Lecture_08_Activations#7. Separately Regulated Activations: GLU and Gated FFNs|Lecture 8 §7]], which settled what nonlinearity goes inside the FFN (feed-forward network) sublayer and ended on gated FFNs such as SwiGLU. This lecture steps back to the FFN as a whole: what job it does next to attention, why a dense FFN makes every token pay for every parameter, and how a Mixture-of-Experts (MoE) layer replaces one FFN with many, plus a router that picks a few of them per token. The second half follows the full deck through routing arithmetic, how an MoE is trained when selection is discrete, load balancing and expert capacity, and the family of MoE variants.*

> [!warning] Possible source inconsistency: missing slides
> Built from the full deck `006-FFN-MOE.pdf` ([[#1. The Position-Wise Feed-Forward Network|§1]]–[[#5.3 The routing mechanism: a softmax gate|§5.3]] were first written from the shorter `006-FFN-part 1.pdf`, whose slides match). Slide numbers 6, 16 and 58 are absent from the PDF, presumably hidden slides; no content gap is visible around them.

The deck's reading list: Jacobs et al. (1991), *Adaptive Mixtures of Local Experts* (MoE foundations); Shazeer et al. (2017), *Outrageously Large Neural Networks* (sparse MoE and routing); Fedus et al. (2022), *Switch Transformers* (top-1 routing and load balancing); Jiang et al. (2024), *Mixtral of Experts* (top-2 routing); Dai et al. (2024), *DeepSeekMoE* (expert specialisation); DeepSeek-AI (2024), *DeepSeek-V3 Technical Report* (modern routing and balancing); Gale et al. (2023), *MegaBlocks* (efficient MoE execution); and Maarten Grootendorst's *A Visual Guide to Mixture of Experts*.

## Topic Inventory

- 1. The Position-Wise Feed-Forward Network
  - 1.1 Attention gathers, the FFN processes
  - 1.2 The FFN equations and shapes
  - 1.3 Why expand, why a nonlinearity, why return to $d$
- 2. From Dense Models to the MoE Question
  - 2.1 Dense models: every token uses everything
  - 2.2 Two motivations for MoE: learn better, scale better
  - 2.3 Gradient interference
  - 2.4 The specialist intuition
- 3. A Worked Example: The Absolute Value with Two Experts
  - 3.1 Approach A: one dense network
  - 3.2 Approach B: two experts and a router
  - 3.3 Observations
- 4. Mixture of Experts in LLMs
  - 4.1 Experts and the router
  - 4.2 Dense layers, the thing MoE replaces
  - 4.3 Sparse layers and sparse models
  - 4.4 What does an expert learn?
  - 4.5 The architecture of experts and routes through layers
- 5. Routing
  - 5.1 The sparse MoE recipe per token
  - 5.2 Combining expert outputs
  - 5.3 The routing mechanism: a softmax gate
  - 5.4 The MoE layer: dense MoE vs sparse MoE
  - 5.5 How the router works: score, select, process, combine
  - 5.6 Pre-routing: the router decides from token features, not expert results
  - 5.7 A first note on renormalization
- 6. Total vs Active Parameters
  - 6.1 Dense models revisited: the same machinery for every token
  - 6.2 Stored vs used parameters
  - 6.3 Efficiency in big-O terms
  - 6.4 Dense FFN layer vs MoE FFN layer, side by side
- 7. Training an MoE
  - 7.1 Two things are learned
  - 7.2 The gradient into an expert
  - 7.3 The gradient into the router weight, and the two gradients
  - 7.4 Subtle point: $w_2$ is not a learned parameter
  - 7.5 The top-$k$ decision is not differentiable
  - 7.6 The overall picture, and why load balancing is needed
  - 7.7 Dense MoE: fully differentiable routing
  - 7.8 Back to renormalization
- 8. Expert Collapse and Load Balancing
  - 8.1 Why some experts starve: a self-reinforcing loop
  - 8.2 The goal and the balancing objective
  - 8.3 The two quantities $f_i$ and $\bar p_i$
  - 8.4 Why multiplying them encourages balance
  - 8.5 We cannot change $f_i$ by gradients, but we can change $\bar p_i$
  - 8.6 How changing probabilities eventually changes assignments
  - 8.7 Why multiply by $N$?
  - 8.8 Does balanced usage mean useful specialisation?
  - 8.9 When is the balancing loss computed?
- 9. Other Load-Balancing Methods and Expert Capacity
  - 9.1 Main approaches
  - 9.2 Routing noise: encourage exploration
  - 9.3 Bias-based balancing: control expert loads directly
  - 9.4 Expert capacity: what if an expert is overloaded?
- 10. Types of MoE
  - 10.1 The family at a glance
  - 10.2 How the main architectures differ
- Numerical Problems
- Formula Summary Sheet
- Summary

---

## 1. The Position-Wise Feed-Forward Network

### 1.1 Attention gathers, the FFN processes

Each Transformer block has two sublayers with different jobs. **Attention** lets a token look at other tokens and gather information from them ([[Lecture_05_Attention_and_Multi_Head_Attention#6. Attention: Intuition|Lecture 5 §6]]). The **feed-forward network (FFN)** then lets each token "take a moment to think" about what it gathered: it applies a learned nonlinear transformation to each token's updated representation.

![[L09_ffn_block.png|180]]

*The FFN block: a linear layer expands the token vector, ReLU acts on each hidden unit, and a second linear layer projects back to width $d$.*

The FFN is **position-wise**: it runs on each token's vector independently and applies the same transformation, with the same weights, at every position in a layer. "Independently" means the FFN never reads another position directly. It does not need to, because attention has already mixed context into its input.

> **Attention mixes information between tokens; the FFN transforms features within each token.**

### 1.2 The FFN equations and shapes

The original Transformer FFN is two linear maps with a ReLU between them:

$$
\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2 = \max(xW_1 + b_1,\, 0)\,W_2 + b_2 .
$$

ReLU acts as a gatekeeper: positive values pass unchanged and negative values become zero ([[Lecture_08_Activations#4. ReLU|Lecture 8 §4]]). The general form replaces ReLU with any nonlinear activation $\phi$ (GELU, SiLU, or a gated variant such as SwiGLU, [[Lecture_08_Activations#7.6 ReGLU, GEGLU and SwiGLU|Lecture 8 §7.6]]). The pipeline is

$$
\underbrace{x}_{1\times d} \xrightarrow{\text{linear expansion}} \underbrace{h}_{1\times m} \xrightarrow{\text{nonlinear activation}} \underbrace{a}_{1\times m} \xrightarrow{\text{linear projection}} \underbrace{y}_{1\times d}
$$

and for one token

$$
\underbrace{h}_{1\times m} = \phi\big(\underbrace{x}_{1\times d}\underbrace{W_1}_{d\times m} + b_1\big), \qquad \underbrace{y}_{1\times d} = \underbrace{h}_{1\times m}\underbrace{W_2}_{m\times d} + b_2 .
$$

| Symbol | Meaning |
|---|---|
| $x$ | one token's input vector (row vector), from the attention sublayer |
| $y$ | the token's output vector |
| $d$ | model width (the residual-stream size) |
| $m$ | hidden width, usually larger than $d$ (classically $m = 4d$, e.g. 512 → 2048) |
| $W_1, b_1$ | expansion weights and bias |
| $W_2, b_2$ | projection weights and bias |
| $\phi$ | nonlinear activation |

The slide's equations write $h$ for the activated output, so the diagram's separate $a$ is just $h$ after $\phi$; the notes follow the equations.

### 1.3 Why expand, why a nonlinearity, why return to $d$

| Design choice            | Reason                                                                                                                                                                                                                                         |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Expand** $d \to m$     | More hidden units let the network learn more combinations of features.                                                                                                                                                                         |
| **Nonlinear activation** | Without it the two linear maps collapse into one: $(xW_1 + b_1)W_2 + b_2 = x(W_1W_2) + (b_1W_2 + b_2)$, a single linear map ([[Lecture_08_Activations#1.2 Affine layers collapse\|Lecture 8 §1.2]]). Expansion alone adds no expressive power. |
| **Return to** $d$        | The output must match the input's size so it can be added back on the residual connection ([[Lecture_07_Residuals_and_Normalization#1. Residual Connections\|Lecture 7 §1]]) and fed to the next layer.                                        |

**Limitation:** the FFN cannot access another token directly; it relies on the context already in its input. And within a layer, the same FFN weights process every position.

---

## 2. From Dense Models to the MoE Question

The FFN of [[#1. The Position-Wise Feed-Forward Network|§1]] is the same for every token. That is simple, but it means the only way to give the model more FFN capacity is to make that one shared network bigger, and every token then pays for it.

### 2.1 Dense models: every token uses everything

In a conventional **dense** Transformer, every token passes through the same attention and FFN in each layer.

- **Motivation:** a straightforward architecture with regular, hardware-friendly computation (large dense matrix multiplies, which GPUs do well).
- **Limitation:** making the model wider or deeper increases **both** parameter capacity **and** per-token computation. Parameters and compute are tied.

This raises the question that motivates MoE: *can we add more learned parameters without making every token use all of them?*

### 2.2 Two motivations for MoE: learn better, scale better

A **Mixture of Experts (MoE)** splits the work across several sub-networks, called **experts**, instead of one network. Its original appeal (Jacobs et al., 1991) was about learning, not saving compute: can a complex problem be solved better by several specialised networks than by one general network that has to learn everything? Keeping the two motivations apart makes the evolution Dense FFN → Dense MoE → Sparse MoE easier to follow.

1. **Learn better (specialisation and learning efficiency).** Different inputs may benefit from different transformations. Giving each expert its own parameters can encourage specialisation and reduce interference between conflicting learning signals.
2. **Scale better (capacity without proportional compute).** A **sparse MoE** holds many more parameters but activates only a small subset for each token.

*(Expanding on the terms: a **dense MoE** runs every expert on every token and mixes all their outputs, so it serves motivation 1 only; a **sparse MoE** runs only a few experts per token, which also serves motivation 2.)*

### 2.3 Gradient interference

Take two tasks, mathematics and translation. In a dense FFN both update the same parameters $\theta$. Their gradients **conflict** when

$$
\nabla_\theta L_{\text{math}}^{\top}\, \nabla_\theta L_{\text{translation}} < 0,
$$

i.e. the two gradient vectors point more than 90° apart. Then a step that lowers one loss initially raises the other. *(Why: a gradient step on math moves $\theta$ by $-\eta\nabla L_{\text{math}}$; to first order the translation loss changes by $-\eta\,\nabla L_{\text{translation}}^\top \nabla L_{\text{math}}$, which is positive when the dot product is negative.)*

> [!warning] Possible source inconsistency
> The slide glosses the condition as "one is negative while the other is positive". The gradients are vectors, so the precise statement is that their **dot product** is negative (an obtuse angle), not that one gradient is negative and the other positive.

In an MoE, suppose math mostly activates Expert 1 and translation mostly activates Expert 2. Each task now mostly updates separate parameters, which can reduce interference, and over time the separation can encourage specialisation:

$$
\text{Different experts} \longrightarrow \text{Less parameter sharing} \longrightarrow \text{Potentially less interference} \longrightarrow \text{Better specialisation}
$$

The arrows describe a possible outcome, not something the architecture guarantees.

### 2.4 The specialist intuition

In a dense layer every token, whether "photosynthesis", "democracy", "multiplication" or "beautiful", goes through the same FFN $y = F(x;\theta)$. That does not mean the same transformation: the nonlinearities already make the output input-dependent. But all those different transformations have to share one set of parameters.

The MoE intuition: instead of training one person to be an expert in mathematics, biology, linguistics and economics, train a specialist in each, and when a problem arrives, consult the right specialist or specialists. The analogy captures the motivation, not a guarantee. Experts are not assigned human-readable subjects; whatever they specialise in is learned from data and can be far subtler than disciplines or vocabulary categories ([[#4.4 What does an expert learn?|§4.4]] shows what they actually learn).

---

## 3. A Worked Example: The Absolute Value with Two Experts

To see the MoE idea without language-model detail, take a regression target made of two rules:

$$
y = \begin{cases} x, & x \ge 0 \\ -x, & x < 0 \end{cases} \quad\text{i.e.}\quad y = |x| .
$$

### 3.1 Approach A: one dense network

A purely linear model $\hat y = ax + b$ cannot fit both rules: the positive side needs slope $+1$ and the negative side slope $-1$, and one affine map has one slope. Adding hidden neurons and a nonlinearity fixes this; a ReLU network represents it exactly:

$$
|x| = \text{ReLU}(x) + \text{ReLU}(-x).
$$

(For $x \ge 0$ the second term is 0 and the first is $x$; for $x < 0$ the first is 0 and the second is $-x$.) So nothing here is impossible for a dense FFN. The question is whether there is a different way to organise the computation.

### 3.2 Approach B: two experts and a router

Define two **experts**, each a simple function,

$$
E_1(x) = x, \qquad E_2(x) = -x,
$$

and a **router** that outputs a weight for each:

$$
p_1(x) = \begin{cases} 1 & x \ge 0 \\ 0 & x < 0 \end{cases}, \qquad p_2(x) = 1 - p_1(x).
$$

The output is the router-weighted sum of expert outputs:

$$
\boxed{y = p_1(x)E_1(x) + p_2(x)E_2(x)}
$$

For positive inputs the router selects Expert 1; for negative inputs, Expert 2. Each expert learns a simple transformation and the router learns which one applies. That is the essence of MoE. The point is not that a dense network cannot learn $|x|$ (it can); MoE **explicitly decomposes** the problem into simpler component functions plus a learned selection mechanism.

### 3.3 Observations

- **A dense network is not less expressive.** A large enough dense network can learn the full mixture too. MoE's advantage is structural: <mark style="background: #FFF3A3A6;">each expert only handles one source, the sources don't interfere during training, and only one expert runs per input.</mark>
- **Nobody tells the model which source an input came from.** There is no separate model trained per distribution and combined afterwards.<mark style="background: #FFF3A3A6;"> The gate and experts are trained together, and the gate learns to infer the source from the input's features.</mark> (In the toy example the router uses the sign of $x$.)
- **The gain comes from features that identify the source.** If a token's features say nothing about where it came from, routing cannot help.

---

## 4. Mixture of Experts in LLMs

The toy example had a hand-written router and two one-line experts. In an LLM both are learned networks, and the MoE layer slots in where the FFN used to be. This section uses Maarten Grootendorst's *A Visual Guide to Mixture of Experts*.

### 4.1 Experts and the router

An MoE layer has two parts:

- **Experts:** each FFN layer now has a set of experts, of which a subset is chosen per token. Experts are typically FFNs themselves.
- **Router** (or **gate network**): decides which tokens go to which experts.

![[L09_router_mechanism.png|260]]

*The router is a small FFNN followed by a softmax. Its probabilities here are 0.45, 0.19, 0.05 and 0.31; FFNN 1 is activated and its output is multiplied by $p = 0.45$.*

Each layer of an MoE LLM has its own set of (somewhat specialised) experts. An expert is not a whole LLM, just a sub-model inside one layer of the architecture.

### 4.2 Dense layers, the thing MoE replaces

In a standard decoder-only Transformer the FFN sits after a LayerNorm in each decoder block (pre-norm, [[Lecture_07_Residuals_and_Normalization#3.8 Pre-norm|Lecture 7 §3.8]]). It lets the model use the contextual information attention created and transform it further to capture more complex relationships. To do this it expands its input, e.g. 512 → 2048 (×4) → 512:

![[L09_dense_ffnn_expansion.png|520]]

*Inside a pre-norm decoder block, the dense FFNN expands 512 values to 2048 (×4) and projects back to 512.*

### 4.3 Sparse layers and sparse models

The traditional FFN is called **dense** because all its parameters (weights and biases) are used for every input; nothing is left out. **Sparse** models activate only a portion of their parameters per input, and MoE is the main way of doing this. Conceptually, chop the dense hidden layer into pieces (experts), retrain, and activate only a subset per token:

| Dense                         | Sparse                         |
| ----------------------------- | ------------------------------ |
| ![[L09_dense_model.png\|260]] | ![[L09_sparse_model.png\|260]] |

*Left: in the dense model every hidden unit is used for every input. Right: the hidden layer is split into four experts and only Expert 4 is activated for this input; Experts 1–3 are not.*

The idea is that each expert learns different information during training, and at inference only the experts most relevant to the input run. For "What is 1 + 1?" the router might send the token to a "numbers" expert and leave the others idle.

### 4.4 What does an expert learn?

Experts learn much more fine-grained things than whole domains, which is why the name "expert" is sometimes called misleading. An expert is not a "Psychology" or "Biology" expert; at most it picks up word-level, syntactic patterns:

![[L09_experts_syntax.png|520]]

*Layer-1 experts in an encoder pick up word-level categories: punctuation, verbs, conjunctions and visual descriptions.*

More precisely, an expert specialises in handling **specific tokens in specific contexts**.

**Encoder models (Switch Transformer).** The Switch Transformer paper (Fedus et al., 2022, an MoE that routes each token to one expert) found <mark style="background: #FFF3A3A6;">encoder experts specialising in punctuation, conjunctions and articles, verbs, visual descriptions (colour, spatial position), and counting and numbers</mark>:

![[L09_expert_specialization_switch.png|520]]

*Switch Transformer encoder experts and the tokens routed to them: punctuation (layers 2 and 6), conjunctions and articles (layers 3 and 6), verbs (layer 1), visual descriptions (layer 0), and counting and numbers (layer 1).*

**Decoder models (Mixtral 8x7B).** Decoder experts do not show this kind of specialisation, but they are not interchangeable either. In the Mixtral paper (Jiang et al., 2024; 8 experts per layer, 2 used per token), colouring each token by its first-choice expert shows<mark style="background: #FFF3A3A6;"> experts used consistently for certain kinds of tokens, tracking syntax</mark> (e.g. indentation, `self`, keywords in code) more than domain:

![[L09_mixtral_expert_colouring.png|360]]

*Mixtral code with each token coloured by its first-choice expert: the same expert keeps handling syntax such as indentation, `self` and keywords.*

### 4.5 The architecture of experts and routes through layers

It is convenient to picture experts as slices of one dense hidden layer, but in practice each expert is usually a whole FFN. Since an LLM has many decoder blocks, a token passes through several experts, one set per layer, before the output is produced:

![[L09_expert_architecture.png|560]]

*Left: a sparse layer with four experts, each a small FFNN; for "What is 1 + 1?" only the "Numbers" expert is activated. Right: across 12 decoder blocks the token is routed to a different expert in each layer.*

Different tokens choose different experts, so they take different **paths** through the network. The decoder block itself changes only in that the single FFN is replaced by several FFNs, one per expert:

![[L09_moe_routes.png|600]]

*Left: across three passes, different tokens take different paths through the experts of each layer. Right: the dense decoder block and its sparse version, where the single FFNN is replaced by FFNN 1–4.*

---

## 5. Routing

The pictures in [[#4. Mixture of Experts in LLMs|§4]] assumed a router that sends each token to the right expert. This section states the per-token procedure and how the router chooses.

### 5.1 The sparse MoE recipe per token

A typical sparse MoE Transformer replaces some FFNs with a collection of FFNs (experts) plus a small learned router. For each token, at each MoE layer:

1. The router scores the experts using that token's current representation.
2. It selects a few experts, often one or two.
3. Only those experts process the token.
4. Their outputs are combined using routing weights.

### 5.2 Combining expert outputs

With two experts selected, e.g. experts 2 and 5,

$$
y = 0.7\,E_2(x) + 0.3\,E_5(x),
$$

where $E_2(x)$ and $E_5(x)$ are the expert FFN outputs and 0.7, 0.3 are illustrative routing weights. This is the [[#3.2 Approach B: two experts and a router|§3.2]] formula with most weights set to zero.

- **Motivation:** increase parameter capacity while keeping per-token computation modest.
- **The model can hold much more processing capacity than any one token uses.** With $N$ experts and $k$ chosen per token, expert parameters grow with $N$ but per-token expert compute grows only with $k$.
- **"Expert" does not guarantee a human-readable specialty.** Nobody assigns "math" to one expert and "Urdu" to another; behaviour emerges in training.

### 5.3 The routing mechanism: a softmax gate

The **router** sits just before the experts and is trained to choose which expert(s) handle each token. It is itself a small FFN (often a single linear layer, *expanding on the slide*): it maps the token vector $x$ to one score per expert and applies a softmax to get a probability for each expert,

$$
p(x) = \text{softmax}(xW_r), \qquad W_r \in \mathbb{R}^{d\times N}.
$$

It selects the best-matching expert(s), and the layer returns the selected expert's output **multiplied by its gate value**. In the slide's figure the probabilities are 0.45, 0.19, 0.05, 0.31; FFNN 1 is selected and the output is $0.45 \cdot E_1(x)$. Multiplying by the gate value is what lets the router learn: the output depends on $p$, so gradients flow back into $W_r$ (with a hard choice and no multiplier, the router would get no gradient; [[#7.8 Back to renormalization|§7.8]] works this out).

### 5.4 The MoE layer: dense MoE vs sparse MoE

The router together with the experts (of which only a few are selected) makes up the **MoE layer**, which takes the FFN's place inside the decoder block:

![[L09_moe_layer_in_decoder.png|560]]

*The decoder block keeps its LayerNorm, masked self-attention and residual adds; only the FFN slot becomes an MoE layer. Inside it the router's bar chart picks FFNN 1, whose output is multiplied (⊗) by its gate value.*

An MoE layer comes in two sizes. Both use a router, but a **sparse MoE** selects only a few experts, whereas a **dense MoE** selects them all, potentially with different weights (different "distributions" over the experts). Given a set of tokens, a dense MoE distributes every token across all experts; a sparse MoE sends each token to only a few.

![[L09_dense_vs_sparse_moe.png|480]]

*Dense MoE: all three FFNNs run and every output is weighted and summed. Sparse MoE: FFNN 2 is greyed out, so only FFNN 1 and FFNN 3 run for this token.*

In current LLMs, "MoE" almost always means a sparse MoE, because running only a subset of experts is computationally cheaper, which is an important trait at LLM scale.

### 5.5 How the router works: score, select, process, combine

Let $N$ be the number of experts and $k$ the number selected per token. A simple router first computes one score per expert,

$$
\underbrace{s}_{1\times N} = \underbrace{x}_{1\times d}\,\underbrace{W_r}_{d\times N},
$$

and a softmax converts these scores into positive weights that sum to 1 (the same softmax as in attention, [[Lecture_05_Attention_and_Multi_Head_Attention#7.5 Step 4: retrieve, and the full formula|Lecture 5 §7.5]]). The router then selects the top $k$ experts. For an illustrative four-expert layer with $k = 2$:

| Expert | Router probability | Selected with $k = 2$? |
|---|---|---|
| 1 | 0.10 | No |
| 2 | 0.60 | Yes |
| 3 | 0.20 | Yes |
| 4 | 0.10 | No |

If we **renormalize** the selected weights (divide each by the sum of the selected probabilities, so they sum to 1 again),

$$
w_2 = \frac{0.60}{0.80} = 0.75, \qquad w_3 = \frac{0.20}{0.80} = 0.25, \qquad y = 0.75\,E_2(x) + 0.25\,E_3(x).
$$

Each expert output has shape $1\times d$ (every expert is an FFN that returns to the model width, [[#1.3 Why expand, why a nonlinearity, why return to $d$|§1.3]]), so the outputs can be added.

### 5.6 Pre-routing: the router decides from token features, not expert results

The router scores experts **before** they execute. It predicts which ones will be useful from the token representation alone; it does not run every expert and then pick the best result (which would cost as much as a dense MoE). A tiny numerical example with the weights of [[#5.5 How the router works: score, select, process, combine|§5.5]]: if $E_2(x) = [4, 0]$ and $E_3(x) = [2, 6]$, then

$$
y = 0.75\,[4, 0] + 0.25\,[2, 6] = [3 + 0.5,\ 0 + 1.5] = [3.5,\ 1.5].
$$

This is a combination of **hidden representations**, not a vote between complete textual answers: the experts are FFNs inside one layer, so $y$ is just the next residual-stream update, which later layers keep transforming.

### 5.7 A first note on renormalization

The slides flag three nuances here and return to them after the training section ([[#7.6 The overall picture, and why load balancing is needed|§7.6]]):

1. The **Switch Transformer** (top-1) keeps the selected expert's original softmax probability, which lets task-loss gradients reach the router. It does not renormalize.
2. Top-$k$ with $k \ge 2$ often renormalizes the selected probabilities, but this is not mandatory.
3. Top-1 can use a weight of 1, but then ordinary task-loss backpropagation cannot train the router through that weight; another learning mechanism is needed if the router is trainable.

---

## 6. Total vs Active Parameters

Routing decides which experts a token uses. The payoff, and the costs, show up when we count what is stored against what is computed.

### 6.1 Dense models revisited: the same machinery for every token

In a dense Transformer every token uses the same attention and FFN modules at each layer ([[#2.1 Dense models: every token uses everything|§2.1]]). This does **not** mean their activations are identical: different inputs activate different features, even though the same matrix multiplications execute. The advantages are substantial:

- Regular computation batches efficiently on GPUs.
- Every token provides a learning opportunity to the same FFN.
- Training and deployment avoid the complexity of routing tokens between experts.

The limitation appears when we want more capacity. For the conventional FFN, ignoring biases,

$$
\text{FFN parameters} = dm + md = 2dm,
$$

and the per-token compute is also proportional to $dm$ (about $2 \cdot 2dm$ FLOPs, Problem 1). Increasing $m$ therefore increases both parameters and computation per token. Dense scaling couples **how much the model can store in its parameters** with **how much computation each token must perform**. Better activations, quantization and efficient kernels help, but they do not fundamentally remove this coupling.

### 6.2 Stored vs used parameters

Imagine an MoE layer with eight equally sized experts, selecting two per token:

| Quantity | What happens |
|---|---|
| Stored expert parameters | All eight experts |
| Expert computation per token | Two experts |
| Other work | Router, attention and remaining shared components |

This uses roughly one-quarter of the expert computation of running all eight. In general, if each expert has $P$ parameters,

$$
\text{total expert parameters} = NP, \qquad \text{expert parameters used per token} = kP .
$$

For eight experts of 10 million parameters each and top-2 routing, the layer stores 80 million expert parameters and uses 20 million per token. That gives two different comparisons:

| Comparison | What the MoE offers |
|---|---|
| Against one 10-million-parameter dense FFN | Eight times the expert parameters, but roughly twice the expert computation |
| Against a dense 80-million-parameter FFN | Similar expert parameter count, but roughly one-quarter of its matrix computation |

These comparisons exclude routing and shared components, and do not imply equal quality.

The main limitations follow directly:

- **Memory:** all experts still need storage and must be available (loaded on some device), even though each token uses only two.
- **Uneven routing:** a few experts may receive most tokens; balancing mechanisms ([[#8. Expert Collapse and Load Balancing|§8]]–[[#9. Other Load-Balancing Methods and Expert Capacity|§9]]) encourage broader use.
- **Communication:** in large models experts live on different GPUs, and moving tokens to them and back can be expensive.
- **Training complexity:** routing and expert learning must stabilise together ([[#7. Training an MoE|§7]]–[[#8. Expert Collapse and Load Balancing|§8]]).

Consequently, **fewer active parameters do not automatically translate into proportionally lower latency.**

### 6.3 Efficiency in big-O terms

For $T$ tokens, $N$ experts, $k$ chosen experts, model width $d$ and conventional experts of hidden width $m$:

$$
\text{Expert computation} \sim O(Tkdm), \qquad \text{Router computation} \sim O(TdN).
$$

*(Why: each token runs $k$ FFNs of cost $\propto dm$; the router is one $d\times N$ multiply per token.)* Increasing $N$ while holding $k$ fixed grows expert capacity without proportionally increasing expert computation. Routing, storage and communication still grow with $N$. And sparse FFNs do nothing about attention's sequence-length costs ([[Lecture_06_Decoding_KV_Cache_MQA_GQA_FlashAttention#3. Time Complexity of Attention|Lecture 6 §3]]) or its KV-cache requirements ([[Lecture_06_Decoding_KV_Cache_MQA_GQA_FlashAttention#4. KV Cache|Lecture 6 §4]]); MoE only changes the FFN sublayer.

### 6.4 Dense FFN layer vs MoE FFN layer, side by side

![[L09_dense_ffn_vs_moe_ffn_layer.jpg|720]]

*Left: one shared FFN (up-projection $h = xW_1 + b_1$, activation $\varphi$, down-projection $y = hW_2 + b_2$) processes every token, so all computation sits in one FFN and every token uses the same parameters. Right: the router computes $p = \text{softmax}(xW_r + b_r)$ over four experts; only Experts 2 and 3 (top-$k$) run, and their outputs are combined as $y = \sum_{i\in\text{TopK}(x)} w_i E_i(x)$. Computation is spread across experts, each token uses a subset of expert parameters, and the layer has sparse activation with larger total capacity.*

---

## 7. Training an MoE

[[#6. Total vs Active Parameters|§6]] counted the cost of a forward pass. Training raises a different question: the layer contains a discrete "pick the top $k$" step, so how do gradients reach the experts and the router?

### 7.1 Two things are learned

Two different things are being learned in an MoE layer:

1. **Experts learn what to compute.**
2. **The router learns which experts to trust, and by how much.**

Suppose for one token the router chooses Experts 2 and 3, and the MoE output is

$$
y = w_2 e_2 + w_3 e_3,
$$

where $e_2, e_3$ are the outputs produced by Experts 2 and 3, $w_2, w_3$ are the router-generated weights, $y$ is the final MoE output and $L$ is the eventual training loss. The computational graph has two branches that meet at $y$:

$$
x \to \begin{cases} \text{Expert 2} \to e_2 \\ \text{Expert 3} \to e_3 \end{cases}, \qquad x \to \text{Router} \to (w_2, w_3), \qquad (e_2, e_3, w_2, w_3) \to y = w_2e_2 + w_3e_3 \to L .
$$

So the error at $L$ flows backward along **both** branches.

### 7.2 The gradient into an expert

By the chain rule through $y$, and since $\partial y/\partial e_2 = w_2$ (the other term does not contain $e_2$),

$$
\frac{\partial L}{\partial e_2} = \frac{\partial L}{\partial y}\frac{\partial y}{\partial e_2} = \boxed{w_2\,\frac{\partial L}{\partial y}} .
$$

The gradient sent into Expert 2 is **scaled by how much the router used Expert 2's output**. With $w_2 = 0.8$, Expert 2 receives a relatively strong learning signal; with $w_2 = 0.05$, a much weaker one. The expert's output comes from its own parameters, $e_2 = E_2(x; \theta_2)$, so one more chain-rule step gives the update for those parameters:

$$
\frac{\partial L}{\partial \theta_2} = \frac{\partial L}{\partial e_2}\frac{\partial e_2}{\partial \theta_2} = w_2\,\frac{\partial L}{\partial y}\,\frac{\partial e_2}{\partial \theta_2}.
$$

This is how Expert 2 learns to change what it produces. *(In the vector case $e_2, y \in \mathbb{R}^{1\times d}$ and $\partial e_2/\partial\theta_2$ is a Jacobian, but the scaling by the scalar $w_2$ is unchanged.)*

### 7.3 The gradient into the router weight, and the two gradients

Differentiating $y$ with respect to the weight instead, $\partial y/\partial w_2 = e_2$, so

$$
\frac{\partial L}{\partial w_2} = \frac{\partial L}{\partial y}\frac{\partial y}{\partial w_2} = \boxed{e_2\,\frac{\partial L}{\partial y}}
$$

*(for vectors this is the dot product $\frac{\partial L}{\partial y}\cdot e_2$, a scalar, because $w_2$ is a scalar)*. The loss tells us whether increasing or decreasing Expert 2's contribution would have helped. The two gradients answer different questions:

| Gradient | Question it answers |
|---|---|
| $\dfrac{\partial L}{\partial e_2} = w_2\dfrac{\partial L}{\partial y}$ | "How should Expert 2 change its answer?" |
| $\dfrac{\partial L}{\partial w_2} = e_2\dfrac{\partial L}{\partial y}$ | "Should I have relied more or less on Expert 2's answer?" |

Each gradient is scaled by the *other* factor of the product $w_2e_2$: the expert's signal by how much it was used, the weight's signal by how large the expert's output was.

### 7.4 Subtle point: $w_2$ is not a learned parameter

$w_2$ and $w_3$ are not stored parameters; they are computed from the token. The router computes logits $r = W_r x$, i.e. $r_1, r_2, \dots, r_N$ *(the slide writes $W_rx$ with column vectors; this is the same as $xW_r$ in [[#5.3 The routing mechanism: a softmax gate|§5.3]] with the matrix transposed)*, and for the two selected experts

$$
w_2 = \frac{e^{r_2}}{e^{r_2} + e^{r_3}}, \qquad w_3 = \frac{e^{r_3}}{e^{r_2} + e^{r_3}} .
$$

*(This is exactly the renormalized top-2 of [[#5.5 How the router works: score, select, process, combine|§5.5]]: dividing the full-softmax probabilities $e^{r_i}/\sum_j e^{r_j}$ of the selected experts by their sum cancels the full denominator and leaves a softmax over the selected logits only.)* The real gradient path is therefore

$$
L \to y \to (w_2, w_3) \to (r_2, r_3) \to W_r, \qquad \frac{\partial L}{\partial W_r} = \frac{\partial L}{\partial y}\,\frac{\partial y}{\partial w}\,\frac{\partial w}{\partial r}\,\frac{\partial r}{\partial W_r}.
$$

The $w_i$ are ordinary differentiable real numbers such as 0.7 and 0.3, and $r_2 \uparrow \Rightarrow w_2 \uparrow$. So a negative $\partial L/\partial w_2$ becomes an update to $W_r$ that says: "next time you see something like this token, give Expert 2 a little more probability."

### 7.5 The top-$k$ decision is not differentiable

Suppose the router initially produces $(0.05, 0.45, 0.40, 0.10)$, so $E_2$ and $E_3$ are selected. The operation

$$
\text{Top-2}\big((0.05, 0.45, 0.40, 0.10)\big) = \{2, 3\}
$$

is **discrete**. A tiny change such as $0.45 \to 0.451$ does not continuously change "Expert 2 is selected"; it stays selected, so the derivative of the selection with respect to the score is zero. And when two values cross, $0.3999 \to 0.4001$, an expert may suddenly enter or leave the top-$k$, a jump. So ordinary differentiation through the *identity* of the selected experts is problematic. But once Experts 2 and 3 have been chosen, the output

$$
y = w_2E_2(x) + w_3E_3(x)
$$

is an ordinary differentiable function and gradients flow through it. Training therefore **differentiates along the chosen computation path**, treating the selection as fixed.

**Worked example.** Let $e_2 = 4$, $e_3 = 1$ (scalars for simplicity) and $w_2 = 0.6$, $w_3 = 0.4$. Then $y = 0.6(4) + 0.4(1) = 2.8$. Suppose the backward pass gives $\partial L/\partial y = -2$ (the loss would fall if $y$ were larger). For the experts,

$$
\frac{\partial L}{\partial e_2} = 0.6(-2) = -1.2, \qquad \frac{\partial L}{\partial e_3} = 0.4(-2) = -0.8,
$$

so both experts receive gradients, with Expert 2 receiving the stronger one because it contributed more. For the weights,

$$
\frac{\partial L}{\partial w_2} = e_2(-2) = -8, \qquad \frac{\partial L}{\partial w_3} = e_3(-2) = -2 .
$$

The loss is telling the routing mechanism that changing $w_2$ has a much larger effect on the result, because Expert 2's output is 4 whereas Expert 3's is only 1. These gradients then propagate through the softmax into the router logits and ultimately the router weights $W_r$. *(Because $w_2 + w_3 = 1$, the softmax turns these into a push on $r_2$ relative to $r_3$: $\partial L/\partial r_2 = w_2w_3(\partial L/\partial w_2 - \partial L/\partial w_3) = 0.24(-6) = -1.44$, so gradient descent raises $r_2$ and Expert 2's share grows.)*

### 7.6 The overall picture, and why load balancing is needed

| Path | Question |
|---|---|
| Loss → expert output → expert parameters | "What should this expert compute?" |
| Loss → mixture weight → router logits → router parameters | "How much should I use this expert?" |

The one thing ordinary backprop does not directly learn through is the **top-$k$ selection**, because it is discrete. Two consequences:

- An expert that is not selected for a token normally gets **no main-task gradient** from that token.
- Real MoEs therefore also commonly use **auxiliary routing / load-balancing losses**, which are important for preventing the router from repeatedly sending everything to the same few experts ([[#8. Expert Collapse and Load Balancing|§8]]).

### 7.7 Dense MoE: fully differentiable routing

In a dense MoE the whole routing operation becomes differentiable, because there is no discrete top-$k$ selection: every expert runs and every softmax weight is used.

| Property | Dense MoE | Sparse MoE |
|---|---|---|
| Experts evaluated | All $N$ | Only $k$ |
| Routing | Continuous softmax | Discrete top-$k$ + continuous weights |
| Differentiability | End-to-end | Along selected paths |
| Expert computation per token | $O(N)$ | $O(k)$ |
| Primary advantage | Smooth learning | Computational efficiency |

Sparse MoE deliberately introduces a discrete selection step to reduce computation. It sacrifices differentiability through expert selection while keeping gradient-based learning for the selected experts and their routing weights.

### 7.8 Back to renormalization

With the two gradients of [[#7.3 The gradient into the router weight, and the two gradients|§7.3]] in hand, the three nuances of [[#5.7 A first note on renormalization|§5.7]] can be checked. Take router logits $r = W_r x$ with probabilities $p = (0.1, 0.6, 0.2, 0.1)$ and top-1 routing, so Expert 2 is selected.

**Before renormalization (Switch Transformer).** The output keeps the softmax probability, $y = p_2E_2(x) = 0.6\,e_2$. Then

$$
\frac{\partial L}{\partial e_2} = 0.6\,\frac{\partial L}{\partial y}, \qquad \frac{\partial L}{\partial p_2} = e_2\,\frac{\partial L}{\partial y}.
$$

The expert learns, with a gradient scaled by 0.6, and since $p_2 = \text{softmax}(W_rx)_2$, the second gradient propagates back to $W_r$. Both the expert and the router learn.

**After top-1 renormalization.** The single selected weight becomes $w_2 = p_2/p_2 = 1$, so $y = 1\cdot e_2 = e_2$. The expert still receives its gradient, and it is no longer scaled down by 0.6:

$$
\frac{\partial L}{\partial e_2} = \frac{\partial L}{\partial y}\underbrace{\frac{\partial y}{\partial e_2}}_{1} = \frac{\partial L}{\partial y}.
$$

The router path is where things change. By the chain rule through $w_2$,

$$
\frac{\partial L}{\partial p_2} = \frac{\partial L}{\partial y}\,\frac{\partial y}{\partial w_2}\,\frac{\partial w_2}{\partial p_2}, \qquad \frac{\partial y}{\partial w_2} = e_2, \qquad \frac{\partial w_2}{\partial p_2} = \frac{\partial}{\partial p_2}\Big(\frac{p_2}{p_2}\Big) = 0,
$$

so $\partial L/\partial p_2 = \frac{\partial L}{\partial y}\cdot e_2 \cdot 0 = 0$. **No task-loss gradient reaches $W_r$ through this path**: the weight is the constant 1 whatever the router outputs. With $k \ge 2$ renormalization is harmless for the router, because each renormalized weight $p_i/\sum_{j\in\text{sel}}p_j$ still depends on the *relative* sizes of the selected probabilities ([[#7.4 Subtle point: $w_2$ is not a learned parameter|§7.4]]).

**What actual models do:**

| Architecture | Routing | Weight treatment |
|---|---|---|
| Switch Transformer | Top-1 | Original softmax probability |
| Mixtral 8×7B | Top-2 | Renormalizes selected weights |
| DeepSeek-V2 | Top-$k$ | Supports non-renormalized weights |
| DeepSeek-V3 | Top-8 | Normalizes selected sigmoid scores and applies a scaling factor |

*(DeepSeek-V3 scores experts with a sigmoid per expert instead of one softmax over all experts, so its selected scores must be normalized to sum to 1; the extra scaling factor rescales the result.)*

**Top-1 with a unit weight is still possible.** A hard top-1 architecture can use $y = E_s(x)$, where $s$ is the selected expert; this is mathematically equivalent to renormalizing the single selected probability to 1. The router then has to learn some other way, for example:

- **Method A: auxiliary routing losses** (an extra loss on the router's probabilities, like the balancing loss of [[#8. Expert Collapse and Load Balancing|§8]]);
- **Method B: the straight-through estimator (STE)**, which uses the hard choice in the forward pass but, in the backward pass, pretends the choice was the soft probability so a gradient can pass *(expanding on the slide's name only)*.

The course does not cover these further.

---

## 8. Expert Collapse and Load Balancing

[[#7.6 The overall picture, and why load balancing is needed|§7.6]] left a gap: experts that are never selected never get a task gradient. Left alone, that gap turns into a feedback loop. (The section divider's slogan, "We work on our work, and our work works on us!", describes it: what an expert is trained on shapes what it becomes good at, which shapes what it is sent next.)

### 8.1 Why some experts starve: a self-reinforcing loop

Experts generally begin without predefined specialties, and small initial routing differences can grow:

1. An expert receives slightly more tokens.
2. It gets more training.
3. It becomes more useful for those inputs.
4. The router sends it still more tokens.
5. The other experts become increasingly underutilised.

Meanwhile, neglected experts receive little practice and become less competitive: a vicious cycle. This is **routing collapse** or **expert collapse**: the model contains many experts but uses only a few. Early sparse-MoE work (Shazeer et al., 2017) explicitly identified this feedback loop. Consider 100 tokens routed top-1 across four experts:

| Expert | Tokens received | Consequence |
|---|---|---|
| A | 85 | Potential bottleneck |
| B | 10 | Underused |
| C | 5 | Very little training |
| D | 0 | No task learning from this batch |

There are two separate costs: **wasted learning capacity** (parameters stored but untrained) and **uneven hardware workload** (the GPU holding A is overloaded while others idle). And training only for immediate prediction quality can undermine the development of experts that could become useful later.

### 8.2 The goal and the balancing objective

We want the router to send each token to the most suitable expert, but not to become so dependent on a few experts that the others hardly get trained. A load-balancing loss adds a second learning objective next to the usual one:

- **Task loss:** are we producing the correct answer?
- **Balancing loss:** are we giving all the experts reasonable opportunities to contribute and learn?

$$
\boxed{L_{\text{total}} = L_{\text{task}} + \lambda L_{\text{balance}}}
$$

$L_{\text{task}}$ encourages correct predictions, $L_{\text{balance}}$ encourages broader expert utilisation, and $\lambda$ sets how strongly we care about balance relative to task performance. When $\lambda = 0$ the router gets no encouragement from this term to distribute work; when $\lambda$ is too large it may prioritise balancing over selecting the most appropriate expert. The aim is not to make every expert equally useful for every token; it is to avoid systematically starving experts across a collection of tokens.

### 8.3 The two quantities $f_i$ and $\bar p_i$

The Switch Transformer introduces

$$
\boxed{L_{\text{balance}} = N\sum_{i=1}^{N} f_i\,\bar p_i}
$$

where $N$ is the number of experts. The key is that $f_i$ and $\bar p_i$ measure two different things. Simplify to two experts and ten tokens:

| | Expert 1 | Expert 2 |
|---|---|---|
| Tokens actually assigned | 8 | 2 |
| $f_i$: assignment fraction | 0.8 | 0.2 |
| $\bar p_i$: average router probability | 0.7 | 0.3 |

- **$f_i$** is the fraction of tokens actually sent to Expert $i$ by top-1 routing, e.g. $f_1 = 8/10 = 0.8$. It is a **discrete** quantity: each token either was assigned to Expert 1 or wasn't.
- **$\bar p_i$** is the average softmax probability the router assigned to Expert $i$ across all ten tokens, computed **before** the top-1 selection. Expert 1 might receive probabilities 0.9, 0.8, 0.7, … on individual tokens, averaging to 0.7. Even tokens ultimately assigned to Expert 2 contribute their Expert-1 probabilities to this average. Unlike $f_i$, $\bar p_i$ is **differentiable**.

### 8.4 Why multiplying them encourages balance

For the example,

$$
L_{\text{balance}} = 2(f_1\bar p_1 + f_2\bar p_2) = 2[(0.8)(0.7) + (0.2)(0.3)] = 2(0.56 + 0.06) = 1.24 .
$$

With perfectly balanced routing, $f_1 = f_2 = 0.5$ and $\bar p_1 = \bar p_2 = 0.5$, so $L_{\text{balance}} = 2[(0.5)(0.5) + (0.5)(0.5)] = 1$. In this example balanced routing gives the smaller loss. The intuition: the product is large when the expert that already gets most tokens (large $f_i$) is also the one the router is most confident about (large $\bar p_i$), i.e. when assignment and confidence pile up on the same expert.

### 8.5 We cannot change $f_i$ by gradients, but we can change $\bar p_i$

Treating the assignment fractions $f_i$ as fixed during backpropagation (they come from the discrete top-1 choice),

$$
\frac{\partial L_{\text{balance}}}{\partial \bar p_i} = N f_i, \qquad \frac{\partial L_{\text{balance}}}{\partial \bar p_1} = 2(0.8) = 1.6, \qquad \frac{\partial L_{\text{balance}}}{\partial \bar p_2} = 2(0.2) = 0.4 .
$$

A larger $f_i$ creates a larger penalty for increasing that expert's average routing probability. Because the probabilities must sum to 1, shifting probability from the overused expert to the underused one reduces the loss. Suppose a gradient update moves the average probabilities from $(0.7, 0.3)$ to $(0.65, 0.35)$ while the hard assignments stay the same:

$$
L_{\text{balance}} = 2[(0.8)(0.65) + (0.2)(0.35)] = 2(0.52 + 0.07) = 1.18,
$$

down from 1.24.

The gradient flows along this path; all of these operations are differentiable, and the discrete $f_i$ are held fixed during the backward pass:

![[L09_balance_gradient_flow.png|420]]

*Balancing loss → average probabilities $\bar p_i$ → softmax probabilities $p_i(x)$ → router logits $r_i$ → router parameters $W_r$, where $r = W_rx$, $p_i(x) = e^{r_i}/\sum_j e^{r_j}$ and $\bar p_i = \frac{1}{T}\sum_{x\in B} p_i(x)$ over the $T$ tokens of batch $B$.*

### 8.6 How changing probabilities eventually changes assignments

We change probabilities by updating $W_r$. The balancing loss sends gradients back along $L_{\text{balance}} \to p \to r \to W_r$, and gradient descent uses the combined task and balancing gradients:

$$
W_r^{\text{new}} = W_r^{\text{old}} - \eta\,\frac{\partial L_{\text{total}}}{\partial W_r}.
$$

As $W_r$ changes, the router produces different probabilities for the tokens it sees: a token previously given $[0.8, 0.2]$ might, after updates, receive $[0.6, 0.4]$. Eventually some tokens switch experts as their probabilities cross. A token at $[0.51, 0.49]$ goes to Expert 1; after successive updates the router might produce $[0.49, 0.51]$, and now it goes to Expert 2. The gradient never passed through the discrete selection; it gradually changed the continuous router probabilities until the top-1 decision flipped. That is how balancing is achieved despite non-differentiable expert selection.

### 8.7 Why multiply by $N$?

Without the factor $N$, perfectly balanced routing across $N$ experts ($f_i = \bar p_i = 1/N$) gives

$$
\sum_{i=1}^{N}\frac{1}{N}\cdot\frac{1}{N} = \underbrace{\frac{1}{N^2} + \cdots + \frac{1}{N^2}}_{N\text{ terms}} = \frac{N}{N^2} = \frac{1}{N},
$$

which shrinks as experts are added. Multiplying by $N$ makes the loss at perfectly uniform routing equal to $N \cdot \frac{1}{N} = 1$ regardless of the number of experts, which keeps the auxiliary loss on a comparable scale across expert counts (so the same $\lambda$ means roughly the same thing).

A technical qualification: this is a heuristic balancing objective, not a perfect mathematical measure of imbalance. Uniform routing gives 1, but some non-uniform combinations of assignment fractions and probabilities can also give values below 1. *(Example: $N = 2$, $f = (0.6, 0.4)$, $\bar p = (0.4, 0.6)$ gives $2(0.24 + 0.24) = 0.96$. This can happen when the tokens sent to Expert 1 win it only narrowly while the others strongly favour Expert 2.)*

### 8.8 Does balanced usage mean useful specialisation?

No. Even if all four experts receive exactly 25% of the tokens, they could still be learning essentially identical functions. Load balancing only encourages equitable usage; it does not directly require experts to become different from one another. Specialisation, when it emerges, comes from the interaction of routing, expert training and the task objective.

### 8.9 When is the balancing loss computed?

**Not after every token.** Consider a training batch of 100 tokens:

1. The router processes all 100 tokens using its current $W_r$.
2. For each token it computes expert probabilities and selects an expert.
3. We compute $f_i$ and $\bar p_i$ over those 100 tokens.
4. We compute $L_{\text{balance}}$ and combine it with the task loss.
5. We run backpropagation and update the parameters.

So one training batch produces one balancing-loss computation per MoE layer (or per routing group, depending on the implementation). **Nor are $f_i$ and $\bar p_i$ running statistics** across training history, in the standard Switch Transformer formulation; they are computed afresh for each batch:

| Training step | Tokens in batch | Assignments to Expert 1 | $f_1$ |
|---|---|---|---|
| 1 | 100 | 80 | 0.80 |
| 2 | 100 | 65 | 0.65 |
| 3 | 100 | 53 | 0.53 |
| 4 | 100 | 49 | 0.49 |

At step 2, $f_1$ is not $(80 + 65)/200$; it uses only the 100 tokens of batch 2. Likewise $\bar p_i$ is the average router probability over the current batch. Each batch gives a fresh balancing signal, and over many updates the router can learn to distribute tokens more evenly (as the falling $f_1$ shows).

---

## 9. Other Load-Balancing Methods and Expert Capacity

The auxiliary loss of [[#8. Expert Collapse and Load Balancing|§8]] balances through gradients. Two other families act on the routing scores directly, and a capacity limit handles whatever imbalance remains.

### 9.1 Main approaches

- **Auxiliary loss:** penalise unbalanced routing through gradients (covered in [[#8. Expert Collapse and Load Balancing|§8]]).
- **Routing noise:** encourage exploration so that initially neglected experts get opportunities to learn.
- **Bias-based balancing:** adjust expert-selection preferences directly according to measured utilisation.

### 9.2 Routing noise: encourage exploration

Perturb the router scores during training:

$$
\tilde s_i = s_i + \epsilon_i,
$$

where $s_i$ is Expert $i$'s original routing score and $\epsilon_i$ is (trainable) random noise. The example is **KeepTopK** from the original sparsely-gated MoE (Shazeer et al., 2017): *(expanding on the slide)* there the noise is Gaussian with a learned, per-expert scale, $H(x)_i = (xW_g)_i + \text{StandardNormal}()\cdot\text{Softplus}((xW_{\text{noise}})_i)$, so "trainable" refers to the noise magnitude; KeepTopK then keeps the top $k$ noisy scores and sets the rest to $-\infty$ before the softmax, so unselected experts get weight 0.

![[L09_routing_noise.png|460]]

*The noisy router output $H(x)$ is the input $x$ times the router weights $W$ plus a small amount of Gaussian noise $n$.*

- It occasionally selects experts that would otherwise just miss the top $k$.
- It gives less-used experts opportunities to receive training gradients.
- Too much noise can disrupt useful specialisation.

Noise encourages exploration; it does not directly guarantee balanced loads. This approach was used in the original sparsely-gated MoE work.

### 9.3 Bias-based balancing: control expert loads directly

Instead of modifying the training loss, add a selection bias $b_i$ for each expert, $\tilde s_i = s_i + b_i$, and let the router select experts using the adjusted scores $\tilde s_i$. After each training batch,

$$
b_i \leftarrow \begin{cases} b_i - \gamma & \text{if expert } i \text{ is overused} \\ b_i + \gamma & \text{if expert } i \text{ is underused} \end{cases}
$$

where $\gamma$ is the bias adjustment rate. In this design:

- $W_r$ still learns routing scores through normal task-loss gradients.
- $b_i$ is adjusted directly from observed expert loads, **not** through backpropagation.
- The adjusted scores decide **which** experts are selected, while the original scores decide **how** their outputs are weighted.

This is the approach used in DeepSeek-V3 (often called auxiliary-loss-free balancing, since no balancing term enters the loss). Example with three experts and top-2 routing:

| Expert | Original score $s_i$ | Bias $b_i$ | Adjusted score |
|---|---|---|---|
| E1 (overused) | 0.90 | −0.30 | 0.60 |
| E2 | 0.70 | 0.00 | 0.70 |
| E3 (underused) | 0.60 | +0.20 | 0.80 |

Without the biases we select E1 and E2; with them, E3 and E2. But after selection the output weights come from the **original** scores (with any required normalization), not the biased ones; e.g. normalizing $0.70$ and $0.60$ gives weights $0.54$ and $0.46$ *(illustrative arithmetic, not on the slide)*. We can correct *who* gets selected without directly changing the learned assessment of *how useful* each selected expert is.

### 9.4 Expert capacity: what if an expert is overloaded?

Even with load balancing, the router may send an expert more tokens than its allocated computation buffer can hold. The fix is to give each expert a maximum **capacity**:

$$
\boxed{C = \Big\lceil c\,\frac{Tk}{N}\Big\rceil}
$$

where $T$ is the number of tokens in the routing group, $k$ the experts selected per token, $N$ the total number of experts, and $c$ the **capacity factor**, extra room for imbalance. $Tk/N$ is the number of assignments each expert would get under perfect balance; $c > 1$ adds slack. Example: 100 tokens, 4 experts, top-1, $c = 1.25$:

$$
C = \Big\lceil 1.25 \times \frac{100}{4}\Big\rceil = \lceil 31.25\rceil = 32,
$$

so each expert can process at most 32 tokens.

![[L09_capacity_overflow_chart.png|620]]

*100 tokens routed to 4 experts with a 32-token buffer each. Expert 1 receives 85 tokens: 32 are processed and 53 overflow. Experts 2–4 receive 8, 4 and 3, leaving $24 + 28 + 29 = 81$ unused slots.*

Expert 1 has 53 overflowed assignments even though the other experts together have 81 unused slots. Each expert has its own capacity, so unused capacity elsewhere cannot automatically take assignments intended for Expert 1. Overflowed tokens are handled in one of three ways:

1. **Skip:** <mark style="background: #FFF3A3A6;">skip the overflowing expert computation</mark>. The token representation still continues through the Transformer's residual connection ([[Lecture_07_Residuals_and_Normalization#1.4 What they do at inference|Lecture 7 §1.4]]) (the MoE update is just missing for that token in that layer).
2. **Reroute:** <mark style="background: #FFF3A3A6;">send the token to another expert with spare capacity</mark>, if the routing algorithm supports it.
3. **Dropless execution:** allow variable workloads and <mark style="background: #FFF3A3A6;">use specialised GPU operations to process all assignments efficiently</mark> (e.g. MegaBlocks, Gale et al., 2023, which uses block-sparse matrix multiplies). <mark style="background: #ABF7F7A6;">Trade-off: no dropped expert computations, but uneven expert workloads can still create bottlenecks</mark>.

**Load balancing tries to prevent overload. Expert capacity manages overload. Dropless execution avoids discarding assignments, but does not necessarily balance the workload.**

| Most tokens go to FFNN 1 | FFNN 1 full → reroute to FFNN 4 | FFNN 1 and 4 full → skip to next layer |
|---|---|---|
| ![[L09_capacity_reroute_1.png\|300]] | ![[L09_capacity_reroute_2.png\|300]] | ![[L09_capacity_reroute_3.png\|300]] |

*Grootendorst's illustration with expert capacity 3: FFNN 1 receives the majority of tokens; once it reaches capacity, "of" is sent to FFNN 4 instead; once FFNN 1 and FFNN 4 are both full, the next token ("Summarize") skips the experts and goes to the next layer through the residual path.*

---

## 10. Types of MoE

Everything so far used token-choice top-$k$ routing. MoE is not a single architecture but a family, differing in how experts are constructed, selected, combined and trained, all answering one question: given several expert networks, how do we decide which experts should process a particular input?

### 10.1 The family at a glance

| Architecture | Main idea | Key characteristic |
|---|---|---|
| Dense MoE | Every expert processes every token | Fully differentiable routing |
| Sparse top-1 MoE | Choose only one expert per token | Maximum sparsity |
| Sparse top-$k$ MoE | Choose a small number of experts | Multiple expert contributions |
| Expert-choice MoE | Experts choose which tokens to process | Better control of expert load |
| Soft MoE | Route through continuous, weighted token mixtures | Avoid hard token-to-expert selection |
| Hierarchical MoE | Use multiple levels of routers | Organise experts into groups |
| Shared + routed experts | Combine always-active experts with selected experts | Shared knowledge plus specialisation |

These are not mutually exclusive: an MoE can use top-$k$ routing, hierarchical organisation and shared experts at the same time.

### 10.2 How the main architectures differ

Take four experts $E_1, \dots, E_4$.

**A. Dense MoE: everyone contributes.**

$$
y = p_1E_1(x) + p_2E_2(x) + p_3E_3(x) + p_4E_4(x).
$$

All four experts process the token and the router sets their relative contributions. Fully differentiable, but expensive because every expert runs. A dense MoE is **not** the same as a conventional dense FFN: the former has several FFNs and a router; the latter has one FFN shared across tokens.

**B. Sparse top-1: one expert is enough.** With $p = [0.1, 0.6, 0.2, 0.1]$, only Expert 2 is evaluated, $y = p_2E_2(x)$. The Switch Transformer popularised this. It simplifies computation and communication, but creates challenges around overloading particular experts ([[#9.4 Expert capacity: what if an expert is overloaded?|§9.4]]).

**C. Sparse top-$k$: let a few experts collaborate.** With the same probabilities and $k = 2$, select Experts 2 and 3: $y = w_2E_2(x) + w_3E_3(x)$ with renormalized $w_2 = 0.75$, $w_3 = 0.25$ ([[#5.5 How the router works: score, select, process, combine|§5.5]]). Several experts contribute while only a subset is evaluated. Mixtral 8×7B uses eight experts and selects two per token at each MoE layer.

**D. Expert-choice: reverse the selection.** Top-$k$ routing asks "which experts should process this token?"; expert-choice routing asks "**which tokens should this expert process?**" <mark style="background: #FFF3A3A6;">Each expert picks its preferred tokens up to a predetermined capacity</mark>. Experts can therefore receive equal numbers of tokens, while individual tokens can be processed by different numbers of experts (possibly zero). This addresses load imbalance more directly than token-choice routing, though it introduces different routing constraints *(e.g. <mark style="background: #ABF7F7A6;">in autoregressive decoding an expert choosing among a batch's tokens would see tokens from the future</mark>, the same leakage issue as Soft MoE below)*.

**E. Soft MoE: avoid hard token selection.** <mark style="background: #ABF7F7A6;">Instead of assigning discrete tokens to experts</mark>,<mark style="background: #FFF3A3A6;"> soft routing builds continuously weighted combinations of tokens and sends those combinations to experts</mark>. Gradients flow through the routing mechanism without hard assignments. The influential 2024 Soft MoE approach (Puigcerver et al.) demonstrated this mainly in vision Transformers. It needs care in autoregressive language models, <mark style="background: #FFB86CA6;">because mixing tokens could leak information from future tokens into earlier positions</mark>.

**F. Shared + routed experts.** Some knowledge may be broadly useful, while other knowledge benefits from specialisation. Instead of having every expert compete to learn everything, add an always-active shared expert:

$$
y = E_{\text{shared}}(x) + \sum_{i\in\text{TopK}(x)} w_i E_i(x).
$$

The shared expert learns broadly applicable transformations, freeing the routed experts to learn more specialised functions. This is one of the central ideas of **DeepSeekMoE** (Dai et al., 2024), which also uses smaller, more **fine-grained experts** (many small experts, more of them selected per token) to allow more flexible combinations.

**Hierarchical MoE.** Rather than one router choosing among all experts, routing happens in stages: <mark style="background: #FFF3A3A6;">input</mark> → choose an <mark style="background: #FFF3A3A6;">expert group</mark> → <mark style="background: #FFF3A3A6;">choose an expert within the group</mark>. This can organise very large numbers of experts, though its benefits depend on routing design and hardware efficiency. Hierarchical MoEs go back to Jordan and Jacobs in the 1990s.

---

## Numerical Problems

**Problem 1 (FFN size and cost).** A dense FFN has $d = 512$, $m = 2048$. Count its parameters and the FLOPs per token for the two matrix multiplies (count a multiply-add as 2 FLOPs).

$W_1$: $512 \times 2048 = 1{,}048{,}576$; $b_1$: 2048; $W_2$: 1,048,576; $b_2$: 512. FLOPs: $2 \times (2 \cdot 512 \cdot 2048) = 4{,}194{,}304$.

$$\boxed{2{,}099{,}712 \text{ parameters},\ \approx 4.19 \text{ MFLOPs per token}}$$

**Problem 2 (collapse without $\phi$).** Show that removing $\phi$ gives a single linear layer and name its weight and bias.

$(xW_1 + b_1)W_2 + b_2 = x(W_1W_2) + (b_1W_2 + b_2)$, with $W_1W_2 \in \mathbb{R}^{d\times d}$.

$$\boxed{W = W_1W_2,\quad b = b_1W_2 + b_2}$$

The $m$-wide hidden layer adds nothing.

**Problem 3 (gradient conflict).** $\nabla_\theta L_{\text{math}} = (1, 2)$, $\nabla_\theta L_{\text{trans}} = (-3, 1)$. Do they conflict? With a step of size $\eta$ on math only, how does the translation loss change to first order?

Dot product: $1\cdot(-3) + 2\cdot 1 = -1 < 0$, so they conflict. $\Delta L_{\text{trans}} \approx -\eta\,(-3,1)\cdot(1,2) = -\eta(-1)$.

$$\boxed{\text{conflict};\ \Delta L_{\text{trans}} \approx +\eta}$$

The math step makes translation slightly worse.

**Problem 4 ($|x|$ two ways).** Evaluate the ReLU form and the two-expert form at $x = -3$ and $x = 2$.

ReLU form: $\text{ReLU}(-3) + \text{ReLU}(3) = 0 + 3 = 3$; $\text{ReLU}(2) + \text{ReLU}(-2) = 2$. Experts: $x = -3 \Rightarrow p_1 = 0, p_2 = 1$, $y = E_2(-3) = 3$; $x = 2 \Rightarrow p_1 = 1$, $y = E_1(2) = 2$.

$$\boxed{y(-3) = 3,\quad y(2) = 2 \text{ in both forms}}$$

**Problem 5 (weighted combination).** $E_2(x) = (1, -2)$, $E_5(x) = (3, 0)$, weights 0.7 and 0.3. Find $y$.

$0.7(1,-2) + 0.3(3,0) = (0.7 + 0.9,\ -1.4 + 0)$.

$$\boxed{y = (1.6,\ -1.4)}$$

**Problem 6 (softmax router).** Router logits for 4 experts are $(1.2, 0.3, -1.0, 0.8)$. Find the probabilities and the top-1 output. Then for top-2, renormalise the two chosen weights to sum to 1 (as Mixtral does, [[#5.5 How the router works: score, select, process, combine|§5.5]]).

$e^{1.2} = 3.320$, $e^{0.3} = 1.350$, $e^{-1} = 0.368$, $e^{0.8} = 2.226$; sum 7.264. Probabilities $\approx (0.457, 0.186, 0.051, 0.306)$, close to the slide's figure. Top-1: Expert 1, $y = 0.457\,E_1(x)$. Top-2: Experts 1 and 4, $0.457/0.763 = 0.599$ and $0.306/0.763 = 0.401$.

$$\boxed{\text{top-1: } 0.457E_1(x);\quad \text{top-2: } 0.599E_1(x) + 0.401E_4(x)}$$

**Problem 7 (capacity vs compute).** Replace the FFN of Problem 1 with 8 such experts and top-2 routing. Find total expert parameters, active expert parameters per token, and the router's weight count (no bias).

Total: $8 \times 2{,}099{,}712 = 16{,}797{,}696$. Active: $2 \times 2{,}099{,}712 = 4{,}199{,}424$. Router: $512 \times 8 = 4096$.

$$\boxed{\approx 16.8\text{M total},\ \approx 4.2\text{M active }(25\%),\ 4096 \text{ router weights}}$$

Capacity is 8× a dense FFN while per-token compute is 2×, and the router is tiny.

---

## Formula Summary Sheet

| Concept | Formula |
|---|---|
| Original FFN | $\text{FFN}(x) = \max(xW_1 + b_1, 0)W_2 + b_2$ |
| General FFN | $h = \phi(xW_1 + b_1)$, $y = hW_2 + b_2$; $x, y \in \mathbb{R}^{1\times d}$, $h \in \mathbb{R}^{1\times m}$ |
| Collapse without $\phi$ | $(xW_1 + b_1)W_2 + b_2 = x(W_1W_2) + (b_1W_2 + b_2)$ |
| FFN parameters | $2dm + m + d$ |
| Gradient conflict | $\nabla_\theta L_A^\top \nabla_\theta L_B < 0$ |
| $\lvert x\rvert$ by ReLU | $\lvert x\rvert = \text{ReLU}(x) + \text{ReLU}(-x)$ |
| Two-expert MoE | $y = p_1(x)E_1(x) + p_2(x)E_2(x)$, $p_2 = 1 - p_1$ |
| Sparse MoE output | $y = \sum_{i \in \text{selected}} p_i(x)E_i(x)$, e.g. $0.7E_2(x) + 0.3E_5(x)$ |
| Router | $p(x) = \text{softmax}(xW_r)$, $W_r \in \mathbb{R}^{d\times N}$ |
| Top-1 output | $y = p_{i^*}(x)E_{i^*}(x)$, $i^* = \arg\max_i p_i(x)$ |
| Router scores | $s = xW_r$, $s \in \mathbb{R}^{1\times N}$ |
| Top-$k$ renormalization | $w_i = p_i / \sum_{j\in\text{TopK}} p_j$, e.g. $0.60/0.80 = 0.75$ |
| Dense FFN parameters (no bias) | $dm + md = 2dm$ |
| Total vs active expert parameters | $NP$ stored, $kP$ used per token |
| MoE compute | Experts $O(Tkdm)$, router $O(TdN)$ |
| Gradient into expert | $\partial L/\partial e_i = w_i\,\partial L/\partial y$; $\partial L/\partial \theta_i = w_i\,\frac{\partial L}{\partial y}\frac{\partial e_i}{\partial \theta_i}$ |
| Gradient into router weight | $\partial L/\partial w_i = e_i\,\partial L/\partial y$ |
| Router gradient path | $\frac{\partial L}{\partial W_r} = \frac{\partial L}{\partial y}\frac{\partial y}{\partial w}\frac{\partial w}{\partial r}\frac{\partial r}{\partial W_r}$, $r = W_rx$ |
| Top-1 renormalized | $w = p_s/p_s = 1 \Rightarrow \partial L/\partial p_s = 0$ |
| Shared + routed | $y = E_{\text{shared}}(x) + \sum_{i\in\text{TopK}(x)} w_iE_i(x)$ |
| Total loss | $L_{\text{total}} = L_{\text{task}} + \lambda L_{\text{balance}}$ |
| Switch balancing loss | $L_{\text{balance}} = N\sum_i f_i\bar p_i$; uniform routing gives 1 |
| Average router probability | $\bar p_i = \frac{1}{T}\sum_{x\in B} p_i(x)$ |
| Balancing gradient | $\partial L_{\text{balance}}/\partial \bar p_i = Nf_i$ ($f_i$ held fixed) |
| Router update | $W_r^{\text{new}} = W_r^{\text{old}} - \eta\,\partial L_{\text{total}}/\partial W_r$ |
| Routing noise | $\tilde s_i = s_i + \epsilon_i$ |
| Bias-based balancing | $\tilde s_i = s_i + b_i$; $b_i \leftarrow b_i \mp \gamma$ if over/underused |
| Expert capacity | $C = \lceil c\,Tk/N\rceil$, e.g. $\lceil 1.25\cdot 100/4\rceil = 32$ |

---

## Summary

- **The position-wise FFN ([[#1. The Position-Wise Feed-Forward Network|§1]])**
  - Expand $d \to m$, apply a nonlinearity, project back $m \to d$; the same weights at every position.
  - Attention mixes information **between** tokens; the FFN transforms features **within** each token.
  - Without $\phi$ the two linear maps collapse into one; returning to $d$ is what lets the residual add work.

- **Why MoE: dense coupling and two motivations ([[#2. From Dense Models to the MoE Question|§2]], [[#6.1 Dense models revisited: the same machinery for every token|§6.1]])**
  - Dense FFN: $2dm$ parameters, and per-token compute also $\propto dm$, so capacity and compute grow together.
  - Two motivations: **learn better** (less gradient interference, specialisation) and **scale better** (more parameters, same per-token compute).
  - Gradient conflict means a negative **dot product** of the two gradient vectors, not one gradient being negative.
  - A dense network is not less expressive ($|x| = \text{ReLU}(x) + \text{ReLU}(-x)$); MoE's gain is structural.

- **Experts and what they learn ([[#3. A Worked Example: The Absolute Value with Two Experts|§3]]–[[#4. Mixture of Experts in LLMs|§4]])**
  - An MoE layer = experts (usually whole FFNs) + a router; each layer has its own experts, so tokens take different paths.
  - Explicit decomposition into simple functions plus a learned selector, as in the $|x|$ example; gate and experts train together and the gate infers the source from input features.
  - Dense layer: every parameter is used for every input. Sparse model: only a portion runs per input, and MoE is the main way to get one.
  - Experts are not topic experts. Switch encoder experts specialise in token types (punctuation, verbs, numbers); Mixtral decoder experts track syntax more than domain.
  - Routing helps only if a token's features reveal which "source" it belongs to.

- **Routing: score, select, process, combine ([[#5. Routing|§5]])**
  - $s = xW_r$, softmax, pick top $k$, run only those experts, output $\sum w_iE_i(x)$ (each $E_i(x)$ is $1\times d$).
  - Dense MoE runs every expert with soft weights; sparse MoE runs only $k$. "MoE" in LLMs almost always means sparse.
  - The router predicts usefulness from the token **before** experts run; the output mixes hidden vectors, not answers.
  - The selected expert's output is multiplied by its gate value, which is what lets task-loss gradients reach $W_r$.
  - Renormalizing the top-$k$ probabilities is common for $k \ge 2$ (Mixtral) but optional; Switch (top-1) keeps the raw probability.

- **Total vs active parameters ([[#6. Total vs Active Parameters|§6]])**
  - $NP$ stored, $kP$ used; 8 experts top-2 ≈ one-quarter of the expert compute of running all eight.
  - Compute: experts $O(Tkdm)$, router $O(TdN)$; raising $N$ at fixed $k$ grows capacity, not expert compute.
  - Fewer active parameters ≠ proportionally lower latency (memory, uneven routing, communication, training complexity).
  - MoE does nothing for attention's sequence-length cost or the KV cache.

- **Training an MoE ([[#7. Training an MoE|§7]])**
  - Expert gradient $\partial L/\partial e_i = w_i\,\partial L/\partial y$ ("how should the expert change?"); router gradient $\partial L/\partial w_i = e_i\,\partial L/\partial y$ ("rely on it more or less?").
  - $w_i$ is computed from logits $r = W_rx$, so the router gradient reaches $W_r$ through the softmax.
  - Training differentiates along the chosen path and treats the top-$k$ choice as fixed.
  - Top-$k$ selection itself has no useful gradient; an unselected expert gets no task gradient from that token.
  - Top-1 with renormalization gives $w = 1$, so $\partial L/\partial p = 0$ and the router learns nothing from the task loss (needs auxiliary losses or STE). With $k \ge 2$ renormalization still passes gradient.
  - Dense MoE is end-to-end differentiable but costs $O(N)$ experts per token.

- **Expert collapse and the balancing loss ([[#8. Expert Collapse and Load Balancing|§8]])**
  - Collapse: more tokens → more training → more useful → more tokens; neglected experts starve. Costs: wasted capacity and uneven hardware load.
  - $L_{\text{total}} = L_{\text{task}} + \lambda L_{\text{balance}}$, with $L_{\text{balance}} = N\sum_i f_i\bar p_i$ (Switch); uniform routing gives 1.
  - $f_i$ (discrete share of tokens) is held fixed; the gradient $Nf_i$ pushes probability $\bar p_i$ away from overused experts until top-1 decisions flip.
  - $\bar p_i$ averages over **all** tokens, including those sent elsewhere, and is computed before selection.
  - Computed once per batch per MoE layer, fresh each batch, not a running average.
  - The loss is a heuristic (some non-uniform settings score below 1), and balanced usage does not imply different experts.

- **Other balancing methods and capacity ([[#9. Other Load-Balancing Methods and Expert Capacity|§9]])**
  - Routing noise $\tilde s_i = s_i + \epsilon_i$ (noisy top-$k$, Shazeer 2017): exploration, no balance guarantee.
  - Bias-based $\tilde s_i = s_i + b_i$ (DeepSeek-V3): biases are set from observed loads, not backprop; they change **who** is selected, while the original scores set the **weights**.
  - Capacity $C = \lceil cTk/N\rceil$; overflow is skipped (residual path), rerouted, or handled dropless (MegaBlocks).
  - One expert can overflow while others have spare slots, because capacity is per expert.

- **The MoE family ([[#10. Types of MoE|§10]])**
  - Dense, top-1 (Switch), top-$k$ (Mixtral 8×7B, top-2), expert-choice, soft MoE, hierarchical, shared + routed (DeepSeekMoE, with fine-grained experts).
  - Every variant answers "which experts should process this input?" differently, and they can be combined.
  - A dense MoE (many FFNs + router) is not a dense FFN (one shared FFN).
  - Expert-choice balances expert load but a token may get zero experts; soft MoE must avoid leaking future tokens in autoregressive models.
