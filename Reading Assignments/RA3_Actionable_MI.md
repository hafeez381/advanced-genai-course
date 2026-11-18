---
type: study-aid
course: '[[_CS5326 Advanced GenAI|CS5326 Advanced GenAI]]'
source:
- '[[RA3.pdf]]'
tags:
- course/cs5326
---

%% nav:start %%
[[_CS5326 Advanced GenAI|CS5326 Advanced GenAI]] · ← [[RA2_DPO]]
Related: [[Lecture_05_Attention_and_Multi_Head_Attention#3. Multi-Head Attention: The First Pass|Lecture 5 (MHA)]], [[Lecture_09_FFNs_Dense_and_MoE#1.2 The FFN equations and shapes|Lecture 9 (FFN)]]
%% nav:end %%

# Locate, Steer, and Improve — A Practical Survey of Actionable Mechanistic Interpretability in LLMs (Zhang et al., Findings of ACL 2026)

*Scope: Abstract through the Limitation section (pp. 10317–10326), plus the useful parts of the appendices in [[#9. Appendix|§9]]: challenges and the evaluation framework (A), future directions (B, briefly), object details (D) and method details (E). Acknowledgement, References, Appendix C (Figures 2–3, outlines of the survey itself) and Table 3 (the 200+ paper list) are excluded.*

## Topic Inventory

- 1. Abstract and Introduction
  - 1.1 Mechanistic interpretability as an observational science
  - 1.2 Two gaps in existing reviews
  - 1.3 The Locate, Steer, and Improve pipeline
- 2. Interpretable Objects of LLMs
  - 2.1 Token embedding and the residual stream
  - 2.2 Multi-head attention
  - 2.3 The FFN as a key–value memory, and neurons
  - 2.4 Sparse autoencoder features
  - 2.5 Notation summary
- 3. Localizing Methods
  - 3.1 Magnitude analysis
  - 3.2 Causal attribution
  - 3.3 Gradient detection
  - 3.4 Probing
  - 3.5 Vocabulary projection
  - 3.6 Circuit discovery
  - 3.7 The six methods side by side
- 4. Steering Methods
  - 4.1 Amplitude manipulation
  - 4.2 Targeted optimization
  - 4.3 Vector arithmetic
  - 4.4 The three methods side by side
- 5. Applications — Improve Alignment
  - 5.1 Safety and reliability
  - 5.2 Fairness and bias
  - 5.3 Persona and role
- 6. Applications — Improve Capability
  - 6.1 Multilingualism
  - 6.2 Knowledge management
  - 6.3 Logic and reasoning
- 7. Applications — Improve Efficiency
  - 7.1 Efficient training
  - 7.2 Efficient inference
- 8. Conclusion and Limitation
  - 8.1 Conclusion
  - 8.2 Limitation
- 9. Appendix
  - 9.1 Challenges for actionable MI (A)
  - 9.2 A minimal evaluation framework (A)
  - 9.3 Broadening the architectural scope (B)
  - 9.4 Information flow in Transformer blocks (D.1)
  - 9.5 SAEs in practice (D.2)
  - 9.6 Vocabulary projection case studies (E.1)
  - 9.7 Deriving steering vectors (E.2)
  - 9.8 Causal tracing (E.3)
  - 9.9 Gradient detection and Integrated Gradients (E.4)
  - 9.10 Edge Attribution Patching (E.5)

---

## 1. Abstract and Introduction

### 1.1 Mechanistic interpretability as an observational science

Large Language Models (LLMs) have achieved remarkable success, yet their internal decision-making remains largely opaque. That lack of transparency poses significant risks and limits our ability to optimize them efficiently. **Mechanistic Interpretability (MI)** aims to **reverse-engineer** a model into understandable components: not just *what* it outputs, but *which* internal parts compute it and *how*.

Existing reviews fall into two groups. One focuses on the theory and foundations of MI and gives technical roadmaps for dissecting Transformers. These reviews prioritize **discovery**, understanding the inner workings, and so treat MI as an **observational science**, leaving practical applications underexplored. A second, newer group starts to bridge understanding and use, discussing how MI can help downstream tasks or specific domains.

### 1.2 Two gaps in existing reviews

The authors name two limitations of those reviews:
1. They **lack a sufficient categorization and clear definition** of MI methods in practical application contexts.
2. Their **coverage of applications is often incomplete**, and the illustration too general, so readers struggle to turn mechanistic insights into actionable interventions.

They argue the gap matters now in particular. LLM progress has mostly come from external scaling (larger models, more data, more compute), and this paradigm is increasingly hitting bottlenecks (the diminishing returns of [[Lecture_10_Neural_Scaling_Laws#5.2 How expensive the next halving is|Lecture 10 §5.2]]). Further advances, they argue, must come from a deeper understanding of internal mechanisms: shifting from scaling the model's "outside" to surgically improving it from the "inside". The field lacks a unified guide that categorizes these internal methods and gives a concrete pipeline for active model improvement.

### 1.3 The Locate, Steer, and Improve pipeline

The paper proposes the **"Locate, Steer, and Improve"** pipeline, which turns MI from passive observation into actionable engineering. Its three contributions:

1. **Pipeline-driven framework.** Define the **interpretable objects** inside an LLM ([[#2. Interpretable Objects of LLMs|§2]]), then sort methods into **Localizing** (diagnosis, [[#3. Localizing Methods|§3]]) and **Steering** (intervention, [[#4. Steering Methods|§4]]). For each technique the paper gives its **Methodological Formulation**, its **Applicable Objects** and its **Scope**, so readers can see quickly how to implement it.
2. **Comprehensive application paradigms.** Applications are surveyed under three themes: **Improve Alignment**, **Improve Capability** and **Improve Efficiency** ([[#5. Applications — Improve Alignment|§5]]–[[#7. Applications — Improve Efficiency|§7]]). Instead of a general overview, each scenario is summarized by its representative **MI paradigms**, the recurring locate-then-steer patterns.
3. **Insights and resources.** A discussion of challenges and future directions (in the appendix, outside this note's scope), and a curated, method-tagged collection of over 200 papers.

![[RA3_fig1_overview.png|760]]

*The paper's structure as a pipeline. Left: the interpretable objects of a decoder-only Transformer (residual stream, MHA, FFN, and derived objects: neurons, SAE features, circuits). Middle: six localizing methods feed three steering methods. Right: the three application themes and their eight sub-areas.*

---

## 2. Interpretable Objects of LLMs

Before any method can locate or steer something, the "somethings" must be defined. The paper formulates them for the **decoder-only** Transformer, the predominant architecture of contemporary LLMs, and every later method is described by which of these objects it applies to.

### 2.1 Token embedding and the residual stream

**Token embedding.** The **embedding matrix** $W_E \in \mathbb R^{\lvert\mathcal V\rvert\times d_{\text{model}}}$ maps each of the $\lvert\mathcal V\rvert$ vocabulary tokens to a $d_{\text{model}}$-dimensional vector. For token $t_i$ the **token embedding**, which is also the initial residual-stream state, is

$$x_i^0 = W_E[t_i] + p_i,$$

where $W_E[t_i]$ is the row of $W_E$ for that token and $p_i$ the positional embedding. (Modern LLMs such as Llama instead apply **RoPE**, rotary position embeddings, directly to queries and keys; see [[Lecture_03_From_Sinusoidal_Limitations_to_RoPE|Lecture 3]].)

**Transformer block.** An LLM stacks $L$ layers, each with an MHA (multi-head attention) block and an FFN (feed-forward network) block. The primary communication channel is the **residual stream state** $x^l$, updated at layer $l$ as

$$x^{l,\text{mid}} = x^l + h^l_{\text{attn}}(x^l) \qquad\qquad\text{(1)}$$
^eq-1

$$x^{l+1} = x^{l,\text{mid}} + h^l_{\text{ffn}}(x^{l,\text{mid}}) \qquad\qquad\text{(2)}$$
^eq-2

where $x^{l,\text{mid}}$ is the intermediate state after the MHA block. Normalization is omitted, as it often is in high-level MI analyses. This is the residual connection of [[Lecture_05_Attention_and_Multi_Head_Attention#4. Residual Connections and Layer Normalization|Lecture 5 §4]], read as a shared channel.

**Why the additive structure matters for MI.** Unrolling [[#^eq-1|eq. (1)]] and [[#^eq-2|eq. (2)]] gives $x^L = x^0 + \sum_l h^l_{\text{attn}} + \sum_l h^l_{\text{ffn}}$: the final state is a *sum* of every component's output. *(Expanding on this:)* so the final prediction can be decomposed into individual component contributions, and each head or FFN can be studied by what it writes into the stream. Most methods in [[#3. Localizing Methods|§3]] rely on this; [[#9.4 Information flow in Transformer blocks (D.1)|§9.4]] gives the read/write picture.

### 2.2 Multi-head attention

MHA lets tokens gather information from other positions through $H$ independent heads ([[Lecture_05_Attention_and_Multi_Head_Attention#3. Multi-Head Attention: The First Pass|Lecture 5 §3]]). For head $h$ at layer $l$ the learnable weights are $W_Q^{l,h}, W_K^{l,h}, W_V^{l,h} \in \mathbb R^{d_{\text{model}}\times d_{\text{head}}}$ and the output projection $W_O^{l,h} \in \mathbb R^{d_{\text{head}}\times d_{\text{model}}}$. With $N$ tokens, the **attention score matrix** $A^{l,h} \in \mathbb R^{N\times N}$ is

$$A^{l,h} = \operatorname{softmax}\!\left(\frac{(x^lW_Q^{l,h})(x^lW_K^{l,h})^\top}{\sqrt{d_{\text{head}}}}\right). \qquad\qquad\text{(3)}$$
^eq-3

The **head output** is $h^{l,h}_{\text{attn}} = A^{l,h}(x^lW_V^{l,h})W_O^{l,h}$, and the block output sums the heads, $h^l_{\text{attn}} = \sum_{h=1}^{H} h^{l,h}_{\text{attn}}$. Writing MHA as a sum over heads (instead of a concatenation followed by one $W_O$) is the same computation split by head, and it makes each head's contribution to the residual stream separate.

### 2.3 The FFN as a key–value memory, and neurons

FFNs act as position-wise feature transformers ([[Lecture_09_FFNs_Dense_and_MoE#1.2 The FFN equations and shapes|Lecture 9 §1.2]]):

$$h^l_{\text{ffn}} = \sigma(x^{l,\text{mid}}W_{\text{up}}^l)\,W_{\text{down}}^l \qquad\qquad\text{(4)}$$
^eq-4

where $x^{l,\text{mid}}$ is the input, $h^l_{\text{ffn}}$ the output, $\sigma$ the activation function, $W_{\text{up}}^l \in \mathbb R^{d_{\text{model}}\times d_{\text{ff}}}$ and $W_{\text{down}}^l \in \mathbb R^{d_{\text{ff}}\times d_{\text{model}}}$. Modern LLMs often use SwiGLU, with an extra gating matrix $W_{\text{gate}}^l$ ([[Lecture_08_Activations|Lecture 8]]); the paper presents the standard FFN for simplicity.

Following Geva et al. (2021, 2022), $W_{\text{up}}$ and $W_{\text{down}}$ are viewed as a **key–value dictionary**. **Neuron $j$** is the atomic unit made of a pair of weights: the **key weight** $k_j^l$ (the $j$-th column of $W_{\text{up}}^l$, as the paper's text says "row", see the note below) and the **value weight** $v_j^l$ (the $j$-th row of $W_{\text{down}}^l$). The intermediate vector $s^l = \sigma(x^{l,\text{mid}}W_{\text{up}}^l)$ holds the **neuron activations**.

*(Expanding on the key–value reading:)* writing [[#^eq-4|eq. (4)]] neuron by neuron, $h^l_{\text{ffn}} = \sum_j s^l_j\, v_j^l$ with $s^l_j = \sigma(x^{l,\text{mid}}\cdot k_j^l)$. Each key is a pattern detector: its activation says how strongly the input matches. Each value is what that neuron writes into the residual stream when it fires. So the FFN looks up memories (values) by similarity to stored patterns (keys).

> [!warning] Possible source inconsistency
> The paper calls $k_j^l$ "the $j$-th row of $W_{\text{up}}^l$" and $v_j^l$ "the $j$-th column of $W_{\text{down}}^l$". With the shapes the paper gives ($W_{\text{up}} \in \mathbb R^{d_{\text{model}}\times d_{\text{ff}}}$, row vectors multiplied on the left), neuron $j$'s key is a **column** of $W_{\text{up}}$ (length $d_{\text{model}}$) and its value a **row** of $W_{\text{down}}$. Rows and columns are swapped relative to the row-vector convention; the meaning, one key and one value vector per neuron, is unaffected.

### 2.4 Sparse autoencoder features

Individual objects such as a neuron activation $s^l$ or a residual state $x^l$ are often **polysemantic**: one neuron responds to several unrelated concepts. This comes from **superposition** (Elhage et al., 2022): the model represents more features than it has dimensions by storing them as overlapping, nearly orthogonal directions. A **Sparse Autoencoder (SAE)** gives a principled way to disentangle such representations into **monosemantic** features (one concept each).

SAEs are trained **layer-wise**. To reconstruct $x^l$:

$$a = \sigma(x^lW_{\text{enc}} + b_{\text{enc}}), \qquad \hat x^l = aW_{\text{dec}} + b_{\text{dec}} \qquad\qquad\text{(5)}$$
^eq-5

with learnable $W_{\text{enc}} \in \mathbb R^{d_{\text{model}}\times d_{\text{SAE}}}$ and $W_{\text{dec}} \in \mathbb R^{d_{\text{SAE}}\times d_{\text{model}}}$, where $d_{\text{SAE}} \gg d_{\text{model}}$ (an overcomplete dictionary). The objective is reconstruction error plus a sparsity penalty:

$$\mathcal L = \lVert x^l - \hat x^l\rVert_2^2 + \lambda\lVert a\rVert_1.$$

The **SAE feature** $f_j$ is the $j$-th row of $W_{\text{dec}}$, a semantic direction in the residual stream, and the **feature activation** $a_j$ (the $j$-th entry of $a$) measures how strongly it is present in the input. *(Expanding on why this works:)* the $\ell_1$ penalty forces only a few of the many features to be active for any input, so each feature is pushed to stand for one concept; the reconstruction term makes sure the features still explain $x^l$. Practical details (expansion factor, dead latents, feature absorption, pre-trained SAE suites) are in [[#9.5 SAEs in practice (D.2)|§9.5]].

### 2.5 Notation summary

| Object | Item | Notation |
|---|---|---|
| Token embedding | embedding matrix; token $i$ embedding (input) | $W_E$; $x_i^0$ |
| Residual stream | residual stream state; intermediate state (post-attention) | $x^l$; $x^{l,\text{mid}}$ |
| Attention | Q, K, V, O weight matrices; attention score matrix; head output; MHA block output | $W_Q^{l,h}, W_K^{l,h}, W_V^{l,h}, W_O^{l,h}$; $A^{l,h}$; $h^{l,h}_{\text{attn}}$; $h^l_{\text{attn}}$ |
| FFN | up-projection (key) matrix; down-projection (value) matrix; FFN block output | $W_{\text{up}}^l$; $W_{\text{down}}^l$; $h^l_{\text{ffn}}$ |
| Neuron | neuron activation; $j$-th key weight; $j$-th value weight | $s^l$; $k_j^l$; $v_j^l$ |
| SAE feature | $j$-th feature; $j$-th feature activation | $f_j$; $a_j$ |
| Circuit | computational graph of objects and edges | $(\mathcal O, \mathcal E)$ |

Objects are either **static** (parameters: $W_E$, $W_Q^{l,h}$, $k_j^l$, $f_j$, …) or **dynamic** (computed per input: $x^l$, $A^{l,h}$, $s^l$, $a_j$, …). The distinction matters below, because some methods only work on one kind.

---

## 3. Localizing Methods

With the objects defined, the first stage of the pipeline is **localizing**: identifying which objects are responsible for a specific behaviour. This lays the foundation for later analysis and steering. Six methods follow, ordered roughly from cheapest and least rigorous to most structured.

### 3.1 Magnitude analysis

**Formulation.** Score each object $o_j$ (a weight or activation) with a scalar function $s_j = f(o_j)$, and keep the high scorers (e.g. $\arg\operatorname{top}k_j\, s_j$) for inspection or intervention. Common metrics are the $L_p$-norm, $s_j = \lVert o_j\rVert_p$, and the max value, $s_j = \max_k\lvert(o_j)_k\rvert$.

**Objects and scope.** It applies to both static parameters and dynamic representations:
1. **Parameters:** per-weight or per-row norms highlight **outliers**, the same outliers that trouble quantization ([[ATML_Lecture_13 - Model Compression#4.9 When quantization struggles — the outlier problem|ATML L13 §4.9]]).
2. **Representations:** ranking neurons or SAE features by activation norm localizes specialized features, and the norm of a head's output, $\lVert h^{l,h}_{\text{attn}}\rVert$, identifies influential heads.
3. **Layer-wise distances:** comparing representations across contrastive inputs, $\lVert x^l - x'^l\rVert$, localizes layers where task-specific information diverges most; comparing consecutive layers, $\lVert x^l - x^{l+1}\rVert$, identifies **redundant computation** (a layer that barely changes the stream).

Magnitude analysis is **training-free but data-dependent**: no training cost, but the input data decides which objects look salient. It is the same idea as magnitude pruning ([[ATML_Lecture_13 - Model Compression#3.4 How to score groups|ATML L13 §3.4]]): large means important.

### 3.2 Causal attribution

Magnitude says an object is large, not that it matters. Causal attribution tests that directly.

**Formulation.** Let $F(\cdot)$ be a scalar model output of interest (e.g. a target token's logit) and $o$ an internal object. Its **causal effect** is

$$\Delta F(o) = F\big(\mathrm{do}(o \leftarrow \tilde o)\big) - F(o),$$

where the intervention $\mathrm{do}(\cdot)$ (from causal inference: *set* $o$ to a value, rather than observe it) is either **patching**, swapping in a counterfactual state, or **ablation**, setting it to zero. $\tilde o$ is the modified state. A large $\lvert\Delta F(o)\rvert$ means $o$ is a **critical mediator** of the behaviour.

**Objects and scope.** It mainly targets dynamic objects ($x^l$, $h^l_{\text{ffn}}$, $h^{l,h}_{\text{attn}}$) and circuit edges.
- **Patching-based attribution** replaces an object computed on the original input with the same object computed on a counterfactual input, isolating specific information pathways. Patching systematically across layers localizes where task-specific information, such as a fact, is introduced. *(Expanding on an example:)* run "The Eiffel Tower is in" and "The Colosseum is in"; patch layer $l$'s state from the first run into the second; if the output flips from "Rome" to "Paris", layer $l$ carries the fact. The paper's worked version, causal tracing, is in [[#9.8 Causal tracing (E.3)|§9.8]].
- **Ablation-based attribution** zeros out objects such as specific heads or circuit edges and measures the performance drop, to determine whether they are **necessary**.

It gives rigorous causal responsibility but is **computationally expensive**, scaling linearly with the number of objects (one forward pass per intervention). That cost is what motivates gradient detection as a fast approximation for initial screening.

### 3.3 Gradient detection

**Formulation.** Score object $o_j$ by the local sensitivity of a scalar target $F(x)$ (a logit or a loss): $s_j(x) = \phi\big(\nabla_{o_j}F(x), o_j\big)$. Two common instantiations are $s_j = \lVert\nabla_{o_j}F(x)\rVert$ and $s_j = \nabla_{o_j}F(x)^\top o_j$ (Sundararajan et al., 2017). The second comes from the first-order Taylor approximation

$$F(o_j + \Delta o_j) \approx F(o_j) + \nabla_{o_j}F(x)^\top\Delta o_j.$$

*(Expanding on this:)* set $\Delta o_j = -o_j$, i.e. ablate the object. The predicted change is $-\nabla_{o_j}F^\top o_j$, so gradient × value is a one-backward-pass **estimate of the ablation effect** $\Delta F$ of [[#3.2 Causal attribution|§3.2]], for every object at once. This is why it can screen before causal attribution. It is the same score as SNIP in pruning ([[ATML_Lecture_13 - Model Compression#3.4 How to score groups|ATML L13 §3.4]]). The full derivation, and Integrated Gradients, are in [[#9.9 Gradient detection and Integrated Gradients (E.4)|§9.9]].

**Objects and scope.** $F$ is differentiable with respect to any object in [[#2.5 Notation summary|§2.5]], so it applies to inputs, activations and parameters: on $x_i^0$ and $x^l$ it ranks influential tokens, positions and layers; on neurons ($s^l$, $v_j^l$) it localizes knowledge- or context-sensitive neurons and traces upstream dependencies; on head outputs and parameter blocks ($W_{Q/K/V/O}^{l,h}$, $W_{\text{up}}^l$, $W_{\text{down}}^l$) it ranks salient attention and FFN modules, individual parameters or whole blocks.

It is **data-dependent and costs extra compute** for backpropagation, so it is often a lightweight ranking step before causal attribution. Pairing it with causal attribution also gives a contrastive signal through the counterfactual logit gap $\lvert\text{logit}(x) - \text{logit}(x_{\text{cf}})\rvert$.

### 3.4 Probing

The first three methods ask how much an object *affects* the output. Probing asks what information an object *contains*.

**Formulation.** Train a predictor $g_\psi$ to decode a label $y$ from an internal vector $z \in \mathbb R^{d_{\text{model}}}$ (e.g. $x^l$), for example a linear softmax probe $\hat y = g_\psi(z) = \operatorname{softmax}(W_Pz)$, on a dataset $D = \{(z, y)\}$. For localization, probes are trained on each candidate object (layer, head, FFN) and the objects are ranked by decoding accuracy, usually followed by causal attribution for validation.

**Objects and scope.** Internal vectors and activations: the residual stream ($x^l$, $x^{l,\text{mid}}$), head outputs, FFN outputs and neuron activations. Layer-wise probes track where contextual knowledge or conflict signals become decodable in $x^l$; head- and module-level probes localize capabilities or risks; probes on SAE feature activations $a_j$ enable concept decoding.

Its scope is **supervised decoding**: it needs labelled data and extra training. *(Expanding on the caveat:)* decodable does not mean used: the model may carry information it never acts on, which is why probing is paired with causal attribution to test **functional necessity**.

### 3.5 Vocabulary projection

**Formulation.** The canonical technique is the **Logit Lens** (nostalgebraist, 2020). It treats the unembedding matrix $W_U \in \mathbb R^{d_{\text{model}}\times\lvert\mathcal V\rvert}$ as a universal decoder for intermediate states: for any $z \in \mathbb R^{d_{\text{model}}}$,

$$p = \operatorname{softmax}(zW_U).$$

*(Expanding on why this is legitimate:)* the residual stream ([[#2.1 Token embedding and the residual stream|§2.1]]) is a shared space that the final layer decodes with $W_U$; reading an intermediate state with the same $W_U$ shows what the model "would predict if it stopped here".

**Objects and scope.** On $x^l$ it traces how the prediction evolves layer by layer and highlights the layers where traits emerge. On a head output $h^{l,h}_{\text{attn}}$ it shows what the head injects into the stream, characterizing functional heads such as the **Name Mover Head** (copies a name from context to the output) and **Copy Suppression Heads** (lower the logit of tokens already copied). Projecting neuron value weights $v_j^l$ or SAE features $f_j$ interprets them by the tokens they promote.

Its scope is **direct semantic mapping**: unlike probing, it gives a **zero-shot** view with no auxiliary training. Case studies for each object type are in [[#9.6 Vocabulary projection case studies (E.1)|§9.6]].

### 3.6 Circuit discovery

The previous methods score objects one at a time. Circuit discovery looks for how they connect.

**Formulation.** A circuit is a **mechanistic pathway**: structured, directed dependencies among internal objects that implement a behaviour. Let $(\mathcal O, \mathcal E)$ be the computational graph over objects $\mathcal O$ and directed edges $\mathcal E$, where $e_{ij}$ is signal flow from $o_i$ to $o_j$. A circuit $\mathcal C \subseteq \mathcal E$ is **faithful** if restricting computation to $\mathcal C$ (patching or ablating every other edge) preserves the target output $F(x)$ or task performance.

**Objects and scope.** It targets **edges** between objects (between $x^l$, head outputs and FFN outputs, among attention/FFN modules, or among SAE features). It is put into practice in three ways:
1. **Intervention-based search** for causally necessary edges (activation patching or ablation search);
2. **Attribution-based edge scoring**, ranking edges by estimated influence (gradient-based attributions and position-aware refinements);
3. **Feature-based replacement models** (SAE or **transcoder** variants, an SAE trained to imitate a whole MLP's input–output map), which lift circuit discovery to sparse features and allow attribution graphs and prompt-specific tracing.

Its scope is **causal subgraph identification**: extracting a minimally sufficient subnetwork, across layers, that mediates a behaviour. It relies on causal attribution for validation and often uses gradient signals to propose candidate edges, so it combines [[#3.2 Causal attribution|§3.2]] and [[#3.3 Gradient detection|§3.3]] at the level of edges; [[#9.10 Edge Attribution Patching (E.5)|§9.10]] shows how.

### 3.7 The six methods side by side

| Method | Question it answers | Needs training? | Cost | Typical objects |
|---|---|---|---|---|
| Magnitude analysis | Which objects are large? | no (data-dependent) | lowest | weights, activations, layers |
| Causal attribution | Does changing it change the output? | no | high, linear in #objects | $x^l$, head/FFN outputs, edges |
| Gradient detection | How sensitive is the output to it? | no | one backward pass | inputs, activations, parameters |
| Probing | What information does it contain? | yes, a probe | labelled data + training | $x^l$, head/FFN outputs, $s^l$, $a_j$ |
| Vocabulary projection | Which tokens does it promote? | no (zero-shot) | low | $x^l$, head outputs, $v_j^l$, $f_j$ |
| Circuit discovery | Which pathway implements it? | sometimes (SAE/transcoder) | high | edges between objects |

A pattern runs through the survey: a cheap method proposes candidates (magnitude, gradient, probing) and causal attribution confirms them.

---

## 4. Steering Methods

Localization says *where* a behaviour lives. Steering manipulates those objects to change the model's outputs. The three methods differ in what they change (activations or weights) and whether the change is temporary or permanent.

### 4.1 Amplitude manipulation

**Formulation.** Change the magnitude of a target object $o$ to $\tilde o$, in one of two ways:
- **Ablation or patching**, $\tilde o \in \{0,\ \mathbb E[o],\ o_{\text{tgt}}\}$: zero it, replace it with its mean, or replace it with a target value, to suppress or alter the information flowing through $o$.
- **Scaling**, $\tilde o = \alpha\cdot o$: a coefficient $\alpha$ amplifies or attenuates its downstream influence during the forward pass.

The operations are identical to those in causal attribution ([[#3.2 Causal attribution|§3.2]]), but the **objective differs**: attribution uses them to *diagnose* causality, steering applies them to *pre-localized* objects to *intervene* on behaviour.

**Objects and scope.** Ablation and patching target object outputs ($x^l$, $h^l_{\text{ffn}}$, $h^{l,h}_{\text{attn}}$), activations ($s^l$, $a_j$) and circuit edges, removing or swapping their influence. This mitigates unwanted behaviours such as language or knowledge confusion, or cuts computation to accelerate inference. Scaling mostly targets $s^l$, $a_j$, or the steering vectors of [[#4.3 Vector arithmetic|§4.3]], for fine-grained control over specific attributes.

It is **inference-time activation control**: training-free, and nothing is stored in the weights.

### 4.2 Targeted optimization

**Formulation.** Permanently update a weight subset $w$. Given a steering dataset $D_{\text{steer}}$, minimize a task loss $\mathcal L$:

$$w_{\text{new}} \leftarrow w - \eta\cdot\nabla_w\mathcal L(D_{\text{steer}}).$$

Focus is enforced by **freezing the remaining weights** $\theta\setminus w$, or by giving $w$ a higher learning rate $\eta$. This is selective fine-tuning ([[ATML_Lecture_10 - Fine-Tuning#7.2 Selective parameter tuning|ATML L10 §7.2]]) with $w$ chosen by localization rather than by a fixed rule.

**Objects and scope.** $w$ is often spread across several layers, at two granularities:
1. **Fine-grained objects:** a neuron's key/value weights ($k_j^l$, $v_j^l$) or specific matrices such as $W_{\text{down}}^l$, motivated by findings that these weights densely encode specific factual knowledge and capabilities.
2. **Coarse-grained objects:** every weight in a functional component (a specific head, an FFN, a whole Transformer block), which adapts general processing mechanisms and is needed for broad, complex behaviours.

It is characterized by **persistence and surgical precision**. Unlike amplitude manipulation it rewrites parameters with gradient training, which enables precise memory editing and focused capability gains while minimizing collateral damage to unrelated traits.

### 4.3 Vector arithmetic

**Formulation.** Assuming concepts or skills are encoded **linearly** (the linear representation hypothesis), steer a target object $z$, a hidden state or a parameter, by

$$\hat z = z + \alpha\cdot v,$$

where $v$ is the **steering vector**, the direction of the target attribute, and $\alpha$ the intervention strength.

**Objects and scope.**
1. **Hidden states** ($x^l$ and $h^{l,h}_{\text{attn}}$): $v$ is usually a **contrastive activation mean**, the difference between average activations on inputs with and without the attribute (Rimsky et al., 2024), or an SAE feature $f_j$ for the target concept.
2. **Parameters:** $v$ is a **task vector** from model merging, the element-wise difference between a fine-tuned model and its pre-trained base, $v = \theta_{\text{ft}} - \theta_{\text{base}}$. It encapsulates a transferable skill or behaviour; adding it to another model transfers the skill, subtracting it removes it.

How $v$ is computed in each case is in [[#9.7 Deriving steering vectors (E.2)|§9.7]]. Its scope is **additive directionality**. Unlike targeted optimization, the paper describes it as a steering force that pushes the model towards an attribute without permanently altering weights. *(For the parameter case, adding a task vector does change the stored weights; it is "non-permanent" in that it is a fixed, removable addition requiring no further training.)*

### 4.4 The three methods side by side

| | Amplitude manipulation | Targeted optimization | Vector arithmetic |
|---|---|---|---|
| Operation | $\tilde o \in \{0, \mathbb E[o], o_{\text{tgt}}\}$ or $\alpha o$ | $w \leftarrow w - \eta\nabla_w\mathcal L$ | $\hat z = z + \alpha v$ |
| Acts on | activations, outputs, edges | a localized weight subset | hidden states or parameters |
| Training | none | gradient training | none (vector from data or a fine-tune) |
| Persistence | per forward pass | permanent | additive, removable |
| Scope (paper's label) | inference-time activation control | persistence and surgical precision | additive directionality |

---

## 5. Applications — Improve Alignment

The rest of the survey shows the pipeline in use. Each application area is summarized as a few **paradigms**, combinations of a localizing method from [[#3. Localizing Methods|§3]] and a steering method from [[#4. Steering Methods|§4]]. Alignment comes first: making the model safer, fairer and controllable in persona.

### 5.1 Safety and reliability

Two steering paradigms dominate.

1. **Component-level manipulation**, via amplitude manipulation and targeted optimization. Work here localizes specialized attention heads, neurons and SAE features that encode toxicity, harmfulness or hallucination. Mitigation is then usually **amplitude manipulation** to ablate or downscale those activations during the forward pass. Targeted optimization offers a **persistent** alternative, updating parameters within the identified safety circuits, neurons, layers or vectors.
2. **Vector arithmetic.** Studies have established that high-level concepts such as **truthfulness, refusal and instruction following** are mediated by specific directions in latent space. Injecting steering vectors along them can inhibit hallucinations, correct failed refusals, or improve instruction following, bolstering safety and reliability while preserving general capabilities.

### 5.2 Fairness and bias

The dominant paradigm has **two steps**: use **causal attribution** and **magnitude analysis** to find bias-carrying objects, then **amplitude manipulation** or **targeted optimization** to suppress their effect.

For data-centric biases (gender, demographic, cultural), causal attribution (zero ablation, activation patching) traces how biased signals are mediated by specific heads, FFNs or residual states. Some studies extend this to broader bias categories by first scoring internal structures such as circuit edges with magnitude analysis, then validating with causal attribution, the screen-then-confirm pattern of [[#3.7 The six methods side by side|§3.7]]. Mitigation is either inference-time amplitude manipulation (suppressing or down-weighting activations) or targeted optimization that persistently updates the identified subsets.

### 5.3 Persona and role

Two steering paradigms, the same pair as safety in [[#5.1 Safety and reliability|§5.1]] but in the opposite order of emphasis:

1. **Vector arithmetic** on global persona representations. Counterfactual inputs with opposing personas give steering vectors, which are added to the residual stream to amplify or suppress attributes during the forward pass, controlling **sycophancy** (telling users what they want to hear), role-playing ability and character traits.
2. **Component-level manipulation.** Pinpoint the components responsible for specific values or personalities, then **amplitude manipulation** selectively activates or suppresses those neurons, or **targeted optimization** fine-tunes them to embed objective-aligned behaviour persistently.

Beyond steering, MI also **evaluates** psychological traits: probing locates persona-specific representations, and the similarity between internal states and persona vectors serves as a metric of how well the model aligns with a psychological profile.

---

## 6. Applications — Improve Capability

Alignment shapes *how* the model behaves; the next theme makes it *better* at tasks: languages, knowledge and reasoning.

### 6.1 Multilingualism

Two paradigms:
1. **Amplitude manipulation of language-specific features.** Language data is used to pinpoint **language-specific neurons** or SAE features via **magnitude analysis** of their activations; zeroing or scaling them enhances multilingual performance and controls the output language.
2. **Vector arithmetic on the residual stream.** To find where to intervene, works use **vocabulary projection** and magnitude analysis to identify the crucial layers, often in the **late layers**. Steering vectors there shift representations along language-related directions, improving multilingual understanding and mitigating **language inconsistency** (answering in the wrong language). Most works build these vectors from parallel language data; recent ones build them directly from language-specific neurons or SAE features, linking the two paradigms.

### 6.2 Knowledge management

Three paradigms, using targeted optimization, amplitude manipulation and vector arithmetic:

1. **Updating** (precise knowledge editing and memory rewriting). Localize editable carriers with causal attribution or probing (optionally helped by gradient detection or vocabulary projection), then rewrite them persistently with targeted optimization (e.g. ROME/MEMIT-style edits of FFN weights, Meng et al., 2022, 2023). Alternatively, edit at the activation level with amplitude manipulation: gated activation control of components, or SAE-based interventions for **unlearning** and detoxification, with relevant features found by magnitude analysis.
2. **Retention** (preventing interference under continual updates). Identify interference circuits with circuit discovery or causal attribution, then suppress the conflict sources with head- or FFN-level amplitude manipulation (pruning or controlled attention). To limit drift, continual adaptation restricts targeted optimization to selected modules or uses frozen-backbone representation edits; probing diagnostics help forecast and evaluate side effects.
3. **Consolidation** (combining skills or models). Lightweight probing identifies compatible carriers (aligned parameter subspaces, transferable feature bases), which are then composed persistently with vector arithmetic (task vectors, [[#4.3 Vector arithmetic|§4.3]]), aggregating complementary skills while mitigating interference.

### 6.3 Logic and reasoning

Three paradigms:
1. **Arithmetic reasoning.** Localize operator or reasoning carriers with causal attribution or magnitude analysis, then restrict **targeted optimization** to the identified heads, FFNs or layers.
2. **Steering reasoning strategies.** Find strategy-relevant directions or sparse features with magnitude analysis or cross-model feature discovery, then modulate inference by injecting strategy vectors (vector arithmetic) or scaling sparse features (amplitude manipulation).
3. **Reasoning reliability.** Lightweight diagnosis: probing or stepwise verification flags erroneous intermediate steps and triggers targeted re-generation before errors propagate. Gradient detection that quantifies token influence along a chain-of-thought (CoT) trajectory can also serve, potentially, as a diagnostic of whether the model attends to relevant logic or to spurious correlations.

---

## 7. Applications — Improve Efficiency

The last theme uses the same localization to spend *less* compute, in training and in inference.

### 7.1 Efficient training

Two paradigms:
1. **Sparse fine-tuning** via targeted optimization. Unlike **PEFT** methods ([[ATML_Lecture_10 - Fine-Tuning#7. Parameter-Efficient Fine-Tuning (PEFT)|ATML L10 §7]]) that add external modules such as LoRA, these fine-tune **intrinsic subsets** of the model, often matching or exceeding full fine-tuning. Gradient detection or magnitude analysis pinpoints task-specific neurons, and only their key/value weights ($k_j^l$, $v_j^l$) are updated. More granular approaches isolate extremely sparse subsets: about **0.1% of neurons** via causal attribution, attention heads chosen by $A^{l,h}$, or subgraphs from circuit discovery. Substantial gains are attainable by targeting the right mechanistic components.
2. **Training-dynamics monitoring** via magnitude analysis. Validation loss can miss critical **phase transitions**, so internal metrics track them: monitoring the attention pattern $A^{l,h}$ of **induction heads** (heads that complete repeated patterns, "…A B … A → B") signals the onset of in-context learning, while weight norms, gradients or activations predict generalization phases and warn against premature stopping.

### 7.2 Efficient inference

Two paradigms:
1. **Selective computation**, using magnitude analysis or circuit discovery to find redundant calculations and induce sparsity at two levels.
   - **Data level:** gradient detection or magnitude analysis prunes tokens and the **KV cache** ([[Lecture_06_Decoding_KV_Cache_MQA_GQA_FlashAttention#4. KV Cache|Lecture 6 §4]]); circuit discovery preserves the attention heads critical for performance.
   - **Model level:** magnitude analysis or probing of residual streams or router activations lets the model **dynamically skip redundant layers** and inefficient **MoE experts** ([[Lecture_09_FFNs_Dense_and_MoE#4.1 Experts and the router|Lecture 9 §4.1]]); identifying specialized neurons allows executing only critical subnetworks.
2. **Adaptive quantization**, chiefly **layer-wise mixed-precision quantization**: bit-widths are allocated by layer sensitivity. Sensitivity is measured with magnitude analysis of outliers or of $x^l$, gradient detection, or vocabulary projection. Robust layers get lower bit-widths to maximize efficiency; sensitive layers keep higher precision. This is a concrete answer to the open "which bits for which layer" problem of [[ATML_Lecture_13 - Model Compression#4.10 Mixed precision, combining methods, and architecture search|ATML L13 §4.10]].

---

## 8. Conclusion and Limitation

### 8.1 Conclusion

The survey reframes MI from a mainly observational endeavour into a practical, actionable paradigm. Under the unified **Locate, Steer, and Improve** pipeline it shows how interpretable objects can be precisely **localized**, causally **manipulated**, and used to **improve** alignment, capability and efficiency. The authors' reading of the literature is that many recent advances, from safety and persona alignment to knowledge editing and sparse fine-tuning, are most effective when grounded in explicit mechanistic intervention. Challenges and future directions are in Appendix A (outside this note's scope).

### 8.2 Limitation

- **Architecture and modality coverage.** The survey covers MI for **dense** LLMs and does not systematically cover methods specific to other architectures and modalities. **MoE** models add routing and sparsely activated experts, and **vision** models rely on modality-specific representations and structures, both of which pose distinct interpretability challenges. Many of the methods are conceptually general, but a full treatment of these architectures is left to future work.
- **No standard evaluation of localization.** The field lacks unified benchmarks or standardized evaluation protocols for localization methods, which makes it hard to compare approaches or to tell whether the identified components are **causally optimal**. This also affects applications, where interventions often rely on a single localization method without formal guarantees. Some works partly mitigate this by using several localization techniques and checking that they converge on similar model regions, but principled evaluation remains an open challenge. The appendix expands on both points ([[#9.1 Challenges for actionable MI (A)|§9.1]]–[[#9.3 Broadening the architectural scope (B)|§9.3]]).

---

## 9. Appendix

The main text names each method in a paragraph. The appendices give the challenges behind the Limitation section, a concrete evaluation recipe, and the working details of the objects and methods. They are collected here in the paper's order (A, B, D, E), with links back to the main-text sections they deepen.

### 9.1 Challenges for actionable MI (A)

MI is **not a universal solution**. Four challenges and boundary conditions may limit its scalability, reliability and practical impact.

1. **Scalability and granularity.** MI remains hard to scale beyond low-level components. Individual neurons and learned features are increasingly well characterized, but identifying **higher-level computational structures** still relies heavily on manual inspection; automated methods need substantial human intervention and do not generalize robustly across prompts, tasks or models. Beyond 100B parameters, fine-grained causal localization ([[#3.2 Causal attribution|§3.2]]) becomes prohibitively expensive, forcing a trade-off between **granularity and computational feasibility**. SAEs add their own cost, since a surrogate model must be trained, growing with model size and feature dimension. Exact attribution would in principle need exhaustive interventions, which is infeasible at modern scale, so most analyses work at coarser granularity or rely on heuristic approximations (such as [[#3.3 Gradient detection|§3.3]] and [[#9.10 Edge Attribution Patching (E.5)|§9.10]]).
2. **Distributed mechanisms vs localized sparsity.** Many methods force representations into a small set of monosemantic components so they can be interpreted. Aggressive sparsity can **prune or obscure components that genuinely belong to the mechanism** but do not fit a sparse pattern. So sparsity buys interpretability at the risk of missing distributed or "inactive" subcomponents. Metrics that balance sparsity against mechanistic completeness are an open problem.
3. **Intervention robustness and side effects.** Editing and steering often lack robustness and predictability: a change aimed at one behaviour can have unintended effects on other tasks or domains. For instance, Yu and Ananiadou (2025) show that modifying a very small number of neurons can substantially degrade overall language performance. Accurate localization plus steering without collateral disruption remains a central technical challenge.
4. **Evaluation protocols.** There is no robust framework for assessing the **faithfulness** of localization and explanation methods. Some benchmarks exist, but there is no consensus metric for whether an identified component truly is the causal mechanism, and without mechanism-level ground truth, rigorous validation and comparison are inherently hard. This is the second limitation of [[#8.2 Limitation|§8.2]], and [[#9.2 A minimal evaluation framework (A)|§9.2]] is the authors' preliminary answer.

### 9.2 A minimal evaluation framework (A)

The authors propose a minimal framework covering three task settings. For each it fixes the features to localize or steer, the primary metric, the essential side-effect checks, and example datasets.

| Capability | Feature to localize/steer | Primary metric | Side-effect check | Example datasets |
|---|---|---|---|---|
| Math reasoning | arithmetic and reflection features | accuracy, reflection-token ratio | general capabilities | GSM8K, AIME25 |
| Safety | refusal features | refusal rate, attack success rate | over-refusal | SORRY-Bench, OR-Bench |
| Knowledge | factual associations | edit success rate | locality, fluency | CounterFact, ZsRE |

**Using it to evaluate localization.** Identify candidate reflection features in math reasoning, ablate them, and measure the change in reflection-token frequency on AIME25: the size of the change is a quantitative measure of **causal faithfulness**.

**Using it to compare steering methods.** Apply different steering methods to the **same localized features** and compare accuracy or behaviour shifts, e.g. amplifying a reflection feature by amplitude manipulation ([[#4.1 Amplitude manipulation|§4.1]]), injecting it as a steering vector ([[#4.3 Vector arithmetic|§4.3]]), and localized weight updates ([[#4.2 Targeted optimization|§4.2]]). Holding the localization fixed isolates the effect of the steering method.

**Side effects are a first-class requirement:**
- **Math reasoning:** improving reflection or arithmetic must not degrade other domains, so the steered model is also evaluated on general-knowledge and safety benchmarks.
- **Safety:** the goal is not only a higher refusal rate on harmful prompts (SORRY-Bench) but also avoiding **over-refusal**, rejecting benign prompts. Overly aggressive safety steering can seriously hurt helpfulness; OR-Bench measures this.
- **Knowledge editing:** **locality** is the main side-effect metric, whether editing one fact (e.g. a president's name) changes unrelated knowledge; **fluency** checks that generation has not become repetitive or ungrammatical.

### 9.3 Broadening the architectural scope (B)

The Limitation section left MoE and vision models to future work; Appendix B sketches how the framework would extend. Most localizing and steering methods are general in spirit and transfer once the new architecture's interpretable objects are identified.
- **MoE models** ([[Lecture_09_FFNs_Dense_and_MoE#4.1 Experts and the router|Lecture 9 §4.1]]) add natural new objects: router states, routing activations, routing decisions and expert activations. Magnitude analysis, for example, can move from neuron activations to router activations to find influential routes or experts, which are then steered by intervening on those expert modules.
- **Multimodal LLMs (MLLMs)** add vision encoders and cross-modal modules, so their objects go beyond language-side activations to include visual token representations and cross-modal interactions.

### 9.4 Information flow in Transformer blocks (D.1)

To locate and steer objects, one must first understand how information moves through the model. The **residual stream** ([[#2.1 Token embedding and the residual stream|§2.1]]) is the central communication channel, a "highway", and each sub-layer performs a specific operation on it:

![[RA3_fig4_info_flow.png|720]]

*Left: the stream $x_i$ for each token flows upward, with Attn and FFN blocks reading from it and adding their output back. Right, zoomed: an attention layer, where heads score other positions through $W_{QK}^{l,h}$ and move information through $W_{OV}^{l,h}$; and an FFN, where keys $W_{\text{in}}^l$ match the input and values $W_{\text{out}}^l$ write to the stream. Top: the final state is decoded by $W_U$ and a softmax. (After Ferrando et al., 2024.)*

- **Attention heads, "the router".** A head **reads** from the residual stream of previous tokens through the subspace projections $W_Q^{l,h}, W_K^{l,h}$ and **writes** the attended information to the current position through $W_V^{l,h}, W_O^{l,h}$. Heads mainly manage **information routing** and contextual dependencies. *(In the figure, $W_{QK} = W_QW_K^\top$ and $W_{OV} = W_VW_O$ are the two products that matter: the first decides where to look, the second what to copy.)*
- **FFN, "the processor".** FFNs act on each position independently, as key–value memories ([[#2.3 The FFN as a key–value memory, and neurons|§2.3]]): the first layer projects the stream into a high-dimensional state of **knowledge keys**, and the second writes the retrieved knowledge back.

The additive structure, $x^{l+1} = x^l + \text{MHA}(x^l) + \text{FFN}(x^l)$, means every feature in the residual stream is a **linear combination of the outputs of all previous components**, which is what justifies disentangling it with SAEs.

### 9.5 SAEs in practice (D.2)

[[#2.4 Sparse autoencoder features|§2.4]] defined the SAE; this is the motivation and practice behind it.

![[RA3_fig5_sae.png|680]]

*An SAE attached to one layer of a frozen LLM: the dense activation $z$ is encoded by $W_{\text{enc}}$ into a wider, sparse code $h(z)$ and decoded by $W_{\text{dec}}$ into $\hat z$, trained with $\mathcal L(z) = \lVert z - \hat z\rVert_2^2 + \alpha\lVert h(z)\rVert_1$. (After Shu et al., 2025; this figure writes $z$, $h(z)$ and $\alpha$ for the main text's $x^l$, $a$ and $\lambda$.)*

- **Why: resolving superposition.** Networks represent more features than they have neurons by encoding them as nearly orthogonal directions, which makes neurons polysemantic. An SAE projects the low-dimensional dense activation into a **higher-dimensional sparse space**, "unpacking" the superposition. The resulting granular features become an **actionable vocabulary**: they can be targeted for steering.
- **Where.** The SAE is an independent module attached to one object of a frozen LLM, and can be applied to nearly any of them: $s^l$, $x^l$, $x^{l,\text{mid}}$, $h^{l,h}_{\text{attn}}$, $h^l_{\text{attn}}$, $h^l_{\text{ffn}}$.
- **Expansion factor.** The ratio $d_{\text{SAE}}/d_{\text{model}}$ is a critical hyperparameter, typically **16× to 128×**, to have room for the many features hidden in superposition.
- **Training challenges.**
  - **Dead latents:** many feature neurons never activate, wasting capacity. Mitigated by **ghost gradients** (an extra loss term that gives dead features a gradient) or periodically **resampling** dead neurons.
  - **Feature absorption:** a broadly activating feature suppresses more specific ones (e.g. a general feature absorbs a narrower one it contains). Architectures such as **Gated SAEs**, **Top-K** and **BatchTopK SAEs** (keep exactly the $k$ largest activations instead of an $\ell_1$ penalty), **JumpReLU SAEs** and **Binary SAEs** improve feature quality and reconstruction.
- **Pre-trained resources.** **Gemma Scope** (SAEs for every layer of Gemma 2) and **Llama Scope** (every layer of Llama 3) are public; Anthropic's "**Golden Gate Claude**" showed a single amplified feature (the Golden Gate Bridge) taking over a model's behaviour, a vivid case of feature steering.

### 9.6 Vocabulary projection case studies (E.1)

Four case studies show the logit lens of [[#3.5 Vocabulary projection|§3.5]] applied to each kind of object.

![[RA3_fig6_vocab_projection.png|720]]

*(a) Logit lens on the residual stream of a multilingual model translating into Chinese: rows are layers (1 at the bottom, 31 at the top), cells show the top decoded token. Early layers decode input-surface tokens, middle layers decode English words ("flower"), and only the last layers decode the Chinese output. (b) Projecting one SAE feature through $W_U$: its top positive logits are all "food" tokens, its negative ones unrelated words. (After Wendler et al., 2024 and Shu et al., 2025.)*

- **Layer-wise (residual stream): the latent language of Transformers.** Wendler et al. (2024) found three phases in a translation task: early layers handle the **surface form** of the input language; middle layers process meaning in an abstract, **English-centric** concept space, even for non-English tasks; final layers rotate back to the **target language**. English serves as an internal pivot for abstract reasoning in multilingual LLMs, which is why multilingual steering works best in late layers ([[#6.1 Multilingualism|§6.1]]).
- **Head-wise: indirect object identification (IOI).** Wang et al. (2023) reverse-engineered how GPT-2 small completes "When Mary and John went to the store, John gave a drink to → Mary". Projecting each head's output onto the vocabulary revealed **Name Mover Heads** (e.g. head 9.9) that write the correct name "Mary" into the stream, and **Negative Name Mover Heads** (e.g. head 10.7) that attend to the name but write a *negative* vector to suppress it. *(Head 10.7 is the head McDougall et al. later characterized as a copy-suppression head, [[#3.5 Vocabulary projection|§3.5]].)* The distinction is only visible through vocabulary projection, because attention patterns alone look the same for both.
- **Neuron-wise: FFNs as key–value memories.** Geva et al. (2021) projected columns of $W_{\text{down}}$ (the value vectors $v_j$) into the vocabulary and found that single neurons promote **semantically related token clusters** (e.g. one neuron boosting "press", "news", "media"). FFN updates thus refine predictions by composing pre-learned distributions layer by layer.
- **Feature-wise: interpreting SAE features.** The logit contribution $l_j = f_jW_U$ of feature $f_j$ ranks the tokens it promotes; a feature that raises "Food", "food" and "Foods" (panel b) encodes a food concept.

### 9.7 Deriving steering vectors (E.2)

[[#4.3 Vector arithmetic|§4.3]] injected a direction $v$; the two main ways to get $v$ are these.

**Contrastive activation means** ("mass-mean shift" or "activation addition"). Take a prompt set $D^+$ that elicits the target behaviour (e.g. sycophantic responses) and $D^-$ that elicits the opposite. Collect activations $x^l$ at a chosen layer for both, and take the difference of the centroids:

$$v = \mu^+ - \mu^- = \frac{1}{\lvert D^+\rvert}\sum_{x_i\in D^+} x_i^l - \frac{1}{\lvert D^-\rvert}\sum_{x_j\in D^-} x_j^l. \qquad\qquad\text{(6)}$$
^eq-6

$v$ is the "direction of sycophancy". Adding $\alpha v$ to the residual stream at inference shifts the state toward the centroid of the positive behaviour, inducing the trait without changing weights. *(Expanding on why the difference of means works:)* anything the two sets share (topic, syntax, length) cancels in the subtraction, leaving mainly the direction along which they differ.

**SAE features.** SAEs isolate steering directions more precisely because they have already disentangled the stream.

![[RA3_fig7_sae_steering.png|720]]

*(a) Compare the SAE codes $h(z)$ of a neutral state $z$ and $h(z')$ of a concept-rich state $z'$; the features that differ most give the steering vector. (b) Add it to the residual stream during the forward pass. (c) Prompted for an angry review, the steered model writes a happy one ("Happiness" feature), a confused one ("Confusion"), or points out the phone does not exist ("Fact"). (After Shu et al., 2025.)*

Prompts with the target concept (e.g. "Happiness") and without it give a concept-rich state $z'$ and a baseline $z$. Comparing their sparse activations $h(z')$ and $h(z)$ identifies a subset $J$ of features with significant differences, and the steering vector is the weighted sum of their decoder columns:

$$v = \sum_{j\in J}\delta_j\, W_{\text{dec}}[:, j],$$

where $\delta_j$ is feature $j$'s activation difference between the two states. It is injected as $\hat x = x + \alpha v$. Compared with [[#^eq-6|eq. (6)]], it keeps only the differences that line up with interpretable features, discarding the rest of the centroid gap. *(Notation: the main text defines $f_j$ as the $j$-th **row** of $W_{\text{dec}} \in \mathbb R^{d_{\text{SAE}}\times d_{\text{model}}}$; this appendix writes "columns" $W_{\text{dec}}[:, j]$, the transposed convention. Both mean feature $j$'s $d_{\text{model}}$-dimensional direction.)*

### 9.8 Causal tracing (E.3)

[[#3.2 Causal attribution|§3.2]] introduced patching; **causal tracing** (Meng et al., 2022) is its representative implementation, used to find where a fact such as "The Space Needle is in → Seattle" is stored.

![[RA3_fig8_causal_tracing.png|740]]

*Top: (a) the clean run predicts "Seattle"; (b) noise on the subject tokens ("Space Needle", marked \*) corrupts it; (c) one clean hidden state is patched into the corrupted run; (d) if the output is fixed, that state carried the fact. Bottom: the recovered probability of "Seattle" for every layer and token when restoring (e) the whole state, (f) only the MLP, (g) only attention, in GPT-2-XL. MLPs at the last subject token light up early ("early site"); attention at the last token lights up late ("late site").*

It is **restoration** ("denoising") patching, in three steps:
1. **Corrupted run.** Add Gaussian noise to the embeddings of the subject tokens ("Space Needle"), which erases the fact: the probability of "Seattle" drops sharply.
2. **Patched run.** For one layer $l$ and position $i$, copy the clean hidden activation $h_i^l$ into the corrupted computation.
3. **Effect.** The **Indirect Effect (IE)** is how much of the original probability of "Seattle" the restoration recovers. A high IE means the state at $(l, i)$ carries the fact.

Repeating this for every layer and position gives the heatmaps, which show **two localized mechanisms** for factual recall:
- **Early site, subject tokens:** a strong effect in **early MLP layers** at the subject's last token; the model decodes the entity and retrieves its properties early.
- **Late site, last token:** a second peak in **late attention layers** at the final position ("downtown"); attention moves the retrieved information to where the prediction is made.

This is the result behind editing facts by updating mid-layer FFN weights ([[#6.2 Knowledge management|§6.2]]), consistent with the key–value view of FFNs.

### 9.9 Gradient detection and Integrated Gradients (E.4)

[[#3.3 Gradient detection|§3.3]] stated the gradient × value score; Appendix E.4 derives it and its main extension. Gradient detection is a **first-order Taylor approximation** to intervention effects, and **Integrated Gradients (IG)** is a path-integrated extension that is exact ("complete") relative to a baseline.

**Taylor derivation.** Fix an input $x$ and an object $o_j \in \mathbb R^d$, and change it additively, $o_j \to o_j + \Delta o_j$. If $F$ is twice differentiable near $o_j$, Taylor's theorem with remainder gives, for some $\xi \in (0, 1)$,

$$F(o_j + \Delta o_j) - F(o_j) = \nabla_{o_j}F(o_j)^\top\Delta o_j + \underbrace{\tfrac12\,\Delta o_j^\top H_{o_j}F(o_j + \xi\Delta o_j)\,\Delta o_j}_{\text{higher-order remainder}} \qquad\qquad\text{(7)}$$
^eq-7

where $H_{o_j}F$ is the Hessian. Gradient detection keeps the first term as a fast proxy and drops the remainder. Along the line $g(\alpha) = F(o_j + \alpha\Delta o_j)$ the chain rule gives $g'(0) = \nabla_{o_j}F^\top\Delta o_j$, so the first term is the **directional derivative** of $F$ along $\Delta o_j$.

**Grad × input.** Treating removal as $\Delta o_j = -o_j$ gives $F(0) - F(o_j) \approx -\nabla_{o_j}F^\top o_j$, which motivates the signed score $s_j = \nabla_{o_j}F^\top o_j$ (or $\lvert s_j\rvert$ when only size matters). [[#^eq-7|Eq. (7)]] makes the conditions explicit: the approximation is good when the **curvature** (Hessian) and the **perturbation size** are small, and removing a whole object is not a small perturbation. Scores can be split by coordinate, $\nabla_{o_j}F^\top o_j = \sum_{k=1}^d \frac{\partial F}{\partial(o_j)_k}(o_j)_k$, and aggregated over a dataset with $\mathbb E_x[s_j]$, $\mathbb E_x[\lvert s_j\rvert]$ or rank statistics.

**Integrated Gradients.** A gradient at a single point can be uninformative under **saturation** (e.g. a sigmoid at 0.999 has near-zero gradient though the input matters), and it does not say what "absence" means. IG fixes both by attributing the difference $F(o_j) - F(o'_j)$ against a chosen **baseline** $o'_j$. On the straight path $\gamma(\alpha) = o'_j + \alpha(o_j - o'_j)$, $\alpha \in [0, 1]$, coordinate $k$ gets

$$\mathrm{IG}_k(o_j; o'_j) = (o_j - o'_j)_k\int_0^1\frac{\partial F(\gamma(\alpha))}{\partial\gamma_k}\,d\alpha. \qquad\qquad\text{(8)}$$
^eq-8

**Completeness.** Since $\frac{d}{d\alpha}F(\gamma(\alpha)) = \nabla_\gamma F(\gamma(\alpha))^\top(o_j - o'_j)$, summing [[#^eq-8|eq. (8)]] over $k$ and applying the fundamental theorem of calculus,

$$\sum_{k=1}^d \mathrm{IG}_k(o_j; o'_j) = \int_0^1\frac{d}{d\alpha}F(\gamma(\alpha))\,d\alpha = F(o_j) - F(o'_j).$$

IG splits the baseline-to-input change in $F$ **exactly** across the coordinates, with no remainder, unlike [[#^eq-7|eq. (7)]]. In practice the integral is a Riemann sum with $m$ steps:

$$\mathrm{IG}_k \approx (o_j - o'_j)_k\cdot\frac1m\sum_{t=1}^m\frac{\partial F(\gamma(t/m))}{\partial\gamma_k}.$$

With $o'_j = 0$ and a gradient that barely changes along the path, the average equals the endpoint gradient and IG reduces to grad × input: grad × input is IG's locally linear special case. Because IG explains $F(o_j) - F(o'_j)$, the **baseline is part of the explanation** and must reflect a sensible notion of "absence" for that object.

**Choices of target $F$.** Gradient detection is modular in $F$:

| Target | $F(x)$ |
|---|---|
| Loss | $-\log p(y^\star\mid x)$ or a cross-entropy objective |
| Logit margin | $\text{logit}_y(x) - \text{logit}_{y_{\text{foil}}}(x)$ |
| Counterfactual gap | $\text{logit}_y(x) - \text{logit}_y(x_{\text{cf}})$ |
| Distribution shift | $D_{\mathrm{KL}}\big(p(\cdot\mid x)\,\Vert\,p(\cdot\mid x_{\text{cf}})\big)$, or symmetric variants |

### 9.10 Edge Attribution Patching (E.5)

[[#3.6 Circuit discovery|§3.6]] listed attribution-based edge scoring; **Edge Attribution Patching (EAP)** (Syed et al., 2024) and its IG variant **EAP-IG** make it concrete, applying [[#9.9 Gradient detection and Integrated Gradients (E.4)|§9.9]] to edges.

**Set-up.** A circuit is a subset $\mathcal E_{\text{sub}} \subset \mathcal E$ of edges that mediates task performance. An edge $e_{u\to v}$ carries the activation $a_u$ of a **sender** node $u$ (a head or MLP neuron) into the input $z_v$ of a **receiver** node $v$. Its score is the change in a task metric $R$ (e.g. a logit difference) when the edge is patched.

**The bottleneck.** Exact activation patching replaces the edge's activation with its value on a corrupted input $x_{\text{corr}}$ while everything else runs on the clean input $x_{\text{clean}}$. Doing this for every edge costs $\lvert\mathcal E\rvert$ forward passes, $O(\lvert\mathcal E\rvert)$, which is prohibitive for large models.

**Linear EAP.** Approximate the patching effect with a first-order Taylor expansion:

$$S_{\text{linear}}(u\to v) \approx \underbrace{\big(a_u(x_{\text{clean}}) - a_u(x_{\text{corr}})\big)}_{\text{sender activation difference}}\cdot\underbrace{\left.\frac{\partial R}{\partial z_v}\right\rvert_{x_{\text{clean}}}}_{\text{receiver input gradient}} \qquad\qquad\text{(9)}$$
^eq-9

Two forward passes (for $a_u$ on both inputs) and one backward pass (for $\nabla R$) score **all** edges at once, $O(1)$ in the number of edges. It is grad × input from [[#9.9 Gradient detection and Integrated Gradients (E.4)|§9.9]] with $\Delta o = a_u(x_{\text{clean}}) - a_u(x_{\text{corr}})$.

**EAP-IG.** Linear EAP assumes the model is linear between the corrupted and clean states; saturation or strong non-linearity can make the gradient at $x_{\text{clean}}$ vanish or mislead. EAP-IG averages the gradient along the interpolation $x_\alpha = x_{\text{corr}} + \alpha(x_{\text{clean}} - x_{\text{corr}})$:

$$S_{\text{IG}}(u\to v) = \big(a_u(x_{\text{clean}}) - a_u(x_{\text{corr}})\big)\cdot\int_0^1\left.\frac{\partial R}{\partial z_v}\right\rvert_{x_\alpha}d\alpha \approx \big(a_u(x_{\text{clean}}) - a_u(x_{\text{corr}})\big)\cdot\frac1n\sum_{k=1}^n\left.\frac{\partial R}{\partial z_v}\right\rvert_{x_{k/n}}$$

with $n$ steps (e.g. $n = 50$).

**Workflow.**
1. **Activation collection:** forward passes on $x_{\text{clean}}$ and $x_{\text{corr}}$ give $\Delta a_u$ for every sender.
2. **Integrated gradients:** $n$ forward and backward passes on the interpolated inputs $x_{k/n}$ give the average gradient $\nabla_{z_v}R$ for every receiver.
3. **Edge scoring:** combine the two for every edge, $S_{\text{IG}}(u\to v) = \Delta a_u\odot\nabla_{z_v}$, summed over dimensions to give one number per edge.
4. **Pruning:** rank edges by $\lvert S_{\text{IG}}\rvert$ and keep those above a threshold; they form the circuit.

EAP-IG needs $n$ backward passes, so it is slower than linear EAP, but it is still far faster than brute-force patching and much more faithful where the model is non-linear.

---

## Formula Summary Sheet

| Concept | Formula | Where |
|---|---|---|
| Token embedding | $x_i^0 = W_E[t_i] + p_i$ | [[#2.1 Token embedding and the residual stream\|§2.1]] |
| Residual stream update ([[#^eq-1\|eq. (1)]], [[#^eq-2\|(2)]]) | $x^{l,\text{mid}} = x^l + h^l_{\text{attn}}(x^l)$, $x^{l+1} = x^{l,\text{mid}} + h^l_{\text{ffn}}(x^{l,\text{mid}})$ | [[#2.1 Token embedding and the residual stream\|§2.1]], [[Lecture_05_Attention_and_Multi_Head_Attention#4. Residual Connections and Layer Normalization\|L5 §4]] |
| Attention score ([[#^eq-3\|eq. (3)]]) | $A^{l,h} = \operatorname{softmax}\big((x^lW_Q^{l,h})(x^lW_K^{l,h})^\top/\sqrt{d_{\text{head}}}\big)$ | [[#2.2 Multi-head attention\|§2.2]], [[Lecture_05_Attention_and_Multi_Head_Attention#3. Multi-Head Attention: The First Pass\|L5 §3]] |
| Head and block output | $h^{l,h}_{\text{attn}} = A^{l,h}(x^lW_V^{l,h})W_O^{l,h}$, $h^l_{\text{attn}} = \sum_h h^{l,h}_{\text{attn}}$ | [[#2.2 Multi-head attention\|§2.2]] |
| FFN ([[#^eq-4\|eq. (4)]]) | $h^l_{\text{ffn}} = \sigma(x^{l,\text{mid}}W_{\text{up}}^l)W_{\text{down}}^l = \sum_j s_j^l v_j^l$ | [[#2.3 The FFN as a key–value memory, and neurons\|§2.3]], [[Lecture_09_FFNs_Dense_and_MoE#1.2 The FFN equations and shapes\|L9 §1.2]] |
| SAE ([[#^eq-5\|eq. (5)]]) | $a = \sigma(x^lW_{\text{enc}} + b_{\text{enc}})$, $\hat x^l = aW_{\text{dec}} + b_{\text{dec}}$ | [[#2.4 Sparse autoencoder features\|§2.4]] |
| SAE loss | $\mathcal L = \lVert x^l - \hat x^l\rVert_2^2 + \lambda\lVert a\rVert_1$ | [[#2.4 Sparse autoencoder features\|§2.4]] |
| Magnitude score | $s_j = \lVert o_j\rVert_p$ or $\max_k\lvert(o_j)_k\rvert$; keep $\arg\operatorname{top}k_j\,s_j$ | [[#3.1 Magnitude analysis\|§3.1]] |
| Causal effect | $\Delta F(o) = F(\mathrm{do}(o\leftarrow\tilde o)) - F(o)$ | [[#3.2 Causal attribution\|§3.2]] |
| Gradient score | $s_j = \lVert\nabla_{o_j}F\rVert$ or $\nabla_{o_j}F^\top o_j$; $F(o+\Delta o) \approx F(o) + \nabla F^\top\Delta o$ | [[#3.3 Gradient detection\|§3.3]], [[ATML_Lecture_13 - Model Compression#3.4 How to score groups\|ATML L13 §3.4]] |
| Linear probe | $\hat y = \operatorname{softmax}(W_Pz)$ | [[#3.4 Probing\|§3.4]] |
| Logit lens | $p = \operatorname{softmax}(zW_U)$ | [[#3.5 Vocabulary projection\|§3.5]] |
| Faithful circuit | $\mathcal C \subseteq \mathcal E$ preserves $F(x)$ when all other edges are ablated | [[#3.6 Circuit discovery\|§3.6]] |
| Amplitude manipulation | $\tilde o \in \{0, \mathbb E[o], o_{\text{tgt}}\}$ or $\tilde o = \alpha o$ | [[#4.1 Amplitude manipulation\|§4.1]] |
| Targeted optimization | $w_{\text{new}} \leftarrow w - \eta\nabla_w\mathcal L(D_{\text{steer}})$, rest frozen | [[#4.2 Targeted optimization\|§4.2]] |
| Vector arithmetic | $\hat z = z + \alpha v$; task vector $v = \theta_{\text{ft}} - \theta_{\text{base}}$ | [[#4.3 Vector arithmetic\|§4.3]] |
| Contrastive steering vector ([[#^eq-6\|eq. (6)]]) | $v = \mu^+ - \mu^- = \frac{1}{\lvert D^+\rvert}\sum_{D^+}x^l - \frac{1}{\lvert D^-\rvert}\sum_{D^-}x^l$ | [[#9.7 Deriving steering vectors (E.2)\|§9.7]], [[#4.3 Vector arithmetic\|§4.3]] |
| SAE steering vector | $v = \sum_{j\in J}\delta_j W_{\text{dec}}[:, j]$, then $\hat x = x + \alpha v$ | [[#9.7 Deriving steering vectors (E.2)\|§9.7]] |
| Indirect effect (causal tracing) | probability of the correct object recovered by restoring clean $h_i^l$ into the corrupted run | [[#9.8 Causal tracing (E.3)\|§9.8]] |
| Taylor with remainder ([[#^eq-7\|eq. (7)]]) | $F(o+\Delta o) - F(o) = \nabla F^\top\Delta o + \tfrac12\Delta o^\top H F(o+\xi\Delta o)\Delta o$ | [[#9.9 Gradient detection and Integrated Gradients (E.4)\|§9.9]], [[#3.3 Gradient detection\|§3.3]] |
| Integrated Gradients ([[#^eq-8\|eq. (8)]]) | $\mathrm{IG}_k = (o - o')_k\int_0^1 \partial F(\gamma(\alpha))/\partial\gamma_k\,d\alpha$ | [[#9.9 Gradient detection and Integrated Gradients (E.4)\|§9.9]] |
| IG completeness | $\sum_k \mathrm{IG}_k = F(o) - F(o')$ | [[#9.9 Gradient detection and Integrated Gradients (E.4)\|§9.9]] |
| Linear EAP ([[#^eq-9\|eq. (9)]]) | $S(u\to v) \approx (a_u(x_{\text{clean}}) - a_u(x_{\text{corr}}))\cdot\partial R/\partial z_v\rvert_{x_{\text{clean}}}$ | [[#9.10 Edge Attribution Patching (E.5)\|§9.10]], [[#3.6 Circuit discovery\|§3.6]] |
| EAP-IG | $S_{\text{IG}} \approx \Delta a_u\cdot\frac1n\sum_{k=1}^n \partial R/\partial z_v\rvert_{x_{k/n}}$, $x_\alpha = x_{\text{corr}} + \alpha(x_{\text{clean}} - x_{\text{corr}})$ | [[#9.10 Edge Attribution Patching (E.5)\|§9.10]] |

---

## Summary

- **The pipeline ([[#1. Abstract and Introduction|§1]])**
  - Past MI reviews treat MI as observation; this survey makes it engineering: **Locate → Steer → Improve**.
  - Two gaps addressed: no clear categorization of MI methods for practice, and application coverage too general to act on.
  - Motivation: external scaling is hitting bottlenecks, so improve the model "from the inside".
  - Each method is described by its formulation, applicable objects and scope.

- **Interpretable objects ([[#2. Interpretable Objects of LLMs|§2]])**
  - The residual stream is additive ($x^{l+1} = x^l + h_{\text{attn}} + h_{\text{ffn}}$), so the output decomposes into per-component contributions.
  - MHA is a sum of heads, each writing $A^{l,h}(x^lW_V)W_O$ into the stream.
  - An FFN neuron is a key (pattern detector) and a value (what it writes); $h_{\text{ffn}} = \sum_j s_j v_j$.
  - Superposition makes neurons polysemantic; an overcomplete SAE with an $\ell_1$ penalty yields monosemantic features $f_j$ with activations $a_j$.

- **Localizing ([[#3. Localizing Methods|§3]])**
  - Magnitude: score by norm or max; training-free but data-dependent; finds outliers, influential heads, redundant layers.
  - Causal attribution: $\Delta F$ under patching or ablation; rigorous but linear in the number of objects.
  - Gradient detection: gradient × value is a first-order estimate of the ablation effect, so it screens before causal attribution.
  - Probing: trains a decoder, measures what is decodable, not what is used; needs labels and validation.
  - Logit lens: decode any state with $W_U$, zero-shot; finds name-mover and copy-suppression heads.
  - Circuit discovery: a faithful, minimal subgraph of edges, found by intervention, attribution or SAE/transcoder replacement.

- **Steering ([[#4. Steering Methods|§4]])**
  - Amplitude manipulation: zero, mean, target or scale activations at inference; same operations as attribution, different goal.
  - Targeted optimization: gradient-update only the localized weights; persistent, surgical.
  - Vector arithmetic: add $\alpha v$, with $v$ from contrastive means, SAE features or task vectors; assumes linear concepts.

- **Alignment ([[#5. Applications — Improve Alignment|§5]])**
  - Safety: ablate or retrain toxicity/hallucination components, or steer truthfulness, refusal and instruction-following directions.
  - Fairness: locate bias carriers by causal attribution (with magnitude screening), then suppress or retrain them.
  - Persona: steering vectors from opposing personas control sycophancy and traits; persona neurons are scaled or fine-tuned; probing and persona-vector similarity evaluate traits.

- **Capability ([[#6. Applications — Improve Capability|§6]])**
  - Multilingualism: scale language-specific neurons/features, or steer late layers along language directions to fix language inconsistency.
  - Knowledge: update (locate, then rewrite weights or activations), retain (suppress interference circuits, restrict updates), consolidate (merge via task vectors).
  - Reasoning: fine-tune localized arithmetic carriers, steer strategy features, and probe steps to trigger re-generation.

- **Efficiency ([[#7. Applications — Improve Efficiency|§7]])**
  - Sparse fine-tuning updates intrinsic subsets (as little as 0.1% of neurons) instead of adding PEFT modules.
  - Monitoring induction-head attention or weight norms catches phase transitions that validation loss misses.
  - Inference: prune tokens/KV cache, skip layers and MoE experts, and allocate quantization bits by layer sensitivity.

- **Limitation ([[#8. Conclusion and Limitation|§8]])**
  - Covers dense LLMs only; MoE and vision models are left to future work.
  - No standard benchmark for localization, so components cannot be shown to be causally optimal; cross-method convergence is a partial check.

- **Appendix ([[#9. Appendix|§9]])**
  - Challenges: causal localization does not scale past ~100B parameters; sparsity can hide distributed mechanisms; edits have side effects; no agreed faithfulness metric.
  - Evaluation recipe: per task, a feature, a primary metric and a side-effect check (math: accuracy vs general capability; safety: refusal vs over-refusal; editing: success vs locality and fluency).
  - MoE and multimodal models add new objects (router states, visual tokens) to which the same methods can extend.
  - Heads route (read via $W_{QK}$, write via $W_{OV}$); FFNs process; the stream is a linear sum of all component outputs.
  - SAEs: 16–128× expansion; fight dead latents (ghost gradients, resampling) and feature absorption (Top-K, JumpReLU, Gated SAEs); Gemma Scope and Llama Scope are public.
  - Logit lens findings: multilingual models think in an English-centric middle space; name-mover heads write a name, negative name movers suppress it; FFN values promote related token clusters.
  - Steering vectors: difference of class means of $x^l$, or a weighted sum of the SAE decoder directions that differ between concept-rich and neutral prompts.
  - Causal tracing: noise the subject, restore one clean state, measure recovered probability; facts sit in early MLPs at the subject token and are moved by late attention.
  - Grad × input is the first-order Taylor term (error set by curvature and step size); IG integrates the gradient from a baseline and is exact (completeness), with grad × input as its linear special case.
  - EAP scores all edges with 2 forward and 1 backward pass via $\Delta a_u\cdot\nabla_{z_v}R$; EAP-IG averages the gradient along the clean–corrupted path for faithfulness under non-linearity.
