---
type: lecture
course: '[[_CS5326 Advanced GenAI|CS5326 Advanced GenAI]]'
source:
- '[[003-Attention-1.pdf]]'
tags:
- course/cs5326
---

%% nav:start %%
[[_CS5326 Advanced GenAI|CS5326 Advanced GenAI]] · ← [[Lecture_04_Long_Context_Extension]] · → [[Lecture_06_Decoding_KV_Cache_MQA_GQA_FlashAttention]]
Concepts: [[Self-Attention]], [[Multi-Head Attention]], [[Scaled Dot-Product Attention]], [[Layer Normalization]]
%% nav:end %%

# Attention and Multi-Head Attention (CS5326 — Advanced GenAI and Agents, Lecture 5)

*Lectures 1-4 were all about position, one input to attention. This lecture opens the attention box itself: the intuition, the encoder-decoder pipeline picture by picture, then scaled dot-product and multi-head attention derived properly (including the $\sqrt d$ that YaRN (Yet another RoPE extensioN)'s temperature modified).*

## Topic Inventory

- 1. Revision: Why Self-Attention Exists
- 2. Building the Encoder Picture: Query, Key, Value
- 3. Multi-Head Attention: The First Pass
- 4. Residual Connections and Layer Normalization
- 5. The Decoder
- 6. Attention: Intuition
- 7. Scaled Dot-Product Attention, Derived
  - 7.1 Notation
  - 7.2 Steps 1-2: project, then compare
  - 7.3 Step 3: scale by $\sqrt d$
  - 7.4 The trouble with large scores
  - 7.5 Step 4: retrieve, and the full formula
  - 7.6 Numerical example
  - 7.7 The additive mask
- 8. From One Head to Many, Formally
  - 8.1 The single-head limitation
  - 8.2 Polysemy and homonymy
  - 8.3 Per-head equations and narrow heads
  - 8.4 Concatenation and the output projection
- 9. Vectorizing Self-Attention ($d_k$, $d_v$ notation)
- 10. Multi-Head Attention in $d_k$, $d_v$ Notation
- 11. Residual Connections, Revisited
- 12. Numerical Problems
- 13. Formula Summary Sheet
- 14. Cheat Sheet & Recall Aids

**Readings:** Vaswani et al. 2017 ([1706.03762](https://arxiv.org/abs/1706.03762)); Liu et al. 2018 ([1801.10198](https://arxiv.org/abs/1801.10198)); Shazeer 2019, MQA ([1911.02150](https://arxiv.org/abs/1911.02150)); Ainslie et al. 2023, GQA ([2305.13245](https://arxiv.org/abs/2305.13245)); Milakov & Gimelshein 2018, online softmax ([1805.02867](https://arxiv.org/abs/1805.02867)); Dao et al. 2022, FlashAttention ([2205.14135](https://arxiv.org/abs/2205.14135)).

---

## 1. Revision: Why Self-Attention Exists

**A need for context.** "The chicken didn't cross the street because *it* was too tired." Does *it* mean the chicken or the street? ("I hate tired streets, by the way.") Grammar allows both; only world knowledge decides. Humans do this effortlessly; machines don't. **Self-attention** lets a Transformer look at other positions for clues that give **a better encoding of a word**. The antecedent's content gets mixed into the vector at the position of "it". *(This is a Winograd schema: change "tired" to "wide" and the answer flips while the syntax stays identical.)*

**What it replaces.** RNNs (recurrent neural networks) carry context in a hidden state, so token 1 reaches token 50 only through 49 sequential updates. Self-attention gives a **direct one-hop link** between every pair and computes all positions at once. The price is $n^2$ comparisons, which the "efficient attention" readings (MQA (multi-query attention), GQA (grouped-query attention), FlashAttention (an exact attention algorithm that avoids writing the full score matrix to GPU memory)) exist to reduce.

**The four-step recipe**, on "the pizza came out of the room and smelled good". To re-encode "the":
1. Dot product of "the" with every word, **including itself**.
2. Softmax, so the weights sum to 1.
3. Scale each word's vector by its weight.
4. Sum, giving the new identity of "the".

![[L05_slide5.png|600]]

*Arcs from every word converge on the target "the" (with a self-loop). The slides repeat this for every word; the message is **"for all of the words, in parallel"**.*

Each word's computation depends only on the input embeddings, never on another word's output. So all $n$ computations are one matrix multiply, and that parallelism is why Transformers displaced RNNs.

---

## 2. Building the Encoder Picture: Query, Key, Value

§1 said each token gathers information through dot products, but not dot products between *what*. Using the raw embedding on both sides would force one fixed notion of similarity, so each token is given three learned roles.

Starting point: one-hot input over `{do, eat, not, <\s>}` → Embeddings Lookup → + Positional Encodings → $h$. A token now knows **what** it is and **where** it is, but can't yet find the tokens it needs ("I'm looking for a match!"). The matching function is **learned** ("I wonder if we could learn it!!!").

| Vector | Google analogy | Role |
|---|---|---|
| **Query** | the search you type | what this token is looking for |
| **Key** | keywords on a page | how this token advertises itself to be found |
| **Value** | the page returned | what this token delivers once chosen |

$$q_{pos} = h_{pos}W^Q,\quad k_{pos} = h_{pos}W^K,\quad v_{pos} = h_{pos}W^V$$

- **Within one head of one layer, the same $W^Q, W^K, W^V$ are used at every position.** Different heads and layers have their own. So the parameter count doesn't grow with $n$.
- A word's query is also matched with **its own key** (self-similarity).
- It all runs **in parallel** for the whole encoder.

**Steps:** scaled dot product of each query with all keys gives a **score**; softmax (which **preserves order**) gives weights summing to 1; the weighted sum of values is the **self-attention** vector:

$$x \to \text{Embed} \to +PE \to h \to \{Q,K,V\} \to Q\cdot K \to \text{Softmax} \to \textstyle\sum_j \text{Sim}(Q_i, K_j)V_j \to SA_i$$

For position 0: $\text{Softmax}(Q_0K_0, Q_0K_1, Q_0K_2)$ gives $\text{Sim}(Q_0,K_0)V_0 + \text{Sim}(Q_0,K_1)V_1 + \text{Sim}(Q_0,K_2)V_2 = SA_{00}$. Positions 1 and 2 use the same weights with $Q_1$, $Q_2$.

---

## 3. Multi-Head Attention: The First Pass

§2 gives each token exactly **one** attention distribution. Text usually has several relationships going on at once, so one distribution is not enough.

One set of Q/K/V matrices is an **attention head**. Text has multiple layers of relationships, so replicate the head: **multi-head attention**. The original paper used **8 heads**.

![[L05_slide22.png|600]]

*$d_{model} = 512$, 8 heads, $d_k = 512/8 = 64$ each. **A trade-off:** fewer heads (e.g. 2, 256 dims each) give richer per-head representations but fewer patterns; more heads (e.g. 16, 32 dims each) capture more patterns with less capacity per head. 8 × 64 was found **empirically** (Vaswani et al.).*

**Multiple parses.** Why several heads? One sentence can need several incompatible readings at once.

![[L05_slide23.png|600]]

*"Do we have to turn left from here or have we left the street behind?" Head 1 and head 2 focus on different occurrences of "left" (a direction vs a past-tense verb).*

Ambiguous sentences ("Ready to go crazy?"):
- **Time flies like an arrow.** Noun + verb, or "time flies" (the insects) that like arrows, or an imperative.
- **Old men and women.** Does "old" scope over men only, or over both?
- **I shot an elephant in my pajamas.** Who is wearing the pajamas?
- **Locked in a vault for fifty years, the owner decided to sell the jewels.** A dangling modifier: the jewels were locked, not the owner.
- **Plunging a thousand feet into the gorge, we saw Yosemite falls.** Same structure: the falls plunge, not "we".

A single softmax distribution must compromise between the readings; several heads can hold them simultaneously.

![[L05_slide24.png|600]]

*Each head has its **own learned $W_Q, W_K, W_V$**, which is why heads attend to different things. Example: head A syntactic (subject-verb: "The cats … are"), head B positional/local ("New York"), head C semantic ("doctor … hospital"). These specializations **may emerge in training; they are not designed**.*

**Concatenate and project:** $X$ → 8 heads → $Z_0..Z_7$ → Concat → $\times W^O$ → $Z$ (512-dim).

![[L05_slide25.png|650]]

*Illustrated Transformer: (1) input sentence, (2) embed, (3) split into 8 heads by multiplying $X$ (or $R$) with each head's weights, (4) attention per head, (5) concat the $Z$s and multiply by $W^O$. **Encoders other than #0 skip the embedding and start from the output of the encoder below.***

![[L05_slide26.png|600]]

*Why $W^O$: (1) a **learned combination** instead of naive concatenation; (2) **dimension preservation**, since output 512 = input 512, enabling residuals and stacking; (3) **cross-head interaction**, since the dense projection mixes information across heads.*

**Why not eight full-width heads?** Suppose each head projected to 512 instead of 64:
- **It costs 8× more:** $W_Q, W_K, W_V$ become $512\times512$, the $n^2d_k$ score cost grows 8×, and $W_O$ becomes $4096\times512$. Eight narrow heads cost the same as one wide head.
- **It is more expressive, not less.** A 64-wide head is a **rank-64 special case** of a 512-wide one ($W_QW_K^\top$ has rank ≤ 64). **Narrowness doesn't cause specialization**: heads differ because of initialization and training, and are partly redundant even at 64 dims (Voita et al. 2019; Michel et al. 2019).
- **So it's a budget decision.** Vaswani Table 3 row A varied $h \in \{1,4,8,16,32\}$ at fixed total width: 8 heads beat 1 head by **≈ 0.9 BLEU** (Bilingual Evaluation Understudy, an n-gram overlap score for machine translation) and slightly beat 16 and 32. **Full-width heads were never tested.** Row B shows $d_k < 64$ hurts.
- **$hd = D$ is a convention, not a rule.** $W_O$ maps $h\,d_v \to D$ for any $h$, and $d_k \ne d_v$ is allowed. Bhojanapalli et al. 2020 widen $d_k$ and gain at long lengths.

---

## 4. Residual Connections and Layer Normalization

![[L05_slide28.png|650]]

*The margin complaints are three separate problems. "We are big, fat numbers now… Normalize us!" means magnitudes drift. "Hey! Remind me what my position was!" means attention output is a weighted average that dilutes the token's own positional signal, and the skip connection adds the positional $h$ back. "Also, deal with the vanishing gradients" means the identity path carries gradients.*

The middle point is why this belongs in a positional-encoding course: without residuals, position would be averaged away layer by layer.

$$\boxed{\text{Output} = \text{LayerNorm}(X + \text{MultiHeadAttention}(X))}$$

![[L05_slide29.png|600]]

*The **residual (+X)** prevents vanishing gradients in deep (6+ layer) nets and lets each layer learn a **delta** (what to add), not a full transform. **LayerNorm** normalizes each token to zero mean and unit variance, stabilizes training, allows higher learning rates, and is applied **per token, not per batch** (so its statistics don't depend on the batch or on other sequences' lengths). Without both, you can't stack 12/24/96 layers.*

$$\text{LayerNorm}(x) = \gamma\odot\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}} + \beta$$

$\mu, \sigma^2$ are computed **across features for each token**. $\gamma, \beta$ are a learned scale and shift that let the layer undo the normalization if useful. $\epsilon$ prevents division by zero.

![[L05_slide31.png|550]]

*Summary, "Who am I?": the encoded input carries **word information** (embeddings: words to numbers), **positional information** (positional encodings: order), and **contextual information** (self-attention: relationships). **And we can train in parallel.***

---

## 5. The Decoder

The encoder now produces contextual representations of the input. Generating an output needs two things it lacks: a way to stop a position seeing tokens not generated yet, and a way to read the encoder's output.

A decoder has its **own self-attention** Q/K/V matrices plus a **separate set for encoder-decoder attention**.

**Masked self-attention.** Decoder self-attention is **one-directional**: only previous states and the current one exist at time $t$ (auto-regressive). It is used:
- in **decoder training** with **teacher forcing**, where the gold future is in the tensor but must be hidden;
- in both training and inference of **decoder-only models like GPT** (Generative Pre-trained Transformer, a decoder-only language model).

Without the mask, training would just copy the answer and inference would fail. *(Teacher forcing trains on gold prefixes, but inference conditions on the model's own outputs. That mismatch is **exposure bias**.)*

**Encoder-decoder attention** sits above the masked self-attention: $\text{Softmax}(Q_{D0}K_{E0}, Q_{D0}K_{E1}, Q_{D0}K_{E2})$, giving $EDA_{0j} = \sum_j \text{Sim}(Q_{D0},K_{Ej})V_{Ej}$, then add and normalize. **Q comes from the decoder; K and V come from the encoder.** The encoder's K and V are computed once and reused at every step. There is no causal mask here: every target position may see the whole source. At the very first step, decoder self-attention has a single term, $\text{Sim}(Q_{d0},K_{d0})V_{d0}$.

![[L05_slide44.png|700]]

*Full picture (targets in Urdu: تم, کھاؤ, مت, `<\s>`). The encoder stays fixed; each decoder column runs masked self-attention → encoder-decoder attention → normalize → softmax over the target vocabulary and emits one word ("Ready for the first/second/third output word?").*

At **inference** the columns are strictly sequential (column $t+1$ needs column $t$'s token), which is why KV caching matters. At **training** with teacher forcing, all columns are computed in one pass and the mask is the only thing stopping column 1 from reading column 3.

---

## 6. Attention: Intuition

The diagrams above showed the whole pipeline. The rest of the lecture ("Now, let's dive in mathematically!") derives it properly, starting from an intuition the algebra can be checked against.

**The library.** Every book has a label on its spine (a **key**) and contents (a **value**); a reader has a question (a **query**). She compares her question to every label and, **instead of picking one book, takes a little of every book, more from the best matches, and blends them.** Three steps: **compare**, **normalize**, **blend**. The soft blend is what makes this differentiable: a hard argmax has zero gradient almost everywhere.

**Why separate keys from values?** What makes a token **relevant** and what it **contributes** differ. **Keys are for being found; values are for being used.** (To resolve *it*, you search for an animate noun, but what you mix in is the content of "chicken".)

**Why three projections of the same $X$?** The raw vector is general-purpose, so each token gets **one outfit for asking, one for being asked, one for answering**. *(If $W_Q = W_K$, scores would be symmetric, $s_{ij} = s_{ji}$, which is wrong: *it* needs *chicken* more than the reverse. If $W_V = I$, attention could only return mixtures of the raw embeddings.)*

**Applied to tokens:** "The cat sat on the mat because it was tired." Understanding "**it**" needs "cat" (long-range coreference); understanding "**on**" needs the relation between "sat" and "mat" (local syntax). Two relation types in one sentence, hence multiple heads. The rest of the course follows what goes wrong when this retrieval runs at scale.

---

## 7. Scaled Dot-Product Attention, Derived

### 7.1 Notation

| Symbol | Meaning |
|---|---|
| $n$ | tokens in the sequence |
| $D$ | model width |
| $h$ | number of query heads |
| $d$ | head width, $d = D/h$ |
| $X \in \mathbb{R}^{n\times D}$ | input, one row per token |
| $Q, K, V$ | query, key, value matrices |
| $S$ / $A$ | scores **before** / weights **after** softmax |
| $O$ / $Y$ | one head's output / combined output |

### 7.2 Steps 1-2: project, then compare

$$q_i = x_iW_Q,\ k_i = x_iW_K,\ v_i = x_iW_V,\qquad W_Q, W_K, W_V \in \mathbb{R}^{D\times d}$$

Each matrix is learned and turns $D$ features into $d$ role-specific features (for one head, $d = D$). **A projection is a collection of learned weighted sums.** Interpretations such as "looking for the entity being discussed" are illustrative; nobody assigns meanings to coordinates. In batch form, $Q = XW_Q$, $K = XW_K$, $V = XW_V \in \mathbb{R}^{n\times d}$. **Q and K must share a width** (for the dot product); V can differ.

$$q_ik_j^\top = \sum_{r=1}^d q_{ir}k_{jr},\qquad QK^\top: (n\times d)(d\times n) = n\times n$$

This sums feature-wise agreements. It is a **learned compatibility score, not semantic similarity**: a pronoun can seek a noun. Entry $(i,j)$ answers "how relevant is token $j$ to what token $i$ is seeking?"

### 7.3 Step 3: scale by $\sqrt d$

$$S = \frac{QK^\top}{\sqrt d}$$

A dot product adds $d$ terms, so its magnitude grows with $d$ even when no feature is more informative. **A wider head shouldn't become more confident just because it has more coordinates.**

**Variance argument.** Assume independent coordinates with mean 0 and variance 1. Then $\mathbb E[q_{ir}k_{jr}] = 0$ and $\text{Var}(q_{ir}k_{jr}) = \mathbb E[q^2]\mathbb E[k^2] = 1$. Summing $d$ independent terms adds variances:

$$\text{Var}(q_ik_j^\top) = d\ \ (\sigma = \sqrt d),\qquad \boxed{\text{Var}\Big(\frac{q_ik_j^\top}{\sqrt d}\Big) = \frac dd = 1}$$

(Dividing by $c$ divides the variance by $c^2$.)

**Why $\sqrt d$ and not $d$:** fluctuations grow like $\sqrt d$ even though $d$ terms are added. Dividing by $d$ **overcorrects**: variance becomes $1/d$, scores become similar, softmax goes nearly uniform, and attention degenerates into averaging.

The assumptions are an **approximation**, mostly relevant at initialization. Training correlates features and changes variances, and even at initialization Q and K come from the same input. So $\sqrt d$ is a sensible fixed correction, **not a guarantee of unit variance throughout training**. (That gap is why YaRN added a second multiplier.)

### 7.4 The trouble with large scores

$$A_{ij} = \frac{e^{S_{ij}}}{\sum_{r=1}^n e^{S_{ir}}}$$

$$\text{softmax}([1,2,3]) \approx [0.090, 0.245, 0.665],\qquad \text{softmax}([8,16,24]) \approx [0.0000001, 0.000335, 0.999665]$$

The **ranking is unchanged, but the result is nearly a hard selection**. Learning also stalls:

$$\frac{\partial A_{ij}}{\partial S_{ij}} = A_{ij}(1-A_{ij})$$

This is small near 0 or 1: 0.223 at $A = 0.665$ versus $3.35\times10^{-4}$ at $0.999665$, about 665× less gradient. Scaling keeps head width alone from pushing softmax into this **saturated regime**. This was the original Transformer paper's motivation.

### 7.5 Step 4: retrieve, and the full formula

$$o_i = \sum_j A_{ij}v_j,\qquad O = AV: (n\times n)(n\times d) = n\times d$$

$$\boxed{O = \text{softmax}\Big(\frac{QK^\top}{\sqrt d}\Big)V}$$

**Softmax operates separately on each row** (each row is one query's budget over sources). Read it from the inside out: **Compare** ($QK^\top$) → **Scale** ($/\sqrt d$) → **Normalize** (softmax) → **Retrieve** ($\cdot V$).

### 7.6 Numerical example

$q = [\sqrt2, 0]$, $k_1 = [1,0]$, $k_2 = [0,1]$, $k_3 = [1,1]$, $d = 2$. Scaled scores: $[1, 0, 1]$.

$$A = \frac{[e, 1, e]}{2e+1} \approx [0.422, 0.155, 0.422]$$

With $v_1 = [2,0]$, $v_2 = [0,2]$, $v_3 = [2,2]$: $o = 0.422[2,0] + 0.155[0,2] + 0.422[2,2] = \boxed{[1.689, 1.155]}$.

The zero-score token still gets 15.5%. **A score of zero is not an instruction to ignore a token** ($e^0 = 1$). Excluding a token requires $-\infty$.

### 7.7 The additive mask

$$S = \frac{QK^\top}{\sqrt d} + M,\qquad M_{ij} = 0 \text{ (allowed)},\ -\infty \text{ (forbidden)},\ e^{-\infty} = 0$$

This excludes padding and future tokens. Adding the mask **before** softmax means the remaining weights **automatically renormalize to 1**; zeroing after softmax would leave rows summing to less than 1. (In code, $-\infty$ is a large negative number such as $-10^9$, to avoid NaN (not a number) on fully masked rows.) **Causal mask:** $M_{ij} = -\infty$ for $j > i$.

---

## 8. From One Head to Many, Formally

### 8.1 The single-head limitation

A single head gives **one distribution per query**, and that same distribution mixes **every coordinate** of the values. But a token may need several things at once: an entity, a location, a syntactic relation.

> "Let the token ask several different questions, and keep the answers separate until we combine them."

This is a coupling constraint: $o_i = \sum_jA_{ij}v_j$ uses one scalar $A_{ij}$ for all $d$ coordinates of $v_j$. Widening the head doesn't relax it; only adding softmaxes does.

**MHA** runs several attentions in parallel, each with its **own Q/K/V projections** and so its own notion of relevance (e.g. the cat, its location, grammatical relations). These are **possible learned behaviours, not predetermined job descriptions**, so for polysemous or homonymous tokens a head **may or may not** learn the distinction.

### 8.2 Polysemy and homonymy

| | Polysemy | Homonymy |
|---|---|---|
| Meaning | one word, **related** senses | different words, same form, **unrelated** |
| Examples | paper (material / newspaper / article), mouth (person / river), head (body / leader / top) | bank (money / river), bat (animal / sports), sole (only / fish / shoe; some items debatable) |
| Origin | usually one | usually different |
| Dictionary | one headword, several senses | separate (often numbered) headwords |
| Quick test | explainable by metaphor or extension | feels like pure coincidence |

A single embedding must serve every sense, so context has to disambiguate, and attention is how it does ("bank" attends to "river" or "deposit").

### 8.3 Per-head equations and narrow heads

$$Q^{(a)} = XW_Q^{(a)},\ K^{(a)} = XW_K^{(a)},\ V^{(a)} = XW_V^{(a)},\qquad W^{(a)} \in \mathbb{R}^{D\times d}$$

$$O^{(a)} = \text{softmax}\Big(\frac{Q^{(a)}K^{(a)\top}}{\sqrt d} + M\Big)V^{(a)} \in \mathbb{R}^{n\times d}$$

The superscript $(a)$ is a head index, not an exponent. **Each head has its own softmax.** The mask $M$ is shared across heads.

**Why narrower heads ($D = 512$, $h = 8$, $d = 64$)?** Every head sees the **full 512-dim input** and learns its own projection to 64. It is **not a fixed slice** of the input (the reshape in code happens *after* a dense projection). Eight full-width heads would give $8\times512 = 4096$ combined width, so we set $hd = D$ and spread a fixed budget across several retrievals.

**Why better than one wide head:** one head might retrieve $0.5v_{cat} + 0.5v_{mat}$, while two heads keep $o^{(1)} \approx v^{(1)}_{cat}$ and $o^{(2)} \approx v^{(2)}_{mat}$ separate, and the model decides how to combine them. The average is lossy (you can't recover which part came from where); separate retrievals aren't. **Different feature groups gather information with different attention patterns.**

### 8.4 Concatenation and the output projection

$$O_{concat} = \text{Concat}(O^{(1)},\dots,O^{(h)}) \in \mathbb{R}^{n\times hd} = \mathbb{R}^{n\times D}$$

**Concatenation preserves the separate answers**; averaging would merge them.

$$Y = O_{concat}W_O,\qquad W_O \in \mathbb{R}^{hd\times D} = \mathbb{R}^{D\times D}$$

$W_O$ **mixes** the features without widening them (8 × 64 = 512 → 512). An equivalent view splits $W_O$ into $h$ blocks of size $d\times D$:

$$Y = \sum_{a=1}^h O^{(a)}W_O^{(a)}$$

"Concatenate, then project" = "project each head into model space, then sum" (block matrix multiplication). The second view shows heads only meet as a sum. $W_O$ also restores width $D$ for the residual add.

$$\boxed{\text{MHA}(X) = \text{Concat}(O^{(1)},\dots,O^{(h)})W_O}$$

| Stage | Shape ($D = 512$, $h = 8$, $d = 64$) |
|---|---|
| Input | $n\times512$ |
| Q, K or V for one head | $n\times64$ |
| Attention weights for one head | $n\times n$ ← the only quadratic entry |
| Output of one head | $n\times64$ |
| Concatenated | $n\times512$ |
| Final | $n\times512$ |

---

## 9. Vectorizing Self-Attention ($d_k$, $d_v$ notation)

The same operation in Lilian Weng's notation. **Careful: here $d$ is the model width** (§7's $D$), and $d_k$, $d_v$ are head widths.

- **Query:** "here's what I'm interested in". **Key:** "this is what I have". **Value:** "if you find me interesting, here's what I'll communicate".
- **YouTube analogy:** the search text (query) is matched against titles and descriptions (keys), and the best videos (values) are returned.

From positionally encoded $X_{pos} \in \mathbb{R}^{1\times d}$:

$$k_{pos} = X_{pos}W^K \in \mathbb{R}^{1\times d_k},\quad q_{pos} = X_{pos}W^Q \in \mathbb{R}^{1\times d_k},\quad v_{pos} = X_{pos}W^V \in \mathbb{R}^{1\times d_v}$$

$d_q = d_k$ is required; for simplicity take $d_k = d_v$. A query must be compared with **all** keys ("imagine Google searching only one webpage"). Stacking gives $X \in \mathbb{R}^{n\times d}$, $Q, K \in \mathbb{R}^{n\times d_k}$, $V \in \mathbb{R}^{n\times d_v}$, and every pairwise score in one product:

$$QK^\top = \begin{bmatrix}\vec q_1\cdot\vec k_1 & \cdots & \vec q_1\cdot\vec k_n\\ \vdots & \ddots & \vdots\\ \vec q_n\cdot\vec k_1 & \cdots & \vec q_n\cdot\vec k_n\end{bmatrix} \in \mathbb{R}^{n\times n}$$

**Scores to weights:** softmax each row, as with classification logits, after dividing by $\sqrt{d_k}$ for stable gradients:

$$\text{Softmax}\Big(\frac{QK^\top}{\sqrt{d_k}}\Big) = [s_{i,j}] \in \mathbb{R}^{n\times n},\qquad \sum_j s_{pos,j} = 1$$

> [!warning] Possible source inconsistency
> The slide text says "divide by the dimensionality of the key vector", but the formula uses $\sqrt{d_k}$ (dividing by $d_k$ overcorrects, §7.3). Also note that $s_{i,j}$ here are **post**-softmax weights, which §7 called $A_{ij}$.

**Multiply by the values:**

$$H = \text{softmax}\Big(\frac{QK^\top}{\sqrt{d_k}}\Big)V \in \mathbb{R}^{n\times d_v},\qquad \vec H_i = \sum_{j=1}^n s_{i,j}\,\vec v_j$$

**One $d_v$-sized "smoothie" per token.** The same scalar $s_{i,j}$ multiplies every coordinate of $\vec v_j$ (the coupling from §8.1).

**Attention itself has no learnable parameters.** It is weighted averages of the rows of $V$, one per query, made only of matrix operations. All learning lives in $W^Q, W^K, W^V, W^O$. That is also why it works for any $n$.

---

## 10. Multi-Head Attention in $d_k$, $d_v$ Notation

$$\text{Multihead}(Q,K,V) = \text{Concat}(\text{head}_1,\dots,\text{head}_h)W^O \in \mathbb{R}^{n\times h d_v}\times\mathbb{R}^{hd_v\times d} = \mathbb{R}^{n\times d}$$
$$\text{head}_i = \text{Attention}(XW_i^Q, XW_i^K, XW_i^V) \in \mathbb{R}^{n\times d_v}$$
$$W_i^Q, W_i^K \in \mathbb{R}^{d\times d_k},\quad W_i^V \in \mathbb{R}^{d\times d_v},\quad W^O \in \mathbb{R}^{hd_v\times d}$$

**Shape check:**
- $\text{head}_i$ has the row count of $QW_i^Q$ ($n$) and the column count of $VW_i^V$ ($d_v$).
- Concatenating $h$ heads gives $n\times hd_v$.
- Multiplying by $W^O$ gives $n\times d$: $n$ queries in, $n$ responses out.
- $d_k$ is contracted away inside $QK^\top$ and never appears in the output, which is why it can differ from $d_v$.

Each head has its own learned transformations.

![[L05_slide81.png|500]]

*Lena Voita: "I saw cat on mat" enters multi-head attention, where **heads work independently**.*

**Three uses of MHA in the Transformer:**

| Use | Q from | K, V from | Mask |
|---|---|---|---|
| Encoder self-attention | encoder | encoder | padding only (bidirectional) |
| Decoder masked self-attention | decoder | decoder | causal + padding |
| Encoder-decoder attention | **decoder** | **encoder** | source padding |

All three are the **same function**, $\text{softmax}(QK^\top/\sqrt d + M)V$; only the sources and the mask differ. GPT-style models use only row 2.

---

## 11. Residual Connections, Revisited

Every sub-layer (MHA **and** FFN (feed-forward network)) is wrapped:

$$\boxed{\text{Output} = \text{LayerNorm}(x + \text{Sublayer}(x))}$$

- $x$ is the sub-layer input.
- $\text{Sublayer}(x)$ is the MHA or FFN output.
- $x + \text{Sublayer}(x)$ is the **residual connection**, which helps gradients flow and avoids vanishing or exploding gradients.

The pattern appears twice per encoder layer and three times per decoder layer. (The FFN itself and the decoder's full math come in [[Lecture_06_Decoding_KV_Cache_MQA_GQA_FlashAttention|Lecture 6]], which then asks what all of this costs at generation time.) *(This is **post-norm**. Most models since about 2020 use **pre-norm**, $x + \text{Sublayer}(\text{LN}(x))$, which trains more stably at depth.)*

---

## 12. Numerical Problems

**P1 — Scaling changes shape, not ranking.** $d = 16$, raw scores $[4, 2, 1]$. Scaled by 4: $[1, 0.5, 0.25]$, and $e$-values $2.718, 1.649, 1.284$ (sum 5.651).

$$\boxed{\text{scaled} \approx [0.481, 0.292, 0.227];\ \text{unscaled} \approx [0.844, 0.114, 0.042]}$$

Same order; without scaling the top key takes 84% instead of 48%.

**P2 — Why $\sqrt d$.** $d = 64$ gives $\text{Var} = 64$ and $\sigma = 8$. Scores 8 vs 0 give $e^8/(e^8+1) = 0.99966$, already saturated. Dividing by $\sqrt d$: $\sigma = 1$. Dividing by $d$: $\text{Var} = 1/64$, $\sigma = 0.125$, and 3-key weights ≈ [0.35, 0.33, 0.32], nearly uniform.

$$\boxed{\sigma: 8 \to 1 \ (\sqrt d) \text{ vs } 0.125 \ (d)}$$

**P3 — Full forward pass.** $q = [2,1]$; $k_1 = [2,0]$, $k_2 = [0,2]$, $k_3 = [1,1]$; $v_1 = [1,0]$, $v_2 = [0,1]$, $v_3 = [1,1]$.
- Raw scores $[4, 2, 3]$; divided by $\sqrt2$: $[2.828, 1.414, 2.121]$.
- $e$-values $16.918, 4.113, 8.342$ (sum 29.373), so $A = [0.576, 0.140, 0.284]$.
- $o = [0.576 + 0.284,\ 0.140 + 0.284]$.

$$\boxed{o \approx [0.860, 0.424]}$$

It leans toward $v_1$, and lies inside the convex hull of the values.

**P4 — Causal mask.** $S = \begin{bmatrix}1.0&2.0&0.5\\0.4&1.4&2.0\\0.2&0.7&1.2\end{bmatrix}$.
- Row 1 → $[1, -\infty, -\infty]$ gives $[1, 0, 0]$.
- Row 2 → $[0.4, 1.4, -\infty]$: $e^{0.4} = 1.492$, $e^{1.4} = 4.055$.

$$\boxed{A_1 = [1,0,0],\ A_2 = [0.269, 0.731, 0]}$$

Both rows sum to 1 automatically. Shortcut: a two-way softmax is $\sigma(1.4 - 0.4) = 0.731$.

**P5 — MHA layer cost.** $D = 512$, $h = 8$, $d = 64$, $n = 1000$.
- Q/K/V: $3\times512\times512 = 786{,}432$; $W_O$: 262,144. Total $4D^2 = 1{,}048{,}576$.
- Weights: $hn^2 = 8\times10^6$ entries ≈ 16 MB (megabytes) in fp16.
- $QK^\top$: $2hn^2d = 1.024\times10^9$ FLOPs (floating-point operations).

$$\boxed{4D^2 \approx 1.05\text{M params};\ 16\text{ MB};\ 1.02\text{ GFLOP}}$$

Parameters don't depend on $n$; activations and FLOPs grow with $n^2$.

**P6 — Full-width heads.** Each head's Q/K/V becomes $512\times512$: $8\times786{,}432 = 6{,}291{,}456$. $W_O$ becomes $4096\times512 = 2{,}097{,}152$. Total 8,388,608.

$$\boxed{= 8\times1{,}048{,}576}$$

**P7 — Gradient starvation.** $A(1-A)$: $0.665 \to 0.2228$; $0.999665 \to 3.35\times10^{-4}$. The maximum is 0.25 at $A = 0.5$.

$$\boxed{\approx 665\times \text{ less gradient; max at } A = 0.5}$$

The gradient vanishes at **both** extremes.

---

## 13. Formula Summary Sheet

| Concept | Formula |
|---|---|
| Projections | $Q = XW_Q$, $K = XW_K$, $V = XW_V$; $W \in \mathbb{R}^{D\times d}$ |
| Score | $q_ik_j^\top = \sum_r q_{ir}k_{jr}$; $QK^\top \in \mathbb{R}^{n\times n}$ |
| Scaled scores | $S = QK^\top/\sqrt d$ |
| Variance | $\text{Var}(q_ik_j^\top) = d$; after scaling = 1 |
| Softmax / derivative | $A_{ij} = e^{S_{ij}}/\sum_r e^{S_{ir}}$; $\partial A_{ij}/\partial S_{ij} = A_{ij}(1-A_{ij})$ |
| Attention | $O = \text{softmax}(QK^\top/\sqrt d)V$; $o_i = \sum_j A_{ij}v_j$ |
| Mask | $S = QK^\top/\sqrt d + M$; causal $M_{ij} = -\infty$ for $j > i$ |
| Per head | $O^{(a)} = \text{softmax}(Q^{(a)}K^{(a)\top}/\sqrt d + M)V^{(a)}$ |
| MHA | $\text{Concat}(O^{(1)},\dots,O^{(h)})W_O$ $= \sum_a O^{(a)}W_O^{(a)}$ |
| $d_k, d_v$ form | $\text{head}_i = \text{Attention}(XW_i^Q, XW_i^K, XW_i^V) \in \mathbb{R}^{n\times d_v}$; $W^O \in \mathbb{R}^{hd_v\times d}$ |
| Split | $d = D/h$ (512/8 = 64) |
| Residual + LN | $\text{LayerNorm}(x + \text{Sublayer}(x))$ |
| LayerNorm | $\gamma\odot(x-\mu)/\sqrt{\sigma^2+\epsilon} + \beta$, per token |
| Cost | params $4D^2$; weights $hn^2$; $QK^\top$ FLOPs $2hn^2d$ |

---

## 14. Cheat Sheet & Recall Aids

- **Story:** attention = compare, scale, normalize, blend. MHA = several such lookups with separate softmaxes. Residual + LayerNorm lets you stack them.
- **Mnemonic "CSNR" (Compare, Scale, Normalize, Retrieve)** plus **"Q asks, K advertises, V delivers."**
- **Mnemonic "E, MD, ED":** encoder self-attention; masked decoder self-attention; encoder-decoder attention (**Q from decoder, K/V from encoder**).
- **Anchors:** 512 / 8 / 64; 8 heads beat 1 head by ≈ 0.9 BLEU; full-width heads cost 8× and were never tested; MHA has $4D^2$ params.
- **Gotcha:** divide by $\sqrt d$, not $d$ ($d$ overcorrects toward uniform).
- **Gotcha:** a score of 0 ≠ ignore; use $-\infty$ **before** softmax (it renormalizes automatically).
- **Gotcha:** $S$ (pre-softmax) vs $A$; the §9 slides' $s_{ij}$ are post-softmax.
- **Narrow heads:** not forced specialization (rank-64 ⊂ rank-512), just a better use of a fixed budget.
- **Concat, not average:** keeps which head found what; $W_O$ mixes the heads and restores width $D$.
- **The residual carries position**, as well as gradients.
- **Attention itself is parameter-free.** Head roles **may emerge**; they are not designed.

**Self-test**
- [ ] Derive $\text{Var}(q_ik_j^\top) = d$ and name where each assumption is used.
- [ ] What goes wrong dividing by 1, $\sqrt d$, $d$?
- [ ] Hand-compute a forward pass from $q$, 3 keys, 3 values.
- [ ] Why add the mask before softmax?
- [ ] Every tensor shape for 512/8/64; which is quadratic?
- [ ] The single-head coupling limitation.
- [ ] Concat-then-project = project-then-sum.
- [ ] Q/K/V sources and mask for all three MHA uses.
