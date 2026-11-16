---
type: lecture
course: '[[_CS5326 Advanced GenAI|CS5326 Advanced GenAI]]'
source:
- '[[003-Attention-2.pdf]]'
tags:
- course/cs5326
---
l
%% nav:start %%
[[_CS5326 Advanced GenAI|CS5326 Advanced GenAI]] · ← [[Lecture_05_Attention_and_Multi_Head_Attention]] · [[Lecture_07_Residuals_and_Normalization]] →
Concepts: [[KV Cache]], [[Multi-Query Attention]], [[Grouped-Query Attention]], [[FlashAttention]]
%% nav:end %%

# Decoding, Attention Cost, KV Cache, MQA/GQA and FlashAttention (CS5326, Lecture 6)

*Continues from [[Lecture_05_Attention_and_Multi_Head_Attention]], which built scaled dot-product and multi-head attention and sketched the decoder. This lecture finishes the decoder math, then asks what attention costs at generation time and covers the three standard fixes: KV (key-value) cache, MQA (multi-query attention)/GQA, and FlashAttention (an exact attention algorithm that avoids writing the full score matrix to GPU memory).*

## Topic Inventory

- 1. The Decoder Block, Formally
  - 1.1 A few extras from the original paper
  - 1.2 Masked multi-head self-attention
  - 1.3 Cross-attention
  - 1.4 Feed-forward network and output projection
- 2. Autoregressive Generation
  - 2.1 Probability of a continuation
  - 2.2 Next-token distribution and the causal mask
  - 2.3 Teacher forcing, prefill and decode
- 3. Time Complexity of Attention
  - 3.1 Notation and the matmul cost rule
  - 3.2 Cost of each stage
  - 3.3 Total cost, causal masking, cross-attention
  - 3.4 Cost of naive generation
- 4. KV Cache
  - 4.1 Why old K and V never change
  - 4.2 What to cache (and what not to)
  - 4.3 Worked example and cost reduction
  - 4.4 Memory cost
- 5. MQA and GQA
  - 5.1 Multi-Query Attention
  - 5.2 How heads still specialize
  - 5.3 Grouped-Query Attention
  - 5.4 Dimensions and savings
- 6. FlashAttention
  - 6.1 The real bottleneck: memory traffic
  - 6.2 Tiling
  - 6.3 Online softmax
  - 6.4 Worked tiled example
  - 6.5 What it saves and what it does not
- 7. Numerical Problems
- 8. Formula Summary Sheet
- 9. Cheat Sheet & Recall Aids

---

## 1. The Decoder Block, Formally

### 1.1 A few extras from the original paper

The original Transformer trained on a vocabulary of about **37,000 tokens** with long inputs and outputs, so it **normalized often**: after positional encodings and after each self-attention sub-layer. It also added a **feed-forward network** (hidden layer) inside every encoder and decoder block to give the model more parameters for complex data.

A decoder block (repeated $N = 6$ times) has three sub-layers, each wrapped in residual + LayerNorm (layer normalization). Let $Y$ be the decoder input (shifted target embeddings + positions) and $\text{Enc}$ the final encoder output.

### 1.2 Masked multi-head self-attention

The math is the encoder's self-attention with a **mask** $M$ added to the scores before softmax:

$$
Q = YW^Q,\quad K = YW^K,\quad V = YW^V,\qquad \text{MaskedAttn}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V
$$

$$
Y' = \text{LayerNorm}\big(Y + \text{MaskedMultiHead}(Y)\big)
$$

The encoder sees all tokens at once. The decoder generates one token at a time and cannot know future tokens, so position $i$ may only attend to positions $1,\dots,i$. $M$ has $0$ on and below the diagonal and $-\infty$ above:

$$
\frac{1}{\sqrt{d_k}}QK^\top + M =
\frac{1}{\sqrt{d_k}}\begin{bmatrix} q_1\!\cdot\! k_1 & -\infty & \cdots & -\infty \\ q_2\!\cdot\! k_1 & q_2\!\cdot\! k_2 & \cdots & -\infty \\ \vdots & & \ddots & \vdots \\ q_T\!\cdot\! k_1 & q_T\!\cdot\! k_2 & \cdots & q_T\!\cdot\! k_T \end{bmatrix}
$$

![[L06_masked_self_attention.png|420]]

*Each token "looks" only at itself and earlier tokens; future tokens (red ✗) are masked before the softmax, then the gathered context updates the token's representation.*

This is the additive mask from [[Lecture_05_Attention_and_Multi_Head_Attention#7.7 The additive mask|Lecture 5 §7.7]], now applied inside the decoder.

**Why $-\infty$ gives exactly zero weight.** With $f(x_i) = e^{x_i}/\sum_j e^{x_j}$, a masked entry has numerator $e^{-\infty} = 0$, while the denominator still contains the finite unmasked terms, so it is a positive finite number. The masked weight is $0$, and the remaining weights renormalize to sum to 1 among the allowed positions. Without the mask the model would "cheat" in training by reading the answer.

### 1.3 Cross-attention (encoder-decoder attention)

Queries come from the decoder; keys and values come from the encoder:

$$
Q = Y'W^Q,\quad K_{enc} = \text{Enc}\,W^K,\quad V_{enc} = \text{Enc}\,W^V,\qquad Y'' = \text{LayerNorm}\!\left(Y' + \text{softmax}\!\left(\frac{QK_{enc}^\top}{\sqrt{d_k}}\right)V_{enc}\right)
$$

No mask is needed: the whole source sentence is known. This is how the decoder "reads" the input, e.g. when generating the Urdu word for "cat" (بلی), cross-attention focuses on "cat" in the encoder output. It aligns output with input.

### 1.4 Feed-forward network and output projection

$$
\text{FFN}(y) = \max(0,\ yW_1 + b_1)W_2 + b_2,\qquad Y''' = \text{LayerNorm}\big(Y'' + \text{FFN}(Y'')\big)
$$

Same as the encoder FFN: extra non-linearity, applied to each position independently. $Y'''$ feeds the next decoder block. After the last block, project to vocabulary size and softmax per position:

$$
\text{logits} = Y_{final}W_{vocab} + b\ \ (m \times \text{vocab\_size}),\qquad P = \text{softmax}(\text{logits})
$$

Training maximizes the probability of the correct next token with **cross-entropy**. $W_{vocab}$ is usually **tied** to the (transposed) token-embedding matrix, so input and output share one representation.

---

## 2. Autoregressive Generation

We know how tokens exchange information. How does the model write tokens that don't exist yet?

### 2.1 Probability of a continuation

An **autoregressive** model generates one token at a time: prompt "The cat" → "sat"; then "The cat sat" → "on"; and so on. (Assume each word is one token.) With $z_i$ the token at position $i$, prompt length $P$ and $T$ generated tokens, the chain rule gives:

$$
p(z_{P+1},\dots,z_{P+T}\mid z_1\dots z_P) = \prod_{s=1}^{T} p(z_{P+s}\mid z_1\dots z_{P+s-1})
$$

Probability of the continuation = probability of its first token × probability of the second given the first × ...

### 2.2 Next-token distribution and the causal mask

For a prefix of length $t$, let $r_t \in \mathbb{R}^{1\times D}$ be the final-layer representation at the last position. Then

$$
\ell_t = r_t W_{vocab} + b_{vocab},\qquad p(z_{t+1}\mid z_1\dots z_t) = \text{softmax}(\ell_t)
$$

$\ell_t$ are the **logits** (one score per vocabulary token); $W_{vocab}$ has one column per token. Pick a token (argmax or sample), append it, repeat.

The causal mask:

$$
M_{ij} = \begin{cases} 0, & j \le i \\ -\infty, & j > i \end{cases}
$$

| Query position | Key 1 | Key 2 | Key 3 | Key 4 |
|---|---|---|---|---|
| 1 | ✓ | | | |
| 2 | ✓ | ✓ | | |
| 3 | ✓ | ✓ | ✓ | |
| 4 | ✓ | ✓ | ✓ | ✓ |

### 2.3 Teacher forcing, prefill and decode

**Training is parallel.** The full target sequence is already known, so every position's prediction is computed in one forward pass; the mask hides what would be unavailable at generation time. Using ground-truth previous tokens as input is **teacher forcing**.

| Input position | Visible context | Target |
|---|---|---|
| "The" | The | cat |
| "cat" | The cat | sat |
| "sat" | The cat sat | on |

**Inference is sequential**, because the next input depends on the prediction just made. Two phases:

- **Prefill:** process the whole prompt at once (many query positions). The last position gives the distribution for the first generated token.
- **Decode:** process each generated token as it arrives, one new query position per layer per step.

The naive implementation reruns the whole growing prefix each step ("The cat" → "The cat sat" → "The cat sat on"), so "The" and "cat" pass through the same computations again and again. §3 measures how bad that is.

---

## 3. Time Complexity of Attention

### 3.1 Notation and the matmul cost rule

| Symbol | Meaning |
|---|---|
| $n$ | tokens in the sequence being processed |
| $D$ | model width |
| $h$ | number of query heads |
| $d$ | head width, $d = D/h$ |
| $X \in \mathbb{R}^{n\times D}$ | input, one row per token |
| $Q,K,V \in \mathbb{R}^{n\times d}$ | per-head queries, keys, values |
| $S$, $A$ | scores (pre-softmax), weights (post-softmax) |
| $O$, $Y$ | one head's output, combined output |

Two kinds of cost: **projection cost** (making $Q,K,V$ and applying $W^O$) and **attention interaction cost** (queries vs keys, mixing values). Analysis is per layer, per sequence, dense matmul; multiply by $L$ for $L$ layers. Parallel hardware changes wall-clock time, not the arithmetic counted.

**Matmul rule:** $A_{a\times b}B_{b\times c} = C_{a\times c}$ has $ac$ entries, each a length-$b$ dot product ($b$ mults, $b-1$ adds): total $ac(2b-1) = O(abc)$. Example: $(3\times4)(4\times5)$ gives 15 entries × 7 ops = 105 ops; $C_{11} = 1(1)+2(6)+3(11)+4(16) = 110$.

**Toy model used throughout:** $D = 8$, $h = 4$, $d = 2$, prompt "The cat", generating "sat → on → the". $x_i \in \mathbb{R}^{1\times8}$; $q_i,k_i,v_i,o_i \in \mathbb{R}^{1\times2}$. Excludes FFN and vocabulary projection.

### 3.2 Cost of each stage

**Projections.** One head: $(n\times D)(D\times d) \Rightarrow O(nDd)$. Across $h$ heads, using $hd = D$: $O(nD^2)$. $Q$, $K$, $V$ together are still $O(nD^2)$ (factor 3 dropped). The output projection $\text{Concat}(O^{(1)},\dots,O^{(h)})W^O$ is $(n\times D)(D\times D) \Rightarrow O(nD^2)$. Toy: a query projection over 3 tokens is $3\times8\times2 = 48$ multiplications per head.

**Scores.** $QK^\top$: $(n\times d)(d\times n) \Rightarrow O(n^2 d)$ per head: $n$ queries × $n$ keys × $d$ features. Across heads: $O(n^2 D)$. Scaling by $1/\sqrt d$ adds only $O(n^2)$.

**Softmax.** $n$ rows of $n$ entries (max, exp, sum, divide): $O(n^2)$ per head, $O(hn^2)$ total, absorbed by $O(n^2D)$ since $d \ge 1$.

**Value mixing.** $AV$: $(n\times n)(n\times d) \Rightarrow O(n^2 d)$ per head, $O(n^2 D)$ total. With the causal mask the first row is $[1,0,0]$ (token 1 sees only itself), and e.g. $o_{31} = a_{31}v_{11} + a_{32}v_{21} + a_{33}v_{31}$.

### 3.3 Total cost, causal masking, cross-attention

| Operation | Cost (all heads) |
|---|---|
| Q/K/V projections | $O(nD^2)$ |
| Query-key scores | $O(n^2D)$ |
| Softmax | $O(hn^2)$ |
| Weighted value sums | $O(n^2D)$ |
| Output projection | $O(nD^2)$ |

$$
\boxed{O(\underbrace{nD^2}_{\text{transform each token's features}} + \underbrace{n^2D}_{\text{exchange info between tokens}})}
$$

"Attention is quadratic" refers to the second term, but projections are real work and dominate when $D > n$ (the two terms cross at $n = D$).

**Causal masking does not remove the quadratic.** Allowed pairs: $1+2+\dots+n = n(n+1)/2 \approx n^2/2$, still $O(n^2)$. Skipping masked pairs saves a constant factor; computing the full matrix then masking saves nothing. Encoder self-attention and full-sequence causal decoder attention both cost $O(nD^2 + n^2D)$.

**Cross-attention.** $t$ decoder queries, $E$ encoder tokens: $Q \in \mathbb{R}^{t\times d}$, $K,V \in \mathbb{R}^{E\times d}$, score matrix $t\times E$.

| Operation | Dimensions | Cost |
|---|---|---|
| Decoder → queries | $(t\times D)(D\times D)$ | $O(tD^2)$ |
| Encoder → keys | $(E\times D)(D\times D)$ | $O(ED^2)$ |
| Encoder → values | $(E\times D)(D\times D)$ | $O(ED^2)$ |
| Concat → output | $(t\times D)(D\times D)$ | $O(tD^2)$ |
| $QK^\top$, $AV$ (all heads) | $(t\times d)(d\times E)$, $(t\times E)(E\times d)$ | $O(tED)$ |

$$
\text{Cross-attention: } O\big((t+E)D^2 + tED\big)
$$

(Softmax is $O(htE)$, absorbed.) Prefill over $P$ prompt tokens costs $O(PD^2 + P^2D)$.

### 3.4 Cost of naive generation

The slides walk the toy model through each step without caching. At every step, all of $Q$, $K$, $V$ ($t\times 2$ per head), the $t\times t$ score matrix, all output rows, and the $t\times 8$ by $8\times8$ output projection are **recomputed**, although the earlier rows come out identical. E.g. processing "The cat sat": $X$ is $3\times8$, $S$ is $3\times3$, and only row 3 ($o_{31} = a_{31}v_{11}+a_{32}v_{21}+a_{33}v_{31}$) is new.

One full-prefix evaluation at length $t$ costs $O(tD^2 + t^2D)$. Summing over generation:

$$
C = O\!\left(\sum_{t=P}^{P+T-1}(tD^2 + t^2D)\right),\qquad \sum_{t=1}^{T} t = O(T^2),\ \ \sum_{t=1}^{T} t^2 = O(T^3)
$$

$$
\text{Fixed } P:\quad \boxed{C_{\text{no cache}} = O(T^2D^2 + T^3D)}
$$

The **cubic** term comes from recomputing a growing square matrix at every step.

---

## 4. KV Cache

The naive loop recomputes things that never change. The KV cache stores them.

![[L06_kv_cache.png|700]]

*Left: without a cache, Q/K/V for all 5 tokens are recomputed to produce $o_5$. Middle: prefill computes and stores K, V once; each decode step computes only $q_5,k_5,v_5$, attends over cached + new K/V, then appends $k_5,v_5$. Right: it trades memory for speed; the cache grows with context and still has to be read every step.*

### 4.1 Why old K and V never change

In a causal Transformer, **appending a future token does not change any earlier token's representation**. When "on" is appended, "The" still attends only to "The", "cat" to "The cat", "sat" to "The cat sat". Their allowed inputs are the same, so their outputs are the same.

This holds at every layer by induction: at layer 1 the earlier tokens and positions are unchanged and see the same prefix, so their outputs are unchanged; layer 2 then receives the same inputs, and so on. (Assumes deterministic inference, fixed parameters, and the same position convention for existing tokens.) So earlier keys and values are fixed: compute once, store, reuse. The same reuse happens at each attention layer.

### 4.2 What to cache (and what not to)

For one head the new output needs only:

$$
o_t = \text{softmax}\!\left(\frac{q_t K_{1:t}^\top}{\sqrt d}\right)V_{1:t}
$$

- **Keys** decide which positions matter; **values** carry the information. Cache both.
- **Old queries:** not cached. They were used to produce old outputs and do not appear in the new output.
- **Old attention weights:** not cached. $q_t k_j^\top \ne q_{t-1}k_j^\top$ in general, so each new query computes its own scores. The cache keeps reusable token information; each new token decides how to use it.

**One decode step at one layer.** With $K_{cache}, V_{cache} \in \mathbb{R}^{(t-1)\times d}$:

$$
q_t = x_tW^Q,\ k_t = x_tW^K,\ v_t = x_tW^V;\qquad K_{cache} \leftarrow \begin{bmatrix}K_{cache}\\ k_t\end{bmatrix},\ V_{cache} \leftarrow \begin{bmatrix}V_{cache}\\ v_t\end{bmatrix};\qquad o_t = \text{softmax}\!\left(\frac{q_tK_{cache}^\top}{\sqrt d}\right)V_{cache}
$$

Repeat per head, combine, continue through the layer. In ordinary **training** the KV cache is not used: all K, V are computed in parallel and kept only as part of the forward pass for backprop. It is an **inference** optimization.

### 4.3 Worked example and cost reduction

After prefilling "The cat", keep $K_{cache}, V_{cache} \in \mathbb{R}^{2\times2}$. To process "sat": project only $x_3$ ($1\times8$ by $8\times2$) into $q_3, k_3, v_3$; append; compute one score row $[s_{31}, s_{32}, s_{33}] = q_3K_{cache}^\top/\sqrt2$ (no mask needed, all keys are in the past); then one output row $o_3 = \text{softmax}(\cdot)\,V_{cache}$. This is **the same expression as without the cache**; the earlier rows simply are not recomputed. The output projection shrinks too: concatenate only the new row's 4 head outputs ($1\times8$) and multiply by $W^O$ ($8\times8$) instead of $4\times8$.

| Per head | Without cache | With cache |
|---|---|---|
| Each Q/K/V projection | $(t\times D)(D\times d)$: $O(tDd)$ | $(1\times D)(D\times d)$: $O(Dd)$ |
| Scores | $(t\times d)(d\times t)$: $O(t^2d)$ | $(1\times d)(d\times t)$: $O(td)$ |
| Softmax | $O(t^2)$ | $O(t)$ |
| Value mixing | $(t\times t)(t\times d)$: $O(t^2d)$ | $(1\times t)(t\times d)$: $O(td)$ |
| **Total** | $O(tDd + t^2d)$ | $O(Dd + td)$ |

Across heads, including $W^O$: $O(tD^2 + t^2D) \to O(D^2 + tD)$ per step.

Toy count for generating sat, on, the (per head, dense scores including prefill):

| Prediction | Projected rows (no cache) | Projected rows (cache) | Score entries (no cache) | Score entries (cache) |
|---|---|---|---|---|
| sat | 2 | 2 | 4 | 4 |
| on | 3 | 1 | 9 | 3 |
| the | 4 | 1 | 16 | 4 |
| **Total** | **9** | **4** | **29** | **11** |

Skipping masked pairs gives 19 vs 10; same conclusion.

$$
C_{\text{cache}} = O\!\left(PD^2 + P^2D + \sum_{t=P+1}^{P+T-1}(D^2 + tD)\right)\ \xrightarrow{\text{fixed } P}\ \boxed{O(TD^2 + T^2D)}
$$

vs $O(T^2D^2 + T^3D)$ without. The arithmetic saving is real; actual speedup also depends on hardware utilization and memory traffic.

### 4.4 Memory cost

Toy: $t = 4$, $h = 4$, $d = 2$. Each token stores 4 key vectors and 4 value vectors of size 2:

$$
\underbrace{2}_{K,V} \times \underbrace{4}_{\text{tokens}} \times \underbrace{4}_{\text{heads}} \times \underbrace{2}_{\text{dims}} = 64 \text{ numbers per layer},\qquad \boxed{\text{MHA cache} = 2thd = 2tD}
$$

We stopped recomputing, but every new query still has to **read** all these numbers. Could the query heads share some of this memory?

---

## 5. MQA and GQA

### 5.1 Multi-Query Attention

In MHA each head $i$ has its own $Q_i = XW_i^Q$, $K_i = XW_i^K$, $V_i = XW_i^V$, so 32 heads means 32 sets of cached K/V. Queries are not cached, so they add no storage. Decoding then suffers from a **large KV cache** and, often worse, **huge memory bandwidth**: every new token must fetch all cached K/V.

**MQA** asks why every query head needs its own K and V. Keep $h$ different queries, share one key and one value:

$$
Q_1, Q_2, \dots, Q_h \quad\text{but}\quad K_1 = \dots = K_h = K,\ \ V_1 = \dots = V_h = V
$$

Per token: $h$ queries $q_t^{(1)},\dots,q_t^{(h)}$ but one $k_t$ and one $v_t$. Cache one K/V set instead of $h$: with 32 heads, about 32× smaller. The price: all heads see the same K/V representation, which can reduce quality. The shared projections are **trained as part of the MQA architecture**. You do not get them by deleting three heads from a trained MHA model.

In the toy model (position 4): four different $W^{Q,a}$ produce $q_4^{a}$ for $a = 1..4$, but the key projection and value projection happen **once**. Each head computes its own scores $s_{4j}^{a} = q_4^{a}K^\top/\sqrt2$ against the **same** $2\times4$ shared key matrix and mixes the **same** $4\times2$ shared value matrix. Queries, scores, weights and outputs carry a head index; shared keys and values do not. Outputs are concatenated and projected with $W^O$ as before.

### 5.2 How heads still specialize

MHA lets each head define its own "what am I looking for?" ($Q$), "what do I expose for matching?" ($K$), and "what do I contribute if selected?" ($V$). With shared K/V, can heads still specialize in syntax, semantics, coreference? **Yes, because the queries still differ.**

The shared key for "bank" can hold many properties at once: $k_{bank} = [\text{nounness}, \text{financial meaning}, \text{location}, \text{syntax}, \text{semantic role}, \dots]$. A syntactic head's $q^{(1)}$ aligns with the syntactic components, a semantic head's $q^{(2)}$ with the semantic ones. Since $q^{(1)} \ne q^{(2)}$, $q^{(1)}\cdot k_j$ and $q^{(2)}\cdot k_j$ give different patterns: head 1 bank → preceding adjective, head 2 bank → money, head 3 bank → river.

Outputs also differ: $o^{(i)} = \sum_j \alpha_{ij}v_j$ with $\alpha_{ij} = \text{softmax}(q^{(i)}k_j^\top)$. Different queries give different weights, so different mixtures of the same values. Two-key illustration with $K = I$:

$$
\tfrac{1}{\sqrt2}[1\ 0]\begin{bmatrix}1&0\\0&1\end{bmatrix} = [1/\sqrt2\ \ 0],\qquad \tfrac{1}{\sqrt2}[0\ 1]\begin{bmatrix}1&0\\0&1\end{bmatrix} = [0\ \ 1/\sqrt2]
$$

Same keys, opposite preferences. *Library analogy:* MHA gives each researcher a different question, a differently indexed library and differently encoded books. MQA gives a different question but the same library and books; they still reach different conclusions because they pull different books.

**Limitation.** In MHA, a syntax head learns $W_K^{syntax}, W_V^{syntax}$ and a semantic head learns $W_K^{semantic}, W_V^{semantic}$, so the same word is literally represented differently per head. MQA keeps specialization only through $Q$ and attention patterns, and gives up **head-specific K/V subspaces**. The single K/V must be rich enough to serve every head.

### 5.3 Grouped-Query Attention

**GQA** is the compromise between MHA and MQA: let groups of query heads share one K/V pair. With 32 query heads: MHA $= 32Q, 32K, 32V$; MQA $= 32Q, 1K, 1V$; GQA might use $32Q, 8K, 8V$ (each K/V shared by 4 query heads):

$$
\underbrace{Q_1,Q_2,Q_3,Q_4}_{\text{group 1}} \to K_1,V_1,\qquad \underbrace{Q_5,\dots,Q_8}_{\text{group 2}} \to K_2,V_2,\ \dots
$$

Related heads (syntax-oriented, semantic-oriented, lexical-oriented) can keep somewhat different K/V, while the cache still shrinks 4×. GQA often keeps most of MHA's quality with much of MQA's inference speed, which is why large LLMs (large language models) use it.

- **MHA:** every head gets its own worldview. Best flexibility, biggest cache.
- **MQA:** every head asks its own question of one shared worldview. Smallest cache, least flexibility.
- **GQA:** groups of related heads share a worldview. Middle ground.

Toy GQA: $h_{kv} = 2$; query heads 1, 2 use group 1, heads 3, 4 use group 2. Each group computes its own $k_4^{g}, v_4^{g}$ from $W^{K,g}, W^{V,g}$ ($8\times2$), giving caches $K^{1}, K^{2}, V^{1}, V^{2}$, each $4\times2$. Heads 1, 2 score against $K^{1}$ and mix $V^{1}$; heads 3, 4 use $K^{2}, V^{2}$. We still get four 2-feature outputs, concatenated and projected as before.

![[L06_mha_mqa_gqa.png|700]]

*MHA: each $q_i$ has its own $k_i, v_i$. MQA: all 8 query heads hit one shared $k$ and $v$. GQA ($h = 8$, $h_{kv} = 2$): heads 1-4 share $k_1, v_1$, heads 5-8 share $k_2, v_2$. Per-head width $d = D/h$ is the same in all three.*

### 5.4 Dimensions and savings

The **head size does not change**; only the number of K/V heads does. With $D = 4096$, $h = 32$: $d = 128$.

| | $Q$ | $K$ | $V$ | Example ($D{=}4096$) |
|---|---|---|---|---|
| MHA | $h\times d$ | $h\times d$ | $h\times d$ | $K: 32\times128$ |
| GQA | $h\times d$ | $h_{kv}\times d$ | $h_{kv}\times d$ | $K: 8\times128$ |
| MQA | $h\times d$ | $1\times d$ | $1\times d$ | $K: 1\times128$ |

So in MQA the shared K/V are dimension $d$, **not** $D$: per token $Q: D$, $K: d$, $V: d$.

**Q: If MQA's value is only 128-dim, how does the output get back to $D = 4096$?** All 32 query heads attend to the same $V$ with different weights, so each produces its own $o_i \in \mathbb{R}^{128}$. Concatenating gives $o_1 \| \dots \| o_{32} \in \mathbb{R}^{32\times128 = 4096}$. MQA shrinks what is **stored** as K/V, not the number or width of head outputs.

**Cache size** (per layer, numbers):

$$
\text{MHA: } 2thd \qquad \text{GQA: } 2th_{kv}d \qquad \text{MQA: } 2td
$$

Toy ($t=4, h=4, d=2$): MHA 64, GQA ($h_{kv}=2$) 32, MQA 16.

**Exact per-new-token work** (one K head: $(1\times8)(8\times2) = 16$ mults):

| Quantity | MHA | MQA | GQA |
|---|---|---|---|
| Query heads | 4 | 4 | 4 |
| KV heads | 4 | 1 | 2 |
| Cached K/V numbers | 64 | 16 | 32 |
| Q-projection mults | 64 | 64 | 64 |
| K-projection mults | 64 | 16 | 32 |
| V-projection mults | 64 | 16 | 32 |
| Output-projection mults | 64 | 64 | 64 |
| **Total projection** | **256** | **160** | **192** |
| Score mults | 32 | 32 | 32 |
| Value-mixing mults | 32 | 32 | 32 |

Projection work per new token $\propto \underbrace{D^2}_{Q} + \underbrace{Dh_{kv}d}_{K} + \underbrace{Dh_{kv}d}_{V} + \underbrace{D^2}_{W^O}$. Scores still need $h\times t\times d = tD$ multiplications, since every query head still attends. All three keep the bound $O(D^2 + tD)$. K/V sharing changes **constants, memory and flexibility**: it cuts K/V projection work, cache storage, and cache traffic (if the kernel exploits sharing). It does not remove per-query-head computation.

---

## 6. FlashAttention

KV cache and MQA/GQA shrink what decoding stores and recomputes. The remaining cost is the full $n\times n$ attention in training and prefill.

### 6.1 The real bottleneck: memory traffic

For $n$ tokens, $Q,K,V \in \mathbb{R}^{n\times d}$ but $S, A \in \mathbb{R}^{n\times n}$. The first instinct, "attention is slow because $n^2$ arithmetic is expensive", is true but incomplete. **On GPUs (graphics processing units), moving these huge matrices through memory can matter as much as the arithmetic.**

GPU hierarchy: Registers → L1/shared SRAM (static random-access memory, the small fast on-chip memory) → L2 → HBM (high-bandwidth memory, the GPU's large main memory). Moving right, capacity grows and access gets slower. Analogy: an **SM** (Streaming Multiprocessor, one compute block of an NVIDIA GPU) is a worker station, **SRAM** is a tiny desk beside it, **HBM** is a huge warehouse.

- **VRAM** (video RAM, i.e. GPU memory) is the GPU's main off-chip memory. **HBM** (High Bandwidth Memory) is one kind of VRAM ($\text{HBM} \subset \text{VRAM}$); some GPUs use GDDR6/6X (graphics DDR memory: cheaper, with lower bandwidth than HBM) instead (H100: HBM; RTX 4090: GDDR6X).
- **SRAM** = Static RAM. On NVIDIA GPUs it is the unified per-SM pool that implements L1 cache and programmer-controlled shared memory. L2 is on-chip too, but much larger and shared by the whole GPU.

| GPU | Main memory | L1/shared SRAM per SM | Max shared per SM | L2 |
|---|---|---|---|---|
| A100 | 80 GB HBM2e | 192 KB | 164 KB | 40 MB |
| H100 SXM | 80 GB HBM3 | 256 KB | 228 KB | 50 MB |
| H200 | 141 GB HBM3e | ~256 KB | 228 KB | 50 MB |
| B200 | 180 GB HBM3e | 256 KB | 228 KB | 126 MB |
| RTX 4090 | 24 GB GDDR6X | 128 KB | configurable | 72 MB |

H100: $80\,\text{GB} / 256\,\text{KB} = 80\times1024\times1024/256 \approx 327{,}680\times$ more capacity in HBM, but SRAM is far faster and next to the arithmetic units. A full $n\times n$ matrix cannot fit in SRAM, so FlashAttention loads small tiles instead: HBM → tile into SRAM → $Q_{tile}K_{tile}^\top$ → online softmax → $O_{tile}$, then the next tile.

**Standard attention** does: $QK^\top$ → write huge $S$ to HBM → read back → softmax → write huge $A$ → read again → $AV$. Huge intermediates keep moving between warehouse and workers.

![[L06_flashattention.png|700]]

*Left: standard attention writes and reads the $N\times N$ score and softmax matrices to HBM. Right: FlashAttention streams $Q$, $K$, $V$ tiles into SRAM, computes partial scores, updates a running max/sum (online softmax), accumulates the output, and never materializes the $N\times N$ matrix. Same exact result, far less HBM traffic, still $O(N^2)$.*

### 6.2 Tiling

**Never materialize the complete $n\times n$ matrix in HBM.** Split into blocks $Q_i, K_j, V_j$ small enough for SRAM, compute $S_{ij} = Q_iK_j^\top$, use it at once to update the softmax and output, then discard it:

$$
\text{compute all} \to \text{store all} \to \text{reload all} \quad\Longrightarrow\quad \text{load tile} \to \text{compute} \to \text{consume} \to \text{discard}
$$

This is what **IO (input/output, i.e. memory reads and writes)-aware attention** means.

### 6.3 Online softmax

**The obstacle:** $a_i = e^{s_i}/\sum_j e^{s_j}$ needs every score. How can block 1 be finalized before later blocks are seen?

**The fix:** the denominator $Z = \sum_j e^{s_j}$ does not depend on $i$, so pull it out:

$$
O = \sum_i \frac{e^{s_i}}{\sum_j e^{s_j}}v_i = \frac{1}{Z}\sum_i e^{s_i}v_i = \frac{\mathcal{A}}{Z},\qquad \mathcal{A} = \sum_i e^{s_i}v_i,\ \ Z = \sum_j e^{s_j}
$$

Both numerator and denominator are **sums**, and sums split across blocks: $\mathcal{A} = \mathcal{A}_1 + \mathcal{A}_2$, $Z = Z_1 + Z_2$, so

$$
\boxed{O = \frac{\mathcal{A}_1 + \mathcal{A}_2}{Z_1 + Z_2}}
$$

which is exactly the all-at-once result. **Softmax probabilities are not accumulated; the numerator and denominator are.** That is why online softmax is exact, not approximate.

Mental model. Wrong: "compute a partial softmax, apply it to $V$, and hope to fix it later." Right: keep a running weighted average plus enough information (how much probability mass supports it) to renormalize when new scores arrive: $O_{new} = \alpha\,O_{old} + \beta\,O_{block}$ with $\alpha + \beta = 1$. Old contributions are reweighted exactly without storing old scores.

**Concrete example.** Scores $[2, 1, 4, 3]$, scalar values $[10, 20, 30, 40]$. Block 1 $= [2, 1]$: local softmax $[0.731, 0.269]$. FlashAttention does **not** say token 1 gets 73.1% of the final attention. It keeps $Z_1 = e^2 + e^1$ and $\mathcal{A}_1 = 10e^2 + 20e^1$, i.e. the block's attention **mass**, not just its internal proportions. Block 2 $= [4, 3]$: $Z_2 = e^4 + e^3$, $\mathcal{A}_2 = 30e^4 + 40e^3$. Then

$$
O = \frac{10e^2 + 20e^1 + 30e^4 + 40e^3}{e^2 + e^1 + e^4 + e^3} \approx 30.31
$$

which is the ordinary full softmax. What happened to 73.1% / 26.9%? The full softmax is $[0.0871, 0.0321, 0.6439, 0.2369]$. Inside block 1, $0.0871/(0.0871+0.0321) \approx 0.731$ and $0.0321/0.1192 \approx 0.269$: the ratio $e^2/e^1 = e$ was right all along. What was unknown was the **block's share** of total mass, $0.0871 + 0.0321 = 11.92\%$, so the final weights are $11.92\% \times 73.1\% = 8.71\%$ and $11.92\% \times 26.9\% = 3.21\%$. Because $\sum e^{s_i}v_i$ is accumulated instead of committed probabilities, the values get rescaled into exactly the right proportions.

**Numerical stability.** Large $e^{s_i}$ overflow, so softmax uses $p_i = e^{s_i - m}/\sum_j e^{s_j - m}$ with $m = \max_j s_j$ (equal to plain softmax, since $e^{-m}$ cancels). FlashAttention tracks $m$ online. If a later block raises the max from $m_{old}$ to $m_{new}$, old accumulators are rescaled by $e^{m_{old} - m_{new}}$; no old scores are revisited. **Blockwise $\mathcal{A}, Z$ accumulation gives exactness; the running max gives numerical safety.**

Per query $i$, maintain:

$$
m = \text{largest score seen},\quad \ell = \sum_{\text{seen } j} e^{s_{ij} - m},\quad u_c = \sum_{\text{seen } j} e^{s_{ij} - m}v_{jc}
$$

Merging an old and new block: $m' = \max(m_{old}, m_{new})$, $\beta_{old} = e^{m_{old} - m'}$, $\beta_{new} = e^{m_{new} - m'}$,

$$
\ell' = \beta_{old}\ell_{old} + \beta_{new}\ell_{new},\qquad u_c' = \beta_{old}u_{old,c} + \beta_{new}u_{new,c},\qquad o_{ic} = u_c/\ell
$$

The rescale works because $e^{s - m_{old}}\cdot e^{m_{old} - m'} = e^{s - m'}$: both blocks are put on the same max. Only $m$, $\ell$ and $u$ are kept, not individual attention weights.

### 6.4 Worked tiled example

Back to prefill/training: all four positions of "The cat sat on" together, one head, $d = 2$. With $\tfrac{1}{\sqrt2}QK^\top$ and

$$
Q = \sqrt2\begin{bmatrix}1&0\\0&1\\1&1\\1&0\end{bmatrix},\quad K = \begin{bmatrix}1&0\\2&1\\3&0\\4&1\end{bmatrix},\quad V = \begin{bmatrix}1&0\\0&1\\1&1\\2&1\end{bmatrix}
\quad\Rightarrow\quad S_{masked} = \begin{bmatrix}1&-\infty&-\infty&-\infty\\0&1&-\infty&-\infty\\1&3&3&-\infty\\1&2&3&4\end{bmatrix}
$$

> [!note] Reconstructed $Q$
> The $Q$ entries were not legible in the text extraction; this $Q$ is back-solved from the slide's score matrix and reproduces it exactly.

Goal: the same $4\times2$ output without storing the $4\times4$ $S$ or $A$ (in general $n\times n$ per head). Use $2\times2$ tiles.

**Tile (queries 1-2, keys 1-2).** Scores $\begin{bmatrix}1&2\\0&1\end{bmatrix}$, mask → $\begin{bmatrix}1&-\infty\\0&1\end{bmatrix}$. Row maxima 1, 1; exponentiate → $\begin{bmatrix}1&0\\e^{-1}&1\end{bmatrix}$ (unnormalized). Times $V_{1:2}$ → $u = \begin{bmatrix}1&0\\e^{-1}&1\end{bmatrix}$, $\ell_1 = 1$, $\ell_2 = 1 + e^{-1}$. The tile (queries 1-2, keys 3-4) is fully masked and **skipped**. Normalize: $o_1 = [1, 0]$, $o_2 = [e^{-1}, 1]/(1+e^{-1}) \approx [0.2689, 0.7311]$. Done.

**Tile (queries 3-4, keys 1-2).** Scores $\begin{bmatrix}1&3\\1&2\end{bmatrix}$, all allowed. $m_3 = 3$, $m_4 = 2$; exponentiate → $\begin{bmatrix}e^{-2}&1\\e^{-1}&1\end{bmatrix}$; times $V_{1:2}$ → $\begin{bmatrix}e^{-2}&1\\e^{-1}&1\end{bmatrix}$.

| Query | $m$ | $\ell$ | $u_1$ | $u_2$ |
|---|---|---|---|---|
| 3 | 3 | $1 + e^{-2}$ | $e^{-2}$ | 1 |
| 4 | 2 | $1 + e^{-1}$ | $e^{-1}$ | 1 |

Incomplete: these queries can still see later keys. Keep the state, discard the tile.

**Tile (queries 3-4, keys 3-4).** Scores $\begin{bmatrix}3&5\\3&4\end{bmatrix}$, mask (3 can't see 4) → $\begin{bmatrix}3&-\infty\\3&4\end{bmatrix}$. $m_{new,3} = 3$, $m_{new,4} = 4$; exponentiate → $\begin{bmatrix}1&0\\e^{-1}&1\end{bmatrix}$; times $V_{3:4}$ → $\begin{bmatrix}1&1\\e^{-1}+2&e^{-1}+1\end{bmatrix}$; $\ell_{new,3} = 1$, $\ell_{new,4} = e^{-1} + 1$.

**Merge query 3.** Max stays 3, so $\beta_{old} = \beta_{new} = 1$: $\ell_3 = 2 + e^{-2}$, $u = [1 + e^{-2},\ 2]$, $o_3 = [1+e^{-2}, 2]/(2+e^{-2}) \approx [0.5317, 0.9366]$.

**Merge query 4.** Max $2 \to 4$: $\beta_{old} = e^{-2}$, $\beta_{new} = 1$. Old $u$ rescaled: $e^{-2}[e^{-1}, 1] = [e^{-3}, e^{-2}]$. Add new: $u = [e^{-3} + e^{-1} + 2,\ e^{-2} + e^{-1} + 1]$, $\ell_4 = e^{-3} + e^{-2} + e^{-1} + 1$:

$$
o_4 = \left[\frac{e^{-3}+e^{-1}+2}{e^{-3}+e^{-2}+e^{-1}+1},\ \frac{e^{-2}+e^{-1}+1}{e^{-3}+e^{-2}+e^{-1}+1}\right] \approx [1.5568,\ 0.9679]
$$

**Check against ordinary attention:**

$$
A \approx \begin{bmatrix}1&0&0&0\\0.2689&0.7311&0&0\\0.0634&0.4683&0.4683&0\\0.0321&0.0871&0.2369&0.6439\end{bmatrix},\qquad AV \approx \begin{bmatrix}1&0\\0.2689&0.7311\\0.5317&0.9366\\1.5568&0.9679\end{bmatrix}
$$

Matches up to rounding, without ever storing $S$ or $A$.

### 6.5 What it saves and what it does not

In practice score tiles live in fast on-chip memory. The kernel (1) loads Q/K/V blocks, (2) computes a score tile, (3) scales and masks, (4) updates per-query $m$, $\ell$, $u$, (5) discards the tile, (6) writes finished outputs. In training, backprop through softmax needs the attention weights $A$. Storing them would bring back the $n\times n$ matrix FlashAttention avoided, so the kernel keeps only the per-row $m$ and $\ell$ and **recomputes the needed tiles in the backward pass**. It does extra arithmetic but far less memory traffic, which is the better trade on a GPU.

**Saves:** the $n\times n$ matrix is never materialized in HBM, giving much lower intermediate memory, far less HBM traffic, better SRAM use, and higher **arithmetic intensity** (computation / bytes moved), so attention is much faster in practice.

**Does not solve:** attention arithmetic is still $O(n^2D)$ ($O(N^2d)$ per head). Every query still meets every allowed key. FlashAttention fixes the **implementation (memory) bottleneck, not the quadratic scaling**.

| Property | Materialized attention | FlashAttention |
|---|---|---|
| Q/K/V and output storage | Linear in $n$ | Linear in $n$ |
| Full score/weight matrices in GPU main memory | $O(hn^2)$ entries | Not materialized |
| Temporary scores | Potentially full matrices | Small tiles |
| Result | Standard | Same, up to floating-point differences |

**Where it helps most:** training and prefill, where many queries attend to many keys. Cached single-token decoding already has only one score row ($q_t$ against $K_{1:t}, V_{1:t}$) per head, so there is no large $n\times n$ matrix to avoid. That step is dominated by *reading the KV cache*, which is exactly the traffic MQA/GQA shrink.

**How the pieces fit:**

| Technique | What it attacks |
|---|---|
| KV cache | Keeps earlier K/V rows; only one new query/output row per step |
| MQA | All query heads share one K/V pair, so less to store and read |
| GQA | Query heads share K/V within groups |
| FlashAttention | Score tiles update running outputs; no full attention matrix |

They are complementary: a model can define K/V sharing with GQA, keep K/V in a cache during generation, and run many-query attention with FlashAttention. To attack the remaining $O(N^2)$ itself you have to cut the number of interactions: sliding-window/local attention, sparse attention, or different long-sequence architectures.

---

## 7. Numerical Problems

### Problem 1: KV cache size of a real model

$D = 4096$, $h = 32$, $L = 32$ layers, context $t = 4096$, fp16 (2 bytes). Find the KV cache for MHA, GQA ($h_{kv} = 8$) and MQA.

$d = 4096/32 = 128$. MHA per layer: $2thd = 2\cdot4096\cdot32\cdot128 = 33{,}554{,}432$ numbers; × 32 layers × 2 bytes $= 2^{31}$ bytes. GQA divides by $h/h_{kv} = 4$, MQA by 32.

$$
\boxed{\text{MHA} = 2\ \text{GiB},\quad \text{GQA} = 512\ \text{MiB},\quad \text{MQA} = 64\ \text{MiB}}
$$

This is per sequence, so batching multiplies it. That is why cache size, not FLOPs (floating-point operations), often limits decoding.

### Problem 2: Projections vs interactions

$D = 4096$. Which term of $O(nD^2 + n^2D)$ dominates at $n = 1024$ and at $n = 32{,}768$?

The ratio $n^2D / nD^2 = n/D$. At $n = 1024$: $0.25$, so projections are 4× the interaction work. At $n = 32{,}768$: $8$, so interactions dominate 8×.

$$
\boxed{\text{crossover at } n = D;\ \ n{=}1024: \text{projections};\ \ n{=}32768: \text{attention}}
$$

### Problem 3: Cached vs uncached generation (toy)

Prompt of $P = 2$ tokens, generate 3 tokens, per head. Count projected rows and dense score entries.

Without cache the prefix length is 2, 3, 4: rows $2 + 3 + 4 = 9$, scores $4 + 9 + 16 = 29$. With cache, prefill is 2 rows / 4 scores, then 1 row + 3 scores, then 1 row + 4 scores.

$$
\boxed{\text{rows: } 9 \to 4,\quad \text{scores: } 29 \to 11}
$$

Consistent with the $O(T^3D) \to O(T^2D)$ reduction.

### Problem 4: Online softmax with a running max

Scores $[2, 1, 4, 3]$, values $[10, 20, 30, 40]$, blocks of 2. Compute $O$ with the $m, \ell, u$ recurrence.

Block 1: $m = 2$, $\ell = 1 + e^{-1} = 1.3679$, $u = 10 + 20e^{-1} = 17.358$. Block 2: $m = 4$, $\ell = 1.3679$, $u = 30 + 40e^{-1} = 44.715$. Merge with $m' = 4$: $\beta_{old} = e^{-2} = 0.1353$, $\beta_{new} = 1$. So $\ell' = 0.1851 + 1.3679 = 1.5530$ and $u' = 2.349 + 44.715 = 47.064$.

$$
\boxed{O = 47.064 / 1.5530 \approx 30.31}
$$

Check: $0.0871(10) + 0.0321(20) + 0.6439(30) + 0.2369(40) = 30.31$. ✓ Close to 30 because score 4 dominates.

### Problem 5: GQA dimensions and work

$D = 4096$, $h = 32$, $h_{kv} = 8$. Per token, how many K numbers are cached per layer, how many query heads share each K/V head, and what is the concatenated output width?

$d = 128$; K per token $= h_{kv}d = 1024$ (vs 4096 in MHA); group size $= h/h_{kv} = 4$; output $= 32 \times 128$.

$$
\boxed{1024\ \text{K numbers/token},\ \ 4\ \text{heads per group},\ \ \text{output } 4096}
$$

### Problem 6: Causal pairs

For $n = 1000$, how many query-key pairs are allowed under a causal mask, as a fraction of $n^2$?

$n(n+1)/2 = 500{,}500$, i.e. about $50.05\%$ of $10^6$.

$$
\boxed{500{,}500\ (\approx 0.5\,n^2)}
$$

Still $O(n^2)$: masking halves the work at most.

---

## 8. Formula Summary Sheet

| Concept | Formula |
|---|---|
| Masked attention | $\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V$, $M_{ij} = 0$ if $j \le i$, else $-\infty$ |
| Decoder sub-layers | $Y' = \text{LN}(Y + \text{MaskedMHA}(Y))$; $Y'' = \text{LN}(Y' + \text{CrossAttn}(Y', \text{Enc}))$; $Y''' = \text{LN}(Y'' + \text{FFN}(Y''))$ |
| FFN | $\max(0, yW_1 + b_1)W_2 + b_2$ |
| Output projection | $P = \text{softmax}(Y_{final}W_{vocab} + b)$ |
| Autoregressive factorization | $p(z_{P+1:P+T}\mid z_{1:P}) = \prod_{s=1}^T p(z_{P+s}\mid z_{1:P+s-1})$ |
| Matmul cost | $(a\times b)(b\times c)$: $ac(2b-1) = O(abc)$ |
| Attention layer cost | $O(nD^2 + n^2D)$ |
| Cross-attention cost | $O((t+E)D^2 + tED)$ |
| Causal pairs | $n(n+1)/2$ |
| Generation, no cache | $O\big(\sum_{t=P}^{P+T-1}(tD^2 + t^2D)\big) = O(T^2D^2 + T^3D)$ |
| Generation, KV cache | $O(PD^2 + P^2D + \sum(D^2 + tD)) = O(TD^2 + T^2D)$ |
| Per-step cost | $O(tD^2 + t^2D) \to O(D^2 + tD)$ |
| Cached decode step | $o_t = \text{softmax}(q_tK_{1:t}^\top/\sqrt d)\,V_{1:t}$ |
| KV cache size (per layer) | MHA $2thd = 2tD$; GQA $2th_{kv}d$; MQA $2td$ |
| Projection work per new token | $D^2 + 2Dh_{kv}d + D^2$ |
| Head width | $d = D/h$ (unchanged in MQA/GQA) |
| Softmax as ratio of sums | $O = \mathcal{A}/Z$, $\mathcal{A} = \sum e^{s_i}v_i$, $Z = \sum e^{s_j}$ |
| Block merge | $O = (\mathcal{A}_1 + \mathcal{A}_2)/(Z_1 + Z_2)$ |
| Stable softmax | $p_i = e^{s_i - m}/\sum_j e^{s_j - m}$, $m = \max s_j$ |
| Online rescale | $m' = \max(m_{old}, m_{new})$, $\beta = e^{m - m'}$, $\ell' = \beta_{old}\ell_{old} + \beta_{new}\ell_{new}$, $u' = \beta_{old}u_{old} + \beta_{new}u_{new}$ |
| Arithmetic intensity | computation performed / bytes moved |

---

## 9. Cheat Sheet & Recall Aids

**One-liners**
- **Causal mask:** $-\infty$ above the diagonal, so softmax gives exactly 0 there and renormalizes the rest.
- **Cross-attention:** Q from decoder, K/V from encoder, no mask.
- **Teacher forcing:** training is parallel because targets are known; the mask stops peeking.
- **Prefill vs decode:** prefill has many queries at once (FlashAttention helps most); decode has one query per step (KV cache helps most).
- **KV cache:** past K/V never change under causality, so compute once and reuse. Cubic generation becomes quadratic.
- **MQA/GQA:** fewer K/V heads, so a smaller cache and less bandwidth. Head width unchanged; same $O(D^2 + tD)$.
- **FlashAttention:** exact attention, tiled in SRAM, online softmax. Fixes memory traffic, not $O(n^2)$.

**Gotchas**
- Cache **K and V**, never Q or attention weights (the new query needs fresh scores).
- Causal masking halves the pairs but is still $O(n^2)$.
- MQA's K/V are dimension $d$, not $D$. Output returns to $D$ via concatenating $h$ head outputs.
- MQA ≠ MHA with heads deleted; the shared projections are trained that way.
- Online softmax accumulates **numerator and denominator**, not probabilities, so it is exact. The running max only adds stability.
- FlashAttention doesn't reduce FLOPs (it even recomputes tiles in backward); it reduces HBM reads/writes.
- "Attention is quadratic" means the $n^2D$ term; $nD^2$ dominates when $n < D$.

**Mnemonic, the three efficiency fixes: "Cache, Share, Tile"**
- **Cache:** KV cache (don't recompute the past)
- **Share:** MQA/GQA (store less of it)
- **Tile:** FlashAttention (move less of it)

**Worldview line:** MHA = every head its own worldview; MQA = own question, one shared worldview; GQA = groups share a worldview.

**Self-test**
- [ ] Write the causal mask and explain why $e^{-\infty}$ gives zero weight after renormalization.
- [ ] Derive $O(nD^2 + n^2D)$ stage by stage.
- [ ] Show why generation without a cache is $O(T^3D)$ and with one is $O(T^2D)$.
- [ ] Argue by induction over layers that past K/V are unchanged.
- [ ] Compute KV cache size for MHA/GQA/MQA given $t, h, h_{kv}, d, L$, bytes.
- [ ] Explain how MQA heads still specialize.
- [ ] Run online softmax with rescaling on a 4-score example.
- [ ] State what FlashAttention does and does not improve.
