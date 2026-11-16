---
type: lecture
course: '[[_CS5326 Advanced GenAI|CS5326 Advanced GenAI]]'
source:
- '[[004-Residuals and Norms.pdf]]'
- '[[004-residuals-and-normalization.html]]'
tags:
- course/cs5326
---

%% nav:start %%
[[_CS5326 Advanced GenAI|CS5326 Advanced GenAI]] · ← [[Lecture_06_Decoding_KV_Cache_MQA_GQA_FlashAttention]] · [[Lecture_08_Activations]] →
Concepts: [[Residual Connection]], [[LayerNorm]], [[RMSNorm]], [[Pre-norm]], [[DeepNorm]]
%% nav:end %%

# Residual Connections and Normalization (CS5326, Lecture 7)

*Follows [[Lecture_06_Decoding_KV_Cache_MQA_GQA_FlashAttention|Lecture 6]], which finished the attention side of the Transformer block. This lecture covers the two pieces that wrap every sublayer, the residual connection and the normalization layer: why they exist, how gradients flow through them, and where the normalization should go (post-, pre-, output- and peri-norm), ending with RMSNorm, BatchNorm and DeepNorm. [[Lecture_08_Activations|Lecture 8]] then opens the FFN sublayer itself.*

## Topic Inventory

- 1. Residual Connections
  - 1.1 Where they sit in the 2017 Transformer
  - 1.2 A world without residuals
  - 1.3 What residuals make easier to learn
  - 1.4 What they do at inference
  - 1.5 Gradients through a scalar residual block
  - 1.6 A numerical example
  - 1.7 Vectors: the 1 becomes an identity matrix
  - 1.8 Summary of the residual argument
- 2. Why and How to Normalize
  - 2.1 Two separate questions
  - 2.2 Why normalize at all
  - 2.3 LayerNorm, step by step
  - 2.4 Learned scale and offset
- 3. Where to Put the Norm
  - 3.1 Post-norm in the original Transformer
  - 3.2 The problem with post-norm
  - 3.3 Digression: dropout
  - 3.4 Learning-rate warm-up
  - 3.5 Why post-norm needed warm-up
  - 3.6 Output norm
  - 3.7 Output norm's problems: residual growth and attention
  - 3.8 Pre-norm
  - 3.9 Pre-norm's problem
  - 3.10 Peri-norm
  - 3.11 Placement summary
- 4. How to Compute the Norm, and Deeper Stacks
  - 4.1 RMSNorm
  - 4.2 RMSNorm's properties and Llama's pre-RMSNorm block
  - 4.3 BatchNorm, and why Transformers use LayerNorm
  - 4.4 DeepNorm
- 5. Numerical Problems
- 6. Formula Summary Sheet
- 7. Cheat Sheet & Recall Aids

---

## 1. Residual Connections

### 1.1 Where they sit in the 2017 Transformer

![[L07_transformer_2017.png|550]]

*The original encoder–decoder Transformer (Vaswani et al., 2017; annotated figure from Lena Voita's NLP course). Every attention and feed-forward sublayer is followed by an "Add & Norm" box: "Add" is the residual connection, "Norm" is LayerNorm. These boxes are the subject of this lecture.*

### 1.2 A world without residuals

Think of each layer as revising a representation that earlier layers have already built, and which may already hold much useful information. Without a residual connection,

$$
y = F(x),
$$

everything passed onward must come through the sublayer $F$ (attention or an FFN). The layer must both preserve the useful existing information and make whatever changes are needed. With a **residual connection** (He et al., 2016, introduced in ResNets for image recognition),

$$
\boxed{y = x + F(x)}
$$

the existing representation is supplied automatically and the layer can concentrate on producing an **update**. Intuition: **keep the current draft and add edits.** Here $x$ is the input to this particular sublayer, not necessarily the original input to the whole model. The running sum carried through the stack is called the **residual stream**.

![[L07_residual_connection.png|380]]

*A residual connection adds a block's input to its output; the red path is the shortcut (skip connection).*

### 1.3 What residuals make easier to learn

Suppose a representation is already close to what the next stage needs:

| Desired behaviour | Without a residual | With a residual |
|---|---|---|
| Preserve the input | Learn $F(x) = x$ | Learn $F(x) = 0$ |
| Make a small adjustment $\Delta(x)$ | Learn $F(x) = x + \Delta(x)$ | Learn $F(x) = \Delta(x)$ |

Producing a small or zero update is easier than reproducing the input through several nonlinear transformations (setting weights near zero is trivial; learning an exact identity through ReLUs and projections is not). So extra layers get a convenient starting point: preserve what is there, then learn useful changes. The updates can still be large or negative; residuals do not restrict layers to small corrections.

### 1.4 What they do at inference

In the forward pass the next layer receives

$$
\underbrace{x}_{\text{existing representation}} + \underbrace{F(x)}_{\text{learned update}} .
$$

For example, the representation of "bank" may already carry useful word information, and attention can add information from "river" elsewhere in the sentence, helping later layers interpret "bank" in context. The residual route carries existing information through many layers while they progressively modify it. Two caveats: it does not *guarantee* information survives (an update can overwrite or cancel parts of $x$), and it skips no computation ($F(x)$ is still evaluated). Its main benefit is making a deep architecture trainable, which is a backward-pass story.

### 1.5 Gradients through a scalar residual block

| Symbol | Meaning |
|---|---|
| $x$ | Input to this residual block (the representation from earlier layers) |
| $F(x)$ | Output of the block's learned transformation (attention or FFN) |
| $y = x + F(x)$ | Output of the block, passed to later layers |
| $L$ | The network's final loss, a single number |
| $\partial L/\partial y$ | Sensitivity of the loss to this block's output; arrives from later layers during backpropagation |
| $\partial L/\partial x$ | Sensitivity of the loss to this block's input; what we pass to earlier layers |

Although $y$ is usually an intermediate representation, it affects the final prediction and therefore the loss.

**Without a residual**, $y = F(x)$ and the chain rule gives $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y}F'(x)$. The gradient must pass through $F$: if $F'$ is small the gradient shrinks, and repeated shrinking over many layers makes early layers learn extremely slowly (the **vanishing gradient** problem).

**With a residual**, a small change in $x$ affects $y$ through both occurrences of $x$: directly and through $F(x)$. Differentiating, and using that the derivative of a sum is the sum of derivatives,

$$
\frac{dy}{dx} = \frac{d}{dx}\big(x + F(x)\big) = \frac{dx}{dx} + \frac{dF(x)}{dx} = 1 + F'(x).
$$

The 1 means a unit change in the input produces a unit change along the direct branch. Connecting to the loss by the chain rule, $\frac{dL}{dx} = \frac{dL}{dy}\frac{dy}{dx}$, substituting, and distributing,

$$
\boxed{\frac{dL}{dx} = \underbrace{\frac{dL}{dy}}_{\text{through the direct branch}} + \underbrace{\frac{dL}{dy}F'(x)}_{\text{through the learned branch}}}
$$

The earlier layer receives a direct gradient contribution even when the route through $F$ contributes very little. *(Expanding on this: across $L$ stacked blocks, the no-residual gradient is a product $\prod_\ell F_\ell'$, which shrinks exponentially if each factor is below 1; with residuals it is $\prod_\ell (1 + F_\ell')$, which expands into a sum of paths that includes the pure-shortcut path with gradient 1.)*

### 1.6 A numerical example

Let $F(x) = 0.1x$ and $x = 2$, so the block outputs $y = 2 + 0.2 = 2.2$. Treat $y$ as the final prediction with target $t = 3$ and loss $L = \tfrac12(y - t)^2$. Then

$$
\frac{dL}{dy} = \tfrac12 \cdot 2(y - t) \cdot 1 = y - t = -0.8, \qquad \frac{dy}{dx} = 1 + 0.1 = 1.1, \qquad \frac{dL}{dx} = (-0.8)(1.1) = -0.88 .
$$

Shown as two contributions, $\frac{dL}{dx} = \underbrace{-0.8}_{\text{direct}} + \underbrace{(-0.8)(0.1)}_{\text{learned}} = -0.8 - 0.08$. Even though the learned branch multiplies the gradient by 0.1, the direct branch passes $-0.8$ onward intact.

### 1.7 Vectors: the 1 becomes an identity matrix

Let the input have two features, $\mathbf{x} = [x_1, x_2]^\top$, $F(\mathbf{x}) = [F_1(\mathbf{x}), F_2(\mathbf{x})]^\top$. The residual addition is elementwise, $\mathbf{y} = [x_1 + F_1(\mathbf{x}),\; x_2 + F_2(\mathbf{x})]^\top$. A vector derivative must say how every output coordinate responds to every input coordinate; this matrix of partial derivatives is the **Jacobian**.

For the direct branch alone, $\mathbf{s}(\mathbf{x}) = \mathbf{x}$: changing $x_1$ changes the copied $x_1$ by the same amount (derivative 1), and changing $x_2$ does not change it (derivative 0). The same holds for row two, so

$$
\frac{\partial \mathbf{x}}{\partial \mathbf{x}} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} = I_2,
$$

and for $d$ features it is $I_d$. The identity matrix is the vector equivalent of the scalar 1: it passes a vector through unchanged. With the learned branch's Jacobian $J_F = \begin{bmatrix} \partial F_1/\partial x_1 & \partial F_1/\partial x_2 \\ \partial F_2/\partial x_1 & \partial F_2/\partial x_2 \end{bmatrix}$, differentiating $\mathbf{y} = \mathbf{x} + F(\mathbf{x})$ gives

$$
\frac{\partial \mathbf{y}}{\partial \mathbf{x}} = I_2 + J_F = \begin{bmatrix} 1 + \frac{\partial F_1}{\partial x_1} & \frac{\partial F_1}{\partial x_2} \\ \frac{\partial F_2}{\partial x_1} & 1 + \frac{\partial F_2}{\partial x_2} \end{bmatrix}.
$$

The extra 1 appears only on the diagonal, because the shortcut connects each input coordinate only to the matching output coordinate.

**Backpropagating.** Let the incoming gradient be the column vector $\mathbf{g} = \nabla_{\mathbf{y}} L = [\partial L/\partial y_1, \partial L/\partial y_2]^\top$ (the loss derivatives collected over all coordinates of $\mathbf{y}$). For $x_1$, the chain rule sums its effects through both outputs:

$$
\frac{\partial L}{\partial x_1} = \frac{\partial L}{\partial y_1}\frac{\partial y_1}{\partial x_1} + \frac{\partial L}{\partial y_2}\frac{\partial y_2}{\partial x_1} = \frac{\partial L}{\partial y_1} + \frac{\partial L}{\partial y_1}\frac{\partial F_1}{\partial x_1} + \frac{\partial L}{\partial y_2}\frac{\partial F_2}{\partial x_1}.
$$

Doing this for every coordinate and collecting the results,

$$
\nabla_{\mathbf{x}} L = (I_d + J_F)^\top \mathbf{g} = I_d^\top \mathbf{g} + J_F^\top \mathbf{g} = \boxed{\underbrace{\mathbf{g}}_{\text{direct}} + \underbrace{J_F^\top \mathbf{g}}_{\text{through } F}}
$$

using $I_d^\top = I_d$ and $I_d\mathbf{g} = \mathbf{g}$. The transpose appears because, for each input coordinate, backpropagation sums over output coordinates (a column of the Jacobian), as the component equation shows. If the learned branch contributes little, earlier layers still receive about $\mathbf{g}$. The total can still shrink or grow, since the two contributions are added (and can cancel); residuals supply a direct contribution, not a guarantee against all gradient problems.

### 1.8 Summary of the residual argument

If the incoming gradient is 2 and $F'(x) = 0.01$: through $F$ alone the earlier layer gets $2 \times 0.01 = 0.02$; with the residual route it gets $2 + 0.02 = 2.02$. Contributions can still cancel or become too large.

- **Forward:** carry the existing representation onward and add a learned update.
- **Backward:** give the learning signal a direct route to earlier layers.

That direct route is exactly why the *placement* of normalization matters next: anything placed on the shortcut sits in that route.

> [!note] Slide numbering
> The deck's page numbers jump from 16 to 20; slides 17–19 are not in the PDF. The companion page [[004-residuals-and-normalization.html|Residuals & Normalization — interactive guide]] covers the same material interactively.

---

## 2. Why and How to Normalize

### 2.1 Two separate questions

Normalization design splits into two independent pathways that do not compete: **where/when** to apply a norm (placement, §3), and **how** to compute it (LayerNorm, RMSNorm, BatchNorm, §2.3 and §4). Any computation can be combined with any placement; Llama, for example, combines RMSNorm with pre-norm placement.

### 2.2 Why normalize at all

Recall the residual update $z = x + F(x)$. As many layers transform and add representations, feature magnitudes can change considerably, so the next sublayer must work on inputs whose scale keeps shifting during training. Normalization gives each sublayer consistently scaled inputs. This helps optimization, but does not guarantee every activation or gradient stays small.

### 2.3 LayerNorm, step by step

**LayerNorm** (Ba, Kiros & Hinton, 2016) works **across the features of one token**. Take a token with three features $z = [1, 2, 3]$. Its mean is $\mu = \frac{1+2+3}{3} = 2$; subtracting it gives $z - \mu = [-1, 0, 1]$; its variance is

$$
\sigma^2 = \frac{(1-2)^2 + (2-2)^2 + (3-2)^2}{3} = \frac{2}{3};
$$

and dividing by the standard deviation $\sigma = \sqrt{2/3} \approx 0.816$,

$$
\hat z = \frac{z - \mu}{\sigma} \approx [-1.225,\; 0,\; 1.225].
$$

The result has mean zero and variance one. In practice a small positive $\epsilon$ is added inside the square root so a near-constant token does not cause division by (almost) zero.

### 2.4 Learned scale and offset

Finally the network learns a useful scale and offset per feature:

$$
\boxed{\text{LN}(z)_i = \gamma_i \, \frac{z_i - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta_i}
$$

- $i$ indexes a feature; $\gamma_i$ is its learned **scale** (gain), $\beta_i$ its learned **offset** (bias).
- $\mu$ and $\sigma^2$ are computed afresh for each token.
- The same $\gamma, \beta$ are shared across all tokens passing through that LayerNorm module.

After the learned scale and offset the output need not have mean 0 or variance 1 (the network can undo the normalization if that helps), and normalization does not make values Gaussian. For an input tensor of shape (batch size, sequence length, $d_{\text{model}}$), Transformer LayerNorm normalizes along the last dimension only; it never mixes statistics between tokens or examples.

---

## 3. Where to Put the Norm

With the computation defined, the placement question is about how normalization interacts with the residual shortcut from §1.

![[L07_norm_placements.png|600]]

*The four placements covered below. Blue is the shortcut, yellow the learned branch $F$, pink the normalization $N$. Post-norm puts $N$ on the shortcut after the sum; the other three keep the shortcut clean and differ in whether $N$ sits before $F$, after it, or both.*

### 3.1 Post-norm in the original Transformer

The original Transformer used

$$
y = \text{LN}\big(x + F(x)\big):
$$

compute the attention or FFN output $F(x)$, add the input $x$, then normalize the sum. This is **post-norm** (normalization after the residual addition). An encoder block, omitting dropout, is

$$
h = \text{LN}_1\big(x + \text{Attention}(x)\big), \qquad y = \text{LN}_2\big(h + \text{FFN}(h)\big),
$$

where the two LayerNorms have their own learned parameters. The original model also applied dropout to sublayer outputs before adding the residual (§3.3).

### 3.2 The problem with post-norm

The residual's benefit was a direct route from $x$ to $y$ with derivative $I$. In post-norm the block has two steps, $z = x + F(x)$ and $y = \text{LN}(z)$: the direct branch bypasses $F$ but still goes through LayerNorm. With incoming gradient $\mathbf{g} = \nabla_y L$, backpropagation first passes it through LayerNorm and only then through the residual addition:

$$
\mathbf{q} = J_{\text{LN}}^\top \mathbf{g}, \qquad \nabla_x L = \mathbf{q} + J_F^\top \mathbf{q},
$$

where $J_{\text{LN}}$ and $J_F$ are the Jacobians of the two operations. So the direct contribution reaching $x$ is $\mathbf{q}$, already transformed by LayerNorm, not the untouched $\mathbf{g}$.

- Across many blocks these normalization Jacobians repeatedly change gradient magnitudes and directions; there is no clean identity path from the loss to early layers.
- Post-norm is therefore more sensitive to initialization and learning-rate choices.
- In particular, Xiong et al. (2020) found large expected parameter gradients near the output at initialization for post-norm, which helps explain its reliance on learning-rate warm-up (§3.4–3.5).

This is not a claim that post-norm always fails: it worked in the original Transformer (and in BERT).

### 3.3 Digression: dropout

**Dropout** lets the network learn with some information temporarily unavailable: during training it randomly sets some activations to zero, with a fresh random choice each time it is applied. Intuition: if the network can always rely on particular features working together, it may over-depend on those combinations; randomly removing some pushes it to develop alternatives that generalize better.

For example, with $F(x) = [2, 4, 6, 8]$ and drop probability $p = 0.5$, one possible result under **inverted dropout** is $[4, 0, 12, 0]$. The survivors are doubled to preserve each value's expected contribution: $0.5(2a) + 0.5(0) = a$. Generally,

$$
\text{Dropout}(a_i) = \frac{m_i a_i}{1 - p}, \qquad m_i = \begin{cases} 0 & \text{with probability } p \\ 1 & \text{otherwise} \end{cases}
$$

At inference, dropout is switched off and every activation is used without extra scaling (that is why the scaling is done at training time). The original Transformer sublayer was

$$
y = \text{LN}\big(x + \text{Dropout}(F(x))\big),
$$

so dropout perturbed the learned update before addition; the base Transformer used $p = 0.1$. (Note that the deck's Lecture 8 writes the same scaling with *keep* probability $q = 1 - p$.)

### 3.4 Learning-rate warm-up

The **learning rate** $\eta$ controls the size of parameter updates; for gradient descent on weights $\theta$, $\theta_{\text{new}} = \theta_{\text{old}} - \eta\nabla_\theta L$. **Warm-up** starts $\eta$ small and increases it gradually. A linear warm-up over $T$ steps is

$$
\eta_t = \eta_{\text{peak}}\frac{t}{T}, \qquad 1 \le t \le T .
$$

With $\eta_{\text{peak}} = 0.001$ and $T = 1000$:

| Training step | Learning rate |
|---|---|
| 100 | 0.0001 |
| 500 | 0.0005 |
| 1000 | 0.001 |

Why? Early in training, gradients and the model's response to parameter changes can be poorly behaved, and large initial updates may destabilize training before useful representations form. This was especially relevant to post-norm Transformers. The original Transformer increased its learning rate linearly for 4,000 steps, then decreased it with an **inverse-square-root schedule** ($\eta \propto 1/\sqrt{t}$). Training and weight updates happen throughout warm-up; it is not a pause. Pre-norm often reduces this sensitivity, but warm-up remains useful.

### 3.5 Why post-norm needed warm-up

The slide's claim, "the pre/post-norm analysis found large expected parameter gradients near the output at initialization for the original post-norm design", means the original post-norm Transformer can make excessively large changes to some weights at the very start of training, so a small initial learning rate is used. Unpacking it:

- Weights start random. A **parameter gradient** $\partial L/\partial w$ says how sensitive the loss is to weight $w$; a large magnitude means a small weight change produces a large loss change.
- "Near the output" means the later layers, closest to the final prediction.
- "Expected" means averaged over random initializations, under the paper's assumptions; not every weight always has a large gradient.
- Crucially, **normalizing forward-pass activations does not guarantee well-scaled backward-pass gradients.**
- Large gradients cause large updates, $w_{\text{new}} = w_{\text{old}} - \eta\,\frac{\partial L}{\partial w}$.

Suppose $w = 1$ with gradient 100:

| Learning rate | Weight change | New weight |
|---|---|---|
| 0.01 | $-1$ | 0 |
| 0.0001 | $-0.01$ | 0.99 |

The first update wipes the weight out. Across many interacting layers, such early jumps can destabilize training. Warm-up limits them by keeping updates small even when gradients are large. Pre-norm gives a direct residual gradient route and better-behaved gradients at initialization in that analysis, reducing the specific instability warm-up was patching (Xiong et al. trained pre-norm Transformers without warm-up).

### 3.6 Output norm

Can we normalize the update while leaving the shortcut untouched? **Output norm** normalizes the update, then adds it:

$$
y = x + \text{Norm}\big(F(x)\big).
$$

Compute $F(x)$, normalize it, add it to $x$. Two attractive properties: the update is normalized before entering the residual stream, and the direct route through $x$ bypasses both $F$ and the normalization. To see the gradient benefit, define $H(x) = \text{Norm}(F(x))$; the block is $y = x + H(x)$, so by §1.7

$$
\nabla_x L = \mathbf{g} + J_H^\top \mathbf{g},
$$

and the direct contribution $\mathbf{g}$ is preserved. The remaining issue is what happens *inside* $F$.

### 3.7 Output norm's problems: residual growth and attention

The accumulated $x$ is not normalized before entering $F$, and normalizing each update does not keep their sum at a fixed magnitude. Repeatedly adding the same normalized vector $u$ gives

$$
x_L = x_0 + Lu,
$$

so each update is controlled but the stream grows linearly with depth.

This matters especially for attention. With fixed projections and no biases, $Q = XW_Q$ and $K = XW_K$. Scaling $X$ by 10 scales both $Q$ and $K$ by 10, so $QK^\top$ scales by 100. That makes softmax far more concentrated, closer to a **hardmax** (all weight on the single largest score). Normalizing the attention *output* afterwards cannot undo a change in *which* values attention selected. Hence successful output-norm designs also control the attention scores:

- **Swin Transformer V2** uses **scaled cosine attention** (scores are cosine similarities of $q$ and $k$ divided by a learned temperature, so input scale cancels).
- **OLMo 2** uses **QK-norm** (query–key normalization: normalize $q$ and $k$ before their dot product), and reports that the reordered norm and QK-norm worked better together than either alone in its experiments.

### 3.8 Pre-norm

What if we normalize the representation before $F$ uses it? **Pre-norm**:

$$
\boxed{y = x + F\big(\text{Norm}(x)\big)}
$$

Take a normalized copy of $x$, let $F$ compute an update from it, add the update to the *original* $x$. This keeps the clean residual gradient route, and $F$ additionally receives a controlled-scale input even as the residual stream grows. For attention, this reduces sensitivity to the overall scale of the incoming representation, though learned projections can still amplify queries and keys. Pre-norm has favourable gradients at initialization and often makes deep Transformers easier to train; it is the GPT-2/GPT-3/Llama default. *(Pre-norm stacks also add one final normalization after the last block, before the output head.)*

### 3.9 Pre-norm's problem

Moving normalization to the input gives up explicit normalization of the update. Even with a well-scaled input, $F$'s learned weights can produce a large output, which goes straight into the residual stream. So the two placements protect different things:

| Design | What it directly normalizes | What remains unnormalized locally |
|---|---|---|
| Output norm | $F$'s update | $F$'s input |
| Pre-norm | $F$'s input | $F$'s update |

Pre-norm is a different allocation of scale control, not an automatic improvement over output norm in every architecture.

### 3.10 Peri-norm

Can we normalize both the input and the update while keeping the shortcut? **Peri-norm** ("peri" = around) normalizes both sides of $F$:

$$
y = x + \text{Norm}_{\text{out}}\Big(F\big(\text{Norm}_{\text{in}}(x)\big)\Big).
$$

Normalize the input to $F$, compute the update, normalize the update, add it to the untouched $x$. This combines input-scale control, update-scale control and a direct residual route; the two norm modules have separate parameters. The Peri-LN paper (Kim et al., 2025) reports improved activation and gradient behaviour in its experiments. Limitations: extra normalization work, and it still does not force the accumulated stream to a constant magnitude or guarantee stability under every recipe. A final normalization after the full stack remains part of the Peri-LN design. *(Related: NormFormer, on the reading list, also adds extra norms inside pre-norm blocks.)*

### 3.11 Placement summary

| Placement | Sublayer equation | Shortcut clean? | Normalizes $F$'s input | Normalizes the update |
|---|---|---|---|---|
| Post-norm | $y = \text{Norm}(x + F(x))$ | No | Yes (via previous block's output) | Yes (with $x$) |
| Output norm | $y = x + \text{Norm}(F(x))$ | Yes | No | Yes |
| Pre-norm | $y = x + F(\text{Norm}(x))$ | Yes | Yes | No |
| Peri-LN | $y = x + \text{Norm}_{\text{out}}(F(\text{Norm}_{\text{in}}(x)))$ | Yes | Yes | Yes |

*(The last three columns are a summary added here from §3.1–3.10.)*

---

## 4. How to Compute the Norm, and Deeper Stacks

### 4.1 RMSNorm

LayerNorm does two things before its learned parameters: it removes the mean, and it controls magnitude using the standard deviation. **RMSNorm** (Zhang & Sennrich, 2019) asks whether controlling magnitude alone is enough. It computes the **root mean square**

$$
\text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2}
$$

and uses

$$
\boxed{\text{RMSNorm}(x)_i = \gamma_i \frac{x_i}{\sqrt{\frac{1}{d}\sum_{j=1}^d x_j^2 + \epsilon}}}
$$

(no offset $\beta$ in the standard form). For the same $x = [1, 2, 3]$, ignoring $\epsilon$, $\text{RMS}(x) = \sqrt{(1 + 4 + 9)/3} \approx 2.160$, so before scaling $\hat x \approx [0.463, 0.926, 1.389]$. Its RMS is one, but its mean is not zero.

### 4.2 RMSNorm's properties and Llama's pre-RMSNorm block

Both the numerator and the denominator differ from LayerNorm: RMSNorm is **not** LayerNorm with the mean subtraction deleted from the numerator, because the RMS about zero is not the standard deviation about the mean (they coincide only when the mean is zero).

- **Strength:** simpler computation (no mean, no $\beta$) while keeping scale normalization.
- **Limitation:** loses LayerNorm's invariance to a common offset. Adding the same constant to every feature leaves LayerNorm's standardized result unchanged but generally changes RMSNorm's (Problem 4). Whether that matters depends on the model and task.
- Both cost $O(d)$ per token; real speed gains depend on implementation and hardware.

A concrete combination is **pre-RMSNorm**, used in Llama 3's reference implementation:

$$
h = x + \text{Attention}\big(\text{RMSNorm}_1(x)\big), \qquad y = h + \text{FFN}\big(\text{RMSNorm}_2(h)\big).
$$

### 4.3 BatchNorm, and why Transformers use LayerNorm

Take three examples with three features each:

$$
X = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix}
$$

LayerNorm normalizes each **row** (one example's features: $[1,2,3]$, then $[4,5,6]$, …). **BatchNorm** (Ioffe & Szegedy, 2015) normalizes each **column** (one feature across the batch: $[1,4,7]$, then $[2,5,8]$, …). It asks: "how large is this feature for this example compared with the same feature across the batch?" For feature $j$ of example $b$,

$$
\text{BN}(X_{bj}) = \gamma_j\frac{X_{bj} - \mu_j}{\sqrt{\sigma_j^2 + \epsilon}} + \beta_j,
$$

with $\mu_j, \sigma_j^2$ computed across the batch. During training BatchNorm uses the current batch's statistics and keeps running estimates; at inference it uses the stored estimates.

Why LayerNorm suits Transformers: each token is normalized independently of the other examples in its batch, and the same computation is used in training and inference. BatchNorm makes outputs depend on batch composition and gives noisy statistics when few values are available (small batches, variable-length padded sequences, and one-token-at-a-time decoding).

### 4.4 DeepNorm

**DeepNorm** (Wang et al., DeepNet, 2022/2024) makes each layer's contribution manageable as depth increases. It modifies post-norm:

$$
\text{post-norm: } y = \text{LN}\big(x + F(x)\big) \qquad\longrightarrow\qquad \text{DeepNorm: } \boxed{y = \text{LN}\big(\alpha x + F(x; W)\big), \quad W_{\text{initial}} = \beta W_{\text{standard}}}
$$

It combines two coordinated changes:

1. **$\alpha$** scales the direct branch by a fixed, architecture-dependent factor (> 1), strengthening the existing representation before addition.
2. **$\beta$** initializes selected weights inside $F$ at a smaller scale (< 1). It applies to both FFN matrices and attention's value and output projections, but not the query and key projections.

The two constants are chosen together according to depth *(e.g. for an encoder-only model with $N$ layers the paper uses $\alpha = (2N)^{1/4}$, $\beta = (8N)^{-1/4}$; outside the slides)*. Details of $\beta$: with $\beta = 0.2$, an initial weight of 0.5 becomes 0.1; the scaling happens once at initialization and the weights then learn normally; $\beta$ is neither learned nor reapplied to the branch output at every forward pass.

Intuition: when hundreds of layers contribute changes, control how strongly each one can alter the overall computation. $\alpha$ gives more weight to what the network already carries; $\beta$ makes its initial modifications restrained. Because LayerNorm follows the addition, increasing $\alpha$ does not just multiply the output by $\alpha$; it changes the balance between branches. The authors trained Transformers with up to 1,000 layers. The initialization recipe is essential (DeepNorm is more than inserting a multiplier), and unlike pre-norm it keeps normalization on the direct route, retaining post-norm's typically better final quality.

---

## 5. Numerical Problems

**Problem 1 (scalar residual gradient).** A block computes $F(x) = 0.5x^2$ at $x = 1$; the output feeds $L = \tfrac12(y - 4)^2$. Find $dL/dx$ with and without the residual.

*Solution.* With the residual, $y = 1 + 0.5 = 1.5$, $dL/dy = y - 4 = -2.5$, $F'(1) = 1$, so $dL/dx = -2.5(1 + 1) = -5.0$ (direct $-2.5$, learned $-2.5$). Without it, $y = F(x) = 0.5$, $dL/dy = -3.5$, $dL/dx = -3.5 \times 1 = -3.5$. Note the gradients differ partly because the forward outputs differ.

$$
\boxed{dL/dx = -5.0 \text{ (residual)}, \quad -3.5 \text{ (plain)}}
$$

**Problem 2 (depth).** Ten stacked blocks each have local derivative $F' = 0.5$. Compare the end-to-end gradient factor with and without residuals; repeat for $F' = -0.5$.

*Solution.* Without residuals, $0.5^{10} \approx 9.8 \times 10^{-4}$, vanishing. With residuals, $(1 + 0.5)^{10} \approx 57.7$, now growing, illustrating that residuals give a direct path but do not guarantee well-scaled gradients. With $F' = -0.5$, $(0.5)^{10}$ again: the contributions partially cancel, as §1.7 warns.

$$
\boxed{9.8\times10^{-4} \text{ vs } 57.7; \quad F' = -0.5: 9.8\times10^{-4}}
$$

**Problem 3 (vector backprop).** $\mathbf{g} = [1, -2]^\top$ and $J_F = \begin{bmatrix} 0.1 & 0.3 \\ 0 & -0.2 \end{bmatrix}$. Compute $\nabla_{\mathbf{x}}L$.

*Solution.* $J_F^\top\mathbf{g} = \begin{bmatrix} 0.1 & 0 \\ 0.3 & -0.2 \end{bmatrix}\begin{bmatrix} 1 \\ -2 \end{bmatrix} = \begin{bmatrix} 0.1 \\ 0.3 + 0.4 \end{bmatrix} = \begin{bmatrix} 0.1 \\ 0.7 \end{bmatrix}$, so $\nabla_{\mathbf{x}}L = \mathbf{g} + J_F^\top\mathbf{g} = [1.1, -1.3]^\top$. Using $J_F$ instead of $J_F^\top$ would wrongly give $[0.1 - 0.6, 0.4] + \mathbf{g} = [0.5, -1.6]$.

$$
\boxed{\nabla_{\mathbf{x}}L = [1.1,\; -1.3]^\top}
$$

**Problem 4 (LayerNorm vs RMSNorm, offset invariance).** Normalize $[1,2,3]$ and $[11,12,13]$ with both (no $\epsilon$, $\gamma = 1$, $\beta = 0$).

*Solution.* LayerNorm: both have $\sigma = \sqrt{2/3}$ after mean removal, so both give $[-1.225, 0, 1.225]$. RMSNorm: $[1,2,3]$ has RMS 2.160, giving $[0.463, 0.926, 1.389]$; $[11,12,13]$ has RMS $\sqrt{434/3} = 12.028$, giving $[0.915, 0.998, 1.081]$. RMSNorm's output changes with the shared offset, and for a large offset almost all variation between features is squashed.

$$
\boxed{\text{LN: identical}; \quad \text{RMSNorm: } [0.463, 0.926, 1.389] \text{ vs } [0.915, 0.998, 1.081]}
$$

**Problem 5 (LayerNorm with $\gamma, \beta$).** Token $z = [4, -2, 0, 6]$, $\gamma = [1, 2, 1, 0.5]$, $\beta = [0, 0, 1, 0]$.

*Solution.* $\mu = 8/4 = 2$; deviations $[2, -4, -2, 4]$; $\sigma^2 = (4 + 16 + 4 + 16)/4 = 10$, $\sigma = 3.162$; $\hat z = [0.632, -1.265, -0.632, 1.265]$. Applying $\gamma, \beta$: $[0.632, -2.530, 0.368, 0.632]$. The output no longer has mean 0 or variance 1, as §2.4 notes.

$$
\boxed{\text{LN}(z) = [0.632,\; -2.530,\; 0.368,\; 0.632]}
$$

**Problem 6 (BatchNorm vs LayerNorm on the same matrix).** For $X = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$, compute both (no $\epsilon$, $\gamma = 1$, $\beta = 0$).

*Solution.* LayerNorm per row: $[1,2]$ has $\mu = 1.5$, $\sigma = 0.5$, giving $[-1, 1]$; likewise $[3,4] \to [-1, 1]$. BatchNorm per column: $[1,3]$ has $\mu = 2$, $\sigma = 1$, giving $[-1, 1]$; $[2,4] \to [-1, 1]$. So LN gives $\begin{bmatrix} -1 & 1 \\ -1 & 1 \end{bmatrix}$ and BN gives $\begin{bmatrix} -1 & -1 \\ 1 & 1 \end{bmatrix}$: LN says "within each example, feature 2 is larger"; BN says "example 2 is larger on every feature".

$$
\boxed{\text{LN} = \begin{bmatrix} -1 & 1 \\ -1 & 1 \end{bmatrix}, \quad \text{BN} = \begin{bmatrix} -1 & -1 \\ 1 & 1 \end{bmatrix}}
$$

**Problem 7 (attention scale and output-norm growth).** (a) A pre-softmax score of 2 between two tokens, with no QK control: what is it after the residual stream grows 3×? (b) An output-norm stack adds a unit-norm update in the same direction for 48 layers to $x_0$ of norm 1 (same direction). Final norm?

*Solution.* (a) $Q$ and $K$ each scale by 3, so scores scale by 9: $2 \to 18$. (b) $\lVert x_L \rVert = 1 + 48 = 49$; normalized updates do not bound the sum (§3.7).

$$
\boxed{18; \quad 49}
$$

**Problem 8 (dropout and warm-up).** (a) Inverted dropout with $p = 0.2$ on $a = [5, 10, 15, 20]$ drops the second entry. Output? (b) The original Transformer warmed up for 4,000 steps; with linear warm-up to $\eta_{\text{peak}} = 6 \times 10^{-4}$, what is $\eta$ at step 1,000, and how large is an update to a weight with gradient 50 then?

*Solution.* (a) Survivors are divided by $1 - p = 0.8$: $[6.25, 0, 18.75, 25]$. (b) $\eta_{1000} = 6\times10^{-4} \times 1000/4000 = 1.5 \times 10^{-4}$; the update is $1.5\times10^{-4} \times 50 = 0.0075$, versus 0.03 at the peak rate.

$$
\boxed{[6.25, 0, 18.75, 25]; \quad \eta = 1.5\times10^{-4}, \; \Delta w = -0.0075}
$$

**Problem 9 (DeepNorm).** Using the paper's encoder-only constants, compute $\alpha$ and $\beta$ for $N = 100$ layers, and the initial value of a value-projection weight that would have been 0.4.

*Solution.* $\alpha = (200)^{1/4} = 3.761$, $\beta = (800)^{-1/4} = 0.188$; the weight starts at $0.188 \times 0.4 = 0.075$. Query/key weights would be left at their standard initialization.

$$
\boxed{\alpha = 3.761, \; \beta = 0.188, \; w_0 = 0.075}
$$

---

## 6. Formula Summary Sheet

| Concept | Formula |
|---|---|
| Residual block | $y = x + F(x)$ |
| Scalar residual derivative | $dy/dx = 1 + F'(x)$ |
| Scalar residual gradient | $\frac{dL}{dx} = \frac{dL}{dy} + \frac{dL}{dy}F'(x)$ |
| Vector Jacobian | $\partial\mathbf{y}/\partial\mathbf{x} = I_d + J_F$ |
| Vector backprop | $\nabla_{\mathbf{x}}L = \mathbf{g} + J_F^\top\mathbf{g}$ |
| LayerNorm | $\gamma_i\frac{z_i - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta_i$, stats over one token's features |
| Post-norm | $y = \text{LN}(x + F(x))$; backprop $\mathbf{q} = J_{\text{LN}}^\top\mathbf{g}$, $\nabla_x L = \mathbf{q} + J_F^\top\mathbf{q}$ |
| Inverted dropout | $m_i a_i/(1-p)$, $m_i \sim \text{Bernoulli}(1-p)$ |
| Original sublayer | $y = \text{LN}(x + \text{Dropout}(F(x)))$, $p = 0.1$ |
| Linear warm-up | $\eta_t = \eta_{\text{peak}}\,t/T$ |
| Output norm | $y = x + \text{Norm}(F(x))$ |
| Output-norm growth | $x_L = x_0 + Lu$ |
| Attention scale | $X \to cX \Rightarrow QK^\top \to c^2 QK^\top$ |
| Pre-norm | $y = x + F(\text{Norm}(x))$ |
| Peri-LN | $y = x + \text{Norm}_{\text{out}}(F(\text{Norm}_{\text{in}}(x)))$ |
| RMS | $\sqrt{\frac1d\sum_i x_i^2}$ |
| RMSNorm | $\gamma_i x_i/\sqrt{\frac1d\sum_j x_j^2 + \epsilon}$ |
| Llama pre-RMSNorm | $h = x + \text{Attn}(\text{RMSNorm}_1(x)), \; y = h + \text{FFN}(\text{RMSNorm}_2(h))$ |
| BatchNorm | $\gamma_j\frac{X_{bj} - \mu_j}{\sqrt{\sigma_j^2 + \epsilon}} + \beta_j$, stats over the batch |
| DeepNorm | $y = \text{LN}(\alpha x + F(x; W)), \; W_{\text{init}} = \beta W_{\text{std}}$ |

---

## 7. Cheat Sheet & Recall Aids

**Story in one line each**
- **Residual:** keep the draft, add edits; backward it adds a gradient path with derivative $I$.
- **LayerNorm:** standardize each token's features, then learned $\gamma, \beta$.
- **Post-norm:** norm sits on the shortcut, so the identity path is lost; needed warm-up.
- **Output norm:** clean shortcut, normalized update, but unnormalized input to $F$ and a growing stream (needs QK control).
- **Pre-norm:** clean shortcut, normalized input, unnormalized update. LLM default.
- **Peri-LN:** both.
- **RMSNorm:** scale only, no mean; Llama uses pre-RMSNorm.
- **DeepNorm:** post-norm with $\alpha x$ upweighted and $\beta$-shrunk init; 1,000 layers.

**Mnemonic for placements — "POPP":** **P**ost (after the sum), **O**utput (after $F$), **P**re (before $F$), **P**eri (both sides of $F$).

**Gotchas**
- Residuals don't skip computation at inference; $F(x)$ is still computed.
- Residuals don't guarantee gradients are fine: $\mathbf{g}$ and $J_F^\top\mathbf{g}$ can cancel or explode.
- Backprop uses $J^\top$, not $J$.
- LayerNorm = across features of one token; BatchNorm = across the batch for one feature.
- After $\gamma, \beta$, LN output need not be mean 0 / variance 1, and it is never "made Gaussian".
- RMSNorm ≠ LayerNorm minus mean subtraction: the denominator also changes. It is not shift-invariant.
- Normalizing activations forward ≠ well-scaled gradients backward.
- Normalizing updates does not bound their sum ($x_0 + Lu$).
- Scaling inputs by $c$ scales attention scores by $c^2$.
- DeepNorm's $\beta$ is a one-time init scale, not learned and not applied each forward pass; it skips $W_Q, W_K$.
- Dropout is off at inference; inverted dropout rescales at training time.

**Recall checklist**
- [ ] Derive $dL/dx = \frac{dL}{dy}(1 + F')$ and the vector form with $J_F^\top$.
- [ ] Normalize $[1,2,3]$ with LayerNorm and RMSNorm by hand.
- [ ] Write all four placement equations and say what each leaves unnormalized.
- [ ] Explain why post-norm needs warm-up (forward norm ≠ backward scale; large output-side gradients at init).
- [ ] Compute a linear warm-up schedule and an inverted-dropout output.
- [ ] Explain why output norm needs QK-norm or cosine attention.
- [ ] State DeepNorm's two changes and which weights $\beta$ touches.
