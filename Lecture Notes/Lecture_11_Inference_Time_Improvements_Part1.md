---
type: lecture
course: "[[_CS5326 Advanced GenAI|CS5326 Advanced GenAI]]"
source:
  - "[[008-Inference-timeImprovement-And-Reasoning-part1_2.pdf]]"
tags:
  - course/cs5326
---

%% nav:start %%
[[_CS5326 Advanced GenAI|CS5326 Advanced GenAI]] · ← [[Lecture_10_Neural_Scaling_Laws]] · → [[Lecture_12_Evaluation_and_Benchmarks]]
Concepts: [[Test-Time Compute]], [[Decoding]], [[In-Context Learning]], [[Chain of Thought]], [[Self-Consistency]]
%% nav:end %%

# Inference-Time Improvements and Reasoning, Part 1 (CS5326, Lecture 11)

*[[Lecture_10_Neural_Scaling_Laws#16. Answer-Time Compute|Lecture 10 §16]] ended by asking how much more a trained model can achieve if we spend extra compute on each question instead of on a bigger model. This lecture answers that question in detail. With the weights frozen, it covers how tokens are read out, what goes into the context, how calls are organized, and how many candidate answers are combined into one.*

> [!info] Overlap with ATML Lecture 12
> About a third of this deck repeats material from [[ATML_Lecture_12 - Test-Time Scaling and Multimodal Reasoning|ATML Lecture 12 (Test-Time Scaling)]]: chain of thought and Plan-and-Solve ([[ATML_Lecture_12 - Test-Time Scaling and Multimodal Reasoning#2. Sequential Scaling — Think Longer|ATML L12 §2]]), self-consistency and best-of-N ([[ATML_Lecture_12 - Test-Time Scaling and Multimodal Reasoning#3. Parallel Scaling — Think Wider|ATML L12 §3]]), and coverage vs selection ([[ATML_Lecture_12 - Test-Time Scaling and Multimodal Reasoning#4.1 Solution coverage vs selection accuracy|ATML L12 §4.1]]). Decoding repeats [[ATML_Lecture_7 - LLM Foundations#6. Decoding|ATML L7 §6]]. Most of the rest is new in this course: contrastive and constrained decoding, the two explanations of in-context learning, the controller view of test-time compute, and the binomial analysis of voting. Each overlapping section below links to its ATML counterpart.

This deck is part 1 of 2. It ends where best-of-N begins; coverage, the three kinds of verifier, search, refinement and reasoning models are in part 2.

## Topic Inventory

- 1. Why Inference Time
  - 1.1 The gap between what a model can do and what it does
  - 1.2 Three reasons to work at this layer
  - 1.3 The four levers
  - 1.4 Fixed computation per token
  - 1.5 Roadmap
- 2. Decoding
  - 2.1 From logits to probabilities
  - 2.2 Greedy decoding and beam search
  - 2.3 Pure sampling and the long tail
  - 2.4 Temperature
  - 2.5 Truncation - top-k, nucleus and min-p
  - 2.6 Contrastive methods
  - 2.7 Three choices of weak predictor
  - 2.8 Classifier-free guidance and the cost of contrast
  - 2.9 Constrained decoding
  - 2.10 The gap that remains after decoding
- 3. In-Context Learning and Prompting
  - 3.1 In-context learning
  - 3.2 View one - the model infers which task it is in
  - 3.3 Worked example - capital or largest city
  - 3.4 How the transformer implements the Bayesian view
  - 3.5 View two - the forward pass runs a learning algorithm
  - 3.6 Worked example - learning y = 2x in one pass
  - 3.7 In-context learning is powerful but brittle
  - 3.8 Chain of thought
  - 3.9 Why chain of thought works
  - 3.10 Variants that add structure to the chain
  - 3.11 The cost of a chain
- 4. The System Around the Model
  - 4.1 The baseline - one prompt, one response
  - 4.2 The token-generation loop
  - 4.3 Model, decoder, controller and evaluator
  - 4.4 Who implements what
- 5. Test-Time Compute
  - 5.1 Six ways to allocate extra effort
  - 5.2 A fixed budget and a cost model
  - 5.3 Inference procedures and the controller
  - 5.4 What an adaptive controller uses
  - 5.5 Two levels of control
  - 5.6 Computation and latency are different
- 6. Sample and Aggregate
  - 6.1 How to sample and how to combine
  - 6.2 Self-consistency
  - 6.3 Worked example - five chains on the pen problem
  - 6.4 Majority of three chains
  - 6.5 Majority of five chains
  - 6.6 The general majority formula
  - 6.7 With two answers, voting cuts both ways
  - 6.8 Plurality - real questions have many wrong answers
  - 6.9 When voting helps, and what it costs
  - 6.10 Universal self-consistency
  - 6.11 Best-of-N - replace the vote with a scorer

---

## 1. Why Inference Time

### 1.1 The gap between what a model can do and what it does

The lecture's question is: **we already have a trained language model; can we obtain better answers from it without changing its parameters?**

The professor's analogy is a student sitting the same exam twice. The first time, every question must be answered instantly, in ink, with no scratch paper, no calculator and no textbook. The second time, the student may use scratch paper and a calculator, look things up, try a hard question several ways, and check the answer before handing it in. **The student's knowledge is identical on both days. The score is not.**

A trained model is in the same position. Its weights hold a fixed body of knowledge and skill, but how much of that skill shows up in the answer depends on the conditions under which we ask. **Inference-time methods** are everything we do after training to close the gap between what a model can do and what it does by default. The weights stay frozen; we change the conditions of the exam.

The gap is not small. The same frozen model can score very differently on the same task depending on how it is used, and that difference is **often larger than the gap between one model generation and the next**.

### 1.2 Three reasons to work at this layer

1. **The weights are often out of reach.** Most strong models sit behind an **API** (Application Programming Interface): we send text and receive text, nothing else. Any improvement has to come from outside.
2. **Training is slow and expensive; inference-time changes are immediate and reversible.** Fine-tuning needs data, compute and evaluation, and a bad fine-tune is costly to undo. A changed prompt, an added verification step or a different sampling setting takes effect on the next request and can be reverted on the one after.
3. **Some capabilities are easier to build around a model than into it.** Fresh knowledge changes daily, so it is better retrieved than memorized. Exact arithmetic is better delegated to a calculator. Long, deliberate reasoning needs working space that a single pass through the network does not provide.

### 1.3 The four levers

If the weights cannot change, only four things can, and every method in this topic pulls one or more of them:

![[L11_four_levers.png|650]]

*The four levers around a frozen model. The input (lever 2) and the readout (lever 1) sit on either side of the model; the computation (lever 3) is the loop that feeds outputs back in; the system (lever 4) is everything around the model: tools, retrievers, tests and judges.*

| Lever | What changes | Covered in |
|---|---|---|
| **The readout** | how tokens are chosen from the output distribution | [[#2. Decoding\|§2]] |
| **The input** | what the model sees in its context window | [[#3. In-Context Learning and Prompting\|§3]] |
| **The computation** | how many tokens and calls are spent per answer, and how they are organized | [[#5. Test-Time Compute\|§5]], [[#6. Sample and Aggregate\|§6]] |
| **The system** | what surrounds the model: tools, verifiers, caches, routers | [[#4. The System Around the Model\|§4]] |

### 1.4 Fixed computation per token

One observation explains most of the lecture. A transformer with $L$ layers pushes every token through the same $L$ layers, whether the question is "what is 2 + 2" or a hard proof. So **a single forward pass gives a fixed amount of computation per token**, however difficult the problem.

There are only two ways to give a hard problem more computation without touching the weights:

1. **Produce more tokens.** Every generated token is another forward pass through the decoder, and that pass can read everything written so far. Writing intermediate work is therefore a way of thinking longer. This happens **inside** the transformer.
2. **Make more calls.** Ask the model several times, in parallel or in sequence, and combine the results. This happens **outside** the transformer.

*(These are the "sequential" and "parallel" families of [[ATML_Lecture_12 - Test-Time Scaling and Multimodal Reasoning#1.4 Four families of test-time scaling|ATML L12 §1.4]].)*

![[ATML_L12_tts_taxonomy.svg|640]]

*Reused from [[ATML_Lecture_12 - Test-Time Scaling and Multimodal Reasoning#1.2 Four design questions|ATML L12 §1.2]]. (a) The four design questions of test-time scaling. (b) The four families: "more tokens" here is the sequential family, "more calls" is the parallel family, and adding a verifier to choose among calls is the hybrid family. Latent scaling (thinking in hidden states) is not covered in this lecture.*

Both create a new problem: once there are many candidate steps or answers, we must decide which to trust. So most methods come as a pair, a way to spend more computation in a structured form and a way to select among the results. The professor's slogan is **generate more, then select well.**

### 1.5 Roadmap

Each method closes a gap left by the one before it. The whole topic spans both parts of the lecture; this note covers the first three rows and the start of the third.

| Topic | The idea | The gap that remains |
|---|---|---|
| Decoding | choose each token more wisely | works one token at a time, so it cannot repair a wrong plan |
| Prompting | change what the model is asked to do, and let it write its reasoning | one chain of reasoning is one sample, and nothing checks it |
| Sample and aggregate | draw many chains and vote or score | independent samples waste compute on chains that failed early |
| Search | evaluate partial solutions and expand only the promising ones | the evaluator is usually the same model that made the mistake |
| Refinement loops | improve one answer using feedback, ideally from outside the model | still scaffolding that we must design by hand |
| Reasoning models | train the model to do the scaffolding internally | no amount of thinking produces a fact the model never saw |
| Context augmentation | bring missing knowledge in through the context window | treats the model as a black box |
| Activation steering | intervene on internal activations (see [[RA3_Actionable_MI#4. Steering Methods\|RA3 §4]]) | every method so far costs more tokens |
| Efficiency | make each token cheaper | the frozen-weights definition has fuzzy edges |
| Boundary cases and agents | test the definition, then run the levers in a loop | leads into the next unit on agents |

![[CS5326_L11_method_map.svg|760]]

*How the methods of this note relate. The four levers sit around the frozen model: the input (§3) and readout (§2) on either side, the system (§4) that runs the loop above it, and the computation (§5) below it. Computation splits into more tokens inside the model (chain of thought) and more calls outside it (sampling). One chain is one unchecked sample, so the dashed arrow leads to the three ways of selecting among many samples (§6): agreement, a scorer, or a judge.*

---

## 2. Decoding

The readout is the lever closest to the model, so the lecture starts there. Most of this section is revision of [[ATML_Lecture_7 - LLM Foundations#6. Decoding|ATML L7 §6]] and the next-token distribution of [[Lecture_06_Decoding_KV_Cache_MQA_GQA_FlashAttention#2.2 Next-token distribution and the causal mask|Lecture 6 §2.2]]; contrastive and constrained decoding (§2.6–§2.9) are new.

### 2.1 From logits to probabilities

At every step the model produces one real number per vocabulary token, the **logits** $z_1, z_2, \dots, z_V$, where $V$ is the vocabulary size. A larger logit means the model favours that token more. Logits can be negative and do not sum to anything in particular, so the **softmax** turns them into probabilities:

$$p_i = \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}} \qquad\qquad\text{(1)}$$
^eq-1

The exponential makes every score positive and stretches the differences between scores; dividing by the sum makes the results add to 1, so they form a probability distribution. **Decoding** is the rule that turns this distribution into one emitted token. That token is appended to the input and the process repeats.

**Worked example.** Four tokens A, B, C, D have logits $2, 1, 0, -1$. Exponentiating gives $e^2 = 7.389$, $e^1 = 2.718$, $e^0 = 1$, $e^{-1} = 0.368$, which sum to $11.475$. Dividing, $p_A = 0.644$, $p_B = 0.237$, $p_C = 0.087$, $p_D = 0.032$.

Each drop of 1 in the logit divides the probability by $e \approx 2.72$. The reason is that for any two tokens the shared denominator cancels:

$$\frac{p_i}{p_j} = \frac{e^{z_i}}{e^{z_j}} = e^{z_i - z_j} \qquad\qquad\text{(2)}$$
^eq-2

So the ratio depends only on the gap $z_i - z_j$, not on the logits themselves or on the other tokens. A gap of 1 gives a ratio of $e$, a gap of 2 gives $e^2 \approx 7.39$. **A fixed gap in logits is a fixed ratio in probabilities.**

### 2.2 Greedy decoding and beam search

**Greedy decoding** picks the highest-probability token at every step. **Beam search** keeps the $b$ most probable partial sequences (the beam width $b$) and extends them all, which approximates the single most probable complete sequence. Both chase likelihood ([[ATML_Lecture_7 - LLM Foundations#6.2 Greedy decoding|ATML L7 §6.2]], [[ATML_Lecture_7 - LLM Foundations#6.4 Beam search|§6.4]]).

This works when there is essentially one right output, as in translation. In open-ended generation it fails recognizably: **the text becomes bland and falls into loops.** The most probable continuation is the most generic one, and once a phrase has been repeated, repeating it again becomes even more probable. **Human text is not the most probable text**; it contains a steady stream of mildly surprising choices.

### 2.3 Pure sampling and the long tail

**Sampling** restores that surprise by drawing the next token at random according to the probabilities. Pure sampling has the opposite problem. The vocabulary has tens of thousands of tokens, and the long tail of very unlikely tokens collectively carries a noticeable share of the probability. The model's estimates in that tail are unreliable, and one bad draw can derail the rest of the text. The practical sampling methods below try to keep the tail under control.

### 2.4 Temperature

**Temperature** $T$ rescales the logits before the softmax:

$$p_i(T) = \frac{e^{z_i/T}}{\sum_{j} e^{z_j/T}} \qquad\qquad\text{(3)}$$
^eq-3

Comparing two tokens, the denominators cancel as in [[#^eq-2|eq. (2)]]:

$$\frac{p_i(T)}{p_j(T)} = e^{(z_i - z_j)/T}$$

The ratio depends only on the logit gap divided by $T$, so temperature stretches or shrinks every gap by the same factor:

| $T$ | Effect on gaps | Distribution |
|---|---|---|
| $T < 1$ | larger | sharpens; the favourite pulls ahead. As $T \to 0$ this becomes greedy decoding |
| $T = 1$ | unchanged | the model's own distribution |
| $T > 1$ | smaller | flattens toward uniform; output becomes more varied and more error-prone |

**Worked example, continued.** With logits $2, 1, 0, -1$ the A–B gap is 1.
- At $T = 1$, $p_A/p_B = e^1 = 2.72$ and the distribution is $(0.644, 0.237, 0.087, 0.032)$.
- At $T = 0.5$ the gap doubles to 2, the ratio becomes $e^2 = 7.39$, and the distribution sharpens to $(0.865, 0.117, 0.016, 0.002)$.
- At $T = 2$ the gap halves to 0.5, the ratio becomes $e^{0.5} = 1.65$, and the distribution flattens to $(0.455, 0.276, 0.167, 0.102)$.

### 2.5 Truncation - top-k, nucleus and min-p

Temperature reshapes the whole distribution but **never removes the tail**. **Truncation** methods cut the tail off, renormalize what is left so it sums to 1, and sample from that.

| Method | Keeps | Behaviour |
|---|---|---|
| **Top-$k$** | the $k$ most probable tokens | simple but rigid: $k = 50$ is too many when the model is certain and too few when many continuations are reasonable |
| **Nucleus (top-$p$)** | the smallest set of top tokens whose probabilities add to at least $p$ | the set shrinks when the model is confident and grows when it is unsure |
| **Min-$p$** | every token with probability at least a fraction of the top token's probability | the threshold scales with the model's confidence |

*(Expanding on this: top-$p$ adapts through the shape of the whole distribution, min-$p$ through the top token alone. When one token dominates, min-$p$'s threshold rises and it cuts hard; when the distribution is flat, the threshold falls and many tokens survive. See also [[ATML_Lecture_7 - LLM Foundations#6.5 Top-$k$ sampling|ATML L7 §6.5]]–[[ATML_Lecture_7 - LLM Foundations#6.6 Top-$p$ (nucleus) sampling|§6.6]].)*

**Worked example.** Six tokens have probabilities $0.50, 0.25, 0.12, 0.08, 0.03, 0.02$.
- **Top-$k$, $k = 3$:** keeps the first three, total $0.87$. Renormalized: $0.50/0.87 = 0.575$, $0.25/0.87 = 0.287$, $0.12/0.87 = 0.138$.
- **Top-$p$, $p = 0.9$:** cumulative sums are $0.50, 0.75, 0.87, 0.95$. The total first reaches 0.9 at the fourth token, so **four** tokens are kept.
- **Min-$p$, ratio 0.2:** threshold $0.2 \times 0.50 = 0.10$. Tokens with probability at least 0.10 survive: the first three.

![[L11_temperature_truncation.png|650]]

*Left: the same logits A–D at $T = 0.5, 1, 2$; lower temperature makes A taller and the rest shorter. Right: the six-token example, with the top-$k$ ($k=3$) and min-$p$ (0.2) cutoffs after $t_3$ and the top-$p$ ($p=0.9$) cutoff after $t_4$. Tokens right of a cutoff are removed before sampling.*

### 2.6 Contrastive methods

Truncation removes tokens the model considers unlikely. It does nothing about tokens the model considers **likely for the wrong reasons**: the generic phrase, the repetition, the popular misconception. We need a way to tell "likely because the model knows something" from "likely because it is the easy thing to say".

The idea has three parts:
1. Ask a **strong predictor** and a deliberately **weak predictor** for the next token, and prefer tokens that only the strong one favours.
2. Both predictors share the same bad habits, and the weak one has little else.
3. Subtracting the weak one cancels the habits and leaves what the strong one specifically knows.

For each candidate token $x$ the **contrastive score** is

$$\text{score}(x) = \log p_{\text{strong}}(x) - \log p_{\text{weak}}(x) \qquad\qquad\text{(4)}$$
^eq-4

and we choose among the highest-scoring tokens. A difference of logarithms is the logarithm of a ratio, so the score asks one question: **how many times more likely is this token under the strong predictor than under the weak one?** A token both like equally scores zero, however likely it is.

**The plausibility safeguard.** A token the strong model gives one in a billion and the weak model one in a trillion would get a huge score, even though the strong model plainly does not want it. So the score is applied only to tokens the strong model already finds plausible, for example those with at least one tenth of the top token's probability; everything else is discarded first. *(This is a min-$p$ filter from [[#2.5 Truncation - top-k, nucleus and min-p|§2.5]] with ratio 0.1, applied before the contrast.)*

**Worked example (illustrative numbers).** The text so far is "Barack Obama was born in Honolulu, Hawaii. He was born in".

| Token | $p_{\text{strong}}$ | $p_{\text{weak}}$ | Score $\ln(p_s/p_w)$ |
|---|---|---|---|
| Hawaii | 0.40 | 0.60 | $\ln(0.40/0.60) = -0.41$ |
| 1961 | 0.30 | 0.05 | $\ln(0.30/0.05) = +1.79$ |
| the | 0.20 | 0.10 | $\ln(0.20/0.10) = +0.69$ |

Greedy decoding on the strong model alone would pick "Hawaii", a repetition. The weak model likes "Hawaii" even more, so it is penalized; only the strong model knows the year, so "1961" is rewarded. All three pass the plausibility check (threshold $0.1 \times 0.40 = 0.04$), and **the winner is 1961**.

![[L11_contrastive_decoding.png|650]]

*Left: the weak model shares the strong model's pull toward repeating "Hawaii" but knows almost nothing about "1961". Right: subtracting log-probabilities gives Hawaii $-0.41$, 1961 $+1.79$, the $+0.69$, rewarding the token only the strong model supports.*

### 2.7 Three choices of weak predictor

The same recipe gives three well-known methods, depending on what plays the weak role:

| Method | The weak predictor | What the contrast isolates |
|---|---|---|
| **Contrastive decoding** | a much smaller model | what the large model knows beyond the small one |
| **DoLa** (Decoding by cOntrasting LAyers) | an earlier layer of the same model, read out through the output head | what the later layers added, which is often the factual knowledge |
| **Classifier-free guidance** | the same model run without the prompt | what the prompt contributed, so the output follows the prompt more closely |

*(Expanding on this: "read out through the output head" is the **logit lens** of [[RA3_Actionable_MI#3.5 Vocabulary projection|RA3 §3.5]]: an intermediate residual-stream state is multiplied by the unembedding matrix to get a next-token distribution early. DoLa uses that early distribution as the weak predictor.)*

### 2.8 Classifier-free guidance and the cost of contrast

**Classifier-free guidance (CFG)** is usually written on logits, with a strength $\gamma$:

$$z_{\text{guided}} = z_{\text{without prompt}} + \gamma\,\big(z_{\text{with prompt}} - z_{\text{without prompt}}\big) \qquad\qquad\text{(5)}$$
^eq-5

Read left to right: start from what the model would say with no prompt, then move in the direction the prompt pushes, $\gamma$ times as far. With $\gamma = 1$ the two unprompted terms cancel and we recover ordinary decoding. With $\gamma > 1$ we deliberately overshoot in the prompt's direction. *(Expanding on this: CFG is the logit form of [[#^eq-4|eq. (4)]]. Taking logs of softmaxes, $\log p_{\text{guided}} = \log p_{\text{w/o}} + \gamma(\log p_{\text{with}} - \log p_{\text{w/o}})$ up to a constant, so the contrastive score is the $\gamma$-weighted term. The name comes from diffusion models, where the same trick strengthens the effect of the text condition.)*

**Worked example.** For three tokens, the logits with the prompt are $(2, 1, 0)$ and without it $(1.5, 1.2, 0)$. The prompt raises the first token by 0.5 and lowers the second by 0.2. With $\gamma = 1.5$:
- first token: $1.5 + 1.5 \times 0.5 = 2.25$;
- second token: $1.2 + 1.5 \times (-0.2) = 0.9$;
- third token: $0$.

After the softmax, the first token's probability rises from $0.665$ under ordinary decoding ($e^2/(e^2 + e^1 + 1) = 7.389/11.107$) to $0.733$ ($e^{2.25}/(e^{2.25} + e^{0.9} + 1) = 9.488/12.948$). The prompt's preference has been amplified.

**Cost and limits of all three contrastive methods.** They need access to logits, which most closed APIs do not expose. They usually need a second forward pass. Pushed too hard, the contrast damages fluency, because the result is no longer the distribution of any real model.

### 2.9 Constrained decoding

The second extension addresses a different failure. Agents work by emitting **structured output**: a tool call, a **JSON** (JavaScript Object Notation) object, a line of **SQL** (Structured Query Language). A single missing brace breaks the program that reads it. Asking politely in the prompt works most of the time, and "most of the time" is not good enough for software.

**Constrained decoding** removes, before sampling each token, every token that would break the required format:
1. A **grammar** or a **JSON schema** defines what valid output looks like.
2. A small **parser** tracks the output so far and lists which tokens may legally come next.
3. The logits of all other tokens are set to $-\infty$, so their probability becomes exactly zero ($e^{-\infty} = 0$ in [[#^eq-1|eq. (1)]]). The model then samples from the legal tokens in the usual way.

**Worked example: a two-option schema.** The schema allows exactly `{"sentiment": "pos"}` or `{"sentiment": "neg"}`. At the first step only the opening `{"` is legal, so it is forced whatever the model would have preferred. The same holds for `sentiment` and the punctuation after it. Then the parser reaches the one place where the schema offers a choice: two tokens are legal, and the model's own probabilities decide between `pos` and `neg`. The closing characters are forced again. **Out of the whole output, the model made one real decision.**

![[L11_constrained_decoding.png|650]]

*Grey boxes are forced because only one token is legal; the model's probabilities matter only at the green pos/neg choice. The red dashed tokens ("The", "I think", "maybe", "positive!", …) are what an unconstrained model might emit; the mask sets their probability to zero.*

**What you get:** output that is **guaranteed to parse**. This is the mechanism behind strict structured outputs and strict tool calling, and it is why tool calls in agents are reliable.

**What you do not get:** correct content. The format is guaranteed and the values are not. There is also a subtler cost: **a format that forces the answer first leaves the model no room to reason.** If the schema has a reasoning field, place it **before** the answer field, so the reasoning is generated first and the answer can depend on it. *(This is chain of thought, [[#3.9 Why chain of thought works|§3.9]], applied inside a schema: tokens written earlier are context for tokens written later, never the other way round.)*

**The two extensions side by side.** Contrastive decoding **re-weights tokens softly**; constrained decoding **forbids tokens outright**. Both operate on the logits at the moment of readout, and they can be combined.

### 2.10 The gap that remains after decoding

Decoding acts one token at a time. It can polish how an answer is expressed, but **it cannot repair a wrong plan**: if the model has set off toward the wrong answer, a better choice of the next word will not turn it around. (The slide illustrates this with an animated GIF of a cartoon character laying track in front of a moving train, which does not reproduce in a static note.) For that, we have to change what the model is asked to do.

---

## 3. In-Context Learning and Prompting

Decoding changed how we read the model's output. This section pulls the second lever of [[#1.3 The four levers|§1.3]] and changes the input.

### 3.1 In-context learning

With the right text in the context window, a frozen model behaves as if it had been trained for a new task. Place a few worked examples of a task in the prompt, and the model will often perform it on a new input, with no gradient update. This is **in-context learning (ICL)**.

**Example prompt.**
- Review: "The battery died within a week." Label: negative
- Review: "Setup took two minutes and it just works." Label: positive
- Review: "The screen cracked on the first drop." Label:

The model continues with "negative". No one trained it on this labelling scheme; the examples alone told it what to do.

This looks strange because in classical machine learning, learning means changing parameters. Here nothing changes, yet behaviour adapts to the task. Two explanations have been proposed, and they are **complementary views of the same phenomenon**.

### 3.2 View one - the model infers which task it is in

During pretraining the model has seen an enormous variety of tasks: translation, question answering, labelling, list completion. Think of each as a hidden task $\theta$ (here $\theta$ is a task label, not the model's parameters). The examples in the prompt are evidence about which $\theta$ is in play. The model's prediction for a new input then behaves like a **weighted average over tasks**, each weighted by how well it explains the examples:

$$p(y \mid x, D) = \sum_{\theta} p(y \mid x, \theta)\; p(\theta \mid D) \qquad\qquad\text{(6)}$$
^eq-6

Here $D = \{(x_i, y_i)\}_{i=1}^{n}$ is the set of $n$ examples, $x$ is the new input and $y$ the output. The first factor asks "if the task were $\theta$, what would the answer be?"; the second asks "given the examples, how likely is it that the task is $\theta$?". The second factor comes from **Bayes' rule**:

$$p(\theta \mid D) = \frac{p(D \mid \theta)\,p(\theta)}{p(D)} = \frac{p(\theta)\prod_{i=1}^{n} p(y_i \mid x_i, \theta)\,\prod_i p(x_i)}{p(D)} \;\propto\; p(\theta)\prod_{i=1}^{n} p(y_i \mid x_i, \theta) \qquad\qquad\text{(7)}$$
^eq-7

Each example multiplies in one more term. A task that explains an example well keeps its weight; a task that explains it badly loses weight quickly, because the terms multiply.

**Derivation of [[#^eq-7|eq. (7)]].** Bayes' rule with hypothesis $\theta$ and evidence $D$ gives $p(\theta \mid D) = p(D \mid \theta)p(\theta)/p(D)$. By the chain rule over the examples,

$$p(D \mid \theta) = \prod_{i=1}^{n} p\big(x_i, y_i \mid \theta, (x_1, y_1), \dots, (x_{i-1}, y_{i-1})\big).$$

Assumption **A1** is that, given the task, the examples are independent of one another, so the history drops out and $p(D \mid \theta) = \prod_i p(x_i, y_i \mid \theta)$. The product rule inside each factor gives $p(x_i, y_i \mid \theta) = p(x_i \mid \theta)\,p(y_i \mid x_i, \theta)$. Assumption **A2** is that the inputs alone say nothing about the task, so $p(x_i \mid \theta) = p(x_i)$. Collecting terms,

$$p(\theta \mid D) = p(\theta)\prod_{i=1}^{n} p(y_i \mid x_i, \theta)\;\times\;\underbrace{\frac{\prod_i p(x_i)}{p(D)}}_{\text{no } \theta \text{ in it}}.$$

The bracket is the same for every $\theta$, which is what $\propto$ hides. Dividing by $\sum_{\theta'} p(\theta')\prod_i p(y_i \mid x_i, \theta')$ restores the equality; that is the "multiply and renormalize" step, the "divide by 0.46" in the example of [[#3.3 Worked example - capital or largest city|§3.3]].

The companion formula [[#^eq-6|eq. (6)]] is the **sum rule** (law of total probability) over $\theta$, plus two more uses of the same assumptions: given $\theta$ and $x$, the answer does not depend on the examples, and the query input alone does not shift the posterior (A2 again).

In words: **start with the prior belief in each task, multiply by how well that task explains each example, then rescale so the weights sum to 1.** One caveat: real models only approximate this. For example, they are sensitive to the order of examples, while the formula is not (a product does not care about order).

### 3.3 Worked example - capital or largest city

Suppose only two tasks are plausible: $\theta_1$ = "name the capital of the country" and $\theta_2$ = "name the largest city of the country". Start with equal prior weight, 0.5 each. Assume a task produces its own correct answer with probability 0.9 and some other given city with probability 0.02.

**Example 1: France → Paris.** Paris is both the capital and the largest city, so both tasks explain it equally well (0.9 each):

$$p(\theta_1 \mid \text{Paris}) = \frac{0.9 \times 0.5}{0.9 \times 0.5 + 0.9 \times 0.5} = 0.5, \qquad p(\theta_2 \mid \text{Paris}) = 0.5.$$

This example carried **no information** about which task we are in.

**Example 2: Australia → Canberra.** Canberra is the capital but not the largest city, so the likelihood is 0.9 under $\theta_1$ and 0.02 under $\theta_2$:

$$p(\theta_1 \mid \text{Canberra}) = \frac{0.9 \times 0.5}{0.9 \times 0.5 + 0.02 \times 0.5} = \frac{0.45}{0.46} = 0.978, \qquad p(\theta_2 \mid \text{Canberra}) = \frac{0.01}{0.46} = 0.022.$$

**Query: Turkey → ?** For the query the formula is the **law of total probability** ([[#^eq-6|eq. (6)]]), not Bayes' rule again. Bayes' rule gave the task weights (0.978 and 0.022); this step averages each task's answer using those weights, with $D$ = the two examples.

Ankara is the capital, so it is correct under $\theta_1$ and "some other city" under $\theta_2$:

$$P(\text{Ankara} \mid D) = 0.9 \times 0.978 + 0.02 \times 0.022 = 0.8802 + 0.0004 = 0.881.$$

Istanbul is the largest city, so it is correct under $\theta_2$ and "some other city" under $\theta_1$:

$$P(\text{Istanbul} \mid D) = 0.02 \times 0.978 + 0.9 \times 0.022 = 0.0196 + 0.0198 = 0.039.$$

One well-chosen example moved the model from undecided to confident.

![[CS5326_L11_bayes_icl.svg|640]]

*The capital-or-largest-city example as bar charts. Left: the task posterior stays at 0.5/0.5 after Paris, which both tasks explain, and jumps to 0.978/0.022 after Canberra, which only the capital task explains. Right: averaging the two tasks' answers with those weights gives Ankara 0.881 and Istanbul 0.039.* The practical lesson: **examples consistent with several tasks teach nothing. Choose examples that tell tasks apart.**

### 3.4 How the transformer implements the Bayesian view

It does not implement it explicitly: there is no stored list of tasks, no prior and no Bayes step inside the network. The formula describes what the output **must look like**, for one reason. The model is trained to predict the next token, and **the best possible next-token predictor on text generated by many hidden tasks is exactly "guess the task from the context, then answer under that guess, weighted by how sure you are."** Training pushes the network toward that behaviour. Inside, attention gathers evidence from each example and the residual stream accumulates it, which is the "multiply in one more term per example" step done as a running sum of logs ($\log \prod = \sum \log$).

### 3.5 View two - the forward pass runs a learning algorithm

In the second view, the examples in the prompt act as a small **training set** and the forward pass acts as the **learner**. The model does not look up a task it already knows; it fits the examples on the spot.

- **How attention makes this possible.** The query position can look at every example, compare itself with each, and combine their labels. That is the core move of many learning algorithms: predict for a new input from the labelled examples most relevant to it.
- **Layers act as iterations.** Each layer refines an internal estimate of the rule linking inputs to outputs; more layers allow more rounds of refinement, like more iterations of a training loop.
- The weights stay frozen. **They store the learning procedure**, acquired during pretraining. The solution to the current task lives only in the activations and disappears when the prompt ends.

This has been demonstrated for small transformers on simple tasks, where they behave like standard learning algorithms. The professor demonstrates it with linear regression.

The examples are pairs $(x_i, y_i)$ and we want a weight vector $w$ with $w \cdot x_i \approx y_i$. The squared-error loss over $n$ examples is

$$L(w) = \frac{1}{2n}\sum_{i=1}^{n}(w \cdot x_i - y_i)^2.$$

Differentiating each squared term brings down a factor of 2, which cancels the $\tfrac12$, and the chain rule contributes $x_i$:

$$\nabla L(w) = \frac{1}{n}\sum_{i=1}^{n}(w \cdot x_i - y_i)\,x_i.$$

Now take **one gradient step from zero**. At $w_0 = 0$ every $w_0 \cdot x_i$ is zero and each bracket becomes $-y_i$, so with learning rate $\eta$

$$w_1 = w_0 - \eta\,\nabla L(w_0) = \frac{\eta}{n}\sum_{i=1}^{n} y_i\,x_i.$$

After one step, the weight vector is a sum of the example inputs, each scaled by its label. Predicting for a new query $x_q$ by a dot product,

$$\hat y_q = w_1 \cdot x_q = \frac{\eta}{n}\sum_{i=1}^{n} y_i\,(x_i \cdot x_q). \qquad\qquad\text{(8)}$$
^eq-8

**Recognize the shape.** The prediction is a weighted sum of the labels $y_i$, where the weight on each label is $x_i \cdot x_q$, the similarity between that example's input and the query. That is exactly what an attention layer computes ([[Lecture_05_Attention_and_Multi_Head_Attention#7.5 Step 4: retrieve, and the full formula|Lecture 5 §7.5]]): the query is compared with each key by a dot product, and the values are added in proportion. **Take the keys to be the example inputs and the values the example labels, and a single (linear) attention layer performs one step of gradient descent on the examples.** *(Expanding on this: "linear" means without the softmax; ordinary softmax attention normalizes the weights $x_i \cdot x_q$, so the match is exact only for the linear variant.)*

### 3.6 Worked example - learning y = 2x in one pass

One-dimensional data from the rule $y = 2x$: examples $(1, 2), (2, 4), (3, 6)$ and query $x_q = 4$.

- Similarities to the query: $1 \times 4 = 4$, $2 \times 4 = 8$, $3 \times 4 = 12$. The most similar example gets the largest weight.
- Weighted sum of labels: $2 \times 4 + 4 \times 8 + 6 \times 12 = 8 + 32 + 72 = 112$.
- Scale by $\eta/n$. Choosing $\eta = 3/14$ gives $\eta/n = 1/14$, so the prediction is $112/14 = 8$, the right answer for $y = 2x$.

No parameter was updated; the "learning" happened in the arithmetic of one pass over the examples.

![[CS5326_L11_icl_gd_attention.svg|720]]

*The y = 2x example read as attention. The query $x_q = 4$ is compared with each key (example input) by a dot product, giving scores 4, 8, 12. These weight the values (labels 2, 4, 6), the weighted sum is 112, and scaling by $\eta/n = 1/14$ gives the answer 8. This is exactly one gradient-descent step from $w_0 = 0$ followed by a prediction.* *(Expanding on this: $\eta$ was chosen to make the answer exact. One gradient step from zero only lands on the true $w = 2$ when $\eta/n = 1/\sum_i x_i^2 = 1/14$; with another $\eta$ the prediction would have the right direction but the wrong scale, and more layers, like more steps, would correct it.)*

**How far this goes.** For simple function classes such as linear regression, researchers have constructed transformer weights that implement this computation, and trained transformers behave similarly. For real language tasks the Bayesian view is usually the better description. The professor's honest summary: **both mechanisms are established in controlled settings, and how they combine in large models is still being worked out.**

### 3.7 In-context learning is powerful but brittle

Results shift with details that should not matter:
- **Which examples are chosen.** Different sets of examples for the same task can give very different accuracy.
- **Their order.** The same examples in a different order can change the result; models tend to over-weight the most recent examples. (This is the order sensitivity that [[#3.2 View one - the model infers which task it is in|§3.2]] noted the ideal formula lacks.)
- **Their format and label balance.** If most examples carry one label, the model leans toward that label.

One improvement is reliable: instead of a fixed set of examples, **retrieve the examples most similar to the current query**. This is per-query demonstration selection, sometimes called **retrieval-based few-shot prompting**, as in **KATE** (Liu et al.; *(expanding on this:)* KATE embeds the training examples and the query with a sentence encoder and puts the $k$ nearest neighbours in the prompt). Both views of ICL predict that this helps: similar examples are stronger evidence about the right task, and they get large weights $x_i \cdot x_q$ in [[#^eq-8|eq. (8)]].

### 3.8 Chain of thought

**Motivation.** Some questions need several steps. A forward pass does a fixed amount of computation per token ([[#1.4 Fixed computation per token|§1.4]]). If the model must emit the final answer immediately, all the steps have to fit inside the passes that produce those few answer tokens.

**Chain of thought (CoT)** asks the model to write its intermediate steps before the answer, either by showing worked examples that include reasoning or simply by adding an instruction such as "think step by step". (Same idea as [[ATML_Lecture_12 - Test-Time Scaling and Multimodal Reasoning#2.1 The reasoning trace as intermediate context|ATML L12 §2.1]] and [[ATML_Lecture_9 - RLVR, Reasoning LLMs, and Using LLMs#6.2 Prompting techniques|ATML L9 §6.2]].)

**Worked example: the pen problem.** "A shop sells pens at 3 for 20 rupees. Ali buys 12 pens and pays with a 100 rupee note. How much change does he get?"
- **Direct answer:** the model must produce a number at once. A common slip is 40 (from $100 - 3 \times 20$), using the numbers in the question without working out how many groups there are.
- **Chain of thought:** "12 pens is $12/3 = 4$ groups of three. Four groups cost $4 \times 20 = 80$ rupees. The change is $100 - 80 = 20$ rupees. The answer is 20."

Each line is easy. The difficulty was in holding them all at once.

### 3.9 Why chain of thought works

Every token the model writes is appended to the context, and the next forward pass can read it through attention. Writing "4 groups" stores an intermediate result **outside the network**, where later passes can pick it up. The context becomes a **scratchpad**, and the amount of sequential computation grows with the number of tokens written instead of being fixed by the number of layers.

![[L11_cot_scratchpad.png|650]]

*Top: a direct answer must be computed in one bounded pass, with no place to store intermediate results. Middle: with chain of thought, each written step (12/3 = 4, 4 × 20 = 80, 100 − 80 = 20) is read back by later passes. Bottom: zooming into step 1, one forward pass per token; the pass that emits "4" reads "12", "/" and "3" back through attention, so depth grows with the number of tokens.*

Theoretical work supports this picture: **there are problems that a fixed-depth transformer cannot solve in one pass but can solve when allowed to write intermediate steps.** *(Expanding on this: the standard results, e.g. Feng et al. 2023 and Merrill & Sabharwal 2024, show that a constant-depth transformer answering immediately is limited to "shallow" parallel computations, while one that writes a number of CoT steps growing with input size can simulate any step-by-step algorithm of that length.)*

### 3.10 Variants that add structure to the chain

- **Least-to-most** prompting first asks the model to break the problem into easier subproblems, then solves them in order, feeding each answer into the next. For the pen problem: "How many groups of three? What do they cost? What is the change?"
- **Plan-and-solve** separates planning from execution: first write a plan, then carry it out. This reduces skipped steps. ([[ATML_Lecture_12 - Test-Time Scaling and Multimodal Reasoning#2.3 Plan-and-Solve|ATML L12 §2.3]] adds the catch: a wrong plan contaminates the whole trajectory.)
- **Step-back** prompting first asks for the general principle behind the question ("what determines the total price when items are sold in groups?") and then applies it to the specific case.

All three keep a single chain and change only its structure; least-to-most and plan-and-solve both decompose before solving, while step-back abstracts before solving.

### 3.11 The cost of a chain

A chain is only as strong as its weakest link. If each step is correct with probability $q$ and a problem needs $k$ steps, all correct, then because independent probabilities multiply

$$P(\text{chain correct}) = q^{k}. \qquad\qquad\text{(9)}$$
^eq-9

With $q = 0.95$ and $k = 10$ this is $0.95^{10} \approx 0.60$: **a model that is right 95% of the time per step fails four problems in ten at ten steps.** This formula motivates every method that follows.

![[CS5326_L11_chain_reliability.svg|560]]

*Chain reliability $q^k$ against the number of steps. Even at 95% per step the chance of a fully correct chain falls to 0.60 at 10 steps and about 0.21 at 30; at 90% per step it is below 0.35 by step 10. Long chains need per-step accuracy very close to 1, or a way to check and recover, which is what §6 provides.*

**Limits, and the gap that remains.** In all of these methods the weights never move; only the text does. However good the prompt, **one chain of thought is one sample**. An early mistake propagates to the end ([[ATML_Lecture_12 - Test-Time Scaling and Multimodal Reasoning#2.2 Where a single chain fails|ATML L12 §2.2]]), [[#^eq-9|eq. (9)]] makes long chains fragile, and **nothing checks the result**.

---

## 4. The System Around the Model

Before turning to multiple chains, the lecture names the parts of the system that would run them (the professor's "Other parts of the picture: the tools governing the LLM"). This is the fourth lever of [[#1.3 The four levers|§1.3]], and it tells us where each later method intervenes.

### 4.1 The baseline - one prompt, one response

**Example:** "Three boxes contain twelve pencils each. Five students receive two pencils each. How many pencils remain?" The baseline prompt is "Answer the following question: …", and a possible response is **26** ($3 \times 12 - 5 \times 2 = 36 - 10$). At the application level this looks simple: construct a prompt, request one generated response, return it.

### 4.2 The token-generation loop

One generation usually involves many token-generation steps. For a standard **autoregressive** model (one that generates left to right, each token conditioned on all earlier ones), let $x$ be the prompt, $y_t$ the token generated at step $t$, $y_{<t}$ all previously generated tokens, and $\theta$ the fixed model parameters. At each step the model computes

$$\pi_\theta(y_t \mid x, y_{<t}),$$

a probability distribution over the next token (the same object as [[Lecture_06_Decoding_KV_Cache_MQA_GQA_FlashAttention#2.1 Probability of a continuation|Lecture 6 §2.1]]). A decoding procedure selects a token from it, and that token becomes available when generating later tokens.

![[L11_generation_loop.png|420]]

*One application-level generation contains an internal loop: the model with fixed parameters reads the prompt and generated prefix, the decoder selects the next token, and unless a stop condition is reached the token is appended to the prefix and the loop repeats.*

**How to extend this?** Asking for a longer response extends the loop; asking for several responses runs several generation trajectories. Three notes on the baseline:
- A direct answer still requires neural computation.
- A number such as "26" is not necessarily one token; tokenization determines that.
- This is a deliberately simple baseline. A production service may perform additional computation behind one user-visible response.

Consequently, **the length of the displayed answer does not, by itself, reveal the computation used to obtain it.** The model supplies next-token probabilities, but other components may decide how those probabilities become a complete answer.

### 4.3 Model, decoder, controller and evaluator

| Component | What it does | Example |
|---|---|---|
| **Language model** | computes next-token probabilities from the current context | assigns probabilities to possible continuations of a calculation |
| **Decoder** | selects tokens and applies generation settings and stopping rules | greedy selection or stochastic sampling |
| **Controller** | organizes the overall answering procedure | requests five attempts, stores them, and chooses what to do next |
| **Evaluator or tool** | supplies judgments, checks, calculations or observations | scores a solution, executes a test, or evaluates an arithmetic expression |

Here **decoder** means the token-selection procedure of [[#2. Decoding|§2]], **not** the transformer's decoder blocks. The **controller** turns a language model into a larger problem-solving system. It can be a short script, an application backend, provider-operated software, or a person coordinating the steps by hand. The field also calls it a **scaffold**, a **harness** or an **orchestrator**.

![[L11_controller_system.png|600]]

*The controller sits in the middle: it sends a prompt to the language model (fixed weights), gets generated text back through the decoder, sends candidates to the evaluator (votes, tests, scores) and requests to tools (code, search, calculator), and finally returns an answer.*

The controller is ordinary software around the model, and **every method in this lecture is a different controller design**. The model only proposes text; the controller decides what happens next: another generation, an evaluation, a tool call, or returning the answer.

![[L11_controller_sequence.png|650]]

*A possible sequence. The user submits a question; the controller sends the prompt and settings to the model + decoder and gets back a candidate or a proposed tool call. Optionally it executes the tool and continues with the result. Optionally it asks the evaluator to assess the candidate, and loops "revise and reassess if needed, within budget" before returning the selected answer to the user.*

### 4.4 Who implements what

- A **user** can implement simple prompting changes directly.
- An **application developer** usually implements repeatable multi-call procedures.
- A **provider** may implement some of these procedures inside its service.

For example, writing "use a calculator" in the prompt supplies an instruction. **Actual calculator use requires a connected tool and software that executes the operation and returns its result.** Once these responsibilities are visible, we can identify exactly where to intervene.

---

## 5. Test-Time Compute

With the controller named, the lecture can say precisely what "spending more at inference" means and who decides how it is spent. ([[Lecture_10_Neural_Scaling_Laws#16.1 Spending compute per question|Lecture 10 §16.1]] introduced this as "answer-time compute"; [[ATML_Lecture_12 - Test-Time Scaling and Multimodal Reasoning#1.3 What counts as inference compute|ATML L12 §1.3]] lists the same forms.)

### 5.1 Six ways to allocate extra effort

**Test-time compute** is computation spent on the current problem after training. It includes more than the tokens in the final answer.

| Allocation | What the system does | What it might gain |
|---|---|---|
| **Depth** | extends one reasoning trajectory | more intermediate calculations, checking or reconsideration |
| **Breadth** | generates several complete attempts | alternative approaches |
| **Verification** | evaluates answers or steps | evidence about which attempts are trustworthy |
| **Revision** | improves an existing attempt using feedback | correction of identified weaknesses |
| **Search** | explores alternative partial solutions | a better route before completing an answer |
| **Tools** | executes calculations, tests or retrieval | external computation or new information |

### 5.2 A fixed budget and a cost model

Suppose the generation allowance is 6,000 tokens. You could spend it on one 6,000-token attempt, three 2,000-token attempts, or six 1,000-token attempts. **None is universally best.** Some problems require sustained sequential reasoning; others benefit more from trying another approach. Multiple attempts also create a selection problem: which one should you return? *(This matches Snell et al.'s finding in [[Lecture_10_Neural_Scaling_Laws#16.3 How to spend the attempts (Snell et al.)|Lecture 10 §16.3]] that the best split depends on question difficulty.)*

A simple cost approximation makes the components visible:

$$C_{\text{total}} \approx C_{\text{prompt}} + N L c_g + N c_v + C_{\text{tools}} \qquad\qquad\text{(10)}$$
^eq-10

Here $N$ is the number of attempts, $L$ their average generated length, $c_g$ the average cost per generated token, and $c_v$ the evaluation cost per attempt. $C_{\text{prompt}}$ and $C_{\text{tools}}$ account for processing inputs and running tools. This is only an approximation: context length, caching, hardware, batching and model size affect actual cost.

![[CS5326_L11_budget_split.svg|680]]

*Three ways to spend a 6,000-token budget. Width is the length of one attempt (depth); the number of stacked strips is the number of attempts (breadth). Generation cost $NL$ is the same in all three rows, but the rows differ in latency (attempts run in parallel, so latency follows the length of one attempt), in verification cost $Nc_v$, and in whether a selection step is needed.* *(In the 6,000-token example, $NL = 6000$ in all three options, so generation cost is the same; they differ in $N c_v$, the cost of checking more attempts, and in latency, [[#5.6 Computation and latency are different|§5.6]].)*

### 5.3 Inference procedures and the controller

**Test-time compute is the resource; the inference procedure determines its use.** An **inference procedure** specifies how the system uses that computation, for example:
- generate one answer and stop;
- generate five solutions and vote;
- generate a solution, run tests, and revise if a test fails;
- explore alternative partial solutions until a budget is exhausted.

The **controller** of [[#4.3 Model, decoder, controller and evaluator|§4.3]] manages the procedure. It keeps track of the current state and decides what happens next: another generation, an evaluation, a revision, a tool call, or termination. It need not be another neural network and need not make sophisticated decisions. Its behaviour can be fixed or adaptive:

| Allocation | Example | What the controller does |
|---|---|---|
| **Fixed in advance** | generate five solutions, then vote | follows a predetermined sequence, regardless of intermediate results |
| **Conditional** | generate code; revise only if tests fail | chooses the next action using an explicit rule |
| **Adaptive** | decide whether to sample another solution, check a disputed step, or stop | uses available evidence and remaining budget to choose the next expenditure |

### 5.4 What an adaptive controller uses

Suppose the system has generated three solutions; two agree and the third gives a different answer. The controller might have access to:
- the candidates and their intermediate work;
- agreement or disagreement among answers;
- evaluator scores or test results;
- tokens, time or money already spent;
- the remaining budget.

It then chooses an action: generate another candidate, run a check that distinguishes the answers, revise an identified error, or stop. **Crucially, the controller does not ordinarily know whether the current answer is correct.** It decides from imperfect evidence, so designing a good controller is part of the problem.

### 5.5 Two levels of control

| Level | Who determines what happens |
|---|---|
| **Within one generated trajectory** | the model generates the next tokens, which may express calculations, reconsideration, or a proposed tool call |
| **Across generations and external operations** | the surrounding controller manages calls, stored candidates, tool execution, search and stopping |

The controller can consult the same model or another model when choosing actions. But generating the words "I should check this with Python" **does not run Python**. The surrounding system must recognize an actionable request, execute it, and return the result.

### 5.6 Computation and latency are different

Six attempts may run concurrently and finish much sooner than six sequential revisions, while consuming similar or greater total resources. Revision has **dependencies**: the next revision needs the previous answer and its feedback, so it cannot be parallelized.

**Putting it together.** The model proposes or generates work. The inference procedure organizes that work. A controller executes the procedure and, in adaptive systems, chooses the next action from the available evidence. Test-time compute is the resource consumed by the complete process. The next big question is **how to choose the answer to be returned.**

---

## 6. Sample and Aggregate

[[#3.11 The cost of a chain|§3.11]] ended with a fragile object: a single chain of reasoning that may contain an undetected error. The simplest remedy needs no new machinery: draw many chains and combine them. This is the "breadth" allocation of [[#5.1 Six ways to allocate extra effort|§5.1]] (ATML's parallel scaling, [[ATML_Lecture_12 - Test-Time Scaling and Multimodal Reasoning#3. Parallel Scaling — Think Wider|ATML L12 §3]]).

### 6.1 How to sample and how to combine

Every method in this section makes two decisions:
1. **How to sample.** Draw $N$ candidates that differ from one another (temperature above zero, [[#2.4 Temperature|§2.4]]).
2. **How to combine.** Reduce the $N$ candidates to one returned answer, in one of three ways:
   - **by agreement:** return the answer that appears most often. This is **self-consistency** ([[#6.2 Self-consistency|§6.2]]–[[#6.9 When voting helps, and what it costs|§6.9]]);
   - **by a scorer:** score every candidate with a verifier and return the top one. This is **best-of-N** ([[#6.11 Best-of-N - replace the vote with a scorer|§6.11]]);
   - **by a judge:** show all the candidates to a model and let it choose or merge. This covers **universal self-consistency** ([[#6.10 Universal self-consistency|§6.10]]) and ensembles across models.

For each, the question is the same: **when do more samples help, and when do they stop helping?**

![[ATML_L12_parallel.svg|600]]

*Reused from [[ATML_Lecture_12 - Test-Time Scaling and Multimodal Reasoning#3.1 Sampling K trajectories|ATML L12 §3.1]]. One model, $K$ samples made diverse by seeds, decoding settings and temperature $\tau$, then three ways to merge them: majority vote (self-consistency, §6.2), a verifier $V_\phi$ with arg max (best-of-N, §6.11), or feeding them back to the LLM (close to USC, §6.10).*

### 6.2 Self-consistency

**Self-consistency** samples $N$ reasoning chains for the same question at a temperature above zero, so that they differ, then ignores the reasoning, extracts the final answer from each chain, and returns the answer that appears most often.

Why it should work: **there are usually few ways to be right and many ways to be wrong.** Correct chains, however differently worded, arrive at the same answer; chains with errors tend to scatter over many different wrong answers. The correct answer therefore stands out as the most common one, even when it is far from certain in any single chain. *(ATML's view of the same rule is that the vote estimates the answer marginal $P(y \mid x)$ summed over reasoning traces; see [[ATML_Lecture_12 - Test-Time Scaling and Multimodal Reasoning#3.2 Self-consistency (majority voting)|ATML L12 §3.2]].)*

### 6.3 Worked example - five chains on the pen problem

The pen problem of [[#3.8 Chain of thought|§3.8]] (correct change: 20). Five chains at a temperature above zero:

| Chain | Reasoning | Answer |
|---|---|---|
| 1 | "12/3 = 4 groups. 4 × 20 = 80. 100 − 80 = 20." | 20 |
| 2 | "Each group of three costs 20. Four groups cost 80. The change is 20." | 20 |
| 3 | "3 × 20 = 60. 100 − 60 = 40." | 40 |
| 4 | "One pen costs 20/3 rupees, so 12 pens cost 80. The change is 20." | 20 |
| 5 | "12/3 = 4 groups. 4 × 20 = 80. 100 − 80 = 30." | 30 |

Ignore the reasoning and count: 20 appears three times, 40 once and 30 once. **Return 20.** The three correct chains are worded differently but land on the same answer; the two wrong chains land on different answers. (Chain 5 also shows why only the final answer is counted: its reasoning is correct until an arithmetic slip in the last step.)

### 6.4 Majority of three chains

Suppose a question has only two possible answers and each chain is correct with probability $p$, independently of the others. With $N = 3$ chains, the majority is correct when at least two of the three are correct, in two ways:
- **all three correct:** probability $p \times p \times p = p^3$;
- **exactly two correct:** one pattern such as correct, correct, wrong has probability $p \times p \times (1-p)$, and there are 3 patterns, because the wrong chain can be first, second or third. Total: $3p^2(1-p)$.

Adding the cases,

$$P(\text{majority of 3 correct}) = p^3 + 3p^2(1-p).$$

With $p = 0.6$ this is $0.216 + 0.432 = 0.648$. **Three mediocre chains beat one.**

### 6.5 Majority of five chains

Take five independent chains with $p = 0.6$. A strict majority needs three, four or five correct.

**Exactly three correct.** One pattern, such as ✓✓✓✗✗, has probability $0.6^3 \times 0.4^2 = 0.03456$. The number of such patterns is the number of ways to choose which three of the five positions are correct:

$$\binom{5}{3} = \frac{5!}{3!\,2!} = \frac{5 \times 4}{2 \times 1} = 10,$$

for example ✓✓✓✗✗, ✓✓✗✓✗ and ✓✗✓✗✓. In general, the number of ways to choose which $k$ of $N$ chains are correct is $\binom{N}{k} = \frac{N!}{k!(N-k)!}$; the three-chain case used $\binom{3}{2} = 3$. Ten patterns with the same probability give

$$P(\text{exactly 3 correct}) = 10 \times 0.6^3 \times 0.4^2 = 0.3456.$$

**Exactly four correct.** The single wrong chain can sit in any of five positions, $\binom{5}{4} = 5$:

$$P(\text{exactly 4 correct}) = 5 \times 0.6^4 \times 0.4 = 0.2592.$$

**All five correct.** One pattern, $\binom{5}{5} = 1$: $P = 0.6^5 = 0.0778$.

Adding the three cases,

$$P(\text{majority of 5 correct}) = 0.3456 + 0.2592 + 0.0778 = 0.683.$$

Voting over five chains raises accuracy from 0.60 to about 0.68. Three chains gave 0.648 and five give 0.683: **each extra pair of chains helps, but by a little less.**

![[CS5326_L11_binomial_vote.svg|640]]

*The five-chain vote as a binomial distribution. Left, $p = 0.6$: the green bars ($k \ge 3$ correct) are the cases where the majority is right, summing to 0.346 + 0.259 + 0.078 = 0.683. Right, $p = 0.4$: the same bars are the mirror image and sum to only 0.317, so with two possible answers voting makes a below-half model worse (§6.7).*

### 6.6 The general majority formula

Counting every way in which more than half of the $N$ chains are correct,

$$P(\text{majority of } N \text{ correct}) = \sum_{k > N/2} \binom{N}{k}\, p^{k}\,(1-p)^{N-k} \qquad\qquad\text{(11)}$$
^eq-11

where $p^k$ is the probability that a particular $k$ chains are correct, $(1-p)^{N-k}$ the probability that the other $N-k$ are wrong, and $\binom{N}{k}$ counts which $k$ chains are the correct ones. The sum runs over every $k$ that forms a majority. *(This is the tail of a binomial distribution; for odd $N$ there are no ties.)*

### 6.7 With two answers, voting cuts both ways

Evaluating [[#^eq-11|eq. (11)]] for two single-chain accuracies (rows 3 and 5 are the worked examples):

| Chains $N$ | Majority correct, $p = 0.6$ | Majority correct, $p = 0.4$ |
|---|---|---|
| 1 | 0.600 | 0.400 |
| 3 | 0.648 | 0.352 |
| 5 | 0.683 | 0.317 |
| 9 | 0.733 | 0.267 |
| 15 | 0.787 | 0.213 |
| 25 | 0.846 | 0.154 |
| 41 | 0.903 | 0.097 |

Above one half, more chains keep helping, but slowly: **it takes 41 chains to move from 0.60 to 0.90.** The right-hand column is the risk. With only two possible answers, all the wrong chains give the same wrong answer. Below one half, that wrong answer is the most common, so more chains make it win more often. **Voting amplifies the most common answer. It does not know which answer is true.** *(The $p = 0.4$ column is the $p = 0.6$ column subtracted from 1, by the symmetry right ↔ wrong.)*

### 6.8 Plurality - real questions have many wrong answers

The table assumed only two possible answers; real questions have many ways to be wrong. **Majority** means more than half of the votes; **plurality** means more votes than any other answer. In practice, self-consistency returns the **plurality** answer.

Suppose 100 chains produce: correct answer A, 40 votes; wrong answers B, C and D, 20 votes each. **A wins, even though 60% of the chains are wrong**, because the wrong votes are split across different answers. So voting can help even below 50% single-chain accuracy, provided the correct answer is more likely than **each** individual wrong answer.

The realistic case is kinder. Turning the example into probabilities (correct answer 0.4, three wrong answers 0.2 each), no single chain is likely to be right, but voting gives 0.51 at $N = 5$, 0.69 at $N = 15$ and 0.89 at $N = 41$ (ties broken at random). If instead all the wrong chains agree on one answer at 0.6, the same single-chain accuracy of 0.4 is driven down.

![[L11_voting_curves.png|650]]

*Left: two possible answers; voting helps when a single chain is right more often than wrong ($p = 0.7, 0.6, 0.55$ rise toward 1) and hurts otherwise ($p = 0.4$ falls toward 0). Right: two cases with the same single-chain accuracy of 0.4. When the errors scatter over three wrong answers at 0.2 each (green), voting recovers the truth; when they agree on one wrong answer at 0.6 (red), voting amplifies the error.*

### 6.9 When voting helps, and what it costs

**The condition:** voting helps exactly when **the correct answer is more common than any single wrong answer**. It does not need a probability above one half.

**Two caveats:**
- The formula assumes the chains are **independent**. Chains sampled from one model are not: if the model holds a misconception, most chains will share it. Real gains are smaller than the formula suggests, and they level off. (This is the red curve of the figure above: a shared misconception makes the errors agree.)
- The **temperature must be high enough** to produce different chains. At zero temperature every chain is identical and voting does nothing.

**The cost:** $N$ chains cost $N$ times the tokens of one chain ($N L c_g$ in [[#^eq-10|eq. (10)]]). The chains can run in parallel, so the price is paid in compute more than in waiting time ([[#5.6 Computation and latency are different|§5.6]]).

### 6.10 Universal self-consistency

Voting needs answers that can be matched exactly: a number, a multiple-choice letter. Two summaries of the same article never match word for word, so there is nothing to count.

**Universal self-consistency (USC)** replaces the count with a judgment: sample $N$ candidate outputs as before, show all of them to a judge model in one prompt, and ask it to pick the candidate **most consistent with the others**. This extends voting to free-form outputs at the price of **trusting the model's judgment**. *(USC is close to the self-aggregation of [[ATML_Lecture_12 - Test-Time Scaling and Multimodal Reasoning#3.4 Self-aggregation|ATML L12 §3.4]]; the difference is that USC asks the judge to select the most consistent existing candidate, while self-aggregation may write a new answer.)*

### 6.11 Best-of-N - replace the vote with a scorer

A judge that can compare candidates can also score each one on its own merits. Voting uses only agreement; if we can score a candidate, we can do better. **Best-of-N (BoN)** generates $N$ candidates, scores each with a **verifier**, and returns the top one ([[ATML_Lecture_12 - Test-Time Scaling and Multimodal Reasoning#3.3 Best-of-N with a verifier|ATML L12 §3.3]]). The verifier can be:
- a set of **unit tests** that a candidate program must pass;
- a trained **reward model** that outputs a score;
- another model acting as a **judge**.

Whether best-of-N works depends on **two separate events**:
1. **Coverage:** at least one of the $N$ candidates is correct.
2. **Selection:** the verifier returns a correct candidate.

![[ATML_L12_hybrid.svg|560]]

*Reused from [[ATML_Lecture_12 - Test-Time Scaling and Multimodal Reasoning#4.1 Solution coverage vs selection accuracy|ATML L12 §4.1]]. (a) Coverage is a property of the $K$ samples (is a correct one among them?); selection is a property of the verifier $V_\phi$ (does it pick it?). (b) A process verifier scores growing prefixes of one trajectory, which part 2 of this lecture takes up.*

The deck ends here, announcing that coverage is computed first and then three kinds of verifier are examined; that continues in part 2. *(Coverage for independent attempts is $1 - (1-p)^N$, already derived as pass@k in [[Lecture_10_Neural_Scaling_Laws#16.2 Repeated sampling and pass@k|Lecture 10 §16.2]]; the coverage/selection split is [[ATML_Lecture_12 - Test-Time Scaling and Multimodal Reasoning#4.1 Solution coverage vs selection accuracy|ATML L12 §4.1]].)*

---

## Formula Summary Sheet

| Concept | Formula | Where |
|---|---|---|
| Softmax | $p_i = e^{z_i}/\sum_j e^{z_j}$ | [[#2.1 From logits to probabilities\|§2.1]], [[Lecture_06_Decoding_KV_Cache_MQA_GQA_FlashAttention#2.2 Next-token distribution and the causal mask\|L6 §2.2]] |
| Logit gap = probability ratio | $p_i/p_j = e^{z_i - z_j}$ | [[#2.1 From logits to probabilities\|§2.1]] |
| Temperature | $p_i(T) = e^{z_i/T}/\sum_j e^{z_j/T}$; ratio $e^{(z_i - z_j)/T}$ | [[#2.4 Temperature\|§2.4]], [[ATML_Lecture_7 - LLM Foundations#6.3 Sampling with temperature\|ATML L7 §6.3]] |
| Min-$p$ threshold | keep $x$ if $p(x) \ge r \cdot \max_j p_j$ | [[#2.5 Truncation - top-k, nucleus and min-p\|§2.5]], [[#2.6 Contrastive methods\|§2.6]] |
| Contrastive score | $\log p_{\text{strong}}(x) - \log p_{\text{weak}}(x)$ | [[#2.6 Contrastive methods\|§2.6]], [[#2.7 Three choices of weak predictor\|§2.7]] |
| Classifier-free guidance | $z_{\text{guided}} = z_{\text{w/o}} + \gamma(z_{\text{with}} - z_{\text{w/o}})$ | [[#2.8 Classifier-free guidance and the cost of contrast\|§2.8]] |
| ICL as task inference | $p(y \mid x, D) = \sum_\theta p(y \mid x, \theta)\,p(\theta \mid D)$ | [[#3.2 View one - the model infers which task it is in\|§3.2]], [[#3.3 Worked example - capital or largest city\|§3.3]] |
| Task posterior | $p(\theta \mid D) \propto p(\theta)\prod_i p(y_i \mid x_i, \theta)$ | [[#3.2 View one - the model infers which task it is in\|§3.2]] |
| One GD step = linear attention | $\hat y_q = \frac{\eta}{n}\sum_i y_i\,(x_i \cdot x_q)$ | [[#3.5 View two - the forward pass runs a learning algorithm\|§3.5]], [[#3.6 Worked example - learning y = 2x in one pass\|§3.6]] |
| Chain reliability | $P(\text{chain correct}) = q^k$ | [[#3.11 The cost of a chain\|§3.11]] |
| Test-time cost | $C_{\text{total}} \approx C_{\text{prompt}} + NLc_g + Nc_v + C_{\text{tools}}$ | [[#5.2 A fixed budget and a cost model\|§5.2]], [[#6.9 When voting helps, and what it costs\|§6.9]] |
| Majority of 3 | $p^3 + 3p^2(1-p)$ | [[#6.4 Majority of three chains\|§6.4]] |
| Binomial coefficient | $\binom{N}{k} = \frac{N!}{k!(N-k)!}$ | [[#6.5 Majority of five chains\|§6.5]] |
| Majority of $N$ | $\sum_{k > N/2}\binom{N}{k}p^k(1-p)^{N-k}$ | [[#6.6 The general majority formula\|§6.6]], [[#6.7 With two answers, voting cuts both ways\|§6.7]] |
| Coverage (pass@N) | $1 - (1-p)^N$ | [[#6.11 Best-of-N - replace the vote with a scorer\|§6.11]], [[Lecture_10_Neural_Scaling_Laws#16.2 Repeated sampling and pass@k\|L10 §16.2]] |

---

## Summary

- **Why inference time ([[#1. Why Inference Time|§1]])**
  - Weights frozen; we change the "conditions of the exam". The same model can vary more across usage than across model generations.
  - Reasons: weights behind APIs; inference changes are immediate and reversible; facts, arithmetic and long reasoning are easier to build around a model.
  - Four levers: readout, input, computation, system.
  - One forward pass = fixed compute per token. More compute comes only from more tokens (inside) or more calls (outside), so: generate more, then select well.
- **Decoding ([[#2. Decoding|§2]])**
  - Softmax turns a logit gap into a probability ratio $e^{\Delta z}$; temperature divides every gap by $T$ ($T \to 0$ = greedy).
  - Greedy/beam chase likelihood: fine for translation, bland and loopy for open text. Pure sampling risks the unreliable tail.
  - Top-$k$ fixed count; top-$p$ smallest set reaching mass $p$; min-$p$ threshold = ratio × top probability.
  - Contrastive score $\log p_s - \log p_w$ rewards what only the strong predictor knows; it needs a plausibility filter. Weak = small model (contrastive decoding), early layer (DoLa), no prompt (CFG).
  - CFG: $z_{\text{w/o}} + \gamma(z_{\text{with}} - z_{\text{w/o}})$; $\gamma > 1$ amplifies the prompt. All contrast needs logits and a second pass, and can hurt fluency.
  - Constrained decoding masks illegal tokens to $-\infty$: guaranteed format, not correct content. Put the reasoning field before the answer field.
  - Decoding cannot repair a wrong plan.
- **In-context learning ([[#3.1 In-context learning|§3.1]]–[[#3.7 In-context learning is powerful but brittle|§3.7]])**
  - Bayesian view: the posterior over hidden tasks multiplies in one likelihood per example; the prediction averages over tasks. Examples that fit several tasks teach nothing (Paris vs Canberra).
  - It is not implemented explicitly; the optimal next-token predictor must behave this way.
  - Learning-algorithm view: one GD step from zero on linear regression gives $\frac{\eta}{n}\sum_i y_i(x_i \cdot x_q)$, which is linear attention with keys = inputs and values = labels.
  - Brittle to example choice, order and label balance; retrieving similar examples (KATE) is a reliable fix.
- **Chain of thought ([[#3.8 Chain of thought|§3.8]]–[[#3.11 The cost of a chain|§3.11]])**
  - Written steps are a scratchpad read back by attention, so sequential compute grows with tokens, not layers.
  - Variants: least-to-most (decompose), plan-and-solve (plan then execute), step-back (principle first).
  - $P = q^k$: 0.95 per step over 10 steps ≈ 0.60. One chain is one unchecked sample.
- **The system and test-time compute ([[#4. The System Around the Model|§4]], [[#5. Test-Time Compute|§5]])**
  - Four components: model, decoder (token selection, not decoder blocks), controller (scaffold/harness/orchestrator), evaluator or tool.
  - The answer's length does not reveal the compute behind it; writing "use Python" does not run Python.
  - Allocations: depth, breadth, verification, revision, search, tools. Cost ≈ prompt + $NLc_g$ + $Nc_v$ + tools.
  - Controllers are fixed, conditional or adaptive, and decide without knowing whether the answer is correct.
  - Parallel attempts save latency; revisions are sequential.
- **Sample and aggregate ([[#6. Sample and Aggregate|§6]])**
  - Combine by agreement (self-consistency), scorer (best-of-N) or judge (USC).
  - Self-consistency: few ways to be right, many to be wrong; the vote uses final answers only.
  - Majority of $N$ is a binomial tail: $p = 0.6$ gives 0.648 (3), 0.683 (5), 0.903 (41). With $p < 0.5$ and two answers, voting makes things worse.
  - Plurality helps whenever the correct answer beats each single wrong answer (0.4 vs 0.2 ×3 gives 0.89 at $N = 41$); errors that agree get amplified.
  - Caveats: chains from one model are correlated, so real gains are smaller; temperature must be above 0. Cost is $N\times$ tokens, but parallel.
  - USC lets a judge pick the most consistent free-form candidate. Best-of-N needs both coverage and selection (continued in part 2).
