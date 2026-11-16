---
type: lecture
course: "[[_CS5326 Advanced GenAI|CS5326 Advanced GenAI]]"
source:
  - "[[007-Neural Scaling Laws.pdf]]"
tags:
  - course/cs5326
textfold: md-lecture-10-neural-scaling-laws-e7cd8c49
---

%% nav:start %%
[[_CS5326 Advanced GenAI|CS5326 Advanced GenAI]] · ← [[Lecture_09_FFNs_Dense_and_MoE]]
Concepts: [[Scaling Laws]], [[Chinchilla Scaling Laws]], [[Zipf's Law]], [[Power Law]], [[Test-Time Compute]]
%% nav:end %%

# Neural Scaling Laws (CS5326, Lecture 10)

*[[Lecture_09_FFNs_Dense_and_MoE#6. Total vs Active Parameters|Lecture 9 §6]] ended on the split between parameters a model stores and parameters it uses per token, which MoE pulls apart. This lecture asks the planning question that sits above every architecture choice: given a compute budget, how big should the model be, how much data should it see, and how well will it do, before any money is spent?*

<!-- textfold-skip -->
> [!tip]- Quiz prep (2.5 hours) — reading by weight
> Past quizzes run 15 minutes for 10 marks: four 1-mark MCQs on mechanism distinctions or one-step calculations, then two 3-mark "demonstrate" or "compute and explain" questions.
>
> | Sections | Time | Focus |
> |---|---|---|
> | [[#5. Properties of Power-Law Loss\|§5]], [[#6. Power-Law Decay vs Exponential Decay\|§6]], [[#7.2 Three levels of improvement\|§7.2]] | 20 min | Multiply vs add, which axes straighten which curve, the halving cost $2^{1/\alpha}$ |
> | [[#9. Kaplan et al. (2020)\|§9]] | 25 min | Why 2 + 4 FLOPs ([[#9.3 FLOPs and the rule C ≈ 6ND\|§9.3]]); why a steeper data exponent does not mean spend on data ([[#9.9 Why the steeper data exponent does not mean spend on data\|§9.9]]) |
> | [[#10. Chinchilla (Hoffmann et al., 2022)\|§10]], [[#11. Why Kaplan Got a Different Answer (Porian et al., 2024)\|§11]] | 20 min | The balance condition, the 20:1 rule, $N = \sqrt{C/120}$, Porian's three corrections |
> | [[#12. When the Data Runs Out (Muennighoff et al., 2023)\|§12]], [[#13. The Relative Worth of Pretraining (Hernandez et al., 2021)\|§13]] | 15 min | Effective tokens; absolute vs relative worth of pretraining |
> | [[#15. Train-Time or Test-Time — Lifetime Cost\|§15]], [[#16. Answer-Time Compute\|§16]] | 10 min | Break-even inference tokens; pass@k and coverage vs selection |
> | [[#2. Power Laws in Language — Zipf's Law\|§2]]–[[#4. The Pareto Principle and the Pareto Distribution\|§4]], [[#8. Terminology and the General Scaling Law\|§8]], [[#14. The Curse of Multilinguality\|§14]], [[#17. Risks of Scaling\|§17]] | skim | Their bullets in the [[#Summary\|Summary]] only |
> | Calculations below, by hand | 45 min | Note closed, then check against the linked section |
> | [[#Summary\|Summary]] | 15 min | One full read |
>
> **Most likely calculations**
> 1. Zipf corpus size, $N = Vx_VH_V$ (about 121M tokens): [[#2.6 Worked example with 100,000 word types|§2.6]]
> 2. A 10% cut needs 4× more $N$ but only 3× more $D$, and why that sends the budget to $N$: [[#9.9 Why the steeper data exponent does not mean spend on data|§9.9]]
> 3. Chinchilla at GPT-3's budget, $C = 120N^2$ (about 51B parameters, 1T tokens): [[#10.9 What it means in practice|§10.9]]
> 4. 5B unique Urdu tokens repeated to 100B presentations ≈ 60B effective: [[#12.6 An Urdu example and the rules of thumb|§12.6]], with the limits in [[#12.3 Checking the formula at its limits|§12.3]]
> 5. 10× more fine-tuning data: $D_T$ grows 1.51× while $D_T/D_F$ falls from 250 to 37.8: [[#13.4 Absolute worth up, relative worth down|§13.4]]
> 6. 13B vs 7B lifetime saving of 17.5%, break-even at about 583B inference tokens: [[#15.4 Worked example — 13B vs 7B|§15.4]], [[#15.5 Payback time|§15.5]]
> 7. pass@k with $p = 0.2$, and why it isn't accuracy: [[#16.2 Repeated sampling and pass@k|§16.2]]
<!-- /textfold-skip -->

> [!note] Notation in this lecture
> $N$ is the **number of parameters** and $D$ the **number of training tokens**; in [[Lecture_09_FFNs_Dense_and_MoE|Lecture 9]] $N$ was the number of experts. The deck reuses some letters: $E$ is the **excess loss** $L - L_\infty$ in [[#7. Breaking Beyond Power-Law Scaling|§7]] but the **loss floor** from [[#8. Terminology and the General Scaling Law|§8]] onward; $\alpha$ is the exponent of whichever law is on the page (power-law loss, Zipf, temperature sampling, transfer); and $e$ counts **epochs** in [[#8.3 Unique tokens vs token presentations|§8.3]] and [[#12. When the Data Runs Out (Muennighoff et al., 2023)|§12]] while $e^{x}$ is the exponential. Each section states which meaning applies.

## Topic Inventory

- 1. Power-Law Loss
  - 1.1 Planning before spending a few million dollars
  - 1.2 The simplest power law
  - 1.3 Reading linear, semi-log and log–log plots
  - 1.4 Why a power law is straight on log–log axes
- 2. Power Laws in Language — Zipf's Law
  - 2.1 Zipf's rank–frequency law
  - 2.2 The multiplication rule and the long tail
  - 2.3 Why the pattern matters for NLP
  - 2.4 How large must a corpus be
  - 2.5 The harmonic number and the Euler–Mascheroni constant
  - 2.6 Worked example with 100,000 word types
- 3. Coverage From Little Data — the PRUS Corpus
  - 3.1 Greedy set cover over tri-phonemes
  - 3.2 The corpus and its unusual sentences
  - 3.3 Outcomes
- 4. The Pareto Principle and the Pareto Distribution
  - 4.1 Two different questions
  - 4.2 Zipf's law gives a Pareto-like concentration
  - 4.3 The Pareto distribution
- 5. Properties of Power-Law Loss
  - 5.1 Scale invariance and diminishing returns
  - 5.2 How expensive the next halving is
  - 5.3 Realistic curves have a floor
  - 5.4 Why this matters for neural networks
- 6. Power-Law Decay vs Exponential Decay
  - 6.1 Multiply vs add
  - 6.2 Which axes straighten which curve
  - 6.3 Any exponential is straight on a semi-log plot
- 7. Breaking Beyond Power-Law Scaling
  - 7.1 Excess loss and the escalating cost of each halving
  - 7.2 Three levels of improvement
  - 7.3 Selective sampling
  - 7.4 Finding useful examples also costs resources
- 8. Terminology and the General Scaling Law
  - 8.1 What to scale
  - 8.2 Recurring symbols
  - 8.3 Unique tokens vs token presentations
  - 8.4 The general form and the excess loss
  - 8.5 Three regions of a learning curve
- 9. Kaplan et al. (2020)
  - 9.1 The question and the experiments
  - 9.2 Finding 1 — loss follows power laws
  - 9.3 FLOPs and the rule C ≈ 6ND
  - 9.4 The PF-day
  - 9.5 When each single-variable law holds
  - 9.6 Six orders of magnitude
  - 9.7 Finding 2 — spend the budget mostly on parameters
  - 9.8 Reading the compute envelope
  - 9.9 Why the steeper data exponent does not mean spend on data
  - 9.10 Take-aways and caveats
- 10. Chinchilla (Hoffmann et al., 2022)
  - 10.1 Were Kaplan's small models undertrained
  - 10.2 Three ways to find the optimum
  - 10.3 Finding — grow parameters and tokens together
  - 10.4 MMLU
  - 10.5 The parametric loss and its optimum
  - 10.6 IsoFLOP valleys
  - 10.7 Why 20 tokens per parameter
  - 10.8 The disagreement grows under extrapolation
  - 10.9 What it means in practice
- 11. Why Kaplan Got a Different Answer (Porian et al., 2024)
  - 11.1 Hoffmann's explanation
  - 11.2 Three corrections, one at a time
- 12. When the Data Runs Out (Muennighoff et al., 2023)
  - 12.1 The problem and the finding
  - 12.2 Effective tokens
  - 12.3 Checking the formula at its limits
  - 12.4 How the value of repetition decays
  - 12.5 The same discount for parameters
  - 12.6 An Urdu example and the rules of thumb
  - 12.7 Data pruning revisited (Sorscher et al., 2022)
- 13. The Relative Worth of Pretraining (Hernandez et al., 2021)
  - 13.1 Pretraining as a head start measured in data
  - 13.2 Absolute and relative worth
  - 13.3 The fitted transfer law
  - 13.4 Absolute worth up, relative worth down
  - 13.5 The pretraining data matters
  - 13.6 Lessons
- 14. The Curse of Multilinguality
  - 14.1 Mutual benefit and contention
  - 14.2 Temperature sampling
  - 14.3 Tokenizer audits
  - 14.4 Language-aware curation
  - 14.5 Putting the curse into numbers (ATLAS)
  - 14.6 Which languages help one another
  - 14.7 Start from an existing multilingual model or from scratch
- 15. Train-Time or Test-Time — Lifetime Cost
  - 15.1 Trained once, used many times
  - 15.2 Training and inference compute
  - 15.3 Same quality from a smaller model
  - 15.4 Worked example — 13B vs 7B
  - 15.5 Payback time
  - 15.6 Realistic extremes and what deployed models do
  - 15.7 Newer work and cautions
- 16. Answer-Time Compute
  - 16.1 Spending compute per question
  - 16.2 Repeated sampling and pass@k
  - 16.3 How to spend the attempts (Snell et al.)
  - 16.4 Optimizing model size, data and attempts together
- 17. Risks of Scaling
  - 17.1 Measurements are not behaviour
  - 17.2 Goodhart's law and reward overoptimization
  - 17.3 Fine-tuning side effects and cultural representation
  - 17.4 Conclusion

---

## 1. Power-Law Loss

### 1.1 Planning before spending a few million dollars

A large training run costs millions of dollars, and the deck opens with the joke alternative to planning: close your eyes, grab a few million dollars, and throw them at a few million parameters with a few million tokens of data. The lecture's answer is that loss behaves predictably enough with scale that small, cheap runs can forecast large ones. That predictability has a mathematical shape, the power law, so the lecture builds the mathematics first ([[#1. Power-Law Loss|§1]]–[[#7. Breaking Beyond Power-Law Scaling|§7]]) and then the scaling-law papers that use it ([[#8. Terminology and the General Scaling Law|§8]] onward).

### 1.2 The simplest power law

In neural scaling, **power-law loss** means that loss decreases as a resource (data, model size or compute) increases, according to a power law. It describes how a loss such as cross-entropy or perplexity behaves. The simplest form is

$$L(x) = Ax^{-\alpha} = \frac{A}{x^{\alpha}} \qquad\qquad\text{(1)}$$
^eq-1

where $x > 0$ is the amount of a resource, measured in specified units; $L(x)$ is the loss achieved at that level (lower is better); $A > 0$ is a scale constant, numerically the loss at $x = 1$; and $\alpha > 0$ is the **scaling exponent**, which sets how quickly loss improves.

For example, take $L(x) = 4x^{-0.5} = 4/\sqrt{x}$:

| Resource $x$ | Loss $L(x)$ |
|---|---|
| 1 | 4 |
| 4 | 2 |
| 16 | 1 |
| 64 | 0.5 |
| 256 | 0.25 |

Each fourfold increase in resources halves the loss, and the same rule holds wherever you start: going from 1 to 4 and from 64 to 256 both halve it. This "same multiplier, same proportional gain" property is what the rest of [[#1. Power-Law Loss|§1]] and [[#5. Properties of Power-Law Loss|§5]] unpack.

### 1.3 Reading linear, semi-log and log–log plots

"Log plot" can mean taking the logarithm of either axis, so the figure shows all four combinations. On a **linear axis**, equal distances represent equal additions (100, 200, 300). On a **logarithmic axis**, equal distances represent equal multiplications (1, 10, 100). For now follow the solid blue curve, $L = 4x^{-0.5}$; the dashed orange curve, which has a floor, returns in [[#5.3 Realistic curves have a floor|§5.3]].

![[L10_powerlaw_axes.png|650]]

*The same two curves on four axis choices: blue is the pure power law $4x^{-0.5}$, dashed orange adds a floor, $1 + 4x^{-0.5}$, and the grey line marks that floor at 1. Only the log–log panel turns the blue curve into a straight line (slope −0.5). These are illustrative curves, not experimental data.*

| Plot | What you see | Lesson |
|---|---|---|
| Plain, both axes linear | A steep initial drop followed by flattening | Additional units of resource produce progressively smaller absolute improvements |
| Logarithmic loss axis only | The power law remains curved; the low values of the loss are zoomed in | Taking the logarithm of loss alone does not straighten a power law |
| Logarithmic resource axis only | Early resource scales become easier to inspect, but the curve remains curved | Equal resource multipliers are now equally spaced |
| Log–log, both axes logarithmic | The pure power law becomes a straight line | The same resource multiplier produces the same proportional loss reduction at every scale |

### 1.4 Why a power law is straight on log–log axes

Taking logarithms of [[#^eq-1|eq. (1)]] and using the product rule $\log(ab) = \log a + \log b$ and the power rule $\log x^{k} = k\log x$,

$$\log L(x) = \log\!\left(Ax^{-\alpha}\right) = \log A - \alpha \log x.$$

This is a straight-line equation: the horizontal coordinate is $\log x$, the vertical coordinate is $\log L$, the intercept is $\log A$ and the slope is $-\alpha$. The base of the logarithm does not change the slope, provided both axes use the same base, because changing base multiplies both coordinates by the same constant. A power law's defining property is therefore **constant proportional improvement**: each step of fixed size along $\log x$ lowers $\log L$ by the same amount. In practice this is how exponents are measured: plot the runs on log–log axes and read the slope.

---

## 2. Power Laws in Language — Zipf's Law

Power laws are not special to neural networks. Before applying them to training, the deck looks at the most famous power law in language, which also matters for how much data a model needs.

### 2.1 Zipf's rank–frequency law

Take a large collection of text, count how often each word occurs, and sort the words from most to least frequent. Can a word's frequency be predicted from its position in this ranking alone? Assign rank 1 to the most frequent word, rank 2 to the next, and so on. **Zipf's law** (after the linguist George Kingsley Zipf) says that frequency approximately follows

$$f(r) \approx \frac{K}{r^{\alpha}}, \qquad \alpha \approx 1,$$

where $r$ is a word's rank after sorting by frequency, $f(r)$ its occurrence count in the collection, $K$ a scale constant set by the collection (at $r = 1$, $f(r) \approx K$), and $\alpha$ the exponent, often approximately 1. In its simplest form, $f(r) \approx K/r$: the second-ranked word occurs about half as often as the first, the tenth-ranked about one-tenth as often. Real corpora show systematic deviations, so this is an empirical approximation. It is the same shape as [[#^eq-1|eq. (1)]], with rank in place of the resource and frequency in place of the loss.

In an idealized collection where the most frequent word occurs 10,000 times, the law predicts:

| Word rank $r$ | Predicted count $10{,}000/r$ |
|---|---|
| 1 | 10,000 |
| 2 | 5,000 |
| 4 | 2,500 |
| 10 | 1,000 |
| 100 | 100 |
| 1,000 | 10 |

### 2.2 The multiplication rule and the long tail

Comparing a word at rank $r$ with one at rank $kr$, where $k > 1$,

$$\frac{f(kr)}{f(r)} = \frac{K/(kr)}{K/r} = \frac{1}{k}.$$

A word ranked $k$ times farther down the list occurs $1/k$ as often. With $k = 2$ this compares ranks 10 and 20, or 100 and 200; in either case the word with the larger rank number occurs about half as often. This is the **multiplication rule**: every doubling of rank halves frequency, and going from rank 1 to 2 follows the same proportional rule as going from rank 100 to 200.

![[L10_zipf.png|650]]

*Top: doubling the rank halves the count (10,000 → 5,000 → 2,500 → 1,250). Left: on ordinary axes a few words are extremely common and many rare words form the long, low **long tail**. Right: on log–log axes, multiplying the rank by 10 divides the count by 10, so the curve is a straight line with slope −1.*

### 2.3 Why the pattern matters for NLP

Is this surprising? People choose words to express meanings, follow grammar and respond to context, yet when those choices are aggregated a simple rank–frequency pattern appears. The practical implication for NLP is that **a large corpus can contain abundant evidence for common words while still providing little evidence for many rare words.** Adding more random text mostly adds more copies of words already well covered, an idea that returns in [[#7.3 Selective sampling|§7.3]].

### 2.4 How large must a corpus be

Suppose we want a corpus covering $V$ distinct word types, with the least frequent type (rank $V$) occurring $x_V$ times. Under the ideal Zipf law, how often must the most frequent type occur, and how many tokens must the corpus contain? Let $x_r$ be the number of occurrences of the word at rank $r$ and assume exponent 1, so $x_r = x_1/r$. At rank $V$,

$$x_V = \frac{x_1}{V} \;\Longrightarrow\; x_1 = Vx_V.$$

The total corpus size $N$ (here a token count, not parameters) is the sum of all word counts, and factoring out $x_1$ leaves the **harmonic number** $H_V$:

$$N = \sum_{r=1}^{V} x_r = x_1\left(1 + \frac{1}{2} + \frac{1}{3} + \cdots + \frac{1}{V}\right) = x_1 H_V.$$

Substituting $x_1 = Vx_V$,

$$\boxed{N = Vx_V H_V}, \qquad H_V = \sum_{r=1}^{V}\frac{1}{r} \approx \ln V + 0.5772 \qquad\qquad\text{(2)}$$
^eq-2

The required corpus grows a little faster than linearly in the vocabulary, because $H_V$ grows like $\ln V$.

### 2.5 The harmonic number and the Euler–Mascheroni constant

The 0.5772 corrects the difference between the discrete sum $H_V$ and its continuous approximation $\ln V$. A convenient approximation for the sum comes from the area under $1/t$:

$$\int_1^{V}\frac{1}{t}\,dt = \ln V.$$

But adding discrete terms (areas of unit-width rectangles of height $1/r$) is not exactly the same as the smooth area. With $V = 2$, the sum is $1 + \tfrac12 = 1.5$ while the integral is $\ln 2 \approx 0.6931$. Both calculations are exact; they compute different quantities. As $V$ grows their difference settles to a fixed number,

$$\gamma = \lim_{V\to\infty}\left(H_V - \ln V\right) = 0.5772156649\ldots,$$

the **Euler–Mascheroni constant**. The settling is visible numerically:

| $V$ | $H_V$ | $\ln V$ | Difference |
|---|---|---|---|
| 10 | 2.92897 | 2.30259 | 0.62638 |
| 100 | 5.18738 | 4.60517 | 0.58221 |
| 1,000 | 7.48547 | 6.90776 | 0.57772 |
| 100,000 | 12.09015 | 11.51293 | 0.57722 |

So for large $V$, $H_V \approx \ln V + 0.5772$. This is a mathematical correction for approximating the sum, not a parameter fitted to language data. *(Expanding on this: the rectangles of height $1/r$ sit above the curve $1/t$ on each interval $[r, r+1]$, so the sum always exceeds the integral; the excess on each interval shrinks fast enough that the total converges, and its limit is $\gamma$.)*

### 2.6 Worked example with 100,000 word types

Suppose we want $V = 100{,}000$ word types, with the least frequent word occurring $x_V = 100$ times. The most frequent word must occur $x_1 = Vx_V = 100{,}000 \times 100 = 10{,}000{,}000$ times. The harmonic number is $H_{100{,}000} \approx \ln(100{,}000) + 0.5772 \approx 12.0901$. By [[#^eq-2|eq. (2)]] the corpus size is

$$N = 10{,}000{,}000 \times 12.0901 \approx \boxed{120.9\ \text{million tokens}}.$$

These calculations assume exactly $V$ types following ideal Zipf frequencies. For randomly collected data they describe **expected** counts, not guaranteed minimum counts: a random 121-million-token sample will leave some rare types below 100.

---

## 3. Coverage From Little Data — the PRUS Corpus

[[#2.6 Worked example with 100,000 word types|§2.6]] shows that random collection needs about 121 million tokens just to see each of 100,000 types 100 times, because most tokens are repeats of common words. The professor's own corpus-design work asks the opposite question: how can we select a small amount of data that still provides substantial linguistic coverage?

### 3.1 Greedy set cover over tri-phonemes

Raza et al. (2009), *Design and Development of a Phonetically Rich Urdu Speech Corpus*, faced this for speech recognition.

- **Challenge.** Collect broad phonetic coverage without repeatedly recording common contexts, or requiring speakers to read unfamiliar words.
- **Starting resources.** An Urdu text corpus of about 18 million word tokens, a 50,000-word pronunciation lexicon, and 62 phonemes.
- **Coverage unit.** A **tri-phoneme**: a target phoneme together with its immediate left and right phonetic contexts. *(Expanding on this: speech recognizers model a sound in context because the same phoneme is pronounced differently depending on its neighbours, so covering every context, not just every phoneme, is what the acoustic model needs.)*
- **Identify the inventory.** Extract the 18,294 distinct within-word tri-phonemes, including word-edge contexts with silence.
- **Select efficiently.** Use **greedy set covering** to choose words that add uncovered contexts, prioritizing frequent, familiar words so speakers read fluently. *(Expanding on this: set cover asks for the fewest sets, here words, whose union covers every element, here tri-phonemes. It is NP-hard, so the greedy heuristic repeatedly picks the word that covers the most still-uncovered tri-phonemes.)*
- **Takeaway.** Choose recording material according to the missing sound contexts it supplies.

### 3.2 The corpus and its unusual sentences

The **Phonetically Rich Urdu Speech (PRUS) corpus** is 70 minutes of transcribed read speech: 708 greedily created sentences representing all phonemic and tri-phonemic combinations in Urdu. It is based on an 18-million-word corpus of Urdu news articles, has 10,101 tokens with 5,656 unique words, gives phonetic and tri-phonetic coverage and phonetic balance for Urdu, and contains 60 unique phones and 42,289 phone occurrences.

Trained linguists created the sentences by hand, following a greedy approach to accommodate the words chosen by the set-cover algorithm while adding as few extra words as possible. So while grammatically correct, some sentences use unusual word choices. The deck shows a page of these Urdu sentences under the title "Chomsky's Colorless Green Ideas", after Chomsky's (1957) example *Colorless green ideas sleep furiously*, a sentence that is grammatical but semantically odd. Compare the sizes: about 10,000 tokens chosen for coverage, against the roughly 121 million a random Zipfian sample would need for a similar goal in [[#2.6 Worked example with 100,000 word types|§2.6]].

### 3.3 Outcomes

- **Urdu ASR (Automatic Speech Recognition), 2010.** Adding PRUS improved the WER (word error rate, the fraction of words wrongly recognized) on spontaneous speech from 22.9% (spontaneous training data alone) to 18.8%.
- **Deepfake defense, Urdu TTS (Text-to-Speech), ACL 2024.** The professor's team recorded PRUS's 708 sentences with 17 speakers (7 women, 10 men) to train speaker-specific, natural-sounding TTS systems using Tacotron and VITS (two neural TTS architectures).

---

## 4. The Pareto Principle and the Pareto Distribution

Zipf's law describes concentration: a few words take most of the occurrences. That raises a familiar name, the 80/20 rule, and the question of how it relates to power-law loss.

### 4.1 Two different questions

They are related through power laws, but they describe different things. The **Pareto principle**, often called the **80/20 rule**, says that a small fraction of items accounts for a large fraction of the total, for example 20% of people holding 80% of wealth. Its mathematical connection is to the **Pareto distribution**, a power-law distribution. But 80/20 is not a universal ratio: the concentration depends on the distribution's exponent.

| | What it describes | Main idea |
|---|---|---|
| Pareto principle | How a total is distributed across items | A small fraction accounts for a large share |
| Power-law loss, $L(x) = Ax^{-\alpha}$ | How loss changes as resources increase | Each fixed proportional reduction in loss requires a fixed resource multiplier |

The three language-related ideas answer three different questions:

| Concept | Question it answers |
|---|---|
| Zipf's law | How frequent is the word at rank $r$? |
| Pareto distribution | What fraction of word types have frequency at least $t$? |
| Pareto principle | Does a small fraction of types account for a large fraction of all tokens? |

### 4.2 Zipf's law gives a Pareto-like concentration

If $x_r = x_1/r$, the total number of tokens is $x_1 H_V$ (as in [[#2.4 How large must a corpus be|§2.4]]) and the most frequent $m$ types contribute $x_1 H_m$. Therefore

$$\text{share contributed by the top } m \text{ types} = \frac{x_1 H_m}{x_1 H_V} = \boxed{\frac{H_m}{H_V}}.$$

For $V = 100{,}000$, the top 20% means $m = 20{,}000$, and

$$\frac{H_{20{,}000}}{H_{100{,}000}} \approx \frac{10.4807}{12.0901} \approx 86.7\%.$$

So this ideal Zipf model gives "20% of word types account for about 87% of tokens": a Pareto-like concentration, though not exactly 80/20.

### 4.3 The Pareto distribution

A **Pareto distribution** describes a quantity for which most items have small values while a few have very large values, for example many words with low frequencies and a few with enormous frequencies. That description alone is not enough. Its defining feature is: **every time you multiply the threshold by the same factor, the fraction of items exceeding it shrinks by the same factor.** In a hypothetical corpus with 10,000 distinct word types:

| Frequency threshold | Number of word types occurring at least this often |
|---|---|
| 10 occurrences | 10,000 |
| 100 occurrences | 1,000 |
| 1,000 occurrences | 100 |
| 10,000 occurrences | 10 |

Each tenfold rise in the threshold cuts the number of qualifying types tenfold: a Pareto pattern with exponent 1. *(Expanding on this, the link to Zipf: if $f(r) = K/r$, a type has frequency at least $t$ exactly when $K/r \ge t$, that is $r \le K/t$. So the number of types with frequency at least $t$ is $K/t$, a Pareto tail with exponent 1. Zipf's rank–frequency law and the Pareto frequency distribution are the same fact read along different axes.)*

---

## 5. Properties of Power-Law Loss

With that detour done ("and we are back!"), the deck returns to [[#^eq-1|eq. (1)]] and draws out the properties that make it useful for planning training runs.

### 5.1 Scale invariance and diminishing returns

Multiply resources by $k > 1$. Then

$$\frac{L(kx)}{L(x)} = \frac{A(kx)^{-\alpha}}{Ax^{-\alpha}} = \frac{Ak^{-\alpha}x^{-\alpha}}{Ax^{-\alpha}} = \boxed{k^{-\alpha}}.$$

The ratio does not depend on $x$. For $\alpha = 0.5$, every doubling gives $L(2x)/L(x) = 2^{-0.5} \approx 0.707$, so every doubling reduces loss by about 29.3%, whether we go from 10 to 20 resource units or from 1,000 to 2,000. This is **scale invariance**: rescaling the input changes the output by a predictable multiplicative factor while preserving the form of the relationship.

A power law has **diminishing absolute returns, even though proportional returns remain constant**. In the running example, successive fourfold increases produce $4 \to 2 \to 1 \to 0.5$; the absolute reductions are 2, 1, 0.5. Each reduction is smaller, and obtaining it needs a larger absolute resource increase (3, then 12, then 48 units). Mathematically,

$$\frac{dL}{dx} = -\alpha A x^{-\alpha-1}.$$

The derivative is negative, so loss decreases; its magnitude shrinks as $x$ grows, so the curve flattens. The deck's illustration is a climber: easy early gains, harder improvements, costly further gains.

### 5.2 How expensive the next halving is

The exponent tells us how expensive further improvement will be. To halve the loss we need a resource multiplier $k$ with

$$k^{-\alpha} = \frac{1}{2} \;\Longrightarrow\; \boxed{k = 2^{1/\alpha}}.$$

| Exponent $\alpha$ | Resource multiplier needed to halve loss |
|---|---|
| 1 | 2× |
| 0.5 | 4× |
| 0.25 | 16× |
| 0.1 | 1,024× |

A small exponent means loss keeps improving, but substantial improvements can be extremely expensive. The language-model exponents in [[#9.2 Finding 1 — loss follows power laws|§9.2]] are around 0.05–0.1, which is why frontier training runs are so large.

### 5.3 Realistic curves have a floor

A more general model is

$$L(x) = L_\infty + Ax^{-\alpha} \qquad\qquad\text{(3)}$$
^eq-3

where $L_\infty$ is the **limiting loss** as this resource grows without bound, under the assumed setup. It may reflect irreducible uncertainty (for language, genuine unpredictability of the next word) or limits from resources held fixed. Now the power law applies to the **excess loss** above the floor: $L(x) - L_\infty = Ax^{-\alpha}$. The dashed orange curve in [[#1.3 Reading linear, semi-log and log–log plots|§1.3]]'s figure uses $L_\infty = 1$, that is $L(x) = 1 + 4x^{-0.5}$. Its total loss bends toward 1, even on log–log axes. To recover a straight line, plot $L - L_\infty$ against $x$ with both axes logarithmic. With a floor, each doubling removes a fixed fraction of the **remaining reducible loss**, not of the total loss.

### 5.4 Why this matters for neural networks

Kaplan and colleagues ([[#9. Kaplan et al. (2020)|§9]]) found approximate power-law relationships between language-model cross-entropy loss and model size, dataset size and training compute, within the regimes they studied. That makes power laws useful for:

- **Prediction:** estimating larger-scale performance from smaller experiments, when the same regime continues.
- **Planning:** estimating the resources required for a target loss.
- **Comparing methods:** distinguishing a lower loss curve (a smaller $A$) from a steeper scaling exponent (a larger $\alpha$).
- **Diagnosing limitations:** investigating departures from the expected trend, such as a floor or another resource becoming a bottleneck.

These are **empirical approximations**, so a straight segment on a log–log plot does not establish that the law will continue indefinitely.

---

## 6. Power-Law Decay vs Exponential Decay

A power law has a slowly shrinking tail. To see what "slow" means, and what a faster alternative would look like, the deck compares it with exponential decay.

### 6.1 Multiply vs add

In power-law decay, $x$ is the base of a power; in exponential decay, $x$ is in the exponent. Here $A, \alpha, b > 0$ are fixed constants.

| | Power-law decay | Exponential decay |
|---|---|---|
| Formula | $L(x) = Ax^{-\alpha}$ | $L(x) = Ae^{-bx}$ |
| Equivalent form | $A/x^{\alpha}$ | $A/(e^{b})^{x}$ |
| What varies? | The base $x$; the exponent $\alpha$ is fixed | The exponent $x$; the base $e^{b}$ is fixed |

The consequence is how much additional $x$ you need for the same fractional reduction in loss. For a power law, **multiplying** $x$ by a fixed factor gives a fixed fractional reduction, $L(kx)/L(x) = k^{-\alpha}$ ([[#5.1 Scale invariance and diminishing returns|§5.1]]). With $L(x) = 1/x$, halving the loss requires doubling $x$ each time: $x = 1, 2, 4, 8$ gives $L = 1, \tfrac12, \tfrac14, \tfrac18$. For an exponential, **adding** a fixed amount gives a fixed fractional reduction:

$$\frac{L(x+\Delta)}{L(x)} = \frac{Ae^{-b(x+\Delta)}}{Ae^{-bx}} = e^{-b\Delta}.$$

With $L(x) = 2^{-x} = e^{-(\ln 2)x}$, halving the loss requires adding 1 to $x$ each time: $x = 0, 1, 2, 3$ gives $L = 1, \tfrac12, \tfrac14, \tfrac18$. So to repeatedly halve loss, a power law multiplies resources by the same factor each time, an exponential adds the same amount. That is why exponential decay eventually becomes faster than power-law decay: power-law decay is slower at sufficiently large $x$.

### 6.2 Which axes straighten which curve

An exponential $L(x) = Ae^{-bx}$, with decay rate $b > 0$, becomes straight when only the loss axis is logarithmic; a power law needs both axes logarithmic. The deck plots two curves that start together: the power law $L_p(x) = x^{-1}$ and the exponential $L_e(x) = e^{-(x-1)}$. Both equal 1 at $x = 1$ with the same initial slope of −1.

![[L10_power_vs_exp.png|700]]

*The same two curves on three axis scales over $1 \le x \le 10$. Left, linear axes: the exponential (dashed orange) drops toward zero much faster and the power law (blue) keeps a longer tail. Middle, log y-axis: the exponential is straight. Right, log–log: the power law is straight and the exponential bends sharply down.*

On the middle panel the exponential is straight because $\ln L_e(x) = -(x-1) = 1 - x$, linear in $x$. On the right panel the power law is straight because $\ln L_p(x) = -\ln x$, linear in $\ln x$ with slope −1.

| Power-law decay $L_p(x) = 1/x$ | Exponential decay $L_e(x) = e^{-(x-1)}$ |
|---|---|
| A fixed multiplication of $x$ removes a fixed fraction of loss | A fixed addition to $x$ removes a fixed fraction of loss |
| Doubling $x$ halves the loss | Adding $\ln 2 \approx 0.693$ to $x$ halves the loss |
| Successive halvings need increasingly large increments in $x$ | Successive halvings need the same increment in $x$ |

### 6.3 Any exponential is straight on a semi-log plot

The logarithm does not need to "cancel" the exponential's base; it only needs to bring the exponent down, and logarithms do that for every base. Write exponential decay generally as $L(x) = Aq^{x}$ with $A > 0$ and $0 < q < 1$ fixed (for example $q = 1/2$). Taking the natural logarithm,

$$\ln L(x) = \ln(Aq^{x}) = \ln A + \ln(q^{x}) = \underbrace{\ln A}_{\text{intercept}} + \underbrace{(\ln q)}_{\text{slope}}\,x.$$

Since $q$ is fixed, $\ln q$ is just a constant, so this is the straight line $Y = c + mx$. For example $L(x) = 2^{-x}$ gives $\ln L(x) = -x\ln 2 \approx -0.6931x$: nothing cancels, yet the result is linear in $x$. Using $e$ only made the coefficient cleaner earlier, $\ln(Ae^{-bx}) = \ln A - bx$ because $\ln e = 1$. The log axis can also use any base, such as 10: $\log_{10}L(x) = \log_{10}A - bx\log_{10}e$, still a straight line with a different numerical slope. So the precise claim is: **a pure exponential becomes straight when the vertical axis is logarithmic and the horizontal axis stays linear, regardless of the exponential base or the logarithm base.**

---

## 7. Breaking Beyond Power-Law Scaling

[[#6. Power-Law Decay vs Exponential Decay|§6]] showed that exponential decay eventually beats any power law. The natural question is whether training can be changed so that loss falls exponentially rather than as a power law.

### 7.1 Excess loss and the escalating cost of each halving

To **break beyond power-law loss** means to make loss decrease with resources faster than the power-law trend would predict. The strongest meaning is a change in the *form* of the scaling relationship, for example from power-law decay to exponential decay. There is an important distinction between **beating a particular power-law curve** and **moving beyond power-law scaling altogether**. Start from the excess loss of [[#^eq-3|eq. (3)]]:

$$E(x) = L(x) - L_\infty = Ax^{-\alpha},$$

where $x$ is the resource amount, $L(x)$ the total loss, $L_\infty$ its limiting floor, $E(x)$ the remaining reducible loss, and $A, \alpha > 0$ set the curve. Using excess loss keeps the comparison meaningful: we are discussing how quickly we approach the floor.

Recall the difficulty a power law imposes. With $E(x) = x^{-1/2}$, every fourfold increase in resources halves the remaining loss:

| Resource $x$ | Excess loss $E(x)$ |
|---|---|
| 1 | 1 |
| 4 | 1/2 |
| 16 | 1/4 |
| 64 | 1/8 |
| 256 | 1/16 |

The successive resource additions are 3, 12, 48, 192: each new halving is more expensive than the previous one. When researchers talk about going beyond power laws, this escalating cost is what they hope to improve.

### 7.2 Three levels of improvement

| What changes? | Example | What it means |
|---|---|---|
| Lower coefficient | $E_{\text{old}} = x^{-0.5}$, $E_{\text{new}} = 0.5x^{-0.5}$ | Half the excess loss at every resource level, but the same scaling rate |
| Larger exponent | $E_{\text{old}} = x^{-0.5}$, $E_{\text{new}} = x^{-1}$ | Loss improves faster as resources grow, but still follows a power law |
| Different functional form | $E_{\text{old}} = x^{-0.5}$, $E_{\text{new}} = Be^{-bx}$ | Exponential decay eventually outpaces every fixed inverse power of $x$; here $B, b > 0$ |

On a log–log plot these appear as a **parallel downward shift** (lower coefficient), a **steeper straight line** (larger exponent), and a **curve that bends downward and becomes progressively steeper** (exponential). The first two can beat the old scaling *law*; the third goes beyond the power-law *family*.

Why would exponential decay be such a substantial change? For a power law, halving the remaining loss requires multiplying resources by $x_{\text{new}}/x_{\text{old}} = 2^{1/\alpha}$ ([[#5.2 How expensive the next halving is|§5.2]]). For $E(x) = Be^{-bx}$, halving it requires adding

$$x_{\text{new}} - x_{\text{old}} = \frac{\ln 2}{b}.$$

Each halving costs the same resource **multiplier** under a power law but the same resource **addition** under an exponential. That is exactly what the comparison plot in [[#6.2 Which axes straighten which curve|§6.2]] illustrated: the exponential escapes the power law's slowly shrinking tail.

### 7.3 Selective sampling

How might scaling be broken? The first observation is that **more data may add little useful information**. As the model improves, additional randomly sampled examples may mostly repeat patterns it already knows (the Zipfian long tail of [[#2.3 Why the pattern matters for NLP|§2.3]]), while its remaining mistakes may need examples that random sampling rarely supplies. One solution is to **select examples that address those remaining weaknesses**, which could give more improvement per training example.

Consider data scaling, with $x$ the number of training examples. If we keep adding randomly sampled examples, many repeat patterns the model already handles well and carry little information about its remaining mistakes. If instead we can select examples that are especially informative at the model's current stage of learning, a retained example might contribute more than a randomly chosen one. The deck's classroom analogy: "A student could solve another thousand randomly selected arithmetic questions. Or a teacher could identify the student's remaining misconceptions and choose exercises that address them." The analogy explains **why** selection might improve scaling; it does not establish that any particular selection method will produce exponential improvement.

Sorscher and colleagues' *Beyond neural scaling laws: beating power law scaling via data pruning* (2022) studies this possibility. They show theoretically that sufficiently good pruning can potentially yield exponential scaling, and report better-than-power-law behaviour **with retained dataset size** in ResNet experiments on CIFAR-10, SVHN and ImageNet. [[#12.7 Data pruning revisited (Sorscher et al., 2022)|§12.7]] returns to which examples to keep. This is the same idea as the PRUS corpus of [[#3. Coverage From Little Data — the PRUS Corpus|§3]]: choose data by what it adds, not by volume.

### 7.4 Finding useful examples also costs resources

The second observation is a caution. Suppose we inspect and score 1 million examples, select 10,000 informative ones, and train on those 10,000. We may get excellent results **per example used in training**, but we still paid to inspect the million. The resource on the horizontal axis matters: selection may improve loss versus *examples used for training*, but to claim the same improvement versus *total compute or cost*, we must also account for finding, scoring and selecting those examples. Breaking beyond power-law scaling therefore means **changing the learning process so that additional resources reduce the remaining error faster than the established power-law trend**, counting every resource spent.

---

## 8. Terminology and the General Scaling Law

The mathematics is now in place. Before the papers, the deck fixes vocabulary, starting from the question the rest of the lecture answers.

### 8.1 What to scale

Suppose you are given twice the GPU budget you used last time. What exactly do you double: the model, the data, or both? The deck promises a formula, and adds that the answer keeps changing. It changes three times in this lecture: Kaplan says mostly the model ([[#9.7 Finding 2 — spend the budget mostly on parameters|§9.7]]), Chinchilla says both equally, about 1.4× each ([[#10.9 What it means in practice|§10.9]]), and inference cost or data scarcity move the answer again ([[#12. When the Data Runs Out (Muennighoff et al., 2023)|§12]], [[#15. Train-Time or Test-Time — Lifetime Cost|§15]]).

The "6-year journey" of scaling laws describes the predictable relationship between model performance (measured as loss) and three key factors:

- $N$ (parameters): the size of the brain.
- $D$ (data tokens): the experiences it learns from.
- $C$ (compute): time × resources spent training.
- $L$ (loss): how "confused" the model is; lower is better.

### 8.2 Recurring symbols

| Symbol | Meaning |
|---|---|
| $N$ | Number of model parameters; the counting convention must be specified (e.g. with or without embeddings, [[#9.1 The question and the experiments\|§9.1]]) |
| $D$ | Training token **presentations**, including repeated exposure |
| $U$ | **Unique** training tokens, under a stated deduplication policy |
| $C$ | Computation, measured in floating-point operations (FLOPs, [[#9.3 FLOPs and the rule C ≈ 6ND\|§9.3]]) |
| $L$ | Held-out next-token cross-entropy loss, in nats per token (natural-log units); lower is better |
| $E$ | Fitted limiting loss in a scaling model, the $L_\infty$ of [[#^eq-3\|eq. (3)]] |
| $T$ | Total inference tokens processed over the model's deployed lifetime (written $I$ in [[#15.2 Training and inference compute\|§15.2]]) |
| $k$ | Number of candidate answers sampled for a question |

### 8.3 Unique tokens vs token presentations

Suppose a corpus contains one billion tokens, $U = 10^9$. One complete pass presents one billion tokens to the model, $D = 10^9$. Four complete passes give $D = 4\times10^9$, but the underlying corpus is still the same size. In general, with $e$ complete **epochs** (passes over the data),

$$D = eU.$$

Computation grows with the number of presentations. Whether learning improves as much as it would from fresh text is a separate question. Compare four billion tokens of different text presented once with one billion tokens presented four times: both have $D = 4\times10^9$ presentations, but they do not give the same training experience. This distinction becomes central for **low-resource languages**: a prescription requiring more training tokens does not create additional unique Urdu text. For now the deck assumes extra training tokens can be drawn from a sufficiently large, consistently prepared corpus, and relaxes that assumption in [[#12. When the Data Runs Out (Muennighoff et al., 2023)|§12]].

### 8.4 The general form and the excess loss

Imagine training a series of models while varying one resource $X$: model size, dataset size or computation, depending on the experiment. We measure the resulting loss and ask whether a simple relationship describes how performance changes. An important empirical observation is that language-model loss can follow approximate power-law relationships over substantial ranges of model size, data and compute, under the experimental conditions studied (Kaplan et al.). The general illustrative form is [[#^eq-3|eq. (3)]] with the deck's new letters:

$$L(X) = E + \frac{a}{X^{\alpha}} = E + aX^{-\alpha}, \qquad a > 0,\ \alpha > 0. \qquad\qquad\text{(4)}$$
^eq-4

- The term $a/X^{\alpha}$ decreases as $X$ increases; it is the part of the loss that more of this resource can reduce in this model.
- The constant $E$ is the limiting value, $\lim_{X\to\infty} L(X) = E$.
- The **excess loss** above the modeled floor is $\Delta L(X) = L(X) - E$, and substituting gives $\Delta L(X) = aX^{-\alpha}$. This is the quantity that obeys the power law.
- The total loss therefore contains a component that this particular model says cannot be reduced merely by increasing $X$.
- We should not immediately interpret the fitted $E$ as the true irreducible uncertainty of language. It is an asymptote in a fitted model; its interpretation and reliability depend on the assumptions and experiments behind the fit.

### 8.5 Three regions of a learning curve

A power law describes a useful *region* of learning; it need not describe the whole journey from almost no data to saturation. Hestness et al. (2017), *Deep Learning Scaling is Predictable, Empirically*, sketch the full curve.

![[L10_learning_curve_regions.png|520]]

*Generalization error vs training-set size, both on log scales. Left, small-data region: the model sits near the green **best-guess error**. Middle, power-law region: error falls along a straight line. Right, irreducible-error region: the curve flattens onto the red floor. (Hestness et al., 2017, Figure 6.)*

In the small-data region the model performs near the **best-guess baseline**, the error of a predictor that has learned almost nothing from the data (for example, always guessing the most common answer). In the middle region error falls approximately along a straight line on these axes. On the right the curve approaches the error floor and flattens, the $E$ of [[#^eq-4|eq. (4)]].

The same paper gives early language-modelling evidence that this predictability predates modern decoder-only Transformers.

![[L10_hestness_lstm.png|700]]

*Hestness et al. (2017), Figure 2. Both x-axes are training-set size in millions of words (log scale). Left: the best validation loss of 2-layer LSTMs, 4-layer LSTMs and depth-5 RHNs falls along near-parallel power-law trends, $\varepsilon(m) \approx 12\,m^{-0.066}$. Right: the model size that reaches that best loss also grows as a power of the data, $s(m) \propto m^{0.7}$ roughly.*

The left panel shows the best validation loss reached by different LSTM (long short-term memory) and recurrent-network families (RHN, recurrent highway network): their descending trends are similar despite architectural differences. The right panel shows how the model sizes associated with those experiments grow with the dataset, so the useful capacity also changes as more data becomes available. Predictable improvement with data predates modern decoder-only language models.

---

## 9. Kaplan et al. (2020)

The general form of [[#^eq-4|eq. (4)]] needs its constants measured for real language models. The first systematic measurement, and the first answer to "how should a fixed training budget be spent", came from OpenAI. (The same material is summarized in [[ATML_Lecture_7 - LLM Foundations#5.2 Kaplan laws|ATML L7 §5.2]].)

### 9.1 The question and the experiments

Before 2020 nobody could say in advance how much better a model would be at ten times the size or ten times the data. Without that, a multi-million-dollar training run could not be planned from cheap small runs, and nobody knew whether money was better spent on a bigger model, more data or longer training.

Kaplan et al., *Scaling Laws for Neural Language Models* (arXiv:2001.08361), trained hundreds of decoder-only Transformers on WebText2, from 768 to 1.5 billion **non-embedding parameters** (all the model's weights except the token embedding matrix and the positional embeddings) and on 22 million to 23 billion tokens. They varied size, data and training length one at a time and recorded full learning curves. Embeddings are excluded because for the tiniest models the embedding table alone would be millions of parameters and would dwarf everything else, hiding the trend.

In headline form, OpenAI found that loss follows smooth, predictable power laws in each resource:

$$L \propto \frac{1}{N^{\alpha}}\ \text{(more parameters, less confusion)}, \qquad L \propto \frac{1}{D^{\beta}}\ \text{(more data)}, \qquad L \propto \frac{1}{C^{\gamma}}\ \text{(more compute)}.$$

This means performance can be predicted **before** spending millions on training. Their key insights: performance improves smoothly and predictably with scale; larger models are more sample-efficient (they learn more per token); and model size matters more than training time for a fixed compute budget. Their recommendations were to scale model size aggressively, even if undertrained, allocating most compute to larger models trained for fewer steps. This led to GPT-3: 175 billion parameters trained on only 300 billion tokens (a ratio of 1.7 tokens per parameter).

### 9.2 Finding 1 — loss follows power laws

Test loss falls as a smooth power law in each of three resources, over six or more orders of magnitude, as long as the other two are not the bottleneck:

$$L(N) = \left(\frac{N_c}{N}\right)^{0.076}, \qquad L(D) = \left(\frac{D_c}{D}\right)^{0.095}, \qquad L(C) = \left(\frac{C_c}{C}\right)^{0.050}.$$

Here $L$ is held-out cross-entropy in nats per token; $N$ is non-embedding parameters; $D$ is dataset size in tokens; $C$ is training compute in **PF-days** ([[#9.4 The PF-day|§9.4]]); $N_c, D_c, C_c$ are fitted constants that set the height of each line, while the exponents ($\alpha_N = 0.076$, $\alpha_D = 0.095$, $\alpha_C = 0.050$) set its slope. These laws have no floor term; they are [[#^eq-1|eq. (1)]] written as $(\text{constant}/x)^{\alpha}$. By [[#5.2 How expensive the next halving is|§5.2]], halving the loss through compute alone would take $2^{1/0.05} = 2^{20}$, about a million times more compute.

Two further results: shape details such as depth versus width change loss by only a few percent at a fixed $N$, so the parameter count, not the architecture shape, is what matters; and compute is counted with a rule every later paper uses, $C \approx 6ND$.

### 9.3 FLOPs and the rule C ≈ 6ND

A **FLOP** is one floating-point operation: a single arithmetic step, such as an addition or a multiplication, on floating-point (real-valued) numbers. Two usages are easy to confuse. **FLOP / FLOPs** is a count of operations, meaning total work; $C$ in the scaling laws is a count. **FLOP/s** (often written FLOPS) is operations per second, a hardware speed.

In neural networks most of the work is **multiply-accumulate**, $a\cdot b + c$, which counts as 2 FLOPs. The accumulation is the addition into a running sum, not the assignment. Consider one output of a matrix-vector product, $y_i = \sum_j W_{ij}x_j$, computed term by term:

```
acc = 0
for j:
    acc = acc + W[i,j] * x[j]
```

Each loop iteration does one multiply ($W\cdot x$) and one add (acc + product), so it counts as 2 FLOPs; the "accumulate" is that add, folding the new product into the partial sum in `acc`. The assignment (`acc = ...`) just stores the result in a register; it is data movement, not arithmetic, and is not counted. Hardware usually fuses the multiply and add into one instruction (**FMA**, fused multiply-add), but by convention it still counts as 2 FLOPs. Since every weight $W_{ij}$ appears in exactly one such step per token, the **forward pass costs about 2 FLOPs per parameter per token**. (This is the matmul cost rule of [[Lecture_06_Decoding_KV_Cache_MQA_GQA_FlashAttention#3.1 Notation and the matmul cost rule|Lecture 6 §3.1]] applied to one token.)

The backward pass costs about twice the forward pass. For each weight matrix in a layer $y = Wx$, it computes two matrix products, each costing about as much as the forward pass:

- The **gradient with respect to the input**, to pass the error back to the previous layer: $\dfrac{\partial L}{\partial x} = W^{\top}\dfrac{\partial L}{\partial y}$. Every weight takes part in one multiply-accumulate, so 2 FLOPs per parameter per token.
- The **gradient with respect to the weights**, needed for the update: $\dfrac{\partial L}{\partial W} = \dfrac{\partial L}{\partial y}x^{\top}$. Each weight's gradient gets one multiply-accumulate per token, another 2 FLOPs per parameter per token.

That gives 4 FLOPs per parameter per token backward, or $4ND$ over $N$ parameters and $D$ tokens. Adding the forward pass's $2ND$,

$$\boxed{C_{\text{train}} \approx 6ND}, \qquad C_{\text{inference}} \approx 2N \text{ per token}. \qquad\qquad\text{(5)}$$
^eq-5

Serving a trained model needs only the forward pass, about $2N$ FLOPs per token. *(Expanding on this: the rule ignores the attention-score computation, whose cost grows with context length rather than with $N$, and which is small relative to $6N$ per token unless the context is very long. For an MoE model, $N$ here is the **active** parameter count of [[Lecture_09_FFNs_Dense_and_MoE#6.2 Stored vs used parameters|Lecture 9 §6.2]], since only active weights do arithmetic for a token.)*

### 9.4 The PF-day

A **PF-day** (petaflop/s-day) is a unit of total compute: running at $10^{15}$ floating-point operations per second for one full day,

$$1\ \text{PF-day} = 10^{15}\,\frac{\text{FLOP}}{\text{s}} \times 86{,}400\ \text{s} \approx 8.64\times10^{19}\ \text{FLOPs}.$$

It measures an amount of work, not a speed: the "/s" and the "day" cancel, much as a light-year is a distance, not a time. A car at 60 km/h for a day covers $60 \times 24 = 1{,}440$ km, which we could call a "60 km-day". The same PF-day could be 1 petaflop/s for a day, 24 petaflop/s for an hour, or a single slower GPU for months. In Kaplan et al., $C$ is measured in PF-days and estimated as $C \approx 6ND$. For scale, the GPT-3 paper reports its 175B model took roughly 3,640 PF-days to train, around $3\times10^{23}$ FLOPs ($3{,}640 \times 8.64\times10^{19} \approx 3.1\times10^{23}$). The Kaplan experiments span tiny fractions of a PF-day up to a few PF-days.

### 9.5 When each single-variable law holds

Each formula in [[#9.2 Finding 1 — loss follows power laws|§9.2]] describes what happens when one resource is scaled while the other two are plentiful enough not to hold the loss back:

- $L(N)$ holds when the model trains on enough data, for long enough, to approach convergence; loss then depends only on model size.
- $L(D)$ holds when the model is large enough, and trained with **early stopping** (halting when held-out loss stops improving), so that data is the only limit.
- $L(C)$ holds when a fixed compute budget is split optimally between model size and training steps.

If one of the other two is scarce, the curve stops following its power law and plateaus. Take a 1-billion-parameter model trained on only 22 million tokens: making it larger barely helps, because it will just overfit or memorize. Data is the bottleneck, and loss is set by $D$, not $N$.

The paper makes this precise with a combined law:

$$L(N, D) = \left[\left(\frac{N_c}{N}\right)^{\alpha_N/\alpha_D} + \frac{D_c}{D}\right]^{\alpha_D}. \qquad\qquad\text{(6)}$$
^eq-6

Whichever term is larger dominates. Grow $N$ while holding $D$ fixed and the first term shrinks toward zero, but the $D$ term remains, so loss levels off. Each single-variable law is this formula with the other term ($N_c/N$ or $D_c/D$) negligible. *(Check: with $D \to \infty$ the bracket is $(N_c/N)^{\alpha_N/\alpha_D}$, and raising it to $\alpha_D$ gives $(N_c/N)^{\alpha_N}$, which is $L(N)$.)*

How did Kaplan keep the single-variable laws clean? For $L(N)$, each model trained on the full WebText2 training set (about 23 billion tokens); even the largest model, 1.5 billion parameters, is small relative to that, so the $D$ term stayed small. For $L(D)$, a large model trained on subsets of the data with early stopping, so $N$ was large enough that the $N$ term stayed small. So for $L(N)$, Kaplan did not scale $D$ up with $N$; he held $D$ fixed at a value big enough for every model tested.

### 9.6 Six orders of magnitude

An **order of magnitude** is a factor of 10, so "six or more orders of magnitude" means the power law holds across a range where the largest value is at least $10^6$ (a million) times the smallest. For model size, $N$ runs from 768 to about 1.5 billion non-embedding parameters, a ratio of about $2\times10^6$, or 6.3 orders of magnitude, and the loss stays on one straight line on a log–log plot the whole way. Compute spanned an even wider range. Data spanned less: 22 million to 23 billion tokens is only about 3 orders of magnitude. The point is that the trend is robust, not a local fit: a straight line on log–log axes over a millionfold range is strong evidence of a genuine power law, which is what justifies extrapolating it to larger models.

### 9.7 Finding 2 — spend the budget mostly on parameters

On the most efficient path, as compute grows, model size grows as $N_{\text{opt}} \propto C^{0.73}$ and training tokens only as $D_{\text{opt}} \propto C^{0.27}$. Ten times the compute should buy about 5.4 times the parameters and only 1.9 times the data. This is just plugging in a factor of 10:

$$10^{0.73} \approx 5.4, \qquad 10^{0.27} \approx 1.9, \qquad 5.4 \times 1.9 \approx 10.$$

The product recovers the 10× compute, as $C \propto ND$ requires; equivalently, the two exponents must add to 1.

### 9.8 Reading the compute envelope

![[L10_kaplan_envelope.png|450]]

*Kaplan et al.: each light-blue curve is one model size during training, plotted as test loss against compute used so far (PF-days, non-embedding). The black envelope is the lowest loss reachable at each compute; the orange dashed line is its power-law fit, $L = (C_{\min}/2.3\cdot10^{8})^{-0.050}$.*

Each blue curve is one model size during training. A model of size $N$ starts at the left and moves right as it trains: its x-coordinate is the compute spent so far, $C = 6N D_i$ with $D_i$ the tokens seen so far, and its y-coordinate is its test loss at that moment. Moving along one blue curve means training the same model on more data.

A vertical line at any budget $C$ compares all the options. Each blue curve crossing that line is a different way to spend exactly $C$: a small model trained on many tokens, or a large model trained on few. The lowest crossing is the best choice for that budget, and the black envelope is that minimum at every $C$:

$$L(C) = \min_{N} L\big(N \text{ trained until it has used } C\big).$$

The model that touches the envelope at $C$ is $N_{\text{opt}}(C)$, and the tokens it has seen there are $D_{\text{opt}} = C/(6N_{\text{opt}})$. Reading off these touching points across all $C$ gives the $(N_{\text{opt}}, D_{\text{opt}})$ pairs, and that sequence is the efficient path.

**Why the curves peel off.** A small model hits the envelope early. Then it starts to saturate: its curve flattens and rises above the envelope, because extra compute buys it little. At that point a larger model, still improving steeply, takes over the envelope. This handoff to ever-larger models is what Finding 2 quantifies: as $C$ grows, the optimal model grows quickly ($C^{0.73}$) while each model is trained only modestly longer ($C^{0.27}$). The slope of the orange fit on log–log axes, −0.050, is the $\alpha_C$ in $L(C)$.

### 9.9 Why the steeper data exponent does not mean spend on data

$L(D)$ has the steepest exponent (0.095 vs 0.076 for $N$), so the natural intuition is "data has the steeper exponent, so spend on data". It runs backwards: the steep exponent is exactly why less data growth is needed.

1. **The steep $L(D)$ assumes a big model.** $L(D) \propto D^{-0.095}$ was measured with models large enough that $N$ was not the bottleneck ([[#9.5 When each single-variable law holds|§9.5]]). Adding data to a small model does not follow that curve: the model saturates and loss plateaus, as with the small-model curves peeling off the envelope. You cannot get the steep data exponent without also growing $N$.
2. **Under a budget, the two error terms must be balanced.** Think of loss as two contributions, one from finite $N$ and one from finite $D$. Since $C \approx 6ND$, a 10× increase in compute costs the same whether it goes to $N$ or to $D$.
3. **The slow resource needs bigger multipliers.** Suppose you want to cut each term by 10%. To cut $L(N)$ by 10%, the factor $k$ by which to multiply $N$ satisfies
$$k^{-0.076} = 0.9 \;\Rightarrow\; k = 0.9^{-1/0.076} = e^{0.105/0.076} = e^{1.39} \approx 4.0.$$
The same calculation for $D$ gives $e^{0.105/0.095} = e^{1.11} \approx 3.0$. A 10% cut in the $N$ term needs 4× more parameters, the same cut in the $D$ term only 3× more data, because $D$'s exponent is steeper. Each multiplier costs the same compute, so the resource with the smaller exponent needs the larger increase to keep pace. Applying this to the combined law of [[#^eq-6|eq. (6)]] gives $N \propto C^{0.56}$ and $D \propto C^{0.44}$: $N$ already grows faster, even without Kaplan's further effects.
4. **Kaplan's 0.73 adds one more effect.** Larger models are more sample-efficient: they reach a given loss after seeing fewer tokens. So growing $N$ also reduces the data you need, which pushes the split further toward parameters.

In short, a high exponent marks a resource that pays off quickly and saturates quickly. Budget flows to the resource that improves slowly, because that one becomes the bottleneck.

*(The deck states the 0.56 / 0.44 split; the derivation fills it in.)* Minimizing the bracket of [[#^eq-6|eq. (6)]], $a N^{-\alpha_N/\alpha_D} + b D^{-1}$ with $a = N_c^{\alpha_N/\alpha_D}$ and $b = D_c$, subject to $ND \propto C$: substitute $D \propto C/N$ and set the derivative in $N$ to zero. Both terms then scale the same way at the optimum, which forces $D \propto N^{\alpha_N/\alpha_D} = N^{0.8}$. Then $C \propto ND \propto N^{1.8}$, so $N \propto C^{1/1.8} = C^{0.56}$ and $D \propto C^{0.8/1.8} = C^{0.44}$.

### 9.10 Take-aways and caveats

Why did parameters get most of the budget? Three reasons from the paper:

- **Larger models learn more from each token.** Plotting loss against tokens seen, a larger model reaches any given loss after fewer tokens than a smaller one; across their range, sample efficiency improved by a large factor. If a big model squeezes more out of each token, then for a fixed budget it seems better to have a big model see fewer tokens than a small model see many.
- **Training to convergence looked wasteful.** They fitted how loss falls with training steps for each model size. That fit said the best use of a fixed budget is to stop a model while its loss is still about 10% above the value it would eventually reach; past that point the same compute buys more by switching to a larger model. As the budget grows, the cheapest way to lower loss is therefore a bigger model stopped early.
- **Data needs to grow slower than size.** Their joint fit of $N$ and $D$ said that an 8 times larger model needs only about 5 times more data ($D \propto N^{0.74}$, and $8^{0.74} \approx 4.7$).

Take-aways: scale is forecastable (small, cheap runs predict large ones, provided the training recipe is held fixed); improvement is costly (10% lower loss costs a multiple of the resource, so allocation decisions matter); $C \approx 6ND$ for training and about $2N$ per token for inference; and Kaplan's advice to put most new compute into parameters shaped the models of 2020 and 2021, such as GPT-3 at 175B parameters, 300B tokens, about 1.7 tokens per parameter.

**Caveats.** All runs used the same schedule: a fixed 3,000-step warmup and a cosine decay planned for a fixed, long training length, with losses for shorter training read off intermediate checkpoints. Most runs were also small (many under 100M parameters). The next two papers ([[#10. Chinchilla (Hoffmann et al., 2022)|§10]], [[#11. Why Kaplan Got a Different Answer (Porian et al., 2024)|§11]]) show that these choices biased the answer toward big models. The power laws themselves (Finding 1) have held up; it is the allocation (Finding 2) that changed.

---

## 10. Chinchilla (Hoffmann et al., 2022)

Kaplan's allocation rested on runs that were stopped early on a schedule built for longer training. DeepMind re-asked the allocation question with that flaw removed. (Also covered in [[RA1_Chinchilla_Cheatsheet#3. Estimating the Optimal Parameter/Training-Token Allocation|RA1]] and [[ATML_Lecture_7 - LLM Foundations#5.3 Chinchilla laws|ATML L7 §5.3]].)

### 10.1 Were Kaplan's small models undertrained

If Kaplan was right, the future was ever-larger models trained on relatively few tokens; DeepMind's own 280B-parameter **Gopher** had been built in that spirit. DeepMind's "Chinchilla paper", *Training Compute-Optimal Large Language Models* (NeurIPS 2022, arXiv:2203.15556), challenged the Kaplan laws by asking whether Kaplan's smaller models had actually been trained to convergence. They trained 400+ models from 70M to 16B parameters, systematically, and found that they had not: the smaller models were undertrained, and when trained longer they catch up.

To test this, DeepMind trained **Chinchilla** (70B parameters, 1.4T tokens) with the same compute budget as Gopher (280B parameters, 300B tokens). Chinchilla won on virtually every benchmark despite being 4× smaller. By the same reasoning GPT-3 was not using its 175B-parameter capacity: it was a 175B model that should have been about a 40B model trained 4× longer. The optimal allocation is to **scale parameters and data equally**, with a rule of thumb of about **20 tokens per parameter**.

| Model size | Optimal training tokens (~20 per parameter) |
|---|---|
| 1B params | ~20B tokens |
| 7B params | ~140B tokens |
| 70B params | ~1.4T tokens |
| 175B params | ~3.5T tokens |

So GPT-3 (175B parameters, 300B tokens) was significantly undertrained by Chinchilla standards: it should have seen about 3.5T tokens. Chinchilla (70B, 1.4T) outperformed GPT-3 despite being 2.5× smaller, because it was compute-optimally trained.

### 10.2 Three ways to find the optimum

The question: for a fixed compute budget, which pair of model size and token count gives the lowest loss? Hoffmann et al. trained more than 400 models, from about 70M to over 16B parameters, on 5B to 500B tokens, **with every run's learning-rate schedule matched to its own training length** (the fix to Kaplan's caveat). They estimated the optimum in three independent ways:

- **Approach 1.** Train each model size for four different training lengths and, at every compute budget, take whichever run had the lowest loss (the envelope of [[#9.8 Reading the compute envelope|§9.8]], done with properly scheduled runs).
- **Approach 2 (isoFLOP valleys).** Fix nine compute budgets; at each, train models of many sizes, each given exactly the tokens the budget allows, and find the size with the lowest loss ([[#10.6 IsoFLOP valleys|§10.6]]).
- **Approach 3.** Fit a formula for loss as a function of both $N$ and $D$ to all runs, and find its minimum under the budget ([[#10.5 The parametric loss and its optimum|§10.5]]).

### 10.3 Finding — grow parameters and tokens together

Model size and training tokens should grow at the same rate with compute: $N_{\text{opt}} \propto C^{a}$ and $D_{\text{opt}} \propto C^{b}$, with

| | $a$ | $b$ |
|---|---|---|
| Approach 1 | 0.50 | 0.50 |
| Approach 2 | 0.49 | 0.51 |
| Approach 3 | 0.46 | 0.54 |
| Kaplan et al. (2020) | 0.73 | 0.27 |

Here $N_{\text{opt}}, D_{\text{opt}}$ are the compute-optimal parameters and training tokens, $C$ is training compute in FLOPs, and $a, b$ say how fast each grows with compute; they add up to 1 because $N \times D$ is fixed by $C \approx 6ND$. At the optimum there are about 20 training tokens per parameter, at every budget tested. To confirm it, they trained Chinchilla (70B, 1.4T tokens) on about the same compute as Gopher (280B, 300B tokens). Chinchilla beat Gopher on almost every evaluation (for example 67.6% vs 60.0% on MMLU) and, being four times smaller, is four times cheaper to run.

### 10.4 MMLU

**MMLU** (Massive Multitask Language Understanding) is a benchmark introduced by Hendrycks et al. (2020) to measure a model's broad knowledge and reasoning.

- **Format:** about 16,000 four-option multiple-choice questions.
- **Coverage:** 57 subjects across STEM, the humanities, social sciences and professional fields such as law and medicine, from elementary to professional level.
- **Evaluation:** usually 5-shot (five solved examples in the prompt) or zero-shot; accuracy is averaged across subjects.
- **Reference points:** random guessing scores 25%; the authors estimated expert-level human performance at about 90%.
- **Its role in the Chinchilla story:** Chinchilla scored about 67.5% against 60% for Gopher, on roughly the same training compute. A model four times smaller but trained on far more data was clearly better, which showed the gains held up on downstream tasks, not just on test loss.
- **Caveats:** MMLU is now largely saturated, with frontier models near or above the human-expert estimate. A few percent of its questions have wrong or ambiguous labels, and contamination (test questions leaking into training data) is a risk because the questions are public. Harder successors such as MMLU-Pro and GPQA are now used to tell strong models apart.

> [!note] Minor source variation
> The deck gives Chinchilla's MMLU score as 67.6% on one slide and "about 67.5%" on the next; the paper reports 67.6%.

### 10.5 The parametric loss and its optimum

Approach 3 writes loss as a floor plus two separate penalties:

$$L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}} \qquad\qquad\text{(7)}$$
^eq-7

where $L(N, D)$ is the predicted loss of a model with $N$ parameters trained on $D$ tokens; $E$ is the loss floor, the loss left even with unlimited model size and data; $A/N^{\alpha}$ is the penalty for limited capacity, shrinking as the model grows; $B/D^{\beta}$ is the penalty for limited training, shrinking as the model sees more tokens; $A, B$ are the sizes of the two penalties and $\alpha, \beta$ how fast each shrinks. It is [[#^eq-4|eq. (4)]] with one decaying term per resource, added rather than combined inside a bracket as Kaplan's [[#^eq-6|eq. (6)]] does.

At a fixed budget, making the model bigger forces fewer tokens, so the capacity penalty falls while the training penalty rises. The total is lowest where moving a little compute from tokens to parameters, or back, gains exactly as much as it loses.

*(The deck states this balance in words; the derivation makes it exact.)* Substitute $D = C/(6N)$ into [[#^eq-7|eq. (7)]] and differentiate with respect to $N$. Since $dD/dN = -D/N$,

$$\frac{\partial L}{\partial N} = -\frac{\alpha A}{N^{\alpha+1}} + \frac{\beta B}{D^{\beta+1}}\cdot\frac{D}{N} = \frac{1}{N}\left(-\frac{\alpha A}{N^{\alpha}} + \frac{\beta B}{D^{\beta}}\right) = 0 \;\Longrightarrow\; \alpha\frac{A}{N^{\alpha}} = \beta\frac{B}{D^{\beta}}.$$

At the optimum the two penalties are in the fixed ratio $\beta : \alpha$. Rearranging, $D^{\beta} = \tfrac{\beta B}{\alpha A}N^{\alpha}$, so $D \propto N^{\alpha/\beta}$; combined with $ND \propto C$,

$$N_{\text{opt}} \propto C^{\beta/(\alpha+\beta)}, \qquad D_{\text{opt}} \propto C^{\alpha/(\alpha+\beta)}. \qquad\qquad\text{(8)}$$
^eq-8

The exponents add to 1 automatically, and they are equal exactly when $\alpha = \beta$. Chinchilla's published Approach 3 fit has $\alpha \approx 0.34$ and $\beta \approx 0.28$ (with $E \approx 1.69$), which gives $a = 0.28/0.62 \approx 0.46$ and $b \approx 0.54$, the Approach 3 row of [[#10.3 Finding — grow parameters and tokens together|§10.3]].

### 10.6 IsoFLOP valleys

The headline: **the largest model you can afford is not the best model you can afford.**

![[L10_isoflop_valleys.png|650]]

*IsoFLOP valleys computed from a Chinchilla-style fit. Each curve is one budget, from $10^{19}$ to $10^{23}$ FLOPs; x is model size $N$ (log), with tokens set by the budget, $D = C/(6N)$; y is predicted loss. Left of each valley: small $N$, many tokens. Right: large $N$, few tokens. The dots mark the minima, which move right as $N_{\text{opt}} \propto C^{0.51}$.*

Each curve is an **isoFLOP** profile: every point on it costs the same compute. Too small a model (left side) runs out of capacity; too large a model (right side) sees too few tokens. Take-aways: you need the right order of magnitude, not the exact optimum, because the valleys are flat near the bottom; Gopher was about 4 times too big, which is enough to matter; and the minima correspond to approximately 20 tokens per parameter.

### 10.7 Why 20 tokens per parameter

Equal growth rates only say the ratio $D/N$ stays **constant** as budgets grow; they do not say what the constant is. The number 20 is where the valley bottoms actually were:

| Model size | Compute-optimal tokens (Hoffmann's Table 3) | Tokens per parameter |
|---|---|---|
| 400M | 8.0B | 20 |
| 1B | 20.2B | 20 |
| 10B | 205.1B | 21 |
| 67B | 1.5T | 22 |
| 175B | 3.7T | 21 |
| 280B | 5.9T | 21 |

In the formula, the constant comes from the relative sizes of the two penalties ($A$ and $B$, through the balance condition of [[#10.5 The parametric loss and its optimum|§10.5]]) for their data, tokenizer and architecture: with these fitted values, the data penalty is large enough that a balanced model needs about 20 tokens for each parameter. Plugging the corrected fitted constants into the balance condition at Gopher's budget gives about 72B parameters and 1.33T tokens, almost exactly the Chinchilla model they built.

> [!note] Unclear in source — "corrected fitted constants"
> The deck does not say what corrected the constants. A likely reading is the 2024 replication by Besiroglu et al., which found that Approach 3's published constants do not reproduce the paper's own optimum and refitted them, recovering about 20 tokens per parameter. The table's 3.7T for 175B also differs slightly from the 3.5T in the rule-of-thumb table of [[#10.1 Were Kaplan's small models undertrained|§10.1]]: the first is Hoffmann's fitted optimum, the second is just $20 \times 175$B.

### 10.8 The disagreement grows under extrapolation

![[L10_chinchilla_fig1.png|600]]

*Hoffmann et al. (2022), Figure 1: optimal parameters vs training FLOPs, both log. Coloured lines are the three Chinchilla approaches; the dashed black line is Kaplan et al. (2020); stars are real models at their training budgets (Chinchilla 70B, Gopher 280B, GPT-3 175B, Megatron-Turing NLG 530B).*

The two prescriptions nearly agree near $10^{20}$ FLOPs, where most of Kaplan's experiments were run, and diverge by about ten times at frontier budgets. GPT-3, Gopher and Megatron-Turing sit near Kaplan's line, far above Chinchilla's. The disagreement only becomes large under extrapolation, which is exactly where the money is spent: two fits with different slopes agree inside the range where they were measured and drift apart outside it.

### 10.9 What it means in practice

Many large models of 2020 and 2021 were oversized for their compute. With the 20:1 rule and $C \approx 6ND$, a budget fixes both numbers:

$$C = 6N \cdot 20N = 120N^2 \;\Longrightarrow\; N = \sqrt{C/120}, \quad D = 20N.$$

At GPT-3's budget (about $3.15\times10^{23}$ FLOPs) that gives $N = \sqrt{2.6\times10^{21}} \approx 51$B parameters on about 1T tokens, instead of 175B on 300B. (The "40B trained 4× longer" of [[#10.1 Were Kaplan's small models undertrained|§10.1]] is a rougher version of the same estimate: $6 \times 40\text{B} \times 1.2\text{T} \approx 2.9\times10^{23}$ FLOPs.)

Take-aways:

- For a fixed **training** budget, grow parameters and tokens together; twice the budget means about $\sqrt2 \approx 1.4$ times each. This answers [[#8.1 What to scale|§8.1]]'s question.
- Start near 20 tokens per parameter, then confirm with a few small equal-compute runs on your own data; the ratio depends on data, tokenizer and recipe.
- A right-sized model is also cheaper to serve.
- This optimum assumes enough fresh data and counts only training cost. [[#12. When the Data Runs Out (Muennighoff et al., 2023)|§12]] drops the first assumption and [[#15. Train-Time or Test-Time — Lifetime Cost|§15]] the second.

---

## 11. Why Kaplan Got a Different Answer (Porian et al., 2024)

Two careful studies asked the same question, how to split a fixed compute budget between model size and training data, and got very different answers: Kaplan's $N \propto C^{0.73}$, which puts most of a bigger budget into parameters, and Chinchilla's $N \propto C^{0.5}$ at about 20 tokens per parameter. Which methodological difference caused the gap?

### 11.1 Hoffmann's explanation

Hoffmann et al. blamed two things. Kaplan used one schedule length for all runs and read short-training losses off the middle of long schedules, which overstates those losses and distorts the trade-off (a run read mid-schedule has not had its learning rate decayed, so its loss is higher than a run planned for that length would reach). And Kaplan's runs were mostly small, while the curvature that matters shows up at larger scales. The next paper tested this explanation and found a different main cause.

### 11.2 Three corrections, one at a time

Porian, Wortsman, Jitsev, Schmidt and Carmon (NeurIPS 2024) first reproduced Kaplan's result in their own controlled setup, getting an exponent of 0.835. Then they removed one methodological difference at a time and refitted the law after each change.

**1. Uncounted output-layer compute: 0.835 → 0.706.** Kaplan counted only non-embedding parameters, so his $C = 6ND$ leaves out the final projection from $d_{\text{model}}$ to a vocabulary of size $V$, whose training cost per token is about $6d_{\text{model}}V$ (it is a $d_{\text{model}} \times V$ matrix, so [[#^eq-5|eq. (5)]] applies to it too). In a small model this layer is a large share of the work; in a large model it is negligible. The small runs therefore used more compute than was recorded, which bent the low end of the fitted line.

**2. Warmup too long for short runs: 0.706 → 0.602.** Kaplan used a fixed 3,000-step **warmup** (ramping the learning rate up from near zero) for every run. In a short run, those steps are a large fraction of training, so the model barely trains at full learning rate. Within a small budget the largest models get the fewest steps, so they are hurt most. That made small models look best at small budgets, which steepened the line. Scaling warmup to the length of each run removes the distortion.

![[L10_warmup_share.png|480]]

*An illustrative fixed warmup of 2,000 steps takes 50% of a 4,000-step run, 20% of 10,000 steps, 5% of 40,000 and 1% of 200,000: short runs spend much of their budget warming up.*

> [!note] Minor source variation
> The chart uses an illustrative 2,000-step warmup; Kaplan's actual warmup, as the text says, was 3,000 steps. The point (a fixed warmup eats short runs) is the same.

**3. Hyperparameters not tuned per model size: 0.602 → 0.497.** A single learning rate and batch size suits some model sizes better than others. Once each size was tuned, the exponent landed on Chinchilla's 0.5.

So the main cause was not the one Hoffmann named first: compute accounting, warmup and per-size tuning together carry the exponent from Kaplan's value to Chinchilla's.

---

## 12. When the Data Runs Out (Muennighoff et al., 2023)

Chinchilla's recipe assumes every training token is fresh ([[#10.9 What it means in practice|§10.9]]). The deck promised in [[#8.3 Unique tokens vs token presentations|§8.3]] to relax that assumption; this is where it does.

### 12.1 The problem and the finding

At large budgets, 20 tokens per parameter means trillions of tokens, far more unique text than most low-resource languages have. The only way to train longer is to repeat data, and nobody knew what a repeated token is worth. Muennighoff et al., *Scaling Data-Constrained Language Models* (2023), ran 400 training runs, up to 9B parameters and 900B tokens. The unique data was held fixed while model size and number of epochs varied, and each run was compared with training on the same number of fresh tokens.

The finding: up to about **4 epochs**, repeated data is almost as good as new data; after that, returns fade quickly; and by about **40 epochs**, extra compute buys almost nothing. Why does repetition help at all? One pass does not extract everything from a text. The model has changed by the time it sees the text again, so early repeats are unfinished learning, not waste.

### 12.2 Effective tokens

Compute counts every token processed; learning counts **effective** tokens:

$$D' = U + UR_D^{\star}\left(1 - e^{-R_D/R_D^{\star}}\right), \qquad R_D^{\star} \approx 15.4 \qquad\qquad\text{(9)}$$
^eq-9

where $U$ is the number of unique tokens; $D = eU$ (epochs times unique data, [[#8.3 Unique tokens vs token presentations|§8.3]]); $D'$ is the effective data in fresh-token equivalents; $R_D = e - 1$ is the number of passes beyond the first (so epoch 2 with the same $U$ is $R_D = 1$); and $R_D^{\star} \approx 15.4$ is a fitted constant. *(Expanding on this: differentiating, each extra pass adds $U e^{-R_D/R_D^{\star}}$ effective tokens, so a repeated token is worth $e^{-R_D/R_D^{\star}}$ of a fresh one. Its value decays exponentially in the number of repeats.)* After about 15 extra passes, the next repeated token is worth $e^{-1} \approx 37\%$ of a fresh one.

### 12.3 Checking the formula at its limits

**No repetition, $R_D = 0$.** The bracket is $1 - e^{0} = 0$, so $D' = U$, as it should be.

**Little repetition, $R_D \ll R_D^{\star}$.** The Taylor series of $e^{-x}$ around 0 gives $e^{-x} \approx 1 - x$, so $e^{-R_D/R_D^{\star}} \approx 1 - R_D/R_D^{\star}$. Substituting,

$$D' \approx U + UR_D^{\star} - UR_D^{\star}\left(1 - \frac{R_D}{R_D^{\star}}\right) = U + UR_D^{\star} - UR_D^{\star} + UR_D = U(1 + R_D) = eU = D.$$

Early repeats are worth almost as much as fresh tokens: this is the "about 4 epochs" regime.

**Heavy repetition, $R_D \to \infty$.** $e^{-R_D/R_D^{\star}} \to 0$, so $D' \to U(1 + R_D^{\star}) = 16.4U$. Effective data hits a ceiling while compute keeps growing: effective data can never exceed about 16.4 times the unique data, however long you train.

### 12.4 How the value of repetition decays

With the fitted $R_D^{\star} = 15.39$:

| Epochs $e$ | $R_D$ | Effective $D'/U$ | Value retained (effective ÷ processed) $D'/D$ | Value of the next token |
|---|---|---|---|---|
| 1 | 0 | 1.00 | 100% | 100% |
| 4 | 3 | 3.73 | 93% | 82% |
| 16 | 15 | 10.6 | 66% | 38% |
| 40 | 39 | 15.2 | 38% | 8% |
| ∞ | ∞ | 16.4 | 0% | 0% |

The last column is the formula the deck does not show, $e^{-R_D/R_D^{\star}}$ from [[#12.2 Effective tokens|§12.2]]; for example $e^{-3/15.39} \approx 0.82$. The two percentage columns answer different questions: the fourth is the average value of everything processed so far, the fifth the value of the very next repeated token.

### 12.5 The same discount for parameters

The paper applies the same form to parameters: capacity beyond what the unique data can use is discounted, with a smaller fitted scale $R_N^{\star} \approx 5.3$. Intuition: excess parameters lose value even faster than excess epochs, so under data scarcity the optimum is a **smaller** model trained for **more** epochs than a naive fresh-data calculation suggests.

### 12.6 An Urdu example and the rules of thumb

Suppose a curated Urdu corpus has $U = 5$B unique tokens and Chinchilla asks for 100B. That is 20 epochs, so $R_D = 19$, and by [[#^eq-9|eq. (9)]]

$$D' = U\left[1 + R_D^{\star}\left(1 - e^{-R_D/R_D^{\star}}\right)\right] = 5\left[1 + 15.39\left(1 - e^{-19/15.39}\right)\right] \approx 5 \times 11.9 \approx 60\text{B}.$$

You pay for 100B presentations and get the learning value of about 60B effective tokens. Whether that is acceptable depends on what else the compute could buy, including a smaller model or more data from somewhere else.

Rules of thumb:

- Repetition up to about 4 epochs is safe.
- Beyond that, forecast with effective tokens, not tokens processed.
- Always record the unique corpus size, the number of passes and the deduplication policy.
- The formula never predicts loss getting worse, but real runs can overfit, so stop on validation loss.

### 12.7 Data pruning revisited (Sorscher et al., 2022)

When unique data is scarce, the other lever is to make each kept example count, the selective sampling of [[#7.3 Selective sampling|§7.3]]. Sorscher et al. (2022), *Beyond neural scaling laws*:

- **The problem.** With randomly sampled data, error falls as a power law, so each further improvement needs many times more data; most added examples repeat what the model already knows.
- **The idea.** Keep only the most informative examples: score each candidate, for example by how hard it is for the model or how far it lies from typical examples, and prune the rest.
- **The finding.** In theory, with a sufficiently good scoring metric, error can fall **exponentially** with the size of the kept dataset, not as a power law.
- **The twist.** What to keep depends on how much data you have. With **scarce** data, keep the easy, typical examples, because the model needs the basics first. With **abundant** data, keep the hard examples, because the basics are already well covered.

The [[#7.4 Finding useful examples also costs resources|§7.4]] caution still applies: the exponential is measured against kept examples, not against the cost of scoring the whole pool.

---

## 13. The Relative Worth of Pretraining (Hernandez et al., 2021)

Data scarcity has a second answer besides repetition: start from a model already pretrained on something else. This section asks how much that head start is worth.

### 13.1 Pretraining as a head start measured in data

The key is to think of pretraining as a **head start**, and to measure that head start in units of target-task data. Hernandez et al., *Scaling Laws for Transfer* (2021), ask: how much additional target data would a model need to compensate for having no pretraining? They study models pretrained on text and then fine-tuned on Python code, compared with models of the same size trained on Python from scratch. Performance is measured by loss on held-out Python code.

Illustrative numbers: two models with exactly the same number of parameters.

| | Model A | Model B |
|---|---|---|
| Starting point | Already pretrained on text | Random initialization |
| Python training data | 1 million characters | 251 million characters |
| Final Python test loss | Same | Same |

Model A reaches the same result with far less Python data, so in this comparison its pretraining was worth an extra 250 million characters of Python training data. That gives three quantities, the paper's definitions of **fine-tuning data**, **effective data transferred** and **total effective data**:

$$\underbrace{D_F}_{\text{actual fine-tuning data}} + \underbrace{D_T}_{\text{extra data equivalent of pretraining}} = \underbrace{D_E}_{\text{total data needed from scratch}}, \qquad 1\text{M} + 250\text{M} = 251\text{M}.$$

The 250M is a **performance equivalence**: we infer it by finding how much Python data the scratch model needs to match Model A. It does not tell us how many characters were in the original pretraining corpus. This is the same move as effective tokens in [[#12.2 Effective tokens|§12.2]]: express a non-fresh resource in fresh-data equivalents.

### 13.2 Absolute and relative worth

The **absolute worth** of pretraining in the example is $D_T = 250$M. Its worth **relative** to the fine-tuning dataset is

$$\frac{D_T}{D_F} = \frac{250\text{M}}{1\text{M}} = 250:$$

the head start is worth 250 times the Python data actually supplied. Including the actual data, the total effective-data multiplier is

$$\frac{D_E}{D_F} = 1 + \frac{D_T}{D_F} = 251,$$

so 1M actual characters combined with pretraining achieve what 251M characters achieve from scratch, in this example.

### 13.3 The fitted transfer law

In the low-data regime the paper fits

$$D_T = k\,D_F^{\alpha}\,N^{\beta} \qquad\qquad\text{(10)}$$
^eq-10

where $N$ is model size excluding embeddings, $k$ a fitted constant, and $\alpha, \beta$ control how transfer scales with fine-tuning data and model size. For text-to-Python transfer, $\alpha = 0.18$ and $\beta = 0.38$. (These $\alpha, \beta$ are new constants, unrelated to the Chinchilla exponents.)

### 13.4 Absolute worth up, relative worth down

Pretraining's absolute worth can increase while its relative worth decreases. Hold model size fixed, so $D_T \propto D_F^{0.18}$. Supplying 10 times more fine-tuning data multiplies $D_T$ by $10^{0.18} \approx 1.51$:

| Quantity | Initial example | With 10× more fine-tuning data |
|---|---|---|
| Actual data $D_F$ | 1M | 10M |
| Pretraining's equivalent contribution $D_T$ | 250M | 250M × 1.51 ≈ 378M |
| Total effective data $D_E$ | 251M | 378M + 10M ≈ 388M |
| Relative contribution $D_T/D_F$ | 250× | ≈ 37.8× |

Total effective data increased (251M → 388M), pretraining's absolute contribution increased (250M → 378M), and its contribution relative to the actual dataset decreased (250× → 37.8×). There is no contradiction: the head start grew, but the actual dataset grew much faster. A simple analogy: receiving PKR 250 when you have PKR 1 is an enormous relative boost; receiving PKR 378 when you have PKR 10 is a bigger absolute gift but a smaller boost relative to what you already have.

Algebraically,

$$\frac{D_T}{D_F} = \frac{kN^{0.38}D_F^{0.18}}{D_F} = \frac{kN^{0.38}}{D_F^{0.82}},$$

and the denominator grows as fine-tuning data increases, so the relative contribution falls. Model size works in the other direction: holding fine-tuning data fixed, a 10× increase in parameters gives $10^{0.38} \approx 2.40$ times as much effective transferred data under the fitted law. Bigger models transfer more.

### 13.5 The pretraining data matters

[[#^eq-10|Eq. (10)]] describes transfer under particular pretraining conditions. The paper reports two setups:

| Setup | Pretraining material | Reported amount |
|---|---|---|
| Text only | WebText2, Common Crawl, English Wikipedia and Internet Books | 24 billion characters |
| Text + other programming languages | A 50/50 mixture of natural-language text and non-Python code, including C, Java, JavaScript and C++ | The non-Python code corpus contained 240 billion characters, reported as 340 GB |

Changing the pretraining setup changed the measured law (data in characters, $N$ in non-embedding parameters, low-data regime):

| Pretraining → target | $k$ | $\alpha$ | $\beta$ |
|---|---|---|---|
| Text → Python | 19,000 | 0.18 | 0.38 |
| Text + non-Python code → Python | 210,000 | 0.096 | 0.38 |

At the same operating point, $N = 100$ million and $D_F = 1$ million Python characters,

$$D_T^{\text{text}} = 19{,}000\,(10^6)^{0.18}(10^8)^{0.38} \approx 250\text{M characters}, \qquad D_T^{\text{text+code}} = 210{,}000\,(10^6)^{0.096}(10^8)^{0.38} \approx 867\text{M characters}.$$

At this operating point, mixed pretraining provides about 3.46 times as much effective transferred data. (Its smaller $\alpha$ means that advantage shrinks as $D_F$ grows.) We cannot take these coefficients and assume they describe every pretrained model: a different source corpus, training budget or target language requires checking the relationship again.

### 13.6 Lessons

| Situation | Lesson / practical takeaway |
|---|---|
| Scarce fine-tuning data | Pretraining can provide a large head start, equivalent to having much more target data |
| Ample fine-tuning data | Pretraining's relative advantage shrinks; some small pretrained models actually performed worse than scratch models ("**ossification**": pretrained weights that resist adapting to the new task) |
| Increasing fine-tuning data | Effective transferred data can increase while its contribution relative to the actual dataset decreases |
| Increasing model size with scarce target data | Scratch models eventually become data-limited; pretrained models continued improving as size increased, although more slowly |
| Limited compute, with a pretrained model already available | Fine-tuning was more compute-efficient in the low-data regime; this comparison excludes pretraining compute |
| Limited total compute, including new pretraining | The paper does not establish that pretraining plus fine-tuning is the cheapest overall strategy |
| Choosing pretraining material | Source composition matters: text plus non-Python code transferred better to Python than text alone |
| Choosing between more target data and a larger model | Calibrate the transfer law for your setting, then compare predicted gains and actual costs; the paper's coefficients are specific to its experiments |

---

## 14. The Curse of Multilinguality

[[#13. The Relative Worth of Pretraining (Hernandez et al., 2021)|§13]] showed that one kind of data (text, code) transfers to another. Training on many languages at once is the same bet at scale: low-resource languages borrow from high-resource ones. But they also compete.

### 14.1 Mutual benefit and contention

Multilingual training lets languages benefit from shared knowledge, while also sharing limited model capacity and training time. Sharing helps until competition outweighs the benefit.

- **mBERT** (multilingual BERT) showed that a model fine-tuned for a task in one language can perform that task in another, including across scripts (Pires et al., 2019). Some learned representations transfer across languages.
- However, each language also has its own vocabulary and patterns.
- In **XLM-R**'s experiments (Conneau et al., 2020; XLM-RoBERTa, a multilingual masked language model trained on 100 languages), adding languages at fixed model capacity initially helped low-resource languages but eventually degraded performance across all of them. This is the **curse of multilinguality**. Increasing model capacity alleviated it.

Three controls decide how a low-resource language fares, and the deck takes them in turn.

### 14.2 Temperature sampling

**Temperature sampling** controls how often each language gets training time. Sampling directly from the combined corpus lets large languages dominate. Instead use

$$q_i = \frac{n_i^{\alpha}}{\sum_j n_j^{\alpha}}, \qquad 0 < \alpha < 1, \qquad\qquad\text{(11)}$$
^eq-11

where $n_i$ is language $i$'s corpus size and $q_i$ its sampling probability. Smaller $\alpha$ gives a more balanced mixture; $\alpha = 1$ is proportional sampling and $\alpha \to 0$ is uniform over languages. XLM-R used 0.3. *(It is called temperature sampling because $n_i^{\alpha} = n_i^{1/T}$ with temperature $T = 1/\alpha$, as in softmax temperature.)*

Illustration: if English has 100 times as many examples as Sindhi, proportional sampling gives a 100:1 ratio. With $\alpha = 0.5$, $100^{0.5} : 1^{0.5} = 10:1$, so Sindhi receives much more training attention. But its unique corpus is unchanged, so sustained upsampling means revisiting examples more often. Repetition can help learning, but eventually gives diminishing returns and can encourage overfitting, as [[#12. When the Data Runs Out (Muennighoff et al., 2023)|§12]] quantified.

### 14.3 Tokenizer audits

**Tokenizer audits** check how much content each language gets per token. A tokenizer may represent familiar words compactly while splitting other languages into many fragments, so equivalent translated sentences can have very different token counts. Suppose equivalent English and Urdu sentences take 10 and 30 tokens: Urdu consumes three times as many token positions for that example, so a fixed token budget or context window accommodates less equivalent content. An audit measures token counts on matched translations, word fragmentation and character coverage. Equal token budgets need not provide equal amounts of linguistic content, which also means a "tokens per parameter" rule ([[#10.7 Why 20 tokens per parameter|§10.7]]) does not transfer unchanged across languages.

### 14.4 Language-aware curation

**Language-aware curation** controls what survives into the training corpus. Cleaning rules designed for English can wrongly discard valid text in other languages, for example rules assuming English word lengths, punctuation or space-separated words. **FineWeb2** (Hugging Face's multilingual web dataset) adapts language-identification thresholds and quality filters to individual languages, identifies scripts, and deduplicates within languages across crawls; the aim is to remove noise while preserving useful language-specific text.

For a low-resource language these choices interact: **curation** determines the available useful text, **tokenization** its token cost, **sampling** how often it is seen, and **model capacity** how well its patterns can be learned alongside other languages.

### 14.5 Putting the curse into numbers (ATLAS)

**ATLAS** (Longpre et al., 2025) asks how much extra model capacity, data and compute we need to support more languages at comparable performance, turning the multilingual trade-off into a planning rule. Consider an illustrative expansion from 10 to 20 languages, applying the reported multipliers:

| Quantity | Original model | Expanded model |
|---|---|---|
| Languages | 10 | 20 |
| Parameters | 1 billion | 1.18 billion |
| Total training tokens | 100 billion | 166 billion |
| Tokens per language, with equal sampling | 10 billion | 8.3 billion |

The model grows only 18%, because many capabilities can be shared across languages. Each language receives 17% fewer tokens, and learning from other languages helps compensate. But training processes more tokens through a larger model, so by $C \approx 6ND$ ([[#^eq-5|eq. (5)]]),

$$\frac{C_{\text{new}}}{C_{\text{old}}} \approx 1.18 \times 1.66 = 1.96.$$

Twice as many languages requires about twice the training compute, even though model size increases only modestly. The paper expresses this generally as $C_{\text{new}}/C_{\text{old}} \approx r^{0.97}$ when language coverage grows by a factor $r$ (and $2^{0.97} \approx 1.96$).

### 14.6 Which languages help one another

ATLAS also measures and estimates whether training on language A helps or interferes with learning language B. Shared scripts and language families often predict stronger transfer (for example, Indonesian helps Malay), but the benefit can differ by direction. That makes language selection an empirical decision. The deck shows two language family trees for orientation.

![[L10_language_tree.jpg|600]]

*The Indo-European and Uralic language families drawn as a tree (from The Guardian, 2015): branches such as Indo-Iranian, Germanic, Romance and Slavic, with Urdu, Hindi, Punjabi and Sindhi on the Indo-Aryan branch and English on the Germanic one.*

![[L10_indo_european_tree.jpg|500]]

*Part of a family tree of Indo-European languages, coloured by branch (Celtic and Germanic in red, Iranian in yellow, Indo-Aryan in green), listing speaker counts and scripts; Urdu, Hindi, Punjabi and Sindhi sit together near the bottom of the Indo-Aryan group.*

### 14.7 Start from an existing multilingual model or from scratch

ATLAS compares two routes to improving a target language, combining [[#13. The Relative Worth of Pretraining (Hernandez et al., 2021)|§13]]'s head start with this section's capacity sharing:

| Available additional training budget | Observed pattern |
|---|---|
| Smaller budget | An existing multilingual checkpoint benefits from its head start |
| Sufficiently large budget | A same-sized model trained from scratch on the target language can eventually overtake it |

Here, fine-tuning an existing model means continued language-model training on target-language text. The comparison treats the existing checkpoint's original pretraining cost as already paid. In their 2B-parameter experiments the crossover occurred around 144–283 billion additional training tokens, depending on language; this threshold depends on the checkpoint and setup. That is, fine-tuning an existing multilingual model beats training from scratch until roughly 144B to 283B target tokens for a 2B model. This matches the ossification lesson of [[#13.6 Lessons|§13.6]]: the head start matters most when target data is scarce.

---

## 15. Train-Time or Test-Time — Lifetime Cost

Every allocation so far has counted only training compute, the second assumption [[#10.9 What it means in practice|§10.9]] flagged. A deployed model is trained once and used many times, so the next question is what happens when serving cost enters the budget.

### 15.1 Trained once, used many times

Sardana et al. (2024), *Beyond Chinchilla-Optimal*, choose a model **for the cost of its whole working life**:

$$\text{lifetime cost} = \text{training cost} + \text{number of requests} \times \text{average cost per request}.$$

Chinchilla optimizes model size and training data for a training budget; Sardana et al. add the cost of serving users. With enough expected usage, training a smaller model for longer can reach the required quality more cheaply overall: spend more once to save on every request.

Illustrative example: suppose a smaller model achieves the same quality, costs \$1 million extra to train, but saves \$0.002 per request. It recovers that extra investment after $\$1{,}000{,}000 / \$0.002 = 500$ million requests. Below that demand, the extra training may not pay off.

### 15.2 Training and inference compute

Let $N$ be the number of parameters, $D$ the total training tokens processed, and $I$ the total inference tokens processed over the model's lifetime, including inputs and outputs. For a dense Transformer, by [[#^eq-5|eq. (5)]],

$$C_{\text{train}} \approx 6ND, \qquad C_{\text{inference}} \approx 2NI, \qquad \boxed{C_{\text{life}} \approx 6ND + 2NI}. \qquad\qquad\text{(12)}$$
^eq-12

An inference token is cheaper than a training token, but there may be vastly more of them over the model's life. Dividing the two costs,

$$\frac{C_{\text{inference}}}{C_{\text{train}}} = \frac{2NI}{6ND} = \frac{I}{3D}.$$

So inference exceeds training compute when $I > 3D$, and dwarfs it only when $I$ is much larger than $3D$.

### 15.3 Same quality from a smaller model

How can a smaller model achieve the same quality? The fitted loss law of [[#^eq-7|eq. (7)]], $L(N, D) = E + A/N^{\alpha} + B/D^{\beta}$, describes a trade-off between its two adjustable penalties. Reducing $N$ increases $A/N^{\alpha}$ (less capacity), while increasing $D$ decreases $B/D^{\beta}$ (extra training compensates for some of that disadvantage). Different combinations of $N$ and $D$ can reach the same loss. The compensation has limits: an arbitrarily tiny model cannot necessarily reach the target, however long it trains, because $D \to \infty$ only removes the $B$ term and leaves $E + A/N^{\alpha}$. Sardana's optimization is therefore

$$\min_{N, D}\ 6ND + 2NI \quad \text{subject to} \quad L(N, D) \le L_{\text{target}}.$$

In words: among the model-and-data combinations that meet the quality requirement, choose the one with the lowest lifetime compute. "Quality" is represented here by pretraining loss. Note the asymmetry in the objective, $C_{\text{life}} = N(6D + 2I)$: more parameters increase the cost of both training and every future inference, while more training tokens only increase the upfront training cost. As expected inference demand grows, reducing parameters becomes increasingly valuable.

### 15.4 Worked example — 13B vs 7B

The paper reports a 7B-versus-13B illustration in which the two models are taken to achieve the same quality, with the 7B model trained longer:

| Quantity | 13B model | 7B model |
|---|---|---|
| Parameters $N$ | $13\times10^9$ | $7\times10^9$ |
| Training compute | $4.5\times10^{22}$ FLOPs | $5.2\times10^{22}$ FLOPs |
| Training tokens implied by $D = C_{\text{train}}/(6N)$ | ≈ 577B | ≈ 1.24T |
| Expected inference tokens $I$ | 2T | 2T |

The smaller model processes about 2.15 times as many training tokens, but its training compute rises by only about 16%, because each training token costs less to process. Inference costs are

$$C_{\text{inference, 7B}} = 2(7\times10^9)(2\times10^{12}) = 2.8\times10^{22}\ \text{FLOPs}, \qquad C_{\text{inference, 13B}} = 2(13\times10^9)(2\times10^{12}) = 5.2\times10^{22}\ \text{FLOPs}.$$

Adding training and inference (in units of $10^{22}$ FLOPs):

| Compute | 13B model | 7B model |
|---|---|---|
| Training | 4.5 | 5.2 |
| Inference | 5.2 | 2.8 |
| Lifetime total | 9.7 | 8.0 |

The saving is $\frac{9.7 - 8.0}{9.7}\times100 \approx \boxed{17.5\%}$. The extra $0.7\times10^{22}$ training FLOPs ($5.2 - 4.5$) buy $2.4\times10^{22}$ fewer inference FLOPs.

### 15.5 Payback time

Within this approximation we can calculate exactly when the smaller model pays back its extra training. The extra training investment is $\Delta C_{\text{train}} = (5.2 - 4.5)\times10^{22} = 7\times10^{21}$ FLOPs. The saving on each inference token is $2N_{13\text{B}} - 2N_{7\text{B}} = 2(13 - 7)\times10^9 = 12\times10^9$ FLOPs per token. Hence

$$I_{\text{break-even}} = \frac{\text{extra training compute}}{\text{inference saving per token}} = \frac{7\times10^{21}}{12\times10^{9}} \approx \boxed{583\ \text{billion tokens}}.$$

For these two candidates: below about 583B inference tokens, the 13B model uses less lifetime compute; above it, the longer-trained 7B model uses less; at 2T inference tokens, the 7B model saves about 17.5%. This is the token-level version of [[#15.1 Trained once, used many times|§15.1]]'s dollar break-even.

### 15.6 Realistic extremes and what deployed models do

The Llama 3 example shows how far this reasoning can push training duration. Meta reports over 15 trillion training tokens for Llama 3 8B. Using 15T,

$$\frac{D}{N} = \frac{15\times10^{12}}{8\times10^9} = 1{,}875 \text{ tokens per parameter},$$

roughly 94 times the 20-tokens-per-parameter benchmark. Meta explicitly reports that its smaller models continued improving with additional training and were attractive because of inference efficiency.

Can such long runs be planned? Gadre et al. (2024) predicted the validation loss of a 1.4B model trained on 900B tokens (about 640 tokens per parameter) using experiments requiring about 300 times less compute. Their contribution is evidence that scaling-law prediction can remain useful well beyond the usual training-optimal regime. When inference demand is negligible, the optimum approaches the Chinchilla training optimum; as demand increases, it shifts toward smaller models trained on more data, provided they can reach the required quality.

Many models intended for inexpensive deployment receive far more training per parameter than the 20:1 reference. With $r = D/N$ and the multiple of the 20:1 reference $= D/(20N)$:

| Model | Parameters | Reported pretraining tokens | Tokens per parameter | Multiple of 20:1 |
|---|---|---|---|---|
| Chinchilla, 2022 | 70B | 1.4T | 20 | 1× |
| LLaMA 7B, 2023 | 7B | 1T | ≈143 | ≈7.1× |
| Llama 2 7B, 2023 | 7B | 2T | ≈286 | ≈14.3× |
| Llama 3 8B, 2024 | 8B | 15T+ | ≈1,875+ | ≈94×+ |
| Qwen3-0.6B, 2025 | 0.6B | ≈36T | ≈60,000 | ≈3,000× |
| LFM2.5-350M, March 2026 | 0.35B | 28T | ≈80,000 | ≈4,000× |
| LFM3-1.7B, October 2026 | ≈1.7B | 22T | ≈13,000 | ≈650× |

**LFM** stands for Liquid Foundation Model, a model family from Liquid AI designed for efficient deployment, including on phones and laptops. These models combine convolution layers and attention; LFM3 adds a revised architecture combining short convolutions, local attention and global attention.

- **Deployment incentive:** spend additional training compute to improve a model that remains inexpensive to serve.
- **Practical limit:** stop when further training no longer gives enough useful improvement to justify its cost.
- **Interpretation:** a higher tokens-per-parameter ratio describes training intensity. It does not establish better task performance or lower lifetime cost.

Ratios use rounded reported parameter counts. Tokenizers, data quality and architectures differ, and the Liquid models use hybrid architectures, so their ratios are descriptive comparisons, not direct FLOP-equivalence comparisons. Many of these runs are also far beyond the 4-epoch comfort zone of [[#12.1 The problem and the finding|§12.1]] unless the unique data is enormous, which is why data quality and repetition enter the newer laws below.

### 15.7 Newer work and cautions

The deck lists 2026 papers that add useful points:

- **Inference is also an opportunity to improve answers.** Sardana treats inference primarily as a serving cost. Roberts et al.'s *Train-to-Test scaling* (2026, arXiv) jointly considers model size, training duration and the number of sampled answers. A smaller model can afford more attempts under the same inference budget, which can further favour substantial pretraining. Their evidence concerns repeated sampling and **pass@k**, where success means at least one of $k$ attempts is correct ([[#16.2 Repeated sampling and pass@k|§16.2]]).
- **Architecture changes which model is cheapest to serve.** Bian et al. (ICLR 2026) extend scaling laws to architectural choices such as hidden size, the allocation of parameters between attention and feed-forward layers, and grouped-query attention ([[Lecture_06_Decoding_KV_Cache_MQA_GQA_FlashAttention#5.3 Grouped-Query Attention|Lecture 6 §5.3]]). Their optimized architectures achieved up to 42% higher inference throughput than the comparison baseline under their tested conditions. So parameter count alone is insufficient for predicting serving efficiency.
- **Training-token counts need a data-quality interpretation.** *InfoLaw* (ICML 2026) models data mixture, quality and repetition alongside model size and consumed tokens. Repeatedly sampling a small high-quality corpus can eventually reduce the benefit of further training. For low-resource languages, distinguish fresh useful information from simply processing more tokens, as [[#12.2 Effective tokens|§12.2]] did.

**Cautions** before trusting a lifetime-cost plan:

- **Predict the quality users actually need.** Loss measures next-token prediction; task accuracy measures success at something, such as answering Urdu questions. Their relationship must be established empirically for the relevant task. An all-or-nothing score can hide gradual improvement, for example by giving zero credit until an entire answer is correct. Schaeffer et al. (2023) showed that metric choice explains many examined cases of apparent **emergence** (abilities that seem to appear suddenly at some scale); this does not establish that every capability jump is an illusion.
- **Evaluate the model at its deployed precision.** **Quantization** stores weights using fewer bits, saving memory but introducing approximation error (see [[ATML_Lecture_13 - Model Compression#4. Quantization|ATML L13 §4]]). Kumar et al. found that, in their experiments, longer-trained models suffered larger loss increases after quantization, so aggressive compression can erase some gains from extra training. Compare candidates after the intended 3- or 4-bit quantization.
- **Measure whether it runs acceptably on the actual device.** **QLoRA** reduces fine-tuning memory: keep the base model frozen in 4-bit form and train small adapter weights (LoRA, [[ATML_Lecture_10 - Fine-Tuning#8. LoRA|ATML L10 §8]]). **LLM in a Flash** addresses inference memory: keep weights in flash storage and selectively load them into RAM, which makes storage-transfer speed part of the problem.

---

## 16. Answer-Time Compute

Sardana showed that expected deployment costs should influence how a model is trained. But deployment compute can also **improve** answers, not just serve them. So we can ask how to divide compute between building a stronger model beforehand and giving it more opportunities to solve each question afterward. (The same family of methods is treated in depth in [[ATML_Lecture_12 - Test-Time Scaling and Multimodal Reasoning#3. Parallel Scaling — Think Wider|ATML L12 §3–4]].)

### 16.1 Spending compute per question

Once a model is trained, how much additional task performance can we obtain by spending more compute on each question? **Answer-time compute** means spending more effort on a particular question while keeping the model's weights fixed. The model can:

| Strategy | What happens |
|---|---|
| Sample | Generate several complete answers, then select one |
| Search | Explore alternative intermediate reasoning steps |
| Revise | Examine an earlier attempt and try to improve it |

### 16.2 Repeated sampling and pass@k

For independent sampling, let $p$ be the probability of answering a particular question correctly in one attempt. One attempt fails with probability $1 - p$, and all $k$ independent attempts fail with probability $(1-p)^k$. Therefore

$$P(\text{at least one correct answer}) = 1 - (1-p)^k. \qquad\qquad\text{(13)}$$
^eq-13

With $p = 0.20$:

| Attempts $k$ | Probability that at least one is correct |
|---|---|
| 1 | 20% |
| 4 | 59% |
| 8 | 83% |
| 16 | 97% |

But having a correct answer somewhere among the candidates does not mean returning it. A verifier, tests, or another selection method must identify it; randomly choosing one candidate would still give only 20% accuracy in this example. This is the coverage-versus-selection split of [[ATML_Lecture_12 - Test-Time Scaling and Multimodal Reasoning#4.1 Solution coverage vs selection accuracy|ATML L12 §4.1]]. Also, if $p = 0$, repeated sampling from the same distribution cannot help. [[#^eq-13|Eq. (13)]] applies to independent attempts with the same success probability; it does not directly describe adaptive search or revision, whose attempts depend on earlier ones.

### 16.3 How to spend the attempts (Snell et al.)

Snell et al. (2024), *Scaling LLM Test-Time Compute Optimally*, contributed a way to choose how to spend those attempts. On their math tasks, the best balance between independent attempts, revisions and search depended on question difficulty.

- Their adaptive strategy could outperform ordinary **best-of-N** (sample $N$ answers, return the one a verifier scores highest) using 64 generations rather than 256: approximately fourfold compute efficiency, not fourfold accuracy. Difficulty-estimation overhead was excluded from this comparison.
- Their smaller model could also outperform a 14× larger model answering once, when matching training-plus-inference FLOPs. Harder questions and heavier usage generally shifted the advantage toward the larger model in their experiments.

The economic intuition is simple: extra attempts must be paid for on every question. Illustratively, a model 14 times smaller that needs 20 equally long attempts uses more generation compute per question than the larger model answering once: each attempt costs $1/14$ of the larger model's, so 20 attempts cost $20/14 \approx 1.4$ times as much. With heavy usage this recurring cost becomes substantial, the same $I$-dependence as [[#15.2 Training and inference compute|§15.2]].

### 16.4 Optimizing model size, data and attempts together

Roberts et al. (2026) optimize three choices together: $N$ (model size), $D$ (training tokens) and $k$ (answer attempts). A smaller model makes each attempt cheaper; longer training improves each attempt's quality; additional attempts increase the chance of success. Their fitted laws favour substantially overtrained smaller models in the settings studied. This extends Snell's comparison by also optimizing training duration, and extends Sardana's by treating inference as a way to buy quality, not only a cost.

---

## 17. Risks of Scaling

Answer-time search ([[#16. Answer-Time Compute|§16]]) optimizes harder against whatever selects the answer. That is where scaling can go off the tracks.

### 17.1 Measurements are not behaviour

Scaling laws predict improvements in particular measurements, and those measurements do not automatically establish acceptable behaviour. This follows directly from answer-time search: a system can become better at satisfying its evaluator while moving away from what we actually want.

### 17.2 Goodhart's law and reward overoptimization

**Goodhart's law:** when a measure becomes a target, it can stop being a reliable measure of the underlying goal. We want helpful, trustworthy answers, but we optimize a measurable proxy, such as a learned reward score (the reward model of [[ATML_Lecture_8 - LLM Alignment#3. The Reward Model|ATML L8 §3]]). More training or more search can exploit weaknesses in that proxy.

Gao et al. (2023), *Scaling Laws for Reward Model Overoptimization*, quantify this: **a better reward score can hide worse outcomes.** A learned reward model acts as a judge. Increasing optimization initially improves performance but eventually exploits the judge's weaknesses: its score keeps rising while a separate reference reward peaks and falls. Larger reward models generally tolerate more optimization before this happens, a scaling law for the failure itself.

### 17.3 Fine-tuning side effects and cultural representation

- **Useful fine-tuning can weaken existing safeguards** (Qi et al., 2024). Even fine-tuning on benign instruction datasets sometimes increased harmful responses. Improving the intended task therefore does not guarantee that previously learned safety behaviour is preserved.
- **A narrow training change can have broad effects** (Betley et al., 2025). Fine-tuning models to produce insecure code caused harmful behaviour on unrelated questions in their experiments (emergent misalignment, [[ATML_Lecture_10 - Fine-Tuning#9. Emergent Misalignment|ATML L10 §9]]). Evaluate beyond the task you adapted the model for.
- **Speaking a language does not establish cultural representation** (Durmus et al., COLM 2024). Translating survey questions into another language did not reliably make model responses resemble the views of that language's speakers. Linguistic competence and representation of people's perspectives need separate evaluation, and speakers themselves hold diverse views.

The practical implication is to **re-evaluate the complete deployed system** after fine-tuning, quantization, or changes to its verifier or search strategy: check task performance, harmful responses, unnecessary refusals and cultural appropriateness.

### 17.4 Conclusion

Scaling laws help us allocate resources and predict measured performance. Deployment evaluation must establish whether the resulting system behaves acceptably for each task, each language form (different scripts and language mixes), and each community it serves.

---

## Formula Summary Sheet

| Concept | Formula | Where |
|---|---|---|
| Power-law loss | $L(x) = Ax^{-\alpha}$; $\log L = \log A - \alpha\log x$ | [[#1.2 The simplest power law\|§1.2]], [[#1.4 Why a power law is straight on log–log axes\|§1.4]], [[ATML_Lecture_7 - LLM Foundations#5.1 Power laws\|ATML L7 §5.1]] |
| Zipf's law | $f(r) \approx K/r^{\alpha}$, $\alpha \approx 1$; $f(kr)/f(r) = 1/k$ | [[#2.1 Zipf's rank–frequency law\|§2.1]], [[#2.2 The multiplication rule and the long tail\|§2.2]] |
| Zipf corpus size | $N = Vx_V H_V$, $H_V \approx \ln V + 0.5772$ | [[#2.4 How large must a corpus be\|§2.4]], [[#2.6 Worked example with 100,000 word types\|§2.6]] |
| Euler–Mascheroni constant | $\gamma = \lim_{V\to\infty}(H_V - \ln V) = 0.5772\ldots$ | [[#2.5 The harmonic number and the Euler–Mascheroni constant\|§2.5]] |
| Token share of top $m$ types | $H_m/H_V$ | [[#4.2 Zipf's law gives a Pareto-like concentration\|§4.2]] |
| Scale invariance | $L(kx)/L(x) = k^{-\alpha}$ | [[#5.1 Scale invariance and diminishing returns\|§5.1]], [[#6.1 Multiply vs add\|§6.1]] |
| Slope of power-law loss | $dL/dx = -\alpha Ax^{-\alpha-1}$ | [[#5.1 Scale invariance and diminishing returns\|§5.1]] |
| Multiplier to halve loss | $k = 2^{1/\alpha}$ | [[#5.2 How expensive the next halving is\|§5.2]], [[#7.2 Three levels of improvement\|§7.2]] |
| Loss with a floor | $L(x) = L_\infty + Ax^{-\alpha}$ | [[#5.3 Realistic curves have a floor\|§5.3]], [[#7.1 Excess loss and the escalating cost of each halving\|§7.1]] |
| Exponential decay | $L(x+\Delta)/L(x) = e^{-b\Delta}$; halve by adding $\ln 2/b$ | [[#6.1 Multiply vs add\|§6.1]], [[#7.2 Three levels of improvement\|§7.2]] |
| Exponential on semi-log axes | $\ln(Aq^{x}) = \ln A + (\ln q)x$ | [[#6.3 Any exponential is straight on a semi-log plot\|§6.3]] |
| Epochs | $D = eU$ | [[#8.3 Unique tokens vs token presentations\|§8.3]] |
| General scaling law | $L(X) = E + aX^{-\alpha}$, $\Delta L = aX^{-\alpha}$ | [[#8.4 The general form and the excess loss\|§8.4]] |
| Kaplan single-variable laws | $L(N) = (N_c/N)^{0.076}$, $L(D) = (D_c/D)^{0.095}$, $L(C) = (C_c/C)^{0.050}$ | [[#9.2 Finding 1 — loss follows power laws\|§9.2]], [[ATML_Lecture_7 - LLM Foundations#5.2 Kaplan laws\|ATML L7 §5.2]] |
| Training and inference compute | $C_{\text{train}} \approx 6ND$; $\approx 2N$ FLOPs per inference token | [[#9.3 FLOPs and the rule C ≈ 6ND\|§9.3]], [[#14.5 Putting the curse into numbers (ATLAS)\|§14.5]], [[#15.2 Training and inference compute\|§15.2]] |
| PF-day | $1\ \text{PF-day} = 10^{15} \times 86{,}400 \approx 8.64\times10^{19}$ FLOPs | [[#9.4 The PF-day\|§9.4]] |
| Kaplan combined law | $L(N,D) = [(N_c/N)^{\alpha_N/\alpha_D} + D_c/D]^{\alpha_D}$ | [[#9.5 When each single-variable law holds\|§9.5]], [[#9.9 Why the steeper data exponent does not mean spend on data\|§9.9]] |
| Kaplan allocation | $N_{\text{opt}} \propto C^{0.73}$, $D_{\text{opt}} \propto C^{0.27}$ | [[#9.7 Finding 2 — spend the budget mostly on parameters\|§9.7]] |
| Multiplier for a 10% cut | $k = 0.9^{-1/\alpha}$ (4.0 for $N$, 3.0 for $D$) | [[#9.9 Why the steeper data exponent does not mean spend on data\|§9.9]] |
| Chinchilla parametric loss | $L(N,D) = E + A/N^{\alpha} + B/D^{\beta}$ | [[#10.5 The parametric loss and its optimum\|§10.5]], [[#15.3 Same quality from a smaller model\|§15.3]], [[ATML_Lecture_7 - LLM Foundations#5.3 Chinchilla laws\|ATML L7 §5.3]] |
| Chinchilla optimum | $\alpha A/N^{\alpha} = \beta B/D^{\beta}$; $N_{\text{opt}} \propto C^{\beta/(\alpha+\beta)}$, $D_{\text{opt}} \propto C^{\alpha/(\alpha+\beta)}$ | [[#10.5 The parametric loss and its optimum\|§10.5]], [[#10.3 Finding — grow parameters and tokens together\|§10.3]] |
| 20:1 rule | $D \approx 20N$, $C = 120N^2$, $N = \sqrt{C/120}$ | [[#10.7 Why 20 tokens per parameter\|§10.7]], [[#10.9 What it means in practice\|§10.9]] |
| Output-layer compute | $\approx 6d_{\text{model}}V$ per training token | [[#11.2 Three corrections, one at a time\|§11.2]] |
| Effective tokens | $D' = U + UR_D^{\star}(1 - e^{-R_D/R_D^{\star}})$, $R_D^{\star} \approx 15.4$; ceiling $16.4U$ | [[#12.2 Effective tokens\|§12.2]], [[#12.3 Checking the formula at its limits\|§12.3]], [[#12.6 An Urdu example and the rules of thumb\|§12.6]] |
| Value of next repeated token | $e^{-R_D/R_D^{\star}}$ | [[#12.2 Effective tokens\|§12.2]], [[#12.4 How the value of repetition decays\|§12.4]] |
| Transfer bookkeeping | $D_F + D_T = D_E$; $D_E/D_F = 1 + D_T/D_F$ | [[#13.1 Pretraining as a head start measured in data\|§13.1]], [[#13.2 Absolute and relative worth\|§13.2]] |
| Transfer law | $D_T = kD_F^{\alpha}N^{\beta}$; $D_T/D_F = kN^{\beta}/D_F^{1-\alpha}$ | [[#13.3 The fitted transfer law\|§13.3]], [[#13.4 Absolute worth up, relative worth down\|§13.4]], [[#13.5 The pretraining data matters\|§13.5]] |
| Temperature sampling | $q_i = n_i^{\alpha}/\sum_j n_j^{\alpha}$, $0 < \alpha < 1$ | [[#14.2 Temperature sampling\|§14.2]] |
| ATLAS compute growth | $C_{\text{new}}/C_{\text{old}} \approx r^{0.97}$ | [[#14.5 Putting the curse into numbers (ATLAS)\|§14.5]] |
| Lifetime compute | $C_{\text{life}} \approx 6ND + 2NI = N(6D + 2I)$; $C_{\text{inf}}/C_{\text{train}} = I/(3D)$ | [[#15.2 Training and inference compute\|§15.2]], [[#15.3 Same quality from a smaller model\|§15.3]] |
| Sardana objective | $\min_{N,D} 6ND + 2NI$ s.t. $L(N,D) \le L_{\text{target}}$ | [[#15.3 Same quality from a smaller model\|§15.3]] |
| Break-even inference tokens | $I = \Delta C_{\text{train}} / (2\Delta N)$ | [[#15.5 Payback time\|§15.5]] |
| Tokens per parameter | $r = D/N$; multiple of 20:1 $= D/(20N)$ | [[#15.6 Realistic extremes and what deployed models do\|§15.6]] |
| pass@k (independent attempts) | $1 - (1-p)^k$ | [[#16.2 Repeated sampling and pass@k\|§16.2]], [[ATML_Lecture_12 - Test-Time Scaling and Multimodal Reasoning#3.1 Sampling K trajectories\|ATML L12 §3.1]] |

---

## Summary

- **Power-law loss ([[#1. Power-Law Loss|§1]], [[#5. Properties of Power-Law Loss|§5]])**
  - $L = Ax^{-\alpha}$: each fixed **multiplier** of resource gives the same **proportional** loss cut ($k^{-\alpha}$), at every scale.
  - Straight only on log–log axes, slope $-\alpha$; logging the loss alone does not straighten it.
  - Diminishing absolute returns but constant proportional returns; halving costs $2^{1/\alpha}$, so $\alpha = 0.05$ means a millionfold more compute.
  - Real curves have a floor: the power law applies to $L - L_\infty$, and each step removes a fraction of the reducible part only.
  - Uses: prediction, planning, comparing methods (lower curve vs steeper slope), diagnosing bottlenecks; still an empirical fit, not a guarantee.

- **Zipf, coverage and Pareto ([[#2. Power Laws in Language — Zipf's Law|§2]]–[[#4. The Pareto Principle and the Pareto Distribution|§4]])**
  - Word frequency $\approx K/r$: doubling the rank halves the count; a straight line of slope −1 on log–log axes.
  - A big corpus has plenty of evidence for common words and little for rare ones.
  - Corpus size for $V$ types each seen at least $x_V$ times: $Vx_VH_V$ with $H_V \approx \ln V + 0.5772$ (121M tokens for 100,000 types × 100), expected not guaranteed.
  - PRUS: greedy set cover over 18,294 tri-phonemes gives full Urdu coverage in about 10,000 tokens (708 sentences); WER 22.9% → 18.8%.
  - Pareto principle (concentration of a total) ≠ power-law loss (loss vs resource); ideal Zipf gives 20% of types ≈ 87% of tokens. A Pareto distribution shrinks the fraction above a threshold by a fixed factor when the threshold is multiplied.

- **Breaking beyond power laws ([[#6. Power-Law Decay vs Exponential Decay|§6]]–[[#7. Breaking Beyond Power-Law Scaling|§7]], [[#12.7 Data pruning revisited (Sorscher et al., 2022)|§12.7]])**
  - Power law: halve by **multiplying** resource; exponential: halve by **adding** a fixed amount, so exponentials eventually win. Exponentials are straight on semi-log axes, for any base.
  - Three levels: lower coefficient (parallel shift), larger exponent (steeper line), different form (curve bends down). Only the third leaves the power-law family.
  - Data pruning can in theory give exponential scaling in **kept** data; keep easy examples when data is scarce, hard ones when it is abundant.
  - Scoring the full pool costs resources too; claims must name the x-axis.

- **Vocabulary and the general law ([[#8. Terminology and the General Scaling Law|§8]])**
  - $N$ parameters, $D$ token presentations, $U$ unique tokens, $C$ FLOPs, $L$ held-out loss in nats/token, $E$ fitted floor; $D = eU$.
  - $L(X) = E + aX^{-\alpha}$; $E$ is a fitted asymptote, not necessarily the true entropy of language.
  - Learning curves have three regions: best-guess, power-law, irreducible error; LSTMs already showed the power-law region (Hestness 2017).

- **Kaplan et al. 2020 ([[#9. Kaplan et al. (2020)|§9]])**
  - $L$ follows power laws in $N$, $D$, $C$ (exponents 0.076, 0.095, 0.050) over 6+ orders of magnitude, when the other two are not the bottleneck.
  - $C \approx 6ND$: 2 FLOPs per parameter per token forward, 4 backward (input gradient + weight gradient); inference $\approx 2N$ per token. 1 PF-day ≈ $8.64\times10^{19}$ FLOPs.
  - Combined law $[(N_c/N)^{\alpha_N/\alpha_D} + D_c/D]^{\alpha_D}$: the larger term dominates, so growing one resource alone plateaus.
  - Allocation $N \propto C^{0.73}$, $D \propto C^{0.27}$ (10× compute → 5.4× params, 1.9× data); led to GPT-3's 1.7 tokens per parameter.
  - A steeper exponent means a resource that saturates fast, so budget flows to the slow one; the combined law alone gives $C^{0.56}$/$C^{0.44}$, sample efficiency pushes to 0.73.
  - Caveats: one fixed schedule with fixed warmup, mostly small runs. Finding 1 survived; Finding 2 did not.

- **Chinchilla and the correction ([[#10. Chinchilla (Hoffmann et al., 2022)|§10]]–[[#11. Why Kaplan Got a Different Answer (Porian et al., 2024)|§11]])**
  - Schedules matched to each run's length; three approaches all give $a \approx b \approx 0.5$ and about 20 tokens per parameter.
  - Chinchilla (70B, 1.4T) beat Gopher (280B, 300B) at equal compute, e.g. MMLU 67.6% vs 60.0%, and is 4× cheaper to serve.
  - $E + A/N^{\alpha} + B/D^{\beta}$: optimum where $\alpha A/N^{\alpha} = \beta B/D^{\beta}$, giving $N \propto C^{\beta/(\alpha+\beta)}$; equal growth rates fix the ratio, the fitted $A, B$ set it at ≈20.
  - IsoFLOP valleys are flat near the bottom: get the order of magnitude right; Gopher was ~4× too big.
  - $N = \sqrt{C/120}$: GPT-3's budget → ≈51B on ≈1T tokens. Twice the budget → ≈1.4× each.
  - Kaplan and Chinchilla agree near $10^{20}$ FLOPs and differ ~10× at frontier budgets.
  - Porian et al.: output-layer compute (0.835→0.706), fixed warmup (→0.602), untuned hyperparameters (→0.497) explain the gap.

- **Data limits and transfer ([[#12. When the Data Runs Out (Muennighoff et al., 2023)|§12]]–[[#13. The Relative Worth of Pretraining (Hernandez et al., 2021)|§13]])**
  - Repeats are nearly as good as fresh data up to ~4 epochs, fade after, near worthless by ~40.
  - Effective tokens $D' = U + UR^{\star}_D(1 - e^{-R_D/R_D^{\star}})$; a repeat is worth $e^{-R_D/R_D^{\star}}$; ceiling $16.4U$. 100B presentations of 5B Urdu tokens ≈ 60B effective.
  - Excess parameters are discounted faster ($R_N^{\star} \approx 5.3$): with scarce data, use a smaller model for more epochs; stop on validation loss.
  - Pretraining = head start measured in target data: $D_F + D_T = D_E$; $D_T = kD_F^{\alpha}N^{\beta}$ (text→Python $\alpha = 0.18$, $\beta = 0.38$).
  - More fine-tuning data raises pretraining's absolute worth but lowers its relative worth; bigger models transfer more; text + code transfers ≈3.5× better than text at the example point; ossification can make small pretrained models worse with ample data.

- **Multilinguality ([[#14. The Curse of Multilinguality|§14]])**
  - Languages share knowledge and compete for capacity; at fixed capacity, adding languages eventually hurts all (XLM-R); more capacity helps.
  - Levers: temperature sampling $q_i \propto n_i^{\alpha}$ (XLM-R $\alpha = 0.3$; upsampling means repetition), tokenizer audits (equal tokens ≠ equal content), language-aware curation (FineWeb2).
  - ATLAS: 2× languages ≈ 1.18× params, 1.66× tokens, ≈2× compute ($r^{0.97}$). Transfer is empirical and can be asymmetric (Indonesian → Malay).
  - Fine-tuning an existing multilingual 2B model beats scratch until ~144–283B target tokens.

- **Lifetime cost ([[#15. Train-Time or Test-Time — Lifetime Cost|§15]])**
  - $C_{\text{life}} \approx 6ND + 2NI$; inference exceeds training when $I > 3D$. Parameters cost twice (training and every request); data costs once.
  - 7B trained longer vs 13B: +16% training, −46% inference, −17.5% lifetime at 2T tokens; break-even ≈583B inference tokens.
  - Deployed models train far past 20:1 (Llama 3 8B ≈1,875; Qwen3-0.6B ≈60,000); ratios describe training intensity, not quality. Gadre et al. show over-trained runs are still predictable.
  - Check the metric users need (loss ≠ task accuracy; emergence is often a metric artifact), evaluate at deployed precision (long-trained models quantize worse), and on the real device.

- **Answer-time compute ([[#16. Answer-Time Compute|§16]])**
  - Sample, search or revise with fixed weights. pass@k $= 1 - (1-p)^k$ (20% → 97% at $k = 16$), but a selector must find the correct answer, and $p = 0$ cannot be fixed by sampling.
  - Snell: difficulty-adaptive allocation ≈4× more compute-efficient than best-of-N; a small model can beat a 14× larger one at matched FLOPs, but heavy usage and hard questions favour the larger model.
  - Roberts et al. optimize $N$, $D$ and $k$ together and favour overtrained small models.

- **Risks ([[#17. Risks of Scaling|§17]])**
  - Scaling laws predict a measurement, not acceptable behaviour; search optimizes the evaluator.
  - Goodhart: proxy reward keeps rising while true reward peaks and falls (Gao et al.); larger reward models delay it.
  - Benign fine-tuning can erode safety (Qi et al.); narrow fine-tuning can cause broad misalignment (Betley et al.); language competence ≠ cultural representation (Durmus et al.).
  - Re-evaluate the whole deployed system, per task, language form and community, after any fine-tune, quantization or verifier change.
