# Look Only When Needed: Certificate-Guided Visual Routing for Eficient Difusion VLMs

Donghyeon Cho<sup>1,\*</sup>, Joowon Kim<sup>1,\*</sup>, Seungho Shin<sup>1</sup> and Eunho Yang<sup>1,2,†</sup>

<sup>1</sup>KAIST, <sup>2</sup>AITRICS

\*Equal Contribution, <sup>†</sup>Corresponding Author

Difusion vision-language models (DVLMs) process their entire input at every denoising step. Since visual tokens make up 89–97% of this input on the benchmarks we study, most of the decoding computation is spent repeatedly processing the same image. Visual token pruning methods partially relieve this burden by removing part of the visual tokens, but they lose the removed information and still process the remaining image at every step. To measure how often the image is actually needed, we compare LLaDA-V’s predictions with and without the image at every decoding step and find that removing the image changes a token being unmasked at only 22.8% of steps on average and at as few as 7.9%. We therefore propose LOWN, a decoding method that decides at every step whether to process the image and how many tokens to unmask. LOWN predicts the masked tokens without the image and processes the image only when this prediction disagrees with the last image-conditioned prediction. Agreement certifies the earlier prediction for reuse, and the number of certified tokens sets how many are unmasked in that step. We further observe that the need for the image difers across benchmarks, so LOWN’s adaptive decision is needed rather than a fixed schedule. Without training or a tuned threshold, LOWN retains 99.6% of LLaDA-V’s score on ten benchmarks at 35.3% of the decoding FLOPs on average. LOWN also generalizes across LLaDA-based DVLMs, composes orthogonally with visual token pruning, and runs under key-value caching for further eficiency gains. Code and project page are available at https://hyeon9698.github.io/LOWN/

## 1. Introduction

Recent difusion language models match similarly sized autoregressive models while refining several masked positions in parallel over denoising steps (Nie et al., 2025; Ye et al., 2025a). In difusion vision-language models (DVLMs) such as LLaDA-V (You et al., 2026) and others (Yang et al., 2025a; Yu et al., 2025), bidirectional attention prevents direct use of the exact key-value caching found in autoregressive models, so standard inference processes the full multimodal sequence at every step. Visual tokens dominate this sequence, accounting for 89–97% of LLaDA-V’s input across our benchmarks (Figure 1b), while a forward without them costs only 3–10% as much. DVLM decoding therefore spends most of its compute repeatedly processing an unchanged visual input.

Existing methods reduce this cost through visual token pruning (Chang et al., 2026; Xu et al., 2026; Kim et al., 2026) (Figure 1a), intermediate-state caching (Wu et al., 2026a; Liu et al., 2026; Nguyen-Tri et al., 2026), or unmasking more tokens per step (Kim et al., 2025; Li et al., 2026b; Wu et al., 2026b; Wang et al., 2026; Lee et al., 2026). These approaches make image-conditioned forwards cheaper, reuse their computation, or run fewer decoding steps, but none asks whether the current step needs the image. We ask whether removing the image would change the model’s decision at the current decoding state.

Comparing LLaDA-V’s predictions with and without the image along the same decoding trajectory reveals substantial redundancy (Section 3.2). The two give the same top-1 token at 83.4% of the positions selected for unmasking, and 85.0% of final tokens had already appeared as top-1 predictions before unmasking. Within a trajectory, image-dependent steps tend to cluster early rather than recur periodically, while their overall share varies from 7.9% on MMBench to 43.8% on DocVQA. A fixed schedule therefore finds it hard to capture when the image is needed. These observations motivate deciding image access per input and step from predictions the model has already computed.

We propose LOWN (Look Only When Needed), a decoding method built on this redundancy. LOWN stores the top-1 predictions and confidences from the latest image-conditioned forward, which we call the FULL forward, as an anchor and tests it with a cheap TEXT-ONLY forward at each step. Agreement certifies the anchor prediction for reuse, while disagreement triggers a new FULL forward. The confidence-ordered run of agreements also determines how many tokens to unmask. LOWN requires neither training nor a task-specific threshold and can be added to visual token pruning or key-value caching because it reduces how often a FULL forward is needed (Figure 1b).

![](images/88cf5511e6e050c9f8afc36ad434cd0c0d32ea00ad53f391617d1a656dc3045a.jpg)

![](images/fbb498ae664275438c72298e131b71bcd8380343f6003ebbc7dcaebc0844c6a3.jpg)

![](images/0a37af8c44c65cbfe1780e09ad5e4abbd0cff35bcab3aaa8de14fe27067fb958.jpg)

![](images/96d6c199243261c1d05de91c5358f28d10c41f4fbf90a4bfd447d8abcd134e2c.jpg)

![](images/ad3145c7c58d8abd0d039b240b1e289f9981f3df78c7f917bfa2d2854551a5d9.jpg)  
(a) Overview of decoding methods for difusion VLMs.

![](images/66782ed27291c045490aa3dd6d15e7f9bf95ddb57d6de7fa8c56db81f86766a8.jpg)  
(b) Token budget and cost axes.

Baseline ZOO-Prune RedVTP LOWN (Ours) LOWN (Ours) + Token Pruning  
![](images/63f60d9cf7aee8ca9075e54e9f6a4d7ab46f0a8766acce8b07da10d2196064bc.jpg)  
(c) Relative cost at matched scores.

![](images/d20966b40cda8f4f9934911e1b853dc0b5bb607cf3c8e8c55b550641ac7354ef.jpg)  
(d) Relative performance at matched FLOPs.  
Figure 1: Look Only When Needed (LOWN). (a) Decoding with a difusion VLM. Token pruning reduces the visual tokens processed during decoding, whereas LOWN decides at each step whether the visual tokens need to be processed at all and otherwise runs a TEXT-ONLY forward. (b) Composition of the input, averaged over ten benchmarks. Pruning reduces how many visual tokens each forward processes, LOWN whether a forward processes them at all. (c) Decoding FLOPs speedup. Each pruning method is shown at LOWN’s score (Appendix C.3), and LOWN + Token Pruning uses RedVTP (�=50%). (d) Quality relative to LLaDA-V when every method is run at the operating point closest to LOWN’s decoding FLOPs from above. LOWN keeps 99.6% on average.

We evaluate LOWN on ten benchmarks spanning multiple-choice, document and chart understanding, mathematical reasoning, and open-ended captioning. On LLaDA-V, it retains 99.6% of the baseline score at 0.35× the decoding FLOPs (Figures 1c and 1d). Against accelerated methods given at least its FLOPs budget, it scores highest on eight of the ten benchmarks. Combined with key-value caching or visual token pruning, it further cuts the partner’s FLOPs by 1.8–2.5×, and with the key-value cache the pair decodes 5.6–7.0× faster than the model alone. The same rule transfers to LaViDa and VidLaDA.

Our contributions are as follows.

• We show through paired FULL and TEXT-ONLY forward comparisons that most DVLM decoding steps can skip image access, while the necessary steps vary across trajectories and are not well captured by fixed schedules.

• We introduce LOWN, a training-free and threshold-free method that uses prediction agreement to jointly decide when to process the image and how many tokens to unmask.

• We validate LOWN across ten benchmarks and three DVLM backbones, retaining 99.6% of LLaDA-V’s score at 0.35× its decoding FLOPs, composing orthogonally with visual token pruning, and running under key-value caching.

## 2. Related Work

Difusion language and vision-language models. Discrete difusion and masked difusion models generate sequences through iterative token recovery (Austin et al., 2021; Sahoo et al., 2024; Shi et al., 2024). This paradigm has scaled to large language models (Nie et al., 2025; Ye et al., 2025a) and extended to image and video understanding (DifusionGemma Team, 2026; You et al., 2026; Li et al., 2025; Yu et al., 2025; He et al., 2026). These models establish the DVLM setting studied here. LOWN addresses the complementary inference question of how often denoising must reprocess the visual input.

Eficient inference for difusion LMs and VLMs. Visual-token reduction shortens the visual sequence in vision transformers and autoregressive VLMs (Bolya et al., 2023; Chen et al., 2024; Zhang et al., 2025; Xing et al., 2025; Shang et al., 2025; Yang et al., 2025b). ZOO-Prune (Kim et al., 2026) combines projector-level sensitivity estimates with diversity-aware selection.

For difusion VLMs, RedVTP prunes using attention from still-masked response tokens, while $\mathrm { D ^ { 3 } T o M }$ uses previously decoded tokens to guide visual-token selection and merging (Xu et al., 2026; Chang et al., 2026). These methods determine which visual tokens a forward processes, but they lose the removed information (Endo et al., 2025; Li et al., 2026a).

Other work keeps the visual input and reduces the number of steps or the cost of a forward. Decoding schedules unmask multiple tokens per step when confidence, cross-step stability, or attention to the image supports it (Wei et al., 2026; Wu et al., 2026a; Yu et al., 2025; Wu et al., 2026b; Wang et al., 2026; Lee et al., 2026). Prophet (Li et al., 2026b) unmasks all remaining tokens once the average top-two logit gap in the answer region is large enough. KLASS (Kim et al., 2025) combines confidence with cross-step distribution stability and is our schedule baseline. Caching methods reuse intermediate states across denoising steps (Ma et al., 2025; Liu et al., 2026; Nguyen-Tri et al., 2026; Li et al., 2025), and we compose LOWN with Fast-dLLM (Wu et al., 2026a), a blockwise approximate key-value cache. Pruning and caching act inside a forward, on the tokens it processes or the states it reuses, and schedules on how many tokens a step unmasks. LOWN decides both whether a step processes the image and how many tokens it unmasks from the agreement between a TEXT-ONLY and the last image-conditioned prediction, and, keeping only predictions, it runs on top of pruning and caching.

## 3. Background and Motivation

## 3.1. Preliminaries

Starting from a fully masked response, a DVLM repeatedly predicts response tokens over $T$ denoising steps while conditioning on fixed visual and prompt tokens (Sahoo et al., 2024; Nie et al., 2025; You et al., 2026). Let � denote the visual tokens, � the prompt tokens, � the response, $x _ { t }$ the partially masked response at step $t ,$ and $M _ { t }$ its masked positions. A FULLforward over $( v , c , x _ { t } )$ gives the top-1 prediction and confidence at each $i \in M _ { t }$

$$
\hat { y } _ { t , i } ^ { \mathrm { f u l l } } = \underset { w } { \arg \operatorname* { m a x } } p _ { \theta } ( y ^ { i } = w \mid v , c , x _ { t } ) , \qquad \hat { s } _ { t , i } ^ { \mathrm { f u l l } } = \underset { w } { \arg \operatorname* { m a x } } p _ { \theta } ( y ^ { i } = w \mid v , c , x _ { t } ) .\tag{1}
$$

LLaDA-V runs a FULL forward at every step and unmasks the � most confident positions (Nie et al., 2025; You et al., 2026), where � is the response length divided by $T .$

## 3.2. When Does a DVLM Need the Image?

To measure how often the DVLM uses the image, we compare two forward passes along LLaDA-V’s standard decoding trajectory on all ten benchmarks of Section 5. The FULL forward evaluates $p _ { \theta } ( \cdot \mid v , c , x _ { t } )$ , whereas the TEXT-ONLYforward removes the visual tokens while preserving the position IDs and evaluates $p _ { \theta } ( \cdot \mid c , x _ { t } )$ At every $i \in M _ { t } ,$ , we record the top-1 predictions from both forwards, $\hat { y } _ { t , i } ^ { \mathrm { f u l l } }$ and $\hat { y } _ { t , i } ^ { \mathrm { t e x t } }$ , together with the FULL confidence $\hat { s } _ { t , i } ^ { \mathrm { f u l l } }$ . The FULL forward alone determines the decoding trajectory, and the TEXT-ONLY outputs are recorded without afecting its decisions. Appendix A gives the full protocol and per-benchmark values. The image rarely changes a decision. Of the positions the model unmasks, 83.4% receive the same top-1 token from the FULL and the TEXT-ONLY forward, averaged over the ten benchmarks, so at most steps the prediction follows from the text alone and the presence of the visual tokens rarely changes the token being unmasked. Moreover, 85.0% of the final tokens had already appeared as the top-1 prediction at the same position before the step at which they were unmasked. The baseline nevertheless continued to recompute those positions with the image until unmasking them.

![](images/92d1e7116f3d33e0e7be0e63e5e0bbe7c81a5ae17771be9d9e3568851c7cb51c.jpg)  
(a) Where and when the image changes a prediction.

![](images/d8aea762490921e8035e33a58da44b9002764cdb49edfb8de78d78d3bdb361fa.jpg)  
(b) Image-dependent steps.  
Figure 2: Visual dependence along the decoding trajectory of LLaDA-V. (a) Share of masked positions at which the FULL and TEXT-ONLY top-1 predictions disagree, over token position (front to back) and denoising step (early to late). (b) Share of steps on which the image changed at least one unmasked decision, per benchmark, with 95% confidence intervals.

The need for image access varies along the decoding trajectory. On eight of the ten benchmarks, disagreement between the two forwards is largest in the early denoising steps and decreases as the response takes shape. This pattern is not universal, as illustrated by DeCapBench in Figure 2a. The afected token positions also difer across tasks, and the share of image-dependent steps ranges from 7.9% on MMBench to 43.8% on DocVQA, 22.8% on average (Figure 2b). Variation across steps, positions, and benchmarks therefore makes a fixed image-access schedule poorly matched to the observed trajectories and motivates input- and step-adaptive image access.

A method that skips the image therefore needs a test that runs at every step, and the TEXT-ONLY forward that exposed the redundancy is a natural candidate, since it costs 3–10% of a FULL forward. The next section turns the agreement between the two forwards into that test.

## 4. Method

LOWN separates two decisions that standard DVLM decoding makes with a single FULL forward, whether the current response state needs the image and how far decoding can advance (Figure 3). It retains the last image-conditioned prediction as an anchor, tests that anchor with a TEXT-ONLY forward, and runs a FULL forward only when the test cannot certify enough tokens. The same test controls how many tokens are unmasked, allowing a successful TEXT-ONLY step to advance farther than the baseline. LOWN changes neither the model parameters nor the attention pattern.

## 4.1. Anchor and Certificate

Whenever LOWN runs a FULL forward, it retains its predictions �¯ and confidences �¯ on the positions that remain masked, and we call this pair the anchor. At step �, LOWN runs a TEXT-ONLY forward and orders the masked positions by decreasing anchor confidence �¯. A candidate token � is certified when its current TEXT-ONLY prediction agrees with the anchor, $\hat { y } _ { t , i } ^ { \mathrm { t e x t } } = \bar { y } _ { i }$ . The two predictions difer in whether they see the image and in the response state they were computed from, so their agreement approximates two conditions at once, that the image is not needed for this token and that the anchor remains valid after the state has changed, without an absolute confidence threshold. In the measurement of Section 3.2, a fresh FULL forward overturns 4.1–11.0% of the tokens certified against a one-step-old anchor, and in LOWN’s own runs it overturns 2.2–4.7% of the tokens unmasked on TEXT-ONLY steps (Section 5.3).

The longest confidence-ordered prefix of certified candidates forms $\mathcal { U } ^ { \mathrm { t e x t } }$ . The first disagreement ends the prefix, so the acceptance length $\lvert \mathcal { U } ^ { \mathrm { t e x t } } \rvert$ is set by the current input and step rather than by a constant. The scan also applies a repetition guard, detailed in Appendix B.1. The certificate is checked against a prediction that ages as the response changes, and its error grows with that age (Section 5.3). LOWN therefore replaces the entire anchor whenever it runs a FULL forward, so that every certificate is checked against the most recent FULL forward.

![](images/3cbfef179a5c84b24305013ce87234dd7a10779a3b46bfd963d939c4f6e99e87.jpg)  
Figure 3: Overview of LOWN. TEXT-ONLY agreement certifies anchor tokens for image-free unmasking, and an insuficient certified prefix triggers a FULL forward that refreshes the anchor and unmasks the positions where the two forwards disagree first.

## 4.2. Certificate-Guided Decoding

Each step runs a TEXT-ONLY forward and forms the certified prefix $\mathcal { U } ^ { \mathrm { t e x t } }$ of Section 4.1. The length of this prefix decides whether the step also needs a FULL forward. When the prefix contains at least � tokens, where � is the number of tokens the baseline unmasks per step, the step skips the FULL forward and accepts the whole prefix, unmasking every certified position to its anchor prediction rather than only � of them. This whole-prefix acceptance is what lets one step unmask more than � tokens. When the prefix is shorter than $k ,$ the anchor no longer certifies enough tokens for a step, so the step runs a FULL forward, replaces the anchor with its predictions and confidences, and unmasks � positions to their FULL predictions in disagreement-first order. Positions whose TEXT-ONLY and FULL predictions difer, $\hat { y } _ { t , i } ^ { \mathrm { t e x t } } \neq \hat { y } _ { t , i } ^ { \mathrm { f u l l } }$ , come first, and the remaining positions follow FULL confidence. These � positions form $\mathcal { U } ^ { \mathrm { f u l l } }$

The resulting state update is

$$
x _ { i }  \{ \begin{array} { l l } { \bar { y } _ { i } , } & { i \in \mathcal { U } ^ { \mathrm { t e x t } } , \quad | \mathcal { U } ^ { \mathrm { t e x t } } | \geq k , } \\ { \hat { y } _ { t , i } ^ { \mathrm { f u l l } } , } & { i \in \mathcal { U } ^ { \mathrm { f u l l } } , \quad | \mathcal { U } ^ { \mathrm { t e x t } } | < k . } \end{array}\tag{2}
$$

Both branches unmask at least � tokens, so LOWN never takes more than the baseline’s $T$ steps, while the TEXT-ONLY branch can take fewer. Appendix B.1 provides the detailed algorithm.

## 4.3. Cost Analysis

Let $\mathcal { C } ( n )$ denote the FLOPs of one forward over � tokens. The FULL and TEXT-ONLY input lengths are $N _ { F } = | v | + | c | + | x _ { t } |$ and $N _ { T } = \left| c \right| + \left| x _ { t } \right|$ . If the baseline runs � FULL forwards, its cost is $F _ { \mathrm { b a s e } } = T \cdot { \mathcal { C } } ( N _ { F } )$ LOWN runs $n _ { T }$ TEXT-ONLY forwards and adds a FULL forward on $n _ { F }$ of those steps, which gives $F _ { \mathrm { L O W N } } =$ $n _ { F } \cdot \mathcal { C } ( N _ { F } ) + n _ { T } \cdot \mathcal { C } ( N _ { T } )$ . Relative FLOPs are therefore

$$
R = { \frac { F _ { \mathrm { L O W N } } } { F _ { \mathrm { b a s e } } } } = { \frac { n _ { F } } { T } } + { \frac { n _ { T } } { T } } { \frac { { \mathcal { C } } ( N _ { T } ) } { { \mathcal { C } } ( N _ { F } ) } } = f + \sigma \rho .\tag{3}
$$

![](images/5dc48d694de7a406610232a15adc8c74c4eba0ea900e5edd9289062f3b36a4d8.jpg)  
Figure 4: Performance and eficiency on LLaDA-V. Gray dashed lines denote the baseline. Triangles and squares denote RedVTP and ZOO-Prune retention settings, respectively. Arrows connect each fixed-pruning setting to its composition with LOWN. KLASS is matched to LOWN’s FLOPs.

Here $f = n _ { F } / T$ is the frequency of FULL forwards, $\sigma = n _ { T } / T \leq 1$ is the relative number of steps, and $\rho = \mathcal { C } ( N _ { T } ) / \mathcal { C } ( N _ { F } )$ is the unit cost of a TEXT-ONLY forward. The certificate lowers $f ,$ and accepting more than � tokens can lower �. The input determines $\rho ,$ which decreases as visual tokens occupy more of the FULL input. Appendix B gives � and the per-benchmark $\rho .$

Visual token pruning changes only the FULL term. Let $\pi = \mathcal { C } ( N _ { F } ^ { \mathrm { p r u n e d } } ) / \mathcal { C } ( N _ { F } )$ be the FULL-forward cost with the pruned visual input. Pruning alone costs � relative to the baseline, while its composition with LOWN costs � $\cdot f + \sigma \rho ;$ , and both are charged the pruning overhead once. Pruning reduces the cost of each FULL forward, whereas LOWN reduces how often one is run. Section 5 evaluates this composition.

## 5. Experiments

Our experiments test whether LOWN preserves quality while reducing decoding cost. Section 5.1 defines the models, benchmarks, baselines, and evaluation protocol. Section 5.2 evaluates standalone LOWN, its composition with visual token pruning and key-value caching, and its transfer to other image and video DVLMs. Section 5.3 compares image-access schedules, isolates visual routing and adaptive acceptance, and analyzes visual information preservation and certificate reliability.

## 5.1. Implementation Details

Benchmarks and evaluation. We evaluate LOWN on LLaDA-V (You et al., 2026) and LaViDa (Li et al., 2025) across ten benchmarks, namely MathVista (Lu et al., 2024), ScienceQA (Lu et al., 2022), ChartQA (Masry et al., 2022), MME (Fu et al., 2025), MMBench (Liu et al., 2024), DocVQA (Mathew et al., 2021), GQA (Hudson and Manning, 2019), LLaVA-Bench (Liu et al., 2023), CapArena (Cheng et al., 2025), and DeCapBench (Ye et al., 2025b). We additionally evaluate VidLaDA-8B (He et al., 2026) on EgoSchema (Mangalam et al., 2023) and MLVU (Zhou et al., 2025). We evaluate quality using each benchmark’s task-specific metric and apply the same scorer to all methods within a comparison. Analytical decoding FLOPs are reported relative to each backbone’s baseline.

Table 1: Performance at matched decoding FLOPs on LLaDA-V. The upper block compares standalone methods at LOWN’s budget, and the lower block at the budget of LOWN with RedVTP (�=50%). Competitors run on a discrete hyperparameter grid, so each cell uses the cheapest setting with at least LOWN’s FLOPs (Appendix C.3). Cells show scores with FLOPs relative to LLaDA-V below. Best scores are bold, second best underlined, and lowest FLOPs bold.
<table><tr><td>Method</td><td>Math Vista</td><td>Science QA</td><td>Chart QA</td><td>MME</td><td>MM Bench</td><td>Doc VQA</td><td>Cap Arena</td><td>LLaVA- Bench</td><td>GQA</td><td>DeCap Bench</td><td>Avg.</td></tr><tr><td>LLaDA-V (Vanilla)</td><td>59.10</td><td>76.50</td><td>78.16</td><td>1719.49</td><td>70.88</td><td>83.91</td><td>-88.67</td><td>58.20</td><td>53.31</td><td>33.18</td><td>100.0%</td></tr><tr><td colspan="10">FLOPs matched to LOWN</td><td></td></tr><tr><td>RedVTP</td><td>51.70</td><td>70.70</td><td>74.32</td><td>1469.78</td><td>56.70</td><td>83.39</td><td>-91.00</td><td>56.90</td><td>50.97</td><td>33.03</td><td>93.0%</td></tr><tr><td rowspan="3">ZOO-Prune</td><td>(×0.354)</td><td>(×0.286)</td><td>(×0.641)</td><td>(×0.256)</td><td>(×0.280)</td><td>(×0.520)</td><td>(×0.411)</td><td>(×0.505)</td><td>(×0.260)</td><td>(×0.411)</td><td>(×0.392)</td></tr><tr><td>53.70</td><td>73.87</td><td>76.44</td><td>1582.41</td><td>67.27</td><td>78.73</td><td>-93.17</td><td>54.30</td><td>50.16</td><td>31.02</td><td>94.2%</td></tr><tr><td>(×0.403)</td><td>(×0.348)</td><td>(×0.641)</td><td>(×0.299)</td><td>(×0.317)</td><td>(×0.483)</td><td>(×0.417)</td><td>(×0.490)</td><td>(×0.238)</td><td>(×0.411)</td><td>(×0.405)</td></tr><tr><td>KLASS</td><td>56.70</td><td>75.01</td><td>73.64</td><td>1641.99</td><td>68.81</td><td>75.36</td><td>-96.83</td><td>42.50</td><td>49.52</td><td>30.18</td><td>91.9%</td></tr><tr><td rowspan="2">LOWN (Ours)</td><td>(×0.346)</td><td>(×0.318)</td><td>(×0.669)</td><td>(×0.316)</td><td>(×0.331)</td><td>(×0.487)</td><td>(×0.476)</td><td>(×0.536)</td><td>(×0.249)</td><td>(×0.500)</td><td>(×0.423)</td></tr><tr><td>56.00</td><td>78.83</td><td>78.36</td><td>1655.08</td><td>69.93</td><td>82.31</td><td>-85.67</td><td>57.40</td><td>53.24</td><td>34.43</td><td>99.6%</td></tr><tr><td></td><td>(×0.325)</td><td>(×0.243)</td><td>(×0.576)</td><td>(×0.231)</td><td>(×0.270)</td><td>(×0.452)</td><td>(×0.388)</td><td>(×0.454)</td><td>(×0.205)</td><td>(×0.391)</td><td>(×0.353)</td></tr><tr><td colspan="10">FLOPs matched to LOWN (Ours) ) + RedVTP (r=50%)</td><td></td></tr><tr><td>RedVTP</td><td>48.40</td><td>65.49 (×0.202)</td><td>66.00 (×0.400)</td><td>1275.08 (×0.171)</td><td>56.70</td><td>81.62</td><td>-91.33</td><td>54.40</td><td>48.05</td><td>31.16</td><td>87.8%</td></tr><tr><td rowspan="2">ZOO-Prune</td><td>(×0.268)</td><td></td><td></td><td>1487.83</td><td>(×0.280) 64.09</td><td>(×0.299)</td><td>(×0.232) -95.50</td><td>(×0.278)</td><td>(×0.175)</td><td>(×0.231)</td><td>(×0.253)</td></tr><tr><td>46.10</td><td>72.48</td><td>71.68</td><td></td><td></td><td>68.82</td><td></td><td>54.90</td><td>47.88</td><td>29.12</td><td>88.8%</td></tr><tr><td>KLASS</td><td>(×0.208) 52.70</td><td>(×0.224) 73.48</td><td>(×0.418) 68.92</td><td>(×0.214) 1584.56</td><td>(×0.205) 65.29</td><td>(×0.282) 76.65</td><td>(×0.268) -96.67</td><td>(×0.287) 42.50</td><td>(×0.152) 48.93</td><td>(×0.265) 29.02</td><td>(×0.252) 89.3%</td></tr><tr><td>+ RedVTP (r=50%)</td><td>(×0.207)</td><td>(×0.210)</td><td>(×0.428)</td><td>(×0.191)</td><td>(×0.207)</td><td>(×0.274)</td><td>(×0.247)</td><td>(×0.290)</td><td>(×0.154)</td><td>(×0.258)</td><td>(×0.247)</td></tr><tr><td>LOWN (Ours)</td><td>54.90</td><td>75.56</td><td>74.88</td><td>1709.75</td><td>69.67</td><td>82.24</td><td>-85.17</td><td>55.00</td><td>53.42</td><td>33.61</td><td>98.3%</td></tr><tr><td></td><td></td><td></td><td></td><td>(×0.166)</td><td>(×0.200)</td><td>(×0.265)</td><td>(×0.219)</td><td></td><td></td><td></td><td></td></tr><tr><td>+ RedVTP (r=50%)</td><td>(×0.195)</td><td>(×0.186)</td><td>(×0.375)</td><td></td><td></td><td></td><td></td><td>(×0.256)</td><td>(×0.150)</td><td>(×0.218)</td><td>(×0.223)</td></tr></table>

Baselines. We compare RedVTP (Xu et al., 2026), ZOO-Prune (Kim et al., 2026), KLASS (Kim et al., 2025), and D<sup>3</sup>ToM (Chang et al., 2026). All methods share prompts and response budgets within each comparison. Where multiple operating points are available, we select settings by FLOPs rather than task scores and assign each comparator at least the corresponding LOWN budget. We apply the same principle to LLaDA-V and LaViDa and evaluate Fast-dLLM (Wu et al., 2026a) separately in the caching comparison.

LOWN implementation. For standalone LOWN on LLaDA-V, we use batch size 1 and greedy decoding with temperature and classifier-free guidance scale set to zero, without key-value caching. The image is encoded once. FULL forwards reuse its visual embeddings, whereas TEXT-ONLY forwards remove those embeddings while preserving the original position IDs of the retained prompt and response tokens. Appendix C provides the complete evaluation protocol, baseline settings, and decoding details.

## 5.2. Main Results

Results on LLaDA-V. The upper block of Table 1 reports LOWN’s standalone results. LOWN retains 99.6% of the baseline score on average at 0.353× the baseline FLOPs. It scores highest among the accelerated methods on eight of ten benchmarks. LOWN uses one adaptive rule without additional tuning, whereas KLASS reports benchmark-specific thresholds. The 99.6% average includes four benchmarks on which LOWN scores above the baseline, and capping each benchmark at 100% so that these gains do not ofset losses elsewhere gives 98.6%. Visual routing alone exceeds the baseline on ScienceQA and CapArena, and disagreement-first order lifts ChartQA and DeCapBench above it, raising the score on seven of the ten benchmarks (Table 14). On a FULL step, this order first unmasks the positions where the image changed the prediction, so image-dependent tokens are not deferred.

Composition with visual token pruning. We combine LOWN with RedVTP and ZOO-Prune without retraining or changing their retention ratios. With RedVTP at 50% retention, LOWN reaches a normalized average of 98.3 at 0.223× the baseline FLOPs and scores highest on all ten benchmarks (Table 1). It further reduces FLOPs by 57.6% for RedVTP at 50% retention and 60.6% for ZOO-Prune at 33% retention (Table 11). For a visual-token pruner, further cost reductions require discarding more visual information. LOWN instead reduces how often the retained visual tokens are processed, allowing less aggressive pruning to outperform a more aggressive setting in both quality and compute. On MathVista, RedVTP at 50% retention with LOWN scores 54.9 at 0.195× FLOPs, against 51.7 at 0.354× for RedVTP at 30% retention. Figure 4 shows the broader trade-of.

![](images/7e18ddf8984226b0bc16f982002bd3f237bea2026d07e1737d1f3e35d1c7e338.jpg)  
FLOPs (relative to baseline) ↓  
Figure 5: Performance and eficiency on LaViDa. Gray dashed lines denote the baseline. Triangles, squares, and diamonds denote RedVTP, ZOO-Prune, and D<sup>3</sup>ToM, respectively. Composition markers use RedVTP at 75% and ZOO-Prune at 33% retention.

Table 2: Composition of LOWN and Fast-dLLM on LLaDA-V.
<table><tr><td rowspan="2">Method</td><td colspan="3">MathVista</td><td colspan="3">LLaVA-Bench</td><td colspan="3">DeCapBench</td></tr><tr><td>Score ↑</td><td>Time (s) ↓</td><td>FLOPs↓</td><td>Score ↑</td><td>Time (s) ↓</td><td>FLOPs↓</td><td>Score ↑</td><td>Time (s) ↓</td><td>FLOPs↓</td></tr><tr><td>LLaDA-V</td><td>59.10</td><td>7.56</td><td>1.0000</td><td>58.20</td><td>6.99</td><td>1.0000</td><td>33.18</td><td>14.64</td><td>1.0000</td></tr><tr><td>Fast-dLLM</td><td>57.30</td><td>2.14</td><td>0.0856</td><td>63.00</td><td>2.02</td><td>0.0876</td><td>35.19</td><td>4.22</td><td>0.1104</td></tr><tr><td>Fast-dLLM + LOWN</td><td>57.10</td><td>1.23 (-42.4%)</td><td>0.0457 (-46.6%)</td><td>58.00</td><td>1.25(-38.1%)</td><td>0.0489(–-44.1%)</td><td>33.35</td><td>2.10 (-50.1%)</td><td>0.0453(-58.9%)</td></tr></table>

Composition with key-value caching. LOWN also provides plug-and-play acceleration on top of key-value caching, since the cache lowers the cost of each forward and LOWN’s acceptance lowers the number of forwards. We retain Fast-dLLM’s cache block and refresh settings and combine its token selection with LOWN’s acceptance rule. Across MathVista, LLaVA-Bench, and DeCapBench, adding LOWN reduces mean generation time by 38.1– 50.1% (1.61–2.01× further speedup) and FLOPs by 44.1–58.9% relative to Fast-dLLM (Table 2, Appendix D). The combined scores are within 2.0 points of LLaDA-V, but lower than Fast-dLLM on all three tasks, with a decrease of 5.0 points on LLaVA-Bench.

Other models. LaViDa. LOWN transfers to LaViDa without changing its decision rule, retaining 99.7% of the baseline score on average at 0.174× its decoding FLOPs (Figure 5). On MMBench and GQA, LOWN scores 77.18 versus 76.95 and 56.54 versus 56.69 at only 0.088× and 0.091× the FLOPs. As on LLaDA-V, adding LOWN to RedVTP at 75% or ZOO-Prune at 33% retention cuts their FLOPs by a further 77.3% and 72.2% at quality indices of 99.5 and 99.8. These results show that adaptive image access transfers across DVLM backbones and remains complementary to visual token pruning. Appendix E reports the complete results.

Table 3: Video-understanding results of LOWN on VidLaDA-8B.
<table><tr><td></td><td colspan="2">EgoSchema</td><td colspan="2">MLVU</td></tr><tr><td>Method</td><td>Score ↑</td><td>FLOPs↓</td><td>Score ↑</td><td>FLOPs↓</td></tr><tr><td>VidLaDA</td><td>60.6</td><td>1.000</td><td>41.8</td><td>1.000</td></tr><tr><td>LOWN (Ours)</td><td>60.2</td><td>0.255 (-74.5%)</td><td>43.3</td><td>0.272 (−72.8%)</td></tr></table>

VidLaDA. We apply LOWN to VidLaDA-8B on two video benchmarks under a controlled 128-token decoding protocol (Table 3, Appendix C.4). LOWN uses 0.255× the baseline decoding FLOPs on EgoSchema and 0.272× on MLVU, with scores of 60.2 and 43.3 compared with the baseline’s 60.6 and 41.8.

![](images/821779752084e90cb7f9baf9080fe66e55c31575eac2828939f0fcaf1a96a639.jpg)  
Figure 6: Qualitative comparison with visual token pruning (RedVTP). In LOWN’s output, blue underlines mark tokens unmasked on FULL forward steps, while the remaining tokens are unmasked on TEXT-ONLY forward steps.

Wall-clock latency. Across the ten LLaDA-V benchmarks, standalone LOWN reduces generation time by 57.4% on average and up to 73.2%. Appendix D repeats Table 1 with each competitor given at least LOWN’s generation time, for standalone methods and pruning compositions.

## 5.3. Ablation and Analysis

Table 4: Comparison of LOWN with fixed and random image-access schedules on LLaDA-V. Cells report score and relative FLOPs.
<table><tr><td></td><td>MathVista</td><td>LLaVA-Bench</td><td>DeCapBench</td><td></td></tr><tr><td>LOWN (Ours)</td><td>56.00 (×0.325)</td><td>57.40 (×0.454)</td><td>34.43 (×0.391)</td><td></td></tr><tr><td>Period 2</td><td>57.00 (×0.541)</td><td>56.10 (×0.516)</td><td>33.43 (×0.522)</td><td></td></tr><tr><td>Period 4</td><td>52.00(×0.321)</td><td>49.00(×0.274)</td><td>27.79 (×0.284)</td><td></td></tr><tr><td>Period 8</td><td>48.50 (×0.211)</td><td>41.30 (×0.154)</td><td>19.09 (×0.164)</td><td></td></tr><tr><td>Random</td><td>53.40 (×0.349)</td><td>53.80 (×0.456)</td><td>30.48 (×0.389)</td><td></td></tr></table>

Image-access scheduling. Table 4 compares LOWN with periodic and random image access. Period � uses the image every � steps, while Random matches LOWN’s average FULL-forward count. Against this matched random schedule, LOWN improves task-normalized quality by 4.9–13.0% on all three tasks. Period 4 uses similar compute to LOWN on MathVista but scores 4.0 points lower.

Visual routing and adaptive acceptance. Table 5 separates two decisions of LOWN on LLaDA-V. Visual routing is the certificate’s choice of the steps that run a FULL forward, and adaptive acceptance is wholeprefix acceptance on TEXT-ONLY steps together with disagreement-first order on FULL steps (Section 4.2, Appendix F.1). Visual routing alone, which keeps the baseline’s order and unmasks � tokens per step, reduces FLOPs by 27–68%. Adaptive acceptance removes a further 21–43% of the remaining cost, scores higher on seven of the ten benchmarks, and loses at most 2.8% of normalized quality on the others. Most of the saving therefore comes from skipping FULL forwards, and adaptive acceptance adds a task-dependent reduction. This is not the same as running fewer steps, since we find that uniformly using two or four times fewer steps degrades performance across benchmarks (Table 12).

Visual information preservation. Visual token pruning permanently removes part of the visual representation before decoding, whereas LOWN skips image access only at selected steps and can consult the complete image later. The distinction is most apparent on tasks that require reading or composing fine-grained visual evidence, where pruning incurs its largest quality losses (Figure 5 and Table 13). In Figure 6, LLaDA-V and LOWN use the relevant visual evidence and answer correctly, whereas RedVTP prunes that evidence and answers fluently but incorrectly. Here LOWN concentrates FULL-forward steps on visually grounded content, while surrounding language is handled by TEXT-ONLY forwards.

Table 5: Ablation of visual routing and adaptive acceptance on LLaDA-V. Each setting reports scores and, below, relative FLOPs with the reduction from the preceding setting in parentheses.
<table><tr><td>Visual routing</td><td>Adaptive acceptance</td><td>Math Vista</td><td>Science QA</td><td>Chart QA</td><td>MME</td><td>MM Bench</td><td>Doc VQA</td><td>Cap Arena</td><td>LLaVA- Bench</td><td>GQA</td><td>DeCap Bench</td></tr><tr><td>X</td><td>X</td><td>59.10</td><td>76.50</td><td>78.16</td><td>1719.5</td><td>70.88</td><td>83.91</td><td>-88.67</td><td>58.20</td><td>53.31</td><td>33.18</td></tr><tr><td>√</td><td>x</td><td>57.60</td><td>77.39</td><td>77.76</td><td>1624.6</td><td>68.30</td><td>83.50</td><td>-86.83 0.5040 (−49.6%) 0.3554(−64.5%) 0.7330 (-26.7%) 0.3383 (-66.2%) 0.3759 (−62.4%) 0.6743 (−32.6%) 0.6833 (−31.7%) 0.7002 (−30.0%) 0.3203 (−68.0%) 0.6840 (−31.6%)</td><td>57.90</td><td>52.53</td><td>33.02</td></tr><tr><td>√</td><td>√</td><td>56.00 0.3252 (−35.5%)0.2428 (−31.7%)0.5759 (−21.4%)0.2307 (−31.8%)0.2701 (−28.1%) 0.4517 (−33.0%)0.3885 (−43.1%)0.4543 (−35.1%)0.2045 (−36.2%)0.3909 (−42.9%)</td><td>78.83</td><td>78.36</td><td>1655.1</td><td>69.93</td><td>82.31</td><td>-85.67</td><td>57.40</td><td>53.24</td><td>34.43</td></tr></table>

Certificate reliability. On the baseline decoding trajectories of Section 3.2, we compare each certified token with a fresh FULL prediction. With a one-step-old anchor, the disagreement rate is 4.1–11.0%, and it generally increases with anchor age, reaching 25.3% on DocVQA when the anchor is eight steps old. On LOWN’s own trajectories, the same rate is 2.2–4.7% for the tokens it unmasks on TEXT-ONLY steps. In practice the anchor rarely gets old, since every FULL forward replaces it and 92–97% of these tokens use an anchor from the preceding step, none older than five (Table 16). An oracle schedule that uses the image only on the steps where it changes an unmasked decision costs 0.33× the baseline against LOWN’s measured 0.35×, and, since it still runs one forward per baseline step, it is not a lower bound. Appendix F.2 gives per-benchmark results.

## 6. Conclusion

DVLMs need not process visual tokens at every denoising step. Along LLaDA-V’s decoding trajectories, most unmasked decisions agree between FULL and TEXT-ONLY forwards, but the steps requiring visual input vary across tasks, motivating adaptive rather than fixed image access. LOWN turns this agreement into a trainingand threshold-free rule for deciding when to run a FULL forward and how many tokens to unmask. Across ten benchmarks, it retains 99.6% of LLaDA-V’s score at 0.35× the decoding FLOPs. The same rule transfers unchanged to LaViDa and extends to VidLaDA under a controlled video protocol, while also composing orthogonally with visual token pruning and running under key-value caching. These results establish imageaccess scheduling as a broadly applicable axis for eficient DVLM decoding.

Limitations. LOWN reuses anchor predictions that agree with the TEXT-ONLY forward, so a reused token can difer from a fresh FULL prediction, more often as the anchor ages (Appendix F.2). It also reduces cost less on chart and document reading, where visual dependence persists and more steps need a FULL forward (Appendix E.5). The same agreement decides how many FULL forwards a response needs, so its cost cannot be set before decoding. Combining LOWN with visual token pruning, however, partly addresses this, as its retention ratio gives some control over cost.

## References

Jacob Austin, Daniel D. Johnson, Jonathan Ho, Daniel Tarlow, and Rianne van den Berg. Structured denoising difusion models in discrete state-spaces. In Advances in Neural Information Processing Systems, 2021.

Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, and Judy Hofman. Token merging: Your ViT but faster. In The Eleventh International Conference on Learning Representations, 2023.

Shuochen Chang, Xiaofeng Zhang, Qingyang Liu, and Li Niu. D<sup>3</sup>ToM: Decider-guided dynamic token merging for accelerating difusion MLLMs. In Proceedings of the AAAI Conference on Artificial Intelligence, 2026.

Liang Chen, Haozhe Zhao, Tianyu Liu, Shuai Bai, Junyang Lin, Chang Zhou, and Baobao Chang. An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large vision-language models. In European Conference on Computer Vision, 2024.

Kanzhi Cheng, Wenpo Song, Jiaxin Fan, Zheng Ma, Qiushi Sun, Fangzhi Xu, Chenyang Yan, Nuo Chen, Jianbing Zhang, and Jiajun Chen. CapArena: Benchmarking and analyzing detailed image captioning in the LLM era. In Findings of the Association for Computational Linguistics: ACL 2025, 2025.

DifusionGemma Team. DifusionGemma technical report. arXiv preprint arXiv:2608.00146, 2026.

Mark Endo, Xiaohan Wang, and Serena Yeung-Levy. Feather the throttle: Revisiting visual token pruning for vision-language model acceleration. In IEEE/CVF International Conference on Computer Vision, 2025.

Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, Yunsheng Wu, Rongrong Ji, Caifeng Shan, and Ran He. MME: A comprehensive evaluation benchmark for multimodal large language models. In Advances in Neural Information Processing Systems (Datasets and Benchmarks Track), 2025.

Zhihao He, Tieyuan Chen, Kangyu Wang, Ziran Qin, Yang Shao, Chaofan Gan, Shijie Li, Zuxuan Wu, and Weiyao Lin. VidLaDA: Bidirectional difusion large language models for eficient video understanding. arXiv preprint arXiv:2601.17868, 2026.

Drew A. Hudson and Christopher D. Manning. GQA: A new dataset for real-world visual reasoning and compositional question answering. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019.

Seo Hyun Kim, Sunwoo Hong, Hojung Jung, Youngrok Park, and Se-Young Yun. KLASS: KL-guided fast inference in masked difusion models. In Advances in Neural Information Processing Systems, 2025.

Youngeun Kim, Youjia Zhang, Huiling Liu, Aecheon Jung, Sunwoo Lee, and Sungeun Hong. ZOO-Prune: Training-free token pruning via zeroth-order gradient estimation in vision-language models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026.

Insu Lee, Wooje Park, Wonseok Shin, Jinwoo Son, and Byonghyo Shim. Visual information-guided parallel decoding for difusion multimodal large language models. arXiv preprint arXiv:2608.26580, 2026.

Duo Li, Zuhao Yang, Xiaoqin Zhang, Ling Shao, and Shijian Lu. A comprehensive study on visual token redundancy for discrete difusion-based multimodal large language models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings, 2026a.

Pengxiang Li, Yefan Zhou, Dilxat Muhtar, Lu Yin, Shilin Yan, Li Shen, Yi Liang, Soroush Vosoughi, and Shiwei Liu. Difusion language model knows the answer before it decodes. In International Conference on Learning Representations, 2026b.

Shufan Li, Konstantinos Kallidromitis, Hritik Bansal, Akash Gokul, Yusuke Kato, Kazuki Kozuka, Jason Kuen, Zhe Lin, Kai-Wei Chang, and Aditya Grover. LaViDa: A large difusion language model for multimodal understanding. In Advances in Neural Information Processing Systems, 2025.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In Advances in Neural Information Processing Systems, 2023.

Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, Kai Chen, and Dahua Lin. MMBench: Is your multi-modal model an all-around player? In European Conference on Computer Vision, 2024.

Zhiyuan Liu, Yicun Yang, Yaojie Zhang, Junjie Chen, Chang Zou, Qingyan Wei, Shaobo Wang, Yichen Zhu, and Linfeng Zhang. dLLM-Cache: Accelerating difusion large language models with adaptive caching. In International Conference on Machine Learning, 2026.

Pan Lu, Swaroop Mishra, Tony Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. In Advances in Neural Information Processing Systems, 2022.

Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. MathVista: Evaluating mathematical reasoning of foundation models in visual contexts. In International Conference on Learning Representations, 2024.

Xinyin Ma, Runpeng Yu, Gongfan Fang, and Xinchao Wang. dKV-Cache: The cache for difusion language models. In Advances in Neural Information Processing Systems, 2025.

Karttikeya Mangalam, Raiymbek Akshulakov, and Jitendra Malik. EgoSchema: A diagnostic benchmark for very long-form video language understanding. In Advances in Neural Information Processing Systems, volume 36, pages 46212–46244, 2023.

Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. ChartQA: A benchmark for question answering about charts with visual and logical reasoning. In Findings of the Associationfor Computational Linguistics: ACL 2022, 2022.

Minesh Mathew, Dimosthenis Karatzas, and C. V. Jawahar. DocVQA: A dataset for VQA on document images. In IEEE/CVF Winter Conference on Applications of Computer Vision, 2021.

Quan Nguyen-Tri, Mukul Ranjan, and Zhiqiang Shen. Attention is all you need for KV cache in difusion LLMs. In International Conference on Learning Representations, 2026.

Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. Large language difusion models. In Advances in Neural Information Processing Systems, 2025.

Subham Sekhar Sahoo, Marianne Arriola, Yair Schif, Aaron Gokaslan, Edgar Marroquin, Justin T. Chiu, Alexander Rush, and Volodymyr Kuleshov. Simple and efective masked difusion language models. In Advances in Neural Information Processing Systems, 2024.

Yuzhang Shang, Mu Cai, Bingxin Xu, Yong Jae Lee, and Yan Yan. LLaVA-PruMerge: Adaptive token reduction for eficient large multimodal models. In IEEE/CVF International Conference on Computer Vision, 2025.

Jiaxin Shi, Kehang Han, Zhe Wang, Arnaud Doucet, and Michalis K. Titsias. Simplified and generalized masked difusion for discrete data. In Advances in Neural Information Processing Systems, 2024.

Chengcheng Wang, Tingzhang Luo, Wenhao Li, Jianyuan Guo, and Chang Xu. TACG: Trajectory-aware commit gating for difusion language model decoding. arXiv preprint arXiv:2607.03236, 2026.

Qingyan Wei, Yaojie Zhang, Zhiyuan Liu, Puyu Zeng, Yuxuan Wang, Biqing Qi, Dongrui Liu, and Linfeng Zhang. Accelerating difusion large language models with SlowFast sampling: The three golden principles. In International Conference on Learning Representations, 2026.

Chengyue Wu, Hao Zhang, Shuchen Xue, Zhijian Liu, Shizhe Diao, Ligeng Zhu, Ping Luo, Song Han, and Enze Xie. Fast-dLLM: Training-free acceleration of difusion LLM by enabling KV cache and parallel decoding. In International Conference on Learning Representations, 2026a.

Yifan Wu, Yufeng Zhang, and Kenli Li. CORA-Dif: Confidence-oriented residual acceptance for eficient difusion language model inference. arXiv preprint arXiv:2608.11235, 2026b.

Long Xing, Qidong Huang, Xiaoyi Dong, Jiajie Lu, Pan Zhang, Yuhang Zang, Yuhang Cao, Conghui He, Jiaqi Wang, Feng Wu, and Dahua Lin. Conical visual concentration for eficient large vision-language models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025.

Jingqi Xu, Jingxi Lu, Chenghao Li, Sreetama Sarkar, Souvik Kundu, and Peter A. Beerel. RedVTP: Training-free acceleration of difusion vision-language models inference via masked token-guided visual token pruning. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings, 2026.

Ling Yang, Ye Tian, Bowen Li, Xinchen Zhang, Ke Shen, Yunhai Tong, and Mengdi Wang. MMaDA: Multimodal large difusion language models. In Advances in Neural Information Processing Systems, 2025a.

Senqiao Yang, Yukang Chen, Zhuotao Tian, Chengyao Wang, Jingyao Li, Bei Yu, and Jiaya Jia. VisionZip: Longer is better but not necessary in vision language models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025b.

Jiacheng Ye, Zhihui Xie, Lin Zheng, Jiahui Gao, Zirui Wu, Xin Jiang, Zhenguo Li, and Lingpeng Kong. Dream 7B: Difusion large language models. arXiv preprint arXiv:2508.15487, 2025a.

Qinghao Ye, Xianhan Zeng, Fu Li, Chunyuan Li, and Haoqi Fan. Painting with words: Elevating detailed image captioning with benchmark and alignment learning. In International Conference on Learning Representations, 2025b.

Zebin You, Shen Nie, Xiaolu Zhang, Jun Zhou, Zhiwu Lu, Ji-Rong Wen, and Chongxuan Li. LLaDA-V: Large language difusion models with visual instruction tuning. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026.

Runpeng Yu, Xinyin Ma, and Xinchao Wang. Dimple: Discrete difusion multimodal large language model with parallel decoding. arXiv preprint arXiv:2505.16990, 2025.

Yuan Zhang, Chun-Kai Fan, Junpeng Ma, Wenzhao Zheng, Tao Huang, Kuan Cheng, Denis Gudovskiy, Tomoyuki Okuno, Yohei Nakata, Kurt Keutzer, and Shanghang Zhang. SparseVLM: Visual token sparsification for eficient vision-language model inference. In International Conference on Machine Learning, 2025.

Junjie Zhou, Yan Shu, Bo Zhao, Boya Wu, Zhengyang Liang, Shitao Xiao, Minghao Qin, Xi Yang, Yongping Xiong, Bo Zhang, Tiejun Huang, and Zheng Liu. MLVU: Benchmarking multi-task long video understanding. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13691–13701, 2025.

## A. Measurement Protocol and Additional Observations

## A.1. Measurement Protocol

We decode each benchmark with LLaDA-V’s released procedure, greedy at temperature zero with zero classifierfree guidance and low-confidence remasking, using the response lengths and step counts of Appendix C. At every step we run the TEXT-ONLY forward alongside the FULL forward on the same partially masked response, with the visual tokens removed and the position IDs of all other tokens kept. Only the FULL forward decides which positions are unmasked, and the TEXT-ONLY outputs never enter the decoding state, so the trajectory is the baseline’s by construction. At every masked position and step we log the top-1 token and its softmax confidence from both forwards, together with the positions the FULL forward unmasks.

## A.2. Per-Benchmark Observations

Table 6 lists, for each benchmark, the quantities that Section 3.2 and Figure 1b summarize. The averages in the main text are unweighted means over the ten benchmarks. The FULL and TEXT-ONLY forwards agree on 71.8–92.1% of unmasked decisions, and 74.0–90.2% of final tokens were already the top-1 prediction before being unmasked. The latter is lowest on the three open-ended benchmarks, LLaVA-Bench, CapArena, and DeCapBench. Figure 7 shows the disagreement maps of Figure 2a for all ten benchmarks.

Table 6: Per-benchmark observations on LLaDA-V. Visual tokens is their mean percentage of the FULL input. Unchanged decisions is the share of unmasked positions at which the FULL and TEXT-ONLY top-1 predictions agree. Image-dependent steps is the share of steps on which the image changed at least one unmasked decision. Already top-1 is the share of final tokens that were the top-1 prediction at their position before the step that unmasked them. Brackets give 95% confidence intervals, and the last row is the unweighted mean.
<table><tr><td>Benchmark</td><td>Visual tokens (%)</td><td>Unchanged decisions</td><td>Image-dependent steps</td><td>Already top-1</td></tr><tr><td>MathVista</td><td>89.0</td><td>89.0 [88.6, 89.3]</td><td>17.9 [17.2, 18.5]</td><td>90.2 [89.9, 90.6]</td></tr><tr><td>ScienceQA</td><td>91.6</td><td>91.8 [91.0, 92.5]</td><td>8.2 [7.5, 9.0]</td><td>88.2 [87.2, 89.0]</td></tr><tr><td>ChartQA</td><td>96.1</td><td>75.5 [74.2, 76.7]</td><td>24.5 [23.3, 25.8]</td><td>88.4 [87.4, 89.3]</td></tr><tr><td>MME</td><td>97.0</td><td>89.4 [88.5, 90.3]</td><td>10.6 [9.7, 11.5]</td><td>89.6 [88.7, 90.4]</td></tr><tr><td>MMBench</td><td>91.3</td><td>92.1 [91.3, 92.8]</td><td>7.9 [7.2, 8.7]</td><td>88.6 [87.6, 89.5]</td></tr><tr><td>DocVQA</td><td>96.6</td><td>71.8 [70.9, 72.7]</td><td>43.8 [42.4, 45.2]</td><td>86.4 [85.7, 87.1]</td></tr><tr><td>CapArena</td><td>94.8</td><td>79.6 [79.1, 80.0]</td><td>32.7 [32.0, 33.5]</td><td>76.9 [76.5, 77.4]</td></tr><tr><td>LLaVA-Bench</td><td>96.2</td><td>76.3 [74.9, 77.6]</td><td>37.8 [35.6, 40.0]</td><td>74.0 [72.6, 75.3]</td></tr><tr><td>GQA</td><td>96.5</td><td>89.1 [88.2, 90.0]</td><td>10.9 [10.0, 11.8]</td><td>89.4 [88.5, 90.2]</td></tr><tr><td>DeCapBench</td><td>94.9</td><td>79.5 [79.1, 79.9]</td><td>33.4 [32.7, 34.1]</td><td>78.0 [77.6, 78.5]</td></tr><tr><td>Mean</td><td>94.4</td><td>83.4</td><td>22.8</td><td>85.0</td></tr></table>

![](images/33c809bb0d9f5716d643a6b5d7b8781eee0370827523602098550d368be44923.jpg)  
Figure 7: Where and when the image changes a prediction, all ten benchmarks. Share of masked positions at which the FULL and TEXT-ONLY top-1 predictions disagree, over token position (front to back) and denoising step (early to late), as in Figure 2a.

## A.3. Measurement on LaViDa

The measurement of Section 3.2 was repeated on LaViDa over the full evaluation sets of all ten benchmarks, with the response lengths and step counts of the LLaDA-V protocol so that the two models are measured on the same schedules. Table 7 lists the per-benchmark values. Each per-benchmark observation holds. On the eight question-answering benchmarks each measure shows less image dependence than on LLaDA-V, apart from already top-1 on MathVista, which is level, and on the two captioning benchmarks the two models are close. The image changes an unmasked decision on fewer steps, 1.3–30.6% against 7.9–43.8% on LLaDA-V, the two forwards agree on 81.6–98.7% of unmasked decisions, and 78.6–93.7% of final tokens were the top-1 prediction before they were unmasked.

Table 7: Per-benchmark observations on LaViDa, in the format of Table $^ { 6 , }$ with the response lengths and step counts of the LLaDA-V measurement and over the full evaluation sets. Brackets give 95% confidence intervals, and the last row is the unweighted mean.
<table><tr><td>Benchmark</td><td>Visual tokens (%)</td><td>Unchanged decisions</td><td>Image-dependent steps</td><td>Already top-1</td></tr><tr><td>MathVista</td><td>71.1</td><td>94.0 [93.8, 94.1]</td><td>10.4 [10.1, 10.7]</td><td>89.8 [89.6, 90.0]</td></tr><tr><td>ScienceQA</td><td>76.4</td><td>98.7 [98.6, 98.8]</td><td>1.3 [1.2, 1.4]</td><td>93.7 [93.5, 94.0]</td></tr><tr><td>ChartQA</td><td>88.8</td><td>83.6 [83.2, 84.0]</td><td>16.4 [16.0, 16.8]</td><td>92.6 [92.4, 92.9]</td></tr><tr><td>MME</td><td>87.9</td><td>96.1 [95.9, 96.3]</td><td>3.9 [3.7, 4.1]</td><td>93.7 [93.5, 94.0]</td></tr><tr><td>MMBench</td><td>78.4</td><td>96.8 [96.6, 96.9]</td><td>3.2 [3.1, 3.4]</td><td>93.7 [93.5, 93.9]</td></tr><tr><td>DocVQA</td><td>88.3</td><td>89.4 [89.3, 89.6]</td><td>15.2 [14.9, 15.4]</td><td>91.6 [91.4, 91.7]</td></tr><tr><td>CapArena</td><td>81.1</td><td>82.0 [81.7, 82.3]</td><td>29.8 [29.3, 30.2]</td><td>78.6 [78.3, 78.9]</td></tr><tr><td>LLaVA-Bench</td><td>86.3</td><td>91.3 [90.4, 92.1]</td><td>13.6 [12.2, 15.1]</td><td>88.1 [87.1, 89.0]</td></tr><tr><td>GQA</td><td>88.8</td><td>95.8 [95.7, 95.9]</td><td>4.2 [4.1, 4.3]</td><td>93.1 [92.9, 93.2]</td></tr><tr><td>DeCapBench</td><td>80.7</td><td>81.6 [81.2, 81.9]</td><td>30.6 [30.0, 31.2]</td><td>78.6 [78.2, 79.0]</td></tr><tr><td>Mean</td><td>82.8</td><td>90.9</td><td>12.9</td><td>89.3</td></tr></table>

## B. Method Details and Cost Accounting

This appendix gives the complete decoding procedure, restates the cost model of Section 4.3 and derives its forward cost �(�), and writes visual token pruning in the same form as Equation 3.

## B.1. Detailed Decoding Algorithm

Each response starts fully masked, with its length and baseline unmasking schedule set by the benchmark protocol. The first iteration runs TEXT-ONLY followed by FULL on the same response, establishing the anchor and unmasking � positions. Confidence is the softmax probability of the top-1 token. The repetition guard ends the certified prefix before a candidate whose token ID matches either immediate neighbor that is already unmasked or was accepted earlier in the same scan. The guard is needed because the positions accepted on one TEXT-ONLY step are predicted independently of one another, so two adjacent positions can receive the same token, and without the guard such pairs would be unmasked together and appear as repeated words in the response. Algorithm 1 gives the full procedure.

## B.2. Cost of One Forward

Section 4.3 writes the cost of LOWN relative to the baseline as $R = f + \sigma \rho$ (Equation 3). The baseline runs � FULL forwards over $N _ { F } = | v | + | c | + | x _ { t } |$ tokens. LOWN runs � TEXT-ONLY forwards over $N _ { T } = \left| c \right| + \left| x _ { t } \right|$ tokens and adds a FULL forward on �<sub>�</sub> of those steps, so $f = n _ { F } / T$ is the frequency of FULL forwards, $\sigma = n _ { T } / T$ the relative number of steps, and $\rho = \mathcal { C } ( N _ { T } ) / \mathcal { C } ( N _ { F } )$ the cost of a TEXT-ONLY forward relative to a FULL one, where �(�) is the FLOPs of one forward over � tokens. A FULL step is charged both forwards. This section gives �(�).

We count matrix multiplications only, with one multiply-add as two FLOPs. For a model with � layers, hidden width $d ,$ FFN width $m ,$ and vocabulary � (for LLaDA-V, �=32, �=4096, �=12288, $\scriptstyle | \mathcal { V } | = 1 2 6 , 3 4 9 )$ , one

Algorithm 1 LOWN decoding   
Require: visual tokens �, prompt tokens �, tokens unmasked per step �   
Ensure: response �   
1: � ← fully masked response; $\hat { y } ^ { \mathrm { t e x t } } \gets$ arg max $p _ { \theta } ( \cdot \mid c , x )$ ◁ draft for the first forward   
2: while � has masked positions do   
3: $\bar { y } \gets$ arg max $p _ { \theta } ( \cdot \mid v , c , x ) , \bar { s }  p _ { \theta } ( \bar { y } \mid v , c , x )$ ◁ FULL forward: update anchor   
4: $\dot { \mathcal { U } } ^ { \mathrm { f u l l } } \gets$ top-� masked positions by ${ \bar { s } } ,$ those with $\hat { y } _ { i } ^ { \mathrm { t e x t } } \neq \bar { y } _ { i }$ first   
5: $x _ { i } \gets \bar { y } _ { i }$ for $i \in \mathcal { U } ^ { \mathrm { f u l l } }$   
6: repeat   
7: $\hat { y } ^ { \mathrm { t e x t } }$ ← arg max $p _ { \theta } ( \cdot \mid c , x )$ ◁ TEXT-ONLY forward   
8: $M \gets$ masked positions in decreasing $\bar { s }$   
9: �<sup>text</sup> ← longest prefix of � with $\hat { y } _ { i } ^ { \mathrm { t e x t } } = \bar { y } _ { i }$ ◁ certificate   
10: if $| \mathcal { U } ^ { \mathrm { t e x t } } | \geq k$ then $x _ { i } \gets \bar { y } _ { i }$ for $i \in \mathcal { U } ^ { \mathrm { t e x t } }$ ◁ anchor kept   
11: if $x$ has no masked positions then break   
12: until $| \mathcal { U } ^ { \mathrm { t e x t } } | < k$ ◁ a certificate fails   
13: end while   
14: return �

layer over � tokens costs

$$
\begin{array} { r } { \mathcal { C } _ { \mathrm { l a y e r } } ( n ) \ = \ \underbrace { 8 n d ^ { 2 } + 4 n ^ { 2 } d } _ { \mathrm { a t t e n t i o n } } + \underbrace { 6 n d m } _ { \mathrm { F F N } } , \qquad \mathcal { C } ( n ) \ = \ L \cdot \mathcal { C } _ { \mathrm { l a y e r } } ( n ) + 2 n d | \mathcal { V } | , } \end{array}\tag{4}
$$

where $8 n d ^ { 2 }$ is the Q/K/V/O projections, $4 n ^ { 2 } d$ is $Q K ^ { \top }$ and $A V$ and is quadratic in length, 6��� is the three matrices of the gated FFN, and the last term of $\mathcal { C } ( n )$ is the LM head. The quadratic term is kept because it is what makes $\rho$ smaller than the length ratio $N _ { T } / N _ { F }$ , since removing � shrinks $n$ by roughly $9 { - } 3 3 \times$ and the quadratic term by roughly the square of that. The count excludes the vision encoder, which runs once per sample and would add 0.16–1.89% of the baseline decoding cost on the ten benchmarks, the same for every method on a benchmark, and non-matmul operations such as softmax and normalization. Table 8 reports the mean $\rho$ per benchmark, and Appendix D reports measured generation time.

Table 8: Mean TEXT-ONLY/FULL forward-cost ratio on LLaDA-V. For each sample, we compute $\rho _ { i } = \mathcal { C } ( p _ { i } + o _ { i } ) / \mathcal { C } ( p _ { i } + v _ { i } + o _ { i } )$ using its prompt length $p _ { i } ,$ visual-token count $v _ { i } ,$ and configured response length $o _ { i } .$ . Each row averages these ratios over the benchmark’s samples, and � is the analytical forward cost in Equation 4.
<table><tr><td>Benchmark</td><td>Samples</td><td>Mean  $\rho$  (%)</td></tr><tr><td>MathVista</td><td>1,000</td><td>10.07</td></tr><tr><td>ScienceQA</td><td>2,017</td><td>7.92</td></tr><tr><td>ChartQA</td><td>2,500</td><td>2.54</td></tr><tr><td>MME</td><td>2,374</td><td>3.15</td></tr><tr><td>MMBench</td><td>4,329</td><td>7.04</td></tr><tr><td>DocVQA</td><td>5,349</td><td>3.02</td></tr><tr><td>CapArena</td><td>600</td><td>4.63</td></tr><tr><td>LLaVA-Bench</td><td>60</td><td>3.35</td></tr><tr><td>GQA</td><td>12,578</td><td>3.20</td></tr><tr><td>DeCapBench</td><td>400</td><td>4.49</td></tr></table>

## B.3. Other Methods in the Same Form

Relative to the unpruned baseline, visual token pruning costs $\pi ,$ LOWN costs $f + \sigma \rho ,$ and the two together cost $\pi f + \sigma \rho .$ . Pruning fixes its budget � before decoding starts, whereas LOWN decides $f$ and � during decoding, per input and per step. Key-value caching instead lowers the cost of every forward that reuses the cache, and LOWN’s acceptance lowers the number of forwards, which is the saving measured in Table 2.

## C. Experimental Setup

## C.1. Benchmarks and Settings

Table 8 lists the evaluated sample counts for LLaDA-V. We use greedy decoding with temperature zero and classifier-free guidance zero, with low-confidence remasking. The configured response length and baseline denoising steps are 100/50 for MathVista, 32/16 for DocVQA, 64/32 for LLaVA-Bench, 128/64 for CapArena and DeCapBench, and 16/16 for ScienceQA, ChartQA, MME, MMBench, and GQA. ChartQA generation ends at the first newline. The baseline unmasks � tokens per step, the response length divided by the number of steps, so � = 2 on MathVista, DocVQA, LLaVA-Bench, CapArena, and DeCapBench and � = 1 on the others. MathVista uses the model’s reasoning prompt, and the other tasks use the non-reasoning setting. These response budgets are held fixed within each comparison. For standalone LLaDA-V, we use the authors’ RedVTP code and adapt the released ZOO-Prune and KLASS implementations. The released ZOO-Prune code targets autoregressive LLaVA models. We keep its token scoring and selection, select the visual tokens once before decoding, and use the retained tokens at every denoising step. Reported FLOPs use the tokens actually retained and include the scoring cost. Composition comparisons target the corresponding LOWN budget and include existing of-grid settings. Table 9 specifies the retention and KLASS thresholds selected for Table 1. The Avg. column in Table 1 is the unweighted mean of 100×score/baseline score, except that CapArena uses 100+score−baseline score.

## C.2. Judges

MME reports perception plus cognition for LLaDA-V and perception alone for LaViDa, while CapArena reports a win–loss score. We score ScienceQA, ChartQA, MME, DocVQA, and GQA with task-specific rules. For LLaDA-V ScienceQA, we extract answers with the deterministic multiple-choice parser from lmms-eval v0.7.1. MathVista extracts final answers with GPT-4o-2024-08-06 before scoring, and CapArena is judged with the same model. MMBench uses rule matching followed by GPT-3.5-turbo-0125 when needed, scoring unmatched answers as incorrect, and LLaVA-Bench is judged with GPT-4o-2024-11-20. DeCapBench uses Qwen3.5-35B-A3B for its fact-based evaluation. All methods within each reported comparison use the same scorer.

## C.3. Baseline Settings

Each competitor exposes a budget hyperparameter, the retention ratio � for RedVTP and ZOO-Prune and the confidence and KL thresholds (�, �) for KLASS. We run each on a fixed grid whose ends keep the method within its working range, a 10% grid of � for RedVTP and ZOO-Prune on LLaDA-V together with existing of-grid settings, and a grid of threshold pairs for KLASS. Because the grid is discrete, a competitor’s FLOPs cannot be set equal to LOWN’s. For every benchmark we therefore select the cheapest grid setting whose FLOPs are at least those of the LOWN row it is compared with, and scores play no part in the selection. Every competitor cell in Tables 1 and 13 consequently spends at least as much compute as LOWN, so a competitor that scores below LOWN does so at equal or higher cost. Table 9 lists the selected settings on LLaDA-V, and Table 13 shows the selected retention under each LaViDa cell.

Figure 1c matches scores instead of FLOPs. For each benchmark the target is LOWN’s score, and each pruning method is drawn at its cheapest measured setting, up to 75% retention, whose score is at least the target. When no setting up to 75% reaches the target, the 75% setting is drawn, so its bar is an upper bound on the speedup the method would have at LOWN’s score. Only measured settings are used, without interpolation.

## C.4. Video Evaluation

We evaluate the 500-question EgoSchema subset and the 502-question MLVU test set. We use VidLaDA-8B with 32 frames per video, sampled at one frame per second and uniformly subsampled when needed. Each frame yields 210 visual tokens after spatial pooling. We adopt a controlled decoding protocol with a 128-token response, 128 baseline difusion steps, and block length 32, following the eficiency-ablation setup of He et al. (2026). Decoding is greedy, with zero classifier-free guidance and low-confidence remasking. The baseline’s parallel decoding is disabled, and LOWN uses adaptive acceptance within each block. We ask the model to reason inside <think> and </think> before answering with an option letter. This protocol difers from VidLaDA’s released 64-frame, two-token evaluation.

Table 9: Settings behind the competitor cells of Table 1 (FLOPs matched) and Table 10 (time matched). For RedVTP, � is the visual-token retention percentage. For ZOO-Prune, the released code sets the budget to �% of a fixed 2,880 tokens rather than of each image’s visual tokens, so the fraction actually retained varies with the image. For KLASS, (�, �) denotes its confidence and KL thresholds. The second and fourth blocks use the budget of LOWN composed with RedVTP at � = 50%.
<table><tr><td></td><td></td><td>MathVista ScienceQA</td><td>ChartQA</td><td>MME</td><td>MMBench</td><td>DocVQA</td><td>CapArena LLaVA-B.</td><td></td><td>GQA</td><td>DeCapB.</td></tr><tr><td colspan="9">FLOPs matched to LOWN</td></tr><tr><td>RedVTP r</td><td>30</td><td>20</td><td>60</td><td>20</td><td>20</td><td>50</td><td>40</td><td>50</td><td>20</td><td>40</td></tr><tr><td>ZOO-Prune r</td><td>30</td><td>20</td><td>80</td><td>30</td><td>20</td><td>60</td><td>50</td><td>60</td><td>20</td><td>50</td></tr><tr><td>KLASS (τ, κ)</td><td>(0.2, 0.2)</td><td>(0.2, 1)</td><td>(0.2, 1)</td><td>(0.2, 0.3)</td><td>(0.2, 0.4)</td><td>(0.2, 0.1)</td><td>(0.2, 0.5)</td><td>(0.2, 1)</td><td>(0.2, 1)</td><td>(0.2, 0.4)</td></tr><tr><td colspan="9">FLOPs matched to LOWN + RedVTP (r = 50%)</td></tr><tr><td>RedVTP r</td><td></td><td>10</td><td></td><td></td><td>20</td><td>25</td><td>20</td><td>25</td><td>10</td><td>20</td></tr><tr><td>ZOO-Prune r</td><td>20 11.12</td><td>11.12</td><td>30 50</td><td>10 20</td><td>11.12</td><td>33.34</td><td>30</td><td>33.34</td><td>11.12</td><td>30</td></tr><tr><td>KLASS + RedVTP (τ, κ)</td><td>(0.2, 0.1)</td><td>(0.2, 0.4)</td><td>(0.1, 0.2)</td><td>(0.2, 0.3)</td><td>(0.2, 0.4)</td><td>(0.2, 0.1)</td><td>(0.2, 0.5)</td><td>(0.2, 1)</td><td>(0.2, 1)</td><td>(0.2, 0.4)</td></tr><tr><td colspan="9"></td></tr><tr><td>Time matched to LOWN</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>RedVTP r ZOO-Prune r</td><td>30 30</td><td>10 11.12</td><td>50</td><td>10</td><td>10</td><td>50 60</td><td>50</td><td>60</td><td>10</td><td>50 60</td></tr><tr><td>KLASS (τ, κ)</td><td>(0.3, 0.1)</td><td>(0.2, 0.4)</td><td>75 (0.2, 0.5)</td><td>30 (0.2, 0.3)</td><td>22.23 (0.2, 0.1)</td><td>(0.3, 0.1)</td><td>60 (0.2, 0.5)</td><td>70 (0.2, 0.5)</td><td>22.23 (0.2, 0.1)</td><td>(0.2, 0.4)</td></tr><tr><td colspan="9">Time matched to LOWN + RedVTP (r = 50%)</td></tr><tr><td>RedVTP r</td><td>10</td><td>10</td><td>40</td><td>10</td><td>10</td><td>30</td><td>25</td><td>40</td><td>10</td><td>25</td></tr><tr><td>ZOO-Prune r</td><td>11.12</td><td>11.12</td><td>70</td><td>30</td><td>22.23</td><td>50</td><td>33.34</td><td>40</td><td>22.23</td><td>33.34</td></tr><tr><td>KLASS + RedVTP (τ, κ)</td><td>(0.4, 0.1)</td><td>(0.5, 0.1)</td><td>(0.2, 0.1)</td><td>(0.3, 0.1)</td><td>(0.5, 0.1)</td><td>(0.5, 0.1)</td><td>(0.2, 0.1)</td><td>(0.2, 0.1)</td><td>(0.3, 0.1)</td><td>(0.2, 0.1)</td></tr></table>

The two methods are evaluated on identical questions and frames. We score only the answer span after the closing tag, using the released task parsers. EgoSchema reports accuracy. MLVU reports the mean accuracy over its nine task types (M-Avg). For each benchmark, the FLOPs ratio divides LOWN’s summed decoding FLOPs by the baseline’s sum over the same questions. It includes the decoder and LM head and excludes the vision encoder.

## D. Wall-Clock Measurements

## D.1. Generation Time

All LLaDA-V timing experiments were run on the same server with NVIDIA H200 GPUs, with one experiment per GPU at a time. We measure per-example generation time at batch size 1 with a host clock around the model’s generation call. CUDA is synchronized before the timer starts and after generation returns. The interval includes decoding, multimodal preparation, and vision encoding performed within the call. Model loading and evaluation scoring are excluded. Each task mean averages all per-example times from one recorded execution of the full evaluation set. FLOPs count the decoder, LM head, and method-specific matrix multiplications, so their scope difers from the timing interval. The cache comparison in Table 2 includes 1,000 MathVista examples, 60 LLaVA-Bench questions, and 400 DeCapBench examples.

## D.2. Quality and Generation Time

Table 10 repeats the comparison of Table 1 with each competitor selected by measured time. For each competitor, among its measured retentions (down to the floors of Table 1) and KLASS thresholds, we take the setting with the shortest mean generation time that is at least LOWN’s, so no competitor is given less time than LOWN. When even the fastest setting is slower than LOWN, that setting is used. Accuracy plays no part in the selection. Cells within a few percent of LOWN’s time should be read as equal in cost. With every competitor given at least LOWN’s generation time on every benchmark, LOWN keeps the highest average in both blocks, 99.6% and 98.3% of LLaDA-V against at most 94.6% and 92.8% for the competitors, and scores highest on seven and eight of the ten benchmarks, respectively.

Table 10: Performance at matched generation time on LLaDA-V. The upper block compares standalone methods at LOWN’s generation time, and the lower block at that of LOWN with RedVTP (�=50%). Competitors run on a discrete h id h ll h i i h h h d i b O ’ ( bl ) d LOWN gives the highest normalized average in both blocks. Cells show scores with mean generation time relative to LLaDA-V below. Best scores are bold, second best underlined, and lowest times bold.
<table><tr><td>Method</td><td>Math Vista</td><td>Science QA</td><td>Chart QA</td><td>MME</td><td>MM Bench</td><td>Doc VQA</td><td>Cap Arena</td><td>LLaVA- Bench</td><td>GQA</td><td>DeCap Bench</td><td>Avg.</td></tr><tr><td>LLaDA-V (Vanilla)</td><td>59.10</td><td>76.50</td><td>78.16</td><td>1719.49</td><td>70.88</td><td>83.91</td><td>88.67</td><td>58.20</td><td>53.31</td><td>33.18</td><td>100.0%</td></tr><tr><td colspan="10">Time matched to LOWN</td><td></td></tr><tr><td rowspan="3">RedVTP</td><td>51.70</td><td>65.49</td><td>72.20</td><td>1275.08</td><td>40.12</td><td>83.39</td><td>-88.50</td><td>57.60</td><td>48.05</td><td>33.88</td><td>88.7%</td></tr><tr><td>(×0.394)</td><td>(×0.374)</td><td>(×0.674)</td><td>(×0.300)</td><td>(×0.366)</td><td>(×0.566)</td><td>(×0.507)</td><td>(×0.612)</td><td>(×0.318)</td><td>(×0.504)</td><td>(×0.461)</td></tr><tr><td>53.70</td><td>72.48</td><td>76.00</td><td>1582.41</td><td>68.73</td><td>78.73</td><td>-94.67</td><td>56.70</td><td>50.50</td><td>31.15</td><td>94.6%</td></tr><tr><td rowspan="2">ZOO-Prune</td><td>(×0.417)</td><td>(×0.331)</td><td>(×0.690)</td><td>(×0.352)</td><td>(×0.393)</td><td>(×0.548)</td><td>(×0.490)</td><td>(×0.587)</td><td>(×0.301)</td><td>(×0.488)</td><td>(×0.460)</td></tr><tr><td>56.90</td><td>76.05</td><td>73.56</td><td>1641.99</td><td>68.56</td><td>76.67</td><td>-96.83</td><td>42.20</td><td>49.55</td><td>30.18</td><td>92.2%</td></tr><tr><td rowspan="2">KLASS</td><td>(×0.493)</td><td>(×0.337)</td><td>(×0.676)</td><td>(×0.309)</td><td>(×0.387)</td><td>(×0.560)</td><td>(×0.450)</td><td>(×0.573)</td><td>(×0.284)</td><td>(×0.479)</td><td>(×0.455)</td></tr><tr><td>56.00</td><td>78.83</td><td>78.36</td><td>1655.08</td><td>69.93</td><td>82.31</td><td>-85.67</td><td>57.40</td><td>53.24</td><td>34.43</td><td>99.6%</td></tr><tr><td rowspan="2">LOWN (Ours)</td><td>(×0.380)</td><td>(×0.331)</td><td>(×0.666)</td><td>(×0.295)</td><td>(×0.362)</td><td>(×0.537)</td><td>(×0.441)</td><td>(×0.532)</td><td>(×0.268)</td><td>(×0.446)</td><td>(×0.426)</td></tr><tr><td>Time matched to LOWN (Ours)</td><td></td><td></td><td></td><td> + RedVTP (r=50%)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="10"></td><td>32.28</td></tr><tr><td></td><td>41.70 (×0.279)</td><td>65.49 (×0.374)</td><td>70.80 (×0.635)</td><td>1275.08 (×0.300)</td><td>40.12 (×0.366)</td><td>82.14 (×0.408)</td><td>-89.33 (×0.299)</td><td>57.10</td><td>48.05</td><td>(×0.296)</td><td>86.0%</td></tr><tr><td rowspan="2">ZOO-Prune</td><td>46.10</td><td>72.48</td><td>74.92</td><td>1582.41</td><td>68.73</td><td>75.97</td><td>-96.17</td><td>(×0.436) 52.90</td><td>(×0.318) 50.50</td><td>29.56</td><td>(×0.371) 91.5%</td></tr><tr><td>(×0.264)</td><td>(×0.331)</td><td>(×0.648)</td><td>(×0.352)</td><td>(×0.393)</td><td>(×0.453)</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>KLASS</td><td>54.70</td><td>74.67</td><td>70.64</td><td>1606.29</td><td>68.38</td><td>82.07</td><td>(×0.301) -96.00</td><td>(×0.357) 47.20</td><td>(×0.301) 50.74</td><td>(×0.299) 30.13</td><td>(×0.370) 92.8%</td></tr><tr><td>+ RedVTP (r=50%)</td><td>(×0.327)</td><td>(×0.370)</td><td>(×0.657)</td><td>(×0.306)</td><td>(×0.364)</td><td>(×0.418)</td><td>(×0.296)</td><td>(×0.371)</td><td>(×0.261)</td><td>(×0.299)</td><td>(×0.367)</td></tr><tr><td>LOWN (Ours)</td><td>54.90</td><td>75.56</td><td>74.88</td><td>1709.75</td><td>69.67</td><td>82.24</td><td>-85.17</td><td>55.00</td><td>53.42</td><td>33.61</td><td>98.3%</td></tr><tr><td>+ RedVTP (r=50%)</td><td>(×0.262)</td><td>(×0.319)</td><td>(×0.577)</td><td>(×0.276)</td><td>(×0.333)</td><td>(×0.398)</td><td>(×0.284)</td><td>(×0.354)</td><td>(×0.256)</td><td>(×0.278)</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>(×0.334)</td></tr></table>

## D.3. Cache Configuration and Composition

Fast-dLLM uses a block spanning the configured output, prefix refresh every 32 forwards, and its factor-based token selection with factor 1. The combined implementation retains these cache settings, maintains separate image-conditioned and TEXT-ONLY prefix caches, and takes the union of Fast-dLLM’s selected tokens and LOWN’s accepted tokens.

## E. Additional Results

## E.1. Fixed-Retention Pruning Compositions

Table 11 measures the efect of adding LOWN to the unpruned model and to each fixed pruner, using the corresponding configuration without LOWN as the reference. LOWN retains much of its standalone compute reduction when applied on top of pruning, with average quality close to that of each reference configuration.

Table 11: Relative gains from adding LOWN on LLaDA-V and LaViDa. Each row is measured against the listed configuration without LOWN. FLOPs ratio is the mean of the per-benchmark FLOPs ratios, and reduction is one minus this mean. Quality is the mean normalized index with the reference set to 100.
<table><tr><td>Configuration before LOWN</td><td>FLOPs ratio ↓</td><td>Reduction ↑</td><td>Quality index ↑</td></tr><tr><td>LLaDA-V</td><td></td><td></td><td></td></tr><tr><td>LLaDA-V (unpruned)</td><td>0.353</td><td>64.7%</td><td>99.6</td></tr><tr><td>RedVTP (50% retention)</td><td>0.424</td><td>57.6%</td><td>100.9</td></tr><tr><td>ZOO-Prune (33% retention)</td><td>0.394</td><td>60.6%</td><td>98.7</td></tr><tr><td>LaViDa</td><td></td><td></td><td></td></tr><tr><td>LaViDa (unpruned)</td><td>0.174</td><td>82.6%</td><td>99.7</td></tr><tr><td>RedVTP (75% retention)</td><td>0.227</td><td>77.3%</td><td>99.5</td></tr><tr><td>ZOO-Prune (33% retention)</td><td>0.278</td><td>72.2%</td><td>99.8</td></tr></table>

## E.2. Fewer Denoising Steps

Table 12 runs LLaDA-V with �/2 and �/4 denoising steps on the CapArena, LLaVA-Bench, and DeCapBench inputs of Table 1. LOWN’s FLOPs lie between the two settings, and LOWN scores above both on all three benchmarks, including �/2 at 0.500×.

Table 12: LLaDA-V with fewer denoising steps on CapArena, LLaVA-Bench, and DeCapBench. �/2 and �/4 run half and a quarter of the baseline’s � steps on the inputs of Table 1. Scores use the judges of Table 1, and FLOPs are relative to LLaDA-V. DeCapBench averages the captions its oficial evaluator judges valid, 399, 398, 392, and 399 of 400 for the four rows.
<table><tr><td></td><td colspan="2">CapArena</td><td colspan="2">LLaVA-Bench</td><td colspan="2">DeCapBench</td></tr><tr><td>Method</td><td>Score ↑</td><td>FLOPs↓</td><td>Score ↑</td><td>FLOPs↓</td><td>Score ↑</td><td>FLOPs↓</td></tr><tr><td>LLaDA-V (T steps)</td><td>-88.67</td><td>1.000</td><td>58.20</td><td>1.000</td><td>33.18</td><td>1.000</td></tr><tr><td>LLaDA-V (T/2)</td><td>-92.33</td><td>0.500</td><td>45.40</td><td>0.500</td><td>32.17</td><td>0.500</td></tr><tr><td>LLaDA-V (T/4)</td><td>-99.00</td><td>0.250</td><td>26.20</td><td>0.250</td><td>25.84</td><td>0.250</td></tr><tr><td>LOWN (Ours)</td><td>-85.67</td><td>0.388</td><td>57.40</td><td>0.454</td><td>34.43</td><td>0.391</td></tr></table>

## E.3. LaViDa

Figure 8 shows all ten benchmarks, and Table 13 gives the FLOPs-matched comparison in the format of Table 1.

![](images/11602d101a4f90d28066f830b59f0b232928d5d7d444eb5bbddd3bcb2c2339bd.jpg)  
FLOPs (relative to baseline) ↓  
Figure 8: Performance and eficiency on LaViDa, all ten benchmarks. The markers follow Figure 5.

Table 13: Performance at matched decoding FLOPs on LaViDa, in the format of Table 1. Methods are compared at LOWN’s budget. Each cell shows the benchmark score, with the selected retention � and FLOPs relative to the LaViDa baseline below. Competitors run on a discrete grid of retention ratios, so each uses its cheapest measured setting whose FLOPs are at least LOWN’s (Appendix C.3). D<sup>3</sup>ToM uses constant 10% retention on MathVista, LLaVA-Bench, CapArena, and DeCapBench and its adaptive schedule on the remaining tasks. MME reports the perception score. Best scores are bold, second best underlined, and lowest FLOPs bold.

<table><tr><td>Method</td><td>Math Vista</td><td>Science QA</td><td>Chart QA</td><td>MME</td><td>MM Bench</td><td>Doc VQA</td><td>Cap Arena</td><td>LLaVA- Bench</td><td>GQA</td><td>DeCap Bench</td><td>Avg.</td></tr><tr><td>LaViDa (Vanilla)</td><td>43.80</td><td>72.34</td><td>67.16</td><td>1353.03</td><td>76.95</td><td>63.54</td><td>-93.17</td><td>38.80</td><td>56.69</td><td>30.18</td><td>100.0%</td></tr><tr><td colspan="10">FLOPs matched to LOWN</td><td></td></tr><tr><td>RedVTP</td><td>36.60 r=25%(×0.432)</td><td>71.34 r=25%(×0.448)</td><td>20.40 r=10%(×0.244)</td><td>987.28 r=10%(×0.248)</td><td>72.23 r=25%(×0.435)</td><td>18.55 r=10%(× 0.225)</td><td>-99.33 r=20%(×0.347)</td><td>28.60 r=10%(×0.226)</td><td>50.14 r=25%(×0.367)</td><td>22.58 r=25% (×0.389)</td><td>73.9% (×0.336)</td></tr><tr><td rowspan="3">ZOO-Prune</td><td>39.20</td><td>71.05</td><td>39.56</td><td>1220.76</td><td></td><td></td><td></td><td></td><td></td><td>28.55</td><td>84.3%</td></tr><tr><td></td><td></td><td></td><td></td><td>73.76</td><td>30.62</td><td>-95.67</td><td>30.20</td><td>52.23</td><td></td><td></td></tr><tr><td>r=11%(×0.341)</td><td>r=11%(×0.324)</td><td>r=10% (×0.217)</td><td>r=11%(×0.232)</td><td>r=11% (×0.309)</td><td>r=10%(× 0.225)</td><td>r=15% (×0.321)</td><td>r=11%(×0.245)</td><td>r=11%(×0.227)</td><td>r=22%(×0.379)</td><td>(×0.282)</td></tr><tr><td rowspan="2">D3ToM</td><td>36.80</td><td>72.14</td><td>24.24</td><td>1244.92</td><td>73.64</td><td>24.56</td><td>-98.00</td><td>32.00</td><td>52.28</td><td>23.31</td><td>79.3%</td></tr><tr><td>(×0.387)</td><td>(×0.469)</td><td>(×0.392)</td><td>(×0.396)</td><td>(×0.456)</td><td>(×0.376)</td><td>(×0.339)</td><td>(×0.302)</td><td>(×0.391)</td><td>(×0.341)</td><td>(×0.385)</td></tr><tr><td rowspan="2">LOWN (Ours)</td><td>42.90</td><td>72.63</td><td>66.16</td><td>1341.15</td><td>77.18</td><td>62.49</td><td>–92.67</td><td>40.40</td><td>56.54</td><td>29.65</td><td>99.7%</td></tr><tr><td>(×0.229)</td><td>(×0.091)</td><td>(×0.211)</td><td>(×0.078)</td><td>(×0.088)</td><td>(×0.130)</td><td>(×0.316)</td><td>(×0.185)</td><td>(×0.091)</td><td>(×0.323)</td><td>(×0.174)</td></tr></table>

## E.4. Qualitative Samples

Figures 9 and 10 extend the comparison of Figure 6 to more examples. In Figure 9, RedVTP answers the short-answer questions incorrectly and, in open-ended captioning, describes objects and attributes that do not match the image. LOWN gives the same answers as LLaDA-V and describes the same main objects. In Figure 10, RedVTP processes the image at every step, but after pruning it processes only the visual tokens it retains, which exclude the x-axis tick 4.0 that answers the question. Its response is fluent but ends with the incorrect answer 2.5. LOWN processes the image on fewer steps, but each FULL forward covers the complete image.

![](images/67a6e6df3c8e4cb3ebe553fe22e3bcb882e75258e0a938a19809d6def5d9e383.jpg)

![](images/9f05e1b9d85354e94b7ba30c6367b46646ddd02eada2c6be59034ad7a2e1aed4.jpg)  
Figure 9: Additional qualitative comparisons with visual token pruning (RedVTP). The upper three examples are short-answer questions, and the lower three are open-ended captioning. Green marks correct answers and content consistent with the image, and red marks incorrect answers and content inconsistent with the image.

![](images/c6d373d3f137c0d806f4852b7ad13132b9e307e243dff6be296f9f9110f62005.jpg)

![](images/2b1e8ed51275651ccd3be15b48d16655534c5e1c28b70f38b1abce59acbd4c89.jpg)  
Figure 10: Qualitative comparison with visual token pruning (RedVTP) and its pruned image. The upper right darkens the visual tokens removed by RedVTP, and the inset enlarges the region around the x-axis tick 4.0. In LOWN’s output, blue underlines mark tokens unmasked on FULL forward steps, while the remaining tokens are unmasked on TEXT-ONLY forward steps.

## E.5. Limitation Example

Figure 11 shows the DocVQA input on which LOWN saves the least. The question asks for a number in a table, and LOWN runs a FULL forward on 14 of its 15 steps. LOWN answers correctly but costs 0.90× the baseline FLOPs, against 0.45× over the whole benchmark (Table 1).

![](images/c3da96c20aa133aba392129bf4b4382aaa4da070859fbc46ba295d7ab02ebfa9.jpg)  
Figure 11: A limitation example on DocVQA. In LOWN’s output, blue underlines mark tokens unmasked on FULL forward steps, while the remaining tokens are unmasked on TEXT-ONLY forward steps. FLOPs are relative to LLaDA-V on this input.

## F. Additional Ablations and Analysis

## F.1. Order of FULL-Step Unmasking

Table 14 separates the two parts of adaptive acceptance in Table 5. Disagreement-first order removes 14–35% of the cost of visual routing alone while scoring higher on seven of the ten benchmarks, and whole-prefix acceptance removes a further 2–19%.

Table 14: Table 5 with adaptive acceptance split into disagreement-first order on FULL steps and whole-prefix acceptance on TEXT-ONLY steps (Section 4.2), on LLaDA-V. Each setting reports scores and, below, relative FLOPs with the reduction from the preceding setting in parentheses.
<table><tr><td rowspan="2">Visual routing</td><td colspan="2">Adaptive acceptance</td><td rowspan="2">Math Vista</td><td rowspan="2">Science QA</td><td rowspan="2">Chart QA</td><td rowspan="2">MME</td><td rowspan="2">MM Bench</td><td rowspan="2">Doc VQA</td><td rowspan="2">Cap Arena</td><td rowspan="2">LLaVA- Bench</td><td rowspan="2">GQA</td><td rowspan="2">DeCap Bench</td></tr><tr><td></td><td>Disagreement Whole prefix</td></tr><tr><td>x</td><td>x</td><td>X</td><td>59.10</td><td>76.50</td><td>78.16</td><td>1719.5</td><td>70.88</td><td>83.91</td><td>-88.67</td><td>58.20</td><td>53.31</td><td>33.18</td></tr><tr><td>V</td><td>x</td><td>X</td><td>57.60</td><td>77.39</td><td>77.76</td><td>1624.6</td><td>68.30</td><td>83.50</td><td>-86.83</td><td>57.90</td><td>52.53</td><td>33.02</td></tr><tr><td></td><td></td><td></td><td>57.10</td><td>78.19</td><td>78.32</td><td>1688.5</td><td>69.85</td><td>82.33</td><td>-81.17</td><td>57.50</td><td>53.43</td><td>0.5040 (-49.6%) 0.3554(−64.5%)0.7330 (-26.7%) 0.3383 (−66.2%) 0.3759 (−62.4%)0.6743 (-32.6%) 0.6833 (-31.7%) 0.7002(-30.0%) 0.3203 (-68.0%) 0.6840 (-31.6%) 34.71</td></tr><tr><td>V</td><td>√</td><td>X</td><td></td><td>0.3910 (-22.4%) 0.3006 (−15.4%) 0.5891 (−19.6%) 0.2755 (−18.6%) 0.3236 (−13.9%) 0.4789 (-29.0%) 0.4471 (-34.6%) 0.5067 (-27.6%) 0.2469 (22.9%) 0.4450 (−-34.9%)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>√</td><td>√</td><td>56.00</td><td>78.83 0.3252 (−16.8%) 0.2428 (−19.2%) 0.5759 (−2.2%) 0.2307 (−16.3%) 0.2701 (−16.5%) 0.4517 (−5.7%) 0.3885 (-13.1%) 0.4543 (−10.3%) 0.2045 (-17.2%) 0.3909 (−12.2%)</td><td>78.36</td><td>1655.1</td><td>69.93</td><td>82.31</td><td>-85.67</td><td>57.40</td><td>53.24</td><td>34.43</td></tr></table>

## F.2. Certificate Reliability

Coverage, $\epsilon ( \Delta ) _ { : }$ , and $R _ { \mathrm { r e f } }$ are computed on the logged baseline trajectories of Appendix A.1, and the acceptance error and anchor age on LOWN’s own decoding of the same samples (Tables 15 and 16).

At anchor age one the certificate certifies 43–72% of the eligible positions, $\epsilon ( \Delta )$ rises from age one to four on every benchmark, and the acceptance error stays below �(1) on every benchmark. On DocVQA, LOWN costs less than the oracle schedule because adaptive acceptance ends the trajectory early.

Table 15: Certificate reliability on LLaDA-V. Eligible positions are the masked positions that were already masked $\Delta$ steps earlier. Coverage is the share of eligible positions whose TEXT-ONLY prediction agrees with a one-step-old anchor. $\epsilon ( \Delta )$ is the share of agreeing positions at which a fresh FULL forward predicts a diferent token when the anchor is $\Delta$ steps old. $R _ { \mathrm { r e f } } = f ^ { \ast } + ( 1 - f ^ { \ast } ) \rho + 1 / T$ is the cost of the oracle schedule of Section $5 . 3 ,$ which runs a FULL forward only on the fraction $f ^ { * }$ of steps where the image changes an unmasked decision and pays � elsewhere, plus one initial FULL forward. For ChartQA, whose baseline stops at the end-of-sequence token after $7 . 0$ of its 16 steps on average, $f ^ { * }$ and $T$ are taken over the steps the baseline actually runs $( f ^ { \ast } = 3 2 . 3 \% )$ . Acceptance error is the share of the tokens LOWN unmasks on TEXT-ONLY steps that a fresh FULL forward on the state they were accepted from would predict diferently, and $R _ { \mathrm { L O W N } }$ is LOWN’s measured cost. Costs are relative to the baseline and rates are in percent.
<table><tr><td>Benchmark</td><td>Coverage</td><td>€(1)</td><td>∈(2)</td><td>€(4)</td><td>∈(8)</td><td> $R _ { \mathrm { r e f } }$ </td><td>Accept. error</td><td> $R _ { \mathrm { L O W N } }$ </td></tr><tr><td>MathVista</td><td>58.6</td><td>7.3</td><td>9.5</td><td>12.2</td><td>15.4</td><td>0.284</td><td>2.2</td><td>0.325</td></tr><tr><td>ScienceQA</td><td>63.5</td><td>6.4</td><td>8.3</td><td>9.2</td><td>8.8</td><td>0.218</td><td>4.7</td><td>0.243</td></tr><tr><td>ChartQA</td><td>47.0</td><td>10.6</td><td>14.0</td><td>16.6</td><td>15.2</td><td>0.490</td><td>4.2</td><td>0.576</td></tr><tr><td>MME</td><td>63.8</td><td>5.9</td><td>7.6</td><td>10.0</td><td>13.6</td><td>0.193</td><td>3.9</td><td>0.231</td></tr><tr><td>MMBench</td><td>65.3</td><td>6.3</td><td>7.3</td><td>7.5</td><td>6.9</td><td>0.217</td><td>3.9</td><td>0.270</td></tr><tr><td>DocVQA</td><td>43.4</td><td>11.0</td><td>15.5</td><td>20.2</td><td>25.3</td><td>0.517</td><td>3.0</td><td>0.452</td></tr><tr><td>CapArena</td><td>72.1</td><td>5.2</td><td>6.8</td><td>8.7</td><td>10.8</td><td>0.374</td><td>3.5</td><td>0.388</td></tr><tr><td>LLaVA-Bench</td><td>54.0</td><td>9.5</td><td>12.6</td><td>16.1</td><td>19.5</td><td>0.430</td><td>4.2</td><td>0.454</td></tr><tr><td>GQA</td><td>64.7</td><td>4.1</td><td>5.4</td><td>7.0</td><td>11.1</td><td>0.200</td><td>2.3</td><td>0.205</td></tr><tr><td>DeCapBench</td><td>70.6</td><td>5.5</td><td>7.3</td><td>9.2</td><td>11.6</td><td>0.380</td><td>3.4</td><td>0.391</td></tr><tr><td>Mean</td><td></td><td></td><td></td><td></td><td></td><td>0.330</td><td></td><td>0.353</td></tr></table>

Table 16: Anchor age of the tokens LOWN accepts on TEXT-ONLY steps, in LOWN’s own runs on LLaDA-V. Age is the number of LOWN steps since the FULL forward that produced the anchor. Shares are in percent, and the last row pools the accepted tokens of all benchmarks.
<table><tr><td>Benchmark</td><td>Age 1</td><td> $\tt A g e \leq 2$ </td><td>Max age</td></tr><tr><td>MathVista</td><td>93.8</td><td>99.5</td><td>5</td></tr><tr><td>ScienceQA</td><td>92.9</td><td>99.4</td><td>3</td></tr><tr><td>ChartQA</td><td>94.9</td><td>99.9</td><td>3</td></tr><tr><td>MME</td><td>94.2</td><td>99.5</td><td>4</td></tr><tr><td>MMBench</td><td>91.8</td><td>99.4</td><td>3</td></tr><tr><td>DocVQA</td><td>96.7</td><td>99.6</td><td>4</td></tr><tr><td>CapArena</td><td>95.4</td><td>99.8</td><td>3</td></tr><tr><td>LLaVA-Bench</td><td>93.9</td><td>99.7</td><td>4</td></tr><tr><td>GQA</td><td>94.8</td><td>99.5</td><td>4</td></tr><tr><td>DeCapBench</td><td>95.3</td><td>99.9</td><td>3</td></tr><tr><td>All</td><td>94.8</td><td>99.7</td><td>5</td></tr></table>