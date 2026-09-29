# FROM PERCEPTION TO INTEGRATION: REVISITING THE INTERNAL DYNAMICS OF REASONING IN VISION-LANGUAGE MODELS

Rong Yu Xu<sup>1</sup>, Prayag Tiwari<sup>2</sup> & Shaolei Zhang<sup>3∗</sup>

<sup>1</sup>Shenzhen College of International Education s23372.xu@stu.scie.com.cn

<sup>2</sup>Halmstad University, Sweden prayag.tiwari@hh.se

<sup>3</sup>Renmin University of China zhangshaolei98@ruc.edu.cn

## ABSTRACT

Vision-language models (VLMs) can answer simple visual questions, but often struggle when one question requires several visual judgments. We study this gap with controlled tasks for feature binding, numerosity, spatial relations, and amodal completion, together with a Composite task that combines them. Matched counterfactual image pairs isolate changes in the visual evidence needed to answer. Across four models, direct answers, hidden-state readouts, and state interventions show that the individual judgments can be made without explicit reasoning and that intervening on the corresponding states can affect the answer. During reasoning, the Composite answer becomes decodable from hidden states and usable from shortened traces, often before the model stops on its own. We train a small detector to predict this readiness and stop reasoning at that point. On MMStar and RealWorldQA, this reduces mean reasoning tokens by 79.1% and 74.5%, while average accuracy rises by 3.13 and 3.30 percentage points, respectively. These findings connect the internal development of answer readiness to a practical rule for allocating reasoning computation.

## 1 INTRODUCTION

People recognize many objects quickly, but some images require further visual processing. In primates, object information can appear within 150 ms during an initial feedforward pass, while harder images require recurrent processing (Thorpe et al., 1996; DiCarlo et al., 2012; Lamme and Roelfsema, 2000; Kar et al., 2019). VLMs also have a fixed number of layers in each forward pass. Generating more tokens gives them additional steps of computation. This suggests a question: which visual judgments can a VLM make before it starts reasoning, and which improve as reasoning proceeds?

Visual benchmarks show which questions models answer correctly (Johnson et al., 2017; Huang et al., 2025). Mechanistic studies show where visual information appears inside a model (Neo et al., 2024; Zhang et al., 2025b). We ask how the model uses that information to reach an answer (Figure 1). Across four VLMs, accuracy without reasoning is 87.2–91.4% on four separate visual tasks, compared with a 31.3% guess rate. Yet average accuracy falls to 58.6% on a single Composite question with two answer choices, left or right. This question requires the model to combine several visual judgments to select the correct side. High accuracy on the individual tasks therefore does not ensure a correct answer when those judgments must be used together. A correct final answer does not reveal when the model first became able to give it.

We test Feature Binding, Numerosity, Spatial Relations, and Amodal Completion: identifying an object’s features, comparing quantities, judging positions, and recognizing a partly hidden shape (Treisman and Gelade, 1980; Feigenson et al., 2004; Logan, 1994; Kellman and Shipley, 1991). Our Composite task combines these demands by asking which side of a partly occluded scene contains more red circles (Figure 2, rightmost column). We also use matched image pairs that change the correct answer while keeping the rest of the scene fixed (Gardner et al., 2020; Bitton et al., 2021). Hidden-state readouts and state interventions test what information the model carries and uses. Answers from shortened reasoning traces show when it can use that information to answer the Composite question.

![](images/7e9c70d153f144f4cac0380674c5dbe162cc829b2ce7560f0f1e93adf33e556b.jpg)  
Figure 1: Overview of our study. Controlled visual tasks test individual judgments and their combination in Composite scenes. Linear readouts and counterfactual state interventions examine what visual information is represented and affects the answer. During native reasoning, readouts and answers from shortened traces track when the Composite answer becomes available. A hidden-state readiness detector uses this principle to stop reasoning early on MMStar and RealWorldQA.

Reasoning raises Composite pair-consistent accuracy by 40.8 points on average. It is also costly: full traces average 1,650 reasoning tokens on MMStar, and 16.6% of Composite runs reach the generation limit without a final answer. We find that a shortened trace can often support a correct answer before the model stops. We call the earliest tested point where this happens answer readiness and ask whether the model’s state can detect it during generation.

We propose Answer-Readiness-Guided Early Stopping. A linear detector uses the current hidden state and its change since the previous checkpoint to predict whether the model can answer correctly now. If the prediction crosses a threshold selected on validation data, the model stops reasoning and generates its answer. This approach builds on earlier methods that stop when an answer appears correct or sufficient (Zhang et al., 2025a; Yang et al., 2026; Xiang et al., 2026).

Across four VLMs on MMStar and RealWorldQA, stopping shortens mean reasoning by 79.1% and 74.5% while average accuracy rises by 3.13 and 3.30 points.

## 2 RELATED WORK

Multimodal reasoning and visual evaluation. Controlled benchmarks test how VLMs combine visual information (Johnson et al., 2017; Schiappa et al., 2024; Huang et al., 2025). Multimodal chain-of-thought methods ask models to explain their answers or reason through image regions and objects (Zhang et al., 2023; Shao et al., 2024; Li et al., 2025). Other evaluations find that more reasoning helps some visual questions but not others (Jiang et al., 2025; Jin et al., 2026). These studies measure final answers; they do not show when reasoning first makes an answer possible.

Interpreting visual representations in VLMs. Researchers use causal tracing and token interventions to locate information inside language models and VLMs (Meng et al., 2022; Basu et al., 2024; Neo et al., 2024). Related work studies how visual information moves through direct and text-mediated pathways (Zhang et al., 2025b; Salazar et al., 2026). Layer-wise readouts also separate visual grounding from later processing and answer formatting (Yu and Lee, 2025). These studies focus on a single forward pass. They do not track how the model uses visual information while generating a reasoning trace.

Efficient reasoning and early exit. Hidden-state readouts can predict reasoning success or check intermediate answers (Afzal et al., 2025; Zhang et al., 2025a). Other methods stop when a trial answer has high confidence or appears sufficient (Yang et al., 2026; Xiang et al., 2026). For VLMs, Bi et al. (2026) estimate when reasoning helps and how long it should continue, then intervene on attention heads. These methods do not use a direct measurement of when the visual answer becomes available during reasoning.

Our analysis measures when a Composite answer becomes usable during reasoning. That measurement provides the target for a detector that decides when to stop.

## 3 INTERPRETING COMPOSITE VISUAL REASONING

## 3.1 INTERPRETABILITY ANALYSIS SETUP

Each task isolates one answer-determining variable while holding the remaining scene content constant.

Feature Binding. Feature Binding asks which color belongs to a marked object. People can notice a color and a shape but sometimes combine them incorrectly (Treisman and Gelade, 1980; Treisman and Schmidt, 1982). Each image contains several colored shapes; a black outline marks the target. The model reports its color.

Numerosity. Numerosity asks which side of a divider has more dots. Even infants can compare such quantities before learning number words (Feigenson et al., 2004; Xu and Spelke, 2000). We vary the relationship between dot count and total dot area, so area alone does not always reveal the answer. The model responds left or right rather than giving an exact count.

Spatial Relations. Spatial Relations asks where a target lies relative to another object. Such judgments require selecting the two relevant objects (Carlson-Radvansky et al., 1999; Logan, 1994). Black and gray outlines mark the target and reference, while their colors and shapes vary. The model chooses left, right, above, or below.

Amodal Completion. Amodal Completion asks what whole shape is partly hidden behind an occluder. People can infer the whole from visible fragments (Kellman and Shipley, 1991; Sekuler and Palmer, 1992). We partly cover a geometric shape and ask the model to choose its complete form. The Composite task needs this judgment for partly hidden objects.

Composite Task. The Composite task asks which side of a divider has more red circles among colored, partly hidden shapes. To answer, the model must recognize each shape, match colors to shapes, assign objects to a side, and compare the two counts. This resembles “challenge images” in vision research: the parts may be easy to recognize, but combining them can require more processing (Kar et al., 2019; Kar and DiCarlo, 2021).

Counterfactual Pairs. For state interventions, we use matched image pairs that differ in the feature that determines the answer while keeping the rest of the scene fixed (Gardner et al., 2020; Bitton et al., 2021). Feature Binding, Numerosity, and Spatial Relations use separate datasets for direct answers and interventions. Amodal Completion and Composite use paired datasets throughout. Figure 2 shows one pair per task; Appendix A gives the construction rules and splits.

![](images/051f1897b605b50c214581a9c9e5580ed7b86f656c52095cf53782760d270dc4.jpg)  
Figure 2: Every column shows a counterfactual pair in which the task-relevant variable and the correct answer change together while the remaining scene content is preserved.

## 3.2 VISUAL JUDGMENTS WITHOUT REASONING

The four component judgments provide three complementary tests without explicit reasoning: directanswer accuracy, linear readouts of hidden states, and interventions that measure whether changing those states affects the answer (Belinkov, 2022).

Component judgments are available without reasoning. We freeze each model, disable thinking, and ask it to answer the four tasks directly. All four models perform above the task-specific guess levels (Table 1). They do best on Feature Binding and Spatial Relations and make more errors on Numerosity and Amodal Completion.

Table 1: All four component judgments can be performed above the guess level without explicit reasoning. Answer accuracy (%) on 1,000 images per task and model; guess is uniform random selection over the answer choices (two for Numerosity, four for the others).
<table><tr><td>Task</td><td>Qwen3.5-4B</td><td>Qwen3.5-9B</td><td>Gemma-4-E4B</td><td>Gemma-4-12B</td><td>Guess</td></tr><tr><td>Feature Binding</td><td>98.0</td><td>99.8</td><td>98.6</td><td>99.9</td><td>25.0</td></tr><tr><td>Numerosity</td><td>79.6</td><td>85.2</td><td>72.4</td><td>86.5</td><td>50.0</td></tr><tr><td>Spatial Relations</td><td>99.8</td><td>100.0</td><td>97.5</td><td>89.5</td><td>25.0</td></tr><tr><td>Amodal Completion</td><td>76.8</td><td>80.5</td><td>81.4</td><td>72.9</td><td>25.0</td></tr><tr><td>Avg</td><td>88.6</td><td>91.4</td><td>87.5</td><td>87.2</td><td>31.3</td></tr></table>

Primitive variables are linearly recoverable across depth. Direct answers show that a model can make these judgments, but do not show where it represents them. We therefore average its image-token hidden states at each layer:

$$
v _ { i } ^ { ( \ell ) } = \frac { 1 } { | \mathcal { T } _ { i } | } \sum _ { t \in \mathcal { T } _ { i } } h _ { i , t } ^ { ( \ell ) } ,\tag{1}
$$

where $h _ { i , t } ^ { ( \ell ) }$ is the representation of sample i at layer ℓ and token position $t ,$ and $\mathcal { T } _ { i }$ contains its image-token positions. We standardize these vectors using training-set statistics to obtain $z _ { i } ^ { ( \ell ) }$ , whose rows form the training matrix $Z _ { \ell } .$ . For each model and task, we fit an independent ridge linear readout at every layer, keeping the VLM frozen and fitting only the readout matrix:

$$
\boldsymbol { W } _ { \ell } ^ { * } = \arg \operatorname* { m i n } _ { \boldsymbol { W } } \| Z _ { \ell } \boldsymbol { W } - \boldsymbol { Y } \| _ { F } ^ { 2 } + \lambda \| \boldsymbol { W } \| _ { F } ^ { 2 } , \qquad \boldsymbol { f } _ { \mathrm { r e a d } } \big ( \boldsymbol { z } _ { i } ^ { ( \ell ) } \big ) = \arg \operatorname* { m a x } _ { c } \big ( \boldsymbol { z } _ { i } ^ { ( \ell ) } \boldsymbol { W } _ { \ell } ^ { * } \big ) _ { c } .\tag{2}
$$

Here Y contains the correct task label: color, side with more dots, spatial direction, or complete shape. The parameter λ controls regularization, and $f _ { \mathrm { r e a d } }$ is the fitted readout. The accuracy of $f _ { \mathrm { r e a d } }$ measures how well that label can be decoded from the frozen model’s states.

We test each readout on held-out samples and keep paired images in the same split. As a control, we fit a separate readout using question tokens without an image. For Amodal Completion, the readout predicts one of six shape labels, whereas direct answering chooses among four options in the question. Appendix A.2 gives the splits and fitting details.

![](images/526a426a55e96139615b770e5eec941d770afa84b4e381087820c8b232c62360.jpg)  
Figure 3: Top: test accuracy of readouts using image states (solid) or question-only states (dashed); gray dotted lines mark chance. Bottom: how often matched donor states move scores toward the donor’s answer compared with control donors, in percentage points. Shading shows pair-bootstrap 95% CIs. Layer depth is normalized within each model.

In the upper row of Figure 3, each task label is decodable from several layers. Numerosity and complete-shape accuracy is high; Feature Binding and Spatial Relations vary more across layers. Question-only controls stay near chance for the first three tasks. They score higher for Amodal Completion because the candidate list itself gives some information about the shape. Image-conditioned readouts outperform these controls through most layers. State interventions test whether this decoded information also affects the answer.

State interventions affect answers at some layers. Reading a task label from a state does not prove that the model uses that state to answer. We test this with matched image pairs. One image is the receiver; the other, with a different correct answer, is the donor. At a selected decoder block, we replace the receiver’s image-token states with the donor’s and continue the forward pass:

$$
f _ { \mathrm { p a t c h } } ( r , d , \ell ) \quad \mathrm { a s s i g n s } \quad H _ { r , \mathcal { T } _ { r } } ^ { ( \ell ) } \gets H _ { d , \mathcal { T } _ { d } } ^ { ( \ell ) } .\tag{3}
$$

We swap states in both directions and retain only pairs for which both original answers are correct under the intervention scoring protocol. This leaves a small set whose size varies by model. As a control, we also use a donor from another pair that shares the receiver’s label.

An intervention can shift answer scores even if the top answer stays the same (Heimersheim and Nanda, 2024). We measure this shift with $m = s ( y _ { d } ) - s ( y _ { r } )$ , the score for the donor’s answer minus the score for the receiver’s answer. We then compare how often the matched donor and the control donor increase this margin:

$$
\Delta _ { \mathrm { p a t c h } } = 1 0 0 \left[ \mathrm { P r } ( m _ { \mathrm { t a r g e t } } > m _ { \mathrm { b a s e } } ) - \mathrm { P r } ( m _ { \mathrm { r a n d o m } } > m _ { \mathrm { b a s e } } ) \right] .\tag{4}
$$

Here $m _ { \mathrm { b a s e } }$ is the receiver’s score before replacement. A positive $\Delta _ { \mathrm { p a t c h } }$ means that matched donors push scores toward their answers more often than control donors do. It measures how often scores move, not how far they move or how often the final choice flips. Appendix A.2 gives the scoring and matching rules.

In the lower row of Figure 3, matched donors move scores toward their answers more often than control donors in early and middle layers, but less consistently at the final tested block. Some task labels remain decodable in late layers even when matched-donor replacement no longer shows a consistent advantage over control replacement. Thus, decodability and the measured intervention effect can differ across depth. The results suggest that the Composite task is difficult because the model must combine the component judgments, not simply recognize their parts. A readout alone cannot establish that the model will use the information. We then test whether a shortened reasoning trace already supports the correct answer.

## 3.3 DEVELOPMENT OF COMPOSITE ANSWER READINESS

Reasoning improves counterfactual consistency. We compare direct answers with the model’s normal reasoning on the same held-out Composite pairs. Sample accuracy counts correct images. Pair-consistent accuracy counts a pair only when both images are answered correctly.

Table 2: Native thinking improves counterfactual consistency. Values are accuracy (%) with pair-bootstrap 95% CIs; $\Delta$ is the reasoning-induced change in percentage points. Budget hit counts thinking-mode runs that reach the generation limit, out of 200 per model; the summary row pools all 800 runs.
<table><tr><td rowspan="2">Model</td><td colspan="3">Sample Accuracy</td><td colspan="3">Pair-Consistent Accuracy</td><td>Budget hit</td></tr><tr><td>Non-thinking</td><td>Thinking</td><td>∆</td><td>Non-thinking</td><td>Thinking</td><td>∆</td><td>Count (%)</td></tr><tr><td>Qwen3.5-4B</td><td>58.5 [53.0,64.0]</td><td>80.0 [74.0,86.0]</td><td>↑21.5</td><td>26.0 [18.0,35.0]</td><td>67.0 [58.0,76.0]</td><td>↑41.0</td><td>38 (19.0%)</td></tr><tr><td>Qwen3.5-9B</td><td>66.5 [61.5,71.5]</td><td>94.0 [90.5,97.0]</td><td>↑27.5</td><td>36.0 [27.0,45.0]</td><td>89.0 [83.0,95.0]</td><td>↑53.0</td><td>11 (5.5%)</td></tr><tr><td>Gemma-4-E4B</td><td>52.0 [49.0,55.0]</td><td>73.5 [67.0,80.0]</td><td>↑21.5</td><td>7.0 [2.0,12.0]</td><td>57.0 [47.0,67.0]</td><td>↑50.0</td><td>0 (0.0%)</td></tr><tr><td>Gemma-4-12B</td><td>57.5</td><td>54.5</td><td>↓3.0</td><td>15.0</td><td>34.0</td><td>↑19.0</td><td>84 (42.0%)</td></tr><tr><td>Avg</td><td>[54.0,61.0] 58.6</td><td>[47.0,62.0] 75.5</td><td>↑16.9</td><td>[8.0,22.0] 21.0</td><td>[25.0,43.0] 61.8</td><td>↑40.8</td><td>133 (16.6%)</td></tr></table>

Reasoning improves pair-consistent accuracy for all four models (Table 2). Sample accuracy rises for three models but falls for Gemma-4-12B. This model also often reaches the generation limit; these results do not show whether that limit causes its lower sample accuracy. Final answers still leave one question open: when during reasoning could the model first give the correct answer?

Tracing when the Composite answer becomes usable. We examine each model’s reasoning trace at five checkpoints: before reasoning (PRE), early, middle, at the last reasoning token (LATE), and just before the final choice (ANSWER). Validation data select one layer per model based on how well it represents the answer before reasoning. We use that same layer at every checkpoint and on all test samples. This analysis includes only pairs with valid checkpoints and parsed answers in both traces, so its sample sizes can differ from the direct-answer test. Appendix A.3 defines the checkpoints and selected layers; Appendix A gives the splits.

At each checkpoint, a separate linear readout predicts the correct side from the selected hidden state.   
We use the same feature standardization, ridge objective, and pair-preserving splits as in Section 3.2.   
Test accuracy measures how well the answer is decodable from that state.

We also test whether the model can answer from the reasoning generated so far. At each checkpoint, we truncate the trace after that checkpoint, append the same answer cue, and score the two possible answers with the frozen model:

$$
f _ { \mathrm { f o r c e } } ( i , t ) = \operatorname * { a r g m a x } _ { c \in \{ \mathrm { l e f t } , \mathrm { r i g h t } \} } s _ { \theta } ( c \mid x _ { i } , r _ { i , \le t } , a ) , \qquad U ( t ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbf { 1 } \big [ f _ { \mathrm { f o r c e } } ( i , t ) = y _ { i } \big ] .\tag{5}
$$

Here $x _ { i }$ is the image and prompt, $r _ { i , \leq t }$ is the reasoning kept through checkpoint t, and a is the fixed answer cue, which gives the format but not the answer. The score $s _ { \theta }$ compares the two choices. Unlike the readout from one hidden state, $f _ { \mathrm { f o r c e } }$ uses the full retained trace, including any intermediate reasoning in text. Its accuracy measures whether the answer is usable at each checkpoint. This offline test needs the correct answer for scoring; that answer is unavailable when the model is deployed.

Decodability and usability develop during reasoning. Before reasoning, neither the hidden-state readout nor the shortened-trace test is consistently accurate across models (Figure 4). Both improve during reasoning, though at different speeds. Usability rises earlier for the Qwen models and Gemma 4-12B, and later for Gemma-4-E4B. By LATE, both decodability and usability are high for all four models.

![](images/48defefbc22c5e97354ff9cbb40cc86ca78f705c5ad2adf2d366cd06189c378c.jpg)  
Figure 4: Each panel shows one model at five reasoning checkpoints. Blue: accuracy of a hidden-state readout for the Composite answer. Orange: accuracy when the model answers from a trace cut at that checkpoint. Shading shows pair-bootstrap 95% CIs; the dotted line marks chance.

Accuracy does not always improve from LATE to ANSWER: it stays high for the Qwen models but falls for both Gemma models. Correct answers are often available at earlier checkpoints, before the model finishes reasoning. Moreover, 16.6% of Composite thinking runs reach the generation limit without a final answer (Table 2). These results motivate predicting usability during generation.

## 4 ANSWER-READINESS-GUIDED EARLY STOPPING

The shortened-trace test shows that an answer can be usable before reasoning ends. Figure 1 shows how this finding becomes a stopping rule. For each model and dataset, a small detector reads the current hidden state and its change at a layer selected on Composite data. Its predicted readiness score determines when to stop: once it crosses a threshold chosen on validation data, the model generates its answer. This resembles the use of an evidence threshold in studies of human perceptual decisions (O’Connell et al., 2012).

Predicting readiness without the correct answer. The offline test $f _ { \mathrm { f o r c e } }$ uses the correct answer to measure readiness. At inference time, the detector must predict readiness without that answer. For training, we stop recorded traces at fixed checkpoints, ask the frozen VLM to answer, and label each checkpoint by whether that answer is correct. The detector uses the current state, its change since the previous checkpoint, and the elapsed reasoning length:

$$
\begin{array} { r } { f _ { \mathrm { s t a t e } } ( i , t ) = \big [ h _ { i , t } ^ { ( \ell ) } , ~ d _ { i , t } ^ { ( \ell ) } , ~ t / B \big ] , \qquad f _ { \mathrm { r e a d y } } ( i , t ) = \sigma \big ( w ^ { \top } \widetilde { f } _ { \mathrm { s t a t e } } ( i , t ) + b \big ) , } \end{array}\tag{6}
$$

Here t counts generated reasoning tokens, $d _ { i , t } ^ { ( \ell ) }$ is the change since the previous checkpoint, and B is the generation budget. At the first checkpoint, we use a zero vector as the previous state, so $d _ { i , \Delta } ^ { ( \ell ) } = h _ { i , \Delta } ^ { \overline { { ( \ell ) } } }$ ; subsequent checkpoints use $d _ { i , t } ^ { ( \ell ) } \bar { = } h _ { i , t } ^ { ( \ell ) } - h _ { i , t - \Delta } ^ { ( \ell ) }$ . Training and online inference use the same convention. The tilde denotes standardization using training data. Only the detector weights w and bias b are trained.

Detector training. With checkpoint labels $y _ { i , t } = \mathbf { 1 } [ \hat { a } _ { i , t } ^ { \mathrm { f o r c e d } } = a _ { i } ^ { * } ]$ , we minimize

$$
\mathcal { L } = - \frac { 1 } { M } \sum _ { ( i , t ) \in \mathcal { D } _ { \mathrm { t r a i n } } } \left[ \alpha y _ { i , t } \log f _ { \mathrm { r e a d y } } ( i , t ) + ( 1 - y _ { i , t } ) \log \left( 1 - f _ { \mathrm { r e a d y } } ( i , t ) \right) \right] ,\tag{7}
$$

where $\hat { a } _ { i , t } ^ { \mathrm { f o r c e d } }$ is the forced answer, $a _ { i } ^ { * }$ is its reference, M counts training checkpoints, and α is their negative-to-positive label ratio. Feature normalization and class weights use only training checkpoints.

Stopping at a validated decision bound. We check the detector every $\Delta = 2 5 6$ reasoning tokens. At the first checkpoint where $f _ { \mathrm { r e a d y } } ( i , t ) \geq \eta$ , we close the reasoning channel and ask the VLM for its final answer. The interval is fixed; validation data select only the threshold η. We choose the threshold that minimizes mean reasoning length while keeping accuracy within a set tolerance of full thinking. If the score never crosses it, generation continues until the model stops naturally or reaches the token budget.

## 5 EXPERIMENTS

## 5.1 EXPERIMENTAL SETUP

We compare each model’s adaptive stopping run with its full-thinking run using the same token budget and answer extraction rule. MMStar (Chen et al., 2024) has multiple-choice questions about perception, science, and mathematics. RealWorldQA (x.ai) uses photographs and includes both multiple-choice and short-answer questions. We test all 1,500 MMStar and 765 RealWorldQA questions with two-fold cross-validation. One fold supplies detector training and threshold validation data; the other is held out for testing. All checkpoints from one question, and RealWorldQA questions with the same image, stay in the same fold. Each question therefore receives one held-out prediction per condition. Both conditions use a B = 6,144-token budget. Appendix B gives the split counts.

Threshold selection and answer transition. Validation selects the threshold that uses the fewest mean reasoning tokens while allowing at most 0.5 percentage points less accuracy than full thinking. When the detector triggers, we close the thinking channel and append Therefore, the answer is. Training labels and adaptive inference use this same transition. Each model, dataset, and evaluation direction has its own detector and threshold.

Answer scoring and token accounting. Within each benchmark, both conditions use the same answer extraction rule, which does not depend on the correct answer. MMStar uses the final choice letter. RealWorldQA accepts a choice or short answer, as the question requires. We extract answers after the thinking channel closes and also accept direct short answers for RealWorldQA. A missing or unparseable answer counts as wrong.

Reasoning length counts tokens before the thinking channel closes. If it never closes, we count all generated tokens as reasoning. A direct answer has zero reasoning tokens. Total output length also includes answer tokens. We average lengths over all held-out questions, including those answered before the first checkpoint.

## 5.2 MAIN RESULTS

Shorter reasoning with higher average accuracy. On MMStar, every model uses fewer reasoning tokens with adaptive stopping, and average accuracy rises (Table 3). Gemma-4-12B gains the most accuracy (+8.87 points), while Qwen3.5-4B is nearly unchanged (+0.07). Total output tokens fall by 23.3–84.9% across models (Table 4). The detector stops at the first 256-token check on 15.9–94.7% of questions, depending on the model. Appendix C.1 reports each test fold separately.

Table 3: Adaptive stopping cuts reasoning cost on both benchmarks while raising average accuracy. All held-out cases per model (1,500 MMStar and 765 RealWorldQA); accuracy in %, reasoning tokens as means, ∆ in percentage points, reduction as the token decrease.

<table><tr><td></td><td></td><td colspan="3">Accuracy (%)</td><td colspan="3">Reasoning Tokens</td></tr><tr><td>Benchmark</td><td>Model</td><td>Full</td><td>Stop</td><td>∆</td><td>Full</td><td>Stop</td><td>Reduction</td></tr><tr><td rowspan="5">MMStar</td><td>Qwen3.5-4B</td><td>64.87</td><td>64.93</td><td>↑0.07</td><td>2,041</td><td>255</td><td>87.5%</td></tr><tr><td>Qwen3.5-9B</td><td>66.87</td><td>69.87</td><td>↑3.00</td><td>1,951</td><td>255</td><td>86.9%</td></tr><tr><td>Gemma-4-E4B</td><td>58.07</td><td>58.67</td><td>↑0.60</td><td>753</td><td>585</td><td>22.3%</td></tr><tr><td>Gemma-4-12B</td><td>56.93</td><td>65.80</td><td>↑8.87</td><td>1,854</td><td>286</td><td>84.6%</td></tr><tr><td>Avg</td><td>61.69</td><td>64.82</td><td>↑3.13</td><td>1,650</td><td>345</td><td>79.1%</td></tr><tr><td rowspan="5">RealWorldQA</td><td>Qwen3.5-4B</td><td>72.81</td><td>75.42</td><td>↑2.61</td><td>1,696</td><td>285</td><td>83.2%</td></tr><tr><td>Qwen3.5-9B</td><td>72.94</td><td>77.91</td><td>↑4.97</td><td>1,606</td><td>248</td><td>84.6%</td></tr><tr><td>Gemma-4-E4B</td><td>58.17</td><td>57.12</td><td>↓1.05</td><td>442</td><td>346</td><td>21.8%</td></tr><tr><td>Gemma-4-12B</td><td>60.13</td><td>66.80</td><td>↑6.67</td><td>859</td><td>293</td><td>65.9%</td></tr><tr><td>Avg</td><td>66.01</td><td>69.31</td><td>↑3.30</td><td>1,151</td><td>293</td><td>74.5%</td></tr></table>

Results on short-answer questions. RealWorldQA tests whether the rule also works with short answers (Table 3). All four models use fewer reasoning tokens, and total output falls by 19.9–83.7% (Table 4). Average accuracy rises, although Gemma-4-E4B loses 1.05 points. The rule reduces output length for both answer formats; its accuracy effect varies by model.

Table 4: Mean total output tokens and detector-triggered stopping (% of all questions).
<table><tr><td></td><td colspan="3">Total output tokens</td><td colspan="2">Stopping (%)</td></tr><tr><td>Model</td><td>Full</td><td>Stop</td><td>Reduction</td><td>Any check</td><td>First check</td></tr><tr><td>MMStar</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen3.5-4B</td><td>2,159</td><td>326</td><td>84.9%</td><td>95.1%</td><td>94.7%</td></tr><tr><td>Qwen3.5-9B</td><td>2,066</td><td>315</td><td>84.8%</td><td>94.7%</td><td>94.3%</td></tr><tr><td>Gemma-4-E4B</td><td>944</td><td>724</td><td>23.3%</td><td>24.7%</td><td>15.9%</td></tr><tr><td>Gemma-4-12B</td><td>1,974</td><td>351</td><td>82.2%</td><td>76.3%</td><td>68.1%</td></tr><tr><td>RealWorldQA</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen3.5-4B</td><td>1,699</td><td>299</td><td>82.4%</td><td>79.1%</td><td>71.1%</td></tr><tr><td>Qwen3.5-9B</td><td>1,610</td><td>263</td><td>83.7%</td><td>79.3%</td><td>78.4%</td></tr><tr><td>Gemma-4-E4B</td><td>448</td><td>359</td><td>19.9%</td><td>38.2%</td><td>29.5%</td></tr><tr><td>Gemma-4-12B</td><td>865</td><td>314</td><td>63.7%</td><td>39.9%</td><td>33.1%</td></tr></table>

Where the accuracy gains come from. Table 5 separates questions by whether full thinking closes its reasoning channel. On MMStar, adaptive stopping reduces missing answers from 297 to 24 for Qwen3.5-4B and from 276 to 44 for Qwen3.5-9B. For Qwen3.5-4B the 96 correct answers gained on open traces are almost exactly offset by 95 lost on closed traces, leaving a net gain of one; Qwen3.5-9B gains 121 on open traces against 76 lost on closed ones. Gemma-4-12B gains on both groups: 100 correct answers on open traces and 33 on closed ones, while Gemma-4-E4B gains 20 on open traces and loses 11 on closed ones. On RealWorldQA, Gemma-4-E4B has no open full-thinking traces to recover; it loses eight correct answers on closed traces.

Table 5: Answer completion and changes in correct-answer counts. Missing includes traces without a parsed answer. Open/closed refer to full thinking; direct answers count as closed. Gains are stopping minus full thinking.
<table><tr><td></td><td colspan="2">Missing answer</td><td colspan="2">Full status</td><td colspan="3">Net correct gain</td></tr><tr><td>Model</td><td>Full</td><td>Stop</td><td>Unclosed</td><td>Closed</td><td>Unclosed</td><td>Closed</td><td>Total</td></tr><tr><td colspan="8">MMStar</td></tr><tr><td>Qwen3.5-4B</td><td>297</td><td>24</td><td>267</td><td>1,233</td><td>96</td><td>-95</td><td>1</td></tr><tr><td>Qwen3.5-9B</td><td>276</td><td>44</td><td>243</td><td>1,257</td><td>121</td><td>-76</td><td>45</td></tr><tr><td>Gemma-4-E4B</td><td>104</td><td>87</td><td>32</td><td>1,468</td><td>20</td><td>-11</td><td>9</td></tr><tr><td>Gemma-4-12B</td><td>318</td><td>56</td><td>207</td><td>1,293</td><td>100</td><td>33</td><td>133</td></tr><tr><td colspan="8">RealWorldQA</td></tr><tr><td>Qwen3.5-4B</td><td>112</td><td>8</td><td>112</td><td>653</td><td>48</td><td>-28</td><td>20</td></tr><tr><td>Qwen3.5-9B</td><td>96</td><td>12</td><td>96</td><td>669</td><td>53</td><td>-15</td><td>38</td></tr><tr><td>Gemma-4-E4B</td><td>3</td><td>29</td><td>0</td><td>765</td><td>0</td><td>-8</td><td>-8</td></tr><tr><td>Gemma-4-12B</td><td>25</td><td>20</td><td>16</td><td>749</td><td>9</td><td>42</td><td>51</td></tr></table>

## 6 CONCLUSION

VLMs can make the component visual judgments without explicit reasoning, but combining them is harder. During reasoning, the Composite answer becomes decodable from hidden states and usable from a shortened trace, often before the model stops on its own. A small detector predicts this point and stops reasoning early. Across MMStar and RealWorldQA, it reduces reasoning tokens and improves average accuracy, partly by producing answers from traces that otherwise end without one.

## REFERENCES

Anum Afzal, Florian Matthes, Gal Chechik, and Yftah Ziser. Knowing before saying: Llm representations encode information about chain-of-thought success before completion. In Findings of the Associationfor Computational Linguistics: ACL 2025, pages 12791–12806. Association for Computational Linguistics, 2025. doi: 10.18653/v1/2025.findings-acl.662.

Samyadeep Basu, Martin Grayson, Cecily Morrison, Besmira Nushi, Soheil Feizi, and Daniela Massiceti. Understanding information storage and transfer in multi-modal large language models. arXiv preprint arXiv:2406.04236, 2024. doi: 10.48550/ARXIV.2406.04236.

Yonatan Belinkov. Probing classifiers: Promises, shortcomings, and advances. Computational Linguistics, 48(1):207–219, 2022. ISSN 1530-9312. doi: 10.1162/coli a 00422.

Jing Bi, Luchuan Song, Dingxin Zhang, Pinxin Liu, Guangyu Sun, Lianggong Bruce Wen, Weidong Cai, Chen Chen, and Chenliang Xu. Does reasoning improve seeing? understanding when visionlanguage models benefit from thinking. In Proceedings ofthe International Conference on Machine Learning 2026, 2026.

Yonatan Bitton, Gabriel Stanovsky, Roy Schwartz, and Michael Elhadad. Automatic generation of contrast sets from scene graphs: Probing the compositional consistency of gqa. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 94–105. Association for Computational Linguistics, 2021. doi: 10.18653/v1/2021.naacl-main.9.

Laura A. Carlson-Radvansky, Eric S. Covey, and Kathleen M. Lattanzi. “what” effects on “where”: Functional influences on spatial relations. Psychological Science, 10(6):516–521, 1999. ISSN 1467-9280. doi: 10.1111/1467-9280.00198.

Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, and Feng Zhao. Are we on the right way for evaluating large visionlanguage models? arXiv preprint arXiv:2403.20330, 2024. doi: 10.48550/ARXIV.2403.20330.

James J. DiCarlo, Davide Zoccolan, and Nicole C. Rust. How does the brain solve visual object recognition? Neuron, 73(3):415–434, 2012. ISSN 0896-6273. doi: 10.1016/j.neuron.2012.01.010.

Lisa Feigenson, Stanislas Dehaene, and Elizabeth Spelke. Core systems of number. Trends in Cognitive Sciences, 8(7):307–314, 2004. ISSN 1364-6613. doi: 10.1016/j.tics.2004.05.002.

Matt Gardner, Yoav Artzi, Victoria Basmov, Jonathan Berant, Ben Bogin, Sihao Chen, Pradeep Dasigi, Dheeru Dua, Yanai Elazar, Ananth Gottumukkala, Nitish Gupta, Hannaneh Hajishirzi, Gabriel Ilharco, Daniel Khashabi, Kevin Lin, Jiangming Liu, Nelson F. Liu, Phoebe Mulcaire, Qiang Ning, Sameer Singh, Noah A. Smith, Sanjay Subramanian, Reut Tsarfaty, Eric Wallace, Ally Zhang, and Ben Zhou. Evaluating models’ local decision boundaries via contrast sets. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2020, pages 1307–1323. Association for Computational Linguistics, 2020. doi: 10.18653/v1/2020.findings-emnlp.117.

Stefan Heimersheim and Neel Nanda. How to use and interpret activation patching. arXiv preprint arXiv:2404.15255, 2024. doi: 10.48550/ARXIV.2404.15255.

Jen-Tse Huang, Dasen Dai, Jen-Yuan Huang, Youliang Yuan, Xiaoyuan Liu, Wenxuan Wang, Wenxiang Jiao, Pinjia He, Zhaopeng Tu, and Haodong Duan. Human cognitive benchmarks reveal foundational visual gaps in mllms. arXiv preprint arXiv:2502.16435, 2025. doi: 10.48550/ARXIV. 2502.16435.

Dongzhi Jiang, Renrui Zhang, Ziyu Guo, Yanwei Li, Yu Qi, Xinyan Chen, Liuhui Wang, Jianhan Jin, Claire Guo, Shen Yan, Bo Zhang, Chaoyou Fu, Peng Gao, and Hongsheng Li. Mme-cot: Benchmarking chain-of-thought in large multimodal models for reasoning quality, robustness, and efficiency. In International Conference on Machine Learning, pages 27793–27830, 2025.

Zhuoran Jin, Kejian Zhu, Hongbang Yuan, Yupu Hao, Pengfei Cao, Yubo Chen, Kang Liu, and Jun Zhao. Look light, think heavy: What multimodal chain-of-thought reasoning can and cannot do. In Proceedings ofthe 64th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 8567–8608. Association for Computational Linguistics, 2026. doi: 10.18653/v1/2026.acl-long.387.

Justin Johnson, Bharath Hariharan, Laurens van der Maaten, Li Fei-Fei, C. Lawrence Zitnick, and Ross B. Girshick. Clevr: A diagnostic dataset for compositional language and elementary visual reasoning. 2017.

Kohitij Kar and James J. DiCarlo. Fast recurrent processing via ventrolateral prefrontal cortex is needed by the primate ventral stream for robust core visual object recognition. Neuron, 109(1): 164–176.e5, 2021. ISSN 0896-6273. doi: 10.1016/j.neuron.2020.09.035.

Kohitij Kar, Jonas Kubilius, Kailyn Schmidt, Elias B. Issa, and James J. DiCarlo. Evidence that recurrent circuits are critical to the ventral stream’s execution of core object recognition behavior. Nature Neuroscience, 22(6):974–983, 2019. ISSN 1546-1726. doi: 10.1038/s41593-019-0392-5.

Philip J Kellman and Thomas F Shipley. A theory of visual interpolation in object perception. Cognitive Psychology, 23(2):141–221, 1991. ISSN 0010-0285. doi: 10.1016/0010-0285(91) 90009-d.

Victor A.F. Lamme and Pieter R. Roelfsema. The distinct modes of vision offered by feedforward and recurrent processing. Trends in Neurosciences, 23(11):571–579, 2000. ISSN 0166-2236. doi: 10.1016/s0166-2236(00)01657-x.

Zejun Li, Ruipu Luo, Jiwen Zhang, Minghui Qiu, Xuanjing Huang, and Zhongyu Wei. Vocot: Unleashing visually grounded multi-step reasoning in large multi-modal models. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 3769–3798. Association for Computational Linguistics, 2025. doi: 10.18653/v1/2025.naacl-long.192.

Gordon D. Logan. Spatial attention and the apprehension of spatial relations. Journal of Experimental Psychology: Human Perception and Performance, 20(5):1015–1036, 1994. ISSN 0096-1523. doi: 10.1037/0096-1523.20.5.1015.

Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. Locating and editing factual associations in gpt. arXiv preprint arXiv:2202.05262, 2022. doi: 10.48550/ARXIV.2202.05262.

Clement Neo, Luke Ong, Philip Torr, Mor Geva, David Krueger, and Fazl Barez. Towards interpreting visual information processing in vision-language models. arXiv preprint arXiv:2410.07149, 2024. doi: 10.48550/ARXIV.2410.07149.

Redmond G O’Connell, Paul M Dockree, and Simon P Kelly. A supramodal accumulation-to-bound signal that determines perceptual decisions in humans. Nature Neuroscience, 15(12):1729–1735, 2012. ISSN 1546-1726. doi: 10.1038/nn.3248.

Israfel Salazar, Stella Frank, Dan Oneata, Desmond Elliott, and Constanza Fierro. Pathways of visual information flow in vision-language models. arXiv preprint arXiv:2607.03358, 2026. doi: 10.48550/ARXIV.2607.03358.

Madeline Schiappa, Raiyaan Abdullah, Shehreen Azad, Jared Claypoole, Michael Cogswell, Ajay Divakaran, and Yogesh S. Rawat. Probing conceptual understanding of large visual-language models. 2024.

Allison B. Sekuler and Stephen E. Palmer. Perception of partly occluded objects: A microgenetic analysis. Journal of Experimental Psychology: General, 121(1):95–111, 1992. ISSN 0096-3445. doi: 10.1037/0096-3445.121.1.95.

Hao Shao, Shengju Qian, Han Xiao, Guanglu Song, Zhuofan Zong, Letian Wang, Yu Liu, and Hongsheng Li. Visual cot: Advancing multi-modal language models with a comprehensive dataset and benchmark for chain-of-thought reasoning. arXiv preprint arXiv:2403.16999, 2024. doi: 10.48550/ARXIV.2403.16999.

Simon Thorpe, Denis Fize, and Catherine Marlot. Speed of processing in the human visual system. Nature, 381(6582):520–522, 1996. ISSN 1476-4687. doi: 10.1038/381520a0.

Anne Treisman and Hilary Schmidt. Illusory conjunctions in the perception of objects. Cognitive Psychology, 14(1):107–141, 1982. ISSN 0010-0285. doi: 10.1016/0010-0285(82)90006-8.

Anne M. Treisman and Garry Gelade. A feature-integration theory of attention. Cognitive Psychology, 12(1):97–136, 1980. ISSN 0010-0285. doi: 10.1016/0010-0285(80)90005-5.

x.ai. Grok-1.5 vision preview, 2024.

Yang Xiang, Yixin Ji, Ruotao Xu, Dan Qiao, Zheming Yang, Juntao Li, and Min Zhang. When is thinking enough? early exit via sufficiency assessment for efficient reasoning. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 23541–23556. Association for Computational Linguistics, 2026. doi: 10.18653/v1/ 2026.acl-long.1080.

Fei Xu and Elizabeth S. Spelke. Large number discrimination in 6-month-old infants. Cognition, 74 (1):B1–B11, 2000. ISSN 0010-0277. doi: 10.1016/s0010-0277(99)00066-9.

Chenxu Yang, Qingyi Si, Yongjie Duan, Zheliang Zhu, Chenyu Zhu, Qiaowei Li, Minghui Chen, Zheng Lin, and Weipinng Wang. Dynamic early exit in reasoning models. International Conference on Learning Representations, pages 88170–88210, 2026.

Zhuoran Yu and Yong Jae Lee. How multimodal llms solve image tasks: A lens on visual grounding, task reasoning, and answer decoding. arXiv preprint arXiv:2508.20279, 2025. doi: 10.48550/ ARXIV.2508.20279.

Anqi Zhang, Yulin Chen, Jane Pan, Chen Zhao, Aurojit Panda, Jinyang Li, and He He. Reasoning models know when they’re right: Probing hidden states for self-verification. arXiv preprint arXiv:2504.05419, 2025a. doi: 10.48550/ARXIV.2504.05419.

Zhi Zhang, Srishti Yadav, Fengze Han, and Ekaterina Shutova. Cross-modal information flow in multimodal large language models. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 19781–19791. IEEE, 2025b. doi: 10.1109/cvpr52734.2025.01842.

Zhuosheng Zhang, Aston Zhang, Mu Li, Hai Zhao, George Karypis, and Alex Smola. Multimodal chain-of-thought reasoning in language models. arXiv preprint arXiv:2302.00923, 2023. doi: 10.48550/ARXIV.2302.00923.

## A DATASETS AND MECHANISTIC ANALYSIS

Table 6 summarizes the synthetic datasets and their analysis splits.

Table 6: Synthetic dataset sizes and analysis splits. Counts are images. Primitive splits are used for linear probing, and the Composite split is used for reasoning-trajectory analysis before trace filtering. Behavioral evaluation uses all 1,000 images per primitive and the 200-image Composite test set. Paired images remain in the same split.

<table><tr><td></td><td></td><td></td><td colspan="3">Analysis Split</td><td></td></tr><tr><td>Dataset</td><td>Images</td><td>Choices</td><td>Train</td><td>Validation</td><td>Test</td><td>Split Unit</td></tr><tr><td>Feature Binding</td><td>1,000</td><td>4</td><td>800</td><td>一</td><td>200</td><td>Image</td></tr><tr><td>Numerosity</td><td>1,000</td><td>2</td><td>800</td><td>一</td><td>200</td><td>Image</td></tr><tr><td>Spatial Relations</td><td>1,000</td><td>4</td><td>800</td><td>一</td><td>200</td><td>Image</td></tr><tr><td>Amodal Completion</td><td>1,000</td><td>4</td><td>800</td><td></td><td>200</td><td>Pair</td></tr><tr><td>Composite</td><td>1,000</td><td>2</td><td>640</td><td>160</td><td>200</td><td>Pair</td></tr></table>

## A.1 PRIMITIVE DATASET CONSTRUCTION

We generate each dataset so the feature that determines the answer varies while other scene detail are controlled. The settings for each task follow.

Feature Binding. Feature Binding uses four target colors (red, blue, green, and yellow), four shapes, 4–8 objects per image, and three layout templates. Target shape, size, and position also vary.

Numerosity. Numerosity uses three ratios between the larger and smaller dot counts (2.0, 1.5, and 1.25). For each ratio, total dot area either agrees with the count difference, is similar on both sides, or points in the opposite direction. Dots do not overlap, and each side is equally often the one with more dots.

Spatial Relations. Spatial Relations uses black and gray outlines for the target and reference, with their colors and shapes sampled independently. Its conditions cross four directions, three target–reference distances (60, 90, and 120 pixels), and three scene sizes (4, 6, and 8 objects).

Amodal Completion. Amodal Completion uses six shapes and six kinds of occluder. Each shape is 25–65% hidden but has at least two visible parts. The two images in a pair have different complete shapes. They share the target’s color, center, size, and rotation settings, the occluder, the intended amount of occlusion, and the order of four answer choices. The actual hidden area can differ between the shapes.

Behavioral and probe splits. The VLM stays frozen during direct answering and readout fitting. We balance splits by target color for Feature Binding; by count ratio, area condition, and answer for Numerosity; and by relation, distance, and object count for Spatial Relations. Amodal Completion has 500 matched pairs: 400 for training and 100 for testing, with occluder types balanced. Paired images always stay in the same split.

Counterfactual pairs for primitive interventions. Feature Binding, Numerosity, and Spatial Relations each have a separate intervention set of 125 matched pairs: 100 training and 25 test pairs. Each pair keeps the scene context fixed while changing the feature that determines the answer. Amodal interventions use the 100 held-out Amodal Completion pairs, which change the complete shape under the same occluder. We intervene in both directions and apply the criteria in Section 3.2 to choose pairs for the primary analysis.

## A.2 PRIMITIVE READOUTS AND INTERVENTIONS

This subsection gives the settings for Section 3.2. Direct-answer tests disable thinking, request a short response, and allow at most 40 generated tokens. We accept a normalized choice letter or a matching word answer. Table 7 summarizes the readouts. For standardized training features X and one-hot labels Y, we solve

$$
W = X ^ { \top } ( X X ^ { \top } + 1 0 0 I ) ^ { - 1 } Y ,
$$

and predict the label maximizing zW for a standardized test vector z.

Table 7: Primitive linear-readout settings. Dataset splits are given in Table 6.
<table><tr><td>Setting</td><td>Specification</td></tr><tr><td>Inputs</td><td>No-thinking image and question</td></tr><tr><td>Visual features</td><td>Mean-pooled image-token states at each tested depth</td></tr><tr><td>Question-only control</td><td>Mean-pooled question span, including candidates, with no image</td></tr><tr><td>Standardization</td><td>Per-dimension training mean and standard deviation</td></tr><tr><td>Label classes</td><td>4 for Feature Binding, 2 for Numerosity, 4 for Spatial Relations, and 6 for Amodal Completion</td></tr></table>

Patching protocol. The no-thinking prompt ends in an answer cue, and we score the next-token choice. Each choice score is the maximum logit over the last token IDs of uppercase/lowercase and leading-space variants of its letter. Pairs and controls satisfy the following criteria:

• Both intervention directions are available and both unpatched answers are correct.

• The original donor and receiver margins differ by more than $1 0 ^ { - 6 }$ in both directions.

• Matched-random donors come from the same test set, belong to another pair, and share the receiver’s semantic label. Amodal controls additionally match question-token length.

The lower row of Figure 3 averages the two directional indicator differences within each pair. Table 8 gives the retained sample sizes.

Table 8: Retained counterfactual pairs for primitive interventions.
<table><tr><td>Task</td><td>Qwen3.5-4B</td><td>Qwen3.5-9B</td><td>Gemma-4-E4B</td><td>Gemma-4-12B</td></tr><tr><td>Feature Binding</td><td>24</td><td>25</td><td>22</td><td>25</td></tr><tr><td>Numerosity</td><td>22</td><td>23</td><td>18</td><td>24</td></tr><tr><td>Spatial Relations</td><td>25</td><td>25</td><td>22</td><td>25</td></tr><tr><td>Amodal Completion</td><td>48</td><td>62</td><td>72</td><td>75</td></tr></table>

## A.3 COMPOSITE DATA AND REASONING CHECKPOINTS

The Composite analysis uses scenes that require several component judgments. We also define fixed points where we can cut off a reasoning trace and test whether the model can answer.

Composite dataset. The Composite task has 1,000 images in 500 matched pairs. Each side shows 8–12 partly hidden red or blue shapes; the model must choose the side with more red circles. Within a pair, we change which shapes have which colors to reverse the answer. Positions, shapes, sizes, rotations, occluders, and overall color and shape counts remain fixed. Occlusion ranges from 25% to 60%. We reserve 100 pairs (200 images) for direct-answer testing. The other 400 pairs provide 320 training and 80 validation pairs for reasoning-trace analysis, with validation selecting the analysis layer. We retain a pair only if both traces have a parsed answer and usable checkpoints. Sample counts can therefore differ by model. Table 9 defines the five checkpoints.

Table 9: Reasoning-trajectory checkpoints. T counts reasoning tokens, and positions are one-based.
<table><tr><td>Checkpoint</td><td>State position</td></tr><tr><td>PRE</td><td>Last prompt token before reasoning</td></tr><tr><td>EARLY</td><td>Reasoning token  $\lfloor { T } / 3 \rfloor$ </td></tr><tr><td>MIDDLE</td><td>Reasoning token [2T/3]</td></tr><tr><td>LATE</td><td>Last reasoning token, T</td></tr><tr><td>ANSWER</td><td>Token immediately preceding the first final-choice token</td></tr></table>

Validation selects one layer per model: L17 for Qwen3.5-4B, L16 for Qwen3.5-9B, L1 for Gemma-4- E4B, and L20 for Gemma-4-12B. Test samples do not affect this choice. The selected depth differs across models, so one layer cannot represent the Composite answer equally well in every architecture.

MMStar and RealWorldQA use separate early-stopping splits, described in Appendix B.

## B EARLY-STOPPING DETAILS

## B.1 DATASETS AND CROSS-VALIDATION

We use every MMStar and RealWorldQA question in two-fold cross-validation. Within one fold, we train the detector and select its threshold; we test on the other fold (Table 10). All checkpoints from one question stay together, as do RealWorldQA questions that share an image. Every model uses the same folds.

Table 10: Early-stopping evaluation splits. Counts are questions. Each source fold is divided into detector training and threshold validation, and the other fold is held out for testing.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Evaluation</td><td colspan="2">Source fold</td><td>Held-out fold</td></tr><tr><td>Training</td><td>Validation</td><td>Test</td></tr><tr><td rowspan="2">MMStar</td><td>A→B</td><td>600</td><td>150</td><td>750</td></tr><tr><td>B→A</td><td>600</td><td>150</td><td>750</td></tr><tr><td rowspan="2">RealWorldQA</td><td>A→B</td><td>306</td><td>77</td><td>382</td></tr><tr><td>B→A</td><td>305</td><td>77</td><td>383</td></tr></table>

Each question contributes one held-out prediction per condition. Pooled accuracy weights questions equally across the two evaluation directions.

## C ADDITIONAL EARLY-STOPPING RESULTS

The main text combines the two test folds and analyzes answer completion. Here we report each fold separately.

## C.1 PERFORMANCE ACROSS EVALUATION FOLDS

Tables 11 and 12 give accuracy and mean reasoning length for each test fold. Reasoning length falls in every model and dataset, but accuracy changes differ between folds.

Table 11: Per-direction held-out MMStar results. Each row evaluates 750 held-out cases. Accuracy values are percentages and reasoning-token counts are means. ∆ denotes the change from full thinking to adaptive stopping in percentage points.
<table><tr><td></td><td></td><td colspan="3">Accuracy (%)</td><td colspan="2">Reasoning Tokens</td></tr><tr><td>Model</td><td>Train→Test</td><td>Full</td><td>Stop</td><td>Δ</td><td>Full</td><td>Stop</td></tr><tr><td rowspan="2">Qwen3.5-4B</td><td>A→B</td><td>64.13</td><td>64.13</td><td>0.00</td><td>2,101</td><td>256</td></tr><tr><td>B→A</td><td>65.60</td><td>65.73</td><td>↑0.13</td><td>1,982</td><td>254</td></tr><tr><td rowspan="2">Qwen3.5-9B</td><td>A→B</td><td>66.93</td><td>69.73</td><td>↑2.80</td><td>1,979</td><td>256</td></tr><tr><td>B→A</td><td>66.80</td><td>70.00</td><td>↑3.20</td><td>1,922</td><td>254</td></tr><tr><td rowspan="2">Gemma-4-E4B</td><td>A→B</td><td>57.47</td><td>57.47</td><td>0.00</td><td>767</td><td>437</td></tr><tr><td>B→A</td><td>58.67</td><td>59.87</td><td>↑1.20</td><td>739</td><td>733</td></tr><tr><td rowspan="2">Gemma-4-12B</td><td>A→B</td><td>57.60</td><td>66.67</td><td>↑9.07</td><td>1,822</td><td>327</td></tr><tr><td>B→A</td><td>56.27</td><td>64.93</td><td>↑8.67</td><td>1,885</td><td>246</td></tr></table>

Table 12: Per-direction held-out RealWorldQA results. A→B evaluates 382 questions, and B→A evaluates 383. Accuracy values are percentages and reasoning-token counts are means. ∆ denotes the change from full thinking to adaptive stopping in percentage points.
<table><tr><td colspan="2"></td><td colspan="3">Accuracy (%)</td><td colspan="2">Reasoning Tokens</td></tr><tr><td>Model</td><td>Train→Test</td><td>Full</td><td>Stop</td><td>∆</td><td>Full</td><td>Stop</td></tr><tr><td rowspan="2">Qwen3.5-4B</td><td>A→B</td><td>71.73</td><td>74.87</td><td>↑3.14</td><td>1,692</td><td>247</td></tr><tr><td>B→A</td><td>73.89</td><td>75.98</td><td>↑2.09</td><td>1,700</td><td>322</td></tr><tr><td rowspan="2">Qwen3.5-9B</td><td>A→B</td><td>73.30</td><td>78.27</td><td>↑4.97</td><td>1,521</td><td>252</td></tr><tr><td>B→A</td><td>72.58</td><td>77.55</td><td>↑4.96</td><td>1,692</td><td>244</td></tr><tr><td rowspan="2">Gemma-4-E4B</td><td>A→B</td><td>57.33</td><td>56.81</td><td>↓0.52</td><td>448</td><td>347</td></tr><tr><td>B→A</td><td>59.01</td><td>57.44</td><td>↓1.57</td><td>436</td><td>344</td></tr><tr><td rowspan="2">Gemma-4-12B</td><td>A→B</td><td>59.95</td><td>63.87</td><td>↑3.93</td><td>781</td><td>217</td></tr><tr><td>B→A</td><td>60.31</td><td>69.71</td><td>↑9.40</td><td>936</td><td>369</td></tr></table>