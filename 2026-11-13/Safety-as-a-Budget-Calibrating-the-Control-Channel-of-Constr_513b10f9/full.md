# Safety as a Budget: Calibrating the Control Channel of Constrained GRPO for Vision-Language Models

Bo Han Lanzhou University hanb2025@lzu.edu.cn

(a) Level the budget reads

![](images/eea759156b291538a923beac7799cb2023c90978b007cb5e6ee2712a61593790.jpg)

(b) Multipliers (ε = 0.10, 3 seeds)  
![](images/a900d506aaec4f828a7bee46ba7252cad0d3b0836148eec23a15c61030905461.jpg)

(c) Held-out harm, same scorer  
![](images/b08be56f256336383a1228c8014cf72ac3bda5d621837a6b8e1d0307d45b16c7.jpg)  
Figure 1. One open scorer, one budget $( \varepsilon _ { h } = \varepsilon _ { r } = 0 . 1 0 )$ , the same reward and optimizer; only the control channel differs. (a) On held-out labeled responses the readout overstates refusal by 0.16; the map restores the rate. (b) Raw, the refusal multiplier climbs; calibrated, it stays near zero and the harm multiplier settles. (b, c) Constrained GRPO advantages, three seeds. (c) Held-out harm against standard GRPO without a budget; thin lines: seeds, thick line: mean.

## Abstract

Reinforcement learning can teach vision-language models (VLMs) to decline harmful image-text requests, but how safe the result becomes is set indirectly, through reward weights found by trial runs. Stating safety as budgets on the rate of harmful assistance and the rate of over-refusal, enforced by Lagrange multipliers inside GRPO with an off-the-shelf scorer as the cost, is the natural fix, and itfails twice. The multiplier compares the scorer’s mean with the budget, and scorers builtfor ranking get the mean wrong: afixed-point analysis shows that the mean bias, not the budget, then decides whether the multiplier settles, saturates or switches off. And standard GRPO standardizes the penalized reward within each group, which cancels the multiplier before it reaches the gradient. Two-channel constrained GRPO repairs both: the base reward keeps the raw score for the within-group comparison, the penalty and the constraint estimate pass through a monotone mapfitted on judge-labeled training responses, and the advantage keeps the multiplier. On a Qwen3-VL-2B policy with a training-free readout of an open 4B model as the scorer, held-out harm on 1,037

adversarial promptsfallsfrom 8.9% under standard GRPO to 4.8%, lower in every seed, where the uncalibrated budget gives 7.6%; over-refusal stays at most 2% on 400 benign queries and MMBench is unchanged. The gain is largest where the intent sits in the image $( 1 0 . 1 \%  5 . 1 \% )$ and grows on the unseen HADES benchmark (28.9% → 12.8%). Order-preserving distortions of the scores reproduce each predicted controller state in every seed, and the results hold under a map fitted on another judge’s labels, an imagereading judge and a human audit.

## 1. Introduction

Vision-language models answer requests whose intent may sit in the text, in the image, or in the combination: a typographic card that spells out a procedure, a photo paired with an innocuous instruction [4, 17, 20]. Reinforcement learning from a safety reward is a natural way to teach such a model to decline the harmful cases, and Group Relative Policy Optimization (GRPO) [21] makes it practical when the reward is an external scorer. What it does not provide is a way to say how safe the result should be. The reward mixes a harm term with a helpfulness term, the mix is a weight, and the weight is found by training, evaluating and retrying. The outcome is a point on a safety–helpfulness curve that nobody chose, and over-refusal of benign queries that merely look risky [13] is a frequent side effect.

Deployment requirements are stated differently. An operator asks for a model that assists with at most a few percent of harmful requests and refuses at most a few percent of legitimate ones. These are rates, and constrained optimization handles rates directly [1, 2]: maximize helpfulness subject to E[harm] $\leq \varepsilon _ { h }$ and $\mathbb { E } [ \mathrm { r e f u s a l } ] \leq \varepsilon _ { r } ,$ with a Lagrange multiplier per constraint that rises while its budget is violated and relaxes once it is met. The natural recipe adds such budgets to GRPO with an off-the-shelf scorer as the cost. We show that this recipe fails twice, explain both failures, and repair them.

A scorer enters a constrained run twice. In the learning channel, GRPO compares the responses to one prompt, so the scorer must rank them; in the control channel, the dual update compares the scorer’s mean with the budget, so the mean must be the rate. The first failure is in the control channel. Scorers are built and evaluated for ranking, while the level alone decides where the multiplier settles (Proposition 1): at an interior fixed point a scorer whose mean is off by b steers toward $\varepsilon - b ,$ , and a large enough bias pins the multiplier at its cap or switches the constraint off. A trainingfree readout of an open 4B model [15] overstates refusal by 0.16; its refusal multiplier climbs to 1.5–3.0 although judged over-refusal is 2%, and the budget barely improves on standard GRPO (7.6% against 8.9% held-out harm over three seeds). The second failure is in the learning channel: GRPO standardizes the penalized reward within each group, which cancels the factor $1 + \lambda _ { h }$ , so the multiplier climbs to 3.2–4.7 without moving the policy and the budget does not lower harm (10.2% against 8.9%, three seeds). Twochannel constrained GRPO repairs both. The base reward keeps the raw score, whose fine-grained order drives the within-group comparison; the penalty and the constraint estimate pass through a monotone map fitted on judge-labeled responses to training prompts, which corrects the level they read without flattening that order (200 labels per question bring the bias of an open readout to about 0.01); and the advantage keeps the multiplier: standardizing the summed reward within each group cancels it, so we leave advantages unscaled [19] or standardize each objective before weighting it [3].

With both repairs, and reward, optimizer and budget otherwise unchanged, held-out harm on 1,037 adversarial prompts falls to 4.8%, lower than standard GRPO in every seed (∆ = −4.2 points, 95% CI [−5.1, −3.2]), with overrefusal of at most 2% on 400 benign queries and MMBench unchanged (Fig. 1, Table 1). With unscaled advantages it reaches 6.1% at less than half the KL, matching a dedicated decision model (Jev [23], 5.8%) while running locally. The gain is largest where the intent is hardest to see, on attacks that write it into the image (10.1% → 5.1%; Fig. 4), and grows on the unseen HADES benchmark, whose attacks hide the intent in optimized images (28.9% → 12.8%, Fig. 3).

Order-preserving distortions of the scores reproduce each controller state that Proposition 1 predicts, in every seed; on a calibrated scorer the budget orders held-out harm (12.0%, 7.6%, 6.5% for $\varepsilon _ { h } ~ = ~ 0 . 2 0 , 0 . 1 0 , 0 . 0 5 )$ ; and the results hold under a map fitted on another judge’s labels, an imagereading judge and a human audit.

## Contributions.

• We explain why safety budgets fail inside GRPO: the fixed points of the dual update make the scorer’s mean bias, not the budget, decide the controller state, and group standardization cancels the multiplier; experiments confirm each predicted state in every seed (Secs. 3.2, 3.3, 4.4, 4.6).

• We propose two-channel constrained GRPO, which repairs both failures by keeping raw scores in the base reward, calibrating the level that the penalty and the dual update read, and using an advantage that keeps the multiplier (Sec. 3).

• With a training-free open readout as the scorer, it lowers held-out harm from 8.9% under standard GRPO to 4.8%, lower in every seed, with the largest gains on image-borne attacks and on the unseen HADES benchmark (Secs. 4.2, 4.3).

## 2. Related Work

Safety of Vision-Language Models. Safety tuning for VLMs has relied on curated data: VLGuard [30] fine-tunes on safe and unsafe image-text pairs, and SPA-VL [27] collects preference pairs across harm categories. Adversarial benchmarks place the harmful intent in the image or in the image-text combination [4, 17, 20], and MOSSBench [13] measures the opposite failure, refusals of benign queries that look risky. These lines fix the target when the data are collected; we state it as a pair of budgets and enforce it during training.

Decision Models as Judges and Rewards. Large language model (LLM) judges are the default scorers for open-ended behavior [28], and safety classifiers such as Llama Guard [8] return coarse labels. Decision models return calibrated probabilities over typed questions [23]. Just Ask Jev [6] finds that a single generic question detects ten kinds of alignment failure zero-shot at a median AUROC of 0.886 and a fraction of LLM-judge cost; other studies compare its robustness with LLM judges and classifiers [11, 14, 25], and LLM-as-Jev [15] reads Jev-style probabilities from frozen open models, the readout our open scorer follows. Visual Jev Rewards [12] turns decision-model probabilities into a GRPO reward for multi-subject image generation. These works evaluate scorers by ranking; a budget also reads their level.

Constrained Policy Optimization. Constrained Markov decision processes [1] pair a reward with cost budgets and solve the Lagrangian by primal-dual updates, with PID variants that damp multiplier oscillation [22]. Safe RLHF [2] applies this to helpfulness and harmlessness with learned reward and cost models, and Safe RLHF-V [9] extends it to multimodal models with a cost model trained on multimodal human preferences. For GRPO, Constrained GRPO [3] shows that scalarizing rewards before group standardization couples the multipliers and proposes scalarizing standardized advantages; we use it and the unscaled form, and show that both need a calibrated control channel (Sec. 4.6); CARGO-VL [10] constrains unsupported answering and over-deferral of VLMs with binary costs. Our costs come from an off-theshelf scorer never trained for the task; we show how its mean bias decides the state the dual update reaches and correct it without touching the base reward. Unscaled advantages follow Dr. GRPO [19]; calibration and isotonic recalibration follow [5, 26].

## 3. Method

## 3.1. Safety as a Pair of Budgets

A policy $\pi _ { \theta }$ answers image-text prompts x drawn from a harmful set H and a benign set B; which set a prompt belongs to comes from the data. For a response $y ,$ a scorer returns three probabilities: $p _ { h } ( x , y )$ that y gives substantive, actionable help with the harmful request; $p _ { r } ( x , y )$ that y declines or deflects the request; and $p _ { f } ( x , y )$ that y usefully addresses it. The scorer judges the response only; the harmfulness of the prompt is never asked, because decision models over-flag benign requests [14] and a reward that read the prompt would push toward refusing everything.

The base reward is

$$
r _ { 0 } ( x , y ) = \left\{ \begin{array} { l l } { 1 - p _ { h } } & { x \in \mathcal { H } , } \\ { w p _ { f } + ( 1 - w ) ( 1 - p _ { r } ) } & { x \in \mathcal { B } , } \end{array} \right.\tag{1}
$$

with $w = 0 . 6 5$ throughout. We maximize it subject to two budgets,

$$
\begin{array} { r } { \underset { \theta } { \operatorname* { m a x } } \ \mathbb { E } [ r _ { 0 } ] \quad \mathrm { s . t . } \quad \underbrace { { \mathbb { E } } _ { x \in \mathcal { H } } [ p _ { h } ] } _ { g _ { h } ( \theta ) } \leq \varepsilon _ { h } , \qquad \underbrace { { \mathbb { E } } _ { x \in { \cal B } } [ p _ { r } ] } _ { g _ { r } ( \theta ) } \leq \varepsilon _ { r } , } \end{array}\tag{2}
$$

and solve the Lagrangian $\mathcal { L } = \mathbb { E } [ r _ { 0 } ] - \lambda _ { h } ( g _ { h } - \varepsilon _ { h } ) - \lambda _ { r } ( g _ { r } -$ $\textstyle \varepsilon _ { r } )$ by alternating a GRPO step on θ with projected dual ascent on $\lambda = ( \lambda _ { h } , \lambda _ { r } )$ . The policy step uses the per-response reward

$$
r ( x , y ) = r _ { 0 } ( x , y ) - \lambda _ { h } \tilde { p } _ { h } \mathbb { k } \lvert \mathcal { k } \rvert _ { \vphantom { \frac { 1 } { \lambda } } } x \in \mathcal { H } ] - \lambda _ { r } \tilde { p } _ { r } \mathbb { k } \lvert \mathcal { k } \rvert _ { \vphantom { \frac { 1 } { \lambda } } } \mathrm { [ } x \in \mathcal { B } \big ] ,\tag{3}
$$

and after every optimization step each multiplier moves by

$$
\lambda _ { k } \gets \mathrm { c l i p } \big ( \lambda _ { k } + \eta \big ( \hat { g } _ { k } - \varepsilon _ { k } \big ) , 0 , \lambda _ { \operatorname* { m a x } } \big ) ,\tag{4}
$$

where $\hat { g } _ { k }$ is an exponential moving average (EMA, coefficient 0.3) of the batch mean of $\tilde { p } _ { k } = c _ { k } ( p _ { k } )$ , the probability after the map of Sec. $3 . 2 ~ ( c _ { k }$ is the identity without calibration), $\eta \ : = \ : 0 . 5$ and $\lambda _ { \operatorname* { m a x } } = 1 0$ . The penalty is this probability itself rather than a thresholded indicator, so the constraint and the gradient see the same quantity. The budget acts on the rate the scorer reports on training prompts; Sec. 4 measures held-out rates with an independent judge.

## 3.2. Two Channels: Ranking for the Gradient, Level for the Budget

The scorer enters training twice. Within a group, the reward of Eq. 3 compares responses, so what matters there is ranking: whether a more harmful response receives a higher $p _ { h }$ . In the dual update of $\mathrm { E q . ~ } 4 , \hat { g } _ { k }$ is compared with $\varepsilon _ { k } .$ , so what matters there is level: whether the mean of $p _ { k }$ is the rate of the event. The two properties are independent, and the second decides what a budget means. Let $h ( \pi ) = \mathbb { E } _ { \pi } [ z ]$ be the rate of the event under reference labels $z , \hat { h } ( \pi ) = \mathbb { E } _ { \pi } [ p ]$ the scorer’s mean on the same responses, and $b ( \pi ) = \hat { h } ( \pi ) - h ( \pi )$ its mean bias.

Proposition 1. Ignoring sampling and smoothing noise, a fixed point $( \pi ^ { * } , \lambda ^ { * } )$ of Eq. 4 is in one of three states: (i) interior, $0 < \lambda ^ { * } < \lambda _ { \operatorname* { m a x } }$ and $h ( \pi ^ { * } ) = \varepsilon - b ( \pi ^ { * } )$ ; (ii) inactive, $\lambda ^ { * } = 0$ and $h ( \pi ^ { * } ) \leq \varepsilon - b ( \pi ^ { * } )$ ; (iii) saturated, $\lambda ^ { * } = \lambda _ { \operatorname* { m a x } }$ and $h ( \pi ^ { * } ) \geq \varepsilon - b ( \pi ^ { * } )$   
Proof. A fixed point of the projected update satisfies $\hat { h } = \varepsilon$ in the interior, $\hat { h } \le \varepsilon$ at 0 and $\hat { h } \ge \varepsilon \mathrm { a t } \lambda _ { \mathrm { m a x } } ;$ substitute $\hat { h } = h + b . \bigtriangledown$

The proposition turns calibration into testable predictions. With $b \approx 0$ the controller settles in state (i) and ε is the rate it enforces. A bias that pushes $\varepsilon - b$ below every attainable rate leaves the constraint violated and drives λ to the cap, state (iii), where $\lambda _ { \mathrm { m a x } }$ rather than ε sets the outcome. A negative bias with $\varepsilon - b$ above the rate the policy already has satisfies the constraint from the start and switches it off, state (ii). None of this depends on ranking: a monotone transform such as $p \mapsto p ^ { \gamma }$ leaves every AUROC unchanged and moves b at will, which is how Sec. 4.4 tests the three states.

Calibrate the Control Channel, Keep the Learning Channel. The proposition also says where a correction belongs. We pass the constraint estimate and the penalty through a monotone map $c _ { k } , \hat { g } _ { k } = \mathrm { E M A } \big ( \overline { { c _ { k } ( p _ { k } ) } } \big )$ and $\lambda _ { k } c _ { k } ( p _ { k } )$ in Eq. 3, and keep the raw probability in the base reward $r _ { 0 }$ The map is fitted on labels from the evaluation judge for responses sampled along an unconstrained run on the training prompts; the trajectory, rather than the initial policy alone, covers the responses the constrained run will produce, and none of these prompts is evaluated. An isotonic map [26] is piecewise constant, so inside $r _ { 0 }$ it would give distinct responses identical scores and erase the within-group comparison that the raw score carries; in the penalty and the constraint it sets the level the multiplier acts on (Sec. 4.4 tests both placements). The same labels also tell a practitioner in advance what a budget will do: the bias b of the raw scorer on them predicts which of the three states a given budget will reach, because the refusal bias stays put along a run (0.14–0.18 in every quarter), while the harm bias moves with the policy, so the map is fitted along a trajectory.

![](images/5b1b75843bf68d41d7c671b4805f330d5d17a53a131a02bb5322f307a6704340.jpg)  
Figure 2. Two-channel constrained GRPO. (a) A scorer reads the request and the response. (b) Raw probabilities set the base reward $r _ { 0 } ;$ the penalty uses the mapped level c(p), and unscaled advantages let the multipliers scale the gradient. (c) The map sets the level the dual update compares with the budgets

## 3.3. The Multiplier Must Reach the Gradient

GRPO computes, for the G responses to one prompt, $A _ { i } =$ $( r _ { i } - { \bar { r } } ) / s$ . On a harmful prompt without a map, Eq. 3 gives $r _ { i } = 1 - ( 1 + \lambda _ { h } ) p _ { h , i } , { \mathrm { s o } } A _ { i } = - ( 1 + \lambda _ { h } ) ( p _ { h , i } - \bar { p } _ { h } ) / s$ . With the usual group scale s = std(r) the factor $\left( 1 + \lambda _ { h } \right)$ cancels: every harmful group contributes unit-scale advantages whatever the multiplier says. With a batch scale, a larger $\lambda _ { h }$ inflates s and shrinks the advantages of benign prompts instead of strengthening those of harmful ones. We set $s = 1$ as in Dr. GRPO [19], so that the multiplier scales the harmfulprompt gradient by $1 + \lambda _ { h }$ and the benign-prompt gradient through $\lambda _ { r } ,$ which is what the Lagrangian prescribes. Constrained GRPO [3] keeps the multiplier another way, standardizing each objective within its group before weighting it, $A _ { i } = z ( r _ { 0 } ) _ { i } - \lambda _ { h } z ( \tilde { p } _ { h } ) _ { \tilde { s } }$ . Sec. 4.6 tests the four forms.

## 3.4. Scorers

The state passed to the decision model is the user request followed by the response; the policy sees the image, the reward reads the response text. Assistance is delivered in the response, so whether a response gives actionable help or refuses is decidable from its text, whichever modality carried the request. The human audit of Sec. 4.7, whose annotator sees the image, supports this choice: the text judge agrees with the annotator more closely $( \kappa = 0 . 8 5 )$ than an imagereading judge does (0.71). Each question is a binary question with an explicit statement of what true and false mean, which the decision-model interface requires for the probability to be meaningful. Harmful prompts are asked only $p _ { h }$ , benign prompts only $p _ { r }$ and $p _ { f }$ . We instantiate the method with two scorers that answer the same three questions: the decision model Jev 1.13 (typesafe/jev-1.13), and a training-free readout of a frozen Qwen3.5-4B that runs locally, following LLM-as-Jev [15]: the options are numbered, the prompt ends with an answer prefix, and each option’s probability is the softmax over the summed log-probabilities of its identifier.

## 4. Experiments

## 4.1. Setup

Policy and Training. We fine-tune Qwen3-VL-2B-Instruct with GRPO and LoRA [7] (rank 16, all linear layers, learning rate $5 \times 1 0 ^ { - 5 }$ , KL coefficient 0.02), 4 samples per prompt, 8 prompts per optimization step and 80 steps, with unscaled advantages unless stated otherwise; standard GRPO, which standardizes the rewards within each group, is the baseline. A system prompt asks for answers under 120 words, which keeps completions inside the 384-token limit.

Training Prompts. Harmful prompts are mined for difficulty: from 598 MM-SafetyBench [17] candidates spread over its thirteen categories, we keep the 130 on which four samples of the base model disagree under the evaluation judge (at least one harmful, at least one not), since only such prompts produce a non-zero group advantage, and train on 120 of them. The mining step excludes every evaluation prompt and its underlying harmful question. Benign prompts are 200 MOSSBench [13] queries; the remaining third of MOSSBench, taken as every third item in a type-stratified order, is held out.

<table><tr><td rowspan="2">Scorer</td><td rowspan="2">Control channel</td><td colspan="2">Held-out (%)</td><td>Train labels</td><td>Training (end)</td></tr><tr><td>Harm</td><td>Over-ref.</td><td> $b _ { h } \ : / \ : b _ { r }$ </td><td> $\lambda _ { h } / \lambda _ { r }$ </td></tr><tr><td>Open readout [15]</td><td>no budget</td><td>13.3</td><td>2.0</td><td></td><td>1</td></tr><tr><td>Open readout</td><td>raw</td><td>12.7</td><td>2.0</td><td> $- 0 . 0 2 / + 0 . 1 6$ </td><td> $3 . 0 9 / 2 . 8 0$ </td></tr><tr><td>Open readout</td><td>calibrated</td><td>7.1</td><td>4.0</td><td> $+ 0 . 0 1 / + 0 . 0 1$ </td><td>2.09 / 0.00</td></tr><tr><td>Open readout</td><td>raw, no εr</td><td>9.0</td><td>3.0</td><td></td><td> $- / -$ </td></tr><tr><td>Open readout</td><td> $\mathrm { r a w } , \varepsilon _ { r } = 0 . 2 6$ </td><td>9.5</td><td>3.0</td><td></td><td> $2 . 4 4 / 0 . 0 0$ </td></tr><tr><td>Open readout</td><td> $\mathrm { f i x e d } \lambda _ { h } = 2$ </td><td>9.7</td><td>4.0</td><td></td><td>2/0</td></tr><tr><td>Jev [23]</td><td>raw</td><td>7.6</td><td>3.0</td><td> $- 0 . 0 3 / + 0 . 0 6$ </td><td>1.42 / 0.45</td></tr><tr><td colspan="6">Three seeds, mean; open readout unless noted</td></tr><tr><td>Standard GRPO [21] no budget</td><td></td><td>8.9</td><td>1.3</td><td></td><td></td></tr><tr><td>CGRPO [3]</td><td>raw</td><td>7.6</td><td>2.7</td><td></td><td>2.10 / 2.42</td></tr><tr><td>CGRPO</td><td>calibrated</td><td>4.8</td><td>4.0</td><td></td><td>1.92 / 0.00</td></tr><tr><td>Unscaled</td><td>no budget</td><td>13.1</td><td>2.3</td><td></td><td></td></tr><tr><td>Unscaled</td><td>raw</td><td>11.9</td><td>1.3</td><td></td><td>2.96 / 2.90</td></tr><tr><td>Unscaled</td><td>calibrated</td><td>6.1</td><td>4.0</td><td></td><td>2.48 / 0.01</td></tr><tr><td>Unscaled</td><td>Llama-fitted map</td><td>6.6</td><td>5.0</td><td></td><td>2.34 / 0.00</td></tr><tr><td>Unscaled, Jev</td><td>raw</td><td>7.5</td><td>2.7</td><td></td><td>1.83 / 0.33</td></tr></table>

Table 1. One budget $( \varepsilon _ { h } = \varepsilon _ { r } = 0 . 1 0 )$ . Top: unscaled advantages, seed 0. Bottom: three seeds; standard GRPO standardizes the reward within each group, CGRPO standardizes each objective before weighting it. Calibrated: the penalty and the constraint estimate pass through an isotonic map; the base reward keeps the raw score. b: mean bias against judge labels of training responses, whole run (calibrated row: held-out half). λ: last 15 steps.

Held-Out Evaluation. Harm is measured on 1,037 adversarial prompts: MM-SafetyBench (299, sampled across categories), FigStep [4] (300), JailbreakV-28K [20] (194, stratified across attack formats) and the contrast set of MOSS-Bench (244), whose items add explicit malice to MOSS-Bench scenarios, so refusing is correct. Over-refusal is measured on the 100 held-out benign MOSSBench queries. Responses are decoded greedily. General capability is checked on 300 MMBench [18] questions. An independent LLM judge (Qwen3.8-Max) labels each response refusal, safe answer or harmful; it is never a reward. Items blocked by the judge provider’s content filter (11–12 of 1,137 per model) are excluded, which leaves 1,025–1,026 judged adversarial prompts per model; they are the same prompts in every model, and counting them as harmful for the calibrated readout and as safe for the raw one leaves every comparison unchanged. All models answer the same items, so comparisons between two trained models are paired, and we report discordant counts with exact McNemar tests; across training seeds we report every seed and claim an effect only when it holds in each.

## 4.2. An Open Scorer Becomes a Budget Controller

A training-free readout of an open 4B model ranks harmful responses almost as well as Jev (AUROC 0.84–0.90 on labeled training responses) but misreads the level of refusal:

(a) Where the intent sits  
(b) HADES (unseen)  
![](images/2b46d725a1ba8784f4ae2eaba9357f72398999f6eceddf2c9494d117b12d47a7.jpg)

![](images/bade4c56df0f7c4fa6b7f53f08b7e4b4257af53ae168261412ba7d7f41fcc7f1.jpg)  
Figure 3. Held-out harm of the open readout, standard GRPO against the $\varepsilon _ { h } = \varepsilon _ { r } = 0 . 1 0$ budget; bars: three-seed mean, dots: seeds. (a) By where the intent sits. Image: MM-SafetyBench, JailbreakV image formats $( n ~ = ~ 4 4 1 )$ ; text: FigStep (question stated in the request), JailbreakV text formats (341); image + text: MOSSBench contrast (244). (b) HADES, unseen in training and calibration; outlined: base model. Raw and calibrated: CGRPO advantages.

on held-out benign training responses it reports a mean refusal probability of 0.18 where the labeled rate is 0.02 $( b _ { r } = + 0 . 1 6$ , Table 1; Fig. 1a). Since $\varepsilon _ { r } - b _ { r } < 0$ lies below every attainable rate, Proposition 1 predicts a saturating refusal constraint, and the run shows it: $\lambda _ { r }$ rises to 2.8 by step 80 although judged over-refusal is 2%. Under this spurious pressure against refusing, the harm budget barely moves the policy: with unscaled advantages it ends at 11.9% against 13.1% without a budget, and with CGRPO advantages at 7.6% against 8.9% for standard GRPO, three seeds each, with $\lambda _ { r }$ rising to 1.5–3.0 in every run.

Calibrating the control channel removes the failure. We fit an isotonic map to the readout’s own scores of the same labeled training responses used for Jev, apply it to the penalty and the constraint estimate, and leave the base reward, optimizer and budget unchanged. The map brings both biases to about 0.01 on held-out labels and holds along the run it controls: on judge labels of every training response of a calibrated run, the mapped estimates stay within 0.03 of the labeled rates from step 20 on, and the labeled harm rate of its last 20 steps is 3.2%, inside the budget (seed 2). The refusal constraint now stays inactive and the harm multiplier settles in the interior. With CGRPO advantages, held-out harm falls to 4.8%, lower than standard GRPO in every seed (168 harmful responses removed, 40 added; $\Delta = - 4 . 2$ points, 95% CI [−5.1, −3.2]) and on all four benchmarks, and lower than the raw budget in every seed (124 against 38). With unscaled advantages it falls from 11.9% to 6.1% $( \Delta = - 5 . 8 ,$ , CI [−7.2, −4.4]), 2.8 points below standard GRPO ([−3.8, −1.9]) at a KL of 0.02–0.04 against 0.04– 0.07, matching a dedicated decision model with the same control channel (5.8%, Sec. 4.4). Over-refusal stays at 1–2% on 400 benign queries and MMBench at 78.0–80.7%.

![](images/91eb65fee4551129436bc324f8c0d3eb303681f027eca39ca2ea6151b17c6b81.jpg)  
Figure 4. Held-out prompts answered by the open-readout policy under the raw and the calibrated control channel $( \varepsilon _ { h } = \varepsilon _ { r } = 0 . 1 0 $ seed 0); text-judge labels, confirmed by the image-reading judge. [masked]: actionable content removed.

Calibration Beats Repairing the Budget by Hand. Both channels contribute: calibrating refusal alone gives 8.1%, harm alone 8.7% (it corrects an early underestimate, $b _ { h } =$ −0.24), and both 7.1%. The repairs a practitioner could try instead, each of which requires knowing which constraint is off, do worse: dropping the refusal budget, loosening it to a feasible $\varepsilon _ { r } = 0 . 2 6$ or fixing $\lambda _ { h } = 2$ gives 9.0%, 9.5% and 9.7% $( p \leq 0 . 0 1 4$ against calibration), and over three seeds dropping the refusal budget ends at 9.1% against 6.1%, lower in every seed $( \Delta = - 3 . 0$ points, 95% CI [−4.0, −2.0]). A fixed weight also does not travel between scorers: $\lambda _ { h } = 2$ matches the budget on Jev (8.1%) but not on the open readout (9.7%). The two scorers thus differ only in the control channel: they end at similar harm without a constraint (p = 0.22), Jev wins under a raw one because its level is already close to the rates, and the gap closes under a calibrated one.

## 4.3. Where the Gain Lands

The gain is largest where the intent is hardest to see (Fig. 3a). On the 441 attacks that write the activity into the picture, the request alone reads “list steps in detail to perform this activity”, so a safety signal on the request would see nothing. The calibrated budget lowers harm on them from 10.1% under standard GRPO to 5.1% over three seeds (80 removed, 15 added), where the raw budget gives 9.1%, and from 18.1% to 10.1% on the image formats of JailbreakV alone, outside the benchmark the training prompts come from. The scorer needs no image because a complying policy names what it read: in the unconstrained Jev run, 32 of 59 harmful responses to image-borne attacks state the depicted activity in their first sentences (Fig. 4, top).

<table><tr><td rowspan="2">Probabilities seen by the constraint</td><td colspan="2">Train labels</td><td>Training</td><td colspan="3">Held-out (%)</td></tr><tr><td>mean</td><td>ECE</td><td> $\lambda _ { h } / \lambda _ { r }$ </td><td></td><td>Harm 3 seeds</td><td>Over-ref.</td></tr><tr><td>Overestimating,  $p ^ { 0 . 3 }$ </td><td>0.46</td><td>0.27</td><td>10.0/ 10.0</td><td>6.6</td><td>7.1</td><td>5.0</td></tr><tr><td>Jev as is</td><td>0.16</td><td>0.04</td><td>1.42 / 0.45</td><td>7.6</td><td>7.5</td><td>3.0</td></tr><tr><td>Isotonic map</td><td>0.25†</td><td>0.05†</td><td>1.84 / 0.00</td><td>5.7</td><td>5.8</td><td>4.0</td></tr><tr><td>Underestimating,  $p ^ { 3 }$ </td><td>0.05</td><td>0.14</td><td>0.00 / 0.00</td><td>12.2</td><td>12.2</td><td>3.0</td></tr></table>

Table 2. Order-preserving transforms of the Jev probabilities in the penalty and the constraint estimate only, base reward unchanged $( \varepsilon _ { h } = \varepsilon _ { r } = 0 . 1 0 $ ; AUROC unchanged). Train labels: mean and expected calibration error (ECE) of transformed $p _ { h }$ against judge labels (<sup>†</sup>held-out half). λ: seed 0; every row reaches the same state in all three seeds.

The controller carries over to benchmarks that play no part in training or calibration (Fig. 3b). On HADES [16], which hides the intent in optimized images, harm falls from 28.9% under standard GRPO to 12.8%, lower in every seed (131 removed, 10 added; $\Delta ~ = ~ - 1 6 . 1$ points, 95% CI $[ - 1 9 . 3 , - 1 3 . 2 ] )$ , where the raw budget gives 23.6%; the base model is at 48.4%. With unscaled advantages calibration gives 17.5%. On SIUO [24], where only the image–text combination is unsafe, calibration lowers harm under unscaled advantages from 19.6% to 14.5% under the imagereading judge $( p < 1 0 ^ { - 3 } )$ . Over-refusal on the 300 safe situations of MSSBench [29] stays below 1%.

## 4.4. The Level Decides the Controller State

The first failure is a level effect, and order-preserving transforms isolate it: they test the predictions of Sec. 3.2 while holding ranking fixed (Table 2; trajectories in Fig. 5). With probabilities inflated by $p \mapsto p ^ { 0 . 3 }$ the constraint sees rates far above both budgets for the whole run, and both multipliers climb monotonically to their cap of 10, in each of three seeds. The outcome is then set by $\lambda _ { \mathrm { m a x } }$ rather than by ε: over three seeds the policy lands where the cap puts it (7.1%, against 7.5% for the unmodified reward). With probabilities deflated by $p \mapsto p ^ { 3 }$ the constraint is satisfied from the first steps; $\lambda _ { h }$ peaks below 0.6 and returns to zero, and the constraint switches itself off in every seed. Held-out harm is 12.1–12.4%, no lower than without a budget in any seed, with 191 harmful responses added against 45 removed relative to Jev as is over three seeds $( p = 1 0 ^ { - 2 2 } )$ .

A mild shift of the level confirms that the level, not the order, is what the budget acts on. An isotonic map fitted on half of the labeled training responses raises Jev’s harm level (0.25 against a judged 0.21 on the other half); applied to the penalty and the constraint estimate only, it raises the pressure $( \lambda _ { h } = 1 . 8 4$ against 1.42) and lowers held-out harm to 5.7% (33 against 13 discordant, $p = 0 . 0 0 4 5 )$ ; over three seeds it settles in the interior every time and averages 5.8% against 7.5%, lower in two seeds and level in the third (5.2% against 5.1%). For refusal it lowers the level (0.09 → 0.02)

![](images/ca5c2ddb3747be17f8fa5b908c72e0046280c2adbea45d185c973a78cc8252ce.jpg)

![](images/b6a60c0edda18abcc4e6e87e196452cc7ecf9b87e5d0326b560173b3c3c68410.jpg)  
Figure 5. The three states of Proposition 1 $( \varepsilon _ { h } = 0 . 1 0 $ , seed 0): overestimating saturates $\lambda _ { h } .$ , underestimating switches it off, and calibrated probabilities let it settle in the interior.

<table><tr><td></td><td colspan="2">Held-out (%)</td><td colspan="2">Training (end)</td><td colspan="2">vs. no budget</td></tr><tr><td>Model</td><td></td><td>Harm ↓ Over-ref. ↓</td><td> $\hat { g } _ { h }$ </td><td> $\lambda _ { h }$ </td><td> $- / +$ </td><td>∆ harm, 95% CI</td></tr><tr><td>Base model</td><td>17.1</td><td>3.0</td><td>一</td><td>一</td><td>1</td><td>一</td></tr><tr><td>No budget</td><td>12.2</td><td>4.0</td><td>0.23</td><td>0</td><td>一</td><td>一</td></tr><tr><td>εh = 0.20</td><td>12.0</td><td>2.0</td><td>0.15</td><td>0.00</td><td>29/27</td><td> $- 0 . 2 \left[ - 1 . 6 , + 1 . 2 \right]$ </td></tr><tr><td> $\varepsilon _ { h } = 0 . 1 0$ </td><td>7.6</td><td>3.0</td><td>0.12</td><td>1.42</td><td>59/12</td><td> $- 4 . 6 \left[ - 6 . 2 , - 3 . 0 \right]$ </td></tr><tr><td> $\varepsilon _ { h } = 0 . 0 5$ </td><td>6.5</td><td>3.0</td><td>0.09</td><td>3.78</td><td>66/8</td><td> $- 5 . 7 \ [ - 7 . 3 , - 4 . 1 ]$ </td></tr><tr><td>Fixed weight,  $\lambda _ { h } = 2$ </td><td>8.1</td><td>5.0</td><td>0.09</td><td>2 (fixed)</td><td>63/21</td><td> $- 4 . 1 \left[ - 5 . 9 , - 2 . 4 \right]$ </td></tr><tr><td>Fixed weight,  $\lambda _ { h } = 1 . 4 2$ </td><td>12.0</td><td>4.0</td><td>一</td><td>1.42 (fixed)</td><td>31/29</td><td> $- 0 . 2 \ : [ - 1 . 7 , + 1 . 3 ]$ </td></tr><tr><td>Three seeds, mean</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>No budget</td><td>11.0</td><td>3.0</td><td></td><td>0</td><td></td><td></td></tr><tr><td> $\varepsilon _ { h } = 0 . 1 0$ </td><td>7.5</td><td>2.7</td><td>0.10</td><td>1.83</td><td>160/52</td><td> $- 3 . 5 \left[ - 4 . 6 , - 2 . 5 \right]$ </td></tr></table>

Table 3. Harm budgets on the Jev reward $( \varepsilon _ { r } = 0 . 1 0 )$ . Training: constraint estimate and $\lambda _ { h }$ over the last 15 steps. Fixed weight: constant $\lambda _ { h } = 2 , \lambda _ { r } = 0 . 5 . - / { + }$ : harmful responses removed/added against the run without a budget; bold: $p < 1 0 ^ { - 7 }$ (McNemar). ∆: paired difference with a 95% prompt-bootstrap interval.

and $\lambda _ { r }$ stays at zero.

The map belongs in the penalty and the constraint only: fed into the base reward as well, it flattens the within-group comparison $( \lambda _ { h } \ 0 . 7 8$ instead of 1.84; harm 7.0% against 5.7%, p = 0.029). A smooth Platt map on the open readout points the same way (7.6% in the control channel only, 8.7% in both); over the two pairs, separating the channels removes 58 harmful responses and adds 33 $( p = 0 . 0 1 1 )$ .

## 4.5. The Budget Is a Control Knob

Table 3 shows the budget acting as a control. Tightening $\varepsilon _ { h }$ from 0.20 to 0.10 to 0.05 lowers held-out harm from 12.0% to 7.6% to 6.5%, with the training estimate in the same order (0.15, 0.12, 0.09); against the run without a budget the two tighter budgets remove 59 and 66 harmful responses and add 12 and 8 $( p < 1 0 ^ { - 7 } )$ . The ordering holds on each of the four benchmarks.

The multiplier behaves as the dual update prescribes. Under the loose budget $\lambda _ { h }$ peaks at 0.98, the constraint is met, and $\lambda _ { h }$ is back at zero from step 53, after which the run matches the run without a budget (29 against 27 discordant). Under $\varepsilon _ { h } = 0 . 1 0$ the multiplier rises to 2.0 and relaxes to 1.4 as the estimate approaches the budget; under $\varepsilon _ { h } = 0 . 0 5$ it keeps rising, and the tightest budget is the one still being enforced at the end of training. In judged terms on heldout data the two looser budgets are met $( 1 2 . 0 \% \leq 2 0 \%$ $7 . 6 \% \leq 1 0 \% )$ ; the 0.05 budget has pulled the training estimate to 0.09 with its multiplier still rising at step 80, so at 80 steps it sets the operating point of the run without having been reached yet.

The gain is not paid in refusals of benign queries: overrefusal stays at 2–4% for every budget, while refusals of harmful requests rise from 40.9% to 51.3% and 53.9%. MMBench accuracy is 79.0% for the base model and 79.0– 81.3% for every trained model, none of which refuses an MMBench question.

The Budget Finds the Weight. A fixed penalty can reach the same point if its weight is right, and the dual update tells us which weight that is. Holding $\lambda _ { h }$ at the peak of the $\varepsilon _ { h } = 0 . 1 0$ run $( \lambda _ { h } = 2 , \lambda _ { r } = 0 . 5 )$ gives 8.1% held-out harm, statistically indistinguishable from the budget it was read from (21 against 26 discordant, $p = 0 . 5 6 ) $ ; holding it where the run settles (1.42) leaves harm at 12.0%, no lower than without a budget, so only the dual trajectory reveals which weight works. The weight was an output of the constrained run, whereas the budget was its only input. Moving the budget also moves the point past this weight: $\varepsilon _ { h } = 0 . 0 5$ ends below the fixed weight $( 6 . 5 \% , p = 0 . 0 2 6 )$ and $\varepsilon _ { h } = 0 . 2 0$ switches its own penalty off once met, which a constant weight cannot do.

Across Seeds. Over three seeds the budget lowers heldout harm by 4.6 and 5.7 points in two seeds and leaves the third level $( 1 2 . 2  7 . 6 , 1 0 . 8  5 . 1 , 1 0 . 0  9 . 8 )$ , by 3.5 points on average (prompt-bootstrap 95% CI $[ - 4 . 6 , - 2 . 5 ]$ 160 harmful responses removed against 52 added), with overrefusal at 2–3%. The size of the effect follows the pressure the dual update builds up: early harm estimates of 0.20, 0.26 and 0.36 give mean multipliers of 1.33, 1.60 and 2.36 and reductions of 0.2, 4.6 and 5.7 points, in the same order.

## 4.6. The Advantage Must Keep the Multiplier

The second failure is in the learning channel. Table 4 trains the open readout with four advantage forms under the same budget. Standardizing the summed reward within each group, as standard GRPO does, cancels the multiplier as Sec. 3.3 predicts: under the calibrated budget $\lambda _ { h }$ climbs to 3.2–4.7 while the estimate stays at 0.15–0.21, and held-out harm is no lower than without a budget in any seed (10.2% against 8.9%, three seeds). Batch standardization keeps the multiplier only nominally: no budget, raw and calibrated control end at 8.3%, 7.7% and 7.5% $( p \ge 0 . 3 7 )$ , and on Jev the budget no longer orders the outcome. Two forms let the multiplier act. Unscaled advantages scale the harmful-prompt gradient by $1 + \lambda _ { h }$ and give 7.1% at the lowest KL; standardizing each objective before weighting it (CGRPO) gives 5.0%, and 4.8% against 6.1% over three seeds at about twice the KL. In both, calibration makes the budget work: under raw control they end at 12.7% and 7.4%.

<table><tr><td rowspan="2">Advantage</td><td colspan="3">Held-out harm (%)</td><td rowspan="2">KL (cal.)</td></tr><tr><td>No budget</td><td>Raw</td><td>Calibrated</td></tr><tr><td>Group-std. (GRPO) [21]</td><td>8.0</td><td></td><td>10.6</td><td>0.037</td></tr><tr><td>Batch-std.</td><td>8.3</td><td>7.7</td><td>7.5</td><td>0.032</td></tr><tr><td>Unscaled  $( s = 1 ) [ 1 9 ]$ </td><td>13.3</td><td>12.7</td><td>7.1</td><td>0.024</td></tr><tr><td>Per objective (CGRPO) [3]</td><td>8.0*</td><td>7.4</td><td>5.0</td><td>0.048</td></tr></table>

Table 4. Advantage forms with the open readout under $\varepsilon _ { h } ~ =$ $\varepsilon _ { r } = 0 . 1 0$ (seed 0): held-out harm, and KL to the reference of the calibrated run (last 10 steps). <sup>∗</sup>Without a budget CGRPO reduces to standard GRPO.
<table><tr><td rowspan="2">Model</td><td colspan="2">Harm (%), image-reading judge</td><td rowspan="2">Text judge harm (%)</td></tr><tr><td>rate [95% CI]</td><td>vs. reference, -/+</td></tr><tr><td>No budget</td><td>21.0 [17.8, 24.5]</td><td></td><td>12.2</td></tr><tr><td> $\varepsilon _ { h } = 0 . 2 0$ </td><td>19.7 [16.2, 23.6]</td><td>50/37</td><td>12.0</td></tr><tr><td> $\varepsilon _ { h } = 0 . 1 0$ </td><td>15.0 [11.8, 19.0]</td><td>89/28</td><td>7.6</td></tr><tr><td> $\varepsilon _ { h } = 0 . 0 5$ </td><td>13.7 [10.6, 17.7]</td><td>93/19</td><td>6.5</td></tr><tr><td>Isotonic map,  $\varepsilon _ { h } = 0 . 1 0$ </td><td>11.8 [8.7, 15.9]</td><td>55/22*</td><td>5.7</td></tr><tr><td>Underestimating,  $p ^ { 3 }$ </td><td>22.1 [18.6, 25.8]</td><td>29/101*</td><td>12.2</td></tr><tr><td>Open readout, no budget</td><td>23.2 [19.5, 27.6]</td><td></td><td>13.3</td></tr><tr><td>Open readout,  $\varepsilon _ { h } = 0 . 1 0$ </td><td>20.8 [17.1, 25.1]</td><td>69/44</td><td>12.7</td></tr><tr><td>calibrated control</td><td>12.6 [9.3, 16.8]</td><td>105/22</td><td>7.1</td></tr><tr><td>Jev vs. open,  $\varepsilon _ { h } = 0 . 1 0$ </td><td></td><td>88/29</td><td>一</td></tr></table>

Table 5. Image-reading judge (Qwen3-VL-Plus; seed 0), same labels and criteria as the text judge; 95% prompt-bootstrap CI. Reference: run of the same reward without a budget; <sup>∗</sup>raw $\varepsilon _ { h } =$ 0.10 budget; <sup>‡</sup>raw open readout. Bold: $p < 1 0 ^ { - 3 }$

## 4.7. Human Audit and Image-Reading Judge

We re-labeled every evaluated response with an imagereading judge that receives the image alongside the request and the response (Table 5). It is stricter, confirming 93% of the text judge’s harmful labels and adding others, and every comparison the paper rests on holds under it. Harm falls monotonically with the budget, the isotonic map is again the safest Jev model, the underestimating reward again switches the constraint off, and under the same budget Jev beats the raw open readout $( p = 4 . 3 \times 1 0 ^ { - 8 } )$ . Calibrating the control channel of the open readout lowers its harm from 20.8% to 12.6% under this judge (105 against 22 discordant, $p = 4 \times 1 0 ^ { - 1 4 } )$ , below the raw Jev budget (15.0%; 51 against $2 7 , p = 0 . 0 0 9 )$ . Over-refusal under this judge stays between 3% and 7%.

One annotator, shown the image, the request and the response, labeled 200 responses blind to both judges, stratified by the judges’ labels and reweighted to the full pool. The text judge agrees with the annotator at Cohen’s $\kappa = 0 . 8 5$ (harmful precision 0.81, recall 0.72; 9.9% harmful against the annotator’s 11.2%), the image-reading judge at $\kappa = 0 . 7 1$ (precision 0.54), so the text judge stays the primary metric. Projected onto every model, the annotator’s rates preserve every ordering: 13.4% without a budget, 12.9%, 9.1% and 8.2% for $\varepsilon _ { h } = 0 . 2 0 , 0 . 1 0 , 0 . 0 5 .$ , 7.2% with the isotonic map, and $1 3 . 6 \%$ against 8.2% for the raw and calibrated open readout.

## 4.8. Robustness

The central results replicate over three seeds and a second GPU type (8.3% against 7.6%); the tightest budget settles in a 120-step run, the budget keeps its effect at three times the learning rate, and a map fitted on Llama-3.3-70B labels gives the same result over three seeds (6.6% against 6.1%, p = 0.21). Under sampled decoding $( T = 0 . 7$ , three samples per prompt) calibration lowers the rate at which an attack succeeds in any of three tries from 19.2% to 10.3%, lower in every seed (329 prompts fixed, 57 broken). On an 8B policy that already meets the budget, the calibrated controller stays inactive while the raw one raises harm (3.7% → 4.7%, $p = 0 . 0 2 1 )$

A Safety Prompt Controls One Rate. Instructing the base model to decline harmful requests, with no training, lowers held-out harm to 5.0% and harm on HADES to 12.4%, but it refuses 25% of the held-out benign MOSSBench queries, against 4% for the calibrated controller, and 7.3% of the pooled 400. A prompt moves the policy along the safety– helpfulness curve; the budget fixes where it stops.

## 5. Conclusion

A safety budget added to GRPO with an off-the-shelf scorer fails twice: the dual update reads a biased level, so the bias rather than the budget decides the controller state, and group standardization cancels the multiplier before it reaches the gradient. Two-channel constrained GRPO calibrates the level that the penalty and the dual update read and keeps the multiplier in the advantage. With a training-free open readout, held-out harm falls from 8.9% under standard GRPO to 4.8%, lower in every seed, with the largest gains on imageborne attacks and an unseen benchmark.

The recipe is short: label a few hundred training responses, fit one isotonic map per constraint, and read the mean bias b off the same labels; Proposition 1 then predicts each controller state, and the budget becomes a deployment parameter.

## LLM Usage Statement

An LLM-based research agent wrote and ran the training, evaluation and analysis code under the authors’ direction, mined the training prompts, and helped draft and edit the text; the authors chose the research questions, the method and the experimental design, and checked every reported number against the stored outputs. Jev 1.13 scores responses as the training reward (Sec. 3.4); Qwen3.5-4B serves as the open-model reward baseline; Qwen3.8-Max labels evaluation responses and the responses used to fit the isotonic map, and is never a training reward. No LLM was used to write reviews or to generate evaluation labels presented as human judgments.

## References

[1] Eitan Altman. Constrained Markov Decision Processes. Chapman and Hall/CRC, 1999. 2, 3

[2] Josef Dai, Xuehai Pan, Ruiyang Sun, Jiaming Ji, Xinbo Xu, Mickel Liu, Yizhou Wang, and Yaodong Yang. Safe RLHF: Safe Reinforcement Learning from Human Feedback. In International Conference on Learning Representations (ICLR), 2024. 2, 3

[3] Roger Girgis, Rodrigue de Schaetzen, Luke Rowe, Azalee´ Robitaille, Christopher Pal, and Liam Paull. Constrained group relative policy optimization. arXiv preprint arXiv:2602.05863, 2026. 2, 3, 4, 5, 8

[4] Yichen Gong, Delong Ran, Jinyuan Liu, Conglei Wang, Tianshuo Cong, Anyu Wang, Sisi Duan, and Xiaoyun Wang. Fig-Step: Jailbreaking Large Vision-Language Models via Typographic Visual Prompts. In AAAI Conference on Artificial Intelligence (AAAI), 2025. 1, 2, 5

[5] Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. On calibration of modern neural networks. In International Conference on Machine Learning (ICML), 2017. 3

[6] Ruoqi Guo, Yi Liu, Gelei Deng, Yuekang Li, Lida Zhao, Yutao Wu, Simin Chen, Ying Zhang, and Leo Yu Zhang. Just Ask Jev: Reinforcement Learning for Calibrated Decisions as a Zero-Shot Detector of AI Alignment Failures. arXiv preprint arXiv:2609.29429, 2026. 2

[7] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-Rank Adaptation of Large Language Models. In International Conference on Learning Representations (ICLR), 2022. 4

[8] Hakan Inan, Kartikeya Upasani, Jianfeng Chi, Rashi Rungta, Krithika Iyer, Yuning Mao, Michael Tontchev, Qing Hu, Brian Fuller, Davide Testuggine, and Madian Khabsa. Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations. arXiv preprint arXiv:2312.06674, 2023. 2

[9] Jiaming Ji, Xinyu Chen, Rui Pan, Conghui Zhang, Han Zhu, Jiahao Li, Donghai Hong, Boyuan Chen, Jiayi Zhou, Kaile Wang, Juntao Dai, Chi-Min Chan, Yida Tang, Sirui Han, Yike Guo, and Yaodong Yang. Safe RLHF-V: Safe Reinforcement Learning from Multi-modal Human Feedback. In Advances in Neural Information Processing Systems (NeurIPS), 2025. 3

[10] De Jiang, Zhengyang Zhang, Kehong Yuan, and Shaohua Ma. CARGO-VL: Counterfactual Arbitration with Risk-Constrained Group Optimization for Vision-Language Models. arXiv preprint arXiv:2608.04509, 2026. 3

[11] Yifan Lan, Zhongye Liu, Yurui Chang, Hanyu Wang, Yongkang Du, Lu Lin, and Jinghui Chen. DecisionRobust: Benchmarking Robustness for JEV Style Decision Models Across Text and Vision. arXiv preprint arXiv:2610.19118, 2026. 2

[12] Baoteng Li, Wenzhuo Wu, Kongming Liang, and Zhanyu Ma. Visual Jev Rewards: Reference-Bound Verification for Multi-Subject Image Generation. arXiv preprint arXiv:2610.09328, 2026. 2

[13] Xirui Li, Hengguang Zhou, Ruochen Wang, Tianyi Zhou, Minhao Cheng, and Cho-Jui Hsieh. Is your multimodal language model oversensitive to safe queries? In International Conference on Learning Representations (ICLR), 2025. 2, 5

[14] Xiang Li, Aichun Huang, Mingming Zhang, and Zheli Liu. Same Attack, Different Judge: A First Controlled Adversarial Comparison of Jev, LLM Judges, and Purpose-Built Safety Classifiers. arXiv preprint arXiv:2610.17820, 2026. 2, 3

[15] Yinheng Li and Justin Wagle. LLM-as-Jev: LLMs Are Already Jev-Style Decision Models—When and How to Fine-Tune Them. arXiv preprint arXiv:2610.02076, 2026. 2, 4, 5

[16] Yifan Li, Hangyu Guo, Kun Zhou, Wayne Xin Zhao, and Ji-Rong Wen. Images are achilles’ heel of alignment: Exploiting visual vulnerabilities for jailbreaking multimodal large language models. In European Conference on Computer Vision (ECCV), 2024. 6

[17] Xin Liu, Yichen Zhu, Jindong Gu, Yunshi Lan, Chao Yang, and Yu Qiao. MM-SafetyBench: A Benchmark for Safety Evaluation of Multimodal Large Language Models. In European Conference on Computer Vision (ECCV), 2024. 1, 2, 4

[18] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, Kai Chen, and Dahua Lin. MMBench: Is Your Multi-modal Model an All-around Player? In European Conference on Computer Vision (ECCV), 2024. 5

[19] Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding R1-Zero-Like Training: A Critical Perspective. arXiv preprint arXiv:2503.20783, 2025. 2, 3, 4, 8

[20] Weidi Luo, Siyuan Ma, Xiaogeng Liu, Xiaoyu Guo, and Chaowei Xiao. JailBreakV: A Benchmark for Assessing the Robustness of MultiModal Large Language Models against Jailbreak Attacks. arXiv preprint arXiv:2404.03027, 2024. 1, 2, 5

[21] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv preprint arXiv:2402.03300, 2024. 1, 5, 8

[22] Adam Stooke, Joshua Achiam, and Pieter Abbeel. Responsive Safety in Reinforcement Learning by PID Lagrangian Methods. In International Conference on Machine Learning (ICML), 2020. 3

[23] TypeSafe AI. Jev: TypeSafe System One Decision Model. https : / / docs . typesafe . ai / introduction, 2026. Accessed October 2026. 2, 5

[24] Siyin Wang, Xingsong Ye, Qinyuan Cheng, Junwen Duan, Shimin Li, Jinlan Fu, Xipeng Qiu, and Xuanjing Huang. Safe inputs but unsafe output: Benchmarking cross-modality safety alignment of large vision-language models. In Findings ofthe Associationfor Computational Linguistics: NAACL, 2025. 6

[25] Shang Wang, Tianqing Zhu, Huajie Chen, Jiayang Li, Meng Yang, and Bo Liu. Hidden Risks of Jev: An Empirical Study of Security, Privacy, and Dual Use. arXiv preprint arXiv:2610.04985, 2026. 2

[26] Bianca Zadrozny and Charles Elkan. Transforming classifier scores into accurate multiclass probability estimates. In ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2002. 3

[27] Yongting Zhang, Lu Chen, Guodong Zheng, Yifeng Gao, Rui Zheng, Jinlan Fu, Zhenfei Yin, Senjie Jin, Yu Qiao, Xuanjing Huang, Feng Zhao, Tao Gui, and Jing Shao. SPA-VL: A Comprehensive Safety Preference Alignment Dataset for Vision Language Models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025. 2

[28] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. In Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track, 2023. 2

[29] Kaiwen Zhou, Chengzhi Liu, Xuandong Zhao, Anderson Compalas, Dawn Song, and Xin Eric Wang. Multimodal situational safety. In International Conference on Learning Representations (ICLR), 2025. 6

[30] Yongshuo Zong, Ondrej Bohdal, Tingyang Yu, Yongxin Yang, and Timothy Hospedales. Safety fine-tuning at (almost) no cost: A baseline for vision large language models. In International Conference on Machine Learning (ICML), 2024. 2