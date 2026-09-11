# WHEN IS TEST-TIME ADAPTATION IDENTIFIABLE FROM UNLABELED EVIDENCE?

Kartik Jhawar<sup>∗</sup> & Lipo Wang

Institute for Digital Molecular Analytics and Science School of Electrical and Electronic Engineering Nanyang Technological University, Singapore kartikvi001@e.ntu.edu.sg

## ABSTRACT

Test-time adaptation (TTA) offers many ways to update a deployed model without labels, but choosing the wrong update can make a strong source model worse. Recent methods therefore try to predict which adaptation will work from unlabeled test data. We ask a prior question: does the evidence given to the selector contain enough information to determine the best action at all? We show that this is not guaranteed, even with a perfect selector. If an observation channel makes two deployments look the same while their TTA rankings differ, reliable selection is impossible from that channel; richer evidence can restore the decision only when it resolves the relevant ambiguity. We make this boundary exact in a finite-batch Gaussian TTA model, where doing nothing beats mean recentering for small shifts, <sub>recentering wins beyond a unique critical shift, and the boundary shrinks as 1/</sub>√<sub>n.</sub> Public benchmark studies on CIFAR-100-C and DomainNet-126 show the same failure mode with modern TTA methods: changing only deployment structure can reverse the oracle action while global order-blind evidence remains unchanged. The result is a practical way to separate two failure modes that are usually mixed together: a weak selector versus an information channel that cannot support the desired decision in the first place.

## 1 INTRODUCTION

Test-time adaptation (TTA) updates a trained model using unlabeled data after deployment. Methods such as Tent, EATA, SAR, DeYO, and ROID can improve performance when test data differ from the source distribution (Wang et al., 2021; Niu et al., 2022; 2023; Lee et al., 2024a; Marsden et al., 2024). But the same update can also be harmful. Small batches, class imbalance, temporal correlation, and realistic model-selection choices can change or even reverse the benefit of adaptation (Gong et al., 2022; Lim et al., 2023; Zhao et al., 2023; Sreeram et al., 2026).

This has created a second problem on top of adaptation itself: which TTA procedure should we use? AETTA and TTALine estimate TTA performance without target labels (Lee et al., 2024b; Kim et al., 2024). MORPHEUS goes closer to our setting and predicts which candidate TTA method will perform best from pre-adaptation entropy and representation geometry (Danilowski et al., 2026). Cygert et al. study unsupervised TTA model selection after candidate adaptations have been executed (Cygert et al., 2026). Recent theory and diagnostics also show that TTA can be hard to recover, temporally unstable, or underspecified (Zhou et al., 2026; Sreeram et al., 2026; Shamsi et al., 2026). These works improve the selector or the adaptation procedure. Also, we have stated works to tell us when TTA can become helpful or harmful, but none so far tell us whether an unlabeled selector can recognize, from the evidence available before adaptation, which side of that reversal it is on. We therefore ask a question that comes before both:

Is the unlabeled evidence available before adaptation actually enough to determine which action is best?

This question is not the obvious statement that the best method depends on the candidate family. We keep the candidate actions (our menu of TTA choices - TENT, DeYO, etc.) fixed. The problem is that the selector sees only a chosen summary of the unlabeled deployment data, while the relative per formance of stateful TTA actions can depend on deployment properties—such as batch composition and order—that this summary may not preserve. Two deployments can therefore be indistinguishable through the selector’s evidence and still have opposite action rankings. In that case, a more powerful regressor, a larger neural selector, or more training data for the selector cannot recover information that the observation channel discarded.

We call the oracle action identifiable when the allowed evidence is sufficient to determine at least one oracle-best intervention over the deployment family under study. This is intentionally a decision-level question. We do not claim that identifying an optimal action without identifying all risks is a new statistical principle; partial-identification and decision theory already study optimal decisions under incomplete knowledge (Kasy, 2016; Pu & Zhang, 2021; Han, 2024; Christensen et al., 2026). Domain adaptation theory likewise gives important impossibility and identifiability precedents (Ben-David et al., 2010; Garg et al., 2022; Gulrajani & Hashimoto, 2022; Kong et al., 2022; Hanneke et al., 2023; Dong et al., 2026). Our contribution is to study TTA selection as an information problem: the actions themselves are data-dependent interventions, the observation channel is fixed before choosing an action, and we ask exactly how batch size, shift mechanism, stream structure, and evidence richness change the answer.

Our contribution is to study TTA selection as an information problem: before asking how to build a better selector, we ask whether the unlabeled evidence available to it is sufficient to determine the best action.

## Our contributions are:

• A formulation of TTA action identifiability. We formalize when the oracle-best action from a fixed set of TTA choices is determined by a specified unlabeled observation channel and deployment family.

• Information limits and positive conditions for selection. We show that indistinguishable deployments with different best actions make reliable selection impossible from that evidence, and give the complementary condition under which an optimal action is identifiable.

• A TTA-specific finite-batch boundary. In a tractable KEEP-versus-RECENTER model, we prove a unique transition between when adaptation hurts and helps, show that its critical shift scales as $1 / { \sqrt { n } }$ , and construct different shift mechanisms with the same coarse evidence but opposite preferred actions.

• Public-benchmark evidence of the same information boundary. On CIFAR-100-C and DomainNet-126, the same selected images can require different TTA actions after only the deployment structure changes. Global order-blind evidence cannot capture these reversals, while stream-aware evidence can; under different stochastic deployments, this advantage can disappear.

The practical message is direct. Before asking how to build a stronger TTA selector, we should first ask whether its inputs can support the decision we expect it to make. This separates a modeling failure from an information failure, and tells us when adding richer deployment evidence is more important than adding selector capacity.

## 2 PROBLEM SETUP

Actions. Let $f _ { 0 }$ be a fixed pretrained source model. We have a finite action set

$$
\mathcal { A } = \{ \mathrm { K E E P } , a _ { 1 } , . . . , a _ { K } \} .
$$

Here KEEP means exactly one thing: use thefrozen source model and do not adapt it. Every other action may be a data-dependent TTA procedure. Let $X _ { 1 : n } = ( X _ { 1 } , \ldots , X _ { n } ) $ denote an unlabeled target adaptation batch of size n. Formally, an action can be viewed as a map

$$
\mathcal { M } _ { a } : ( f _ { 0 } , X _ { 1 : n } ) \mapsto f _ { a , X _ { 1 : n } } ,
$$

which takes the source model and an unlabeled adaptation batch and returns the predictor that will be deployed.

In the benchmark experiments, the five empirical actions are SOURCE/KEEP, DeYO, Tent, ROID, and test-time normalization (Lee et al., 2024a; Wang et al., 2021; Marsden et al., 2024). To theoretically analyze the finite-batch trade-offs between estimation noise, shift magnitude, and shift mechanisms in closed form, Section 3.4 introduces RECENTER as an analytically tractable one-dimensional prototype of adaptation (shifting a decision threshold to the unlabeled sample mean). RECENTER is strictly a theoretical device to study KEEP-versus-adaptation boundaries and should not be confused with the empirical benchmark methods or with NEO (Murphy et al., 2026).

A “world”. A world $\theta \in \Theta$ specifies both the target joint distribution $P _ { \theta } ( X , Y )$ and the deployment protocol, including batch construction, ordering, or temporal structure when these affect adaptation. This second part is essential because a stateful TTA action can produce a different predictor from the same image multiset when the images arrive differently.

We define the prospective risk of action a on an independent fresh target point $( X ^ { \prime } , Y ^ { \prime } )$ as

$$
R _ { \theta } ( a ) = \mathbb { E } _ { \theta } [ \ell ( f _ { a , X _ { 1 : n } } ( X ^ { \prime } ) , Y ^ { \prime } ) ] ,
$$

where the expectation includes both the adaptation batch and the fresh evaluation point. The oracle action set is

$$
\mathcal { A } ^ { * } ( \theta ) = \arg \operatorname* { m i n } _ { a \in \mathcal { A } } R _ { \theta } ( a ) .
$$

Target labels are used only after an experiment to measure these risks and define the oracle. A real selector never sees them before choosing an action.

What the selector is allowed to see. Before choosing an action, the selector observes only

$$
Z _ { n } = \phi _ { n } ( f _ { 0 } , X _ { 1 } , \ldots , X _ { n } ) ,
$$

where $\phi _ { n }$ denotes the specified unlabeled evidence channel, which instantiates features such as entropy, confidence margins, source logits, global empirical moments, or sequential order statistics. Let $Q _ { \theta , n }$ be the probability law of $Z _ { n }$ in world θ.

Definition 1 (Oracle-action identifiability). Fix an observation channel and a family of worlds. For an evidence law $Q ,$ , define

$$
\Theta ( Q ) = \{ \theta \in \Theta : Q _ { \theta , n } = Q \} .
$$

The oracle action is identifiable at $Q$ if

$$
\bigcap _ { \theta \in \Theta ( Q ) } { \mathcal { A } } ^ { * } ( \theta ) \neq \varnothing .
$$

If every world has one unique best action, this simply means: any two worlds that look the same through the allowed evidence must have the same best action.

Why we define it this way. This definition asks an information question, not a training question. If we knew the full probability law of the allowed evidence, would it determine a best action? A practical selector sees only finite data, so there is a second statistical problem of learning the decision from limited samples. We keep these questions separate.

It is also useful to separate three levels of difficulty:

$$
\mathrm { T a r g e t / M e c h a n i s m \mathrm { - } I D } \Rightarrow \mathrm { R i s k \mathrm { - } I D } \Rightarrow \mathrm { A c t i o n \mathrm { - } I D } .
$$

Knowing the complete target mechanism is stronger than knowing every action risk, and knowing every action risk is stronger than merely knowing which action wins. The reverse directions need not hold. This hierarchy is a positioning device rather than a claim of a new general decision-theory principle; Section 6 connects it to earlier partial-identification and domain-adaptation work.

Regret. For a selector $s ,$ its oracle regret in world θ is the extra risk caused by its chosen action compared with the oracle:

$$
\mathrm { R e g } _ { \theta } ( s ) = \mathbb { E } \bigg [ R _ { \theta } ( s ( Z _ { n } ) ) - \operatorname* { m i n } _ { a \in \mathcal { A } } R _ { \theta } ( a ) \bigg ] .
$$

The theory below asks when this extra risk can be forced to zero and when some positive error is unavoidable.

## 3 WHEN IS THE ORACLE TTA ACTION IDENTIFIABLE?

## 3.1 IF TWO WORLDS LOOK THE SAME BUT NEED DIFFERENT ACTIONS, SELECTION IS IMPOSSIBLE

Theorem 1 (Exact non-identifiability). Suppose two worlds $\theta _ { 0 } , \theta _ { 1 }$ have exactly the same evidence law, $Q _ { \theta _ { 0 } , n } = Q _ { \theta _ { 1 } , n } ,$ but have different unique oracle actions $a _ { 0 } ^ { * } \neq a _ { 1 } ^ { * }$ . Under an equal prior over the two worlds, every selector based only on $Z _ { n }$ has average oracle-action error at least $1 / 2 .$

If every non-oracle action costs at least $\Delta > 0$ extra risk, then every selector has average oracle regret at least $\Delta / 2$

Simple meaning. The selector sees the same kind of evidence in both worlds, so it has no reliable way to know which world it is in. If the correct actions disagree, it cannot be right in both. Under an equal two-world test, at least half of the error is unavoidable.

The statement can remain true even with unlimited unlabeled data if the full allowed unlabeled observation remains the same. This is the TTA specialization of a classical two-point decision/testing argument; we use it as a limit on pre-action selectors rather than claim the testing inequality itself as new. In our discrete construction, both worlds have the same unlabeled marginal $P ( X ) =$ (0.45, 0.25, 0.30), but the action risks reverse:

<table><tr><td></td><td>KEEP</td><td>ADAPT</td></tr><tr><td>World A</td><td>0.5625</td><td>0.2500</td></tr><tr><td>World B</td><td>0.2100</td><td>0.5275</td></tr></table>

Perfect equality is a strong condition, so we also need an approximate statement.

Corollary 1 (Approximate non-identifiability). Let the two evidence laws be $Q _ { 0 } , Q _ { 1 }$ and let their unique oracle actions differ. Under an equal prior,

$$
\mathbb { P } ( w r o n g o r a c l e a c t i o n ) \ge \frac { 1 - \mathrm { T V } ( Q _ { 0 } , Q _ { 1 } ) } { 2 } .
$$

If every wrong action costs at least ∆, then

$$
\mathbb { E } [ \mathrm { R e g } ] \geq \frac { \Delta } { 2 } \big ( 1 - \mathrm { T V } ( Q _ { 0 } , Q _ { 1 } ) \big ) .
$$

Simple meaning. Total variation (TV) measures how distinguishable two evidence distributions are. TV near zero means they look very similar. The bound says that when the evidence distributions overlap heavily, some action-selection error must remain. Complete proofs of Theorem 1 and Corollary 1 are in Appendix A.1 and A.2, respectively.

## 3.2 THE POSITIVE SIDE: WHEN THE ACTION IS IDENTIFIABLE

Theorem 2 (Compatibility criterion). Fix an evidence law Q. There exists one action, depending only on Q, that has zero oracle regretfor every world compatible with Q ifand only if

$$
\bigcap _ { \theta \in \Theta ( Q ) } { \mathcal { A } } ^ { * } ( \theta ) \neq \varnothing .
$$

If every compatible world has a unique oracle action, this means they must all share the same action.

Simple meaning. After seeing the evidence, imagine making a list of every hidden world that could still have produced it. If all of those worlds agree on at least one best action, then the action is determined even if the full target world is not.

The complete proof is in Appendix A.3. A stronger sufficient condition is a positive worst-case margin:

$$
M _ { a } ( Q ) = \operatorname* { i n f } _ { \theta \in \Theta ( Q ) } \operatorname* { m i n } _ { b \neq a } \left[ R _ { \theta } ( b ) - R _ { \theta } ( a ) \right] > 0 .
$$

Then a is not only optimal in every compatible world; it beats every other action by a positive amount.

Figure 3. Exact finite-n KEEP/RECENTER phase boundary  
![](images/2153bec12e1fb906c78118659fc01c65b8b332f1d03825c32e166c45bb1f1304.jpg)  
Figure 1: Finite-batch KEEP vs. RECENTER phase boundary (Theorem 3). $L e f t { \mathrm { : } }$ Prospective classification risk under Gaussian translation shift $( \mu = 2 , \sigma = 1 )$ . KEEP (blue) is optimal near zero shift, while RECENTER (horizontal lines) wins once $| \delta | > \delta _ { c } ( n )$ . Right: Critical tipping point $\delta _ { c } ( n )$ plotted against $1 / { \sqrt { n } } ,$ confirming the exact $\Theta ( n ^ { - 1 / 2 } )$ finite-batch rate.

## 3.3 WHY IGNORING ORDER CAN HIDE THE CORRECT ACTION

Proposition 1 (Permutation/invariance obstruction). Let g be a transformation such as permuting the order of a deployment batch. Suppose the evidence channel ignores that transformation, so $\phi _ { n } ( g B ) = { \dot { \phi } } _ { n } ( B )$ for every batch B. Ifa world and its transformed version have different unique oracle actions, then the oracle action is not identifiable from $\phi _ { n }$ over any family containing both worlds.

Simple meaning. If the selector throws order away, it cannot react to order. But a stateful TTA method may react strongly to order. Therefore the same image set can look identical to the selector while requiring a different action. This is exactly what our matched CIFAR-100-C and DomainNet-126 experiments are designed to test. The formal proof is in Appendix A.4.

## 3.4 A TTA-SPECIFIC FINITE-BATCH KEEP-VERSUS-RECENTER BOUNDARY

The previous results are general information statements. We now want a concrete TTA mechanism where the best action changes in a predictable way with batch size and shift size. This is the paper’s main TTA-specific theorem.

The Core Trade-off. When adapting to an unlabeled batch, an algorithm faces a fundamental tension:

• KEEP (Frozen Source): Incurs zero estimation noise, but suffers increasing error as the true distribution shift |δ| grows.

• RECENTER (Adaptation): Successfully eliminates the true shift δ, but incurs afinite-sample noisefloor because the shift is estimated from a noisy batch of size n.

1D Gaussian Setup. Consider balanced binary classification $Y \in \{ - 1 , + 1 \}$ with source distributions $\mathcal { N } ( \pm \mu , \sigma ^ { 2 } ) \left( \bar { \mu } , \sigma > 0 \right)$ split by a source threshold at zero. In the target domain, data undergo a

translation shift $\delta \colon$

$$
X = \delta + \mu Y + \epsilon , \quad \epsilon \sim \mathcal { N } ( 0 , \sigma ^ { 2 } ) .\tag{1}
$$

KEEP leaves the threshold at zero. RECENTER receives n unlabeled target samples $X _ { 1 : n }$ , computes the empirical sample mean $\hat { \delta } = \bar { X } _ { n } .$ , and classifies a fresh target point using threshold <sup>ˆ</sup>δ.

The prospective risk of KEEP grows strictly with the true translation |δ|:

$$
R _ { K } ( \delta ) = \frac { 1 } { 2 } \left[ \Phi \left( - \frac { \mu + \delta } { \sigma } \right) + \Phi \left( \frac { \delta - \mu } { \sigma } \right) \right] .\tag{2}
$$

For RECENTER, shifting the threshold cancels $\delta ,$ leaving only the finite-sample estimation error $E _ { n } = \hat { \delta } - \delta$ . With $\begin{array} { r } { S _ { n } = \sum _ { i = 1 } ^ { n } Y _ { i } } \end{array}$ , the exact finite-n risk is independent of δ and given by:

$$
R _ { R } ( n ) = 2 ^ { - n } \sum _ { k = 0 } ^ { n } { \binom { n } { k } } \Phi \left( { \frac { \mu ( 2 k - n ) / n - \mu } { \sigma { \sqrt { 1 + 1 / n } } } } \right) .\tag{3}
$$

Theorem 3 (Finite-batch KEEP/RECENTER boundary). For everyfinite batch size $n \geq 1$ , there exists a unique critical shift $\delta _ { c } ( n ) > 0$ such that $R _ { K } ( \delta _ { c } ( \dot { n _ { } } ) ) = R _ { R } ( n )$ . Furthermore:

$$
| \delta | < \delta _ { c } ( n ) \implies R _ { K } ( \delta ) < R _ { R } ( n ) \quad ( \mathrm { K E E P } w i n s ) ,\tag{4}
$$

$$
| \delta | > \delta _ { c } ( n ) \implies R _ { K } ( \delta ) > R _ { R } ( n ) \quad ( \mathrm { R E C E N T E R } w i n s ) .\tag{5}
$$

As $n \to \infty$ , this phase boundary scales as:

$$
\delta _ { c } ( n ) = \frac { \sqrt { \mu ^ { 2 } + \sigma ^ { 2 } } } { \sqrt { n } } + O ( n ^ { - 3 / 2 } ) = \Theta ( n ^ { - 1 / 2 } ) .\tag{6}
$$

Intuition and the $1 / { \sqrt { n } } \mathbf { L a w } .$ When shift $\delta = 0$ , the source threshold is already optimal. RECEN-TER still estimates a mean from finite samples, introducing random jitter that degrades performance $( R _ { R } ( n ) > R _ { K } ( 0 ) )$ . However, as $| \delta |$ grows, KEEP degrades quadratically, while RECENTER maintains a flat noise floor. The two curves must cross at exactly one critical value $\delta _ { c } ( n ) ( \mathrm { F i g u r e } 1 )$ .

Because the standard error of a sample mean scales as $1 / { \sqrt { n } }$ , the true distribution shift must exceed this statistical noise scale before test-time adaptation provides a net benefit. The full proof in Appendix A.5 uses the exact finite mixture above; the numerical table does not rely on a normal approximation.

## 3.5 THE SAME TARGET MEAN CAN COME FROM TWO DIFFERENT SHIFTS

A change in target mean does not always mean that the whole distribution translated. Suppose the class-conditionals stay fixed at $\mathcal { N } ( \pm \mu , \sigma ^ { 2 } )$ but the positive-class probability changes to $\pi \neq 1 / 2$ Then

$$
m = \mu ( 2 \pi - 1 )
$$

is the new target mean even though neither class distribution moved.

Proposition 2 (Under pure prior shift, mean recentering moves the threshold the wrong way). Under pure class-prior shift with $\bar { \pi } \neq 1 / 2$ , population RECENTER has strictly larger classification risk than KEEP.

Simple meaning. If there are simply more positive examples, the overall mean moves to the right. Mean recentering interprets this as a physical translation and moves the threshold right. But the Bayes-optimal threshold actually moves left because positive examples are now more common. The recentering correction therefore goes in the wrong direction. The Full Proof is in Appendix A.6.

Corollary 2 (Same population mean, opposite best actions). Fix m with $0 < | m | < \mu .$ Compare two worlds:

1. World C: balanced classes with a real translation $\delta = m ,$

2. World P: no translation, but class prior $\pi = ( 1 + m / \mu ) / 2$

Both worlds have the same population target mean m. Population RECENTER is better in World C, while KEEP is better in World P. Therefore a channel that reveals only the population target mean cannot identify the oracle action over afamily containing both worlds.

Remark 1 (What “same mean” does not mean). The two worlds have the same population mean. Their full distributions are not identical, and the probability law of a finite-sample empirical mean need not be identical either. This is intentional. The result shows that mean-only evidence is too coarse; richer statistics may separate the worlds and recover the correct action. The Full Proof is in Appendix A.7.

Together, these results give the main theoretical message: the best TTA action depends not only on the shift itself, but also on batch size, the shift mechanism, the deployment structure, and what information the selector is allowed to use.

## 4 EXPERIMENTAL DESIGN

The experiments are built to test the information question above. They are not presented as a new universal TTA algorithm.

## 4.1 CANDIDATE ACTIONS AND ORACLE EVALUATION

Both public-benchmark studies use the same five choices: SOURCE/KEEP, DeYO, Tent, ROID, and test-time normalization (Lee et al., 2024a; Wang et al., 2021; Marsden et al., 2024). Every action starts from the same frozen source state for a target world. Labels are used only afterward to measure each action’s error and define the oracle. The selector evidence is always label-free and computed before the selected TTA action is executed.

## 4.2 CIFAR-100-C: CONTROLLED DEPLOYMENTS

CIFAR-100-C (Hendrycks & Dietterich, 2019) gives the broad controlled study. We construct 675 worlds: 15 corruptions, three severities, five deployment regimes, and three seeds. The regimes change class-prior conditions and stream structure. In the key matched experiment, an IID world and a correlated world contain exactly the same selected images; only their ordering changes.

This directly tests Proposition 1. A global evidence channel that ignores order cannot distinguish the two arrangements, while stateful TTA actions can behave differently on them.

We compare a hierarchy of source-only evidence. Z1 contains simple global uncertainty summaries. Z2 adds richer global, permutation-invariant source-output statistics. Z3 additionally retains how source-model behavior changes across successive windows. Z4 further adds a fixed projection of the ordered source logits. The selector is evaluated mainly by oracle regret, together with exact/tie-aware action accuracy, negative transfer, benefit capture, bootstrap intervals, and paired tests. Appendix B lists the complete reviewer-check suite.

Selection baselines. SOURCE/KEEP and the best fixed action measure how much can be achieved without world-specific selection. MORPHEUS is the closest direct pre-action comparison and is implemented using the author-confirmed MultiOutput random-forest architecture and both documented Neural-Collapse feature interpretations (Danilowski et al., 2026). We also report the same regression architecture over our Z1–Z4 evidence hierarchy. Cygert et al. are discussed as a strongerinformation-budget comparison because their unsupervised selection statistics are evaluated after candidate adaptations have been run (Cygert et al., 2026); we therefore do not pretend that it is a same-budget pre-action baseline.

## 4.3 DOMAINNET-126: NATURAL DOMAIN SHIFTS

DomainNet-126 (Peng et al., 2019) tests whether the same phenomenon survives beyond corruption shifts. We use 90 worlds from six directional domain transfers, three seeds, and five deployment regimes. Each world contains 2,000 target images, and all TTA actions run with execution batch size 32.

Table 1: CIFAR-100-C controlled LOCO selection. Lower regret is better; action accuracy is the fraction of worlds where the exact oracle action is selected. MORPHEUS uses the final authorconfirmed MultiOutput RF implementation.
<table><tr><td>Selector / evidence</td><td>Regret (pp) ↓</td><td>Action acc. (%) ↑</td></tr><tr><td>SOURCE / KEEP</td><td>4.980</td><td>50.2</td></tr><tr><td>MORPHEUS entropy</td><td>4.135</td><td>53.0</td></tr><tr><td>MORPHEUS NC-text</td><td>1.170</td><td>67.6</td></tr><tr><td>MORPHEUS NC-appendix</td><td>1.358</td><td>66.5</td></tr><tr><td>Z1 (global)</td><td>1.705</td><td>63.7</td></tr><tr><td>Z2 (richer global)</td><td>1.247</td><td>65.0</td></tr><tr><td>Z3 (stream-aware)</td><td>0.312</td><td>76.6</td></tr><tr><td>Oracle</td><td>0.000</td><td>100.0</td></tr></table>

Matched IID/correlated worlds again contain the same selected image multiset and class quota; only order changes. The original stream-aware Z3 representation used a 200-sample evidence window. Final CPU-only reviewer checks then reconstruct the frozen evidence exactly and evaluate a compact 11-number stream summary at fixed windows 32, 64, 128, and 200 without rerunning any TTA method.

The 11 numbers measure variation and drift in SOURCE entropy, confidence, margin, predicted-class composition, and mean logits across successive windows. They use SOURCE outputs only. A nested-CV Ridge regression provides a deliberately simple linear baseline to test whether the result depends on random-forest nonlinearity.

## 5 RESULTS

## 5.1 SAME IMAGES, DIFFERENT DEPLOYMENT, DIFFERENT ORACLE ACTION

The matched experiments give the most direct test of the theory. On CIFAR-100-C, 157 of 270 matched IID/correlated pairs change oracle action even though each pair contains exactly the same selected images; 82 are strong flips. Global permutation-invariant evidence is essentially unchanged across each pair, so it cannot respond to an action reversal caused only by stream structure. On DomainNet-126, the oracle action changes in 33 of 36 same-image matched pairs, including 27 strong flips. Aligned SOURCE predictions and SOURCE error remain unchanged.

These are not synthetic toy outcomes: they arise from public benchmarks and modern TTA actions. They instantiate Proposition 1: an observation channel can discard a deployment variable that matters to the ranking of the actions.

## 5.2 CIFAR-100-C: RICHER EVIDENCE SHARPLY REDUCES ORACLE REGRET

Table 1 reports the frozen leave-one-corruption-out controlled evaluation. Z3 reduces mean regret to 0.312 pp. The final author-confirmed MORPHEUS MultiOutput baselines remain substantially farther from oracle: 1.170 pp for the text Neural-Collapse definition and 1.358 pp for the appendix definition. The entropy-only MORPHEUS variant has 4.135 pp regret. Paired bootstrap intervals and Holm-corrected tests comparing these MORPHEUS variants with Z3 are reported in Appendix B.

The matched strong-flip subset makes the mechanism even clearer. Z3 changes its selected action in 92.7% of the 82 strong pairs and has only 0.093 pp mean pair regret. The author-confirmed MORPHEUS entropy, NC-text, and NC-appendix selectors change action in 0% of these pairs and have 6.189, 2.873, and 2.921 pp pair regret, respectively. Because their evidence is global, the two reorderings look the same to them.

## 5.3 DOMAINNET-126: THE SIGNAL IS SMALL, SIMPLE, AND EXECUTION-SCALE

DomainNet independently reproduces the information-boundary effect under natural domain shifts. Table 2 shows the leave-one-transfer-out result. The original 2,990-dimensional Z3 RF has 0.034 pp regret and 93.3% exact action accuracy. The predeclared 11-dimensional ORDER11 summary at the actual TTA execution scale W = 32 gives the same point result. A nested Ridge model on those 11 numbers remains strong at 0.061 pp regret and 92.2% action accuracy. In contrast, global Z2 remains at 1.933 pp regret and 45.6% action accuracy.

Table 2: DomainNet-126 LOTO selection over 90 worlds. The compact ORDER11 results use only 11 SOURCE-derived stream statistics.
<table><tr><td>Selector / evidence</td><td>Regret (pp) ↓</td><td>Action acc. (%) ↑</td></tr><tr><td>SOURCE / KEEP</td><td>1.933</td><td>45.6</td></tr><tr><td>MORPHEUS entropy</td><td>4.886</td><td>46.7</td></tr><tr><td>Z2 RF (global)</td><td>1.933</td><td>45.6</td></tr><tr><td>Z3 MultiOutput</td><td>0.331</td><td>83.3</td></tr><tr><td>Z3 RF (2,990-D)</td><td>0.034</td><td>93.3</td></tr><tr><td>ORDER11 RF,  $W = 3 2$ </td><td>0.034</td><td>93.3</td></tr><tr><td>Ridge ORDER11,  $W = 3 2$ </td><td>0.061</td><td>92.2</td></tr></table>

On the 27 strong DomainNet matched flips, global Z2 changes action in 0/27 pairs and has 1.942 pp mean pair regret. Z3 and ORDER11-RF change action in all 27 with zero pair regret. Ridge changes action in 26/27 and has 0.044 pp pair regret. The conclusion therefore does not depend on thousands of features, a $W = 2 0 0$ window, or a complicated nonlinear selector. Window, seed, PCA, transfer-holdout, and native-TTA checks are in Appendix B.

## 5.4 NO EVIDENCE REPRESENTATION IS UNIVERSALLY BEST

The controlled matched-world experiment is deliberately designed so that order is decision-relevant. Under randomized stochastic CIFAR deployments, the large Z3 advantage disappears. When selectors are trained and evaluated within the stochastic deployment family, MORPHEUS NC-text obtains 0.210 pp mean regret, NC-appendix 0.293 pp, and Z3 0.338 pp. We keep this counter-result because it is important: richer stream evidence is not automatically superior. Its value depends on whether the deployment family contains ambiguities that the extra information resolves.

This is the empirical form of the paper’s central claim. A selector should not be judged only by its prediction architecture; it should also be judged by whether its observation channel retains the information needed to distinguish deployments whose action rankings differ.

## 6 RELATED WORK AND POSITIONING

TTA reliability and deployment effects. TTA methods such as Tent, EATA, SAR, DeYO, and ROID focus primarily on how to adapt reliably (Wang et al., 2021; Niu et al., 2022; 2023; Lee et al., 2024a; Marsden et al., 2024). NOTE, TTN, and realistic TTA evaluations show that temporal correlation, small batches, class imbalance, and protocol choices can strongly affect adaptation (Gong et al., 2022; Lim et al., 2023; Zhao et al., 2023). Tempora further shows that TTA utility and method rankings can change over time (Sreeram et al., 2026). These findings motivate our question, but they do not characterize whether a specified pre-action evidence channel identifies the oracle intervention.

TTA performance estimation and method selection. AETTA estimates the accuracy of an adapted model without labels, while TTALine uses accuracy/agreement relationships after TTA (Lee et al., 2024b; Kim et al., 2024). MORPHEUS is our closest operational neighbor: it uses pre-adaptation entropy and Neural-Collapse geometry to predict post-adaptation performance and select among TTA procedures (Danilowski et al., 2026). We do not claim to be the first to select a TTA method before adaptation. We ask the prior statistical question: when can any selector using a given pre-action observation channel determine the oracle action at all?

Cygert et al. study realistic unsupervised TTA model selection, but their criteria are evaluated after candidate adaptations are run (Cygert et al., 2026). This is a richer information/computation budget than our pre-action setting, so we treat it as an adjacent comparison rather than forcing a misleading same-budget table. NEO uses latent re-centering as an actual TTA method (Murphy et al., 2026); our RECENTER action is instead a one-dimensional theoretical device used to derive a finite-sample boundary.

Recent limits and underspecification in TTA. Zhou et al. study TTA learnability through recovery complexity: how quickly an adaptation process can return to low excess risk after a shift (Zhou et al., 2026). Shamsi et al. study underspecification of unsupervised TTA objectives and maintain multiple hypotheses (Shamsi et al., 2026). These are close in spirit because they expose limits of unlabeled adaptation, but their objects differ from ours. We ask whether the identity of the best member of a fixed menu of data-dependent interventions is determined by a specified observation channel before choosing an intervention.

Domain adaptation, transfer, and unlabeled performance estimation. Ben-David et al. give foundational impossibility results for domain adaptation, and Garg et al. show that unlabeled OOD performance estimation requires assumptions (Ben-David et al., 2010; Garg et al., 2022). Gulrajani and Hashimoto study identifiability of domain mappings; Kong et al. study partial identifiability of target-domain structure; more recent representation-based work identifies still richer target informa tion under structural assumptions (Gulrajani & Hashimoto, 2022; Kong et al., 2022; Ng et al., 2025). These works target a domain map, target structure, or predictor-level object. Our target is deliberately downstream: the index of the best TTA intervention for a fixed source model.

Hanneke et al. analyze model selection and adaptation to unknown transfer relationships and oracle rates (Hanneke et al., 2023). Dong et al. analyze the hardness of unsupervised domain adaptation through information about target labels and the attainable target predictor (Dong et al., 2026). Their settings are important theoretical neighbors, but neither directly gives our finite-batch KEEP/RECENTER boundary, our pre-action TTA observation-channel criterion, or the matchedstream invariance result.

Decision making under partial identification. The abstract principle that an optimal decision or policy ranking can sometimes be determined without identifying every payoff is well established in statistics and econometrics (Kasy, 2016; Pu & Zhang, 2021; Han, 2024; Christensen et al., 2026). We therefore do not present Action-ID, the two-point lower bound, or the TV inequality as new general decision theory. We use that machinery as a foundation and concentrate the contribution on TTA-specific information boundaries: data-dependent interventions, finite adaptation batches, shift-mechanism reversals, and observation channels that can erase or preserve deployment structure.

## 7 DISCUSSION AND LIMITATIONS

Our results expose a failure mode that is easy to miss when TTA selection is treated only as a prediction problem. A selector can fail because its regression model is weak, but it can also fail because the evidence it receives makes deployments with different action rankings indistinguishable. The second failure cannot be repaired by simply increasing selector capacity.

This matters for current TTA practice. Strong adaptation methods and increasingly sophisticated selection criteria do not remove the need to ask what information reaches the selector. On CIFAR-100-C and DomainNet-126, the same selected images can require different actions after only the stream organization changes. Global evidence can therefore be insufficient even when it contains hundreds of source-output statistics. Conversely, a small stream-aware summary can recover the missing decision signal when order is the relevant hidden variable.

The conclusion is also not that order-aware evidence is always better. Our stochastic experiment shows the opposite: when the deployment family changes, the large controlled advantage can disappear. This makes the claim stronger and more useful, not weaker. Reliable TTA selection requires matching the observation channel to the deployment variations that can change the action ranking.

There are clear limits. The theory uses a finite action menu and a simple one-dimensional Gaussian model. The Gaussian RECENTER action is a mechanism study, not a literal model of every modern TTA algorithm. The experiments use one main source-model family, and the DomainNet action boundary is simpler than CIFAR because correlated worlds strongly favor SOURCE. Broader architectures and action menus are natural future tests.

Finally, the translation/prior-shift construction is intentionally a coarse-evidence result. The two full target distributions differ; only the specified population-mean channel is shared. Richer observations may resolve the ambiguity. That is the central lesson: identifiability belongs to the pair (deployment family, observation channel). The practical design question is therefore not only “which selector should we train?” but also “what information must that selector be allowed to see for the desired TTA decision to be possible?”

## REPRODUCIBILITY STATEMENT

The experiment notebooks freeze the target worlds, actions, random seeds, evidence definitions, selectors, and statistical tests. Full mathematical proofs are included in Appendix A, and the complete experimental audit is summarized in Appendix B. The final anonymous submission will include the code and frozen result artifacts required to reproduce the reported tables.

## AI USE STATEMENT

Generative AI tools were used during the research workflow for code debugging, organization, mathematical checking, and language editing. The authors remain responsible for every theorem assumption, proof, experiment, numerical claim, citation, and manuscript statement.

## ACKNOWLEDGEMENTS

This research is supported by the Ministry of Education, Singapore, under its funding for the Research Centre of Excellence award to the Institute for Digital Molecular Analytics & Science (IDMxS), NTU. Project No. EDUNC-33-18-279-V12-IDMxS.

## REFERENCES

Shai Ben-David, Tyler Lu, Teresa Luu, and David Pal. Impossibility theorems for domain adaptation. In Proceedings ofthe Thirteenth International Conference on Artificial Intelligence and Statistics, volume 9 of Proceedings ofMachine Learning Research, pp. 129–136. PMLR, 2010.

Timothy Christensen, Hyungsik Roger Moon, and Frank Schorfheide. Optimal decision rules when payoffs are partially identified. The Review of Economic Studies, 2026. doi: 10.1093/restud/ rdag017.

Sebastian Cygert, Damian Sojka, Tomasz Trzci´ nski, and Bartłomiej Twardowski. Realistic evaluation´ of test-time adaptation: Unsupervised model selection. In Proceedings ofthe 21st International Conference on Computer Vision Theory and Applications (VISAPP), pp. 75–86. SciTePress, 2026. doi: 10.5220/0014320100004084.

Michal Danilowski, Alexander Murphy, Young D. Kwon, and Abhirup Ghosh. Morpheus: Meta test-time adaptation via neural collapse geometry. In ICLR 2026 Workshop on Test-Time Updates, 2026. URL https://openreview.net/forum?id=YfyApJH1Rj.

Zhiyi Dong, Zixuan Liu, and Yongyi Mao. On the hardness of unsupervised domain adaptation: Optimal learners and information-theoretic perspective. In Proceedings ofthe 4th Conference on Lifelong Learning Agents, volume 330 of Proceedings ofMachine Learning Research, pp. 89–111. PMLR, 2026.

Saurabh Garg, Sivaraman Balakrishnan, Zachary C. Lipton, Behnam Neyshabur, and Hanie Sedghi. Leveraging unlabeled data to predict out-of-distribution performance. In International Conference on Learning Representations, 2022.

Taesik Gong, Jongheon Jeong, Taewon Kim, Yewon Kim, Jinwoo Shin, and Sung-Ju Lee. Note: Ro bust continual test-time adaptation against temporal correlation. In Advances in Neural Information Processing Systems, volume 35, 2022.

Ishaan Gulrajani and Tatsunori Hashimoto. Identifiability conditions for domain adaptation. In Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings ofMachine Learning Research, pp. 7982–7997. PMLR, 2022.

Sukjin Han. Optimal dynamic treatment regimes and partial welfare ordering. Journal ofthe American Statistical Association, 119(547):2000–2010, 2024. doi: 10.1080/01621459.2023.2238941.

Steve Hanneke, Samory Kpotufe, and Yasaman Mahdaviyeh. Limits of model selection under transfer learning. In Proceedings of the 36th Conference on Learning Theory, volume 195 of Proceedings ofMachine Learning Research, pp. 5781–5812. PMLR, 2023.

Dan Hendrycks and Thomas Dietterich. Benchmarking neural network robustness to common corruptions and perturbations. In International Conference on Learning Representations, 2019.

Maximilian Kasy. Partial identification, distributional preferences, and the welfare ranking of policies. The Review ofEconomics and Statistics, 98(1):111–131, 2016. doi: 10.1162/REST a 00528.

Eungyeup Kim, Mingjie Sun, Christina Baek, Aditi Raghunathan, and J. Zico Kolter. Test-time adaptation induces stronger accuracy and agreement-on-the-line. In Advances in Neural Information Processing Systems, volume 37, 2024. doi: 10.52202/079017-3820.

Lingjing Kong, Shaoan Xie, Weiran Yao, Yujia Zheng, Guangyi Chen, Petar Stojanov, Victor Akinwande, and Kun Zhang. Partial identifiability for domain adaptation. In Proceedings ofthe 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pp. 11455–11472. PMLR, 2022.

Jonghyun Lee, Dahuin Jung, Saehyung Lee, Junsung Park, Juhyeon Shin, Uiwon Hwang, and Sungroh Yoon. Entropy is not enough for test-time adaptation: From the perspective of disentangled factors. In International Conference on Learning Representations, 2024a. URL https://openreview. net/forum?id=9w3iw8wDuE.

Taeckyung Lee, Sorn Chottananurak, Taesik Gong, and Sung-Ju Lee. Aetta: Label-free accuracy estimation for test-time adaptation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 28643–28652, 2024b.

Hyesu Lim, Byeonggeun Kim, Jaegul Choo, and Sungha Choi. Ttn: A domain-shift aware batch normalization in test-time adaptation. In International Conference on Learning Representations, 2023.

Robert A. Marsden, Mario Dobler, and Bin Yang. Universal test-time adaptation through weight ¨ ensembling, diversity weighting, and prior correction. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 2555–2565, 2024.

Alexander Murphy, Michal Danilowski, Soumyajit Chatterjee, and Abhirup Ghosh. Neo — nooptimization test-time adaptation through latent re-centering. In International Conference on Learning Representations, 2026.

Ignavier Ng, Yujia Li, Zhixuan Li, Yujia Zheng, Guangyi Chen, and Kun Zhang. A general representation-based approach to multi-source domain adaptation. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 45911–45933. PMLR, 2025.

Shuaicheng Niu, Jiaxiang Wu, Yifan Zhang, Yaofo Chen, Shijian Zheng, Peilin Zhao, and Mingkui Tan. Efficient test-time model adaptation without forgetting. In Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pp. 16888–16905. PMLR, 2022.

Shuaicheng Niu, Jiaxiang Wu, Yifan Zhang, Zhiquan Wen, Yaofo Chen, Peilin Zhao, and Mingkui Tan. Towards stable test-time adaptation in dynamic wild world. In International Conference on Learning Representations, 2023.

Xingchao Peng, Qinxun Bai, Xide Xia, Zijun Huang, Kate Saenko, and Bo Wang. Moment matching for multi-source domain adaptation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 1406–1415, 2019.

Hongming Pu and Bo Zhang. Estimating optimal treatment rules with an instrumental variable: A partial identification learning approach. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 83(2):318–345, 2021. doi: 10.1111/rssb.12413.

Afshar Shamsi, Xiao-Yu Guo, Hamid Alinejad-Rokny, Arash Mohammadi, Damien Teney, and Ehsan Abbasnejad. Multi-hypothesis test-time adaptation to mitigate underspecification. arXiv preprint arXiv:2607.00259, 2026.

Sudarshan Sreeram, Young D. Kwon, and Cecilia Mascolo. Tempora: Characterising the timecontingent utility of online test-time adaptation. In International Conference on Machine Learning, 2026.

Dequan Wang, Evan Shelhamer, Shaoteng Liu, Bruno Olshausen, and Trevor Darrell. Tent: Fully testtime adaptation by entropy minimization. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=uXl3bZLkr3c.

Hao Zhao, Yuejiang Liu, Alexandre Alahi, and Tao Lin. On pitfalls of test-time adaptation. In Proceedings ofthe 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pp. 42058–42080. PMLR, 2023.

Zhi Zhou, Ming Yang, Shi-Yu Tian, Kun-Yang Yu, Lan-Zhe Guo, and Yu-Feng Li. On the learnability of test-time adaptation: A recovery complexity perspective. In Proceedings of the 43rd International Conference on Machine Learning, 2026.

## A APPENDIX

This appendix gives the formal steps. Before each proof we first state the idea in plain language.

## A.1 PROOF OF THEOREM 1

Idea. The selector receives statistically identical evidence in both worlds. Therefore it must use the same action probabilities in both. Since the two correct actions are different, the probability mass assigned to “correct in world $0 ^ { \circ }$ and “correct in world $1 ^ { \circ }$ cannot add to more than one.

Let $p _ { j } ( z )$ be the selector’s conditional probability of outputting $a _ { j } ^ { * }$ after observing $z ,$ including any internal randomization. Because the two evidence laws are the same, call the common law $Q .$ . Then

$$
\mathbb { P } _ { \theta _ { 0 } } \big ( s ( Z _ { n } ) = a _ { 0 } ^ { * } \big ) + \mathbb { P } _ { \theta _ { 1 } } \big ( s ( Z _ { n } ) = a _ { 1 } ^ { * } \big ) = \int \big [ p _ { 0 } ( z ) + p _ { 1 } ( z ) \big ] d Q ( z ) \le 1 ,
$$

because $a _ { 0 } ^ { * } \neq a _ { 1 } ^ { * }$ implies $p _ { 0 } ( z ) + p _ { 1 } ( z ) \le 1$ for every z. Under an equal prior, average correctness is therefore at most $1 / 2$ , so average action error is at least $1 / 2$ . If every wrong action has excess risk at least $\Delta ,$ , regret is at least $\Delta$ whenever the selector is wrong, giving average regret at least $\Delta / 2$ . □

Explicit Discrete Two-World Construction. To instantiate Theorem 1 with an explicit discrete example, let $\mathcal { X } = \{ x _ { 1 } , x _ { 2 } , x _ { 3 } \}$ } with identical target marginal distribution $P ( X ) = ( 0 . \bar { 4 } 5 , 0 . 2 5 , 0 . 3 0 )$ in both worlds. Consider two candidate classifiers:

$$
\begin{array} { r l } { f _ { \mathrm { K E F P } } ( x _ { 1 } ) = 1 , \ } & { f _ { \mathrm { K E E P } } ( x _ { 2 } ) = 0 , \quad f _ { \mathrm { K E E P } } ( x _ { 3 } ) = 0 , } \\ { f _ { \mathrm { A D A P T } } ( x _ { 1 } ) = 0 , \ } & { f _ { \mathrm { A D A P T } } ( x _ { 2 } ) = 1 , \quad f _ { \mathrm { A D A P T } } ( x _ { 3 } ) = 0 . } \end{array}
$$

Let $\eta _ { \theta } ( x ) = \mathbb { P } _ { \theta } ( Y = 1 \mid X = x )$ define the target posterior label distribution:

$$
\begin{array} { r l } { { \mathrm { W o r l d ~ A } \mathrm { : } } } & { { } { \eta _ { A } = ( 0 . 2 5 0 , 0 . 6 7 5 , 0 . 1 8 7 5 ) , } } \\ { { \mathrm { W o r l d ~ B } \mathrm { : } } } & { { } { \eta _ { B } = ( 0 . 7 5 0 , 0 . 3 1 5 , 0 . 0 6 2 5 ) . } } \end{array}
$$

Under 0-1 loss $\begin{array} { r } { R _ { \theta } ( f ) = \sum _ { i = 1 } ^ { 3 } P ( X = x _ { i } ) \mathbb { P } _ { \theta } ( f ( x _ { i } ) \neq Y \mid X = x _ { i } ) } \end{array}$ , the exact prospective risks evaluate to:

$$
R _ { A } ( \mathrm { K E E P } ) = 0 . 4 5 ( 1 - 0 . 2 5 0 ) + 0 . 2 5 ( 0 . 6 7 5 ) + 0 . 3 0 ( 0 . 1 8 7 5 ) = 0 . 5 6 2 5 ,
$$

$$
R _ { A } ( \mathrm { A D A P T } ) = 0 . 4 5 ( 0 . 2 5 0 ) + 0 . 2 5 ( 1 - 0 . 6 7 5 ) + 0 . 3 0 ( 0 . 1 8 7 5 ) = 0 . 2 5 0 0 ,
$$

$$
R _ { B } ( \mathrm { K E E P } ) = 0 . 4 5 ( 1 - 0 . 7 5 0 ) + 0 . 2 5 ( 0 . 3 1 5 ) + 0 . 3 0 ( 0 . 0 6 2 5 ) = 0 . 2 1 0 0 ,
$$

$$
R _ { B } ( \mathrm { A D A P T } ) = 0 . 4 5 ( 0 . 7 5 0 ) + 0 . 2 5 ( 1 - 0 . 3 1 5 ) + 0 . 3 0 ( 0 . 0 6 2 5 ) = 0 . 5 2 7 5 .
$$

Thus, $\mathcal { A } ^ { * } ( \mathrm { W o r l d \ A } ) = \{ \mathrm { A D A P T } \}$ while $\mathcal { A } ^ { * } ( \mathrm { W o r l d \ B } ) = \{ \mathrm { K E E P } \}$ . Since the observation channel $Z _ { n } \sim P ( { \dot { X } } ) ^ { \otimes n }$ is statistically identical in both worlds, any pre-adaptation selector incurs an average action error of at least $1 / 2$ and an expected regret of at least $\Delta / 2 \stackrel { - } { = } 0 . 1 5 8 7 5$

## A.2 PROOF OF COROLLARY 1

Idea. When the two evidence laws are not identical, the selector can partly tell them apart. The best possible two-way discrimination advantage is controlled by total variation. Whatever ambiguity remains becomes an unavoidable action-selection error.

Let $A _ { 0 }$ be the event that the selector outputs $a _ { 0 } ^ { * } .$ Since $a _ { 1 } ^ { * } \neq a _ { 0 } ^ { * } .$

$$
\mathbb { P } _ { 0 } ( \mathrm { c o r r e c t } ) + \mathbb { P } _ { 1 } ( \mathrm { c o r r e c t } ) \le Q _ { 0 } ( A _ { 0 } ) + Q _ { 1 } ( A _ { 0 } ^ { c } ) .
$$

The right side is

$$
1 + Q _ { 0 } ( A _ { 0 } ) - Q _ { 1 } ( A _ { 0 } ) \leq 1 + \mathrm { T V } ( Q _ { 0 } , Q _ { 1 } ) .
$$

Average correctness is therefore at most $( 1 + \mathrm { T V } ) / 2$ , so average error is at least $( 1 - \mathrm { T V } ) / 2$ Multiplying by the minimum wrong-action cost $\Delta$ gives the regret bound. □

## A.3 PROOF OF THEOREM 2

Idea. If all worlds compatible with the evidence share one optimal action, choose it. If they do not share any optimal action, no single decision based only on that same evidence law can be optimal in every compatible world.

If $\textstyle a \in \bigcap _ { \theta \in \Theta ( Q ) } { \mathcal { A } } ^ { * } ( \theta )$ , then the decision rule $d ( Q ) = a$ has zero oracle regret in every compatible world. Conversely, if the intersection is empty, every action fails to be optimal in at least one compatible world. Hence no action-valued function of $Q$ can have zero oracle regret simultaneously across the whole compatible set. □

## A.4 PROOF OF PROPOSITION 1

Idea. An invariant observation maps a batch and its transformed version to exactly the same value. Therefore the transformed worlds have the same evidence law. If their best actions differ, Theorem 1 applies.

Let the raw batch in world θ be B. The transformed world uses the pushforward distribution of $g B$ Since $\phi _ { n } ( g B ) = \phi _ { n } ( B )$ pointwise, both worlds induce the same evidence law. Different unique oracle actions then imply non-identifiability by Theorem 1. □

## A.5 PROOF OF THEOREM 3

Step 1: write the KEEP risk as one function. Define

$$
g ( t ) = { \frac { 1 } { 2 } } \left[ \Phi { \left( - { \frac { \mu + t } { \sigma } } \right) } + \Phi { \left( { \frac { t - \mu } { \sigma } } \right) } \right] .
$$

This is exactly the error of the fixed source threshold when the target translation is t, so $R _ { K } ( \delta ) = g ( \delta )$ The function is even: a left shift and an equally large right shift hurt the symmetric source classifier by the same amount. For $t > 0$

$$
g ^ { \prime } ( t ) = \frac { 1 } { 2 \sigma } \left[ \varphi \bigg ( \frac { t - \mu } { \sigma } \bigg ) - \varphi \bigg ( \frac { t + \mu } { \sigma } \bigg ) \right] > 0 .
$$

The inequality holds because $| t - \mu | < t + \mu$ and the standard normal density decreases as $| x |$ moves away from zero. Therefore g strictly increases with $| t | .$ Also

$$
g ( 0 ) = \Phi ( - \mu / \sigma ) , \qquad \operatorname * { l i m } _ { t \to \infty } g ( t ) = 1 / 2 .
$$

Step 2: understand what RECENTER leaves behind. The estimate is

$$
\widehat { \delta } = \bar { X } _ { n } = \delta + \frac { 1 } { n } \sum _ { i = 1 } ^ { n } ( \mu Y _ { i } + \varepsilon _ { i } ) .
$$

Define the estimation error

$$
E _ { n } = { \widehat { \delta } } - \delta .
$$

After moving the threshold by ${ \widehat { \delta } } ,$ the true translation $\delta$ is cancelled and only the random estimation error remains. Because the evaluation point is independent of the adaptation batch,

$$
R _ { R } ( n ) = \mathbb { E } [ g ( E _ { n } ) ] .
$$

For every finite $n , E _ { n }$ is non-degenerate, so $E _ { n } \neq 0$ almost surely. Since $g ( t ) > g ( 0 )$ whenever $t \neq 0$

$$
R _ { R } ( n ) > g ( 0 ) .
$$

On the other hand $g ( t ) < 1 / 2$ for every finite t, hence $R _ { R } ( n ) < 1 / 2 .$

Because $g$ is continuous and strictly increasing on $t > 0$ , there is exactly one positive value $\delta _ { c } ( n )$ with

$$
\begin{array} { r } { g ( \delta _ { c } ( n ) ) = R _ { R } ( n ) . } \end{array}
$$

Below that value KEEP has lower risk; above it RECENTER has lower risk. This proves the existence, uniqueness, and ordering parts of the theorem.

## Step 3: derive the exact finite-n RECENTER risk. Condition on

$$
S _ { n } = \sum _ { i = 1 } ^ { n } Y _ { i } = s .
$$

Then

$$
E _ { n } \mid S _ { n } = s \sim { \mathcal { N } } \left( { \frac { \mu s } { n } } , { \frac { \sigma ^ { 2 } } { n } } \right) .
$$

For a Gaussian random variable $Z \sim \mathcal { N } ( m , \tau ^ { 2 } )$ , the identity

$$
\mathbb { E } \left[ \Phi \left( \frac { Z - a } { \sigma } \right) \right] = \Phi \left( \frac { m - a } { \sqrt { \sigma ^ { 2 } + \tau ^ { 2 } } } \right)
$$

follows by introducing an independent standard normal variable and combining two Gaussian noises. Averaging over the binomial values $S _ { n } = 2 k - n$ gives Eq. equation 3. Symmetry of $E _ { n }$ makes the two class-error terms equal after expectation.

Step 4: find the large-n scale of the crossing. Near zero, g is smooth and even, so the first non-zero term is quadratic:

$$
g ( t ) = g ( 0 ) + c _ { 2 } t ^ { 2 } + O ( t ^ { 4 } ) , \qquad c _ { 2 } = { \frac { \mu \varphi ( \mu / \sigma ) } { 2 \sigma ^ { 3 } } } > 0 .
$$

The estimation error is a sample mean of $\mu Y + \varepsilon ,$ , whose variance is

$$
v = \mu ^ { 2 } + \sigma ^ { 2 } .
$$

Therefore

$$
\mathbb { E } [ E _ { n } ^ { 2 } ] = { \frac { v } { n } } , \qquad \mathbb { E } [ E _ { n } ^ { 4 } ] = O ( n ^ { - 2 } ) .
$$

Taking expectation in the Taylor expansion gives

$$
R _ { R } ( n ) = g ( 0 ) + c _ { 2 } { \frac { v } { n } } + O ( n ^ { - 2 } ) .
$$

At the crossing, $g ( \delta _ { c } ( n ) ) = R _ { R } ( n )$ . Since $\delta _ { c } ( n )  0$

$$
c _ { 2 } \delta _ { c } ( n ) ^ { 2 } + O ( \delta _ { c } ( n ) ^ { 4 } ) = c _ { 2 } { \frac { v } { n } } + O ( n ^ { - 2 } ) ,
$$

so

$$
\delta _ { c } ( n ) ^ { 2 } = \frac { v } { n } + { \cal O } ( n ^ { - 2 } ) .
$$

Taking the positive square root yields

$$
\delta _ { c } ( n ) = \frac { \sqrt { \mu ^ { 2 } + \sigma ^ { 2 } } } { \sqrt { n } } + { \cal O } ( n ^ { - 3 / 2 } ) .
$$

This proves the theorem.

## A.6 PROOF OF PROPOSITION 2

Idea. A change in class proportions moves the mixture mean and the Bayes decision boundary in opposite directions. Mean recentering follows the mixture mean, so it moves away from the Bayes correction.

Under pure prior shift, $\mathbb { P } ( Y = + 1 ) = \pi$ . For threshold t,

$$
R _ { \pi } ( t ) = \pi \Phi \left( \frac { t - \mu } { \sigma } \right) + ( 1 - \pi ) \Phi \left( \frac { - t - \mu } { \sigma } \right) .
$$

The Bayes-optimal threshold is

$$
t _ { B } = \frac { \sigma ^ { 2 } } { 2 \mu } \log \frac { 1 - \pi } { \pi } .
$$

$\operatorname { I f } \pi > 1 / 2$ , then $t _ { B } < 0$ , while the population mean

$$
m = \mu ( 2 \pi - 1 ) > 0 .
$$

The derivative of $R _ { \pi } ( t )$ changes sign only at $t _ { B }$ and is positive for $t > t _ { B }$ . Therefore $R _ { \pi } ( t )$ is strictly increasing from 0 to m, giving

$$
R _ { \pi } ( m ) > R _ { \pi } ( 0 ) .
$$

Thus moving the threshold to the target mean is worse than keeping it at zero. The case $\pi < 1 / 2$ follows by symmetry. □

## A.7 PROOF OF COROLLARY 2

Idea. We build two worlds with the same reported mean m. In one world the mean moved because the whole distribution translated, so recentering fixes the problem. In the other world the mean moved only because the class proportions changed, so recentering moves the threshold the wrong way.

In World C, balanced translation by m gives population mean m. Population RECENTER uses the true mean and returns the source risk $g ( 0 )$ , while KEEP has $g ( m ) > g ( 0 )$

In World P, choose

$$
\pi = \frac { 1 + m / \mu } { 2 } ,
$$

which is valid for $| m | < \mu$ . Then the target mean is also

$$
\mu ( 2 \pi - 1 ) = m .
$$

But Proposition 2 gives

$$
R _ { \pi } ( 0 ) < R _ { \pi } ( m ) ,
$$

so KEEP is better than population RECENTER. Thus the same population mean corresponds to opposite best actions.

For the finite-batch RECENTER action, the opposite-action conclusion also holds for all sufficiently large n. In World C, Theorem 3 gives $\delta _ { c } ( n ) \to 0$ , so a fixed nonzero m eventually lies above the boundary. In World P, the sample mean converges almost surely to m, and the bounded fresh-point classification loss converges to the strictly worse population-RECENTER risk. □

## B ADDITIONAL EXPERIMENTAL DETAILS

This appendix records the frozen reviewer checks behind the concise main-text tables. No result below is chosen by selecting the most favorable random seed or window after seeing the outcome.

## B.1 CIFAR-100-C: COMPLETE CONTROLLED STUDY

The controlled benchmark has 675 worlds: 15 corruption types × three severities × five deployment regimes × three seeds. The five actions are SOURCE, DeYO, Tent, ROID, and test-time normalization. The oracle counts are SOURCE 339, ROID 277, Tent 34, and DeYO 25 worlds. The matched

IID/correlated experiment contains 270 pairs with exactly the same selected images; 157/270 change oracle action and 82 are strong flips.

The frozen Z3 selector has 0.312 pp mean oracle regret in leave-one-corruption-out evaluation. The final author-confirmed MORPHEUS MultiOutput-RF baselines are 4.135 pp for entropy, 1.170 pp for the text Neural-Collapse definition, and 1.358 pp for the appendix Neural-Collapse definition. Relative to Z3, the mean paired regret differences are approximately +3.823, +0.857, and +1.046 pp; the corresponding transfer-level tests remain significant after Holm correction $( p \approx 3 . 6 6 \times 1 0 ^ { - 4 } )$ .

On the 82 strong same-image flips, frozen Z3 changes its selected action in 92.7% of pairs with 0.093 pp mean pair regret. The author-confirmed MORPHEUS entropy, NC-text, and NC-appendix selectors change action in 0% of these pairs and have mean pair regrets 6.189, 2.873, and 2.921 pp.

Reviewer checks. The frozen audit includes exact/tie-aware oracle metrics; five independent RF seeds; feature-dimension control; paired bootstrap confidence intervals; paired significance tests; conventional untouched CIFAR-100-C sanity checks; and a batch-size frontier. Compressing Z3 to five dimensions is worse than MORPHEUS, while increasing retained information improves performance and roughly 64 dimensions becomes competitive/better. This is useful evidence that the result depends on preserving decision-relevant information rather than simply naming one representation “better.”

The batch-size frontier also confirms that the oracle action itself can move as execution scale changes. Relative to the B = 200 reference, action-switch rates are 50.0% at B = 20, 31.25% at B = 50, 16.67% at B = 100, and 29.17% at B = 500 in the audited subset.

## B.2 CIFAR-100-C: STOCHASTIC COUNTER-RESULT

The controlled stream construction is not the only deployment family we test. Under randomized stochastic deployments, the large Z3 advantage disappears. When selectors are trained and evaluated within the stochastic family, mean regret is 0.210 pp for MORPHEUS NC-text, 0.293 pp for NCappendix, and 0.338 pp for Z3. We retain this result because it rejects a universal-superiority interpretation of Z3 and supports the observation-channel/deployment-family framing.

## B.3 CYGERT ET AL.: STRONGER INFORMATION BUDGET

The Cygert et al. native source/model/data sanity reproduces their published CIFAR-100-C SOURCE behavior closely. Their released evaluation pipeline does not expose every ENT/CON/SND statistic cleanly in our reproducible run, and some released runner paths are inconsistent for candidate methods. We therefore do not alter their scientific source code to force a same-table result. More importantly, their model-selection information is available after candidate adaptations have been executed, whereas our main setting chooses from pre-action evidence. We report this as a stronger-information-budget neighboring protocol rather than a direct same-budget baseline.

## B.4 DOMAINNET-126: FROZEN NATURAL-DOMAIN STUDY

The study uses six directional transfers: real→clipart, real→painting, real→sketch, clipart→sketch, clipart→real, and clipart→painting. With three seeds and five regimes this gives 90 worlds. Each world uses 2,000 images without replacement. All five TTA actions run at execution batch size 32.

The oracle action is SOURCE in 41 worlds, ROID in 48, and DeYO in one. All 36 correlated worlds prefer SOURCE, while the 54 IID worlds are mostly ROID. This makes DomainNet a deliberately clean mechanism confirmation rather than the most difficult possible selector benchmark.

Matched IID/correlated worlds share the same selected images and class quota. The oracle changes in 33/36 pairs and 27 are strong flips. Final numerical checks align samples by original dataset index and verify SOURCE-prediction and SOURCE-error consistency. Canonical permutation invariance is checked on the same saved tensor rather than by demanding bitwise equality between separate GPU forwards.

Main and compact evidence. Across all 90 worlds, global Z2 has 1.933 pp mean regret and 45.6% exact action accuracy. Original Z3 RF has 0.034 pp regret and 93.3% action accuracy. A CPU-only audit reconstructs the original $W = 2 0 0 Z 2 / Z 3$ evidence bitwise exactly, then evaluates the alreadypredeclared 11-number ORDER11 summary at windows 32, 64, 128, and 200. ORDER11-RF gives 0.034 pp/93.3% at $W = 3 2$ , 0.106 pp/91.1% at W = 64, 0.034 pp/93.3% at $W = 1 2 8 ,$ and 0.034 pp/93.3% at $W = 2 0 0$ . Thus the execution-scale W = 32 result is identical to the original W = 200 point prediction.

Adding global Z2 to ORDER11 does not improve the W = 32, 128, or 200 point metrics, which supports the interpretation that stream structure is the decisive signal in this construction. On the 27 strong matched flips, ORDER11 changes action in all 27 with zero mean pair regret at every tested window.

Linear-model check. Nested Ridge regression chooses regularization using only the training transfers. With ORDER11 at $W = 3 \bar { 2 }$ , Ridge has 0.061 pp mean regret, 92.2% exact action accuracy, 96.7% within 0.5 pp of oracle, and no 5-pp catastrophic cases. On the 27 strong matched flips it changes action in 26/27 pairs and has 0.044 pp mean pair regret. The same Ridge architecture on global Z2 remains at 1.933 pp regret, 45.6% action accuracy, and 0/27 action changes.

Robustness and generalization checks. Five RF seeds give identical main Z3 metrics. Fold-only PCA at 5, 16, 32, and 64 dimensions gives the same 0.034 pp/93.3% point result on DomainNet, showing that the mechanism is low-dimensional there. Leave-one-target-domain-out regret is 0.180 pp for clipart, 0 for painting, 0.027 pp for real, and 0 for sketch. Leave-one-source-domain-out is harder for the clipart-source family (1.171 pp regret) but still improves substantially over SOURCE (2.522 pp). The real-source conventional native sequence also behaves normally: ROID improves over SOURCE on the standard domain sequence, confirming that the severe correlated-world adaptation failures are not simply a broken implementation.

## B.5 WHAT IS DELIBERATELY NOT CLAIMED

The studies do not claim that Z3 or ORDER11 is a universally best selector. They also do not claim that every TTA action reversal is caused by order. Their purpose is narrower: to show concrete public-benchmark deployment families where a restricted observation channel collapses worlds with different oracle actions, and to show that adding the missing kind of information can recover the decision in those families.