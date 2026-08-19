# WHEN MORE FOUNDATION MODELS MEANS LESS: DI-AGNOSING AND ADDRESSING MULTI-VIEW FUSION FAILURE

Yibo Liu<sup>∗</sup> Bowen Jiang   
Hainan Campus, Beijing University of Posts and Telecommunications   
Lingshui, Hainan, China   
liuyibo@bupt.edu.cn 1729773236@bupt.edu.cn

## ABSTRACT

Multi-view learning is motivated by the assumption that complementary views can provide richer and more discriminative representations. With the growing availability of pretrained foundation-model encoders, candidate views are no longer limited to a small, fixed set of sensors or hand-crafted features. In this setting, each frozen pretrained encoder can induce a distinct representation view, and practitioners may choose from a large, heterogeneous pool of candidate encoders. The design problem therefore shifts from fusing a small, predefined set of views to selecting which encoders to combine, and how many, from a large candidate pool. Empirically, we find that downstream performance does not improve monotonically as more encoders are fused: gains typically saturate after a small subset of encoders is included. These observations motivate explicitly selecting both the composition and the size of the fused view set. We formulate this problem as view-set composition: jointly determining which candidate views to fuse and how many to include. We then propose KAGES (Kernel-Alignment Greedy Encoder Selector), a labelaware selector that requires no downstream classifier training during selection and greedily adds the encoder yielding the largest marginal improvement in alignment between the fused representation kernel and the label kernel. KAGES consistently outperforms competing selection methods as well as full fusion, and approaches exhaustive oracle selection. We further observe similar selection gains in image retrieval and in the fusion of frozen language-model representations. Overall, our results show that effective multi-view learning over foundation-model pools depends not on fusing more encoders, but on selecting a compact and complementary set of views. Code and results are available at https://github.com/ yibol9768-alt/Quantifying-Representation-Reliability.

## 1 INTRODUCTION

Multi-view learning has traditionally assumed a small, predefined set of views arising from different sensors, modalities, or feature representations, motivated by the premise that complementary views can provide information beyond that available from any single view (Xu et al., 2013; Yan et al., 2021; Trosten et al., 2021). Foundation models fundamentally expand the space of candidate views. In the frozen-encoder setting, each pretrained encoder induces a distinct representation of the same input, and practitioners can draw from large, heterogeneous model pools spanning architectures, pretraining objectives, supervision sources, and modalities (Radford et al., 2021; Oquab et al., 2024; He et al., 2022; Liu et al., 2022; Dubey et al., 2024; Yang et al., 2024). The view set therefore ceases to be merely a fixed input to the learning problem and becomes a design variable itself: beyond deciding how to fuse a predefined set of views, one must determine which encoder-induced views to combine and how many to retain. We formulate this problem as view-set composition. While complementarity motivates combining multiple views, it does not by itself provide an operational criterion for selecting a subset from a large heterogeneous pool or for determining when additional views cease to be beneficial.

![](images/761cb2b4fb55dccb91598de711a710657f866bf06ae352a6aa2c7346302812b8.jpg)  
Figure 1: View-set composition in frozen foundation-model fusion. Left: a heterogeneous pool of frozen vision backbones. Middle: downstream accuracy is non-monotonic in the number of fused views, with most datasets peaking at 3 to 5 encoders and declining, while a single task-aligned encoder beats every fusion on Country211. Right: two failure modes behind the decline. Type I (representation redundancy): a new encoder repeats information already captured. Type II (taskmodel misalignment): a representationally distinct encoder injects task-irrelevant cues, so diversity alone is not effectiveness. The marginal-utility theory of Section 2.2 explains the shape from a single gain-cost crossing (Theorem 1); the two failure modes are formalised in supplementary App. A.3.6.

Recent work on multi-encoder vision systems, including Eagle (Shi et al., 2025), Cambrian-1 (Tong et al., 2024), BRAVE (Kar et al., 2024), and MoVA (Zong et al., 2024), has demonstrated the value of combining representations from multiple pretrained encoders, with some methods further introducing routing or search mechanisms to determine how encoder features are used. These methods typically focus on designing effective fusion or routing strategies over a fixed or manually specified set of encoders, leaving view-set composition over large heterogeneous encoder pools largely unexplored.

The importance of view-set composition becomes apparent as the encoder pool grows. Even along a deliberately favorable fusion path that places previously successful and complementary encoders early, downstream accuracy is often non-monotonic as additional views are added. Encoders added early in the fusion path can improve performance, whereas later encoders may contribute information already captured by the current set or signals that are only weakly aligned with the target task. In either case, the fused representation gains additional view-specific information without a commensurate increase in task-relevant signal, so adding more encoders can ultimately reduce rather than improve downstream accuracy.

This observation does not imply that effective fusion requires a small fixed number of encoders. Rather, it shows that the peak observed along a fusion path is order-dependent. A fixed encoderaddition path may include informative views early and redundant or task-misaligned views later; a task-aware selector can instead prioritize high-value views while deferring less beneficial additions. We therefore cast view-set composition as an ordering problem: a high-quality ordering should prioritize useful encoders so that its prefixes provide competitive fused sets under different compute budgets.

Existing selection strategies are insufficient for this goal. Per-encoder transferability scores (You et al., 2021; Bao et al., 2019; Pandy et al., 2022) evaluate individual encoders independently and ignore their interactions with already-selected encoders. Diversity-based criteria encourage representational difference, but different is not necessarily useful for the downstream task. Exhaustive or validationbased subset search can identify high-performing combinations, but requires training downstream predictors for many candidate subsets and does not scale well to large encoder pools. What is needed is a training-free, set-level criterion that scores the marginal utility of each candidate relative to the current fused set.

To address this challenge, we propose KAGES (Kernel-Alignment Greedy Encoder Selector), a labelaware method for view-set composition over frozen foundation-model encoders. KAGES defines the utility of a fused set as the alignment between its representation kernel and the label kernel. At each step, it adds the candidate encoder with the largest marginal gain in this set-level utility. The resulting procedure produces a task-aligned encoder ordering without requiring downstream classifier training, combinatorial subset search, or fusion-architecture tuning.

The kernel-alignment objective admits a natural relevance–redundancy interpretation. In particular, the marginal gain favors candidates that are well aligned with the labels while exhibiting limited kernel overlap with already-selected encoders. Under linear-kernel fusion by feature concatenation, the fused kernel is additive across views, allowing KAGES to evaluate marginal gains directly at the kernel level rather than retraining a downstream model for each candidate subset (Section 3.2). The same set-level formulation also admits a conditional fixed-budget guarantee: if the pure alignment utility is monotone and has submodularity ratio at least γ, then, for any budget k, the greedy prefix of size k achieves a $( 1 - e ^ { - \gamma } )$ approximation to the optimal size-k subset (Theorem 2, Corollary 1).

Extensive evaluations on image-classification benchmarks show that KAGES consistently outperforms competing multi-view selection methods and full fusion, substantially improving over diversity-based selection while approaching oracle selection. These gains extend beyond classification to image retrieval and frozen large-language-model fusion.

## Contributions.

• A new framing. We recast large-pool frozen foundation-model fusion as multi-view learning at scale and formulate view-set composition—which encoders to fuse and how many—as a central design problem.

• A unified characterisation, with theory. We characterise the failure of the more-is-better assumption in this regime: a gain–cost crossing (Theorem 1) provides a mechanism for an order-dependent peak, while empirical downstream performance subsequently plateaus and, on structured-symbol tasks, can decline even under an oracle ordering.

• A scalable, set-level selection method. We propose KAGES, a label-aware selector that evaluates candidate encoders through their marginal contribution to a set-level kernel-alignment objective, avoiding repeated fitting of downstream predictors for candidate subsets. After per-encoder kernel construction, each candidate is evaluated in $O ( n ^ { 2 } )$ time per greedy step, independent of encoder dimensionality. Under the stated monotonicity and submodularity-ratio conditions, each size-k greedy prefix achieves a conditional $\left( 1 - e ^ { - \gamma } \right)$ approximation to the optimal size-k subset. KAGES remains effective in our largest evaluated pools of $M = 1 5$ candidates.

• Broad empirical validation. We validate KAGES across image classification, image retrieval, and frozen large-language-model fusion, and observe the peak-then-decline phenomenon in both image classification and frozen-LLM fusion. Across these settings, KAGES produces stronger prefixes than diversity-based selection and often outperforms full fusion, with the largest gains in the main classification protocols, while approaching oracle selection.

## 2 PEAK-THEN-DECLINE AND THE NEED FOR VIEW-SET COMPOSITION

We now make the failure of complementarity precise. Once redundancy and finite-sample cost overtake the fresh task signal a new encoder brings along an ordering, every further view lowers the accuracy along that path. We first cast large-pool fusion as the search for a high-quality ordering (Section 2.1), then give a marginal-utility account in which a single gain–cost crossing forces an order-dependent peak (Section 2.2, Theorem 1). We then document the resulting peak-then-decline across five recognition regimes and three sample sizes on a path deliberately built to favour the “more is better” hypothesis (Sections 2.3–2.4); the path is not our method but evidence that the peak is structural and leaves a large improvement open, which KAGES realises in Section 3.

## 2.1 LARGE-POOL FUSION AS AN ORDERED PATH

Let $\mathcal { M } = \{ m _ { 1 } , . . . , m _ { M } \}$ be a fixed pool of frozen pre-trained encoders. For input $x ,$ encoder $m _ { i }$ produces a feature vector $F _ { m _ { i } } ( x ) \in \mathbb { R } ^ { d _ { m } }$ i . Given a downstream classification task with label $Y ,$ a subset $S \subseteq { \mathcal { M } }$ is combined into a fused feature $F _ { S } ( x )$ by a fusion operator (defined in Section 2.3);

a lightweight downstream classifier $f _ { \theta }$ is then trained on a labelled set of size n. For the theoretical analysis below, let

$$
{ \bar { F } } _ { S } ( x ) : = { \bigl ( } F _ { m } ( x ) { \bigr ) } _ { m \in S }
$$

denote the joint collection of the selected encoder features.

An ordering $\pi = ( \pi _ { 1 } , \ldots , \pi _ { K } )$ of $K \leq M$ encoders induces nested prefixes

$$
S _ { k } ^ { \pi } \ = \ \{ m _ { \pi _ { 1 } } , \ldots , m _ { \pi _ { k } } \} , \qquad k = 1 , \ldots , K , \qquad S _ { 0 } ^ { \pi } : = \emptyset .\tag{1}
$$

For this ordering, the peak accuracy along the path is

$$
P _ { n } ( \pi ) = \operatorname* { m a x } _ { 1 \leq k \leq K } \operatorname { A c c } _ { n } \left( S _ { k } ^ { \pi } \right) .\tag{2}
$$

The peak height $P _ { n } ( \pi )$ is a property of the ordering, not only of the underlying set: two permutations of the same encoders induce different nested prefixes and can therefore attain different peaks. Writing $R _ { n } ( S )$ for the corresponding 0–1 test risk, $\bar { \mathrm { A c c } } _ { n } ( S ) = 1 - R _ { n } ( S )$ , so maximising peak accuracy is equivalent to minimising the risk along the path. Section 2.2 writes this minimum path risk in terms of a maximum prefix sum of conditional net marginal utilities along π (Proposition 1). View-set composition is the problem of constructing an ordering whose prefix accuracy curve $k \mapsto \operatorname { A c c } _ { n } ( S _ { k } ^ { \pi } )$ stays as high as possible at every cardinality.

## 2.2 A MARGINAL-UTILITY VIEW OF PEAK-THEN-DECLINE

For a fixed ordering π, the performance peak is determined by the sequence of marginal gains and costs along its prefixes. Writing $S _ { k } = S _ { k } ^ { \bar { \pi } }$ and $m _ { k } : = m _ { \pi _ { k } }$ for brevity, so that $S _ { k } = { \bar { S } } _ { k - 1 } { \bar { \cup } } \left\{ m _ { k } \right\}$ decompose the finite-sample test risk into Bayes and excess-risk parts,

$$
\begin{array} { r } { R _ { n } ( S ) = R ^ { \star } ( S ) + E _ { n } ( S ) , \quad g _ { k } : = R ^ { \star } ( S _ { k - 1 } ) - R ^ { \star } ( S _ { k } ) \geq 0 , \quad c _ { k } : = E _ { n } ( S _ { k } ) - E _ { n } ( S _ { k - 1 } ) . } \end{array}\tag{3}
$$

where $R ^ { \star } ( S )$ is the Bayes risk given the joint representation $\bar { F } _ { S }$ , and $E _ { n } ( S )$ is the corresponding excess-risk term. Subtraction gives the one-step identity

$$
R _ { n } ( S _ { k } ) - R _ { n } ( S _ { k - 1 } ) = c _ { k } - g _ { k } :
$$

adding encoder $m _ { k }$ improves performance exactly when $g _ { k } > c _ { k }$ (supplementary App. A.3.2). Under log loss, the Bayes gain is a conditional mutual information,

$$
g _ { k , \mathrm { l o g } } \ = \ I ( F _ { m _ { k } } ; Y \mid \bar { F } _ { S _ { k - 1 } } ) ,\tag{4}
$$

the information-theoretic counterpart of the algebraic expansion in Proposition 2. This identity shows why ordering matters. The same candidate can have high gain when added before redundant views and low gain after its information has already been covered. A task-aware selector therefore raises the attainable peak by arranging the path so that large positive marginal utilities appear early.

The Bayes gain is non-increasing along a submodular greedy path (supplementary $\mathrm { A p p . \mathbf { A . } 1 } )$ . An OLS linear-probe instance gives

$$
\mathbb { E } R _ { n } ( S _ { k } ) = \nu _ { k } ^ { 2 } \frac { n - 1 } { n - k - 1 } ;
$$

once all signal coordinates have been included, the resulting pure-noise tail has non-decreasing excess-risk increments $c _ { k }$ at fixed n (supplementary App. A.3.3, Step 4). These observations motivate the monotonicity assumptions in the following conditional single-crossing result.

Theorem 1 (Order-specific gain–cost crossing). Fix an ordering π and its nested prefixes $S _ { 1 } ^ { \pi } , \ldots , S _ { K } ^ { \pi }$ Suppose $g _ { k }$ is non-increasing along this path, $c _ { k }$ is non-decreasing, and $a _ { k } : = g _ { k } - c _ { k }$ satisfies $a _ { 1 } > 0 , a _ { K } < 0$ , with no zero-margin ties (margin $\begin{array} { r } { \eta : = \operatorname* { m i n } _ { k } | a _ { k } | > 0 ) } \end{array}$ . Let

$$
k ^ { \star } ( \pi ) : = \operatorname* { m a x } \{ k : a _ { k } > 0 \} .
$$

Then $a _ { k }$ changes sign exactly once, between $k ^ { \star } ( \pi )$ and $k ^ { \star } ( \pi ) + 1$ , and, by the one-step identity,

$$
R _ { n } ( S _ { 0 } ^ { \pi } ) > R _ { n } ( S _ { 1 } ^ { \pi } ) > \cdots > R _ { n } ( S _ { k ^ { \star } ( \pi ) } ^ { \pi } ) < R _ { n } ( S _ { k ^ { \star } ( \pi ) + 1 } ^ { \pi } ) < \cdots < R _ { n } ( S _ { K } ^ { \pi } ) .
$$

Thus, risk decreases strictly up to $k ^ { \star } ( \pi )$ and increases strictly afterwards. The peak value attained by this path is order-specific: changing the ordering changes the sequence $\{ g _ { k } , \bar { c } _ { k } \}$ , and can therefore change both the peak location $k ^ { \star } ( \bar { \pi } )$ and the peak height $P _ { n } ( \pi )$

Selection acts on the entire marginal-utility sequence: a better ordering front-loads larger positive net marginal utilities and can achieve a higher peak before those positive net utilities are exhausted. Writing the path peak as a maximum prefix sum makes this precise.

Proposition 1 (Peak lift by front-loading net utility). For any ordering π, let $a _ { k } ^ { \pi } : = { { g } _ { k } ^ { \pi } } - { { c } _ { k } ^ { \pi } }$ Telescoping the one-step identity from $i = \bar { 1 }$ to k yields

$$
\operatorname* { m i n } _ { 1 \leq k \leq K } R _ { n } ( S _ { k } ^ { \pi } ) = R _ { n } ( S _ { 0 } ^ { \pi } ) - \operatorname* { m a x } _ { 1 \leq k \leq K } \sum _ { i = 1 } ^ { k } a _ { i } ^ { \pi } .
$$

Consequently, an ordering $\pi ^ { \prime }$ with a larger max-prefix-sum ofnet marginal utilities attains a no-worse peak risk than π.

Proof. Sum the one-step identity to obtain

$$
R _ { n } ( S _ { k } ^ { \pi } ) = R _ { n } ( S _ { 0 } ^ { \pi } ) - \sum _ { i = 1 } ^ { k } a _ { i } ^ { \pi } ;
$$

the minimum risk along $\pi$ is at the index maximising the cumulative net utility.

The information form

$$
g _ { k , \mathrm { l o g } } = I ( F _ { m _ { k } } ; Y \mid \bar { F } _ { S _ { k - 1 } } )
$$

interprets each conditional gain as task signal not yet covered by the current set; the one-step identity itself applies to any decomposable risk, including 0–1 loss with a different $g _ { k }$ . Under the singlecrossing conditions above, if $c _ { k } ( n ) = \alpha _ { k } / n$ with $\alpha _ { k } \geq 0$ , increasing n weakly decreases every c<sub>k</sub>, so the peak index

$$
\operatorname* { m a x } \{ k : g _ { k } > c _ { k } ( n ) \}
$$

is non-decreasing in $n ;$ the next subsections show the peaked regime is the rule and that the peak shifts right with n on most datasets.

## 2.3 A FAVORABLE DIAGNOSTIC PATH

Two taxonomies recur throughout the paper. Encoder paradigm partitions pre-training objectives into five classes: (1) language-supervised (CLIP, SigLIP), (2) self-supervised (DINOv2, MAE, Data2Vec), (3) generative or image-prior (Stable-Diffusion VAE encoder), (4) class-supervised (ConvNeXt, ResNet-50, ViT-B/16), and (5) task-expert (SAM, Depth-Anything). Recognition regime partitions downstream classification tasks into five classes: generic object, fine-grained object, texture, geospatial scene, and structured symbol or digit. Supplementary App. A.3.1 gives the full taxonomy and per-paradigm encoder rationales.

The diagnostic pool contains six frozen encoders. Cambrian-1 (Tong et al., 2024) reports that the trio CLIP-ViT-B/16 (Radford et al., 2021), DINOv2-base (Oquab et al., 2024), and ConvNeXt-base (Liu et al., 2022) delivers $\mathbf { a } + \mathbf { 3 } \tan + 5$ point improvement over single-encoder baselines on its vision-centric benchmark CV-Bench, and treats this trio as the main fusion combination of the paper. We adopt it as the first three encoders of our pool. We then add a second representative of each of the three paradigms it spans: SigLIP-B/16 (Zhai et al., 2023) for language supervision, ViT-MAE-base (He et al., 2022) for self-supervision, and ResNet-50 (He et al., 2016) for class supervision. The pool is balanced (two encoders per paradigm), and every individual encoder is competent at image classification. Paradigms (3) generative and (5) task-expert do not appear here. They are reintroduced in the main experiment of Section $^ { 4 , }$ where they serve as adversarial candidates that a competent selector should reject.

We evaluate this pool on five datasets, one per recognition regime: CIFAR-100 (generic object), Oxford Pets (fine-grained object), DTD (texture), Country211 (geo-spatial), and GTSRB (symbol or digit). The same five datasets reappear unchanged in the main experiments. The fusion operator used to produce $F _ { S }$ is the gated-fusion operator of Shi et al. (2025) with a 2-layer MLP, so peak-thendecline cannot reduce to a feature-concatenation artifact.

Encoders enter in a fixed literature-tracked path: CLIP, DINOv2, ConvNeXt, SigLIP, MAE, ResNet 50, the Cambrian-1 trio followed by within-paradigm second representatives, so positions 5–6 act as redundancy probes against 1–3. The path is fixed a priori from published fusion practice, using no held-out task, transferability score, or observation of the evaluation datasets. Its purpose is not to approximate the optimal ordering but to show that even a plausible fusion trajectory has an orderspecific peak below the full pool; Section 3 then asks whether a better, more task-aligned trajectory can be built without downstream retraining.

Table 1: Peak-then-decline at 10-shot across all five recognition regimes. Encoders are added one at a time along the literature-tracked path of Section 2.3. Each cell is gated-fusion accuracy (%) after the indicated set is in place. Bold: per-column peak.
<table><tr><td>Regime Dataset</td><td>generic CIFAR-100</td><td>fine-grained Pets</td><td>texture DTD</td><td>geo-spatial Country211</td><td>symbol GTSRB</td></tr><tr><td>1 (CLIP)</td><td>60.29</td><td>85.04</td><td>64.41</td><td>13.50</td><td>61.15</td></tr><tr><td>2 (+DINOv2)</td><td>80.11</td><td>92.86</td><td>72.67</td><td>9.85</td><td>51.54</td></tr><tr><td>3 (+ConvNeXt)</td><td>80.37</td><td>93.75</td><td>73.52</td><td>9.91</td><td>52.73</td></tr><tr><td>4 (+SigLIP)</td><td>80.18</td><td>93.46</td><td>74.05</td><td>9.82</td><td>52.34</td></tr><tr><td>5 (+MAE)</td><td>80.20</td><td>93.65</td><td>74.17</td><td>9.71</td><td>51.97</td></tr><tr><td>6 (+ResNet-50)</td><td>79.92</td><td>93.49</td><td>74.18</td><td>9.58</td><td>52.58</td></tr><tr><td>Peak position</td><td>3</td><td>3</td><td>6</td><td>1</td><td>1</td></tr></table>

Table 2: n-scaling across all five recognition regimes: peak position along the literature-tracked path at three sample sizes. For each dataset and each of three sample sizes (10 shots per class, 100 shots per class, full training set), we report the peak position along the path and the gated-fusion accuracy (%) at that peak. The right-most column gives the shift ∆ = (peak at full data) − (peak at 10-shot). The peak moves right with n on 3 of 5 datasets (CIFAR-100, Pets, GTSRB); Country211 is single-encoder saturated and stays at $k = 1 ;$ DTD shows a non-monotonic dependence on n (further discussed in the text). Full per-k curves are in supplementary App. A.3.5.
<table><tr><td></td><td colspan="2"> $n = 1 0 – { \mathrm { s h o t } }$ </td><td colspan="2"> $n = 1 0 0 { \cdot } { \mathrm { s h o t } }$ </td><td colspan="2">n = full data</td><td></td></tr><tr><td>Dataset (regime)</td><td>peak k</td><td>Acc</td><td>peak k</td><td>Acc</td><td>peak k</td><td>Acc</td><td>∆</td></tr><tr><td>CIFAR-100 (generic)</td><td>3</td><td>80.37</td><td>3</td><td>84.26</td><td>5</td><td>87.36</td><td>+2</td></tr><tr><td>Pets (fine-grained)</td><td>3</td><td>93.75</td><td>4</td><td>95.86</td><td>5</td><td>95.87</td><td>+2</td></tr><tr><td>DTD (texture)</td><td>6</td><td>74.18</td><td>3</td><td>80.78</td><td>5</td><td>82.53</td><td>-1</td></tr><tr><td>Country211 (geo-spatial)</td><td>1</td><td>13.50</td><td>1</td><td>24.01</td><td>1</td><td>25.65</td><td>0</td></tr><tr><td>GTSRB (symbol / digit)</td><td>1</td><td>61.15</td><td>1</td><td>82.35</td><td>5</td><td>86.55</td><td>+4</td></tr></table>

Classical multi-view learning attributes fusion gain to complementarity (Xu et al., 2013). Our pool spans three paradigms with two competent encoders each, creating conditions intended to favour complementary gains; peak-then-decline therefore appears even on a path designed to favour the “more is better” hypothesis. Figure 1 previews the central observation (per-dataset curves in supplementary App. A.3.5).

## 2.4 EVIDENCE ACROSS REGIMES AND DATA SCALES

The marginal-utility account predicts (i) peak-then-decline whenever marginal gain $g _ { k }$ diminishes faster than cost $c _ { k }$ accumulates and, under the scaling condition stated at the end of Section 2.2, (ii) a right-shift of the crossing as n grows. We test (i) with the 10-shot scan in Table 1 and (ii) with the n-sweep in Table 2, running the same five datasets at 10-shot, 100-shot, and full training set.

Order-specific peaks on a favourable fixed path. Under 10-shot training (Table 1) the peak position is task-dependent: k = 3 on CIFAR-100 and Pets, k = 6 on DTD, and k = 1 on Country211 and GTSRB, where CLIP alone already beats every multi-encoder prefix and each added view has negative net utility along this path. The peak is a property of the ordered marginal-utility sequence, not the model count.

Peak shifts right where finite-sample cost dominates. The n-sweep (Table 2) shows the peak moving right with n on three of five datasets: GTSRB jumps from $k = 1 \mathrm { t o } k = 5$ at full data, CIFAR-100 and Pets shift by +2, Country211 stays at $k = 1$ , and DTD is non-monotone. The lesson is structural: the prefix-curve height at every k depends on the ordering, so a selector that front-loads high-utility views raises the curve everywhere, which KAGES (Section 3) implements.

## 3 KAGES: GREEDY VIEW-SET COMPOSITION BY KERNEL ALIGNMENT

Section 2 showed that large-pool fusion is an ordering problem: a selector’s quality is reflected in the entire prefix-accuracy curve, not in a single best prefix. KAGES constructs such an ordering by greedily maximising a set-level kernel-alignment utility. It is training-free but not label-free: it uses labels only through a label kernel and never trains a downstream classifier for a candidate subset, which makes it deployable at pool sizes where validation-based subset search is infeasible.

## 3.1 SET-LEVEL OBJECTIVE AND GREEDY SELECTION

Let $\mathcal { M } = \{ m _ { 1 } , . . . , m _ { M } \}$ be the candidate pool, $F _ { s } \in \mathbb { R } ^ { n \times d _ { s } }$ the feature matrix of encoder s on n labelled samples, and $Y ^ { ' } \in \{ 0 , 1 \} ^ { n \times C _ { \mathrm { c l s } } }$ <sup>s</sup> the one-hot label matrix. For a subset $S \subseteq { \mathcal { M } }$ , KAGES assigns the set-level score

$$
J ( S ) = U ( \bar { F } _ { S } , Y ) - \tau \sum _ { s \in S } \frac { d _ { s } } { n } ,\tag{5}
$$

where $U ( \bar { F } _ { S } , Y ) ~ \in ~ \mathbb { R }$ measures the task alignment of the joint selected representation $\bar { F } _ { S }$ and the optional second term is a per-encoder finite-sample complexity penalty $( \tau \geq 0 ;$ the default $\tau = 0$ recovers the pure utility and is swept in Section 4). We set $U ( \emptyset , { \dot { Y } } ) : = J ( \emptyset ) : = 0$ by convention. Given a current set S, KAGES scores a candidate m $\notin \cal S$ by the marginal gain $\Delta J ( m \mid S ) : = J ( S \cup \{ m \} ) - J ( S )$ , adds the candidate with the largest marginal gain, and repeats until all candidates are ordered. The output is a complete encoder ordering; downstream users may select any prefix according to their compute budget. When a single score-based deployment size is desired, an optional rule is $\hat { k } _ { J } : = \operatorname* { m i n } \left( \operatorname { a r g m a x } _ { 1 \leq k \leq M } J ( S _ { k } ) \right)$ , which chooses the smallest prefix attaining the largest score along the completed path.

Algorithm 1 KAGES: training-free greedy encoder ordering by set-level marginal utility.   
Require: Pool $\mathcal { M } = \{ m _ { 1 } , . . . , m _ { M } \}$ ; labelled task (X, Y) with n samples; cost coefficient $\tau \geq 0 .$   
1: Compute centred encoder kernels $K _ { s } ^ { c }$ for every $s \in \mathcal { M }$ on the n samples.   
2: Initialize $S _ { 0 } \gets \emptyset , \hat { \pi }  ( )$   
3: for $t = 1 , 2 , \dots , M$ do   
4: $m _ { t } \gets \arg \operatorname* { m a x } _ { m \notin S _ { t - 1 } } \ \Delta J ( m \mid S _ { t - 1 } )$ # marginal-gain greedy step   
5: $S _ { t } \gets S _ { t - 1 } \cup \{ m _ { t } \}$ ; append $m _ { t }$ to πˆ.   
6: end for   
7: Optional: $\hat { k } _ { J } \gets \operatorname* { m i n } \bigl ( \operatorname { a r g m a x } _ { 1 \leq k \leq M } J ( S _ { k } ) \bigr )$   
8: return encoder ordering $\hat { \pi } = ( m _ { 1 } , \dots , m _ { M } )$ and optional score-selected prefix $S _ { \hat { k } _ { J } }$

Algorithm 1 produces a complete ordering of the pool and, optionally, a score-selected prefix. The algorithm itself does not require monotonicity or submodularity; Corollary 1 applies only to the unpenalised alignment utility and only under its stated conditions.

## 3.2 CENTRED KERNEL-TARGET ALIGNMENT AS UTILITY

For non-empty S, we instantiate U with the centred set-level kernel-target alignment,

$$
\begin{array} { r } { U ( \bar { F } _ { S } , Y ) = \mathrm { C K A } \big ( K _ { S } ^ { c } , K _ { Y } ^ { c } \big ) , \qquad K _ { S } ^ { c } = \displaystyle \sum _ { s \in S } K _ { s } ^ { c } , \quad K _ { s } ^ { c } = H F _ { s } F _ { s } ^ { \top } H , \quad K _ { Y } ^ { c } = H Y Y ^ { \top } H , } \end{array}\tag{6}
$$

where $\begin{array} { r } { H = I _ { n } - \frac { 1 } { n } \mathbf { 1 } \mathbf { 1 } ^ { \top } } \end{array}$ is the centering matrix and $K _ { s } ^ { c }$ is the centred linear kernel of encoder s. By the convention above, $U ( \emptyset , Y ) = \bar { 0 ; }$ we assume $\| \check { K } _ { Y } ^ { c } \| _ { F } > 0$ and discard any encoder with $\| \dot { K } _ { s } ^ { c } \| _ { F } = 0$ . For non-zero kernels, centred kernel-target alignment (Cristianini et al., 2002; Kornblith et al., 2019) is bounded in [0, 1], scale-invariant in each kernel individually, and reduces to canonical kernel-target alignment for $| S | = 1$

Linear-kernel additivity. The additive form $\begin{array} { r } { K _ { S } ^ { c } = \sum _ { s \in S } K _ { s } ^ { c } } \end{array}$ is not cosmetic: dropping the centering shows it equals the linear kernel of the concatenated feature matrix,

$$
[ F _ { s _ { 1 } } \ : | \ : \cdots \ : | \ : F _ { s _ { k } } ] [ F _ { s _ { 1 } } \ : | \ : \cdots \ : | \ : F _ { s _ { k } } ] ^ { \intercal } = \sum _ { j = 1 } ^ { k } F _ { s _ { j } } F _ { s _ { j } } ^ { \intercal } ,\tag{7}
$$

so adding an encoder is an $n \times n$ additive update on the running set kernel, independent of $d _ { m } . \mathrm { A }$ perencoder Frobenius-normalised variant $\widetilde { K } _ { s } ^ { c } : = K _ { s } ^ { c } / ( \| K _ { s } ^ { c } \| _ { F } + \varepsilon )$ yields slightly lower empirical AULC on our pool; for non-zero kernels the $\varepsilon = 0$ form is exactly scale-invariant, while the small numerical $\varepsilon > 0$ only stabilises near-zero kernels. We report this variant as an ablation in supplementary App. A.3.10 and use the raw additive variant in the main results.

Running scalars and cost. We maintain $K _ { S } ^ { c }$ and three Frobenius scalars $( A _ { S } = \langle K _ { S } ^ { c } , K _ { Y } ^ { c } \rangle _ { F }$ $B _ { S } = \bar { \langle } K _ { S } ^ { c } , K _ { S } ^ { c } \rangle _ { F } , C = \| K _ { Y } ^ { c } \| _ { F } ^ { 2 } ) ;$ ; each candidate evaluation costs three $n \times n$ Frobenius inner products and one $n \times n$ tensor sum $- O ( n ^ { 2 } )$ per candidate per step, independent of encoder dimension. For 10 encoders at $n = 5 0 0 .$ , one full greedy traversal completes in under one second on a single GPU. Writing $p _ { c } = n _ { c } / n$ for the empirical class proportions, the centred label kernel satisfies $\begin{array} { r } { ( K _ { Y } ^ { \bar { c } } ) _ { i j } = \mathbf { 1 } [ y _ { i } = \bar { y _ { j } } ] - p _ { y _ { i } } - p _ { y _ { j } } + \sum _ { c } \bar { p } _ { c } ^ { 2 } } \end{array}$ , reducing to $\mathbf { \hat { 1 } } [ y _ { i } = y _ { j } ] - 1 / C _ { \mathrm { c l s } }$ for class-balanced samples. Thus $\langle K _ { m } ^ { c } , K _ { Y } ^ { c } \rangle _ { \mathrm { { i } } }$ <sub>F</sub> rewards encoder similarity structure that covaries with the task labels.

## 3.3 ALGEBRAIC STRUCTURE AND GREEDY GUARANTEE

The set-level kernel-target alignment admits an exact algebraic expansion that exposes the relevance– redundancy trade-off without any linear combination of separately estimated proxies. Unless stated otherwise, this subsection analyses the pure alignment utility U, equivalently J with $\tau = 0$ , which is used in the main results. For $\bar { \tau _ { \bar { ~ } > } } 0 , \dot { \Delta J } ( m \mid \bar { S } ) = \Delta U ( m \bar { \mid } S ) \bar { - } \bar { \tau } d _ { m } / n$

Proposition 2 (Algebraic expansion of $\mathrm { C K A } ( K _ { S } ^ { c } , K _ { Y } ^ { c } ) )$ . For any non-empty $S \subseteq { \mathcal { M } }$ with $\| K _ { S } ^ { c } \| _ { F } >$ 0 and $\| K _ { Y } ^ { c } \| _ { F } > 0$

$$
\mathrm { C K A } ( K _ { S } ^ { c } , K _ { Y } ^ { c } ) = \frac { \displaystyle \sum _ { s \in S } \langle K _ { s } ^ { c } , K _ { Y } ^ { c } \rangle _ { F } } { \displaystyle \sqrt { \sum _ { s , s ^ { \prime } \in S } \langle K _ { s } ^ { c } , K _ { s ^ { \prime } } ^ { c } \rangle _ { F } } \cdot \| K _ { Y } ^ { c } \| _ { F } } .\tag{8}
$$

The numerator is the sum ofper-encoder centred kernel-target inner products. The denominator decomposes into per-encoder self-terms $\langle K _ { s } ^ { c } , K _ { s } ^ { c } \rangle _ { F }$ (encoder “kernel mass”) and pairwise crossterms $\bar { \langle } K _ { s } ^ { c } , K _ { s ^ { \prime } } ^ { c } \rangle _ { F } \bar { f } o r \ : s \neq s ^ { \prime }$ (between-encoder kernel similarity).

Proofsketch. By construction $\begin{array} { r } { K _ { S } ^ { c } = \sum _ { s \in S } K _ { s } ^ { c } \left( \mathrm { E q . ~ } 6 \right) } \end{array}$ . Bilinearity of the Frobenius inner product yields $\begin{array} { r } { \langle K _ { S } ^ { c } , K _ { Y } ^ { c } \rangle _ { F } = \sum _ { s \in S } \langle K _ { s } ^ { c } , K _ { Y } ^ { c } \rangle _ { F } } \end{array}$ and $\begin{array} { r } { \langle K _ { S } ^ { c } , K _ { S } ^ { c } \rangle _ { F } = \sum _ { s , s ^ { \prime } \in S } \langle K _ { s } ^ { c } , K _ { s ^ { \prime } } ^ { c } \rangle _ { F } } \end{array}$ . Substituting into the definition of centred kernel-target alignment recovers Eq. (8). □

Exact marginal-gain criterion. For brevity write the five scalars $A _ { S } : = \langle K _ { S } ^ { c } , K _ { Y } ^ { c } \rangle _ { F } , \ B _ { S } : =$ $\langle { \cal K } _ { S } ^ { c } , { \cal K } _ { S } ^ { c } \rangle _ { F } , r _ { m } : = \langle { \cal K } _ { m } ^ { c } , { \cal K } _ { Y } ^ { c } \rangle _ { F } , h _ { m } : = \langle \dot { \cal K } _ { m } ^ { c } , { \cal K } _ { S } ^ { c } \rangle _ { F } , q _ { m } : = \langle { \cal K } _ { m } ^ { c } , { \cal K } _ { m } ^ { c } \rangle _ { F }$ , with $\| \dot { K } _ { Y } ^ { c } \| _ { F } ^ { 2 } = C$ fixed across candidates.

Proposition 3 (Exact CKA marginal-gain criterion). For the unpenalised objective used in the main results $( \tau = 0 )$ , let S be non-empty with $B _ { S } > 0$ and $C > 0$ . Adding m $\not \in S$ to S yields positive alignment gain $\Delta U ( m \mid S ) > 0$ if and only if

$$
r _ { m } > A _ { S } \cdot \left( \sqrt { 1 + \frac { 2 h _ { m } + q _ { m } } { B _ { S } } } - 1 \right) .\tag{9}
$$

Proof. Substitute $K _ { S \cup \{ m \} } ^ { c } = K _ { S } ^ { c } + K _ { m } ^ { c }$ into $U ( S \cup \{ m \} ) = ( A _ { S } + r _ { m } ) / \sqrt { ( B _ { S } + 2 h _ { m } + q _ { m } ) C }$ and require $U ( S \cup \{ m \} ) > U ( S ) = A _ { S } / \sqrt { B _ { S } C }$ . With $A _ { S } \geq 0$ both sides are non-negative; squaring and rearranging yields Eq. (9). Full derivation in supplementary App. A.3.4. □

Adaptive kernel mRMR interpretation. For $A _ { S } > 0$ , when the candidate’s target and set-mass contributions are both small relative to the current set, $r _ { m } / A _ { S } \ll 1$ and $( 2 h _ { m } + q _ { m } ) / B _ { S } \ll 1$ first-order expansion of Eq. (9), with their product treated as second order, gives the linearised criterion

$$
\Delta U ( m \mid S ) \approx \frac { 1 } { \sqrt { B _ { S } C } } \left[ r _ { m } - \frac { A _ { S } } { 2 B _ { S } } \left( 2 h _ { m } + q _ { m } \right) \right] .\tag{10}
$$

KAGES therefore locally implements a parameter-free relevance-minus-redundancy criterion whose redundancy weight $A _ { S } / ( 2 B _ { S } )$ is determined by the current set’s alignment and mass, rather than being fixed ahead of time as in classical mRMR-style selectors (Peng et al., 2005; Brown et al., 2012).

Conditional greedy guarantee. The unnormalised numerator $A _ { S } = \langle K _ { S } ^ { c } , K _ { Y } ^ { c } \rangle _ { F }$ is monotone because the centred Gram matrices are positive semidefinite, but the normalised CKA utility $U ( S )$ need not be monotone: its denominator can outweigh a non-negative numerator increment, as characterised exactly by Proposition 3. We therefore state a conditional guarantee under monotonicity and the standard submodularity-ratio condition of Das & Kempe (2011); the theorem does not assert that every empirical CKA instance satisfies these assumptions.

Theorem 2 (Conditional greedy guarantee under a submodularity ratio). Let $U : 2 ^ { \mathcal { M } } \to \mathbb { R } _ { \geq 0 }$ satisfy $U ( \emptyset ) = 0$ and be monotone non-decreasing. Fix a cardinality budget K. Suppose there is a constant $\gamma \in ( 0 , 1 ]$ such that,for every $L \subseteq { \mathcal { M } }$ and every $T \subseteq { \mathcal { M } } \setminus { \dot { L } }$ with $| T | \leq K$

$$
\sum _ { m \in T } \Delta U ( m \mid L ) \ \geq \ \gamma \big [ U ( L \cup T ) - U ( L ) \big ] .
$$

Then the cardinality-K prefix $S _ { K } ^ { \mathrm { g r e e d y } }$ produced by Algorithm 1 with $\tau = 0$ satisfies

$$
U ( S _ { K } ^ { \mathrm { g r e e d y } } ) \geq \left( 1 - e ^ { - \gamma } \right) \operatorname* { m a x } _ { | S | \leq K } U ( S ) .
$$

Proofsketch. Let O be an optimal set with $| O | \le K$ . The submodularity-ratio condition implies that the sum of the available gains from $O \setminus S _ { k } ^ { \setminus }$ is at least $\gamma [ U ( O ) - U ( S _ { k } ) ]$ . Hence the greedy gain is at least $\gamma / K$ times the current optimality gap, which contracts geometrically and yields the stated $( 1 - e ^ { - \gamma } )$ factor. Full proof in supplementary App. A.1.

Corollary 1 (Prefix-wise fixed-budget guarantee). Under the hypotheses of Theorem 2, for every $k = 1 , \ldots , K$ , the greedy ordering $\pi _ { g }$ produced by Algorithm 1 with $\tau = 0$ satisfies

$$
U ( S _ { k } ^ { \pi _ { g } } ) \geq \left( 1 - e ^ { - \gamma } \right) \operatorname* { m a x } _ { | S | \leq k } U ( S ) .
$$

Corollary 1 is a fixed-budget statement at each prefix size, not a guarantee that empirical CKA is monotone or that its score-selected stopping point is accuracy-optimal. The accuracy-level prefix quality is evaluated empirically in Section 4.

Empirical ordering consistency. KAGES uses empirical marginal gains $\Delta \widehat { J } _ { n } ( m \mid S )$ rather than the marginal gains of a deterministic population target utility $J .$ Under the uniform subgaussian concentration and bias assumptions stated in supplementary $\mathrm { \bar { A p p . A . 2 } } .$ , with scale $\sigma ,$ bias constant $B ,$ and population-path margin $G > 0 , { \mathrm { i f } } n \geq { \dot { ( 4 B / G ) } } ^ { 2 }$ then the empirical ordering matches the population greedy ordering with probability at least $1 ^ { ' } - [ M ( M + 1 ) - \dot { 2 } ] \exp ( - n G ^ { 2 } / ( 3 2 \sigma ^ { 2 } ) )$ . This is an assumption-conditional concentration result, not an unconditional Hoeffding claim for empirical CKA.

## 4 EXPERIMENTS

We evaluate KAGES on the main image-classification protocol of Section 2.3 extended to all five pre-training paradigms, and then test transfer to image retrieval and frozen-LLM fusion.

## 4.1 EXPERIMENTAL SETUP

Datasets. Following the taxonomies introduced in Section 2.3, we evaluate on the same five datasets used by the diagnostic, covering all five recognition regimes one-to-one: Generic Object (CIFAR-100), Fine-grained Object (Oxford Pets), Texture / Material (DTD), Geo-spatial / Scene (Country211), and Structured Symbol / Digit (GTSRB).

Encoder pool. The pool extends the diagnostic pool to all five pre-training paradigms, contributing 10 frozen backbones spanning: language-supervised (CLIP-ViT-B/16 (Radford et al., 2021), SigLIP-B/16 (Zhai et al., 2023)); self-supervised (DINOv2-base (Oquab et al., 2024), ViT-MAE-base (He et al., 2022), Data2Vec (Baevski et al., 2022)); generative (Stable-Diffusion VAE (Rombach et al., 2022)); class-supervised (ConvNeXt-base (Liu et al., 2022), ResNet-50 (He et al., 2016), ViT-B/16 (Dosovitskiy et al., 2021)); task-expert (SAM-ViT-B (Kirillov et al., 2023)). Paradigms C and E reappear as adversarial cases that a competent selector should reject.

Protocols. We report three headline protocols. B uses 16-shot training with a Concat+MLP fusion head (hidden dim 512, 20 epochs). H uses an 8-shot setting with a larger M = 15 candidate pool. F uses the full training set with the same fusion-head family. Together they test whether the orderingquality advantage persists across low-shot, larger-pool, and full-data regimes. Additional protocol and training details are in supplementary App. A.3.1.

Metrics. View-set composition is an ordering problem, so we evaluate selectors by the quality of their entire prefix path. Our two primary ordering metrics are:

• AULC (normalised area under the k–accuracy curve, analogous to AUROC): the mean prefix accuracy along the method’s full ordering.

• Acc@k<sup>⋆</sup>: the peak accuracy along the ordering, the best fused set the path reaches (marked by stars in Figure 2).

AULC does not depend on a stopping rule, so it directly evaluates prefix-path quality. Acc@k<sup>⋆</sup> is a hindsight path diagnostic computed from the downstream accuracy curve; it does not evaluate the optional score-based stopping rule <sup>ˆ</sup>k<sub>J</sub>, for which we make no separate empirical stopping-point claim.

Baselines and seeds. We compare KAGES against multi-view selection strategies operating on the same frozen-encoder pool. The primary baselines are the standard multi-view defaults: Fuse-All embodies the more-is-better default, Diversity selects for representational difference through centred kernel alignment between encoders, and Random is a random-order baseline. Because AULC requires a complete path, Fuse-All is represented by the prespecified fixed-order prefix path used by the implementation, whose final prefix contains the full pool. We additionally compare against two stronger subset-selection baselines, DPP (determinantal point process MAP inference) and Submodular (a facility-location coverage objective). Oracle greedily adds the encoder producing the largest true downstream-accuracy improvement and is a validation-based greedy hindsight reference, not a global upper bound over all subsets. All AULC numbers are means over five random seeds {17, 23, 42, 71, 137}; standard deviations in supplementary App. A.3.11.

## 4.2 MAIN RESULTS

KAGES beats the multi-view defaults on every dataset, and the margin is wide where fusion matters. Table 3 reports per-dataset AULC on all three protocols. KAGES is best in every cell against full fusion, random and diversity selection. The advantage concentrates in the regimes that genuinely reward heterogeneous fusion: on the structured-symbol task GTSRB it beats blind full fusion by +8.9, +15.5 and +6.7 points on B, H and F, and on the geo regime at full data by +3.6. On saturated natural-image tasks such as Pets the ordering still matters but the margin is smaller, since a single strong encoder already approaches the peak. Averaged over the five regimes KAGES leads Fuse-All by +3.9, +5.8 and +3.3 points. The path-curve view (Figure 2) shows the mechanism: KAGES front-loads high-utility encoders and is already strong at small k, while the defaults climb slowly and recover late, if at all.

KAGES also matches or exceeds stronger subset-selection baselines. Beyond the more-is-better defaults, we compare against two strong subset-selection methods on the same pool: DPP MAP inference and a facility-location submodular objective. Both are far better than full fusion, yet KAGES still leads on average AULC on every protocol (B: 66.93 vs 66.13 submodular and 65.21 DPP; H: 64.77 vs 63.49 and 62.13; F: 74.78 vs 73.98 and 73.51), confirming that the label-aware kernel-alignment criterion adds value over diversity- and coverage-driven selection.

Table 3: Per-dataset AULC (%) on all three protocols: KAGES beats the multi-view defaults on every dataset, by a wide margin where heterogeneous fusion matters. Against full fusion (Fuse-All), random and diversity selection, KAGES is best in every cell. The $\Delta$ rows give KAGES minus Fuse-All: the gain is largest on the structured-symbol task GTSRB (+8.9 to +15.5 points) and on the geo regime at full data. Stronger subset-selection baselines (DPP, submodular) are reported in the text; KAGES matches or exceeds them too. Means over 5 seeds {17, 23, 42, 71, 137}; bold: best per column within a protocol.
<table><tr><td>Method</td><td>C100</td><td>Pets</td><td>DTD</td><td>C211</td><td>GTSRB</td><td>Avg</td></tr><tr><td colspan="7">Protocol B (16-shot, Concat+MLP)</td></tr><tr><td>Fuse-All Random</td><td>78.54 77.06</td><td>94.06 92.97</td><td>75.35 74.87</td><td>10.16 11.10</td><td>57.18 59.16</td><td>63.06 63.03</td></tr><tr><td>Diversity KAGES (ours) ∆ vs Fuse-All</td><td>80.61 83.84 +5.3</td><td>90.91 94.70 +0.6</td><td>74.38 77.82 +2.5</td><td>11.27 12.16 +2.0</td><td>59.12 66.12 +8.9</td><td>63.26 66.93 +3.9</td></tr><tr><td colspan="7">Protocol H (8-shot, M=15) Fuse-All</td></tr><tr><td>Random Diversity</td><td>73.89 75.11 78.18</td><td>89.01 91.78 86.91</td><td>70.10 69.84 68.87</td><td>9.65 9.77 9.67</td><td>52.42 54.88 56.67</td><td>59.02 60.28 60.06</td></tr><tr><td>KAGES (ours) ∆ vs Fuse-All</td><td>80.20 +6.3</td><td>92.82 +3.8</td><td>72.36 +2.3</td><td>10.57 +0.9</td><td>67.92</td><td>64.77</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>+15.5</td><td>+5.8</td></tr><tr><td>Protocol F (full data)</td><td>85.07</td><td>95.90</td><td>80.05</td><td></td><td></td><td></td></tr><tr><td>Fuse-All</td><td></td><td></td><td></td><td>16.51</td><td>79.71</td><td>71.45</td></tr><tr><td>Random</td><td>84.66</td><td>95.26</td><td>80.07</td><td>19.43</td><td>81.87</td><td>72.26</td></tr><tr><td>Diversity</td><td>86.21</td><td>94.35</td><td>79.85</td><td>18.51</td><td>81.31</td><td>72.05</td></tr><tr><td>KAGES (ours)</td><td>88.19</td><td>96.25</td><td>82.96</td><td>20.07</td><td>86.41</td><td>74.78</td></tr><tr><td>∆ vs Fuse-All</td><td>+3.1</td><td>+0.4</td><td>+2.9</td><td>+3.6</td><td>+6.7</td><td>+3.3</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

![](images/28adf1c5f24b7b3d032865fe596adf784c84d66e4dac31154738e3f2384333f1.jpg)

![](images/d580a5498aa57535158d0a29d4bff42b0b99ce5ec14b2d62bd0b4122089dfaa4.jpg)

![](images/760743574b83e38aa7127d592a62f1f01f4fbcae7e7a09f9209ed98584a951c0.jpg)

![](images/b5dde927a392c9e7601cbfdc9049ae85ddafaa2e24e4c1db9cdfa4c49ebaf3fd.jpg)

![](images/dda9c2d9ee9ed362fb0159f869a5a2cad9b3062493a4f4bf9cf76f21964a8c85.jpg)  
Figure 2: KAGES front-loads high-utility encoders and stays at the top of the multi-view baseline envelope (Protocol F, full data). Per-dataset prefix accuracy along each method’s ordering. KAGES (navy, star at its peak) is plotted over the gray envelope spanning the standard multi-view defaults (full fusion, random, and diversity selection), with their per-k median in gray. KAGES reaches a strong representation at small k on every regime while the defaults climb slowly, with the clearest separation on the texture and symbol tasks. Country211 is the anti-fusion regime in which CLIP alone saturates geolocation, so all methods converge.

First-pick consistently favours self-supervised encoders, and adversarial paradigms are rejected. On four natural-image regimes (CIFAR-100, Pets, DTD, Country211) KAGES’s first pick is DINOv2- base (paradigm B) on all 5 seeds; on Structured Symbol / Digit (GTSRB) it picks the class-supervised

![](images/57638dcc27940bfb13952c92fd2b47f92014508e351b32b6d5f9725c131fe1ee.jpg)  
Figure 3: Accuracy curves over fused LLM count for MMLU and ARC-Challenge. The gray band spans diversity-greedy and fixed-order baselines, the gray line shows their median, and KAGES is shown in navy with stars marking its peak. The peak-then-decline pattern reproduces at the LLM layer.

ResNet-50 (3/5 seeds) or CLIP (2/5). Paradigms C (generative) and E (task-expert) are never first-picked, consistent with both being structurally orthogonal to classification.

Cost-penalty ablation (τ ). Sweeping $\tau \in \{ 0 , 0 . 0 1 , 0 . 0 5 , 0 . 1 , 0 . 5 , 1 . 0 \}$ (3 seeds) gives mean AULC {66.9, 66.7, 65.5, 64.6, 62.9, 62.5}, maximised at $\tau = 0$ . Larger τ makes the ordering track encoder dimension and degrades fast on GTSRB, so the main results use the pure $\mathrm { C K A } ( K _ { S } ^ { c } , \mathbf { \bar { { K } } } _ { Y } ^ { c } )$ criterion.

## 4.3 BEYOND CLASSIFICATION

KAGES’s advantage extends to two settings outside the recognition-regime taxonomy, and in the language setting the peak-then-decline reappears. (i) Image retrieval: on the same encoder pool with L2-normalised concatenated features and nearest-neighbour search, Recall@1 saturates later than classification, peaking at $k = 8$ on Pets and $k = 1 0$ on both DTD and GTSRB, with KAGES matching the peak in each case (full curves in supplementary App. A.3.7). (ii) Frozen-LLM fusion: treating six base LLMs (Llama-3, Qwen-2.5, Mistral, Gemma, Yi-1.5, DeepSeek) as views and training a linear probe on MMLU and ARC-Challenge, accuracy peaks then declines as in vision; KAGES reaches 75.0% on MMLU at $k = 2$ and 91.0% on ARC at $k = 4$ , hitting the peak at a smaller k than fixed-order or diversity-greedy baselines (Figure 3; full results in supplementary App. A.4).

## 5 RELATED WORK

Foundation-model fusion and transferability scoring. Classical multi-view learning (Xu et al., 2013; Yan et al., 2021; Trosten et al., 2021) fuses a small fixed set of feature views and offers no recipe for selecting views out of a heterogeneous pool. Multi-encoder MLLMs scale this to multiple frozen vision backbones: Eagle (Shi et al., 2025), Cambrian-1 (Tong et al., 2024), BRAVE (Kar et al., 2024), and MoVA (Zong et al., 2024) study encoder combinations or routing within relatively small, predefined expert pools. Training-free transferability scores (LogME (You et al., 2021; 2022), H-Score (Bao et al., 2019), GBC (Pandy et al., 2022), LEEP (Nguyen et al., 2020), NCE (Tran et al., 2019), TransRate (Huang et al., 2022), EMMS (Meng et al., 2023), PACTran (Ding et al., 2022), OTCE (Tan et al., 2021; 2024), DISCO (Zhang et al., 2025)) score a single encoder and provide no notion of how a candidate interacts with an already-selected pool. Pool-level approaches are sparse: OSBORN (Vimal et al., 2023) gives a $( 1 - 1 / e )$ guarantee at a fixed $k = 3 ;$ model-zoo studies (Dong et al., 2022; Chen et al., 2023) select at the recipe level. Across these lines, encoder choice is either constrained to predefined pools or scored one model at a time, without jointly constructing a complete ordering over a large heterogeneous pool. We depart on three counts that, to our knowledge, no prior work combines: we characterise order-dependent peak-then-decline, including an absolute decline on structured-symbol tasks, rather than only diminishing returns on a handful of encoders; we trace it to the frozen feature-fusion regime through a gain–cost crossing; and we turn that account into a single training-free selector with conditional prefix-wise guarantees that produces a high-quality ordering and provides an optional score-based stopping rule from frozen-feature statistics alone.

Kernel and information-theoretic selection. Classical scalar feature selection (mRMR (Peng et al., 2005), JMI (Yang & Moody, 1999), Brown et al. (Brown et al., 2012)) and similarity tools (CKA (Kornblith et al., 2019), HSIC (Gretton et al., 2005), Platonic Hypothesis (Huh et al., 2024)) compare representations or select scalar features but do not produce a path-quality recommendation across prefix sizes. The closest line is greedy kernel-based view selection: HSIC-mRMR / HSIC-Lasso (Song et al., 2012; Yamada et al., 2014) and Kernel-JMI (Brown et al., 2012) select scalar features via hand-weighted pairwise scores at cost growing with feature dimension. KAGES instead treats each encoder’s representation as one atomic unit via the single closed-form pure alignment utility $U ( S ) = \mathrm { C K A } ( \bar { K } _ { S } ^ { c } , K _ { Y } ^ { c } )$ , with kernel-level candidate evaluation independent of encoder dimension (Section 3.2); its exact positive-gain criterion (Proposition 3) reproduces relevance-minusredundancy with adaptive weight $A _ { S } / ( 2 B _ { S } )$ rather than tuned, and its conditional prefix-wise fixed-budget guarantee (Corollary 1) applies at every cardinality for which the stated monotonicity and submodularity-ratio conditions hold.

## 6 DISCUSSION, LIMITATIONS, AND FUTURE WORK

Limitations. The normalised CKA objective is neither generally monotone nor generally submodular because the denominator in Eq. 8 can outweigh a non-negative numerator increment and introduces cross-overlap terms. Our $( 1 - e ^ { - \gamma } )$ guarantee is therefore conditional on monotonicity and a positive submodularity ratio; establishing when empirical frozen-encoder pools satisfy these conditions is future work. The generative and task-expert paradigms are handled only at the selector level: KAGES never first-picks them, and fusing them remains future work.

## 7 CONCLUSION

We have recast multi-view learning for the foundation-model era, where a view is one of thousands of frozen encoders and the design problem becomes view-set composition: which encoders to fuse, and how many. We showed that the classical complementarity principle fails at this scale: the best achievable accuracy peaks at a small set and then plateaus or, on structured-symbol tasks, declines. We proposed KAGES, a training-free selector with encoder-dimension-independent kernel-level candidate evaluation and a conditional $\left( 1 - e ^ { - \gamma } \right)$ prefix-wise fixed-budget guarantee under the stated monotonicity and submodularity-ratio conditions. KAGES ranks first among multi-view selection methods across five regimes, five pre-training paradigms, and three evaluation settings, beating full fusion and diversity-based selection while approaching the oracle. As public model hubs keep growing, choosing which encoders to fuse, rather than how to fuse a fixed few, becomes the central question of multi-view learning, and a training-free, theory-backed answer to it is a step toward using the world’s frozen models at scale.

## ACKNOWLEDGMENTS AND AUTHORSHIP NOTE

We thank Prof. Zongbo Han for his guidance and helpful discussions. Prof. Han declined to be listed as an author of the final manuscript. Consequently, the earlier BMVC 2026 submission was withdrawn after the Area Chair decision and before publication. This preprint lists only the authors who approved the present version.

## REFERENCES

Alexei Baevski, Wei-Ning Hsu, Qiantong Xu, Arun Babu, Jiatao Gu, and Michael Auli. data2vec: A general framework for self-supervised learning in speech, vision and language. In ICML, pp. 1298–1312, 2022.

Yajie Bao, Yang Li, Shao-Lun Huang, Lin Zhang, Lizhong Zheng, Amir Zamir, and Leonidas Guibas. An information-theoretic approach to transferability in task transfer learning. In ICIP, pp. 2309–2313, 2019.

Gavin Brown, Adam Pocock, Ming-Jie Zhao, and Mikel Luján. Conditional likelihood maximisation: A unifying framework for information theoretic feature selection. JMLR, 13:27–66, 2012.

Yimeng Chen, Tianyang Hu, Fengwei Zhou, Zhenguo Li, and Zhiming Ma. Explore and exploit the diverse knowledge in model zoo for domain generalization. In ICML, pp. 4623–4640, 2023.

N. Cristianini, J. Shawe-Taylor, A. Elisseeff, and J. Kandola. On kernel-target alignment. Advances in Neural Information Processing Systems (NeurIPS), 2002.

A. Das and D. Kempe. Submodular meets spectral: Greedy algorithms for subset selection, sparse approximation and dictionary selection. In International Conference on Machine Learning (ICML), 2011.

Nan Ding, Xi Chen, Tomer Levinboim, Beer Changpinyo, and Radu Soricut. PACTran: PAC-Bayesian metrics for estimating the transferability of pretrained models to classification tasks. In ECCV, 2022. arXiv:2203.05126.

Qishi Dong, Awais Muhammad, Fengwei Zhou, Chuanlong Xie, Tianyang Hu, Yongxin Yang, Sung-Ho Bae, and Zhenguo Li. ZooD: Exploiting model zoo for out-of-distribution generalization. In NeurIPS, 2022.

A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations (ICLR), 2021.

Abhimanyu Dubey et al. The Llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Eyal Even-Dar, Shie Mannor, and Yishay Mansour. PAC bounds for multi-armed bandit and Markov decision processes. In COLT, 2002.

Eyal Even-Dar, Shie Mannor, and Yishay Mansour. Action elimination and stopping conditions for the multi-armed bandit and reinforcement learning problems. JMLR, 7:1079–1105, 2006.

Arthur Gretton, Olivier Bousquet, Alex Smola, and Bernhard Schölkopf. Measuring statistical dependence with hilbert-schmidt norms. In ALT, pp. 63–77, 2005.

K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.

Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked autoencoders are scalable vision learners. In CVPR, pp. 16000–16009, 2022.

Long-Kai Huang, Ying Wei, Yu Rong, Qiang Yang, and Junzhou Huang. Frustratingly easy transferability estimation. In ICML, 2022. arXiv:2106.09362.

Minyoung Huh, Brian Cheung, Tongzhou Wang, and Phillip Isola. The platonic representation hypothesis. In ICML, 2024.

Oguzhan Fatih Kar, Alessio Tonioni, et al. BRAVE: Broadening the visual encoding of vision-˘ language models. In ECCV, 2024.

Emilie Kaufmann, Olivier Cappé, and Aurélien Garivier. On the complexity of best-arm identification in multi-armed bandit models. JMLR, 17(1), 2016.

Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, Piotr Dollár, and Ross Girshick. Segment anything. In ICCV, 2023.

Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geoffrey Hinton. Similarity of neural network representations revisited. In ICML, 2019.

Yuchen Liu, Yaoming Zhu, Jiangbo Wang, Enwei Yu, Junxiang Liu, and Junfeng Yan. From CLIP to DINO: Visual encoders shout in multi-modal large language models. arXiv preprint arXiv:2310.08825, 2023.

Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A convnet for the 2020s. In CVPR, pp. 11976–11986, 2022.

Fanqing Meng, Wenqi Shao, Zhanglin Peng, Chonghe Jiang, Kaipeng Zhang, Yu Qiao, and Ping Luo. Foundation model is efficient multimodal multitask model selector. In NeurIPS, 2023. arXiv:2308.06262.

Cuong V Nguyen, Tal Hassner, Matthias Seeger, and Cedric Archambeau. Leep: A new measure to evaluate transferability of learned representations. In ICML, 2020.

Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. TMLR, 2024.

Michal Pandy, Andrea Agostinelli, Jasper Uijlings, Vittorio Ferrari, and Thomas Mensink. Transferability estimation using bhattacharyya class separability. In CVPR, pp. 9162–9172, 2022.

Hanchuan Peng, Fuhui Long, and Chris Ding. Feature selection based on mutual information: Criteria of max-dependency, max-relevance, and min-redundancy. IEEE TPAMI, 27(8):1226–1238, 2005.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In CVPR, 2022.

Min Shi, Fuxiao Liu, Shihao Wang, Shijia Liao, Subhashree Radhakrishnan, et al. Eagle: Exploring the design space for multimodal llms with mixture of encoders. In ICLR, 2025.

Le Song, Alex Smola, Arthur Gretton, Justin Bedo, and Karsten Borgwardt. Feature selection via dependence maximization. JMLR, 13:1393–1434, 2012.

Yang Tan, Yang Li, and Shao-Lun Huang. OTCE: A transferability metric for cross-domain cross-task representations. In CVPR, 2021. arXiv:2103.13843.

Yang Tan, Enming Zhang, Yang Li, Shao-Lun Huang, and Xiao-Ping Zhang. Transferability-guided cross-domain cross-task transfer learning. IEEE TNNLS, 2024. arXiv:2207.05510.

Alexander Tartakovsky, Igor Nikiforov, and Michèle Basseville. Sequential Analysis: Hypothesis Testing and Changepoint Detection. Chapman & Hall/CRC Monographs on Statistics and Applied Probability. CRC Press, 2014.

Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, et al. Cambrian-1: A fully open, vision-centric exploration of multimodal LLMs. In NeurIPS, 2024.

Anh T Tran, Cuong V Nguyen, and Tal Hassner. Transferability and hardness of supervised classification tasks. In ICCV, pp. 1395–1405, 2019.

Daniel J Trosten, Sigurd Lokse, Robert Jenssen, and Michael Kampffmeyer. Reconsidering representation alignment for multi-view clustering. In CVPR, 2021.

KB Vimal, Saketh Bachu, Tanmay Garg, Niveditha Lakshmi Narasimhan, Raghavan Konuru, and Vineeth N Balasubramanian. Building a winning team: Selecting source model ensembles using a submodular transferability estimation approach. In ICCV, pp. 11575–11586, 2023.

Chang Xu, Dacheng Tao, and Chao Xu. A survey on multi-view learning. arXiv preprint arXiv:1304.5634, 2013.

Makoto Yamada, Wittawat Jitkrittum, Leonid Sigal, Eric P. Xing, and Masashi Sugiyama. Highdimensional feature selection by feature-wise kernelized Lasso. Neural Computation, 26(1): 185–207, 2014.

Xiaoqiang Yan, Shizhe Hu, Yiqiao Mao, Yangdong Ye, and Hui Yu. Deep multi-view learning methods: A review. Neurocomputing, 448:106–129, 2021.

An Yang et al. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2024.

Howard Hua Yang and John Moody. Data visualization and feature selection: New algorithms for nongaussian data. In NeurIPS, pp. 687–693, 1999.

Kaichao You, Yong Liu, Jianmin Wang, and Mingsheng Long. Logme: Practical assessment of pre-trained models for transfer learning. In ICML, 2021.

Kaichao You, Yong Liu, Ziyang Zhang, Jianmin Wang, Michael I. Jordan, and Mingsheng Long. Ranking and tuning pre-trained models: A new paradigm for exploiting model hubs. JMLR, 23 (209), 2022. arXiv:2110.10545.

Tao Yu, Zhihe Lu, Xin Jin, Zhibo Chen, and Xinchao Wang. Task residual for tuning vision-language models. In CVPR, pp. 10899–10909, 2023.

Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In ICCV, pp. 11975–11986, 2023.

Renrui Zhang, Wei Zhang, Rongyao Fang, Peng Gao, Kunchang Li, Jifeng Dai, Yu Qiao, and Hongsheng Li. Tip-adapter: Training-free adaption of clip for few-shot classification. In ECCV, 2022.

Renrui Zhang, Xiangfei Hu, Bohao Li, Siyuan Huang, Hanqiu Deng, Hongsheng Li, Yu Qiao, and Peng Gao. Prompt, generate, then cache: Cascade of foundation models makes strong few-shot learners. In CVPR, pp. 15211–15222, 2023.

Tengxue Zhang, Yang Shu, Xinyang Chen, Yifei Long, Chenjuan Guo, and Bin Yang. Assessing pre-trained models for transfer learning through distribution of spectral components. In AAAI, 2025. arXiv:2412.19085.

Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. Learning to prompt for visionlanguage models. IJCV, 130(9):2337–2348, 2022.

Zhuofan Zong et al. MoVA: Adapting mixture of vision experts to multimodal context. In NeurIPS, 2024.

## APPENDIX OVERVIEW

The appendix supports specific portions of the main paper:

• App. A.1: conditional $( 1 - e ^ { - \gamma } )$ greedy approximation guarantee under a standard submodularityratio condition, together with a mutual-information specialization that recovers the $( 1 - 1 / \dot { e } )$ bound.

• App. A.2: ordering-consistency proof for $\Delta J$ (Theorem A.2), supporting the empirical-ordering claim at the end of Section 3.3 of the main paper.

• App. A.3: experimental configuration, full results tables, ordering details, risk decomposition, and proof of the conditional marginal-utility crossing (Theorem 1). Supports Sections 2 and 4 of the main paper.

• App. A.4: cross-modal validation on frozen LLMs (supports Section 4.3).

• App. A.5: future work, online and sequential extensions.

The notation of the main paper carries over: $\mathcal { M }$ is the encoder pool of size M, and πˆ is the KAGES greedy ordering on the labelled task, as defined in Section 3 of the main paper.

## A.1 GREEDY APPROXIMATION GUARANTEE UNDER A SUBMODULARITY RATIO

We give the conditional fixed-budget greedy guarantee used in the main paper. The result is stated for a generic non-negative set utility $\breve { U }$ and applies only when the stated monotonicity and submodularityratio conditions hold. We then give a population mutual-information specialization for which these conditions follow from conditional independence.

For m $\notin S ,$ , write $\Delta U ( m \mid S ) : = U ( S \cup \{ m \} ) - U ( S )$

Theorem A.1 (Conditional greedy guarantee under a submodularity ratio). Let $U : 2 ^ { \mathcal { M } } \to \mathbb { R } _ { > 0 }$ satisfy $U ( \emptyset ) = 0$ and be monotone non-decreasing. Fix a cardinality budget K. Suppose there is a constant $\gamma \in ( 0 , 1 ]$ such that,for every $L \subseteq { \mathcal { M } }$ and every $T \subseteq { \mathcal { M } } \setminus L$ with $| T | \leq K$

$$
\sum _ { m \in T } \Delta U ( m \mid L ) \ \geq \ \gamma \big [ U ( L \cup T ) - U ( L ) \big ] .\tag{A.1}
$$

Let $S _ { 0 } = \emptyset$ and let greedy choose

$$
m _ { k } \in \arg \operatorname* { m a x } _ { m \notin S _ { k - 1 } } \Delta U ( m \mid S _ { k - 1 } ) , \qquad S _ { k } = S _ { k - 1 } \cup \{ m _ { k } \} .
$$

Then

$$
U ( S _ { K } ) \geq \big ( 1 - e ^ { - \gamma } \big ) \operatorname* { m a x } _ { | S | \leq K } U ( S ) .
$$

Proof. Let $O \in$ arg ma $\operatorname { \mathrm { \Sigma } } \times _ { | S | \leq K } U ( S )$ and write $G _ { k } : = U ( O ) - U ( S _ { k } )$ . By monotonicity and $( \mathsf { A } . 1 )$

$$
\sum _ { m \in O \setminus S _ { k } } \Delta U ( m \mid S _ { k } ) \ \geq \ \gamma \big [ U ( S _ { k } \cup O ) - U ( S _ { k } ) \big ] \ \geq \ \gamma G _ { k } .
$$

Since $| O \setminus S _ { k } | \leq K$ , the largest available marginal gain is at least $\gamma G _ { k } / K$ . Greedy therefore satisfies

$$
U ( S _ { k + 1 } ) - U ( S _ { k } ) \geq \frac { \gamma } { K } G _ { k } , \qquad G _ { k + 1 } \leq \left( 1 - \frac { \gamma } { K } \right) G _ { k } .
$$

Iterating from $S _ { 0 } = \emptyset$ gives

$$
G _ { K } \leq \left( 1 - \frac { \gamma } { K } \right) ^ { K } U ( O ) \leq e ^ { - \gamma } U ( O ) ,
$$

which proves the claim.

Corollary A.1 (Prefix-wise fixed-budget guarantee). Under the hypotheses of Theorem A.1, for every $k = 1 , \ldots , K$ , the same greedy ordering satisfies

$$
U ( S _ { k } ) \geq ( 1 - e ^ { - \gamma } ) \operatorname* { m a x } _ { | S | \leq k } U ( S ) .
$$

Proof. Apply Theorem A.1 with the budget set to k. The first k greedy choices are unchanged when the traversal is subsequently continued to budget K. □

For the information-theoretic specialization, let

$$
\bar { F } _ { S } : = ( F _ { m } ) _ { m \in S }
$$

denote the joint collection of the selected encoder features.

Lemma A.1 (Conditional independence implies information submodularity). The set function $f ( S ) : = I ( \bar { F } _ { S } ; Y )$ is normalized and monotone non-decreasing. Ifthe encoderfeatures are mutually independent conditional on $Y _ { \textrm { : } }$ , then $f$ is submodular and hence satisfies (A.1) with $U = f a n d \gamma = 1$

Proof. Normalization is immediate. Monotonicity follows from the chain rule,

$$
f ( A \cup \{ m \} ) - f ( A ) = I ( F _ { m } ; Y \mid \bar { F } _ { A } ) \geq 0 .
$$

For $A \subseteq B$ and m $\notin B$

$$
I ( F _ { m } ; Y \mid \bar { F } _ { A } ) = H ( F _ { m } \mid \bar { F } _ { A } ) - H ( F _ { m } \mid Y , \bar { F } _ { A } ) .
$$

Conditional independence gives $H ( F _ { m } \mid Y , \bar { F } _ { A } ) = H ( F _ { m } \mid Y )$ , and similarly for B. Since conditioning cannot increase entropy, $H ( \dot { F } _ { m } \mid \bar { F } _ { A } ) \ge H ( \dot { F } _ { m } \mid \bar { F } _ { B } ) ,$ , so

$$
I ( F _ { m } ; Y \mid \bar { F } _ { A } ) \ge I ( F _ { m } ; Y \mid \bar { F } _ { B } ) .
$$

This is the diminishing-returns condition for submodularity. A normalized monotone submodular function satisfies (A.1) with $\gamma = 1$ □

Corollary ${ \bf A } . 2 \left( \left( 1 - 1 / e \right) \right.$ information-gain guarantee). Under the conditional-independence condition of Lemma A.1, greedy maximization of $f ( \overline { { S } } ) = I ( \bar { F } _ { S } ; Y )$ satisfies

$$
f ( S _ { K } ^ { \mathrm { g r e e d y } } ) \geq ( 1 - 1 / e ) \operatorname* { m a x } _ { | S | \leq K } f ( S ) .
$$

Scope of the guarantee. Theorem A.1 is conditional: it does not assert that every frozen-encoder pool, or the empirical CKA objective used by KAGES, is automatically monotone or has a positive submodularity ratio. Because CKA contains a normalization denominator, non-negative target alignment of a candidate kernel alone does not guarantee a non-negative marginal CKA gain; the exact one-step condition is derived in App. A.3.4. The theorem and Corollary A.1 provide fixedbudget prefix guarantees only when the stated conditions hold.

## A.2 ORDERING CONSISTENCY FOR $\Delta J$

KAGES (Algorithm 1 in the main paper) acts on the empirical utility $\widehat { J } _ { n }$ computed from n labelled samples. This appendix shows that, under uniform subgaussian concentration of the marginal gains and a non-vanishing per-step population margin, empirical KAGES recovers the population greedy ordering with exponentially high probability.

## A.2.1 SETUP AND NOTATION

Let $\mathcal { M } = \{ 1 , \dots , M \}$ be a fixed encoder pool with $M \ \geq \ 2$ . Write $J ( S )$ for a deterministic population target utility (for example, a population limit of the alignment score when such a limit exists) and ${ \widehat { J } } _ { n } ( S )$ for its n-sample empirical estimator. The theorem below is generic and relies only on the stated concentration and bias assumptions; it does not derive those assumptions for empirical CKA. Marginal gains are

$$
\Delta J ( m \mid S ) : = J ( S \cup \{ m \} ) - J ( S ) , \qquad \Delta \widehat { J } _ { n } ( m \mid S ) : = \widehat { J } _ { n } ( S \cup \{ m \} ) - \widehat { J } _ { n } ( S ) .
$$

The population greedy trajectory is $( m _ { t } ^ { \star } , S _ { t } ^ { \star } ) _ { t = 1 } ^ { M }$ <sub>1</sub> with $S _ { 0 } ^ { \star } = \varnothing$ and

$$
m _ { t } ^ { \star } \in \arg \operatorname* { m a x } _ { m \notin S _ { t - 1 } ^ { \star } } \Delta J ( m \mid S _ { t - 1 } ^ { \star } ) , \qquad S _ { t } ^ { \star } = S _ { t - 1 } ^ { \star } \cup \{ m _ { t } ^ { \star } \} .
$$

The empirical trajectory $( \hat { m } _ { t } , \hat { S } _ { t } )$ is defined analogously using $\Delta \widehat { J } _ { n }$ . Ties are resolved by the same fixed deterministic rule in both trajectories.

## A.2.2 ASSUMPTIONS

Assumption A.1 (Sample independence and pool finiteness). The n labelled observations are i.i.d.   
from a fixed joint distribution, and the pool cardinality M is fixed independently of n.

Assumption A.2 (Uniform subgaussian concentration with bias). For every m $\in \mathcal { M }$ and every $S \subseteq \bar { \mathcal { M } } \backslash \{ m \}$

$$
\operatorname* { P r } \Big [ \Big | \Delta \widehat { J } _ { n } ( m \mid S ) - \mathbb { E } \Delta \widehat { J } _ { n } ( m \mid S ) \Big | > t \Big ] \leq 2 \exp \left( - \frac { n t ^ { 2 } } { 2 \sigma ^ { 2 } } \right) \quad f o r a l l t > 0 ,
$$

with $\sigma > 0$ . Moreover, the bias

$$
b _ { n } ( m , S ) : = \mathbb { E } \Delta \widehat { J } _ { n } ( m \mid S ) - \Delta J ( m \mid S )
$$

satisfies $| b _ { n } ( m , S ) | \leq B / { \sqrt { n } }$ uniformly over $( m , S )$

Assumption A.3 (Greedy ordering margin). For every step $t \in \{ 1 , \ldots , M - 1 \}$

$$
\Delta J ( m _ { t } ^ { \star } \mid S _ { t - 1 } ^ { \star } ) - \operatorname* { m a x } _ { m \nmid S _ { t - 1 } ^ { \star } \cup \{ m _ { t } ^ { \star } \} } \Delta J ( m \mid S _ { t - 1 } ^ { \star } ) \geq G
$$

for some $G > 0 .$

## A.2.3 THEOREM STATEMENT

Theorem A.2 (Ordering consistency for $\Delta J )$ . Under Assumptions A.1–A.3, $i f n \geq ( 4 B / G ) ^ { 2 }$ , then

$$
\operatorname* { P r } \bigl [ \hat { \pi } \neq ( m _ { 1 } ^ { \star } , \ldots , m _ { M } ^ { \star } ) \bigr ] \leq \bigl [ M ( M + 1 ) - 2 \bigr ] \exp \mathopen { } \mathclose \bgroup \left( - \frac { n G ^ { 2 } } { 3 2 \sigma ^ { 2 } } \aftergroup \egroup \right) .
$$

Consequently, a targetfailure probability $\epsilon \in ( 0 , 1 )$ is attained whenever

$$
n \geq \operatorname* { m a x } \left\{ \left( { \frac { 4 B } { G } } \right) ^ { 2 } , { \frac { 3 2 \sigma ^ { 2 } } { G ^ { 2 } } } \log { \frac { M ( M + 1 ) - 2 } { \epsilon } } \right\} .
$$

In particular, the sufficient sample size is ${ \cal O } \big ( ( B ^ { 2 } + \sigma ^ { 2 } [ \log M + \log ( 1 / \epsilon ) ] ) / G ^ { 2 } \big )$

## A.2.4 PROOF

Step 1: deviation from the population marginal gain. $\mathrm { I f } n \geq ( 4 B / G ) ^ { 2 }$ , then $| b _ { n } ( m , S ) | \leq G / 4$ Hence, for any fixed $( m , S )$

$$
\begin{array} { r l r } {  { \operatorname* { P r } \bigg [ \bigg | \Delta \widehat { J } _ { n } ( m \mid S ) - \Delta J ( m \mid S ) \bigg | \geq \frac { G } { 2 } \bigg ] } } \\ & { } & { \leq \operatorname* { P r } \bigg [ \bigg | \Delta \widehat { J } _ { n } ( m \mid S ) - \mathbb { E } \Delta \widehat { J } _ { n } ( m \mid S ) \bigg | \geq \frac { G } { 4 } \bigg ] \leq 2 \exp \bigg ( - \frac { n G ^ { 2 } } { 3 2 \sigma ^ { 2 } } \bigg ) . } \end{array}
$$

Step 2: deterministic good events along the population path. For every population-path step $t \in \{ 1 , \ldots , M - 1 \}$ and every candidate m $\notin S _ { t - 1 } ^ { \star }$ , define

$$
\mathcal { E } _ { t , m } : = \left\{ \left| \Delta \widehat { J } _ { n } ( m \mid S _ { t - 1 } ^ { \star } ) - \Delta J ( m \mid S _ { t - 1 } ^ { \star } ) \right| < \frac { G } { 2 } \right\} ,
$$

and let $\begin{array} { r } { \mathcal { E } : = \bigcap _ { t = 1 } ^ { M - 1 } \bigcap _ { m \not \in S _ { t - 1 } ^ { \star } } \mathcal { E } _ { t , m } } \end{array}$ . These events are defined on the deterministic population greedy path before the empirical path is run, so no conditioning on a data-dependent previous-selection event is needed.

On $\mathcal { E } ,$ the empirical and population paths agree by induction. Suppose $\hat { S } _ { t - 1 } = S _ { t - 1 } ^ { \star }$ . For any remaining competitor m $\neq m _ { t } ^ { \star }$ , the population margin and the two strict $G / 2$ error bounds give

$$
\begin{array} { r l } & { \Delta \widehat { J } _ { n } ( m _ { t } ^ { \star } \mid S _ { t - 1 } ^ { \star } ) - \Delta \widehat { J } _ { n } ( m \mid S _ { t - 1 } ^ { \star } ) } \\ & { \quad > \Delta J ( m _ { t } ^ { \star } \mid S _ { t - 1 } ^ { \star } ) - \Delta J ( m \mid S _ { t - 1 } ^ { \star } ) - G \geq 0 . } \end{array}
$$

Thus the empirical rule selects $m _ { t } ^ { \star }$ . Starting from $S _ { 0 } ^ { \star } = \hat { S } _ { 0 } = \varnothing$ , induction proves equality of the paths through step $M - 1 ;$ the final step has only one candidate.

Step 3: one unconditional union bound. Step 1 and a union bound over the deterministic family of events yield

$$
\begin{array} { r } { \mathrm { P r } ( \mathcal { E } ^ { c } ) \leq 2 \displaystyle \sum _ { t = 1 } ^ { M - 1 } ( M - t + 1 ) \exp \left( - \frac { n G ^ { 2 } } { 3 2 \sigma ^ { 2 } } \right) } \\ { = \left[ M ( M + 1 ) - 2 \right] \exp \left( - \frac { n G ^ { 2 } } { 3 2 \sigma ^ { 2 } } \right) . } \end{array}
$$

Since E implies recovery of the full ordering, this is the claimed failure-probability bound. Solving it for a target failure probability ϵ proves the sample-complexity statement. ■

Remark. The polynomial prefactor is a conservative consequence of a uniform union bound along the population greedy path. The exponential dependence on $n \dot { G } ^ { 2 } / \sigma ^ { 2 }$ and the logarithmic dependence on M are the substantive parts of the result.

## A.3 EXPERIMENTAL DETAILS, THEOREMS, AND FULL TABLES

This appendix expands the experimental sections of the main paper (Sections 2, 4). It contains the configuration, the formal risk decomposition, the proof of the conditional marginal-utility crossing (Theorem 1), and the full results tables underlying the main-text figures.

## A.3.1 EXPERIMENTAL CONFIGURATION

Taxonomies (full). The main paper introduces two taxonomies that recur throughout the experiments. Encoder paradigm partitions pre-trained vision backbones into five categories: (A) language-supervised image–text contrastive models (CLIP (Radford et al., 2021), SigLIP (Zhai et al., 2023)); (B) self-supervised models trained without labels (DINOv2 (Oquab et al., 2024), MAE (He et al., 2022), Data2Vec (Baevski et al., 2022)); (C) generative / image-prior encoders (Stable-Diffusion VAE (Rombach et al., 2022)); (D) class-supervised ImageNet-trained models (ConvNeXt (Liu et al., 2022), ResNet-50 (He et al., 2016), ViT-B/16 (Dosovitskiy et al., 2021)); (E) task-expert encoders trained for dense vision tasks (SAM (Kirillov et al., 2023)). Task recognition regime partitions the five evaluation datasets one-to-one: Generic Object (CIFAR-100), Fine-grained Object (Pets), Texture / Material (DTD), Geo-spatial / Scene (Country211), and Structured Symbol / Digit (GTSRB).

Reference-path justification. The diagnostic reference path CLIP → DINOv2 → ConvNeXt → SigLIP → MAE → ResNet-50 is fixed a priori from published fusion practice and not optimised on the evaluation data. Its first three positions are the Cambrian-1 (Tong et al., 2024) trio (CLIP, DINOv2, ConvNeXt), reported there as a strong vision-centric combination (Shi et al., 2025; Kar et al., 2024; Zong et al., 2024); the last three add a within-paradigm second representative each (SigLIP for language supervision, MAE for self-supervision, ResNet-50 for class supervision), so positions 5–6 act as redundancy probes against positions 1–3. Self-supervised purely-visual features (DINOv2) are added early to recover what contrastive pre-training misses (Liu et al., 2023).

Hardware. All experiments are conducted on a single NVIDIA RTX 5090 GPU with 32GB VRAM.

Model pool. We use 10 frozen pre-trained encoders spanning all five pre-training paradigms (consistent with Section 4 of the main paper): CLIP-ViT-B/16 (Radford et al., 2021) and SigLIP-B/16 (Zhai et al., 2023) (language-supervised); DINOv2-base (Oquab et al., 2024), ViT-MAEbase (He et al., 2022) and Data2Vec (Baevski et al., 2022) (self-supervised); the Stable-Diffusion VAE encoder (Rombach et al., 2022) (generative); ConvNeXt-base (Liu et al., 2022), ResNet-50 (He et al., 2016) and ViT-B/16 (Dosovitskiy et al., 2021) (class-supervised); and SAM-ViT-B (Kirillov et al., 2023) (task-expert). The generative and task-expert encoders are reintroduced here as adversarial candidates that a competent selector should reject. Feature dimensions range from 256 (SD-VAE) to 2048 (ResNet-50).

Datasets. Both the diagnostic protocol (Section 2, 6-encoder pool) and the main-results protocol (Section 4, 10-encoder pool) evaluate on the same five datasets, one per recognition regime: CIFAR-100 (100 classes, generic object), Oxford Pets (37 classes, fine-grained object), DTD (47 classes, texture), Country211 (211 classes, geo-spatial scene) and GTSRB (43 classes, structured symbol/digit). Oxford Flowers-102 additionally appears only in the fair 16-shot comparison against single-model CLIP adaptation (Table A.3), following the standard few-shot adaptation benchmark. All datasets are public benchmarks used under their original train/test splits.

Setting B (Concat + MLP). Features from selected encoders are concatenated (no normalization) and fed to a 2-layer MLP: Linear(input\_dim → 512) → ReLU → Dropout $_ { \cdot ( 0 . 1 ) } $ Linear(512 → num\_classes). Training: Adam optimizer, $\mathrm { l r } = 1 0 ^ { - 3 }$ , weight decay $1 0 ^ { - \dot { 4 } }$ , batch size 128, 20 epochs.

Selection cost. KAGES needs only the centred encoder kernels and a greedy traversal, with no downstream training. Computing the ten $n \times n$ centred kernels takes ${ \sim } 5 \mathrm { s }$ on a single GPU, and the full greedy ordering (three Frobenius inner products per candidate per step, $O ( n ^ { 2 } )$ and independent of encoder dimension) completes in under one second. The total selection cost is a few seconds, against hours for exhaustive evaluation of all $2 ^ { 1 0 } - 1 = 1 0 2 3$ subsets.

Reproducibility. Main-table experiments use five random seeds {17, 23, 42, 71, 137} and report mean ± standard deviation. Paired method comparisons use 10,000 bootstrap resamples for 95% confidence intervals.

## A.3.2 RISK DECOMPOSITION (THEOREM A.3) AND PROOF

For a selected set S, let

$$
\bar { F } _ { S } : = ( F _ { m } ) _ { m \in S }
$$

denote the joint collection of its encoder features. Let $R ^ { \star } ( S )$ be the Bayes risk given $\bar { F } _ { S }$ , and let $R _ { n } ( S )$ be the expected population test risk of the predictor returned by the n-sample training procedure. Define the finite-sample excess risk

$$
E _ { n } ( S ) : = R _ { n } ( S ) - R ^ { \star } ( S ) \geq 0 .
$$

This exact excess term may absorb estimation, approximation, and optimization effects of the specified downstream training procedure.

Theorem A.3 (Risk decomposition). For a nested growth path $S _ { 0 } \subset S _ { 1 } \subset \cdots \subset S _ { K }$ , the change in finite-sample risk at step k decomposes as

$$
{ R } _ { n } ( S _ { k } ) - { R } _ { n } ( S _ { k - 1 } ) = \Delta { E } _ { k } - \delta _ { k } ^ { \star } ,
$$

where

$$
\Delta E _ { k } : = E _ { n } ( S _ { k } ) - E _ { n } ( S _ { k - 1 } ) , \qquad \delta _ { k } ^ { \star } : = R ^ { \star } ( S _ { k - 1 } ) - R ^ { \star } ( S _ { k } ) \geq 0 .
$$

Proof. Subtract $R _ { n } ( S _ { k - 1 } ) = R ^ { \star } ( S _ { k - 1 } ) + { \cal E } _ { n } ( S _ { k - 1 } )$ from ${ \cal R } _ { n } ( S _ { k } ) = { \cal R } ^ { \star } ( S _ { k } ) + { \cal E } _ { n } ( S _ { k } )$ . The non-negativity of $\delta _ { k } ^ { \star }$ follows because $\bar { F } _ { S _ { k - 1 } }$ is a subcollection of $F _ { S _ { k } } \colon$ a Bayes decision rule given $\bar { F } _ { S _ { k } }$ can ignore the newly added feature and reproduce any rule based on $\bar { F } _ { S _ { k - 1 } }$ . The exact increment $\Delta \ddot { E } _ { k }$ need not be non-negative. □

## A.3.3 MARGINAL-UTILITY CROSSING: PROOF OF THEOREM 1

This appendix gives the derivation behind Theorem 1 of the main paper. Along a fixed ordering, write

$$
g _ { k } : = R ^ { \star } ( S _ { k - 1 } ) - R ^ { \star } ( S _ { k } ) , \qquad c _ { k } : = E _ { n } ( S _ { k } ) - E _ { n } ( S _ { k - 1 } ) , \qquad a _ { k } : = g _ { k } - c _ { k } .
$$

Step 1: one-step identity. For every $k ,$

$$
R _ { n } ( S _ { k } ) - R _ { n } ( S _ { k - 1 } ) = c _ { k } - g _ { k } = - a _ { k } .
$$

This follows immediately by subtracting the two risk decompositions in Theorem A.3.

Step 2: log loss makes $g _ { k }$ a conditional mutual information. Under log loss $\ell ( q , y ) = - \log q ( y )$ the Bayes minimizer is the true conditional distribution and

$$
R _ { \log } ^ { \star } ( S ) = H ( Y \mid \bar { F } _ { S } ) .
$$

Since $\bar { F } _ { S _ { k } } = ( \bar { F } _ { S _ { k - 1 } } , F _ { m _ { k } } )$

$$
g _ { k , \mathrm { l o g } } = I ( F _ { m _ { k } } ; Y \mid \bar { F } _ { S _ { k - 1 } } ) .
$$

The chain rule gives

$$
\begin{array} { r } { I ( F _ { m _ { k } } ; Y \mid \bar { F } _ { S _ { k - 1 } } ) = \underbrace { I ( F _ { m _ { k } } ; Y ) } _ { \mathrm { r e l e v a n c e } } - \underbrace { I ( F _ { m _ { k } } ; \bar { F } _ { S _ { k - 1 } } ) } _ { \mathrm { r e d u n d a n c y } } + \underbrace { I ( F _ { m _ { k } } ; \bar { F } _ { S _ { k - 1 } } \mid Y ) } _ { \mathrm { c l a s s - c o n d i t i o n a l ~ a d j u s t m e n t } } . } \end{array}
$$

Step 3: a submodular greedy path has non-increasing Bayes gains. Let $f ( S ) : = I ( \bar { F } _ { S } ; Y )$ and suppose $f$ is submodular along the path. If

$$
m _ { k } \in \arg \operatorname* { m a x } _ { m \notin S _ { k - 1 } } \Delta f ( m \mid S _ { k - 1 } ) ,
$$

then

$$
g _ { k + 1 } = \operatorname* { m a x } _ { m \not \in { S _ { k } } } \Delta f ( m \mid S _ { k } ) \leq \operatorname* { m a x } _ { m \not \in { S _ { k } } } \Delta f ( m \mid S _ { k - 1 } ) \leq g _ { k } .
$$

Thus submodularity plus greedy selection is one sufficient mechanism for the non-increasing-g<sub>k</sub> assumption in Theorem 1.

Step 4: Gaussian OLS instance with an increasing pure-noise tail. Let $X \sim \mathcal { N } ( 0 , I _ { M } ) , Y =$ $\begin{array} { r } { \sum _ { j = 1 } ^ { r ^ { - } } \beta _ { j } X _ { j } + \varepsilon } \end{array}$ with $\varepsilon \sim \mathcal { N } ( 0 , \sigma ^ { 2 } ) , F _ { m _ { j } } ( X ) = X _ { j }$ , and $S _ { k } = \{ m _ { 1 } , . . . , m _ { k } \}$ . For an OLS probe trained on n $> k + 1$ samples,

$$
\mathbb { E } R _ { n } ( S _ { k } ) = \nu _ { k } ^ { 2 } \frac { n - 1 } { n - k - 1 } , \qquad \nu _ { k } ^ { 2 } : = \sigma ^ { 2 } + \sum _ { j = k + 1 } ^ { r } \beta _ { j } ^ { 2 } .
$$

Here $R ^ { \star } ( S _ { k } ) = \nu _ { k } ^ { 2 } .$ , so

$$
E _ { n } ( S _ { k } ) = \nu _ { k } ^ { 2 } \frac { k } { n - k - 1 } .
$$

For every pure-noise addition $k > r , \nu _ { k - 1 } ^ { 2 } = \nu _ { k } ^ { 2 } = \sigma ^ { 2 }$ , and hence

$$
c _ { k } = E _ { n } ( S _ { k } ) - E _ { n } ( S _ { k - 1 } ) = \sigma ^ { 2 } \frac { n - 1 } { ( n - k ) ( n - k - 1 ) } > 0 .
$$

This quantity is strictly increasing with k while $n > k + 1$ . Thus the Gaussian model gives an explicit increasing-cost tail; it does not assert that $c _ { k }$ is non-decreasing for arbitrary signal-bearing additions. As a concrete full-path instance, take $r = 1$ and $n > K + 1$ . Then $g _ { 1 } = \bar { \beta } _ { 1 } ^ { 2 } , \bar { c } _ { 1 } = \sigma ^ { 2 } / ( \bar { n } - 2 )$ , and for every $k \geq 2 , g _ { k } = 0 < c _ { k }$ . Therefore $\beta _ { 1 } ^ { 2 } > \sigma ^ { 2 } / ( n - 2 )$ yields a strict single crossing after the first step.

Step 5: proof of the conditional crossing theorem. Under the assumptions of Theorem $1 , g _ { k }$ is non-increasing and $c _ { k }$ is non-decreasing, so $\ : a _ { k } = g _ { k } - c _ { k } \ :$ is non-increasing. Let

$$
k ^ { \star } : = \operatorname* { m a x } \{ k : a _ { k } > 0 \}
$$

and let $\eta : = \mathrm { m i n } _ { k } \left| a _ { k } \right| > 0$ denote the no-tie margin. Then $a _ { k } \geq \eta > 0$ for $k \leq k ^ { \star }$ and $a _ { k } \leq - \eta < 0$ for $k > k ^ { \star } ;$ ; the sign change occurs between $k ^ { \star }$ and $k ^ { \star } + 1$ . By Step 1, $R _ { n } ( S _ { k } ) - R _ { n } ( S _ { k - 1 } ) = - a _ { k }$ which is strictly negative up to $k ^ { \star }$ and strictly positive afterwards. This proves the asserted orderspecific peak-then-decline path. □

## A.3.4 EXACT CKA MARGIN DERIVATION

We give the full derivation of Proposition 3 in the main paper. Recall the five scalars $A _ { S } : =$ $\langle { \bar { K } } _ { S } ^ { c } , { \bar { K } } _ { Y } ^ { c } \rangle _ { F } , \ : B _ { S _ { \lambda } } : = \langle { \bar { K } } _ { S } ^ { c } , { \bar { K } } _ { S } ^ { c } \rangle _ { F } , \ : \ : \hat { r _ { m } } : = \langle { \bar { K } } _ { m } ^ { c } , { \bar { K } } _ { Y } ^ { c } \rangle _ { F } , \ : h _ { m } : \overset {  } { = } \langle { \bar { K } } _ { m } ^ { c } , { \bar { K } } _ { S } ^ { c } \rangle _ { F } , \ : q _ { m } : = \langle { \bar { K } } _ { m } ^ { c } , { \bar { K } } _ { m } ^ { c } \rangle _ { F } .$ with $C \doteq \| K _ { Y } ^ { c } \| _ { F } ^ { 2 }$ . This derivation concerns the pure alignment utility U, equivalently J with $\tau = 0$

One-step CKA after adding m. Adding m to $S$ gives $K _ { S \cup \{ m \} } ^ { c } = K _ { S } ^ { c } + K _ { m } ^ { c } .$ , so

$$
A _ { S \cup \{ m \} } = A _ { S } + r _ { m } , \qquad B _ { S \cup \{ m \} } = B _ { S } + 2 h _ { m } + q _ { m } , \qquad U ( S \cup \{ m \} ) = \frac { A _ { S } + r _ { m } } { \sqrt { ( B _ { S } + 2 h _ { m } + q _ { m } ) C } } .
$$

Positive-gain condition. Assume $B _ { S } > 0$ and $C > 0$ . Since the centred Gram matrices are positive semidefinite, $A _ { S } , r _ { m } , h _ { m } , q _ { m } \geq 0 .$ . Then $U ( S \cup \{ m \} ) > U ( S )$ is equivalent to $\begin{array} { r } { \frac { A _ { S } + r _ { m } } { \sqrt { B _ { S } + 2 h _ { m } + q _ { m } } } > } \end{array}$ $\scriptstyle { \frac { A _ { S } } { \sqrt { B _ { S } } } }$ , which (with $A _ { S } \geq 0 ,$ , both sides non-negative) is equivalent to $( A _ { S } + r _ { m } ) ^ { 2 } B _ { S } > A _ { S } ^ { 2 } ( B _ { S } +$ $\mathrm { ~ \it { 2 } } \dot { h } _ { m } + q _ { m } ) , \mathrm { i . e . \ } r _ { m } ^ { 2 } B _ { S } + 2 A _ { S } r _ { m } B _ { S } > A _ { S } ^ { 2 } ( 2 h _ { m } + q _ { m } )$ . Treating this as a quadratic in $r _ { m }$ and taking the positive root yields the threshold $r _ { m } ^ { \star } = A _ { S } ( \sqrt { 1 + ( 2 h _ { m } + q _ { m } ) / B _ { S } } - 1 )$ , which is the exact criterion of Proposition 3.

## A.3.5 DIAGNOSTIC-PROTOCOL SCALING TABLES

Table A.1 reports the full per-dataset accuracy curves that underlie Figure 1 of the main paper.

Table A.1: 10-shot accuracy $( \% )$ as encoders are incrementally added along the literature-tracked path (CLIP → DINOv2 → ConvNeXt → SigLIP → MAE → ResNet-50; gated fusion). These are the per-dataset curves underlying Figure 1 and Table 1. Bold: peak. ⋆: CLIP alone is optimal.
<table><tr><td>Dataset</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>Peak</td></tr><tr><td>CIFAR-100</td><td>60.29</td><td>80.11</td><td>80.37</td><td>80.18</td><td>80.20</td><td>79.92</td><td>3</td></tr><tr><td>Pets</td><td>85.04</td><td>92.86</td><td>93.75</td><td>93.46</td><td>93.65</td><td>93.49</td><td>3</td></tr><tr><td>DTD</td><td>64.41</td><td>72.67</td><td>73.52</td><td>74.05</td><td>74.17</td><td>74.18</td><td>6</td></tr><tr><td>Country211★</td><td>13.50</td><td>9.85</td><td>9.91</td><td>9.82</td><td>9.71</td><td>9.58</td><td>1</td></tr><tr><td>GTSRB*</td><td>61.15</td><td>51.54</td><td>52.73</td><td>52.34</td><td>51.97</td><td>52.58</td><td>1</td></tr></table>

## A.3.6 TWO FAILURE MODES (FORMAL CHARACTERIZATION)

The risk decomposition (Theorem $\mathbf { A . } 3 )$ shows that step k degrades performance exactly when the excess-risk increment $c _ { k } = \Delta E _ { k }$ exceeds the Bayes-risk reduction $g _ { k } = \delta _ { k } ^ { \star }$ . Two empirically distinct mechanisms can make this net utility negative.

TYPE I: TASK-CONDITIONAL REDUNDANCY.

Definition A.1 (Task-conditional redundancy). Encoder $m _ { k }$ is task-conditionally redundant with respect to $S _ { k - 1 }$ for label Y if

$$
I ( F _ { m _ { k } } ; Y \mid \bar { F } _ { S _ { k - 1 } } ) = 0 .
$$

Proposition A.1 (Type I removes the Bayes gain). Under log loss, $i f m _ { k }$ is task-conditionally redundant with respect to $S _ { k - 1 }$ , then $g _ { k } = \delta _ { k } ^ { \star } = 0$ and

$$
R _ { n } ( S _ { k } ) - R _ { n } ( S _ { k - 1 } ) = \Delta E _ { k } .
$$

Consequently, the step degrades performance exactly when $\Delta E _ { k } > 0$ . The same conclusion holdsfor any Bayes decision loss whenever $Y$ is conditionally independent of $F _ { m _ { k } }$ given $\bar { F } _ { S _ { k - 1 } }$

Proof. Under log loss,

$$
g _ { k } = H ( Y \mid \bar { F } _ { S _ { k - 1 } } ) - H ( Y \mid \bar { F } _ { S _ { k } } ) = I ( F _ { m _ { k } } ; Y \mid \bar { F } _ { S _ { k - 1 } } ) = 0 .
$$

The one-step identity then gives $R _ { n } ( S _ { k } ) - R _ { n } ( S _ { k - 1 } ) = \Delta E _ { k }$

TYPE II: TASK–MODEL MISALIGNMENT.

Definition A.2 (Task–model misalignment). For a tolerance $\epsilon \geq 0 .$ , encoder $m _ { k }$ is ϵ-task-misaligned relative to $S _ { k - 1 }$ if its conditional Bayes gain satisfies

$$
g _ { k } = R ^ { \star } ( S _ { k - 1 } ) - R ^ { \star } ( S _ { k } ) \leq \epsilon .
$$

Under log loss this is equivalent to $I ( F _ { m _ { k } } ; Y \mid \bar { F } _ { S _ { k - 1 } } ) \le \epsilon$ . The candidate may nevertheless be representationally distinct according to a task-agnostic diversity criterion.

Proposition A.2 (Type II decline when excess cost dominates). If m<sub>k</sub> is ϵ-task-misaligned relative to $S _ { k - 1 }$ and $c _ { k } > \epsilon ,$ then

$$
R _ { n } ( S _ { k } ) - R _ { n } ( S _ { k - 1 } ) = c _ { k } - g _ { k } \geq c _ { k } - \epsilon > 0 .
$$

Proof. The claim follows directly from the one-step identity and the definition $g _ { k } \leq \epsilon .$ □

## A.3.7 IMAGE RETRIEVAL: FULL RECALL@1 CURVES

With the same KAGES ordering used in the main text, L2-normalized concatenated features, and nearest-neighbor search, we report Recall@1 as a function of k on three datasets (Table A.2).

Table A.2: Image retrieval Recall@1 (%) vs. k. Retrieval gains are task-dependent and saturate at larger k than classification.
<table><tr><td>Dataset</td><td>k = 1</td><td>3</td><td>5</td><td>8</td><td>10</td><td>Peak k</td><td>Best R@1</td></tr><tr><td>DTD</td><td>47.4</td><td>66.6</td><td>70.1</td><td>70.5</td><td>74.6</td><td>10</td><td>74.6</td></tr><tr><td>GTSRB</td><td>50.4</td><td>56.8</td><td>55.7</td><td>59.2</td><td>60.2</td><td>10</td><td>60.2</td></tr><tr><td>Pets</td><td>89.8</td><td>93.3</td><td>93.5</td><td>94.3</td><td>94.0</td><td>8</td><td>94.3</td></tr></table>

## A.3.8 FAIR 16-SHOT COMPARISON WITH SINGLE-MODEL ADAPTERS

Table A.3 compares multi-model fusion with recent single-model CLIP adaptation methods under a fair 16-shot setting.

Table A.3: Fair 16-shot comparison: multi-model fusion vs. single-model CLIP adaptation (%). All methods use 16 shots per class. †: from original papers.
<table><tr><td colspan="2">Method</td><td>DTD</td><td>Flowers</td><td>GTSRB</td><td>Avg</td></tr><tr><td rowspan="5">Sige</td><td>CLIP zero-shot†</td><td>44.3</td><td>67.4</td><td></td><td></td></tr><tr><td>CoOp† (Zhou et al., 2022)</td><td>68.7</td><td>95.7</td><td></td><td></td></tr><tr><td>Tip-Ädapter-F† (Zhang et al., 2022)</td><td>73.7</td><td>97.2</td><td></td><td></td></tr><tr><td>TaskRes† (Yu et al., 2023)</td><td>73.4</td><td>97.0</td><td></td><td></td></tr><tr><td>CaFo† (Zhang et al., 2023)</td><td>74.6</td><td>97.5</td><td>一</td><td></td></tr><tr><td rowspan="3">Mlt</td><td>All Models</td><td>77.9</td><td>99.6</td><td>61.6</td><td>79.7</td></tr><tr><td>Random</td><td>77.7</td><td>99.6</td><td>60.3</td><td>79.2</td></tr><tr><td>KAGES</td><td>78.7</td><td>99.7</td><td>72.1</td><td>83.5</td></tr></table>

## A.3.9 FULL LEADERBOARD ON PROTOCOL B

The main paper reports KAGES against the standard multi-view defaults (full fusion, random, diversity) and summarises the two stronger subset-selection baselines (DPP, submodular) in the text. Table A.4 gives the complete per-dataset AULC for all six multi-view methods on Protocol B (16-shot, Concat+MLP), means over five seeds. KAGES is best on every dataset and on the average; the coverage-driven DPP and submodular objectives are the strongest competitors but still trail KAGES, since neither uses the label kernel.

## A.3.10 NORMALIZATION ABLATION

The main paper uses the raw additive variant $\begin{array} { r } { K _ { S } ^ { c } = \sum _ { s \in S } K _ { s } ^ { c } } \end{array}$ . A per-encoder Frobenius-normalised variant $\widetilde { K } _ { s } ^ { c } : = K _ { s } ^ { c } / ( \| K _ { s } ^ { c } \| _ { F } + \varepsilon )$ gives provable per-encoder scale invariance but yields slightly lower empirical AULC on our pool $( \Delta \mathrm { { \bar { A } U L } \bar { C } \approx - 0 . 4 p p }$ on Protocol B). We report it here as an ablation and use the raw additive variant in the main results.

Table A.4: Full per-dataset AULC (%) on Protocol B (16-shot, Concat+MLP), means over 5 seeds {17, 23, 42, 71, 137}. Bold: best per column; underlined: second-best.
<table><tr><td>Method</td><td>C100</td><td>Pets</td><td>DTD</td><td>C211</td><td>GTSRB</td><td>Avg</td></tr><tr><td>Fuse-All</td><td>78.54</td><td>94.06</td><td>75.35</td><td>10.16</td><td>57.18</td><td>63.06</td></tr><tr><td>Random</td><td>77.06</td><td>92.97</td><td>74.87</td><td>11.10</td><td>59.16</td><td>63.03</td></tr><tr><td>Diversity</td><td>80.61</td><td>90.91</td><td>74.38</td><td>11.27</td><td>59.12</td><td>63.26</td></tr><tr><td>DPP</td><td>83.62</td><td>93.21</td><td>72.48</td><td>11.87</td><td>64.88</td><td>65.21</td></tr><tr><td>Submodular</td><td>82.82</td><td>93.31</td><td>77.64</td><td>11.34</td><td>65.53</td><td>66.13</td></tr><tr><td>KAGES (ours)</td><td>83.84</td><td>94.70</td><td>77.82</td><td>12.16</td><td>66.12</td><td>66.93</td></tr></table>

## A.3.11 INTER-SEED STABILITY OF AULC

Across the five seeds {17, 23, 42, 71, 137}, the inter-seed standard deviation of AULC is small and does not affect the method ranking: it is at most 0.3 pp for KAGES on every dataset and below 0.6 pp for all baselines, so the per-dataset margins reported in the main tables are stable across seeds.

## A.4 CROSS-MODAL VALIDATION ON FROZEN LLMS

To test whether KAGES transfers beyond vision, we run a parallel experiment on 6 frozen base LLMs: Qwen2.5-7B (Yang et al., 2024), Mistral-7B-v0.1, Llama-3-8B (Dubey et al., 2024), DeepSeek-7B, Gemma-7B, and Yi-1.5-9B. For each model we extract the last-layer, last-token hidden state on MMLU-500 and ARC-Challenge-500 (both 4-way MCQA, treated as classification), then apply the exact same pipeline as our main experiments: concatenate features of the first k selected models, train a linear probe on an 80/20 split, and report accuracy (Table A.5).

Table A.5: Linear-probe accuracy (%) across k on frozen LLMs. Peak-then-decline reproduces on both tasks. KAGES peaks earlier and higher on MMLU.
<table><tr><td>Dataset</td><td>Method</td><td>k = 1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>Peak</td></tr><tr><td rowspan="3">MMLU-500</td><td>KAGES</td><td>67.0</td><td>75.0</td><td>69.0</td><td>72.0</td><td>71.0</td><td>67.0</td><td>75.0</td></tr><tr><td>diversity-greedy</td><td>67.0</td><td>66.0</td><td>70.0</td><td>69.0</td><td>71.0</td><td>68.0</td><td>71.0</td></tr><tr><td>fixed order</td><td>67.0</td><td>68.0</td><td>72.0</td><td>70.0</td><td>69.0</td><td>67.0</td><td>72.0</td></tr><tr><td rowspan="3">ARC-Ch-500</td><td>KAGES</td><td>88.0</td><td>86.0</td><td>89.0</td><td>91.0</td><td>88.0</td><td>87.0</td><td>91.0</td></tr><tr><td>diversity-greedy</td><td>88.0</td><td>86.0</td><td>89.0</td><td>90.0</td><td>89.0</td><td>87.0</td><td>90.0</td></tr><tr><td>fixed order</td><td>85.0</td><td>88.0</td><td>89.0</td><td>85.0</td><td>91.0</td><td>89.0</td><td>91.0</td></tr></table>

![](images/761234167e467b3ba8e3a2e2bb0f14411824a495f9e31331af82ddedfbeab5d0.jpg)  
Figure A.1: CKA matrices across 6 LLMs. Mistral-Llama CKA reaches 0.82 on MMLU (both English-heavy pretraining), while Gemma (Google, distilled) stays below 0.30 with every other model. This echoes our vision finding: training corpus/objective, not architecture, drives representation similarity.

Findings. (i) The peak-then-decline curve reproduces: MMLU peaks at $k = 2$ for KAGES (75%) and at $k = 3$ to 5 for the baselines, while ARC peaks at $k = 4$ to 5. No configuration benefits from all 6 models. (ii) KAGES’s three-term score beats diversity-greedy by up to 9pp at k = 2 on MMLU, consistent with our main-paper observation that pure diversity is necessary but not sufficient. (iii) The CKA matrix supports the Platonic Representation Hypothesis (Huh et al., 2024) across modalities: Mistral and Llama, both English-dominant base models, cluster tightly, while Gemma’s distillation-heavy training produces the most isolated representation.

## A.5 FUTURE WORK: ONLINE AND SEQUENTIAL EXTENSIONS

The present framework is offline: the labelled task is fixed at n samples, the encoder pool is finite, and the selection is computed in a single pass. Two extensions are natural future directions.

Sequential / streaming labels. If labelled samples arrive sequentially, the ordering πˆ and the optional score-selected size $\hat { k } _ { J }$ can be recomputed as the empirical score path tightens. The sequentialanalysis literature (Tartakovsky et al., 2014) provides change-point detection tools that could be adapted to detect significant shifts in the selection. Active best-arm identification (Kaufmann et al., 2016) provides sample-complexity templates for adaptive selection.

Online ordering and score-based size selection. The ordering and the optional score-selected size can be recomputed as new samples arrive. Under its stated assumptions, Theorem A.2 implies stabilization of the ordering after $\dot { O ( ( B ^ { 2 } + \sigma ^ { 2 } [ \log M + \log ( 1 / \epsilon ) ] ) / G ^ { 2 } ) }$ samples with failure probability at most $\epsilon ;$ it does not by itself prove stabilization or accuracy optimality of $\hat { k } _ { J }$ . An online variant could expose intermediate orderings and score paths with confidence intervals. The connection to bandit best-arm identification (Even-Dar et al., 2002; 2006; Kaufmann et al., 2016) is natural and we leave the formal analysis to a follow-up.