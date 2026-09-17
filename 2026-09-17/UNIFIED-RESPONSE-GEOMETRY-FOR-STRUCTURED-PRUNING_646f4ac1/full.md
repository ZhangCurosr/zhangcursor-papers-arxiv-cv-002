# UNIFIED RESPONSE GEOMETRY FOR STRUCTURED PRUNING

A PREPRINT

Kaixiang Shu Independent Researcher 614729197@qq.com

## ABSTRACT

Structured pruning is commonly formulated as ranking individual channels, although channel responses can be complementary or cancel through downstream mixing. Motivated by these response interactions, we formulate pruning as the selection of a subset with large joint response capacity, followed by a separate functional realization step. Our unified response geometry maps each candidate set to $\dot { M } ( D , \bar { R } ) = D ^ { 1 / 2 } R D ^ { 1 / 2 }$ and uses its determinant together with Schur-greedy residuals to select non-redundant coordinates. The same construction yields two information-conditioned instances: an unlabeled instance based on activation covariance, and a task-conditioned instance that combines activation and gradient variance for response scale with gradient correlation for complementarity. To convert the selected subset into an executable network, we fold predictable removed responses into successor weights through ridge compensation and recalibrate batch-normalization statistics, without fine-tuning the network. On ImageNet ResNet-50, the unlabeled instance reaches 65.4% and 53.9% Top-1 accuracy at 30% and 40% deletion, versus 59.8% and 43.1% for strength-only selection; the task-conditioned instance reaches 67.7% and 56.3% under the same protocol. A six-family screen shows architecture-dependent behavior, with positive relative contrasts in several convolutional and expansion-layer settings and clear boundary cases in windowed attention. These results support response geometry as a conditional principle for structured pruning, with its benefit determined jointly by the observed response and the architecture in which that response is realized.

## 1 Introduction

Structured pruning seeks to remove channels, filters or expansion coordinates so that a pretrained network becomes cheaper to execute while retaining its predictive function. A common formulation turns this problem into a ranking of individual coordinates: a scalar score is computed from weights, activations or task sensitivities, and the lowest-scoring coordinates are removed [1, 2, 3, 4, 5, 6, 7, 8, 2]. This formulation is attractive because it gives a simple and controllable pruning rule. It also makes the computational target explicit: the selected coordinates determine the width of a successor layer and therefore the cost of the resulting network. However, the object that is removed is a set of coordinates whose responses are jointly mixed by later layers. The adequacy of an individual score therefore depends on whether the value of a coordinate can be assessed independently of the coordinates retained with it.

Our earlier analysis of convolutional responses showed why this independence assumption can be restrictive. Under downstream linear mixing, channel responses may contain shared components, complementary directions and components that cancel in the readout [3]. A channel with a modest marginal amplitude can provide a direction that is difficult to reconstruct from the other retained channels. Conversely, a channel with a large amplitude can be largely redundant with the current set, or its contribution can be offset by another response after mixing. These cases have the same failure mode for pruning: the value of a coordinate is determined by the response space formed with its companions, rather than by its marginal strength alone. The relevant question is consequently not which individual channels are strongest, but which subset preserves the largest amount of non-redundant response capacity at a prescribed width.

Existing criteria expose different parts of this problem. Magnitude and activation statistics describe individual response scale [1, 4, 5, 6, 7, 8]; gradient-based criteria measure sensitivity to a task loss [2, 11]; and geometric or diversity-based methods attempt to avoid selecting similar coordinates [12, 1, 14, 15]. These approaches are useful for their respective information regimes, but they generally leave several modeling choices implicit or treat them as separate heuristics:

what response is observed, how its scale is assigned, how relations between responses are measured, and how a selected subset is converted into a functioning network. The unresolved problem is therefore to define a subset-level pruning interface that can express both response scale and response relations, accept either unlabeled or task-conditioned observations, and remain separate from the subsequent functional realization of the pruned network.

We address this problem by formulating structured pruning as the selection of a subset with large joint response capacity. For a diagonal scale matrix D and a relationship matrix ${ \check { R } } .$ , we use the unified response geometry

$$
M ( D , R ) = D ^ { 1 / 2 } R D ^ { 1 / 2 } .
$$

The diagonal term records the scale of each response direction, while the off-diagonal structure records how response directions overlap or complement one another. For a candidate subset, the determinant of its principal matrix measures the volume of the response space jointly retained by that subset. A Schur residual then measures the additional response capacity supplied by a candidate after conditioning on the coordinates already selected. The resulting greedy ordering is therefore defined by a set objective rather than by independent scores, providing a common geometric criterion for deciding which coordinates are worth retaining before the network is edited.

The same interface yields two information-conditioned instances without changing the selection principle. The unlabeled instance obtains both scale and relationships from activation responses, making the criterion applicable when pruning data have no labels. The task-conditioned instance uses activation and gradient variance to define response scale and gradient correlation to describe complementarity relative to the current task. Labels therefore change the response model supplied to the geometry, rather than creating a second pruning principle: the subset objective, determinant/Schur-greedy ordering and realization pipeline remain shared. This separation lets the experiments compare information regimes directly while keeping the combinatorial selection problem fixed.

Selection alone does not specify how the edited network should realize the retained subset. Removing coordinates changes the input seen by the successor layer and can leave normalization statistics calibrated for the unpruned responses. We consequently treat functional realization as a separate step. Predictable components of removed activations are folded into successor weights with a ridge projection, and batch-normalization statistics are recalibrated when such statistics are present. The same realization procedure is applied to every selector, without fine-tuning the backbone or classifier. In this way, the response geometry defines what is retained, while the recovery operator tests whether that geometric choice can be implemented without conflating subset quality with downstream adaptation.

We evaluate the framework in a controlled ImageNet ResNet-50 study, where convolutional channel responses provide the setting directly motivated by the response analysis. In the primary confirmation, the unlabeled instance reaches 65.4% and 53.9% Top-1 accuracy at 30% and 40% deletion, compared with 59.8% and 43.1% for the matched strength-only rule; the task-conditioned instance reaches 67.7% and 56.3% under the same protocol. The experiments also compare against classical pruning baselines, isolate the contribution of functional realization, and examine the relationship between response-geometry descriptors and pruning contrasts. We then screen six architecture families to distinguish transfer cases from boundary cases: expansion coordinates in ViT provide a related linear-mixing setting, whereas windowed attention in Swin-T changes the meaning of a local response coordinate. Together, these studies test one unified claim at two levels: response geometry can improve subset selection when the observed responses contain exploitable complementarity, and the size and direction of the resulting contrast depend on how that response is realized by the architecture.

## 2 Related Work

## 2.1 Importance- and task-aware pruning

The dominant formulation of structured pruning assigns an importance score to each removable coordinate and then selects the coordinates with the largest scores under a width or resource budget. Early saliency methods estimated the change in the objective caused by removing parameters [1, 2, 3, 16]. For convolutional networks, Network Slimming uses batch-normalization scale parameters to induce channel sparsity and then removes channels with small learned scales [4, 8, 6]. Network Trimming instead uses data-dependent neuron responses to identify coordinates that can be removed from a trained model [5, 7, 17, 18, 15]. These approaches established the practical value of structured removal: pruning whole channels or filters produces an actual reduction in layer dimensions and can be implemented by standard dense kernels.

Importance scores have since been refined in several directions. Layer-adaptive magnitude methods allocate a global sparsity budget across layers instead of applying the same local threshold everywhere [19]. Other criteria estimate the sensitivity of a channel or connection using a first-order Taylor approximation, gradient information, or a one-shot loss perturbation [2, 11, 20]. Activation-weighted criteria extend the same idea to data-dependent responses, including settings where the original training objective is not revisited [21]. These methods are complementary in the information they use, but their common decision variable is still a marginal score for one coordinate. The score can express how large, active or task-sensitive a coordinate is in isolation; it does not by itself specify how the coordinate interacts with the other coordinates that will remain after pruning.

Task-aware pruning is especially useful when labeled examples are available because the score can reflect the task’s current decision surface rather than only the magnitude of the internal response. Gradient-based Taylor criteria estimate the first-order change in loss after removing a filter or channel [2], while SNIP and related single-shot approaches use connection sensitivity before training or fine-tuning [11]. Automated policies further learn layer-wise or platform-aware structural decisions, as in MetaPruning, NetAdapt and AMC [22, 23, 24]. Unlabeled or activation-based criteria provide a different operating regime: they estimate response statistics from inputs without requiring labels or loss gradients, which is useful for post-training compression and settings where labels are unavailable. In both regimes, however, the information source is commonly used to produce an independent scalar score. A task-sensitive response can therefore receive a high score even when its task effect is already represented by the selected set.

Our framework keeps the scale information represented by these criteria while changing the selection object. The unlabeled instance uses activation covariance to represent response scale and activation relationships, whereas the taskconditioned instance combines activation and gradient variance for scale with gradient correlation for task-conditioned complementarity. Both instances then use the same subset objective and greedy selection rule. The comparison is therefore between information-conditioned response models supplied to one geometric interface, rather than between unrelated scoring heuristics.

Post-training and training-free pruning provide a complementary context for this distinction. SynFlow and GraSP preserve signal flow or gradient flow without access to training examples [25, 26], while lottery-ticket and movementbased approaches study sparse subnetworks through initialization or fine-tuning dynamics [27, 28]. Recent transformer work extends structural pruning to attention and feed-forward blocks, including compact encoder models and one-shot large-language-model compression [29, 30, 31]. These methods demonstrate that the available response and adaptation regime strongly affect pruning behavior. Our setting is narrower and more controlled: we study post-training structured coordinate selection with a fixed realization operator, so that the effect of a response-set objective can be separated from training dynamics or extensive retraining.

## 2.2 Geometric, diversity, and response-set selection

Several pruning and compression methods move beyond independent scores by discouraging the selection of similar filters or neurons. Geometric-median pruning removes filters that are close to other filters in a feature space, with the intuition that a nearby filter can provide a substitute for the one removed [12]. High-rank feature-map criteria and neuron-importance propagation also use statistics of feature responses or their propagated effects to guide structured removal [15, 17]. Diversity-based compression uses determinantal point processes to favor sets that cover different response directions rather than repeatedly selecting similar elements [1, 14]. These approaches are important precedents for treating the retained set, rather than only its individual members, as an object of optimization.

The motivation for modeling relationships between responses also comes from analyses of downstream mixing in convolutional networks. Our earlier response analysis showed that channel activations can contain shared, complementary and cancelling components when they are combined by later linear operators [3]. This observation is compatible with, but not reducible to, a filter-distance criterion: two channels can be close in one representation and still have different effects after the downstream readout, while responses with modest marginal amplitude can provide a direction that is difficult to reconstruct from the current set.

Our response geometry is related to this line of work through its determinant-based set objective, but it makes a different factorization explicit. The matrix $M ( D , R ) = D ^ { 1 / \bar { 2 } } R D ^ { 1 / 2 }$ separates the scale of each response from the relationship structure measured across responses. Its principal determinant combines individual response capacity with the complementarity of the selected directions, and its Schur residual gives the incremental capacity of a candidate conditioned on the current set. This provides a fixed-width greedy ordering that can be reused with different response sources. A geometric distance or a diversity kernel can discourage redundancy, but it does not by itself determine which response supplies the scale, how task information should enter, or how the selected subset should be realized in the edited network. The present work turns the response-interaction observation into a pruning interface whose selection principle can be tested through a common functional realization pipeline.

## 2.3 Structural surgery and functional recovery

A ranking rule is not by itself a deployable pruned model. Removing a channel changes the input dimension of the successor layer, and the original successor weights and normalization statistics were calibrated for the unpruned response distribution. Structured pruning systems therefore commonly include a reconstruction, normalization update or subsequent adaptation stage after the structural edit [5, 2, 8, 18, 23, 24]. These stages are necessary in practice, but they can also obscure the contribution of the selection criterion if different selectors receive different recovery procedures or if extensive fine-tuning compensates for a weak ranking.

We use functional recovery as a controlled realization step shared by all selectors. A ridge projection predicts the removable response components from the retained ones and folds that prediction into the successor weights; batch-normalization statistics are then recalibrated when the architecture uses running normalization. The recovery operator does not change the selected order and does not fine-tune the backbone or classifier. This separation lets the experiments ask two different questions: which subset has the desired response geometry, and how well can that subset be implemented by the common recovery pipeline? Response geometry defines the subset-level capacity to preserve, while recovery supplies the common realization interface.

## 2.4 Position of this work

The contribution of this paper is a unified response-geometry formulation that connects these lines of work at the level of the retained subset. Relative to marginal importance methods, it adds an explicit representation of cross-coordinate response relations. Relative to geometric and diversity-based selection, it makes the scale–relationship factorization and the response source explicit, and it uses Schur residuals to construct a fixed-width order. Relative to task-aware and unlabeled criteria, it keeps the same selection and realization interface while changing only the observed response model. Relative to reconstruction-based pruning pipelines, it separates the subset objective from the functional recovery applied after structural surgery.

The resulting framework has two final instances in the paper: an unlabeled activation-response instance and a taskconditioned activation/gradient instance. They are not separate principles or historical implementation variants. They are two information-conditioned realizations of one determinant/Schur subset objective, evaluated with one common functional recovery protocol. This organization allows the experiments to compare information regimes directly, trace the response-geometry effect on ResNet-50, and identify where the same principle transfers to expansion coordinates or reaches a boundary in windowed attention.

## 3 Method

Logic of this section. The method has one selection objective and one realization operator, as summarized in Figure 1. We first define the response geometry and its marginal gain, then instantiate the same construction with unlabeled or task-conditioned observations, and finally map the selected coordinates back into a valid network. This ordering makes clear which quantities define a subset and which quantities only restore the network after selection.

## 3.1 Pruning objective and response geometry

Consider a pretrained network and a target layer with $C$ removable coordinates. For a prescribed deletion ratio r, the selector retains $k = \lfloor ( 1 - r ) C \rfloor$ coordinates. Let $S \subseteq \{ 1 , \dots , C \}$ denote a candidate retained set. The selection objective is local: it asks whether the responses indexed by S span a large and non-redundant response space before the network is edited. It is therefore a criterion for constructing a subset, not a direct theorem about the final task loss.

Let $O \in \mathbb { R } ^ { N \times C }$ contain centered observations of one response type, with one row per coordinate observation. For a convolutional activation, spatial positions are included as separate rows. The population covariance and its correlation normalization are

$$
\Sigma _ { O } = \frac { 1 } { N } O ^ { \top } O , \qquad R _ { O } = \mathrm { d i a g } ( \Sigma _ { O } ) ^ { - 1 / 2 } \Sigma _ { O } \mathrm { d i a g } ( \Sigma _ { O } ) ^ { - 1 / 2 } ,
$$

with an explicit zero-variance convention for degenerate directions. Let $D = \mathrm { d i a g } ( d _ { 1 } , \dots , d _ { C } )$ contain the response scale. The unified response-geometry matrix is

$$
M ( D , R ) = D ^ { 1 / 2 } R D ^ { 1 / 2 } .\tag{1}
$$

The diagonal of M records individual response strength, whereas its off-diagonal entries record relations between response directions. The primary instances use the full relationship matrix; diagonal and interpolated matrices are retained only as prespecified controls in the supplement.

## Unified Response Geometry for Structured Pruning

![](images/5379d3859b5929d829497d9e1b7e400bffc0bae7d03ef92c394568d1d05537eb.jpg)  
Figure 1: Overview of Unified Response Geometry for Structured Pruning. Downstream responses can be shared, complementary, or cancelling, so marginal importance does not determine the value of a retained set. The unified matrix combines response scale D and relationship structure $R ;$ determinant-based subset capacity and Schur residuals define the greedy selection. Activation-only and activation–gradient observations instantiate the same objective, after which ridge compensation and BN recovery produce an executable pruned network without backbone or classifier fine-tuning.

For a positive semidefinite M, the combination capacity of S is its squared response volume, with $V ( \emptyset ) = 1$

$$
V ( { \cal S } ) = \operatorname * { d e t } ( M _ { \cal S } { \ o { s } } ) .\tag{2}
$$

For $j \not \in S ,$ the Schur identity gives

$$
\frac { V ( S \cup \{ j \} ) } { V ( S ) } = M _ { j j } - M _ { j S } M _ { S S } ^ { - 1 } M _ { S j } .\tag{3}
$$

The right-hand side is the response variance of $\dot { \mathbf { \zeta } } _ { j }$ that cannot be represented by the selected coordinates. We therefore build an ordering by repeatedly selecting the largest non-negative Schur residual and retaining its first k entries. If numerical rank is exhausted, remaining coordinates are appended in deterministic index order and recorded. For a fixed subset with positive diagonal strengths,

$$
\log V ( S ) = \sum _ { i \in S } \log d _ { i } + \log \operatorname* { d e t } ( R _ { S S } ) ,
$$

which makes the scale–complementarity decomposition explicit. The determinant is a selection objective; it is not interpreted as a finite-deletion output-error bound.

## 3.2 The two response instances

The two proposed rules use identical estimation, ordering, and realization code. They differ in the response used to assign scale and measure complementarity. Let $D _ { x } = \operatorname { d i a g } ( \operatorname { C o v } ( x _ { 1 } ) , \dots , \operatorname { C o v } ( x _ { C } ) )$ . For the unlabeled instance, the observed response is the activation and

$$
M _ { u } = D _ { x } ^ { 1 / 2 } \operatorname { C o r r } ( x ) D _ { x } ^ { 1 / 2 } = \operatorname { C o v } ( x ) .
$$

For the task-conditioned instance, $g _ { i }$ is the derivative of the current loss with respect to coordinate i. We use the task-conditioned scale $D _ { x g } = \operatorname { d i a g } ( \operatorname { C o v } ( x _ { 1 } ) \operatorname { C o v } ( g _ { 1 } ) , \dots , \operatorname { C o v } ( x _ { C } ) \operatorname { C o v } ( g _ { C } ) )$ and set

$$
M _ { s } = D _ { x g } ^ { 1 / 2 } \mathrm { C o r r } ( g ) D _ { x g } ^ { 1 / 2 } .
$$

Algorithm 1: Unified response-geometry structured pruning   
Input: target layer ℓ with C removable coordinates; deletion ratio r; mode   
m ∈ {unlabeled, task − conditioned}; strength bank ${ \mathcal { C } } _ { s } ;$ relationship bank $\mathcal { C } _ { r } \mathrm { : }$ ; adaptation set A   
Output: selected set S and a functionally recovered successor layer   
1 if m = unlabeled then   
2 collect x; set $d _ { i } \gets \mathrm { C o v } ( x _ { i } )$ and $R  \operatorname { C o r r } ( x )$   
3 else   
4 collect x and task gradients g; set $d _ { i } \gets \mathrm { C o v } ( x _ { i } ) \mathrm { C o v } ( g _ { i } )$ and $R  \operatorname { C o r r } ( g )$   
5   
6 Estimate the moments from $\mathcal { C } _ { s }$ and $\mathcal { C } _ { r }$ , form $D  \deg ( d )$ , and set $M \gets D ^ { 1 / 2 } R D ^ { 1 / 2 }$   
7 S ← ∅ // selected coordinates   
8 for t ← 1 to $k = \lfloor ( 1 - r ) C \rfloor$ do   
9 if $S = \emptyset$ then   
10 compute $\delta _ { i } \gets M _ { i i }$ for every $i \not \in S$   
11 else   
12 compute $\delta _ { i } \gets M _ { i i } - M _ { i S } M _ { S S } ^ { - 1 } M _ { S i }$ for every $i \not \in S$   
13   
14 $j \gets \arg \operatorname* { m a x } _ { i \notin S } \delta _ { i } ; S \gets S \cup \{ j \}$   
15 Fit ridge predictors from retained to removed activations on A   
16 Fold the predictors into successor weights, reset BN statistics, and recompute them on A   
17 Evaluate the recovered network and report accuracy, teacher KL, and prediction agreement

Thus the task-conditioned instance uses activation variance for physical amplitude, gradient variance for task sensitivity, and gradient correlation for task-conditioned complementarity. Labels change the observed response and its scale, while the subset size, greedy rule and recovery pipeline remain fixed. The displayed equalities are population identities. In the finite-sample implementation, the diagonal scale and relationship matrix are estimated from the disjoint banks described below, so their product is a split-sample estimate and need not equal a single-bank empirical covariance entry by entry.

The calibration estimator is deliberately separated from functional adaptation. We use an independent strength bank $\mathcal { C } _ { s }$ for $D _ { o }$ and relationship bank $\mathcal { C } _ { r }$ for $R _ { o } ;$ in the confirmation protocol these contain 2,048 and 4,096 images, respectively. Activations and task responses are collected in evaluation mode with a fixed cross-entropy convention. Counts, means, and centered second moments are accumulated across batches before covariance and correlation normalization. This gives one fixed response matrix per target layer and avoids changing the selector when the adaptation set is changed.

The methodological contribution is the separation of four choices that are often conflated in a pruning score: what response is observed, how individual scale is assigned, how redundancy is measured, and how the selected coordinates are realized in the successor layer. The determinant and Schur identities themselves are standard. The framework makes these choices explicit, keeps the selection and realization interface fixed across the two information regimes, and exposes the response factorization to controlled ablations.

## 3.3 Functional realization and recovery

Selection changes the coordinates consumed by the successor layer, so a coordinate order is not yet a deployable pruned network. Let $A _ { S }$ and $A _ { R }$ be the retained and removed activation coordinates collected on an independent adaptation set. We fit a zero-intercept ridge projection

$$
\boldsymbol { B } = \left( \widehat { \mathbb { E } } [ A _ { S } ^ { \top } A _ { S } ] + \rho \boldsymbol { I } \right) ^ { - 1 } \widehat { \mathbb { E } } [ A _ { S } ^ { \top } A _ { R } ] ,\tag{4}
$$

where $\rho$ is proportional to the mean retained second moment. If the successor convolution has input weights $[ W _ { S } , W _ { R } ]$ substituting $A _ { R } \approx A _ { S } B$ yields

$$
\begin{array} { r } { W _ { \mathrm { k e e p } } ^ { \prime } = W _ { S } + W _ { R } B ^ { \top } . } \end{array}\tag{5}
$$

The projection is fitted once per target layer and is never used to alter the selected order. Dependent BN parameters are sliced consistently; residual additions retain their original output width, so the operation removes only the specified input coordinates.

After folding, BN running means and variances are reset and recomputed with cumulative statistics on the adaptation images. We report the compensated network before this step as the raw state and the same network after recalibration as the BN state. Every selector, including L2 and strength-only baselines, receives the same adaptation images, ridge coefficient, BN procedure, target widths, and evaluation images. Thus the method comparison isolates the response-geometry selection rule, while the raw-to-BN transition exposes how functional recovery contributes to the final result.

## 3.4 Assumptions, implementation, and scope

The framework uses response magnitude as an individual scale and correlation as a measure of complementary response directions. The task-conditioned instance adds gradient variance and gradient correlation to describe task sensitivity. These choices are examined with controlled coefficients, numerical checks and the cross-architecture mechanism experiment.

The covariance accumulator costs $O ( N C ^ { 2 } )$ time and $O ( C ^ { 2 } )$ memory for a layer; constructing a full greedy order costs $O ( C ^ { 3 } )$ with rank-one Schur updates. No classifier or backbone weights are trained, and all selection quantities are frozen before evaluation. The primary confirmation therefore tests response geometry under a fixed compensation and BN-recovery pipeline.

## 4 Experiments

The experiments test one claim in progressively broader settings: response geometry should change structured selection when the pruning and recovery pipeline is held fixed, while the magnitude and direction of the contrast should depend on the observation model and architecture. We begin with a controlled multi-seed confirmation on ResNet-50, isolate functional realization from subset selection, and then examine transfer and boundary cases across six architecture families. Unless stated otherwise, accuracies are measured after the common recovery step (BN recalibration where BN is present); teacher KL and prediction agreement are fidelity diagnostics.

## 4.1 Experimental protocol and controlled comparison

The primary confirmation uses one ImageNet-pretrained ResNet-50 checkpoint and 32 internal bottleneck targets, consisting of the paired conv1/conv2 layers in the four residual stages. At a target layer with C removable coordinates, every method retains $k = \lfloor ( 1 - r ) C \rfloor$ coordinates at deletion ratio r. The comparison includes L2 magnitude, a diagonal strength-only rule, and the two final response-geometry instances defined in Section 3.2. The broader six-family screen adds representative magnitude, activation-statistic, geometric-median and gradient-sensitivity baselines [4, 5, 12, 19, 2, 11, 21, 1, 2, 6, 7, 8, 18, 15, 22, 32, 23, 24]. This choice makes the main contrast explicit: L2 and strength-only ignore cross-coordinate response geometry, whereas the two final instances apply one determinant/Schur interface to activation or task-conditioned responses.

The calibration data are partitioned by role. The strength bank contains 2,048 images and determines the diagonal scale; the relationship bank contains 4,096 images and determines the correlation matrix. An independent 3,200-image adaptation bank is used only for ridge compensation and BN recalibration, and a disjoint 2,048-image confirmation bank is used only for evaluation. We repeat the complete calibration and candidate-freezing procedure for three seeds. The checkpoint, target graph and confirmation images are fixed across seeds, so the reported standard deviations quantify calibration variation rather than an additional source of image-level uncertainty.

All selectors are evaluated under the same functional realization. Removed activations are predicted from retained activations with the zero-intercept ridge model, the predictor is folded into the successor convolution, dependent normalization parameters are sliced, and running BN statistics are recomputed on the adaptation bank when BN is present. No classifier or backbone parameter is fine-tuned. This control is essential because a ranking method and a deployable pruned network are different objects: changing the downstream realization can dominate the apparent effect of the ranking. We therefore report the recovered BN endpoint as the primary result and retain raw and teacher-fidelity measurements as diagnostics.

The primary endpoint is ImageNet Top-1 accuracy on the fixed confirmation bank. Teacher KL, centered-logit MSE and prediction agreement are recorded for the same predictions. The seed-level unit of analysis is a paired difference between methods evaluated with the same target structure and confirmation images. This pairing prevents a favorable split or a different recovery state from being mistaken for a selection difference. The protocol and all candidate hashes are frozen before confirmation; the complete moment-estimation and audit details are given in the supplement.

Table 1: Primary BN-recovered Top-1 accuracy $( \mathrm { m e a n } \pm \mathrm { s . d . }$ . over three calibration seeds).
<table><tr><td>Method</td><td>30%</td><td>40%</td></tr><tr><td>L2</td><td> $4 7 . 6 9 \pm 1 . 4 9$ </td><td> $3 1 . 2 5 \pm 0 . 6 9$ </td></tr><tr><td>Strength-only</td><td> $5 9 . 8 1 \pm 2 . 1 7$ </td><td> $4 3 . 1 3 \pm 0 . 3 2$ </td></tr><tr><td>Unlabeled instance</td><td> $6 5 . 4 3 \pm 0 . 2 0$ </td><td> $5 3 . 9 2 \pm 0 . 4 8$ </td></tr><tr><td>Task-conditioned instance</td><td> $6 7 . 6 6 \pm 2 . 1 2$ </td><td> $5 6 . 2 5 \pm 2 . 6 1$ </td></tr></table>

Table 2: Pre-specified paired Top-1 differences across calibration seeds.
<table><tr><td>Comparison</td><td>Deletion</td><td>Mean (pp)</td><td>Positive seeds</td></tr><tr><td>Unlabeled instance — strength-only</td><td>30%</td><td>+5.62</td><td>3/3</td></tr><tr><td>Task-conditioned instance — strength-only</td><td>30%</td><td>+7.85</td><td>3/3</td></tr><tr><td>Unlabeled instance — strength-only</td><td>40%</td><td>+10.79</td><td>3/3</td></tr><tr><td>Task-conditioned instance — strength-only</td><td>40%</td><td>+13.12</td><td>3/3</td></tr></table>

## 4.2 Primary response-geometry result on ResNet-50

Table 1 answers the first question: does adding response geometry to a diagonal strength score change structured-pruning accuracy under the common realization pipeline? At 30% and 40% deletion, the unlabeled instance reaches $6 5 . 4 \bar { 3 } \pm 0 . 2 \bar { 0 }$ and $5 3 . { \dot { 9 } } 2 \pm 0 . 4 8$ Top-1, versus 59.81 ± 2.17 and 43.13 ± 0.32 for strength-only selection. The paired contrasts are $+ 5 . 6 2 \ \mathrm { a n d } + 1 0 . 7 9 $ percentage points, respectively, and all three calibration seeds favor the response-geometry rule. The contrast is larger at the stronger deletion rate, where selecting complementary responses becomes more consequential than preserving only the largest individual variances.

The task-conditioned instance yields the same ordering relative to the diagonal comparator. It reaches $6 7 . 6 6 \pm 2 . 1 2$ and $5 6 . 2 5 \pm 2 . 6 1$ Top-1 at 30% and 40% deletion, corresponding to paired contrasts of +7.85 and +13.12 points; all three seeds favor the task-conditioned rule. Its scale contains both activation and gradient variance, while its relationship term is the gradient correlation. This is the task-conditioned response model used in the confirmation protocol, so the result measures the combined effect of task sensitivity and task-conditioned complementarity rather than the effect of labels alone.

The two information regimes show the same ordering pattern. Activation geometry supplies information beyond a diagonal strength score when labels are unavailable. The task-conditioned instance retains the same volume principle while adding a task response, yielding higher Top-1 accuracy than the diagonal control under the fixed realization pipeline. The comparison is between the two final instances and matched controls; no internal implementation history is treated as a separate method.

## 4.3 Selection versus functional realization

The second question is whether the response-volume ranking remains useful after the selected coordinates are converted into a valid network. Figure 3 and Table 3 trace the same candidate sets through direct slicing, ridge compensation and BN recalibration. Direct slicing removes coordinates from the successor input while leaving the successor weights and running statistics calibrated for the original coordinates. It is a diagnostic of an incomplete realization, so the flow separates structural editing from the recovery process.

Ridge compensation restores the contribution of removed responses through the retained coordinates. For each target layer, the predictor is fit on the independent adaptation bank and folded into the successor convolution without changing the selected order. The compensated state recovers most of the direct-slice loss across the representative methods in the flow diagnostic. Because the predictor and its coefficient are shared across selectors, this recovery measures the value of a common realization operator rather than a method-specific retraining advantage.

BN recalibration supplies a further, distinct correction. Removing coordinates and folding predictors changes the activation distribution seen by downstream normalization, so the running mean and variance from the unpruned network are no longer appropriate. Recomputing them on the same adaptation bank changes the final endpoint and teacher fidelity. The unpruned control passed through the identical BN procedure quantifies the drift introduced by recalibration itself, preventing the BN increment from being attributed entirely to pruning.

The flow experiment thus supports a separation of roles. The determinant and Schur rule defines the subset geometry, ridge compensation makes that subset executable in the successor layer, and normalization recovery restores the operating statistics where the architecture uses running normalization. The three-seed confirmation table remains the primary selection result; the flow experiment evaluates the volume criterion after realization and exposes how each recovery stage contributes to the final endpoint.

![](images/c2eef19635e116e2c4a1c06fd9041987daab38c04985d6699a6c7f911d8a6802.jpg)  
Figure 2: Calibration-seed contrasts for the two final response-geometry instances. Each point is a paired Top-1 difference from strength-only selection on the same confirmation images. Both instances are positive for all three seeds at both deletion rates.

Table 3: Selection versus functional realization in the representative flow diagnostic.
<table><tr><td>Method</td><td>Deletion</td><td>Direct</td><td>Compensated</td><td>+BN</td><td>Change to +BN (pp)</td></tr><tr><td>L2</td><td>30%</td><td>0.6</td><td>39.6</td><td>49.6</td><td>+49.0</td></tr><tr><td>L2</td><td>40%</td><td>0.2</td><td>16.2</td><td>31.6</td><td>+31.3</td></tr><tr><td>Strength-only</td><td>30%</td><td>0.0</td><td>52.1</td><td>59.6</td><td>+59.6</td></tr><tr><td>Strength-only</td><td>40%</td><td>0.2</td><td>25.3</td><td>44.3</td><td>+44.1</td></tr><tr><td>Unlabeled instance</td><td>30%</td><td>0.6</td><td>55.0</td><td>64.1</td><td>+63.5</td></tr><tr><td>Unlabeled instance</td><td>40%</td><td>0.2</td><td>27.9</td><td>50.2</td><td>+50.0</td></tr><tr><td>Task-conditioned instance</td><td>30%</td><td>11.1</td><td>65.3</td><td>68.7</td><td>+57.5</td></tr><tr><td>Task-conditioned instance</td><td>40%</td><td>0.9</td><td>48.2</td><td>56.2</td><td>+55.4</td></tr></table>

## 4.4 Cross-architecture behavior and mechanism boundary

The final question is how far the response-geometry construction transfers beyond the theory-native CNN setting. We screen six distinct families—ResNet-50, ConvNeXt-B, EfficientNet-B3, MobileNetV2, ViT-S/16 and Swin-T [33, 34, 35, 36, 37, 38]—using the same selector, structural surgery and normalization-recovery pipeline, with architecturespecific dependency handling where needed. Each model contributes 18 selection entries at five deletion rates, giving 540 frozen rows. The release uses the final unlabeled and task-conditioned instances across all six models, with the task-conditioned rows regenerated from the final $D _ { x g }$ implementation. It tests transfer and maps the operating range; the multi-seed ResNet-50 experiment supplies the replicated formula-level confirmation.

At 40% deletion, the unlabeled-minus-L2 Top-1 contrasts are 15.2, 33.1, 4.3, −2.7, 24.6 and −10.7 points in the model order above. The corresponding task-conditioned-minus-Fisher contrasts are 8.3, −2.0, 18.8, 9.8, 14.0 and 4.7 points. The unlabeled instance has the highest reported accuracy in the ResNet-50 and ViT-S/16 cases and at the stronger EfficientNet-B3 budgets, whereas APoZ [5], LAMP [19] or magnitude rules are higher in parts of ConvNeXt-B, MobileNetV2 and Swin-T. The task-conditioned instance has the highest reported accuracy in several ResNet-50, EfficientNet-B3 and MobileNetV2 budgets, while Taylor/Fisher-gate [2], OBD [1] or Wanda [21] are higher in parts of ConvNeXt-B, ViT-S/16 and Swin-T. The comparison therefore shows architecture-dependent performance rather than uniform dominance.

The architecture pattern follows the scope of the Project1 motivation. Project1 was derived for CNN channel responses with spatially repeated observations, downstream linear mixing and shared cancellation structure; CNN bottleneck and expansion layers therefore form the theory-native domain of the construction. ViT-S/16 provides a transfer case

## Selection and recovery contribute distinct parts of the final pruning outcome

![](images/9004cca164db24d7d7b72f462a02fe7f68e23c413bc04e10b384d9d3c91a6577.jpg)

![](images/c12044993e7bac988134f56eb9bc74904e462b50d450dcf00fa2c00e0cac756b.jpg)

![](images/76f75e7a407e96cb55bb0fc763e619b63936bcf00480bbc2c705fbe2467e0f9a.jpg)

![](images/39a894017a3b681ec9a6f04cdb6a75e0ad49057d2a5d4a98855097dbab06a334.jpg)  
Figure 3: Selection and functional recovery are separate stages in a representative flow diagnostic. Panel a follows direct slicing, ridge compensation and BN recalibration; panels b and c show the corresponding accuracy and teacher-KL increments; panel d reports the final BN endpoint. The same recovery procedure is applied to every selector, so the figure diagnoses the pipeline rather than replacing the multi-seed selection comparison.

Table 4: Six-family structural-pruning screening. Entries are Top-1 accuracy (%). “Best U/L” is the best method within the unlabeled or label/gradient comparator family at the same deletion rate; the two final response-geometry instances are shown separately so architecture boundaries remain visible.
<table><tr><td>Model</td><td>Unlabeled</td><td>best U</td><td>30% Task-cond.</td><td>best L</td><td>Unlabeled</td><td>best U</td><td>40% Task-cond.</td><td>best L</td><td>Unlabeled</td><td>best U</td><td>50% Task-cond.</td><td></td></tr><tr><td>ResNet-50</td><td>40.9 71.7</td><td>40.9 (Unlabeled instance)</td><td>50.8</td><td>50.8 (Task-conditioned instance)</td><td>18.8</td><td>18.8 (Unlabeled instance)</td><td>27.9</td><td>27.9 (Task-conditioned instance)</td><td>5.4</td><td>5.4 (Unlabeled instance)</td><td>9.9</td><td>9.9 (Task-conditioned instance)</td></tr><tr><td>ConvNeXt-B EfficientNet-B3</td><td>39.7</td><td>72.5 (APoZ) 41.0 (FPGM)</td><td>65.7 61.0</td><td>72.5 (Taylor-gate) 61.0 (Task-conditioned instance)</td><td>47.1 19.4</td><td>47.2 (APoZ) 19.4 (Unlabeled instance)</td><td>49.7 40.0</td><td>56.0 (Fisher-gate) 40.0 (Task-conditioned instance)</td><td>3.0 5.2</td><td>3.8 (APoZ)</td><td>15.2 19.9</td><td>24.0 (OBD)</td></tr><tr><td>MobileNetV2</td><td>15.8</td><td>23.5 (L1)</td><td>42.6</td><td>42.6 (Task-conditioned instance)</td><td>4.7</td><td>7.4 (LAMP)</td><td>16.2</td><td>16.2 (Task-conditioned instance)</td><td>1.2</td><td>5.2 (Unlabeled instance) 1.3 (LAMP)</td><td>4.7</td><td>19.9 (Task-conditioned instance) 4.7 (Task-conditioned instance)</td></tr><tr><td>ViT-S/16</td><td>52.5</td><td>52.5 (Unlabeled instance)</td><td>54.0</td><td>54.0 (Task-conditioned instance)</td><td>31.3</td><td>31.3 (Unlabeled instance)</td><td>36.3</td><td>42.5 (Fisher-gate)</td><td>9.8</td><td>9.8 (Unlabeled instance)</td><td>15.8</td><td>21.3 (Fisher-gate)</td></tr><tr><td>Swin-T</td><td>59.9</td><td>65.2 (LAMP)</td><td>60.0</td><td>65.0 (Wanda)</td><td>44.5</td><td>55.2 (LAMP)</td><td>43.0</td><td>50.6 (Wanda)</td><td>21.2</td><td>33.7 (LAMP)</td><td>22.0</td><td>25.4 (Wanda)</td></tr></table>

because its MLP hidden units are also linear expansion coordinates, and the unlabeled instance is competitive there, but the Project1 GAP and pixel-space argument does not directly cover ViT. Swin-T is a boundary case: windowed token mixing, LayerNorm and residual paths change the functional meaning of a local response coordinate, and LAMP/Wanda have higher accuracy in the reported budgets. This gives an architecture-conditioned interpretation: CNN is the native domain, ViT is an empirically transferable extension, and Swin-T defines a boundary for the present response model.

We next test that interpretation directly. Before pruning, we measure shared low-rank energy, a fixed downstreamreadout cancellation residual and normalized effective covariance rank for each model, aggregate them over prunable layers, and relate every descriptor to matched unlabeled–L2 and task-conditioned–Fisher contrasts. The cancellation residual has the clearest directional association with the unlabeled contrast at 40% deletion $( \rho = 0 . 8 9$ , exact permutation $p = 0 . 0 3 5 , n = 6 )$ . Low-rank energy is not monotone, and task-conditioned associations are mixed; the complete descriptor matrix and all 24 correlations are provided in the supplement. Thus the mechanism experiment supports a measurable cancellation component while rejecting a single-scalar explanation of all architecture behavior. This test i motivated by the superposition and cancellation observations in Project1 [3].

Together, the confirmation and screen show that response geometry changes selection outcomes under the fixed realization protocol used for each study. The replicated ResNet-50 confirmation establishes the main result, while the fixed-budget screen maps its transfer range and architecture-dependent contrasts.

Per-architecture contrasts expose boundaries

L2 Unlabeled instance Fisher Task-conditioned instance

Geometry-driven pruning shows architecture-dependent contrasts Six-family structural-pruning benchmark · solid = unlabeled · dashed = task-conditioned · shaded bands = 30% and 40% operating points

![](images/4c7d3de9675a30c8920a4e0c5f125d4ace98bcf37758aecda2b365edbdb2ae71.jpg)

![](images/5ffbdfeb400e879a2f295252562e553daabde1d96458dccdbf0a88e35c756787.jpg)

![](images/3005315f8b1ade7aa5e4387cfcf1fec89d8a658980d7d2ed1f0ad84619a7d813.jpg)

![](images/13053c688e71c684b4e8ff6c41697cacea8b5a8c00e08023b75a016c1e5eff9c.jpg)

![](images/f85541237d890022a84d7a8b44c60097950f00726d262300f8c3e44582b45fa5.jpg)

![](images/37ca67e891701ddad32b1ae0944978ae8d4c836eab55c3b77a55726bffde3702.jpg)

![](images/dbc51992384bcb3270d520bf002f14f6c952d0d92e372fc387cc2b9574bcb746.jpg)

![](images/ee63f365d5c15837d1fbe89075b1f738b5802266a41540055f7321541dec3354.jpg)  
Figure 4: Six-family cross-architecture screening. Panel A summarizes the median accuracy trajectory and interquartile spread; panel B reports paired 40% contrasts against regime-matched comparators; panels C–H show each architecture. Negative contrasts remain visible to define the operating boundary.

## 5 Discussion

The central implication of the results is that structured pruning can be viewed as a response-modeling problem rather than only a ranking problem. The relevant question is which response directions remain jointly useful at the target width, not which coordinates have the largest marginal scores. In the ResNet-50 confirmation, both information-conditioned instances exceed the strength-only comparator under the same recovery procedure across all calibration seeds. For the unlabeled instance, activation relationships provide selection information beyond individual response variance, which is particularly useful when task feedback is unavailable. The task-conditioned instance applies the same principle to a response model that incorporates task sensitivity: its activation/gradient scale and gradient correlation allow task information to affect both individual strength and complementarity. The common construction therefore connects the two information regimes through one shared question: how much additional response capacity does a coordinate provide alongside those already retained?

The recovery experiment gives this geometric statement an operational meaning. The volume objective favors responses that supply distinct directions at the selected width, while the ridge projection uses retained activations to reproduce predictable components of the removed responses. These operations act on different aspects of compression: selection determines the available response coordinates, and compensation adjusts the successor to use them. BN recalibration then aligns running statistics with the edited network. The progression from direct slicing through compensation to recalibration in Figure 3 illustrates why final accuracy depends on the realization process as well as on the retained set. Keeping the recovery procedure fixed across selectors makes the comparison informative about the selection criterion within that pipeline. This separates the geometric hypothesis from its functional consequence: response capacity defines the property being retained, while recovered accuracy and teacher fidelity test how that property survives in the edited network.

Project1 alignment is measurable, but architecture effects are not one-dimensional Activation/gradient response descriptors versus matched pruning contrasts; each point is one active architecture.

![](images/6070fa342afdbe2c73d0cd817fb497f7fd456d7e97310719b584d9f197d25ebc.jpg)

![](images/8ba44a4fc5cf665218881c9f1c863882b8a8da259c6cab1174186c3f0d47db18.jpg)

![](images/0e3a7f078a3b7c83107a60236bd502ad5b761fcfa8d181920626f324ae03f520.jpg)

![](images/e6acd4b0fd701966af6bf09d18b9577bb579107f8ce82df9ba7cbef79d6a7006.jpg)  
Architecture-mechanism alignment  
Six models · n=6 per panel Dots are parameter-weighted layer medians.

![](images/5ddaf08714a55171b16099b9bc07dfc8982d9611b5e3a7cf7a0c385e08c510c8.jpg)

![](images/252348fd9274e647caecd40f19ef05be4aa1c9951fbc788c16bb13a6fb92ae71.jpg)

![](images/61473f0288f12087c2aaf42f88f37e240342847190a7b6d605d60413c6fabf68.jpg)

![](images/7d8c54d1c7d0fad58b642c5f840e5ca4b5ce98196c7d96c2795e23ece5c8f375.jpg)  
Reading the grid  
Every descriptor × contrast pair is shown. ρ is Spearman rank correlation; p is an exact permutation p-value The grid is descriptive at n=6, not a universal test

![](images/dbb6085677d85d03153a0ddbd5437f1790d2008882ac28b0eacbe330529a4743.jpg)

![](images/093e3c0906afd2b12152b1e813f4ab626c41b5acf5ef4259ba6e7ba7b127b76d.jpg)

![](images/437558dcf6ee6feacec1d8fc105394f07d73251db49351036e0fd46a76ff8e39.jpg)

![](images/34e7433e911b080fcb18b0eacb02a50373f8ff0dd3da00b63d2368074237c821.jpg)  
Observed signal

![](images/25696eba8ef83c3ef00667c78de88ffce98112962fb8575239d373aa7b347811.jpg)

![](images/8ddf345135a5687505dacb28251b85178f743fe96c346db5bf8ca3ff8a3effcf.jpg)

![](images/79d5508c4aaf272bbb2973551289b73500275e8194d60687235c4e625771626a.jpg)

![](images/bede7627b8d56bcad2325355d399f92e1c8510031c55bc3084906ca0b2efa0e6.jpg)  
Cancellation residual tracks Unlabeled contrast at 40% (ρ=+0.89, p=0.035). Other directions are mixed, s no composite alignment score is claimed.

![](images/9ad463e308680f47ef20c54fb87a61d79f3659d15ec5946f01389894def394f7.jpg)

![](images/c96772dcdc07844210c36bc1ef7035e24fe11e4d14ebeb8858a7ebc3dff8b434.jpg)

![](images/7e23638a674f3fa30478a1f5b98d040a398234b6b19cc799f81715c0c5af2514.jpg)

![](images/0e96104c089b7b19bec3ea2dbdb6416fef6457bd9a02dc8ff55c089ca21aea10.jpg)

![](images/fc657ad594053eb73029c3d03d2a420cba581083066bf57ea2cb0e0ca55f5ca8.jpg)

![](images/2b904970030332bf012a9ab5430c29598ac26a0ff77b7ddc3b69b10899a31c6a.jpg)

![](images/f0849446e53d832b78f596523548851105c062438185174521ed615fedb8c4d5.jpg)

![](images/6ba9e63d7edc4c842a04bf0585a760a80986de685a8a93bbe8b8ab7a0e48f64f.jpg)  
Figure 5: Architecture–mechanism alignment. Descriptor–contrast pairs are shown for all six architectures. The strongest directional association is between cancellation residual and the unlabeled contrast at 40%, while low-rank and task-conditioned relationships remain mixed.

The cross-architecture results locate this principle within the response structures that motivated it. Convolutional channels provide spatially repeated observations followed by linear mixing, which connects the CNN setting to the earlier analysis of superposition and cancellation. The competitive unlabeled results on ViT-S/16 extend the empirical evidence to MLP expansion coordinates with a linear successor. Swin-T, together with mixed results in some ConvNeXt B and MobileNetV2 budgets, shows that a shared architecture label alone does not determine the best selector. The coordinate being removed and the surrounding mixing and normalization operations matter for interpreting its response geometry. The mechanism experiment provides a measurable link to this interpretation: cancellation residual has the clearest association with the unlabeled contrast at stronger deletion, whereas shared low-rank energy and effective rank do not consistently explain the ordering across models. This pattern is consistent with a role for downstream response interactions, while the mixed task-conditioned associations indicate that task sensitivity introduces additional structure. The resulting scope is therefore architectural and conditional: the response geometry is most directly motivated for CNN channels, can transfer to related expansion coordinates, and requires a different response model when the meaning of a local coordinate changes.

The formulation also identifies concrete directions for extending its practical range. Estimating full response relationships requires quadratic memory in layer width, and constructing a complete Schur-greedy order has cubic cost. Low-rank or blockwise approximations are therefore natural directions for wide layers, with their quality assessed by the resulting coordinate selections and recovered predictions. The task-conditioned instance additionally requires labels and gradients for the chosen loss, making the calibration task part of the response definition. A complementary modeling question is how to choose the observation unit when normalization or token mixing couples coordinates beyond a single channel. Finally, the link between the volume objective and reconstruction under a fixed downstream readout remain a theoretical question: the present evidence establishes usefulness through the recovered network. These question preserve the framework’s central separation between response modeling, subset selection and functional realization, while identifying where further analysis can make their connection more precise.

## 6 Conclusion

We presented Unified Response Geometry for Structured Pruning, a framework that separates response scale from response relationships and selects retained coordinates through a determinant objective and Schur-greedy ordering. Activation and task-conditioned observations yield two instances of the same construction, followed by functional realization through ridge compensation and applicable BN recalibration. The controlled ImageNet ResNet-50 experiments show that both instances retain higher accuracy than strength-only selection without network fine-tuning. The six-family evaluation finds transfer in additional convolutional and expansion-layer settings while identifying architecture-dependent boundaries, and the mechanism analysis connects part of this variation to downstream response interactions. Together, these findings support a conditional organizing principle for structured pruning: when the observed coordinates contain exploitable complementary responses, the retained set should be selected by its joint response capacity and judged by the function realized by the edited network.

## References

[1] Yann LeCun, John S. Denker, and Sara A. Solla. Optimal brain damage. In Advances in Neural Information Processing Systems, volume 2, 1990.

[2] Song Han, Jeff Pool, John Tran, and William J. Dally. Learning both weights and connections for efficient neural networks. In Advances in Neural Information Processing Systems, volume 28, 2015.

[3] Song Han, Huizi Mao, and William J. Dally. Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding. In International Conference on Learning Representations, 2016.

[4] Zhuang Liu, Jianguo Li, Zehao Shen, Gao Huang, Shouyuan Yan, and Changshui Zhang. Learning efficient convolutional networks through network slimming. In 2017 IEEE International Conference on Computer Vision (ICCV), pages 2755–2763, 2017.

[5] Hengyuan Hu, Rui Peng, Yu-Wing Tai, and Chi-Keung Tang. Network trimming: A data-driven neuron pruning approach towards efficient deep architectures. In 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 113–121, 2016.

[6] Hao Li, Asim Kadav, Igor Durdanovic, Hanan Samet, and Hans Peter Graf. Pruning filters for efficient convnets. In International Conference on Learning Representations, 2017.

[7] Jian-Hao Luo, Jianxin Wu, and Weiyao Lin. Thinet: A filter level pruning method for deep neural network compression. In 2017 IEEE International Conference on Computer Vision (ICCV), pages 5068–5076, 2017.

[8] Yihui He, Xiangyu Zhang, and Jian Sun. Channel pruning for accelerating very deep neural networks. In 2017 IEEE International Conference on Computer Vision (ICCV), pages 1389–1397, 2017.

[9] Pavlo Molchanov, Arun Mallya, Stephen Tyree, Iuri Frosio, and Jan Kautz. Importance estimation for neural network pruning. In 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 11256–11264, 2019.

[10] Kaixiang Shu. Adjoint inversion reveals holographic superposition and destructive interference in cnn classifiers. arXiv preprint arXiv:2604.27529, 2026.

[11] Namhoon Lee, Thalaiyasingam Ajanthan, and Philip H. S. Torr. Snip: Single-shot network pruning based on connection sensitivity. In International Conference on Learning Representations, 2019.

[12] Yang He, Ping Liu, Ziwei Wang, Zhilan Hu, and Yi Yang. Filter pruning via geometric median for deep convolutional neural networks acceleration. In 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 4335–4344, 2019.

[13] Zelda Mariet and Suvrit Sra. Diversity networks: Neural network compression using determinantal point processes. In International Conference on Learning Representations, 2016.

[14] Alex Kulesza and Ben Taskar. Determinantal point processes for machine learning. Foundations and Trends in Machine Learning, 5(2–3):123–286, 2012.

[15] Mingbao Lin, Rongrong Ji, Yan Wang, Yichen Zhang, Baochang Zhang, Yonghong Tian, and Ling Shao. Hrank: Filter pruning using high-rank feature map. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 1529–1538, 2020.

[16] Wei Wen, Chunhuan Wu, Yandan Wang, Yiran Chen, and Hai Li. Learning structured sparsity in deep neural networks. In Advances in Neural Information Processing Systems, volume 29, 2016.

[17] Ruichi Yu, Ang Li, Chun-Fu Chen, Jui-Hsin Lai, Vlad I. Morariu, Xintong Han, Mingfei Gao, Ching-Yung Lin, and Larry S. Davis. Nisp: Pruning networks using neuron importance score propagation. In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9194–9203, 2018.

[18] Yang He, Guoliang Kang, Xuetao Dong, Yanwei Fu, and Yi Yang. Soft filter pruning for accelerating deep convolutional neural networks. In Proceedings ofthe Twenty-Seventh International Joint Conference on Artificial Intelligence, pages 2234–2240, 2018.

[19] Jaeho Lee, Sejun Park, Sangwoo Mo, Sungsoo Ahn, and Jinwoo Shin. Layer-adaptive sparsity for magnitude-based pruning. In International Conference on Learning Representations, 2021.

[20] Chenglong Zhao, Bingbing Ni, Jian Zhang, Qiwei Zhao, Wenjun Zhang, and Qi Tian. Variational convolutional neural network pruning. In 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 591–600, 2019.

[21] Mingjie Sun, Zhuang Liu, Anna Bair, and J. Zico Kolter. A simple and effective pruning approach for large language models. In International Conference on Learning Representations, 2024.

[22] Zechun Liu, Haoyuan Mu, Xiangyu Zhang, Zichao Guo, Xin Yang, Kwang-Ting Cheng, and Jian Sun. Metapruning: Meta learning for automatic neural network channel pruning. In 2019 IEEE/CVF International Conference on Computer Vision (ICCV), pages 3296–3305, 2019.

[23] Tien-Ju Yang, Andrew Howard, Bo Chen, Xiao Zhang, Alec Go, Mark Sandler, Vivienne Sze, and Hartwig Adam. Netadapt: Platform-aware neural network adaptation for mobile applications. In European Conference on Computer Vision (ECCV), pages 285–300, 2018.

[24] Yihui He, Jun Lin, Zhijian Liu, Hanrui Wang, Li-Jia Li, and Song Han. Amc: Automl for model compression and acceleration on mobile devices. In European Conference on Computer Vision (ECCV), pages 784–800, 2018.

[25] Hidenori Tanaka, Daniel Kunin, Daniel L. Yamins, and Surya Ganguli. Pruning neural networks without any data by iteratively conserving synaptic flow. In Advances in Neural Information Processing Systems, volume 33, 2020.

[26] Chaoqi Wang, Guodong Zhang, and Roger Grosse. Picking winning tickets before training by preserving gradient flow. In International Conference on Learning Representations, 2020.

[27] Jonathan Frankle and Michael Carbin. The lottery ticket hypothesis: Finding sparse, trainable neural networks. In International Conference on Learning Representations, 2019.

[28] Victor Sanh, Thomas Wolf, and Alexander M. Rush. Movement pruning: Adaptive sparsity by fine-tuning. In Advances in Neural Information Processing Systems, volume 33, 2020.

[29] Mengzhou Xia, Zexuan Zhong, and Danqi Chen. Structured pruning learns compact and accurate models. In Proceedings ofthe 60th Annual Meeting ofthe Associationfor Computational Linguistics, pages 639–658, 2022.

[30] Elias Frantar and Dan Alistarh. Sparsegpt: Massive language models can be accurately pruned in one-shot. In International Conference on Machine Learning, 2023.

[31] Xinyin Ma, Gongfan Fang, and Xinchao Wang. Llm-pruner: On the structural pruning of large language models. arXiv preprint arXiv:2305.11627, 2023.

[32] Zhuang Liu, Ming Sun, Tinghui Zhou, Gao Huang, and Trevor Darrell. Rethinking the value of network pruning. In International Conference on Learning Representations, 2019.

[33] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 770–778, 2016.

[34] Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A convnet for the 2020s. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 11966–11976, 2022.

[35] Mingxing Tan and Quoc V. Le. Efficientnet: Rethinking model scaling for convolutional neural networks. In International Conference on Machine Learning, pages 6105–6114, 2019.

[36] Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, and Liang-Chieh Chen. Mobilenetv2: Inverted residuals and linear bottlenecks. In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4510–4520, 2018.

[37] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2021.

[38] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pages 9992–10002, 2021.

## Supplementary Material

## A Response estimators and calibration protocol

The supplement provides the implementation detail needed to reproduce the response matrices and to distinguish calibration uncertainty from functional adaptation. The main paper defines the common matrix and selection rule; this section fixes the observation units, moment estimator, gradient convention, and the two independent calibration banks used by the confirmation protocol.

For a convolutional activation $x \in \mathbb { R } ^ { B \times C \times H \times W }$ , each spatial position is treated as one coordinate-response observation. For wide layers, the implementation applies a deterministic evenly spaced cap of 4,096 rows per physical batch to keep the covariance construction tractable; otherwise all available spatial rows are used. The task-gradient tensor uses the same row convention and cap after differentiating the saved activation under the fixed cross-entropy convention. This produces a covariance of task-response variation.

The estimator maintains the sample count, mean and centered second-moment matrix for every target layer. When two accumulators are merged, the between-mean correction is applied explicitly,

$$
S _ { 1 2 } = S _ { 1 } + S _ { 2 } + \frac { n _ { 1 } n _ { 2 } } { n _ { 1 } + n _ { 2 } } ( \mu _ { 2 } - \mu _ { 1 } ) ( \mu _ { 2 } - \mu _ { 1 } ) ^ { \top } .
$$

The covariance is the population convention $S / ( n _ { 1 } + n _ { 2 } )$ used by the implementation. Correlations are normalized from the covariance diagonal; zero-variance directions receive zero off-diagonal correlations and a unit diagonal only when that direction is defined.

Let $\Sigma _ { x }$ and $\Sigma _ { g }$ denote the activation and gradient covariances. The unlabeled instance uses $d _ { i } ^ { ( u ) } = ( \Sigma _ { x } ) _ { i i }$ and $R _ { x } = \operatorname { C o r r } ( x )$ . The task-conditioned instance uses $d _ { i } ^ { ( s ) } = ( \Sigma _ { x } ) _ { i i } ( \Sigma _ { g } ) _ { i i }$ and $R _ { g } = \operatorname { C o r r } ( g )$ , giving

$$
M _ { u } = D _ { x } ^ { 1 / 2 } R _ { x } D _ { x } ^ { 1 / 2 } , \qquad M _ { s } = D _ { x g } ^ { 1 / 2 } R _ { g } D _ { x g } ^ { 1 / 2 } .
$$

The gradient variance in $D _ { x g }$ supplies task-sensitive response scale, while the gradient correlation supplies the relationship term. This factorization is fixed before confirmation. These equations describe the population construction. Because the implementation estimates the diagonal scale and relationship matrix from disjoint calibration banks, the finite-sample product is a split-sample estimate and is not required to match a covariance computed from one bank entry by entry.

The strength bank $\mathcal { C } _ { s }$ and relationship bank $\mathcal { C } _ { r }$ are disjoint. In the confirmation protocol they contain 2,048 and 4,096 images, respectively. The adaptation bank contains 3,200 images and the confirmation bank contains 2,048 held-out validation images. Candidate sets are frozen before confirmation and all methods receive the same target widths, adaptation images, compensation coefficient, normalization-recovery procedure (BN where present) and confirmation images. This separation prevents the recovery set or evaluation outcome from changing the response matrix.

## B Determinant geometry and controlled design choices

The selection objective is the squared volume of the response ellipsoid generated by a candidate subset, consistent with determinant-based diversity selection [1]. If M is positive semidefinite and S is a selected index set, the volume factor is $V ( S ) = \operatorname* { d e t } ( M _ { S S } )$ . For a new coordinate $j ,$ with $V ( \emptyset ) = 1$ , the Schur identity gives

$$
\operatorname * { d e t } ( M _ { S \cup \{ j \} , S \cup \{ j \} } ) = \operatorname * { d e t } ( M _ { S S } ) \left( M _ { j j } - M _ { j S } M _ { S S } ^ { - 1 } M _ { S j } \right) .
$$

The parenthesized term is the residual response variance after projection onto the selected coordinates. The greedy solver therefore appends the largest non-negative residual at each step. If numerical rank is exhausted, the remaining indices are appended in deterministic index order and logged.

For the primary rule, a fixed subset with positive diagonal strengths separates scale and complementarity:

$$
\log \operatorname* { d e t } ( M _ { S S } ) = \sum _ { i \in S } \log d _ { i } + \log \operatorname* { d e t } ( R _ { S S } ) .
$$

For the control study only, we replace R by $R _ { \lambda } = ( 1 - \lambda ) I + \lambda R$ to interpolate between diagonal strength and full geometry. The identity separates the two contributions without changing the primary selector.

The controlled study varies the relationship contribution while holding the strength bank fixed and changing relationship bank size from 1,024 to 4,096 images. The unlabeled $\lambda = 0 . 5$ rule has paired contrasts of +1.42–+1.69 points relative

to its matched diagonal across the four budget/sample-size conditions, with all 12 seed-level differences positive. The full relationship is retained as the primary setting because it is fixed before confirmation and does not depend on accuracy-based tuning; the control results and relationship-repeatability analysis are reported for completeness.

The controlled interpretation is therefore deliberately limited. Activation magnitude supplies physical response scale, gradient variance adds task sensitivity in the task-conditioned instance, and correlation supplies a redundancy relation. The experiments test this factorization under the fixed intervention pipeline.

## C Functional realization and implementation audit

A selected subset becomes deployable only after the coordinates removed from a layer are accounted for in its successor. Let $a _ { S }$ and $a _ { R }$ be retained and removed activation coordinates on the adaptation bank. We fit a zero-intercept ridge predictor,

$$
\boldsymbol { B } = \left( \widehat { \mathbb { E } } [ a _ { S } ^ { \top } a _ { S } ] + \rho \boldsymbol { I } \right) ^ { - 1 } \widehat { \mathbb { E } } [ a _ { S } ^ { \top } a _ { R } ] ,
$$

where $\rho$ is proportional to the mean retained second moment. If the successor convolution has input weights $[ W _ { S } , W _ { R } ]$ replacing $a _ { R }$ by $a _ { S } B$ gives the folded retained weight $W _ { S } + W _ { R } B ^ { \top }$ . The predictor is fit once per target layer and never changes the selected order.

The structural edit slices dependent normalization parameters and preserves the output width of residual additions. For the bottleneck targets, the first successor convolution receives the folded contribution from removed coordinates, while the second target removes its output coordinates and slices the following convolution consistently. The direct-slice state $( t = 0 )$ and compensated state (t = 1) are both retained in the flow diagnostic. No classifier or backbone parameter is trained.

After folding, BN running means and variances are reset and recomputed cumulatively on the adaptation bank. The unpruned network passes through the same BN procedure as a control, because recalibration can itself alter teacher agreement. The primary endpoint is the compensated network after this BN step; raw Top-1, teacher KL, centered-logit MSE and prediction agreement expose the intermediate changes without replacing the primary endpoint.

The implementation audit checks compressed and dense masked forwards before scoring, exact target widths, unpruned identity before BN recovery, deterministic rank-exhaustion behavior, cached image and logit hashes, checkpoint hashes and candidate manifests. The wide-layer GPU Schur solver used by the six-model release agrees with the CPU reference to approximately $1 . 5 \times 1 0 ^ { - 7 }$ in covariance/ordering checks. Together, these checks establish implementation consistency for the reported runs.

## D Confirmation statistics and secondary endpoints

The formal ResNet-50 confirmation uses one ImageNet-pretrained checkpoint and 32 internal bottleneck conv1/conv2 targets. For each of three calibration seeds, the selection, relationship, adaptation and confirmation banks are separate and contain 2,048, 4,096, 3,200 and 2,048 images. The two final instances and the L2 and strength-only controls use 30% and 40% coordinate deletion with $k = \lfloor ( \bar { 1 } - r ) C \rfloor$

The candidate structures are frozen before evaluation. Identical structures share one confirmation forward, while every method receives the same compensation and normalization-recovery procedure (BN where present). The primary endpoint is accuracy after the applicable recovery step, with BN-recovered Top-1 as the CNN endpoint. Teacher KL measures divergence from the unpruned teacher distribution, centered-logit MSE measures logit displacement after removing the teacher mean, and agreement measures the fraction of examples with identical top-1 predictions. These metrics answer different questions and are therefore reported separately rather than collapsed into one score.

Table 5 gives all secondary endpoints for the two final instances and their controls. Table 6 gives paired seed differences for the pre-specified contrasts. Seed-level differences are reported directly; any image bootstrap interval conditions on a fixed calibration seed and is not treated as an independent replication. All paired contrasts for the two final instances relative to strength-only selection are positive for the three calibration seeds.

## E Extended screening and mechanism analysis

The six-family release contains ResNet-50, ConvNeXt-B, EfficientNet-B3, MobileNetV2, ViT-S/16 and Swin-T, with 18 selection entries at each of five deletion rates. All models use the common fixed-budget screening protocol recorded in the release manifest and the final unlabeled and task-conditioned instances, with architecture-specific dependency handling where needed. The task-conditioned rows were regenerated with the final $D _ { x g }$ implementation. The complete tables below report every configured method at the two deletion rates emphasized in the main paper; the release files retain all five rates. The tables expose transfer trends and architecture boundaries, while the replicated ResNet-50 experiment remains the multi-seed formula-level confirmation.

Table 5: Secondary BN metrics averaged over calibration seeds.
<table><tr><td>Method</td><td>Deletion</td><td>Top-1 (%)</td><td>KL↓</td><td>Agreement (%)</td></tr><tr><td>L2</td><td>30%</td><td>47.69</td><td>2.139</td><td>51.35</td></tr><tr><td>L2</td><td>40%</td><td>31.25</td><td>3.272</td><td>33.06</td></tr><tr><td>Strength-only</td><td>30%</td><td>59.81</td><td>1.253</td><td>64.58</td></tr><tr><td>Strength-only</td><td>40%</td><td>43.13</td><td>2.369</td><td>46.13</td></tr><tr><td>Unlabeled instance</td><td>30%</td><td>65.43</td><td>0.890</td><td>71.50</td></tr><tr><td>Unlabeled instance</td><td>40%</td><td>53.92</td><td>1.536</td><td>57.80</td></tr><tr><td>Task-conditioned instance</td><td>30%</td><td>67.66</td><td>0.757</td><td>73.97</td></tr><tr><td>Task-conditioned instance</td><td>40%</td><td>56.25</td><td>1.384</td><td>60.53</td></tr></table>

Table 6: Pre-specified paired Top-1 differences across calibration seeds.
<table><tr><td>Comparison</td><td>Deletion</td><td>Mean (pp)</td><td>Positive seeds</td></tr><tr><td>Unlabeled instance — strength-only</td><td>30%</td><td>+5.62</td><td>3/3</td></tr><tr><td>Task-conditioned instance — strength-only</td><td>30%</td><td>+7.85</td><td>3/3</td></tr><tr><td>Unlabeled instance — strength-only</td><td>40%</td><td>+10.79</td><td>3/3</td></tr><tr><td>Task-conditioned instance — strength-only</td><td>40%</td><td>+13.12</td><td>3/3</td></tr></table>

At 40% deletion, the final task-conditioned-minus-Fisher [2] Top-1 contrasts are 8.3, −2.0, 18.8, 9.8, 14.0 and 4.7 percentage points for ResNet-50, ConvNeXt-B, EfficientNet-B3, MobileNetV2, ViT-S/16 and Swin-T, respectively. The negative ConvNeXt-B value is retained as an architecture boundary; these contrasts come from fixed-budget screening rather than independent-seed estimates.

The architecture pattern follows the response structures motivating the framework: CNN bottleneck and expansion layers are the theory-native setting because Project1 uses spatially repeated channel responses, downstream linear mixing and cancellation; ViT MLP hidden units are a transferable expansion-layer case not directly covered by the Project1 GAP/pixel-space argument; and Swin-T is a boundary where windowed token mixing, LayerNorm and residual paths alter coordinate function.

The mechanism test asks whether the Project1 interpretation predicts the screening pattern [3]. For each model, we aggregate shared low-rank energy, cancellation residual and normalized effective covariance rank over prunable layers with parameter-weighted medians. The contrast endpoints are read from the frozen release, and no descriptor, layer or deletion rate is selected from accuracy. Cancellation residual has the clearest directional association with the unlabeled contrast at 40% (ρ = 0.89, exact permutation p = 0.035, n = 6). Activation low-rank energy is not monotone, and task-conditioned/effective-rank associations are mixed (the 30% correlation is $\rho = 0 . 6 0 , p = 0 . 2 4 )$ . All descriptor–contrast pairs are reported below.

The results show that response geometry changes selection outcomes under the fixed pruning and recovery pipeline, with architecture-dependent contrasts that motivate broader replicated studies.

Table 7: Complete unlabeled comparison at 30% deletion. Top-1 accuracy (%).
<table><tr><td>Method</td><td>ResNet-50</td><td>ConvNeXt-B</td><td>EfficientNet-B3</td><td>MobileNetV2</td><td>ViT-S/16</td><td>Swin-T</td></tr><tr><td>Random</td><td>21.7</td><td>66.3</td><td>31.9</td><td>8.9</td><td>41.6</td><td>60.1</td></tr><tr><td>L1</td><td>5.6</td><td>24.5</td><td>36.6</td><td>23.5</td><td>13.2</td><td>63.0</td></tr><tr><td>L2</td><td>11.9</td><td>50.3</td><td>40.0</td><td>22.7</td><td>20.1</td><td>65.2</td></tr><tr><td>BN-γ</td><td>6.0</td><td>24.5</td><td>33.2</td><td>8.2</td><td>13.2</td><td>63.0</td></tr><tr><td>APoZ</td><td>22.9</td><td>72.5</td><td>39.8</td><td>4.7</td><td>45.6</td><td>63.0</td></tr><tr><td>LAMP</td><td>11.9</td><td>50.3</td><td>40.0</td><td>22.7</td><td>20.1</td><td>65.2</td></tr><tr><td>FPGM</td><td>10.7</td><td>51.2</td><td>41.0</td><td>20.0</td><td>27.9</td><td>62.7</td></tr><tr><td>Unlabeled instance</td><td>40.9</td><td>71.7</td><td>39.7</td><td>15.8</td><td>52.5</td><td>59.9</td></tr></table>

Table 8: Complete label/gradient comparator family at 30% deletion. Top-1 accuracy (%).
<table><tr><td>Method</td><td>ResNet-50</td><td>ConvNeXt-B</td><td>EfficientNet-B3</td><td>MobileNetV2</td><td>ViT-S/16</td><td>Swin-T</td></tr><tr><td>Taylor</td><td>37.0</td><td>67.4</td><td>53.8</td><td>32.5</td><td>43.5</td><td>41.4</td></tr><tr><td>GradNorm</td><td>44.9</td><td>68.3</td><td>44.3</td><td>18.8</td><td>40.7</td><td>51.6</td></tr><tr><td>Fisher</td><td>43.9</td><td>67.4</td><td>52.4</td><td>23.0</td><td>44.1</td><td>55.6</td></tr><tr><td>Taylor-w</td><td>41.7</td><td>68.5</td><td>47.8</td><td>24.3</td><td>49.9</td><td>54.0</td></tr><tr><td>Taylor-gate</td><td>30.9</td><td>72.5</td><td>50.0</td><td>32.6</td><td>47.9</td><td>52.0</td></tr><tr><td>Fisher-gate</td><td>29.9</td><td>69.7</td><td>51.6</td><td>28.7</td><td>53.3</td><td>58.7</td></tr><tr><td>SNIP</td><td>41.7</td><td>68.5</td><td>47.8</td><td>24.3</td><td>49.9</td><td>54.0</td></tr><tr><td>Wanda</td><td>30.4</td><td>24.9</td><td>40.5</td><td>22.1</td><td>10.1</td><td>65.0</td></tr><tr><td>OBD</td><td>44.8</td><td>67.3</td><td>60.8</td><td>37.0</td><td>46.6</td><td>55.5</td></tr><tr><td>Task-conditioned instance</td><td>50.8</td><td>65.7</td><td>61.0</td><td>42.6</td><td>54.0</td><td>60.0</td></tr></table>

Table 9: Complete unlabeled comparison at 40% deletion. Top-1 accuracy (%).
<table><tr><td>Method</td><td>ResNet-50</td><td>ConvNeXt-B</td><td>EfficientNet-B3</td><td>MobileNetV2</td><td>ViT-S/16</td><td>Swin-T</td></tr><tr><td>Random</td><td>4.2</td><td>39.2</td><td>13.5</td><td>1.5</td><td>16.1</td><td>45.9</td></tr><tr><td>L1</td><td>0.9</td><td>2.0</td><td>9.2</td><td>3.0</td><td>4.4</td><td>51.8</td></tr><tr><td>L2</td><td>3.6</td><td>14.0</td><td>15.1</td><td>7.4</td><td>6.7</td><td>55.2</td></tr><tr><td>BN-γ</td><td>0.4</td><td>2.0</td><td>11.8</td><td>1.7</td><td>4.4</td><td>51.8</td></tr><tr><td>APoZ</td><td>5.5</td><td>47.2</td><td>16.8</td><td>1.4</td><td>22.1</td><td>52.2</td></tr><tr><td>LAMP</td><td>3.6</td><td>14.0</td><td>15.1</td><td>7.4</td><td>6.7</td><td>55.2</td></tr><tr><td>FPGM</td><td>3.3</td><td>14.2</td><td>14.5</td><td>4.1</td><td>8.2</td><td>53.3</td></tr><tr><td>Unlabeled instance</td><td>18.8</td><td>47.1</td><td>19.4</td><td>4.7</td><td>31.3</td><td>44.5</td></tr></table>

Table 10: Complete label/gradient comparator family at 40% deletion. Top-1 accuracy (%).
<table><tr><td>Method</td><td>ResNet-50</td><td>ConvNeXt-B</td><td>EfficientNet-B3</td><td>MobileNetV2</td><td>ViT-S/16</td><td>Swin-T</td></tr><tr><td>Taylor</td><td>14.2</td><td>46.6</td><td>31.2</td><td>9.0</td><td>24.9</td><td>19.4</td></tr><tr><td>GradNorm</td><td>24.2</td><td>51.4</td><td>19.0</td><td>6.1</td><td>18.3</td><td>30.0</td></tr><tr><td>Fisher</td><td>19.6</td><td>51.7</td><td>21.2</td><td>6.5</td><td>22.3</td><td>38.4</td></tr><tr><td>Taylor-w</td><td>18.5</td><td>52.1</td><td>14.9</td><td>4.0</td><td>30.2</td><td>34.8</td></tr><tr><td>Taylor-gate</td><td>8.2</td><td>54.5</td><td>23.7</td><td>5.8</td><td>32.2</td><td>31.4</td></tr><tr><td>Fisher-gate</td><td>7.5</td><td>56.0</td><td>22.9</td><td>5.1</td><td>42.5</td><td>42.1</td></tr><tr><td>SNIP</td><td>18.5</td><td>52.1</td><td>14.9</td><td>4.0</td><td>30.2</td><td>34.8</td></tr><tr><td>Wanda</td><td>8.0</td><td>1.1</td><td>10.6</td><td>4.9</td><td>5.0</td><td>50.6</td></tr><tr><td>OBD</td><td>21.1</td><td>52.8</td><td>36.7</td><td>11.5</td><td>27.0</td><td>37.5</td></tr><tr><td>Task-conditioned instance</td><td>27.9</td><td>49.7</td><td>40.0</td><td>16.2</td><td>36.3</td><td>43.0</td></tr></table>

Table 11: Architecture-level response descriptors used in the Project1 alignment analysis. Values are parameter-weighted medians across prunable layers.
<table><tr><td>Model</td><td>Act. low-rank</td><td>Cancellation residual</td><td>Act. eff. rank</td><td>Grad. low-rank</td><td>Grad. eff. rank</td></tr><tr><td>ResNet-50</td><td>0.083</td><td>0.953</td><td>0.122</td><td>0.020</td><td>0.477</td></tr><tr><td>ConvNeXt-B</td><td>0.126</td><td>0.958</td><td>0.019</td><td>0.030</td><td>0.077</td></tr><tr><td>EfficientNet-B3</td><td>0.135</td><td>0.910</td><td>0.024</td><td>0.018</td><td>0.207</td></tr><tr><td>MobileNetV2</td><td>0.119</td><td>0.936</td><td>0.035</td><td>0.014</td><td>0.479</td></tr><tr><td>ViT-S/16</td><td>0.091</td><td>0.969</td><td>0.029</td><td>0.032</td><td>0.074</td></tr><tr><td>Swin-T</td><td>0.428</td><td>0.908</td><td>0.002</td><td>0.018</td><td>0.106</td></tr></table>

Table 12: Spearman correlations between architecture descriptors and matched pruning contrasts. Exact permutation p-values use all 6! permutations; these are descriptive because $n = 6 .$
<table><tr><td>Descriptor</td><td>Unlabeled 30%</td><td>Unlabeled 40%</td><td>Task-cond. 30%</td><td>Task-cond. 40%</td></tr><tr><td>activation low-rank</td><td>-0.60 (0.243)</td><td>-0.49 (0.356)</td><td>-0.37 (0.498)</td><td>-0.14 (0.803)</td></tr><tr><td>cancellation residual</td><td>+0.77 (0.104)</td><td>+0.89 (0.035)</td><td>+0.09 (0.920)</td><td>-0.03 (1.000)</td></tr><tr><td>activation effective rank</td><td>+0.26 (0.659)</td><td>+0.14 (0.803)</td><td>+0.60 (0.243)</td><td>+0.37 (0.498)</td></tr><tr><td>gradient low-rank</td><td>+0.89 (0.035)</td><td>+0.77 (0.104)</td><td>-0.37 (0.498)</td><td>-0.26 (0.659)</td></tr><tr><td>readout alignment</td><td>-0.71 (0.137)</td><td>-0.77 (0.104)</td><td>+0.03 (1.000)</td><td>+0.26 (0.659)</td></tr><tr><td>gradient effective rank</td><td>-0.60 (0.243)</td><td>-0.54 (0.298)</td><td>+0.37 (0.498)</td><td>+0.14 (0.803)</td></tr></table>

Project1 alignment is measurable, but architecture effects are not one-dimensional Activation/gradient response descriptors versus matched pruning contrasts; each point is one active architecture.  
![](images/568f6bf488c1daf64400a1f298f16b30869bf80bea54f84e824b1ec00e740321.jpg)

![](images/b62e663f56c23b28693fae95f2374827ec1a93b5bd407755eb971d9399202c98.jpg)

![](images/2ab57c7f8a39788364441b816d243f6a83554ac22b07f1470ff5a36ad6dbfb75.jpg)

![](images/cc8f7570e6e833b3a964332e10f78e1c20599d1fa905cb2e8761929d38af17d0.jpg)  
Architecture-mechanism alignment  
Six models · n=6 per panel Dots are parameter-weighted layer medians.

![](images/e908a2f264dcc55810482989e2bf11b039d1236dbd83427084d67d606f3e878b.jpg)

![](images/6816eb0d86e8e4821c8f1de7e64db0c6d482a2ef07ec8ea42d8a559113f32426.jpg)

![](images/49c4737fc11967674978b5b678722521b46eead3832a7a65a39815b53adbc45d.jpg)

![](images/6589c27c344974bed77bbc60cb5c3a5416daaf9317914545cce3c1bf3f5b0fa2.jpg)  
Reading the grid  
Every descriptor × contrast pair is shown. ρ is Spearman rank correlation; p is an exact permutation p-value. The grid is descriptive at n=6, not a universal test.

![](images/32f202fe6823e8bbd8df445ea6e3dba4d15322773619f8e4e8c8339d117dac59.jpg)

![](images/9c11187ddddf33829897e0ffeecd84d7253c27601544f3a47cf7252b42645457.jpg)

![](images/26acb40357392375be5c672436765c68f14842008a57faa780e9d8e15301e0db.jpg)

![](images/2d87d8f0c2fb12a9f4e02a988dedc11ee6d35c84bd4a0db534ce66ea32c129b0.jpg)

![](images/612b7ec9c082c343f85d6e6fa2ae0eb5778620aadaf4d3871b6fca3c9d652d90.jpg)  
Observed signal

![](images/439232812ac09e74ad65908d79e2379f055db8b09aaf0c369df231155b6e4a31.jpg)

![](images/145f886e2e4f2eb2bd98f4f2eff90ab03d69c89843571c5a0286ac10674f30fc.jpg)

![](images/73f73e6d553448c3d385d4db536af96f7cfa2dc331815bdea9dfd40dbd3600b1.jpg)  
Cancellation residual tracks Unlabeled contrast at 40% (ρ=+0.89, p=0.035). Other directions are mixed, so no composite alignment score is claimed.

![](images/5e771b739ad99a2b53e9f3434da0d4171180bd48aebcde77392cb72ffbe93eba.jpg)

![](images/7b533e4dd776de59d760e66498ede93c3e39401a86e15ea1af69a07eeb959a7c.jpg)

![](images/6e1f52b6a405cd5b206ffa9e37d34b693ebbf10075891b6971405b909c87df95.jpg)

![](images/10e1a5ea005e1c6ed5f4f7df8422fdafd2d23de60595f33c9058dfcb224cc395.jpg)

![](images/0cd48695de66cf759debef9d7a704ba06a36ec299624e4a92e06f876091ad69f.jpg)

![](images/41cfa2aae15c0009b3fcff507c346240fc50268467de39f923e7129cf599516c.jpg)

![](images/9fbc6bfae16db203fa918cad2f75f16dd2e834601fb71e5c50a5507535494f47.jpg)

![](images/b7044f8a5dd74cfce8590413cde654a158dbaba470e9ad9390047210a059976e.jpg)  
Figure 6: All descriptor–contrast pairs in the six-model mechanism test. Titles report Spearman ρ and exact permutation p; every pair is shown to avoid selective emphasis.

A dense view of method behavior at 30% channel deletion  
![](images/2aa8095421e8499e68dcf21244eae410cf167ac129c7528fbd9e8402d2783c25.jpg)

<table><tr><td rowspan=1 colspan=7>Task-conditioned selection</td></tr><tr><td rowspan=1 colspan=1>37.0</td><td rowspan=1 colspan=1>67.4</td><td rowspan=1 colspan=1>53.8</td><td rowspan=1 colspan=1>32.5</td><td rowspan=1 colspan=1>43.5</td><td rowspan=1 colspan=1>41.4</td><td rowspan=1 colspan=1>42.5B</td></tr><tr><td rowspan=1 colspan=1>44.9</td><td rowspan=1 colspan=1>68.3</td><td rowspan=1 colspan=1>44.3</td><td rowspan=1 colspan=1>18.8</td><td rowspan=1 colspan=1>40.7</td><td rowspan=1 colspan=1>51.6</td><td rowspan=1 colspan=1>44.6</td></tr><tr><td rowspan=1 colspan=1>43.9</td><td rowspan=1 colspan=1>67.4</td><td rowspan=1 colspan=1>52.4</td><td rowspan=1 colspan=1>23.0</td><td rowspan=1 colspan=1>44.1</td><td rowspan=1 colspan=1>55.6</td><td rowspan=1 colspan=1>48.2</td></tr><tr><td rowspan=1 colspan=1>41.7</td><td rowspan=1 colspan=1>68.5</td><td rowspan=1 colspan=1>47.8</td><td rowspan=1 colspan=1>24.3</td><td rowspan=1 colspan=1>49.9</td><td rowspan=1 colspan=1>54.0</td><td rowspan=1 colspan=1>48.8</td></tr><tr><td rowspan=1 colspan=1>30.9</td><td rowspan=1 colspan=1>72.5</td><td rowspan=1 colspan=1>50.0</td><td rowspan=1 colspan=1>32.6</td><td rowspan=1 colspan=1>47.9</td><td rowspan=1 colspan=1>52.0</td><td rowspan=1 colspan=1>49.0</td></tr><tr><td rowspan=1 colspan=1>29.9</td><td rowspan=1 colspan=1>69.7</td><td rowspan=1 colspan=1>51.6</td><td rowspan=1 colspan=1>28.7</td><td rowspan=1 colspan=1>53.3</td><td rowspan=1 colspan=1>58.7</td><td rowspan=1 colspan=1>52.5</td></tr><tr><td rowspan=1 colspan=1>41.7</td><td rowspan=1 colspan=1>68.5</td><td rowspan=1 colspan=1>47.8</td><td rowspan=1 colspan=1>24.3</td><td rowspan=1 colspan=1>49.9</td><td rowspan=1 colspan=1>54.0</td><td rowspan=1 colspan=1>48.8</td></tr><tr><td rowspan=1 colspan=1>30.4</td><td rowspan=1 colspan=1>24.9</td><td rowspan=1 colspan=1>40.5</td><td rowspan=1 colspan=1>22.1</td><td rowspan=1 colspan=1>10.1</td><td rowspan=1 colspan=1>65.0</td><td rowspan=1 colspan=1>27.6</td></tr><tr><td rowspan=1 colspan=1>44.8</td><td rowspan=1 colspan=1>67.3</td><td rowspan=1 colspan=1>60.8</td><td rowspan=1 colspan=1>37.0</td><td rowspan=1 colspan=1>46.6</td><td rowspan=1 colspan=1>55.5</td><td rowspan=1 colspan=1>51.0</td></tr><tr><td rowspan=1 colspan=1>年50.8</td><td rowspan=1 colspan=1>65.7</td><td rowspan=1 colspan=1>★61.0</td><td rowspan=1 colspan=1>吉42.6</td><td rowspan=1 colspan=1>青54.0</td><td rowspan=1 colspan=1>60.0</td><td rowspan=1 colspan=1>57.0</td></tr><tr><td rowspan=1 colspan=7>R-50   CNX-B   EN-B3   MNv2   ViT-S   Swin-T  Median</td></tr></table>

![](images/cf1d53895e0577c40fb01b63540204e9da7395697803678244fad57b0bc1f561.jpg)  
Figure 7: Complete configured methods at 30% coordinate deletion for the six-model screen.

A dense view of method behavior at 40% channel deletion  
![](images/7a513e029f70e80c538296b7ed9a4d3c2b4ed43af2d6754f6812c3b2fc7d241b.jpg)

<table><tr><td rowspan=1 colspan=7>Task-conditioned selection</td></tr><tr><td rowspan=1 colspan=1>14.2</td><td rowspan=1 colspan=1>46.6</td><td rowspan=1 colspan=1>31.2</td><td rowspan=1 colspan=1>9.0</td><td rowspan=1 colspan=1>24.9</td><td rowspan=1 colspan=1>19.4</td><td rowspan=1 colspan=1>22.1 B</td></tr><tr><td rowspan=1 colspan=1>24.2</td><td rowspan=1 colspan=1>51.4</td><td rowspan=1 colspan=1>19.0</td><td rowspan=1 colspan=1>6.1</td><td rowspan=1 colspan=1>18.3</td><td rowspan=1 colspan=1>30.0</td><td rowspan=1 colspan=1>21.6</td></tr><tr><td rowspan=1 colspan=1>19.6</td><td rowspan=1 colspan=1>51.7</td><td rowspan=1 colspan=1>21.2</td><td rowspan=1 colspan=1>6.5</td><td rowspan=1 colspan=1>22.3</td><td rowspan=1 colspan=1>38.4</td><td rowspan=1 colspan=1>21.8</td></tr><tr><td rowspan=1 colspan=1>18.5</td><td rowspan=1 colspan=1>52.1</td><td rowspan=1 colspan=1>14.9</td><td rowspan=1 colspan=1>4.0</td><td rowspan=1 colspan=1>30.2</td><td rowspan=1 colspan=1>34.8</td><td rowspan=1 colspan=1>24.3</td></tr><tr><td rowspan=1 colspan=1>8.2</td><td rowspan=1 colspan=1>54.5</td><td rowspan=1 colspan=1>23.7</td><td rowspan=1 colspan=1>5.8</td><td rowspan=1 colspan=1>32.2</td><td rowspan=1 colspan=1>31.4</td><td rowspan=1 colspan=1>27.5</td></tr><tr><td rowspan=1 colspan=1>7.5</td><td rowspan=1 colspan=1>56.0</td><td rowspan=1 colspan=1>22.9</td><td rowspan=1 colspan=1>5.1</td><td rowspan=1 colspan=1>42.5</td><td rowspan=1 colspan=1>42.1</td><td rowspan=1 colspan=1>32.5</td></tr><tr><td rowspan=1 colspan=1>18.5</td><td rowspan=1 colspan=1>52.1</td><td rowspan=1 colspan=1>14.9</td><td rowspan=1 colspan=1>4.0</td><td rowspan=1 colspan=1>30.2</td><td rowspan=1 colspan=1>34.8</td><td rowspan=1 colspan=1>24.3</td></tr><tr><td rowspan=1 colspan=1>8.0</td><td rowspan=1 colspan=1>1.1</td><td rowspan=1 colspan=1>10.6</td><td rowspan=1 colspan=1>4.9</td><td rowspan=1 colspan=1>5.0</td><td rowspan=1 colspan=1>★50.6</td><td rowspan=1 colspan=1>6.5</td></tr><tr><td rowspan=1 colspan=1>21.1</td><td rowspan=1 colspan=1>52.8</td><td rowspan=1 colspan=1>36.7</td><td rowspan=1 colspan=1>11.5</td><td rowspan=1 colspan=1>27.0</td><td rowspan=1 colspan=1>37.5</td><td rowspan=1 colspan=1>31.8</td></tr><tr><td rowspan=1 colspan=1>★27.9</td><td rowspan=1 colspan=1>49.7</td><td rowspan=1 colspan=1>★40.0</td><td rowspan=1 colspan=1>★16.2</td><td rowspan=1 colspan=1>36.3</td><td rowspan=1 colspan=1>43.0</td><td rowspan=1 colspan=1>38.2</td></tr><tr><td rowspan=1 colspan=7>R-50   CNX-B   EN-B3   MNv2   ViT-S   Swin-T  Median</td></tr></table>

![](images/a681970d9b0ca8ad0642515f28a2b58cf4090989303b351d9dbea7b9d6ae5df3.jpg)  
Figure 8: Complete configured methods at 40% coordinate deletion for the six-model screen.

Unlabeled instance contrasts across deletion budgets Each cell is one architecture; comparator = L2 in the same budget and pipeline. Unlabeled instance − L2  
![](images/31ec9c202ebe8f1b866dca67f172b0d28703ce79f23fbecb158acd5d84d18980.jpg)

![](images/bb3fcc8ae660d78fbba1dc8995a1b882b8e42c82423fe137791e85598602a8cd.jpg)  
Figure 9: Unlabeled-instance Top-1 contrast relative to L2 across deletion rates and architectures. The right panel reports the mean contrast and the fraction of architectures with a positive contrast.

Task-conditioned instance contrasts across deletion budgets Each cell is one architecture; comparator = Fisher in the same budget and pipeline. Task-conditioned instance − Fisher  
![](images/020d077c672705ccccaee0bde90d2fe871cb7bb02e074cabdf487996daca3ebe.jpg)

![](images/04287f7557f8845b2d02fd3391a3343faabc824d2d8697672c19d64f05e9aafb.jpg)  
Figure 10: Task-conditioned Top-1 contrast relative to Fisher across deletion rates and architectures. The right panel reports the mean contrast and the fraction of architectures with a positive contrast.

## References

[1] Zelda Mariet and Suvrit Sra. Diversity networks: Neural network compression using determinantal point processes. In International Conference on Learning Representations, 2016.

[2] Pavlo Molchanov, Arun Mallya, Stephen Tyree, Iuri Frosio, and Jan Kautz. Importance estimation for neural network pruning. In 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 11256–11264, 2019.

[3] Kaixiang Shu. Adjoint inversion reveals holographic superposition and destructive interference in cnn classifiers. arXiv preprint arXiv:2604.27529, 2026.