# Mitigating Spatial Blindness in Multiple Instance Learning for Whole-Slide Image Analysis

Xiangyu Li<sup>a,b,\*</sup>, Jincheng Li<sup>b,d,c,\*</sup>, Xing Xie<sup>b,e</sup>, Bin Guan<sup>b,f</sup>, Shaohua Kevin Zhou<sup>c,d,†</sup>, Rui Yan<sup>c,d,1,†</sup>, Ran Su<sup>a,†</sup>

6a<sub>College</sub> <sub>of</sub> <sub>Intelligence</sub> <sub>and</sub> <sub>Computing,</sub> <sub>Tianjin</sub> <sub>University,</sub> <sub>Tianjin,</sub> <sub>China</sub>

2b<sub>Shanghai Artificial Intelligence Laboratory, Shanghai, China</sub>

0c<sub>School</sub> <sub>of</sub> <sub>Biomedical</sub> <sub>Engineering,</sub> <sub>University</sub> <sub>of</sub> <sub>Science</sub> <sub>and</sub> <sub>Technology</sub> <sub>of</sub> <sub>China,</sub> <sub>Hefei,</sub> <sub>China</sub>

2d<sub>Me</sub>d<sub>ica</sub>l <sub>Imaging, Ro</sub>b<sub>otics, Ana</sub>l<sub>ytic Computing</sub> & <sub>Learning</sub> (<sub>MIRACLE</sub>) <sub>La</sub>b<sub>, USTC Suz</sub>h<sub>ou Institute</sub>f<sub>or A</sub>d<sub>vance</sub>d <sub>Researc</sub>h<sub>, Suz</sub>h<sub>ou, C</sub>h<sub>ina</sub>

<sup>e</sup>Engineering Training Center, Nantong University, Nantong, China

<sup>f</sup>School ofInformation Science and Engineering, University ofJinan, Jinan, China

<sup>3</sup>A R T I C L E I N F O

## A B S T R A C T

VArticle history:

Keywords: Computational pathology, Whole-slide images, Multiple instance learning, Spatial modeling

Whole-slide multiple instance learning can predict diagnostic and prognostic labels from tissue composition while making limited use of spatial organization. Providing coordinates or a graph does not establish that a trained predictor uses tissue neighborhoods. We examine this gap as spatial blindness and hypothesize that early composition learning can weaken supervision of a spatial pathway. We introduce ResTopoMIL, which freezes a learned composition predictor before training a graph residual with a coordinate-reassignment separation loss. Permutation and information analyses distinguish architectural symmetry, available spatial information and the proposed optimization mechanism. On a controlled spatial task, staged training achieved an AUC of 0.987, compared with 0.684 for the same jointly trained architecture. Performance remained high on a composition-only task. Across four WSI classification and five survival benchmarks, ResTopoMIL improved mean primary performance on eight tasks; four improvements remained significant after Holm-adjusted patient-cluster tests. Coordinate-free residual controls recovered some of the gains, while neighborhood destruction reduced performance close to the composition anchor. Neighborhood-preserving transformations and anatomy-stratified perturbations further characterized this dependence. These results support fixing a composition reference as a practical intervention for learning spatial corrections, while retaining additional appearance capacity as a contributing explanation.

© 2026 Elsevier B. V. All rights reserved.

## 1. Introduction

Tissue composition and spatial organization provide complementary evidence for whole-slide prediction. Multiple instance learning (MIL) addresses this task using bags of patch features and slide-level labels [1–4]. This weak supervision makes large histopathology images tractable without dense annotations [5–7].

However, a high diagnostic or prognostic score does not reveal which tissue evidence supports the prediction. Tumor abundance can be informative, whereas gland formation, invasive fronts and tumor–stroma interfaces depend on spatial relationships [8–10].

Accurate prediction from an unordered bag therefore leaves an unresolved question about spatial learning. Attention pooling is invariant to instance order, and self-attention without positional encoding is equivariant before invariant readout [11–13]. These properties allow rich appearance statistics to support prediction without physical coordinates. They do not imply that every order-invariant model ignores space. A graph predictor can remain invariant to node relabeling while depending on its adjacency. The relevant intervention changes feature–coordinate associations, rather than merely changing the order in which paired instances are listed.

![](images/4ceb267249ee8f7ba25950ab3c42d0b94e934e2cfb0ee81ea77a99a9d8e55a72.jpg)  
Fig. 1: Motivation and coordinate-reassignment diagnostic. (a) Coordinate-free MIL receives the same appearance multiset after coordinate reassignment and therefore gives the same prediction; joint relabeling of paired features and coordinates is a diferent operation. The schematic AUC values are illustrative. (b) ResTopoMIL separates a composition reference from a graph correction constrained to distinguish intact from disrupted neighborhoods. (c) The TCGA-BRCA diagnostic holds patch embeddings fixed while reassigning coordinates. Near-invariance is expected when a model’s efective inputs do not change; it does not itself establish failed spatial learning. Table 6 tests spatial baselines with rebuilt coordinate-derived inputs.

Spatial architectures provide access to neighborhoods, but access alone does not establish learned use. Graph networks, positional Transformers, hierarchical aggregators and state-space models introduce spatial or contextual pathways [14–20]. Their predictors can still exploit appearance signals that are easier to learn. We use spatial blindness to describe weak prediction or performance response to neighborhood destruction on a structure-dependent task. Figure 1(c) motivates this behavioral distinction, while comparisons with spatial baselines explicitly rebuild their coordinate-derived inputs. The unresolved gap concerns how training can increase useful neighborhood dependence beyond simply adding another prediction branch.

Our optimization hypothesis is that composition can reduce prediction error before a spatial pathway learns useful relations. For additive logits, the supervised graph update is multiplied by the model’s current prediction error. This creates a conditional route to weak spatial supervision, consistent with shortcutlearning and gradient-starvation accounts [21–24]. The gradient identity motivates an intervention; it does not prove that composition always learns first or that freezing must increase gradients.

ResTopoMIL fixes a composition reference before learning a graph-conditioned correction (Figure 1). A prototype histogram first predicts from the appearance multiset. After this predictor is frozen, a graph branch learns residual logits, while a margin loss separates intact and coordinate-reassigned graph representations. This design makes the training schedule testable within the same architecture and makes neighborhood dependence testable independently of appearance capacity.

The contributions of this study are threefold:

• We formulate spatial blindness as weak neighborhood dependence in WSI MIL and distinguish it from permutation symmetry. A conditional gradient analysis motivates the proposed optimization hypothesis.

• We introduce ResTopoMIL, which freezes a prototypebased composition predictor before learning an additive graph residual. A coordinate-reassignment separation loss encourages the graph branch to distinguish intact from disrupted neighborhoods.

• We evaluate the intervention using controlled composition and spatial tasks, nine WSI prediction benchmarks and matched ablations. Coordinate-free residual controls, neighborhood perturbations and anatomy-stratified tests distinguish spatial dependence from additional appearance capacity.

## 2. Related work

## 2.1. Appearance aggregation under slide-level supervision

MIL learns bag predictions without requiring a label for every instance. Early pathology methods combined patch classifiers, clustering, global pooling or multi-view CNN representations [25–30]. Attention MIL learns instance weights within an orderinvariant readout [4, 11]. Later approaches use dual streams, bag distillation, hard-instance mining and attention regularization to improve discrimination [31–40]. These methods show that appearance aggregation is a strong predictive baseline, making its contribution essential to isolate when evaluating spatial models.

Stronger patch encoders increase the information available to any aggregator. Pathology foundation models and selfsupervised representations improve local embeddings [41–46]. Vision-language and prompt-based methods also broaden slide supervision [47–51]. We therefore hold the patch encoder fixed within each comparison and evaluate classification with both UNI and CTransPath. This protocol targets aggregation and training rather than additional encoder capacity.

## 2.2. Spatial and contextual aggregation

Structured MIL recognizes that relations between instances can afect bag labels [52, 53]. Graph methods represent tissue as a point cloud or neighborhood graph [14–16, 54]. Transformers and low-rank attention model interactions across large bags [17, 18, 55–57]. Hierarchical and context-aware methods exploit WSI pyramids and regional organization [58–63], while statespace models address long-sequence computation [19, 20, 64]. Our question complements these architectural developments by testing whether a trained predictor uses the spatial input it receives.

## 2.3. Spatial learning and optimization

Prior work has explicitly examined order symmetry and positional information. Chen et al. [13] analyzed these properties and proposed restoring instance arrangement through a Siamese formulation. NcIEMIL addressed feature-extraction and aggregation redundancy using filtering and spatial/channel attention [65]. ResTopoMIL instead tests a training intervention when appearance and graph pathways coexist. Its permutation arguments build on established set-model analysis, rather than constituting a new general theory of MIL.

The intervention combines established representation ideas with a diferent optimization schedule. Deep Sets, histograms and morphological prototypes motivate the composition reference [12, 66–68]. Shortcut learning motivates testing whether easy signals dominate joint training [21, 24]. Conditionalinformation identities describe what remains beyond a fixed reference [69, 70]. The contribution lies in connecting these ideas through a staged model and evaluating its neighborhood dependence under matched controls.

## 3. Problem formulation and analysis

The analysis separates three questions: what a model can represent, what spatial information is available, and what optimization may encourage it to learn. This distinction prevents permutation symmetry from being treated as proof of spatial blindness.

## 3.1. Permutation symmetry and neighborhood dependence

A slide is a paired bag $X = \{ ( \mathbf { h } _ { i } , \mathbf { p } _ { i } ) \} _ { i = 1 } ^ { N }$ , where $\mathbf { h } _ { i } \in \mathbb { R } ^ { d }$ is a patch feature and $\mathbf { p } _ { i } \in \mathbb { R } ^ { 2 }$ its coordinate. Let H and R stack these quantities by row. The appearance multiset is $B = \{ \mathbf { h } _ { i } \} _ { i = 1 } ^ { N }$ Composition denotes statistics of B, whereas spatial organization depends on the feature–coordinate pairing.

Proposition 1 (Permutation symmetry does not imply spatial blindness). Let Π be a permutation matrix and ρ an invariant readout. An equivariant coordinate-free map T satisfies $\rho ( T ( \boldsymbol { \Pi } \mathbf { H } ) ) = \rho ( T ( \mathbf { H } ) )$ . An equivariant graph map G satisfies

$$
\rho ( G ( \Pi \mathbf { H } , \Pi \mathbf { A } \Pi ^ { \top } ) ) = \rho ( G ( \mathbf { H } , \mathbf { A } ) ) .\tag{1}
$$

Neither identity requires invariance when A changes while H remainsfixed.

Proof. Attention pooling provides a coordinate-free example. For a shared scorer $^ { a , }$ define $e _ { i } \ = \ a ( \mathbf { h } _ { i } )$ and $\alpha ( \mathbf { H } ) \ =$ softmax $( [ e _ { 1 } , \ldots , e _ { N } ] ^ { \top } )$ . Its pooled representation obeys

$$
\begin{array} { r l } { z ( \boldsymbol { \Pi } \mathbf { H } ) = ( \boldsymbol { \Pi } \mathbf { H } ) ^ { \top } \boldsymbol { \Pi } \alpha ( \mathbf { H } ) } & { { } } \\ { = \mathbf { H } ^ { \top } \alpha ( \mathbf { H } ) = z ( \mathbf { H } ) , } & { { } } \end{array}\tag{2}
$$

because Π<sup>⊤</sup>Π = I. Shared gated scorers have the same property [11].

Self-attention without positional encoding is equivariant before pooling. Let $\mathbf { Q } = \mathbf { H } \mathbf { W } _ { Q } , \mathbf { K } = \mathbf { H } \mathbf { W } _ { K }$ and $\mathbf { V } = \mathbf { H } \mathbf { W } _ { V }$ . Rowwise softmax satisfies softmax(ΠMΠ<sup>⊤</sup>) = Πsoftmax(M)Π<sup>⊤</sup>, including permutation of each row’s normalization sum. Consequently,

$$
\begin{array} { r l } & { T ( \Pi \mathbf { H } ) = \operatorname { s o f t m a x } \left( \frac { \Pi \mathbf { Q } \mathbf { K } ^ { \top } \Pi ^ { \top } } { \sqrt { d _ { k } } } \right) \Pi \mathbf { V } } \\ & { ~ = \Pi T ( \mathbf { H } ) . } \end{array}\tag{3}
$$

Shared tokenwise feed-forward maps, activations and layer normalization preserve equivariance. Invariant pooling gives an invariant bag predictor. A class token is handled with Π =<sup>¯</sup> diag(1 Π), which leaves its output row unchanged [13].

For a GCN, let $\hat { \mathbf { A } } = \tilde { \mathbf { D } } ^ { - 1 / 2 } ( \mathbf { A } + \mathbf { I } ) \bar { \tilde { \mathbf { D } } } ^ { - 1 / 2 }$ . Joint relabeling gives $\mathbf { H } ^ { \prime } = \Pi \mathbf { H }$ and $\hat { \mathbf { A } } ^ { \prime } = \Pi \hat { \mathbf { A } } \Pi ^ { \top }$ . Each shared-weight layer satisfies

$$
\sigma ( \hat { \mathbf { A } } ^ { \prime } \mathbf { H } ^ { \prime } \mathbf { W } ) = \Pi \sigma ( \hat { \mathbf { A } } \mathbf { H } \mathbf { W } ) .\tag{4}
$$

Induction establishes stack equivariance, and invariant readout gives Eq. (1).

Coordinate reassignment tests a diferent property from joint relabeling. For an equivariant graph constructor with consistently resolved distance ties, $\mathbf { A } ( \Pi \mathbf { R } ) = \Pi \mathbf { A } ( \mathbf { R } ) \Pi ^ { \top }$ . Reindexing (H R) together preserves the paired bag. Holding H fixed and permuting only R instead changes which appearances occupy adjacent locations. The coordinate set and unlabeled geometric graph remain unchanged, but feature–neighborhood associations difer. Proposition 1 imposes no equality between the resulting predictions. These symmetry statements assume deterministic evaluation and do not describe stochastic dropout diferences.

## 3.2. Positional access does not ensure learned sensitivity

Positional encoding can remove an architectural invariance guarantee without guaranteeing diferent predictions. For an invariant aggregator F and slot-attached encodings E,

$$
F ( \boldsymbol { \Pi } \mathbf { H } + \mathbf { E } ) = F ( \mathbf { H } + \boldsymbol { \Pi } ^ { \top } \mathbf { E } ) .\tag{5}
$$

The right-hand side need not equal $F ( \mathbf { H } + \mathbf { E } )$ . However, distinct inputs can share an output. Mean pooling gives the counterexample

$$
\frac { 1 } { N } \mathbf { 1 } ^ { \top } ( \Pi \mathbf { H } + \mathbf { E } ) = \frac { 1 } { N } \mathbf { 1 } ^ { \top } ( \mathbf { H } + \mathbf { E } ) .\tag{6}
$$

Repeated encoding rows can also admit nontrivial invariant permutations. Conversely, jointly permuting features and their encodings preserves the output of any invariant F.

Sensitivity must therefore be evaluated for the trained model and the intervention actually supplied. Physical-coordinate reassignment may leave a padded positional grid unchanged. Such a model is an input-preserving control in that test, rather than evidence that its positional module failed to learn. Spatial baselines are evaluated by rebuilding their graph or coordinate-derived sequence.

## 3.3. Spatial information beyond appearance

An appearance predictor can be accurate even when coordinates contain additional label information. Since B is a deterministic projection of X, standard conditional-information identities yield

$$
\begin{array} { r } { I ( Y ; X \mid B ) = H ( Y \mid B ) - H ( Y \mid X , B ) \qquad } \\ { \quad = H ( Y \mid B ) - H ( Y \mid X ) \ge 0 . } \end{array}\tag{7}
$$

For discrete labels, this also equals $\mathbb { E } _ { X } { \mathrm { K L } } ( p ( Y \mid X ) \| p ( Y \mid B ) )$ [13, 70]. It vanishes precisely when these posteriors agree almost surely.

Positive conditional information establishes a potential reduction in optimal expected log loss. It does not require a strict reduction in zero–one classification error, because the most probable class may remain unchanged. A predictor with access to X can always ignore coordinates, so its optimal classification risk cannot exceed that using B. Neither statement guarantees that a finite-data network learns the available spatial signal. We reserve spatial blindness for weak behavioral dependence on neighborhoods on a task where those neighborhoods matter.

## 3.4. A conditional optimization hypothesis

An additive predictor exposes one route by which composition may weaken spatial supervision. Write

$$
f ( X ) = f _ { s t a t } ( B ; \theta _ { s } ) + f _ { t o p o } ( X ; \theta _ { t } ) .\tag{8}
$$

This logit decomposition does not identify separate appearance and topology factors. The graph term can also encode appearance information omitted by the statistical predictor.

Proposition 2 (Residual-error gating of the graph update). For binary cross-entropy, let $r _ { \theta } ( X , Y ) = \sigma ( f ( X ) ) - Y$ . Assumefinite second moments and valid interchange of diferentiation and expectation. Then

$$
\nabla _ { \theta _ { t } } \mathcal { L } = \mathbb { E } [ r _ { \theta } ( X , Y ) \nabla _ { \theta _ { t } } f _ { t o p o } ( X ) ] ,\tag{9}
$$

and

$$
\| \nabla _ { \theta _ { t } } \mathcal { L } \| \leq \sqrt { \mathbb { E } r _ { \theta } ^ { 2 } } \ \sqrt { \mathbb { E } \| \nabla _ { \theta _ { t } } f _ { t o p o } \| _ { F } ^ { 2 } } .\tag{10}
$$

Proof. Set $\hat { p } _ { \theta } ~ = ~ \sigma ( f ( X ) )$ . Diferentiating the binary loss −Y log $\hat { p } _ { \theta } - ( 1 - Y ) \log ( 1 - \hat { p } _ { \theta } )$ with respect to f gives ${ \hat { p } } _ { \theta } - Y$ The chain rule gives Eq. (9). The norm inequality and Cauchy– Schwarz give Eq. (10).

The bound contracts if composition reduces the squared prediction error early while the graph Jacobian remains bounded. This condition motivates testing whether a fixed composition reference improves subsequent graph learning. It does not establish that composition always learns first or compare staged and joint gradient magnitudes. The toy experiment tests the schedule within one architecture, while gradient traces and WSI controls examine its empirical consequences.

## 4. ResTopoMIL

ResTopoMIL separates a fixed composition reference from a trainable graph correction. Figure 2 shows the three design choices needed for this intervention: an order-invariant anchor, staged additive prediction and a coordinate-specific representation constraint. The constraint tests graph dependence beyond the residual objective itself.

## 4.1. A prototype histogram as the composition reference

The statistical stream estimates how much the appearance multiset explains without coordinates. A constrained prototype histogram provides an inspectable reference without assuming that it exhausts all appearance information. Let $C = \left\{ \mathbf { c } _ { k } \right\} _ { k = 1 } ^ { K _ { p r o t o } }$ be a learnable codebook. For each patch, the soft assignment is

$$
a _ { i k } = \frac { \exp ( - \| \mathbf { h } _ { i } - \mathbf { c } _ { k } \| ^ { 2 } / \tau ) } { \sum _ { j = 1 } ^ { K _ { p r o t o } } \exp ( - \| \mathbf { h } _ { i } - \mathbf { c } _ { j } \| ^ { 2 } / \tau ) } .\tag{11}
$$

The temperature τ controls assignment sharpness. Averaging gives the bag histogram and statistical logits,

$$
\mathbf { z } _ { s t a t } = \frac { 1 } { N } \sum _ { i } \mathbf { a } _ { i } , \qquad f _ { s t a t } = \mathop { \mathrm { M L P } } ( \mathbf { z } _ { s t a t } ) .\tag{12}
$$

The histogram records soft prototype proportions, rather than diagnostic class probabilities or spatial locations. Its entries sum to one because each patch assignment is normalized.

Prototype frequencies summarize recurring appearance patterns [68]. MiniBatch K-Means initializes the codebook from training patch embeddings, after which both the prototypes and temperature are learned. A 512-dimensional hidden representation precedes the statistical prediction head. Anchorconvergence tests examine the warm-up duration, and coordinatefree residual controls quantify usable appearance information omitted by this reference.

## 4.2. A graph-conditioned correction

The graph stream gives the correction access to physical neighborhoods while retaining the same patch features. Each patch forms a node, and a coordinate KNN graph supplies the adjacency A. We use two GCN layers,

$$
\mathbf { H } ^ { ( l + 1 ) } = \sigma \big ( \tilde { \mathbf { D } } ^ { - 1 / 2 } \tilde { \mathbf { A } } \tilde { \mathbf { D } } ^ { - 1 / 2 } \mathbf { H } ^ { ( l ) } \mathbf { W } ^ { ( l ) } \big ) ,\tag{13}
$$

where $\tilde { \mathbf { A } } = \mathbf { A } + \mathbf { I } ,$ , D<sup>˜</sup> is its degree matrix and $\mathbf { H } ^ { ( 0 ) } = \mathbf { H }$ . Global mean pooling and a linear head give

$$
\mathbf { z } _ { t o p o } = \frac { 1 } { N } \sum _ { i } \mathbf { H } _ { i } ^ { ( 2 ) } , \qquad f _ { t o p o } = \mathbf { W } _ { t o p o } \mathbf { z } _ { t o p o } + \mathbf { b } _ { t o p o } .\tag{14}
$$

Coordinate neighborhoods distinguish visually similar patches occupying diferent tissue contexts. The default graph has eight nearest neighbors per node before self-loops, and its hidden features have dimension 512. ReLU and dropout with probability 0.25 follow the graph transformations. The shallow branch remains fixed in the joint-versus-staged comparison, allowing the training schedule to be evaluated without changing architecture.

![](images/c7c307685d90cdbdc226e14a0f50369834e14e52245de8c6470cade195117fcf.jpg)  
Fig. 2: ResTopoMIL framework. A frozen patch encoder supplies shared appearance features. The statistical stream averages soft prototype assignments into a composition histogram and is trained in Stage 1, then frozen in Stage 2. The topological stream constructs KNN graphs from intact and reassigned coordinates and processes both views with a shared, trainable two-layer GCN. A margin loss separates their pooled representations. The statistical and intact-graph prediction logits are added before probability conversion, as specified in Eq. (15). The reassigned view and separation loss are used only during training; inference uses the intact graph.

## 4.3. Additive logits with afixed reference

Fixing the anchor prevents the composition predictor from changing throughout residual learning. Stage 1 trains the statistical stream alone for ten epochs. Stage 2 freezes its codebook, temperature, MLP and prediction head, then optimizes the graph branch through

$$
f ( X ) = \operatorname { s g } [ f _ { s t a t } ( B ) ] + f _ { t o p o } ( X ) ,\tag{15}
$$

where sg stops gradients. Both heads produce logits in the same output space. Classification probabilities are obtained only after summation, using Softmax for multiclass prediction or a sigmoid for a single binary logit.

The residual is defined by the predictive objective, rather than by subtracting hidden features or fitting a separate residual label. The supervised error evolves as the graph learns, but the anchor cannot absorb changes in that error. Larger graph learning rates, statistical dropout, curricula and hard-instance mining alter joint optimization while leaving the reference trainable. Table 5 compares these interventions. Freezing changes the learning problem, although Proposition 2 provides no guarantee that its gradients are larger.

## 4.4. Separating intact and reassigned neighborhoods

Residual prediction alone can recover appearance information that the anchor missed. We therefore use a coordinate-specific constraint to discourage the graph stream from ignoring its adjacency. A corrupted bag X<sup>˜</sup> retains H and reassigns coordinates as ΠR, after which the KNN graph is rebuilt. The same trainable GCN processes both views and produces $\mathbf { z } _ { t o p o }$ and $\tilde { \mathbf { z } } _ { t o p o }$

The auxiliary objective separates these representations using a cosine-distance margin,

$$
\mathcal { L } _ { t e x t u r e } = \operatorname* { m a x } \Bigl ( 0 , m - \Bigl [ 1 - \cos ( \mathbf { z } _ { t o p o } , \tilde { \mathbf { z } } _ { t o p o } ) \Bigr ] \Bigr ) .\tag{16}
$$

The loss decreases as similarity falls, and vanishes once the cosine distance reaches m. At the default $m = 0 . 3$ , the zero-loss condition is similarity at most 0.7. This is a separation constraint without an explicit positive-pair attraction term. Coordinate reassignment is not assumed to preserve the tissue label.

Supervision uses the intact view, while the reassigned view enters only the representation comparison. The Stage-2 classification objective is

$$
\mathcal { L } _ { t o t a l } = \mathcal { L } _ { c l s } ( \mathbf { s g } [ f _ { s t a t } ] + f _ { t o p o } , Y ) + \lambda \mathcal { L } _ { t e x t u r e } .\tag{17}
$$

Survival prediction replaces classification loss with the discretetime survival objective. A deterministic graph-ignoring representation cannot distinguish the two views, but representation separation alone does not ensure that prediction uses the diference. Inference therefore uses only the intact graph and sums the two logits once; perturbation experiments test the learned predictor’s dependence on neighborhoods.

## 4.5. Information captured beyond the anchor

The residual objective targets information beyond a fixed reference without identifying that information as spatial. Let $Z _ { s t a t }$ and $Z _ { t o p o }$ denote the branch representations as random variables. The chain rule gives

$$
\begin{array} { r } { I ( Z _ { s t a t } , Z _ { t o p o } ; Y ) = I ( Z _ { s t a t } ; Y ) + I ( Z _ { t o p o } ; Y \mid Z _ { s t a t } ) . } \end{array}\tag{18}
$$

Stage 2 fixes the first term, while the graph branch and its head parameterize

$$
q _ { \phi } ( y \mid Z _ { s t a t } , Z _ { t o p o } ) = \left[ \mathrm { S o f t m a x } ( f _ { s t a t } + f _ { t o p o , \phi } ) \right] _ { y } .\tag{19}
$$

For binary prediction, the positive-class probability is $\sigma ( f _ { s t a t } +$ $f _ { t o p o } )$

Proposition 3 (Residual conditional-information bound). With $Z _ { s t a t }$ and $f _ { s t a t }$ fixed, minimizing the decoder’s cross-entropy maximizes the variational lower bound

$$
\begin{array} { r l } & { I ( Z _ { t o p o } ; Y \mid Z _ { s t a t } ) \geq H ( Y \mid Z _ { s t a t } ) } \\ & { \phantom { a a a a a a a a a a a a a a a a a a a } + \mathbb { E } \log q _ { \phi } ( Y \mid Z _ { s t a t } , Z _ { t o p o } ) . } \end{array}\tag{20}
$$

Proof. Let $Z ~ = ~ ( Z _ { s t a t } , Z _ { t o p o } )$ . The decoder’s expected crossentropy decomposes as

$$
\begin{array} { r l } & { \mathbb { E } [ - \log q _ { \phi } ( Y \mid Z ) ] } \\ & { \quad = H ( Y \mid Z ) + \mathbb { E } _ { Z } { \mathrm { K L } } ( p ( Y \mid Z ) \| q _ { \phi } ( Y \mid Z ) ) . } \end{array}\tag{21}
$$

Nonnegative KL divergence bounds the conditional entropy by cross-entropy. Substitution into $I ( Z _ { t o p o } ; Y \mid Z _ { s t a t } ) \ = \ H ( Y \mid$ $Z _ { s t a t } ) \mathrm { ~ - ~ } H ( Y \mid Z )$ gives Eq. (20). The first term is constant during Stage 2, so maximizing this lower bound is equivalent to minimizing the expected negative log likelihood.

Increasing the bound does not itself guarantee an increase in true conditional mutual information. The graph representation can also encode omitted appearance statistics. Equation (16) supplies a separate spatial constraint, while coordinate-free residuals, graph perturbations and anatomical tests assess the interpretation empirically.

## 5. Experiments

The experiments test the argument at three levels. Controlled bags isolate composition from spatial labels, WSI benchmarks measure predictive utility, and matched ablations and perturbations examine the source of the gains. This order separates successful prediction from evidence of neighborhood use.

## 5.1. Evaluation protocol

## 5.1.1. WSI datasets and prediction tasks

Nine public WSI benchmarks cover diagnostic classification and survival prediction. TCGA-NSCLC contains 1,053 slides, comprising 541 LUAD and 512 LUSC cases, for lung cancer subtyping. TCGA-BRCA contains 1,021 slides for invasive ductal versus invasive lobular carcinoma, contrasting ductal architecture with single-file infiltration patterns. BRACS uses 525 slides in three classes: benign, atypical and malignant [71]. PANDA contains 10,616 slides for Gleason grading, where gland formation and fused-gland architecture are relevant tissue features.

Five TCGA cohorts test survival prediction: KIRC (519 slides), KIRP (259), LUAD (516), STAD (441) and UCEC (537). The cohorts span diverse morphological factors without assuming that all prognostic information is spatial. LUAD includes lepidic, acinar, papillary, micropapillary and solid growth patterns. STAD includes gland diferentiation, structural abnormalities and signet-ring morphology, while UCEC includes variation in the glandular-to-solid component ratio. KIRC includes nuclear-grade variation, and KIRP includes papillary-core organization. Models predict discrete time-to-event risk over four non-overlapping intervals. CAMELYON-16 provides a separate localization evaluation using tumor masks and is not counted among the nine prediction benchmarks.

## 5.1.2. Controlled composition and spatial tasks

Spatial-MNIST-Bag isolates the training hypothesis without pathology-specific annotation. Each bag contains 50 digit instances represented by 512-dimensional pretrained ResNet-18 embeddings and assigned coordinates in [0 1]<sup>2</sup>. Dataset A samples coordinates independently from $U ( [ 0 , 1 ] ^ { 2 } )$ and labels a bag positive if it contains at least one digit $" 9 "$ . Coordinates therefore provide no systematic label information.

Dataset B matches the digit-composition distribution across classes and varies their spatial arrangement. Each bag contains one each of $^ { \bullet } 1 , 3 , 5 , 7 , 9 ^ { , 9 }$ and 45 even digits sampled from {0 2 4 6 8}. Background digits and individual embeddings vary across bags according to the same class-independent sampling rule. Positive bags draw a centroid $c \sim U ( [ 0 . 2 , 0 . 8 ] ^ { 2 } )$ and place the five odd digits around it using Gaussian noise with $\sigma = 0 . 0 5$ Background coordinates remain uniform. Negative bags scatter all 50 instances uniformly. The positive construction encourages a compact odd-digit neighborhood; Gaussian sampling does not guarantee a particular KNN clique.

## 5.1.3. Preprocessing, baselines and implementation

All comparisons within an encoder setting share patch features, patient-level splits and evaluation procedures. Patient-level stratification prevents slides from the same patient appearing in training, validation and test partitions. Tissue is identified using Otsu thresholding in HSV space and divided into nonoverlapping $2 5 6 \times 2 5 6$ patches at 20× magnification. The primary evaluation uses frozen 1024-dimensional UNI embeddings. UNI uses a ViT-Large architecture pretrained with DINOv2 on Mass-100K, comprising more than $\bar { 1 } 0 ^ { 8 }$ patches. A separate classification comparison uses frozen CTransPath features to assess consistency across encoders.

The classification and survival tables compare ResTopoMIL with AB-MIL, CLAM-SB, DS-MIL, TransMIL, ILRA-MIL, MHIM-MIL, DGR-MIL and 2DMambaMIL. DTFD-MIL is additionally included in the localization comparison. PatchGCN supplies a graph baseline in the coordinate-reassignment diagnostic. Optimizer family, preprocessing and evaluation protocols are matched within the reported comparisons.

The default model uses $K _ { p r o t o } = 3 2 , K _ { k n n } = 8 , m = 0 . 3$ and $\lambda = 1 . 0$ . Prototype initialization uses MiniBatch K-Means on a random subset of training patches; the learnable temperature starts at 1.0. After ten statistical warm-up epochs, refinement trains the graph branch for 30 classification epochs or 20 survival epochs. Optimization uses Lookahead with an Adam base, learning rate $2 \times 1 0 ^ { - 4 }$ , weight decay $1 0 ^ { - 4 }$ and cosine annealing. Experiments use PyTorch 1.13, PyTorch Geometric and one NVIDIA RTX 3090 with 24 GB memory.

## 5.1.4. Metrics and statistical analysis

Classification is assessed with accuracy, F1 and AUC; survival prediction uses the concordance index. Unless a caption states otherwise, tables report means and standard deviations over five matched runs. Representative trajectories and endpoints without run-level dispersion are identified separately. Bold and underlined values rank reported means rather than statistical significance.

Patient-cluster inference compares each primary UNI endpoint with the baseline having the highest original mean AUC or C-index. All slides and repeated appearances of a patient are grouped, with shared weights and method-label swaps across methods and runs. We use 10,000 bootstrap samples for marginal 95% confidence intervals and 100,000 permutations for twosided $p = ( b + 1 ) / ( B + 1 )$ . Holm adjustment covers the nine primary endpoints. These intervals quantify patient-sampling uncertainty, while seed standard deviations describe run-to-run variability.

![](images/93a0ffbdfa249b3cfc7026f128eab9b4ef2c87e726f0273d9e53517928268db2.jpg)

Table 1: Controlled composition and spatial tasks. Dataset A requires instance detection without useful coordinates; Dataset B matches digit composition and varies spatial arrangement. Bold and underlined AUCs denote the highest and second-highest reported values per task. Run-level dispersion is unavailable for these endpoints.
<table><tr><td>Method</td><td>AUC</td><td>A: Comp. B: Topo. AUC</td></tr><tr><td>AB-MIL</td><td>0.998</td><td>0.505</td></tr><tr><td>TransMIL</td><td>0.995</td><td>0.532</td></tr><tr><td>ResTopoMIL (Joint)</td><td>0.991</td><td>0.684</td></tr><tr><td>ResTopoMIL</td><td>0.994</td><td>0.987</td></tr></table>

## 5.2. Staged training recovers a controlled spatial rule

The toy tasks test whether the training schedule improves spatial learning while retaining appearance-based prediction. Dataset A rewards instance detection, whereas Dataset B removes a systematic digit-composition rule. Table 1 reports the resulting AUCs.

On Dataset B, staged ResTopoMIL achieved AUC 0.987, compared with 0.684 for the same architecture trained jointly. AB-MIL and TransMIL were near chance at 0.505 and 0.532. Both performed well on Dataset A, with AUCs of 0.998 and 0.995, respectively. ResTopoMIL retained AUC 0.994 on that composition task, although it did not exceed AB-MIL. The matchedarchitecture contrast isolates the schedule’s contribution to learning the controlled spatial label.

## 5.3. Predictive utility across WSI classification tasks

The four classification benchmarks test whether the intervention has value beyond synthetic bags. Table 2 compares aggregation methods under two frozen encoders, holding the feature source fixed within each block.

With UNI, ResTopoMIL achieved the highest mean F1 on all four datasets. Its AUC margins over the strongest baseline were 0.0166 on BRACS, 0.0117 on PANDA and 0.0024 on BRCA. On NSCLC, MHIM-MIL had slightly higher AUC, 0.9759 versus 0.9753, while ResTopoMIL had higher accuracy and F1. The method’s relative performance therefore depended on both task and metric.

With CTransPath, ResTopoMIL led all 12 reported dataset– metric combinations. AUC margins were 0.0135, 0.0053, 0.0075 and 0.0065 on BRACS, PANDA, NSCLC and BRCA, respectively. The model used 1.15 million trainable parameters, compared with 2.67 million for TransMIL and 4.35 million for DGR-MIL. Parameter counts describe model size and are not runtime measurements.

## 5.4. Survival prediction and patient-level reliability

The survival tasks evaluate transfer from diagnostic categories to time-to-event prediction. ResTopoMIL achieved the highest mean C-index on all five cohorts in Table 3. Gains were clearest on LUAD and STAD; the KIRC margin over the strongest baseline was only 0.0002.

Patient-cluster tests bounded the reliability of the primary predictive comparisons (Table 4). BRACS, PANDA, LUAD and STAD retained significance after Holm adjustment. The

Fig. 3: Gradient dynamics on BRCA and PANDA. Upper panels compare joint and staged optimization; the dashed vertical line marks the transition after ten warm-up epochs. The statistical stream is then frozen and graph training begins. Lower panels compare joint graph training with and without $\mathcal { L } _ { t e x t u r e } .$ . The plotted gradient norms are not separated into supervised and auxiliary components and provide descriptive optimization evidence.

remaining positive estimates had confidence intervals containing zero, while NSCLC had a small negative estimate. Across classification AUC and survival C-index, mean primary performance improved on eight of nine endpoints, with statistically supported gains confined to four.

## 5.5. Training schedule, residual capacity and design controls

The ablations test whether gains require the proposed training schedule and exceed added appearance capacity. Table 5 contains 28 settings on PANDA and BRCA, including branch, fusion, graph and hyperparameter controls. Figure 3 provides descriptive training traces for the optimization interpretation.

Training dynamics. In the upper panels, staged training switched from the statistical stream to the graph stream after ten epochs. The graph gradient rose when graph optimization began, while the statistical gradient became zero because that stream was frozen. Joint-training graph gradients generally declined earlier and were weaker after this transition. The lower panels specifically compared joint graph training with and without $\mathcal { L } _ { t e x t u r e } ;$ removing the constraint produced faster gradient decay. These curves do not separate supervised and auxiliary gradients, so they do not identify the source of every gradient increase.

Schedule beyond branch capacity. The staged model exceeded joint optimization in AUC by 0.0127 on PANDA and 0.0065 on BRCA. Neither branch alone matched the complete model’s AUC. Statistical dropout, curriculum scheduling, hard-instance mining and a tenfold graph learning rate narrowed the gap. Hardinstance mining was strongest among these alternatives, reaching 0.9372/0.9792 versus 0.9426/0.9838 for the default. Joint training nevertheless gave slightly higher BRCA F1, 0.9318 versus 0.9308. These comparisons support the schedule’s predictive contribution without establishing superiority on every metric.

Table 2: WSI classification with frozen UNI and CTransPath (CPath) features. Entries are mean±standard deviation over five runs. Params is the trainable parameter count in millions. Bold and underlined means mark the best and second-best distinct values within each encoder block; smaller parameter counts are preferred. Ties share formatting. Rankings do not establish significance.
<table><tr><td rowspan="2">Feature Method</td><td rowspan="2"></td><td rowspan="2">Params</td><td colspan="3">BRACS</td><td colspan="3">PANDA</td><td colspan="3">TCGA-NSCLC</td><td colspan="3">TCGA-BRCA</td></tr><tr><td></td><td></td><td>AUC</td><td></td><td></td><td>AUC</td><td></td><td>F1</td><td>AUC</td><td>Acc</td><td>F1</td><td>AUC</td></tr><tr><td rowspan="9"></td><td>AB-MIL</td><td>.59</td><td></td><td> $. 7 2 7 5 _ { \pm . 0 7 5 9 } . 6 7 4 8 _ { \pm . 0 1 8 5 } . 8 8 0 6 _ { \pm . 0 0 9 1 }$ </td><td></td><td> $. 7 3 2 2 _ { \pm . 0 0 5 9 }$ </td><td></td><td></td><td></td><td></td><td>.6876±.0079.9306±.0017.8988±.0066.8680±.0068.9569±.0073</td><td> $. 9 4 1 4 _ { \pm . 0 0 6 2 }$ </td><td> $. 9 1 8 7 _ { \pm . 0 1 0 6 }$ </td><td> $. 9 7 2 7 _ { \pm . 0 0 1 9 }$ </td></tr><tr><td>CLAM-SB</td><td>.79</td><td>.7371±.0182.7051±.0287</td><td></td><td> $\underline { { . 8 8 4 0 _ { \pm . 0 1 3 1 } } }$ </td><td></td><td></td><td></td><td></td><td></td><td></td><td>.7318±.0041.6814±.0050.9215±.0009.8836±.0085.8529±.0084.9635±.0064.9314±.0062</td><td> $\underline { { 9 2 6 8 } } _ { \pm . 0 1 1 2 }$ </td><td> $\underline { { 9 8 1 4 } } _ { \pm . 0 0 2 7 }$ </td></tr><tr><td>DS-MIL</td><td>1.20</td><td> $6 4 6 0 _ { \pm . 0 1 8 9 }$ </td><td>.6068±.0436.8054±.0225 .7394±.0207.6911±.0231.9309±.0070.8836±.0085 .8627±.0099.9579±.0083 .9409±.0124.9058±.0208</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td> $. 9 7 7 7 _ { \pm . 0 0 3 9 }$ </td></tr><tr><td>TransMIL</td><td>2.67</td><td> $6 5 0 6 _ { \pm . 0 1 7 4 }$ </td><td> $. 5 5 9 2 _ { \pm . 0 3 9 8 }$ </td><td> $. 8 4 5 0 _ { \pm . 0 0 9 8 }$ </td><td></td><td></td><td></td><td></td><td></td><td></td><td>.7090±.0088 .6667±.0133 .9288±.0038 .8933±.0132 .8822±.0132 .9692±.0084 .9409±.0095</td><td> $. 9 0 1 6 _ { \pm . 0 1 6 3 }$ </td><td> $. 9 7 8 7 _ { \pm . 0 1 5 4 }$ </td></tr><tr><td>ILRA-MIL</td><td>3.68</td><td> $6 2 3 0 _ { \pm . 0 2 8 6 }$ </td><td> $. 5 5 2 6 _ { \pm . 0 3 6 8 }$ </td><td> $. 8 0 1 2 _ { \pm . 0 2 1 9 }$ </td><td>.7402±.0096.6880±.0094.9266±.0034.8912±.0157.8829±.0159.9628±.0095</td><td></td><td></td><td></td><td></td><td></td><td> $. 9 4 5 5 _ { \pm . 0 0 5 1 }$ </td><td> $. 9 1 0 8 _ { \pm . 0 0 9 1 }$ </td><td> $. 9 6 2 1 _ { \pm . 0 1 4 8 }$ </td></tr><tr><td>MHIM-MIL</td><td>2.67</td><td> $6 6 9 0 _ { \pm . 0 3 5 8 }$ </td><td> $. 5 8 7 4 _ { \pm . 0 5 0 8 }$ </td><td> $. 8 3 4 0 _ { \pm . 0 1 2 8 }$ </td><td>.6970±.0123 .6372±.0158 .9155±.0020 .8908±.0183 .8703±.0183 .9759±.0030 .9465±.0095</td><td></td><td></td><td></td><td></td><td></td><td></td><td> $. 9 1 1 7 _ { \pm . 0 1 5 5 }$ </td><td> $. 9 7 6 9 _ { \pm . 0 1 2 9 }$ </td></tr><tr><td>DGR-MIL</td><td>4.35 1.27</td><td> $. 7 1 2 6 _ { \pm . 0 3 6 3 }$ </td><td> $. 6 6 8 4 _ { \pm . 0 5 0 9 }$ </td><td> $. 8 2 5 8 _ { \pm . 0 3 8 6 }$ </td><td>.6964±.0129 .6399±.0178 .9043±.0082 .9036±.0225 .8829±.0225 .9390±.0224 .9455±.0051</td><td></td><td></td><td></td><td></td><td></td><td></td><td> $. 9 0 9 9 _ { \pm . 0 0 9 8 }$ </td><td> $. 9 7 2 4 _ { \pm . 0 1 9 7 }$ </td></tr><tr><td>2DMambaMIL</td><td></td><td> $. 7 1 8 5 _ { \pm . 0 2 9 0 }$ </td><td> $. 6 7 1 0 _ { \pm . 0 4 5 0 }$ </td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>.8315±.0310 .7012±.0115 .6450±.0160 .9105±.0075 .9008±.0190.8855±.0210 .9450±.0180.9490±.0060</td><td> $. 9 1 5 0 _ { \pm . 0 1 1 0 }$ </td><td> $. 9 7 5 5 _ { \pm . 0 1 8 0 }$ </td></tr><tr><td>ResTopoMIL</td><td>1.15</td><td> $. 7 4 9 4 _ { \pm , 0 . 0 2 8 } < . 7 1 4 2 _ { \pm , 0 . 0 2 0 } , \ : \ : 9 0 0 6 _ { \pm , \infty , 0 . 5 } , \ : 7 5 4 6 _ { \pm , 0 . 0 9 4 } , \ : 7 0 9 7 _ { \pm , 0 0 8 } , \ : 9 4 2 6 _ { \pm , 0 0 1 0 } , \ : 9 1 5 7 _ { \pm , 0 0 8 } , \ : 9 1 3 5 _ { \pm , 0 0 8 7 } , \ : 9 7 5 3 _ { \pm , 0 0 2 9 } , \ : 9 5 6 8 _ { \pm , 0 0 9 5 } , \ : 9 3 0 8 _ { \pm , 0 1 6 6 }$ </td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td> $\mathbf { . 9 8 3 8 _ { \pm . 0 0 4 9 } }$ </td></tr><tr><td rowspan="9">CPath</td><td>AB-MIL</td><td>.59</td><td>.7250±.0210 .6415±.0225 .8512±.0105 .6854±.0092 .6652±.0085 .9185±.0042 .8810±.0075 .8510±.0082 .9412±.0085 .9250±.0075</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td> $. 9 0 1 0 _ { \pm . 0 1 1 5 }$ </td><td> $. 9 6 0 5 _ { \pm . 0 0 3 5 }$ </td></tr><tr><td>CLAM-SB</td><td>.79</td><td>.7185±.0195 .6120±.0215 .8580±.0125 .6820±.0088 .6610±.0065 .9150±.0038 .8855±.0092 .8580±.0095 .9485±.0075 .9150±.0085</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td> $\underline { { 9 6 8 5 } } _ { \pm . 0 0 4 5 }$ </td></tr><tr><td>DS-MIL</td><td>1.20</td><td>.7305±.0285 .6525±.0320 .8550±.0152 .6815±.0065 .6612±.0052 .9205±.0068 .8850±.0115 .8530±.0098 .9430±.0092 .8910±.0225</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td> $. 9 0 8 5 _ { \pm . 0 1 2 5 }$ </td><td> $9 6 2 2 _ { \pm . 0 0 4 1 }$ </td></tr><tr><td>TransMIL</td><td>2.67</td><td></td><td>.6150±.0185 .5255±.0415 .8125±.0128 .6855±.0105 .6420±.0150 .9024±.0055.8950±.0145 .8655±.0140 .9515±.0075 .9250±.0115</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td> $\underline { { 9 1 1 0 _ { \pm . 0 1 3 5 } } }$ </td><td></td></tr><tr><td>ILRA-MIL</td><td>3.68</td><td></td><td>.5900±.0250 .5105±.0385 .7850±.0210 .7255±.0080 .6710±.0115 .9120±.0045 .8820±.0165 .8710±.0162 .9485±.0082 .9310±.0065 .8950±.0105</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td> $. 8 8 5 0 _ { \pm . 0 1 7 5 }$ </td><td> $. 9 6 4 5 _ { \pm . 0 1 6 0 }$ </td></tr><tr><td>MHIM-MIL</td><td>2.67</td><td></td><td>.6350±.0380 .5510±.0520 .8015±.0165 .6750±.0135 .6155±.0185 .8985±.0035 .8910±.0195 .8550±.0195 .9610±.0048 .9310±.0105 .8950±.0165</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td> $. 9 5 2 5 _ { \pm . 0 1 1 5 }$   $9 6 5 8 _ { \pm . 0 1 4 2 }$ </td></tr><tr><td>DGR-MIL</td><td>4.35 1.27</td><td></td><td>.6800±.0350 .6350±.0485 .7950±.0305 .6720±.0145 .6185±.0195 .8845±.0095 .8850±.0235 .8650±.0245 .9220±.0250.8950±.0115 .8950±.0155</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td> $. 9 6 0 5 _ { \pm . 0 2 1 0 }$ </td></tr><tr><td>2DMambaMIL</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>.6905±.0285 .6420±.0465 .8080±.0255 .6255±.0185.6815±.0125 .8912±.0082 .8910±.0185 .8750±.0215 .9315±.0195 .9350±.0075 .9010±.0125</td><td></td><td></td><td> $. 9 6 4 0 _ { \pm . 0 1 8 5 }$ </td></tr><tr><td>ResTopoMIL</td><td>1.15</td><td></td><td>.7385±.0255 .6952±.0435 .8715±.0085 .7350±.0110 .6885±.0105 .9258±.0025 .9010±.0095 .8955±.0095 .9685±.0032 .9420±.0105 .9150±.0175</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td> $\mathbf { \sigma } _ { \mathbf { \sigma } _ { 2 } 7 5 0 _ { \pm . 0 0 5 2 } }$ </td></tr></table>

Table 3: Survival prediction results. Each cell reports mean±std concordance index over five random seeds on five TCGA cohorts. Higher is better. Bold and underlined means denote the best and second-best values per cohort, respectively.
<table><tr><td>Method</td><td>KIRC</td><td>KIRP</td><td>LUAD</td><td>STAD</td><td>UCEC</td></tr><tr><td>AB-MIL</td><td> $0 . 5 6 9 4 _ { \pm . 0 2 7 9 }$ </td><td> $0 . 7 0 9 1 _ { \pm . 0 5 9 2 }$ </td><td> $0 . 5 9 4 2 _ { \pm . 0 1 2 0 }$ </td><td> $0 . 5 8 7 1 _ { \pm . 0 0 9 0 }$ </td><td> $0 . 6 2 2 0 _ { \pm . 0 2 4 1 }$ </td></tr><tr><td>CLAM-SB</td><td> $0 . 5 7 0 5 _ { \pm . 0 1 0 1 }$ </td><td> $0 . 7 0 3 4 _ { \pm . 0 5 6 5 }$ </td><td> $0 . 6 1 9 2 _ { \pm . 0 1 0 8 }$ </td><td> $0 . 5 8 2 9 _ { \pm . 0 2 4 3 }$ </td><td> $0 . 6 0 3 4 _ { \pm . 0 3 9 8 }$ </td></tr><tr><td>DS-MIL</td><td> $0 . 5 8 3 2 _ { \pm . 0 2 0 7 }$ </td><td> $0 . 6 8 7 5 _ { \pm . 0 4 5 6 }$ </td><td> $0 . 5 6 6 9 _ { \pm . 0 1 4 5 }$ </td><td> $0 . 6 0 9 0 _ { \pm . 0 1 8 7 }$ </td><td> $0 . 6 3 6 8 _ { \pm . 0 4 1 7 }$ </td></tr><tr><td>TransMIL</td><td> $0 . 5 6 7 7 _ { \pm . 0 1 6 2 }$ </td><td> $0 . 7 0 0 0 _ { \pm . 0 2 3 8 }$ </td><td> $0 . 6 1 4 8 _ { \pm . 0 1 2 4 }$ </td><td> $0 . 6 2 0 4 _ { \pm . 0 3 5 9 }$ </td><td> $0 . 6 5 9 5 _ { \pm . 0 4 8 6 }$ </td></tr><tr><td>ILRA-MIL</td><td> $0 . 7 2 7 6 _ { \pm . 0 2 7 1 }$ </td><td> $0 . 7 2 7 2 _ { \pm . 1 3 2 2 }$ </td><td> $0 . 6 0 6 7 _ { \pm . 0 1 5 6 }$ </td><td> $0 . 6 5 8 9 _ { \pm . 0 3 5 4 }$ </td><td> $0 . 6 7 6 7 _ { \pm . 0 3 0 8 }$ </td></tr><tr><td>MHIM-MIL</td><td> $0 . 7 1 7 0 _ { \pm . 0 1 4 3 }$ </td><td> $\underline { { 0 . 8 0 7 3 _ { \pm . 0 5 4 5 } } }$ </td><td> $0 . 5 4 9 1 _ { \pm . 0 2 0 5 }$ </td><td> $0 . 6 3 5 7 _ { \pm . 0 3 2 1 }$ </td><td> $0 . 6 4 7 4 _ { \pm . 0 7 0 4 }$ </td></tr><tr><td>DGR-MIL</td><td> $0 . 7 2 9 8 _ { \pm . 0 0 6 8 }$ </td><td> $0 . 8 0 7 1 { \scriptstyle \pm . 0 2 6 5 }$ </td><td> $0 . 6 2 4 5 _ { \pm . 0 1 1 7 }$ </td><td> $\underline { { 0 . 6 6 2 5 } } _ { \pm . 0 1 2 2 }$ </td><td> $0 . 6 9 7 6 _ { \pm . 0 1 0 4 }$ </td></tr><tr><td>2DMambaMIL</td><td> $\underline { { 0 . 7 3 1 1 } } { \scriptstyle \pm . 0 1 1 0 }$ </td><td> $0 . 8 0 2 7 _ { \pm . 0 2 5 0 }$ </td><td> $\underline { { 0 . 6 2 9 0 } } _ { \pm . 0 0 9 5 }$ </td><td> $0 . 6 5 1 5 _ { \pm . 0 1 5 0 }$ </td><td> $\underline { { 0 . 7 0 2 0 } } { \scriptstyle \pm . 0 1 4 0 }$ </td></tr><tr><td>ResTopoMIL</td><td> $\mathbf { 0 . 7 3 1 3 _ { \pm . 0 1 0 4 } }$ </td><td> $\mathbf { 0 . 8 1 8 2 _ { \pm . 0 2 2 7 } }$ </td><td> $\mathbf { 0 . 6 4 5 7 { \scriptstyle \pm . 0 0 7 0 } }$ </td><td> $\mathbf { 0 . 6 8 0 7 _ { \pm . 0 0 8 6 } }$ </td><td> $\mathbf { 0 . 7 0 5 8 _ { \pm . 0 1 2 8 } }$ </td></tr></table>

Table 4: Paired patient-cluster inference against the strongest original UNI baseline. ∆ is ResTopoMIL minus comparator. Confidence intervals are marginal 95% intervals; $p _ { H }$ is Holm-adjusted across nine endpoints.
<table><tr><td>Endpoint</td><td>Comparator</td><td>Δ</td><td>95% CI</td><td>PH</td></tr><tr><td>BRACS AUC</td><td> $\mathrm { C L A M } { \cdot } \mathrm { S B }$ </td><td>+0.0166</td><td>[0.0082,0.0249]</td><td>0.0025</td></tr><tr><td>PANDA AUC</td><td>DS-MIL</td><td>+0.0117</td><td>[0.0063,0.0170]</td><td>0.0016</td></tr><tr><td>NSCLC AUC</td><td>MHIM-MIL</td><td>-0.0006</td><td>[-0.0024,0.0012]</td><td>1.0000</td></tr><tr><td>BRCA AUC</td><td>CLAM-SB</td><td>+0.0024</td><td>[-0.0015,0.0061]</td><td>0.2582</td></tr><tr><td>KIRC C-index</td><td>2DMambaMIL</td><td>+0.0002</td><td>[-0.0062,0.0067]</td><td>1.0000</td></tr><tr><td>KIRP C-index</td><td>MHIM-MIL</td><td>+0.0109</td><td>[-0.0240,0.0460]</td><td>1.0000</td></tr><tr><td>LUAD C-index</td><td>2DMambaMIL</td><td>+0.0167</td><td>[0.0113,0.0221]</td><td>&lt;0.001</td></tr><tr><td>STAD C-index</td><td>DGR-MIL</td><td>+0.0182</td><td>[0.0108,0.0255]</td><td>&lt;0.001</td></tr><tr><td>UCEC C-index</td><td>2DMambaMIL</td><td>+0.0038</td><td>[-0.0049,0.0121]</td><td>0.9580</td></tr></table>

Fusion and graph design. Additive prediction gave higher PANDA AUC and higher accuracy on both datasets than the tested fusion alternatives. However, MoE, independent classifiers and $\mathrm { C o n c a t + M L P }$ achieved higher BRCA AUCs, respectively 0.9897, 0.9845 and 0.9862 versus 0.9838. Replacing GCN with GAT gave similar PANDA AUC and higher BRCA AUC, 0.9419/0.9889. Training on random graphs reduced AUC to 0.8286/0.8563, while fixing the prototypes also reduced performance. The intervention was not restricted to GCN, and its default fusion was not universally optimal.

Coordinate-specific constraint and sensitivity. Removing $\mathcal { L } _ { t e x t u r e }$ reduced PANDA/BRCA AUC by 0.0279/0.0076 (Panel D). Across the prototype-count, neighbor-count and margin sweeps, PANDA AUC ranged over 0.9285–0.9426, 0.9312– 0.9426 and 0.9325–0.9426. The defaults were not uniformly best: margin 0.4 gave BRCA AUC 0.9898, compared with 0.9838. The sweeps varied one parameter at a time and did not test interactions or loss-weight sensitivity.

## 5.6. Prediction depends on the supplied neighborhoods

Appearance capacity beyond the anchor. Coordinate-free residuals improved prediction even without physical neighborhoods (Panel H). They shared anchor logits, UNI features, five patientlevel splits, optimizer family and Stage-2 budget. A parametermatched two-layer node MLP with mean pooling increased PANDA/BRCA AUC by 0.0086/0.0111 over the anchor. An AB-MIL residual reached 0.9289/0.9752, remaining 0.0137/0.0086 below the full model. The anchor therefore left usable appearance information, but these controls did not recover all graphmodel gains.

Coordinate perturbations test whether the trained predictor uses spatial inputs. Patch embeddings remain fixed, coordinate assignments are permuted, and coordinate-derived inputs are rebuilt without retraining. Figure 4 reports progressive perturbations, while Table 6 retains endpoint comparisons for 2DMambaMIL and PatchGCN.

Performance generally declined, with local fluctuations, as shufling increased on BRCA, PANDA, KIRP and LUAD. On PANDA, complete reassignment reduced 2DMambaMIL/PatchGCN AUC by 0.0073/0.0147, compared with 0.0374 for ResTopoMIL under identical embeddings and permutations. Mean absolute probability changes were 0.0090/0.0170 versus 0.0390; label-flip rates were 1.8%/3.2% versus 7.8%. Larger responses indicated stronger dependence on the supplied arrangement, rather than a ranking of clinical reasoning.

Table 5: Consolidated ablation on PANDA and TCGA-BRCA with UNI features. Reported dispersion is standard deviation over five matched runs. All metrics are higher-is-better; bold and underlined values denote the highest and second-highest distinct reported means in each column, respectively, without implying significance. The default uses $K _ { p r o t o } = 3 2 , K _ { k n n } = 8 ,$ , and m = 0 3; each sweep changes only the named parameter.
<table><tr><td rowspan="2">Variant</td><td colspan="3">PANDA</td><td colspan="3">TCGA-BRCA</td></tr><tr><td>Accuracy</td><td>F1</td><td>AUC</td><td>Accuracy</td><td>F1</td><td>AUC</td></tr><tr><td colspan="7">Reference model</td></tr><tr><td>ResTopoMIL (default)</td><td> $\underline { { 0 . 7 5 4 6 _ { \pm 0 . 0 0 9 4 } } }$ </td><td> $\underline { { 0 . 7 0 9 7 _ { \pm 0 . 0 0 9 8 } } }$ </td><td> $\mathbf { 0 . 9 4 2 6 _ { \pm 0 . 0 0 1 0 } }$ </td><td> $\mathbf { 0 . 9 5 6 8 _ { \pm 0 . 0 0 9 5 } }$ </td><td> $0 . 9 3 0 8 _ { \pm 0 . 0 1 6 6 }$ </td><td> $0 . 9 8 3 8 _ { \pm 0 . 0 0 4 9 }$ </td></tr><tr><td colspan="7">A. Branches and optimization</td></tr><tr><td>Statistical only</td><td> $0 . 6 7 6 1 _ { \pm 0 . 0 0 6 2 }$ </td><td> $0 . 6 1 1 3 _ { \pm 0 . 0 0 7 5 }$ </td><td> $0 . 9 0 2 7 _ { \pm 0 . 0 0 5 1 }$ </td><td> $0 . 9 1 3 8 _ { \pm 0 . 0 0 6 9 }$ </td><td> $0 . 8 8 9 6 _ { \pm 0 . 0 1 1 2 }$ </td><td> $0 . 9 4 8 6 _ { \pm 0 . 0 0 3 5 }$ </td></tr><tr><td>Topological only</td><td> $0 . 7 0 4 2 _ { \pm 0 . 0 0 7 2 }$ </td><td> $0 . 6 5 9 8 _ { \pm 0 . 0 0 8 5 }$ </td><td> $0 . 9 2 1 5 _ { \pm 0 . 0 0 5 5 }$ </td><td> $0 . 9 3 8 2 _ { \pm 0 . 0 0 6 8 }$ </td><td> $0 . 8 9 4 5 _ { \pm 0 . 0 1 0 2 }$ </td><td> $0 . 9 6 0 8 _ { \pm 0 . 0 0 4 8 }$ </td></tr><tr><td>Joint optimization</td><td> $0 . 7 3 9 7 _ { \pm 0 . 0 0 5 5 }$ </td><td> $0 . 7 0 2 5 _ { \pm 0 . 0 0 6 5 }$ </td><td> $0 . 9 2 9 9 _ { \pm 0 . 0 0 5 2 }$ </td><td> $0 . 9 4 3 2 _ { \pm 0 . 0 0 5 5 }$ </td><td> $\mathbf { 0 . 9 3 1 8 _ { \pm 0 . 0 0 6 1 } }$ </td><td> $0 . 9 7 7 3 _ { \pm 0 . 0 0 2 2 }$ </td></tr><tr><td>Joint + statistical dropout</td><td> $0 . 7 4 1 5 _ { \pm 0 . 0 0 7 0 }$ </td><td> $0 . 7 0 3 8 _ { \pm 0 . 0 0 7 5 }$ </td><td> $0 . 9 3 2 5 _ { \pm 0 . 0 0 6 5 }$ </td><td>0.9448±0.0048</td><td> $0 . 9 3 0 2 _ { \pm 0 . 0 0 6 8 }$ </td><td> $0 . 9 7 8 0 _ { \pm 0 . 0 0 3 0 }$ </td></tr><tr><td>Joint + curriculum schedule</td><td> $0 . 7 4 4 2 _ { \pm 0 . 0 0 5 2 }$ </td><td> $0 . 7 0 5 5 _ { \pm 0 . 0 0 5 8 }$ </td><td> $0 . 9 3 4 8 _ { \pm 0 . 0 0 4 5 }$ </td><td> $0 . 9 4 6 5 _ { \pm 0 . 0 0 4 5 }$ </td><td> $0 . 9 3 1 5 _ { \pm 0 . 0 0 5 5 }$ </td><td> $0 . 9 7 8 5 _ { \pm 0 . 0 0 1 8 }$ </td></tr><tr><td>Joint + hard-instance mining</td><td> $0 . 7 4 8 5 _ { \pm 0 . 0 0 6 2 }$ </td><td> $0 . 7 0 9 1 _ { \pm 0 . 0 0 6 5 }$ </td><td> $0 . 9 3 7 2 _ { \pm 0 . 0 0 5 0 }$ </td><td>0.9492±0.0052</td><td> $0 . 9 2 8 8 _ { \pm 0 . 0 0 6 5 }$ </td><td> $0 . 9 7 9 2 _ { \pm 0 . 0 0 2 1 }$ </td></tr><tr><td>Joint optimization (multi-LR)</td><td> $0 . 7 4 6 8 _ { \pm 0 . 0 0 5 1 }$ </td><td> $0 . 7 0 8 4 _ { \pm 0 . 0 0 4 5 }$ </td><td> $0 . 9 3 5 2 _ { \pm 0 . 0 0 4 8 }$ </td><td> $0 . 9 4 7 5 _ { \pm 0 . 0 0 4 2 }$ </td><td> $0 . 9 2 3 5 _ { \pm 0 . 0 0 5 4 }$ </td><td> $0 . 9 7 8 6 _ { \pm 0 . 0 0 1 8 }$ </td></tr><tr><td colspan="7">B. Fusion alternatives</td></tr><tr><td>Gated fusion</td><td> $0 . 7 3 8 9 _ { \pm 0 . 0 0 6 8 }$ </td><td> $0 . 6 9 9 1 _ { \pm 0 . 0 0 7 5 }$ </td><td> $0 . 9 2 2 5 _ { \pm 0 . 0 0 6 2 }$ </td><td> $0 . 9 4 3 2 _ { \pm 0 . 0 0 5 1 }$ </td><td> $0 . 9 0 6 8 _ { \pm 0 . 0 0 6 5 }$ </td><td> $0 . 9 8 0 2 _ { \pm 0 . 0 0 4 2 }$ </td></tr><tr><td>Mixture of experts</td><td> $0 . 7 4 7 1 _ { \pm 0 . 0 0 5 2 }$ </td><td> $0 . 6 9 1 8 _ { \pm 0 . 0 0 4 8 }$ </td><td> $0 . 9 3 0 5 _ { \pm 0 . 0 0 3 5 }$ </td><td> $0 . 9 4 3 2 _ { \pm 0 . 0 0 4 5 }$ </td><td> $0 . 9 1 0 8 _ { \pm 0 . 0 0 5 5 }$ </td><td> $\underline { { 0 . 9 8 9 7 _ { \pm 0 . 0 0 2 8 } } }$ </td></tr><tr><td>Concat + MLP</td><td> $0 . 7 3 9 8 _ { \pm 0 . 0 0 6 9 }$ </td><td> $0 . 6 9 2 8 _ { \pm 0 . 0 0 7 4 }$ </td><td> $0 . 9 2 7 6 _ { \pm 0 . 0 0 5 8 }$ </td><td> $0 . 9 3 0 2 _ { \pm 0 . 0 0 5 7 }$ </td><td> $0 . 9 0 9 8 _ { \pm 0 . 0 0 8 3 }$ </td><td> $0 . 9 8 6 2 _ { \pm 0 . 0 0 3 1 }$ </td></tr><tr><td>Independent classifiers</td><td> $0 . 7 3 5 2 _ { \pm 0 . 0 0 8 5 }$ </td><td> $0 . 6 8 4 5 _ { \pm 0 . 0 0 8 1 }$ </td><td> $0 . 9 2 5 6 _ { \pm 0 . 0 0 6 2 }$ </td><td> $0 . 9 4 6 5 _ { \pm 0 . 0 0 5 8 }$ </td><td> $0 . 9 1 1 2 _ { \pm 0 . 0 0 9 5 }$ </td><td> $0 . 9 8 4 5 _ { \pm 0 . 0 0 2 5 }$ </td></tr><tr><td colspan="7">C. Graph and prototype controls</td></tr><tr><td>GAT backbone</td><td> $0 . 7 4 8 9 _ { \pm 0 . 0 0 6 2 }$ </td><td> $0 . 7 0 7 4 _ { \pm 0 . 0 0 5 8 }$ </td><td> $\underline { { 0 . 9 4 1 9 _ { \pm 0 . 0 0 4 1 } } }$ </td><td> $0 . 9 4 3 2 _ { \pm 0 . 0 0 5 2 }$ </td><td> $0 . 9 0 6 8 _ { \pm 0 . 0 0 6 1 }$ </td><td> $0 . 9 8 8 9 _ { \pm 0 . 0 0 2 5 }$ </td></tr><tr><td>Random graph (training control)</td><td> $0 . 6 8 1 5 _ { \pm 0 . 0 1 4 8 }$ </td><td> $0 . 5 3 6 2 _ { \pm 0 . 0 2 6 0 }$ </td><td> $0 . 8 2 8 6 _ { \pm 0 . 0 1 5 0 }$ </td><td> $0 . 8 5 2 3 _ { \pm 0 . 0 1 8 0 }$ </td><td> $0 . 6 7 4 9 _ { \pm 0 . 0 2 5 0 }$ </td><td> $0 . 8 5 6 3 _ { \pm 0 . 0 1 1 0 }$ </td></tr><tr><td>Fixed prototypes</td><td> $0 . 7 3 3 5 _ { \pm 0 . 0 0 6 5 }$ </td><td> $0 . 6 7 8 1 _ { \pm 0 . 0 0 7 2 }$ </td><td> $0 . 9 3 4 9 _ { \pm 0 . 0 0 5 5 }$ </td><td> $0 . 9 4 4 6 _ { \pm 0 . 0 0 5 6 }$ </td><td> $0 . 9 0 9 8 _ { \pm 0 . 0 0 9 4 }$ </td><td> $0 . 9 7 4 6 _ { \pm 0 . 0 0 3 5 }$ </td></tr><tr><td colspan="7">D. Coordinate-specific constraint</td></tr><tr><td>w/o Ltexture</td><td> $0 . 7 1 5 5 _ { \pm 0 . 0 0 7 2 }$ </td><td> $0 . 6 9 7 0 _ { \pm 0 . 0 0 8 1 }$ </td><td> $0 . 9 1 4 7 _ { \pm 0 . 0 0 6 5 }$ </td><td> $0 . 9 4 1 8 _ { \pm 0 . 0 0 5 4 }$ </td><td> $0 . 9 0 6 5 _ { \pm 0 . 0 0 9 8 }$ </td><td> $0 . 9 7 6 2 _ { \pm 0 . 0 0 3 8 }$ </td></tr><tr><td colspan="7">E. Prototype count</td></tr><tr><td> $K _ { p r o t o } = 8$ </td><td> $0 . 7 4 0 5 _ { \pm 0 . 0 0 7 1 }$ </td><td> $0 . 6 9 1 2 _ { \pm 0 . 0 0 6 3 }$ </td><td> $0 . 9 2 8 5 _ { \pm 0 . 0 0 5 1 }$ </td><td> $0 . 9 4 8 2 _ { \pm 0 . 0 0 5 5 }$ </td><td> $0 . 9 1 4 5 _ { \pm 0 . 0 0 8 9 }$ </td><td> $0 . 9 8 5 2 _ { \pm 0 . 0 0 2 1 }$ </td></tr><tr><td> $K _ { p r o t o } = 1 6$ </td><td> $0 . 7 5 1 2 _ { \pm 0 . 0 0 5 8 }$ </td><td> $0 . 7 0 6 5 _ { \pm 0 . 0 0 4 2 }$ </td><td> $0 . 9 3 9 2 _ { \pm 0 . 0 0 3 4 }$ </td><td> $0 . 9 5 1 5 _ { \pm 0 . 0 0 3 8 }$ </td><td> $0 . 9 2 2 4 _ { \pm 0 . 0 0 6 1 }$ </td><td> $0 . 9 8 8 9 _ { \pm 0 . 0 0 1 5 }$ </td></tr><tr><td> $K _ { p r o t o } = 6 4 $ </td><td> $0 . 7 5 4 4 _ { \pm 0 . 0 0 5 2 }$ </td><td> $\mathbf { 0 . 7 0 9 8 _ { \pm 0 . 0 0 4 8 } }$ </td><td> $0 . 9 4 1 5 _ { \pm 0 . 0 0 3 1 }$ </td><td> $0 . 9 5 2 8 _ { \pm 0 . 0 0 4 2 }$ </td><td> $0 . 9 2 4 5 _ { \pm 0 . 0 0 6 5 }$ </td><td> $0 . 9 8 9 2 _ { \pm 0 . 0 0 1 8 }$ </td></tr><tr><td colspan="7"> $F . ~ G r a p h ~ d e g r e e$ </td></tr><tr><td> $K _ { k n n } = 4$ </td><td> $0 . 7 4 3 8 _ { \pm 0 . 0 0 6 2 }$ </td><td> $0 . 6 9 5 5 _ { \pm 0 . 0 0 5 9 }$ </td><td> $0 . 9 3 1 2 _ { \pm 0 . 0 0 4 5 }$ </td><td> $0 . 9 5 0 2 _ { \pm 0 . 0 0 4 9 }$ </td><td> $0 . 9 1 9 2 _ { \pm 0 . 0 0 7 1 }$ </td><td> $0 . 9 8 7 5 _ { \pm 0 . 0 0 1 9 }$ </td></tr><tr><td> $K _ { k n n } = 1 6$ </td><td> $0 . 7 4 9 5 _ { \pm 0 . 0 0 5 5 }$ </td><td> $0 . 7 0 2 4 _ { \pm 0 . 0 0 5 1 }$ </td><td> $0 . 9 3 6 8 _ { \pm 0 . 0 0 3 7 }$ </td><td> $0 . 9 5 1 8 _ { \pm 0 . 0 0 4 4 }$ </td><td> $0 . 9 2 1 8 _ { \pm 0 . 0 0 6 8 }$ </td><td> $0 . 9 8 8 2 _ { \pm 0 . 0 0 2 2 }$ </td></tr><tr><td colspan="7"> $G . S h u f f e m a r g i n$ </td></tr><tr><td>Margin m = 0.1</td><td> $0 . 7 4 6 2 _ { \pm 0 . 0 0 6 0 }$ </td><td> $0 . 6 9 8 8 _ { \pm 0 . 0 0 5 5 }$ </td><td> $0 . 9 3 2 5 _ { \pm 0 . 0 0 4 8 }$ </td><td> $0 . 9 5 1 0 _ { \pm 0 . 0 0 4 0 }$ </td><td> $0 . 9 2 0 5 _ { \pm 0 . 0 0 6 2 }$ </td><td> $0 . 9 8 8 0 _ { \pm 0 . 0 0 1 8 }$ </td></tr><tr><td> $\mathbf { M a r g i n } m = 0 . 2$ </td><td> $0 . 7 5 3 5 _ { \pm 0 . 0 0 4 8 }$ </td><td> $0 . 7 0 8 2 _ { \pm 0 . 0 0 4 1 }$ </td><td> $0 . 9 4 0 8 _ { \pm 0 . 0 0 3 5 }$ </td><td> $0 . 9 5 3 2 _ { \pm 0 . 0 0 3 5 }$ </td><td> $0 . 9 2 4 8 _ { \pm 0 . 0 0 5 5 }$ </td><td> $0 . 9 8 9 6 _ { \pm 0 . 0 0 1 4 }$ </td></tr><tr><td> $\mathbf { M a r g i n } m = 0 . 4$ </td><td> $\mathbf { 0 . 7 5 4 8 _ { \pm 0 . 0 0 4 9 } }$ </td><td> $0 . 7 0 9 4 _ { \pm 0 . 0 0 4 2 }$ </td><td> $0 . 9 4 1 7 _ { \pm 0 . 0 0 3 3 }$ </td><td> $\underline { { 0 . 9 5 3 6 _ { \pm 0 . 0 0 3 6 } } }$ </td><td> $0 . 9 2 5 1 _ { \pm 0 . 0 0 5 3 }$ </td><td> $\mathbf { 0 . 9 8 9 8 _ { \pm 0 . 0 0 1 4 } }$ </td></tr><tr><td> $\mathbf { M a r g i n } m = 0 . 5$ </td><td> $0 . 7 5 0 5 _ { \pm 0 . 0 0 5 2 }$ </td><td>0.7042±0.0049</td><td> $0 . 9 3 8 5 _ { \pm 0 . 0 0 3 9 }$ </td><td> $0 . 9 5 2 5 _ { \pm 0 . 0 0 3 8 }$ </td><td> $0 . 9 2 3 0 _ { \pm 0 . 0 0 5 8 }$ </td><td> $0 . 9 8 9 0 _ { \pm 0 . 0 0 1 6 }$ </td></tr><tr><td colspan="7">H. Coordinate-free residuals and inference corruption</td></tr><tr><td>Parameter-matched DeepSets residual</td><td> $0 . 6 9 0 4 _ { \pm 0 . 0 0 7 3 }$ </td><td> $0 . 6 2 8 0 _ { \pm 0 . 0 0 8 6 }$ </td><td> $0 . 9 1 1 3 _ { \pm 0 . 0 0 4 3 }$ </td><td> $0 . 9 2 3 4 _ { \pm 0 . 0 0 6 5 }$ </td><td> $0 . 9 0 0 7 _ { \pm 0 . 0 0 9 5 }$ </td><td> $0 . 9 5 9 7 _ { \pm 0 . 0 0 3 2 }$ </td></tr><tr><td>AB-MIL residual</td><td> $0 . 7 3 8 5 _ { \pm 0 . 0 0 6 1 }$ </td><td> $0 . 6 9 3 1 _ { \pm 0 . 0 0 7 0 }$ </td><td> $0 . 9 2 8 9 _ { \pm 0 . 0 0 3 2 }$ </td><td> $0 . 9 4 7 2 _ { \pm 0 . 0 0 5 8 }$ </td><td> $0 . 9 1 8 4 _ { \pm 0 . 0 0 8 5 }$ </td><td> $0 . 9 7 5 2 _ { \pm 0 . 0 0 2 7 }$ </td></tr><tr><td>Default, shuffled graph at inference</td><td> $0 . 6 8 2 9 _ { \pm 0 . 0 0 6 0 }$ </td><td> $0 . 6 1 6 8 _ { \pm 0 . 0 0 7 1 }$ </td><td> $0 . 9 0 5 2 _ { \pm 0 . 0 0 2 9 }$ </td><td> $0 . 9 1 5 5 _ { \pm 0 . 0 0 6 2 }$ </td><td> $0 . 8 9 1 9 _ { \pm 0 . 0 0 8 9 }$ </td><td> $0 . 9 5 0 9 _ { \pm 0 . 0 0 3 7 }$ </td></tr></table>

Table 6: Spatial-baseline AUC before and after complete coordinate reassignment with fixed patch embeddings. Clean and Shuf. denote original and shufled endpoints; $\Delta = { \mathrm { S h u f } } .$ − Clean. Reported endpoints have no run-level dispersion and are not five-seed aggregate estimates.
<table><tr><td rowspan="2">Method</td><td colspan="3">BRACS</td><td colspan="3">PANDA</td><td colspan="3">NSCLC</td><td colspan="3">BRCA</td></tr><tr><td>Clean</td><td>Shuf.</td><td>Δ</td><td>Clean</td><td>Shuf.</td><td>Δ</td><td>Clean</td><td>Shuf.</td><td>Δ</td><td>Clean</td><td>Shuf.</td><td>Δ</td></tr><tr><td>2DMambaMIL</td><td>0.8315</td><td>0.8237</td><td>-0.0078</td><td>0.9105</td><td>0.9032</td><td>-0.0073</td><td>0.9450</td><td>0.9418</td><td>-0.0032</td><td>0.9755</td><td>0.9709</td><td>-0.0046</td></tr><tr><td>PatchGCN</td><td>0.8738</td><td>0.8520</td><td>-0.0218</td><td>0.9274</td><td>0.9127</td><td>-0.0147</td><td>0.9711</td><td>0.9642</td><td>-0.0069</td><td>0.9801</td><td>0.9700</td><td>-0.0101</td></tr></table>

PatchGCN rebuilds its graph; 2DMambaMIL rebuilds its coordinate-derived 2D sequence. ∆ measures sensitivity, not a performance ranking.

Five-run inference corruption confirmed the contrast with coordinate-free residual capacity. Shufling the full model’s graph reduced PANDA/BRCA AUC to 0.9052/0.9509 (Table 5, Panel H), close to the anchor’s 0.9027/0.9486. Coordinate-free controls in Figure 1(c) were nearly invariant because their efective inputs were unchanged. TransMIL’s padded positional grid also need not change under physical-coordinate reassignment [17].

The composition-only toy task provided a negative control for unnecessary spatial dependence. ResTopoMIL’s clean/shufled

![](images/494a525f261a2b52a144ed6a40495014026e5071ecfdd32f242c36220bbab521.jpg)

![](images/38afa454680dc09d708a7ef8ac362b49da449556db70a85776c6405814217612.jpg)

![](images/55ff2741cb022dedd5ad830dc27bc4a43a09044ef89fa6211a53bbbb2d61efd4.jpg)

![](images/9221fed9997996e135da6e9941520281b4d1dbc5e1feed4b6d7f39edf4293715.jpg)  
Fig. 4: Progressive coordinate shufling with fixed patch embeddings. Panels report AUC on TCGA-BRCA/PANDA and C-index on TCGA-KIRP/TCGA-LUAD. Curves are representative-seed trajectories, with no uncertainty bands; they are not five-seed aggregate estimates.

Table 7: PANDA neighborhood controls. Edge retention refers to original KNN edges; ∆ is perturbed minus clean AUC. Only AUC is ranked, with the best and second-best distinct values bold and underlined; retention and ∆ are diagnostics.
<table><tr><td>Perturbation</td><td>Retention</td><td>AUC</td><td>Δ</td></tr><tr><td>None</td><td>100%</td><td>0.9426</td><td>0.0000</td></tr><tr><td>Translation/rotation/reflection/scaling</td><td>100%</td><td>0.9426</td><td>0.0000</td></tr><tr><td>Jitter  $( 0 . 0 5 \times d _ { N N } )$ </td><td>98.8%</td><td>0.9425</td><td>-0.0001</td></tr><tr><td>Jitter  $( 0 . 1 0 \times d _ { N N } )$ </td><td>96.1%</td><td>0.9421</td><td>-0.0005</td></tr><tr><td>Degree-preserving rewiring, 25%</td><td>≈75%</td><td>0.9349</td><td>-0.0077</td></tr><tr><td>Degree-preserving rewiring, 50%</td><td>≈50%</td><td>0.9237</td><td>-0.0189</td></tr><tr><td>Degree-preserving rewiring, 100%</td><td>≈0%</td><td>0.9070</td><td>-0.0356</td></tr><tr><td>Complete coordinate permutation</td><td>≈0%</td><td>0.9052</td><td>-0.0374</td></tr></table>

Dataset A AUC was 0.9940/0.9938. Its mean absolute probability change was 0.0012, with 0.2% label flips. The model thus showed little perturbation response in this controlled setting without a spatial label signal.

## 5.7. Neighborhood-preserving and neighborhood-damaging controls

Preserving neighborhoods left PANDA performance nearly unchanged, while graph damage reduced it (Table 7). Translation, rotation, reflection and uniform scaling retained AUC 0.9426. Jitter at $0 . 1 0 \times d _ { N N }$ retained 96.1% of edges and produced AUC 0.9421. Here $d _ { N N }$ denotes the nearest-neighbor distance used to scale coordinate jitter.

Connectivity- and degree-preserving double-edge swaps within original components reduced AUC to 0.9349, 0.9237 and 0.9070 at 25%, 50% and 100% rewiring. Afected-node counts were controlled, with approximately matched edge-length and node-feature-similarity distributions. These controls associated prediction changes with neighborhood damage, while allowing higher-order or non-pathological correlations to contribute. Run-level dispersion was unavailable for these endpoints.

Table 8: CAMELYON-16 localization (upper panel) and post-hoc anatomystratified graph perturbation (lower panel). FROC uses the reported seven-point patch-grid summary. Masks stratify perturbations after training; higher values are better. Best and second-best values are bold and underlined within each panel; ties share formatting.
<table><tr><td>Method</td><td>Dice</td><td>Specificity</td><td>FROC</td></tr><tr><td>AB-MIL</td><td>0.412</td><td>0.985</td><td>0.3952</td></tr><tr><td>CLAM-SB</td><td>0.459</td><td>0.987</td><td>0.4257</td></tr><tr><td>DS-MIL</td><td>0.259</td><td>0.863</td><td>0.4506</td></tr><tr><td>TransMIL</td><td>0.103</td><td>0.999</td><td>0.4866</td></tr><tr><td>DTFD-MIL</td><td>0.525</td><td>0.999</td><td>0.4712</td></tr><tr><td>MHIM-MIL</td><td>0.548</td><td>0.992</td><td>0.4815</td></tr><tr><td>2DMambaMIL</td><td>0.475</td><td>0.995</td><td>0.4520</td></tr><tr><td>ResTopoMIL</td><td>0.624</td><td>0.999</td><td>0.5483</td></tr><tr><td>Graph condition</td><td>Slide AUC</td><td>Dice</td><td>FROC</td></tr><tr><td>Original</td><td>0.973</td><td>0.624</td><td>0.5483</td></tr><tr><td>Random-edge rewiring</td><td>0.944</td><td>0.581</td><td>0.5210</td></tr><tr><td>Normal-interior rewiring</td><td>0.950</td><td>0.593</td><td>0.5290</td></tr><tr><td>Tumor-interior rewiring</td><td>0.941</td><td>0.570</td><td>0.5120</td></tr><tr><td>Tumor-normal interface</td><td>0.921</td><td>0.522</td><td>0.4810</td></tr></table>

## 5.8. Localization and anatomy-stratified perturbations

CAMELYON-16 tested whether learned patch scores aligned with annotated anatomy [72]. Classifier training used slide labels alone. Oficial tumor masks were used for localization evaluation, validation-threshold selection and post-hoc perturbation stratification, without entering classifier inputs.

Patch scoring and localization protocol. Evaluation used 256 × 256 patches at 20×, excluding patches outside the Otsu tissue mask. A patch was positive when at least 1% of its area overlapped the tumor mask. ResTopoMIL scored each patch using its pre-pooling target-class residual logit, $s _ { i } = \mathbf { W } _ { t o p o , y } ^ { \top } \mathbf { \bar { H } } _ { i } ^ { ( 2 ) }$ . Baselines used their native attention or instance scores. All scores were min–max normalized within each slide, without smoothing, morphological refinement or a tumor-shape prior.

Each method’s Dice threshold maximized mean validation Dice and was then fixed for testing. Connected components in thresholded patch grids defined FROC candidates. A candidate was a true positive if its maximum-score patch lay inside an annotated tumor. The reported FROC summary averaged seven operating points: 1/8, 1/4, 1/2, 1, 2, 4 and 8 false positives per slide. This patch-grid summary difers from the oficial six-point CAMELYON-16 score, which excludes 1/8 [72].

Localization results. ResTopoMIL exceeded the strongest baseline by 0.076 Dice and 0.0617 FROC (Table 8). TransMIL and DTFD-MIL matched its specificity of 0.999, but had lower Dice. These comparisons measured patch-grid lesion coverage and background suppression; run-level uncertainty was unavailable. Figure 5 illustrates native heatmaps for four selected cases, including tumor-core concentration, boundary continuity, invasivefront patterns and background leakage. The cases complement mask-based metrics without estimating cohort-wide visual reliability.

Anatomical perturbation and score relevance. Post-hoc tumor masks stratified connectivity- and degree-preserving graph rewiring. Rewiring tumor–normal interfaces produced the largest AUC, Dice and FROC losses among the reported conditions (Table 8, lower panel). Removing the highest-scoring 10% of patches reduced slide AUC from 0.973 to 0.835, compared with 0.961 after matched-size random removal. Retaining only those patches gave AUC 0.903. These checks linked scores and neighborhood dependence to annotated anatomy, without identifying a unique causal mechanism or providing a full deletion curve.

![](images/85919018756c7e402c9425500385d376d739acdf1cad35a1731fb80ecc2a5813.jpg)  
(c) Case-03  
(d) Case-04  
Fig. 5: Qualitative H&E and native heatmap comparisons across five MIL methods. Warmer colors indicate higher within-slide normalized scores. Non-tissue patches are masked, with no smoothing, morphological refinement, or tumor-shape prior. The examples illustrate contiguity and background leakage, rather than segmentation accuracy. Case labels are internal to this figure.

## 5.9. Anchor convergence and patch-scale robustness

Longer warm-up did not materially alter the composition reference in the tested histogram family. Extending Stage 1 from ten to twenty epochs or using early stopping changed anchor AUC by at most 0.0002 and final AUC by 0.0001 (Table 9). This comparison addressed premature freezing within the chosen anchor, while the coordinate-free residuals showed that the anchor still omitted appearance information.

Across five matched runs, the anchor retained 91.0%±1.2% and 92.7%±1.0% of full-model AUC improvement over a class-prior predictor on PANDA and BRCA. The ratio was $( \mathrm { A U C } _ { s t a t } - . 5 ) / ( \mathrm { A U C } _ { f u l l } - . 5 )$ . The corresponding cross-entropy fraction, $( \mathrm { C E } _ { p r i o r } - \mathrm { C E } _ { s t a t } ) / ( \mathrm { C E } _ { p r i o r } - \mathrm { C E } _ { f u l l } )$ , was $8 6 . 9 \% \pm 2 . 0 \%$ and 91.4%±1.6%. These ratios describe normalized predictive improvements rather than a decomposition of label information.

Matched patch-scale comparisons retained the relative advantage over 2DMambaMIL on BRACS (Table 10). Patient-level splits, UNI extraction, training procedures and five seeds were shared. ResTopoMIL exceeded the comparator on all three mean metrics at both patch sizes. With 512 × 512 patches, its accuracy, F1 and AUC margins were 0.0250, 0.0350 and 0.0220. Both methods gained accuracy with larger patches, but field of view, instance count and graph granularity changed together.

Table 9: Warm-up schedule analysis. Cells give reported AUC without runlevel dispersion. Stat./Full denote the anchor and complete model. Bold and underlined AUCs mark the best and second-best distinct values per column; ties share formatting.
<table><tr><td colspan="3">PANDA</td><td colspan="2">BRCA</td></tr><tr><td>Schedule</td><td>Stat.</td><td>Full</td><td>Stat.</td><td>Full</td></tr><tr><td>10 epochs</td><td>0.9027</td><td>0.9426</td><td>0.9486</td><td>0.9838</td></tr><tr><td>20 epochs</td><td>0.9029</td><td>0.9425</td><td>0.9488</td><td>0.9839</td></tr><tr><td>Early stopping</td><td>0.9028</td><td>0.9427</td><td>0.9487</td><td>0.9838</td></tr></table>

Table 10: Three-class BRACS under matched patch-scale protocols. Entries are mean±standard deviation over five runs. Best and second-best means are bold and underlined within each patch size.
<table><tr><td>Patch Method</td><td></td><td>Accuracy</td><td>F1</td><td>AUC</td></tr><tr><td rowspan="2">256²</td><td>2DMambaMIL</td><td> $0 . 7 1 8 5 _ { \pm 0 . 0 2 9 0 }$ </td><td> $\underline { { 0 . 6 7 1 0 _ { \pm 0 . 0 4 5 0 } } }$ </td><td>0.8315±0.0310</td></tr><tr><td>ResTopoMIL</td><td> $\mathbf { 0 . 7 4 9 4 _ { \pm 0 . 0 2 8 6 } }$ </td><td>0.7142±0.0420 0.9006±0.0055</td><td></td></tr><tr><td rowspan="2">512²</td><td>2DMambaMIL</td><td> $\underline { { 0 . 7 4 2 0 _ { \pm 0 . 0 2 5 0 } } }$ </td><td> $\underline { { 0 . 6 8 6 0 _ { \pm 0 . 0 2 8 0 } } }$ </td><td> $\underline { { 0 . 8 8 4 0 _ { \pm 0 . 0 2 1 0 } } }$ </td></tr><tr><td>ResTopoMIL</td><td> $\mathbf { 0 . 7 6 7 0 _ { \pm 0 . 0 2 2 0 } }$ </td><td> $\mathbf { 0 . 7 2 1 0 _ { \pm 0 . 0 2 5 0 } }$ </td><td> $\mathbf { 0 . 9 0 6 0 _ { \pm 0 . 0 1 8 0 } }$ </td></tr></table>

![](images/ca9fb5d6b17ded47024f6d85a61b2db9ac56cf352e2caf58146f1fe8b4724547.jpg)  
Fig. 6: BRACS feature-space projections. The upper panel uses PCA and the lower uses t-SNE; each shows statistical-stream features above topologicalstream features across five reference splits. Colors denote classes, hollow circles denote statistically hard samples, and stars denote cases corrected by the topological stream. Class separation is descriptive evidence, rather than a test of coordinate dependence.

## 5.10. Representation diagnostics and computational scope

BRACS feature projections showed complementary class information across streams (Figure 6). PCA silhouette scores were approximately −0 05 to −0 09 for the statistical stream and 0 87– 0 89 for the graph stream. The graph stream’s Davies–Bouldin indices were 0 16–0 18, and its t-SNE silhouettes reached 0 92– 0 93. Stars identify cases corrected after statistical-stream errors. These two-dimensional projections describe class separation, while perturbation tests provide the direct evidence of coordinate dependence.

The training constraint requires an additional shufled-graph pass. Deployment constructs one intact graph, runs each stream once and adds their logits. No shufled view, test-time ensemble or patch annotation is required. Wall-clock time, peak-memory measurements and matched-compute comparisons were unavailable, so parameter counts alone do not establish computational eficiency.

## 6. Discussion

The central finding is that the training of a spatial pathway deserves separate evaluation from its architectural availability. The controlled task showed that changing the schedule could substantially improve learning of a spatial label within the same architecture. WSI comparisons established predictive value across several diagnostic and survival endpoints. Residual-capacity and neighborhood tests then bounded what could be attributed to spatial dependence. Together, these findings support fixing a composition reference before learning a graph correction.

The evidence also narrows the optimization interpretation. Proposition 2 gives a conditional explanation for weak supervised graph gradients when prediction error becomes small. It does not establish a universal composition-first learning trajectory. The gradient traces changed after graph training began, but the auxiliary loss also supplied gradients. Their increase cannot therefore be read as isolated recovery of spatially informative classification supervision. The matched toy and schedule comparisons provide the more direct evidence for the proposed training intervention.

Appearance capacity remains part of the residual’s contribution. Coordinate-free residuals improved over the histogram anchor, showing that warm-up convergence did not exhaust the appearance multiset’s predictive information. The full graph model exceeded those controls, and inference corruption reduced performance toward the anchor. Neighborhood-preserving transformations caused little change, whereas anatomical-interface rewiring produced larger losses. These observations connect prediction to supplied neighborhoods while allowing incidental correlations and unmodeled appearance information to contribute.

The approach is most relevant when spatial organization supplies information beyond composition. On a compositiondominant task, a strong appearance predictor may already be suficient, and a graph branch could fit incidental layout. Dataset A provided a controlled negative test, but realistic compositiondominant WSI cohorts remain underrepresented. Anchor capacity and training duration remain design choices. An underfit anchor leaves appearance errors, while overfitting may weaken the useful residual-learning signal. Warm-up stability was established only for the histogram family on PANDA and BRCA; adaptive schedules and stronger appearance references require separate evaluation.

Predictive and clinical scope remain limited by the evaluated data and measurements. Only four of nine primary comparisons remained significant after multiplicity adjustment, and several alternative configurations achieved higher BRCA AUC. All WSI studies were retrospective and used frozen encoders. CAMELYON-16 supplied patch-grid localization and post-hoc anatomical evidence, rather than a segmentation system or clinical validation. Prospective cohorts, jointly tuned encoders and measured computational cost would clarify the intervention’s practical limits. More realistic perturbations and compositiondominant WSI negative controls would further test when neighborhood use is beneficial.

## 7. Conclusion

ResTopoMIL addresses the gap between providing spatial inputs and learning to use them in whole-slide MIL. It fixes a composition predictor, trains an additive graph residual and separates intact from coordinate-reassigned representations. Controlled tasks support the training-schedule hypothesis, while WSI comparisons and perturbations connect predictive gains to neighborhood dependence. Coordinate-free controls retain appearance capacity as a contributing explanation. The results support explicit training interventions and behavioral spatial tests alongside architectural design.

## Author Contribution Statement

Xiangyu Li: Conceptualization, Methodology, Software, Investigation, Formal analysis, Visualization, Writing - original draft. Jincheng Li: Conceptualization, Methodology, Supervision, Writing - review & editing. Xing Xie: Data curation, Software, Validation. Bin Guan: Software, Validation. Shaohua Kevin Zhou: Resources, Validation, Supervision. Rui Yan: Formal analysis, Validation, Methodology, Resources. Ran Su: Conceptualization, Project administration, Supervision.

## Acknowledgments

This research is supported by Open Project Program of the State Key Laboratory of MedicalProteomics(SKLP-0202406), Emerging Frontiers CultivationProgram of Tianjin University Interdisciplinary Center,SeedFoundation of Tianjin University.

## Declaration

During the preparation of this work, the authors used the Qwen model in order to refine the manuscript. After using this tool/service, the author reviewed and edited the content as needed and takes full responsibility for the content of the published article.

## References

[1] T. G. Dietterich, R. H. Lathrop, T. Lozano-Pérez, Solving the multiple instance problem with axis-parallel rectangles, Artificial intelligence 89 (1- 2) (1997) 31–71.

[2] O. Maron, T. Lozano-Pérez, A framework for multiple-instance learning, Advances in neural information processing systems 10 (1997).

[3] G. Campanella, M. G. Hanna, L. Geneslaw, A. Miraflor, V. Werneck Krauss Silva, K. J. Busam, E. Brogi, V. E. Reuter, D. S. Klimstra, T. J. Fuchs, Clinical-grade computational pathology using weakly supervised deep learning on whole slide images, Nature medicine 25 (8) (2019) 1301– 1309.

[4] M. Y. Lu, D. F. Williamson, T. Y. Chen, R. J. Chen, M. Barbieri, F. Mahmood, Data-eficient and weakly supervised computational pathology on whole-slide images, Nature biomedical engineering 5 (6) (2021) 555–570.

[5] L. Pantanowitz, P. N. Valenstein, A. J. Evans, K. J. Kaplan, J. D. Pfeifer, D. C. Wilbur, L. C. Collins, T. J. Colgan, Review of the current state of whole slide imaging in pathology, Journal of pathology informatics 2 (1) (2011) 36.

[6] G. Verghese, J. K. Lennerz, D. Ruta, W. Ng, S. Thavaraj, K. P. Siziopikou, T. Naidoo, S. Rane, R. Salgado, S. E. Pinder, et al., Computationa pathology in cancer diagnosis, prognosis, and prediction–present day and prospects, The Journal of pathology 260 (5) (2023) 551–563.

[7] A. H. Song, G. Jaume, D. F. Williamson, M. Y. Lu, A. Vaidya, T. R. Miller, F. Mahmood, Artificial intelligence for digital and computational pathology, Nature Reviews Bioengineering 1 (12) (2023) 930–949.

[8] D. F. Quail, J. A. Joyce, Microenvironmental regulation of tumor progression and metastasis, Nature medicine 19 (11) (2013) 1423–1437.

[9] J. Cheng, Y. Liu, W. Huang, W. Hong, L. Wang, X. Zhan, Z. Han, D. Ni, K. Huang, J. Zhang, Computational image analysis identifies histopathological image features associated with somatic mutations and patient survival in gastric adenocarcinoma, Frontiers in Oncology 11 (2021) 623382.

[10] X. Wang, Y. Chen, Y. Gao, H. Zhang, Z. Guan, Z. Dong, Y. Zheng, J. Jiang, H. Yang, L. Wang, et al., Predicting gastric cancer outcome from resected lymph node histopathology images using deep learning, Nature communications 12 (1) (2021) 1637.

[11] M. Ilse, J. Tomczak, M. Welling, Attention-based deep multiple instance learning, in: International conference on machine learning, PMLR, 2018, pp. 2127–2136.

[12] M. Zaheer, S. Kottur, S. Ravanbakhsh, B. Poczos, R. R. Salakhutdinov, A. J. Smola, Deep sets, Advances in neural information processing systems 30 (2017).

[13] X. Chen, P. Qiu, W. Zhu, H. Wang, H. Li, X. Dong, X. Sun, X. Yu, Y. Wang, A. Razi, A. Sotiras, Cracking instance jigsaw puzzles: An alternative to multiple instance learning for whole slide image analysis, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 21353–21363.

[14] R. J. Chen, M. Y. Lu, M. Shaban, C. Chen, T. Y. Chen, D. F. Williamson, F. Mahmood, Whole slide images are 2d point clouds: Context-aware survival prediction using patch-based graph convolutional networks, in: International Conference on Medical Image Computing and Computer-Assisted Intervention, Springer, 2021, pp. 339–349.

[15] P. Pati, G. Jaume, A. Foncubierta-Rodriguez, F. Feroce, A. M. Anniciello, G. Scognamiglio, N. Brancati, M. Fiche, E. Dubruc, D. Riccio, et al., Hierarchical graph representations in digital pathology, Medical image analysis 75 (2022) 102264.

[16] M. Adnan, S. Kalra, H. R. Tizhoosh, Representation learning of histopathology images using graph neural networks, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops, 2020, pp. 988–989.

[17] Z. Shao, H. Bian, Y. Chen, Y. Wang, J. Zhang, X. Ji, et al., Transmil: Transformer based correlated multiple instance learning for whole slide image classification, Advances in neural information processing systems 34 (2021) 2136–2147.

[18] R. J. Chen, C. Chen, Y. Li, T. Y. Chen, A. D. Trister, R. G. Krishnan, F. Mahmood, Scaling vision transformers to gigapixel images via hierarchical self-supervised learning, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 16144–16155.

[19] S. Yang, Y. Wang, H. Chen, Mambamil: Enhancing long sequence modeling with sequence reordering in computational pathology, in: International conference on medical image computing and computer-assisted intervention, Springer, 2024, pp. 296–306.

[20] J. Zhang, A. T. Nguyen, X. Han, V. Q.-H. Trinh, H. Qin, D. Samaras, M. S. Hosseini, 2dmamba: Eficient state space model for image representation with applications on giga-pixel whole slide image classification, in: Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 3583–3592.

[21] H. Shah, K. Tamuly, A. Raghunathan, P. Jain, P. Netrapalli, The pitfalls of simplicity bias in neural networks, Advances in Neural Information Processing Systems 33 (2020) 9573–9585.

[22] R. Geirhos, P. Rubisch, C. Michaelis, M. Bethge, F. A. Wichmann, W. Brendel, Imagenet-trained cnns are biased towards texture; increasing shape bias improves accuracy and robustness, in: International conference on learning representations, 2018.

[23] M. Noroozi, P. Favaro, Unsupervised learning of visual representations by solving jigsaw puzzles, in: European conference on computer vision, Springer, 2016, pp. 69–84.

[24] M. Pezeshki, O. Kaba, Y. Bengio, A. C. Courville, D. Precup, G. Lajoie, Gradient starvation: A learning proclivity in neural networks, Advances in Neural Information Processing Systems 34 (2021) 1256–1272.

[25] Y. Xu, J.-Y. Zhu, I. Eric, C. Chang, M. Lai, Z. Tu, Weakly supervised histopathology cancer image segmentation and classification, Medical image analysis 18 (3) (2014) 591–604.

[26] O. Z. Kraus, J. L. Ba, B. J. Frey, Classifying and segmenting microscopy images with deep multiple instance learning, Bioinformatics 32 (12) (2016) i52–i59.

[27] K. Das, S. P. K. Karri, A. G. Roy, J. Chatterjee, D. Sheet, Classifying histopathology whole-slides using fusion of decisions from deep convolutional network on a collection of random multi-views at multi magnification, in: 2017 IEEE 14th International Symposium on Biomedical Imaging (ISBI 2017), IEEE, 2017, pp. 1024–1027.

[28] K. Das, S. Conjeti, A. G. Roy, J. Chatterjee, D. Sheet, Multiple instance learning of deep convolutional neural networks for breast histopathology whole slide classification, in: 2018 IEEE 15th International Symposium on Biomedical Imaging (ISBI 2018), IEEE, 2018, pp. 578–581.

[29] X. Wang, H. Chen, C. Gan, H. Lin, Q. Dou, E. Tsougenis, Q. Huang, M. Cai, P.-A. Heng, Weakly supervised deep learning for whole slide lung cancer image analysis, IEEE transactions on cybernetics 50 (9) (2019) 3950–3962.

[30] C.-L. Chen, C.-C. Chen, W.-H. Yu, S.-H. Chen, Y.-C. Chang, T.-I. Hsu, M. Hsiao, C.-Y. Yeh, C.-Y. Chen, An annotation-free whole-slide training

approach to pathological classification of lung cancer types using deep learning, Nature communications 12 (1) (2021) 1193.

[31] B. Li, Y. Li, K. W. Eliceiri, Dual-stream multiple instance learning network for whole slide image classification with self-supervised contrastive learning, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 14318–14328.

[32] H. Zhang, Y. Meng, Y. Zhao, Y. Qiao, X. Yang, S. E. Coupland, Y. Zheng, Dtfd-mil: Double-tier feature distillation multiple instance learning for histopathology whole slide image classification, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 18802–18812.

[33] W. Tang, S. Huang, X. Zhang, F. Zhou, Y. Zhang, B. Liu, Multiple instance learning framework with masked hard instance mining for whole slide image classification, in: Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 4078–4087.

[34] T. Lin, Z. Yu, H. Hu, Y. Xu, C.-W. Chen, Interventional bag multiinstance learning on whole-slide pathological images, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 19830–19839.

[35] Y. Zhang, H. Li, Y. Sun, S. Zheng, C. Zhu, L. Yang, Attention-challenging multiple instance learning for whole slide image classification, in: European conference on computer vision, Springer, 2024, pp. 125–143.

[36] Y. Zhang, H. Li, Y. Sun, Z. Shui, J. Li, C. Zhu, L. Yang, Aem: attention entropy maximization for multiple instance learning based whole slide image classification, in: International Conference on Medical Image Computing and Computer-Assisted Intervention, Springer, 2025, pp. 45–55.

[37] L. Qu, Y. Ma, X. Luo, Q. Guo, M. Wang, Z. Song, Rethinking multiple instance learning for whole slide image classification: A good instance classifier is all you need, IEEE Transactions on Circuits and Systems for Video Technology 34 (10) (2024) 9732–9744.

[38] D. Shao, R. J. Chen, A. H. Song, J. Runevic, M. Y. Lu, T. Ding, F. Mahmood, Do multiple instance learning models transfer?, Proceedings of Machine Learning Research 267 (2025) 54219–54238.

[39] W. Zhu, P. Qiu, X. Chen, Z. Yang, A. Sotiras, A. Razi, Y. Wang, How efective can dropout be in multiple instance learning?, in: International Conference on Machine Learning, PMLR, 2025, pp. 80090–80106.

[40] W. Zhu, P. Qiu, X. Chen, O. M. Dumitrascu, Y. Wang, Pdl: Regularizing multiple instance learning with progressive dropout layers, arXiv preprint arXiv:2308.10112 (2023).

[41] R. J. Chen, T. Ding, M. Y. Lu, D. F. Williamson, G. Jaume, B. Chen, A. Zhang, D. Shao, A. H. Song, M. Shaban, et al., A generalpurpose self-supervised model for computational pathology, arXiv preprint arXiv:2308.15474 (2023).

[42] M. Kang, H. Song, S. Park, D. Yoo, S. Pereira, Benchmarking selfsupervised learning on diverse pathology datasets, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 3344–3354.

[43] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, et al., Dinov2: Learning robust visual features without supervision, arXiv preprint arXiv:2304.07193 (2023).

[44] M. Caron, H. Touvron, I. Misra, H. Jégou, J. Mairal, P. Bojanowski, A. Joulin, Emerging properties in self-supervised vision transformers, in: Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 9650–9660.

[45] M. Y. Lu, B. Chen, A. Zhang, D. F. Williamson, R. J. Chen, T. Ding, L. P. Le, Y.-S. Chuang, F. Mahmood, Visual language pretrained multiple instance zero-shot transfer for histopathology images, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 19764–19775.

[46] S. Kapse, P. Pati, S. Yellapragada, S. Das, R. R. Gupta, J. Saltz, D. Samaras, P. Prasanna, Gecko: Gigapixel vision-concept contrastive pretraining in histopathology, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 20020–20030.

[47] J. Shi, C. Li, T. Gong, Y. Zheng, H. Fu, Vila-mil: Dual-scale visionlanguage multiple instance learning for whole slide image classification, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 11248–11258.

[48] M. Han, L. Qu, D. Yang, X. Zhang, X. Wang, L. Zhang, Mscpt: Fewshot whole slide image classification with multi-scale and context-focused prompt tuning, IEEE Transactions on Medical Imaging (2025).

[49] B. Wong, J. Kim, H. Fu, M. Y. Yi, Few-shot learning from gigapixel images

via hierarchical vision-language alignment and modeling, in: The Thirtyninth Annual Conference on Neural Information Processing Systems, 2025.

[50] D. Tomar, G. Vray, D. Mahapatra, S. Roy, J.-P. Thiran, B. Bozorgtabar, Slide-level prompt learning with vision language models for few-shot multiple instance learning in histopathology, in: 2025 IEEE 22nd International Symposium on Biomedical Imaging (ISBI), IEEE, 2025, pp. 1–5.

[51] J. Gou, L. Ji, P. Liu, M. Ye, Queryable prototype multiple instance learning with vision-language models for incremental whole slide image classification, in: Proceedings of the AAAI conference on artificial intelligence, Vol. 39, 2025, pp. 3158–3166.

[52] Z.-H. Zhou, Y.-Y. Sun, Y.-F. Li, Multi-instance learning by treating instances as non-iid samples, in: Proceedings of the 26th annual international conference on machine learning, 2009, pp. 1249–1256.

[53] D. Zhang, Y. Liu, L. Si, J. Zhang, R. Lawrence, Multiple instance learning on structured data, Advances in Neural Information Processing Systems 24 (2011).

[54] S. Pal, A. Valkanas, F. Regol, M. Coates, Bag graph: Multiple instance learning using bayesian graph neural networks, in: Proceedings of the AAAI conference on artificial intelligence, Vol. 36, 2022, pp. 7922–7930.

[55] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, I. Polosukhin, Attention is all you need, Advances in neural information processing systems 30 (2017).

[56] Y. Xiong, Z. Zeng, R. Chakraborty, M. Tan, G. Fung, Y. Li, V. Singh, Nyströmformer: A nyström-based algorithm for approximating self-attention, in: Proceedings of the AAAI conference on artificial intelligence, Vol. 35, 2021, pp. 14138–14148.

[57] J. Xiang, J. Zhang, Exploring low-rank property in multiple instance learning for whole slide image classification, in: The Eleventh International Conference on Learning Representations, 2023.

[58] J. Zhang, K. Ma, J. Van Arnam, R. Gupta, J. Saltz, M. Vakalopoulou, D. Samaras, A joint spatial and magnification based attention framework for large scale histopathology classification, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 3776–3784.

[59] Z. Guo, W. Zhao, S. Wang, L. Yu, Higt: Hierarchical interaction graphtransformer for whole slide image analysis, in: International Conference on Medical Image Computing and Computer-Assisted Intervention, Springer, 2023, pp. 755–764.

[60] Z. Buzzard, K. Hemker, N. Simidjievski, M. Jamnik, Paths: A hierarchical transformer for eficient whole slide image analysis, arXiv preprint arXiv:2411.18225 (2024).

[61] M. Tran, S. Wagner, W. Weichert, C. Matek, M. Boxberg, T. Peng, Navigating through whole slide images with hierarchy, multi-object, and multiscale data, IEEE transactions on medical imaging 44 (5) (2025) 2002–2015.

[62] O. Fourkioti, M. De Vries, C. Bakal, CAMIL: Context-aware multiple instance learning for cancer detection and subtyping in whole slide images, in: The Twelfth International Conference on Learning Representations, 2024.

URL https://openreview.net/forum?id=rzBskAEmoc

[63] K. Chen, S. Sun, J. Zhao, Camil: Causal multiple instance learning for whole slide image classification, in: Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 38, 2024, pp. 1120–1128.

[64] A. Gu, K. Goel, C. Re, Eficiently modeling long sequences with structured state spaces, in: International Conference on Learning Representations, 2021.

[65] Q. Sun, D. Jiang, J. Li, R. Yan, Y. He, T. Guan, Z. Cheng, NcIEMIL: Rethinking decoupled multiple instance learning framework for histopathological slide classification, in: Proceedings of the International Conference on Medical Imaging with Deep Learning, Vol. 250 of Proceedings of Machine Learning Research, 2024, pp. 1166–1178.

[66] J. Peeples, W. Xu, A. Zare, Histogram layers for texture analysis, IEEE Transactions on Artificial Intelligence 3 (4) (2021) 541–552.

[67] A. H. Song, R. J. Chen, T. Ding, D. F. Williamson, G. Jaume, F. Mahmood, Morphological prototyping for unsupervised slide representation learning in computational pathology, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 11566–11578.

[68] X.-S. Wei, J. Wu, Z.-H. Zhou, Scalable algorithms for multi-instance learning, IEEE transactions on neural networks and learning systems 28 (4) (2016) 975–987.

[69] C. E. Shannon, A mathematical theory of communication, The Bell system technical journal 27 (3) (1948) 379–423.

[70] T. M. Cover, Elements of information theory, John Wiley & Sons, 1999.

[71] N. Brancati, A. M. Anniciello, P. Pati, D. Riccio, G. Scognamiglio, G. Jaume, G. De Pietro, M. Di Bonito, A. Foncubierta, G. Botti, et al., Bracs: A dataset for breast carcinoma subtyping in h&e histology images, Database 2022 (2022) baac093.

[72] B. Ehteshami Bejnordi, M. Veta, P. Johannes van Diest, B. Van Ginneken, N. Karssemeijer, G. Litjens, J. A. Van Der Laak, C. consortium, M. Hermsen, Q. F. Manson, et al., Diagnostic assessment of deep learning algorithms for detection of lymph node metastases in women with breast cancer, Jama 318 (22) (2017) 2199–2210.