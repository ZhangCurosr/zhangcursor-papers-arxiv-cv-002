# TecoPrompt: Temporal-Conservative Prompt Learning for Vision-Language Models

Zeyi Shao1, Haowen Hua1, Jiaxin Zhang2, John See3 Zeyd Boukhers4, and Cong Yang1B

<sup>1</sup> Soochow University, Suzhou, China <sup>2</sup> NVIDIA, Shanghai, China

Heriot-Watt University Malaysia, Putrajaya, Malaysia

Fraunhofer FIT, Sankt Augustin, Germany

<sup>B</sup> Corresponding author: cong.yang@suda.edu.cn

Abstract. Prompt learning adapts vision-language models, such as CLIP, by adjusting a small set of context tokens. However, under few-shot supervision, even moderate label noise can disrupt prompt optimization. To address this issue, we propose TecoPrompt, a closed-loop robust promptlearning framework that revisits optimal transport (OT) pseudo-labeling from a temporal perspective. TecoPrompt employs an entropic OT plan in the CLIP semantic space to obtain globally consistent label candidates. It verifies the reliability of these candidates by examining trajectory stability: a noisy label is only rewritten if the OT candidate remains unchanged within a K-epoch temporal stability window and passes a confidence gate based on Exponential Moving Average (EMA). This approach helps reduce confirmation bias. The rewritten labels are then integrated back into prompt training using a tri-group objective that includes three loss functions aligned with clean, mid, and noisy subsets. Experiments on seven datasets with synthetic symmetric and asymmetric noise, as well as Food101N, demonstrate significant performance improvements. For example, on the OxfordPets dataset, with 50% asymmetric noise, TecoPrompt achieves an accuracy of 0.843, up from 0.775.

Keywords: Vision-Language Models · Prompt Learning · Few-Shot Learning

## 1 Introduction

Vision-language pre-trained models (VL-PTMs) like CLIP learn to create aligned representations of images and text. This capability allows for open-vocabulary recognition and robust zero-shot transfer [15, 34, 48]. Building upon this alignment, prompt learning has emerged as a practical method for adaptation. It involves updating only a small set of context tokens while keeping the encoders frozen, making it eficient and data-friendly [16, 17, 52, 53]. Additionally, prior research [43] suggests that prompt-based adaptation may be inherently more tolerant to imperfect supervision.

![](images/82aec814f3e5a1a67fe5a54563741402fc07ce7f8a35640636f001a6bc33b631.jpg)  
Fig. 1: Distribution of the maximum number of consecutive epochs during which the OT pseudo-label remains unchanged. For noisy examples, we count runs in which the stable OT label matches the ground-truth (Noisy, correct); for clean examples, we count runs in which the stable OT label is incorrect (Clean, wrong). Across datasets, noisy examples frequently exhibit long stable-correct runs (including the “20+” bin), whereas clean examples rarely sustain long stable-wrong runs.

Despite their successes, most current strategies [11, 25, 42] for learning from noisy labels ofer limited improvements in few-shot prompt learning, where each labeled example is highly valuable. These methods commonly utilize techniques like sample grouping, instance filtering, or robust loss functions to reduce the impact of noisy annotations. While these approaches enhance robustness, they often downweight or eliminate portions of the data, resulting in ineficient use of limited supervision. A more efective solution is label correction, which aims to convert noisy annotations into reliable targets. However, naive or overly aggressive rewriting can mistakenly alter clean labels, worsening confirmation bias, particularly when early predictions are unstable. Therefore, a high-precision and conservative correction strategy is essential to enhance supervision while carefully managing the risk of mis-correction.

Motivated by this, we revisit optimal transport (OT)-based pseudo-labeling using a globally coupled assignment. To analyze the temporal behavior of OT pseudo-labels, we conducted a controlled experiment in which 50% random label noise was injected into the training data. In Fig. 1, when OT pseudo-labels are recomputed across epochs with a globally coupled assignment, some samples demonstrate strong temporal consistency. Notably, noisy samples often display extended periods during which the OT label remains unchanged and aligns with the ground-truth label, while clean samples are less likely to exhibit such long, incorrect stable runs. This significant distinction suggests that OT provides reliable candidate labels, and that temporal consistency can serve as a precise criterion for determining when to implement label corrections during training.

To demonstrate when stability can be trusted for hard rectification, we analyze the supervision trajectory induced by Optimal Transport (OT) through the lens of learning dynamics [3, 45]. We consider the OT candidate distribution as a score vector and monitor the evolution of its maximum value (argmax) during training. With bounded perturbations per epoch, a suficient margin condition ensures the invariance of the argmax. This explains why stable candidates are more reliable. Additionally, we connect OT label flips to forgetting-style dynamics. This analysis supports the use of a K-epoch stability window as a conservative gate and suggests that K should be chosen to enhance precision in rectification [41].

Based on these insights, we propose TecoPrompt (Temporal-Conservative Prompt Learning), a robust and closed-loop framework for few-shot adaptation in the presence of noisy labels. TecoPrompt uses an entropically regularized optimal transport formulation to establish a globally consistent assignment between image embeddings and prompt-conditioned text prototypes. From this, we derive globally consistent candidates. To verify these candidates, TecoPrompt evaluates trajectory stability within a K-epoch window, alongside an exponential moving average (EMA)-based confidence gate, and performs conservative hard rectification. The corrected labels are then fed back into the prompt optimization process using a tri-group objective. This objective combines cross-entropy loss on clean and rectified samples with noise-robust losses on uncertain samples.

Succinctly, the main contributions of this work are as follows: 1) We propose TecoPrompt, a closed-loop robust prompt learning framework that couples global OT-based candidate generation with strict temporal verification for reliable label rectification. 2) We establish suficient conditions that justify hard label rewriting and show how the stability window and confidence gating improve correction precision. 3) We conduct extensive experiments on diverse datasets under various noise settings, consistently achieving substantial performance gains over prior approaches (e.g., 66.6% vs. 55.3% on Flowers102 under 75% asymmetric noise).

## 2 Related Work

Prompt Learning for Vision-Language Models. Prompt learning adapts CLIP-style vision-language models by tuning lightweight prompts while keeping the backbone frozen. A representative line learns continuous textual contexts, exemplified by CoOp and the instance-conditional CoCoOp [52, 53]. Another line extends prompting beyond pure text to strengthen cross-modal adaptation, such as visual prompt tokens and joint prompting across modalities [16, 17]. Recent studies further improve prompt transfer under distribution shifts by introducing source-aware regularization [18] or by explicitly modeling domain adaptation for CLIP [26]. In parallel, label-free test-time prompt tuning updates prompts using unlabeled target data at inference, including TPT and its calibrated variant, C-TPT [36, 47]. Beyond these core directions, prompt learning is increasingly studied under alternative training regimes that reduce annotation dependence or enhance generalization, such as unsupervised prompt distillation [24] and global-local prompting [20], which leverage both holistic and region-level cues.

Learning with Noisy Labels. Noisy supervision often induces memorization and biased optimization in deep networks [1, 4, 49]. Common defenses include robust training dynamics and regularization [13, 27, 44], noise-robust loss design [9, 51], explicit objective correction via noise modeling or transition estimation [32, 46, 50], and data-centric selection or refinement [22, 23, 35, 37].

Prompt tuning’s empirically observed resilience to noisy supervision was highlighted by Wu et al. [43]; JoAPR [10] separates clean/noisy samples via a Gaussian-mixture partitioning and label-refinement pipeline and then retrains on the purified set; NLPrompt enhances prompt learning by using PromptMAE and PromptOT. The PromptOT method purifies noisy labels by dividing the data into clean and noisy subsets. CE is applied to the clean subset, while MAE is used for noisy samples to improve robustness. In contrast, our method difers from NLPrompt in that it integrates temporal consistency into label corrections. TecoPrompt checks label stability over K epochs before committing corrections, ensuring more reliable updates. Additionally, TecoPrompt employs a tri-group loss to enable more nuanced optimization.

Optimal Transport for Global Consistency. Optimal transport (OT) provides a principled global coupling mechanism and can be solved eficiently via entropic regularization and Sinkhorn iterations [6, 33]. OT has been used as a global signal for learning under noisy supervision, such as filtering/reweighting or curriculum-aware transport [2, 8], and has also been introduced into prompt learning to exploit the shared embedding space [3]. Many existing OT-based prompt-learning approaches primarily employ OT for purification, subset assignment, or loss re-weighting. In contrast, our method treats OT assignments as structured, globally consistent label candidates, and further introduces a sample-wise temporal consistency criterion (based on the stability of historical OT assignments) to decide whether to trigger hard label rewriting, aligning with the broader principle of temporal consistency in semi-supervised learning while explicitly controlling confirmation bias [4, 21, 40].

## 3 Formulation of Temporal-Consistency OT Label Rewriting

In this section, we provide a theoretical characterization of OT label rewriting triggered by temporal consistency in TecoPrompt. Building on temporal consistency as a reliability prior widely used in semi-supervised learning [21] and teacher–student consistency targets [40], we formalize when a temporally stable OT assignment can serve as a reliable label candidate and show that strengthening the temporal consistency requirement yields a more conservative yet reliable rewriting set.

## 3.1 Notation.

Scalars are denoted by non-bold letters, vectors by lowercase bold letters, and matrices by uppercase bold letters. The indicator function is written as 1[·], and we use $[ n ] = \{ 1 , 2 , \dots , n \}$ . For each sample $i \in [ N ]$ , let $y _ { i } ^ { \star } \in [ C ]$ denote its (latent) clean label and $\tilde { y } _ { i } \in [ C ]$ the observed (possibly noisy) label. At epoch t, the OT-induced pseudo-label is $\hat { y } _ { i } ^ { ( t ) } \in [ C ]$ , with a confidence score $c _ { i } ^ { ( t ) } \in [ 0 , 1 ]$

Given a temporal stability window of size $K ,$ we define three events to characterize temporal stability and reliability. The temporal consistency event is ${ \mathcal { E } } _ { K } ( i ) = \{ \hat { y } _ { i } ^ { ( t - { \bar { K } } + 1 ) } = \cdots = \hat { y } _ { i } ^ { ( t ) } \neq \tilde { y } _ { i } \}$ , meaning that the pseudo-label remains unchanged within the K-epoch stability window and disagrees with the observed label. The reliability event is $\mathcal { D } _ { K } ( i ) = \{ c _ { i } ^ { ( t - K + 1 ) } \geq \theta , \ldots , c _ { i } ^ { ( t ) } \geq \theta \}$ , requiring the confidence to stay above a threshold θ throughout the same window. The rewrite-candidate event is defined as $\mathcal { A } _ { K } ( i ) = \mathcal { E } _ { K } ( i ) \cap \mathcal { D } _ { K } ( i )$

## 3.2 OT-Forgettability and a $\mathbf { 2 } \times \mathbf { 2 }$ Decomposition.

We characterize sample-wise stability through the OT pseudo-label trajectory. Let $\hat { y } _ { i } ^ { ( s ) } \in [ C ]$ denote the OT-induced pseudo-label of sample i at epoch s. After warm-up, we monitor the trajectory over a window of length $T _ { w }$ starting from epoch $t _ { 0 }$ . We quantify temporal instability using the one-step flip indicator and its flip rate:

$$
d _ { i } ^ { ( s ) } = \mathbf { 1 } \Big [ \hat { y } _ { i } ^ { ( s + 1 ) } \neq \hat { y } _ { i } ^ { ( s ) } \Big ] , \quad \phi _ { i } = \frac { 1 } { T _ { w } } \sum _ { s = t _ { 0 } } ^ { t _ { 0 } + T _ { w } - 1 } d _ { i } ^ { ( s ) } \in [ 0 , 1 ] .\tag{1}
$$

A smaller $\phi _ { i }$ indicates a more stable OT assignment, while a larger value indicates frequent changes. Given a threshold $\phi _ { 0 } \in ( 0 , 1 )$ , we define the OTforgettability indicator as follows:

$$
F _ { i } = \mathbf { 1 } [ \phi _ { i } > \phi _ { 0 } ] .\tag{2}
$$

We call $F _ { i } = 0$ OT-unforgettable and $F _ { i } = 1$ OT-forgettable. Unlike the classic example-forgetting definition based on correctness with respect to $y _ { i } ^ { \star } , \phi _ { i }$ and $F _ { i }$ are directly observable from the OT pseudo-label trajectory.

For theoretical analysis, let $\tilde { y } _ { i }$ be the observed label and $y _ { i } ^ { \star }$ be the ground-truth label. We introduce the latent noise indicator:

$$
Z _ { i } = { \bf 1 } [ \tilde { y } _ { i } \neq y _ { i } ^ { \star } ] .\tag{3}
$$

which is used only to describe the clean/noisy decomposition and is not required by the rewriting rule. The pair $( Z _ { i } , F _ { i } )$ induces a $2 \times 2$ partition:

$$
\displaystyle \frac { \big | F _ { i } = 0 , ~ \mathrm { O T - u n f o r g e t t a b l e } \quad F _ { i } = 1 , ~ \mathrm { O T - f o r g e t t a b l e } } { S _ { c , u } }\tag{4}
$$

Intuitively, $S _ { n , u }$ contains noisy yet OT-unforgettable samples: despite incorrect observed labels, their OT pseudo-labels remain consistent across training, making them the primary targets for conservative label rewriting.

Rewriting objective. Our goal is to correct supervision mainly on the noisy OTunforgettable subset $S _ { n , u } .$ . Meanwhile, the procedure should remain conservative on clean data by requiring stable disagreement and high confidence.

![](images/a3b05b52e12dea7af3e3f183a6400a41457411690f938fed046c44afb3f716d8.jpg)  
Fig. 2: Overview of the proposed TecoPrompt framework. Given frozen CLIP image/text encoders, we compute image embeddings and prompt-conditioned text prototypes, and solve a balanced entropic optimal transport problem to obtain globally consistent pseudo-label distributions. We then blend the OT pseudo-labels with the observed noisy labels using confidence weighting (where confidence scores are EMA-smoothed), perform sample selection to form clean/mid/noisy groups, and optimize prompts with a tri-group objective. In parallel, we track the OT-induced label trajectory over epochs and apply EMA-gated temporal consistency with a K-epoch stability window to conservatively trigger hard-label rewriting, thereby forming a closed-loop refinement that mitigates confirmation bias.

## 3.3 Temporal-Consistency Rewriting.

Following Sec. 3.1, $\boldsymbol { \mathcal { A } } _ { K } ( \boldsymbol { i } )$ collects samples whose OT-induced pseudo-labels are temporally stable and simultaneously disagree with the observed labels.

Therefore, a sample is rewritten only if it is temporally consistent, disagrees with y˜<sub>i</sub>, and maintains high confidence over the history window.

Theorem 1. Under the proposed assumptions, increasing the consistency window size K improves the reliability of the rewritten labels. In particular, for suficiently large K, the probability of correct label rewriting can be made arbitrarily high.

The proof is provided in the Appendix. This result shows that longer OTconsistency trajectories progressively eliminate unstable samples, leading to robust label correction under noise.

## 4 Method

As illustrated in Fig. 2, our core idea is to combine a global view and a sample-wise view: OT provides globally consistent label candidates, while temporal tracking identifies which candidates are suficiently reliable to trigger conservative label rewriting. Our method consists of five components: (i) Text & Image Encoder, (ii) OT-based purification, (iii) EMA-smoothed OT confidence, (iv) temporal consistency label rewriting, and (v) ternary grouping with loss-specific training.

## 4.1 Text & Image Encoder.

We use a CLIP-style prompt-learning setup with a frozen text encoder $h ( \cdot )$ and a frozen image encoder $g ( \cdot )$ . For each class $c \in [ C ]$ , we feed a learnable context prompt p together with a fixed class prompt $p _ { c }$ into the text encoder h to obtain a class-specific text feature:

$$
\mathbf { h } _ { c } = h ( p , p _ { c } ) \in \mathbb { R } ^ { d } .\tag{5}
$$

For each input image $x _ { i } .$ we extract a d-dimensional image representation $\mathbf { g } _ { i } \in \mathbb { R } ^ { d }$ using the frozen image encoder $g ( \cdot )$

We define the similarity between an image feature and a class text feature as their inner product:

$$
\sin ( \mathbf { g } _ { i } , \mathbf { h } _ { c } ) = \langle \mathbf { g } _ { i } , \mathbf { h } _ { c } \rangle .\tag{6}
$$

and obtain the predictive distribution by applying a softmax operator to the resulting similarity vector over all classes:

$$
\begin{array} { r } { { \bf s } _ { i } ( p ) = \mathrm { s o f t m a x } \big ( \mathrm { s i m } ( { \bf g } _ { i } , \{ { \bf h } _ { c } \} _ { c = 1 } ^ { C } ) \big ) \in \mathbb { R } ^ { C } . } \end{array}\tag{7}
$$

In implementation, we stack the class text features $\{ \mathbf { h } _ { c } \} _ { c = 1 } ^ { C }$ and image features $\{ \mathbf { g } _ { i } \} _ { i = 1 } ^ { N }$ into matrices $\mathbf { T } = [ \mathbf { h } _ { 1 } ; \ldots ; \mathbf { h } _ { C } ] \in \mathbb { R } ^ { C \times d }$ and ${ \bf { I } } = [ { \bf { g } } _ { 1 } ; \ldots ; { \bf { g } } _ { N } ] \in { \mathbb { R } } ^ { N \times d }$ 2 and compute the pairwise similarity matrix $\mathbf { S } = \mathbf { T } \mathbf { I } ^ { \top } \in \mathbb { R } ^ { \bar { C } \times N }$

## 4.2 OT-based Purification

We apply entropically regularized OT in the shared vision-language embedding space, where class-wise text features serve as prototypes and image features as samples. Solved eficiently by Sinkhorn iterations [6, 33], OT yields balanced soft pseudo-labels for each sample, providing a lightweight purification signal for data partitioning, confidence estimation, and subsequent refinement.

Given the similarity matrix $\mathbf { S } \in \mathbb { R } ^ { C \times N }$ computed above, we construct the OT cost matrix C via an element-wise negative log transform of $\mathbf { S } ,$ and solve the following balanced OT problem:

$$
\operatorname* { m i n } _ { \mathbf { Q } \geq 0 } | \mathbf { C } , \mathbf { Q } \rangle - \epsilon H ( \mathbf { Q } ) \ \mathrm { ~ s . t . ~ } \ \mathbf { Q } \mathbf { 1 } _ { N } = \frac { 1 } { C } \mathbf { 1 } _ { C } , \ \mathbf { Q } ^ { \top } \mathbf { 1 } _ { C } = \frac { 1 } { N } \mathbf { 1 } _ { N } .\tag{8}
$$

where $\begin{array} { r } { H ( \mathbf { Q } ) = - \sum _ { c = 1 } ^ { C } \sum _ { i = 1 } ^ { N } Q _ { c , i } } \end{array}$ log $Q _ { c , i }$ is the entropy of the transport plan, and $\epsilon > 0$ controls the strength of the entropic regularization. The optimal coupling Q⋆ induces a soft pseudo-label distribution for each sample (columnwise). We denote the OT-induced pseudo-label by $\hat { y } _ { i } ^ { ( t ) } \in [ C ]$ with confidence $c _ { i } ^ { ( t ) } \in [ 0 , 1 ]$ , and take:

$$
\hat { y } _ { i } ^ { ( t ) } = \arg \operatorname* { m a x } _ { c \in [ C ] } Q _ { c , i } ^ { \star } , \quad c _ { i } ^ { ( t ) } = \operatorname* { m a x } _ { c \in [ C ] } Q _ { c , i } ^ { \star } .\tag{9}
$$

as the OT signal used to guide subsequent sample selection and to quantify temporal consistency across iterations.

![](images/4811a902bcb199a0c983a821421d5f694f3accca724f167a25af428e188422e1.jpg)  
Fig. 3: Optimal Transport generates pseudo-label candidates to rewrite noisy annotations. Through iterative updates over epochs, incorrect labels are progressively corrected, reducing the noise rate and improving the quality of supervision.

## 4.3 EMA-Smoothed OT Confidence

Single-epoch OT confidence can be noisy. We therefore maintain an exponential moving average (EMA) confidence for each sample:

$$
\begin{array} { r } { \bar { c } _ { i } ^ { ( t ) } = \left\{ \begin{array} { l l } { c _ { i } ^ { ( t ) } , } & { t = 1 , } \\ { \beta \bar { c } _ { i } ^ { ( t - 1 ) } + ( 1 - \beta ) c _ { i } ^ { ( t ) } , } & { t > 1 . } \end{array} \right. } \end{array}\tag{10}
$$

where $\beta \in [ 0 , 1 )$ controls the smoothing strength. We also keep a short history of OT pseudo-labels and confidences to support temporal consistency checking.

## 4.4 Temporal Consistency Label Rewriting

We trigger hard label rewriting only when the OT pseudo-label is both temporally consistent and suficiently confident, as illustrated in Fig. 3. Following Sec. 3.1, where $\boldsymbol { \mathcal { A } } _ { K } ( \boldsymbol { i } )$ denotes the rewrite-candidate event requiring stability within the K-epoch window and sustained confidence. For eligible samples, we rewrite the observed label by the stable OT candidate:

$$
\tilde { y } _ { i } \gets \hat { y } _ { i } ^ { ( t ) } \quad \mathrm { i f } \ A _ { K } ( i ) \ \mathrm { h o l d s } .\tag{11}
$$

This criterion enforces a conservative rewrite policy by requiring both temporal stability and sustained confidence before updating labels.

warmup. To avoid aggressive early corrections and reduce confirmation bias, we introduce a warmup condition. Let t denote the current epoch index and treat $t < K$ as the warmup stage (warmup length =K-1 epochs), during which we only collect the OT-induced label assignments into the temporal bufer without applying any hard rewrite. Hard OT label rewriting is enabled only once $t \geq K$ , i.e., after accumulating a full K-epoch stability window. In addition to preventing unstable early predictions from triggering irreversible corrections, this strategy allows the model to reach a more reliable regime and enforces multi-epoch consistency as a prerequisite for rewriting.

## 4.5 Ternary Grouping and Training Objectives

After smoothing and consistency checking, and applying label rewriting when triggered, we partition the training set $\mathcal { D }$ into three pairwise disjoint subsets $\mathcal { D } _ { \mathrm { c l n } } .$ $\mathcal { D } _ { \mathrm { m i d } }$ , and $\mathcal { D } _ { \mathrm { n o s } }$ , such that $\mathcal { D } = \mathcal { D } _ { \mathrm { c l n } } \cup \mathcal { D } _ { \mathrm { m i d } } \cup \mathcal { D } _ { \mathrm { n o s } }$ and $\mathcal { D } _ { a } \cap \mathcal { D } _ { b } = \emptyset$ for $a \neq b$

Clean set. A sample is placed into $\mathcal { D } _ { \mathrm { c l n } }$ if its $\mathrm { O T }$ pseudo-label agrees with the observed label and its EMA confidence passes a strict threshold. Here $\tau _ { \mathrm { c l e a n } }$ and $\tau _ { \mathrm { m i d } }$ denote confidence thresholds with $\tau _ { \mathrm { c l e a n } } > \tau _ { \mathrm { m i d } } :$

$$
i \in \mathcal { D } _ { \mathrm { c l n } } \quad \iff \quad \widehat { y } _ { i } ^ { ( t ) } = \widetilde { y } _ { i } \mathrm { a n d } \bar { c } _ { i } ^ { ( t ) } \geq \tau _ { \mathrm { c l e a n } } .\tag{12}
$$

Mid set. The mid set collects samples that are likely usable but not fully reliable. Concretely, we include either agreement samples with moderate EMA confidence or rewrite-eligible samples not already assigned to $\mathcal { D } _ { \mathrm { c l n } }$

$$
\begin{array} { r } { i \in \mathcal { D } _ { \operatorname* { m i d } } \quad \iff \quad \left( { \hat { y } } _ { i } ^ { ( t ) } = \tilde { y } _ { i } \mathrm { a n d } \tau _ { \operatorname* { m i d } } \leq { \bar { c } } _ { i } ^ { ( t ) } < \tau _ { \mathrm { c l e a n } } \right) \mathrm { o r } A _ { K } ( i ) . } \end{array}\tag{13}
$$

Noisy set. All remaining samples are treated as noisy:

$$
\mathcal { D } _ { \mathrm { n o s } } = \mathcal { D } \setminus \left( \mathcal { D } _ { \mathrm { c l n } } \cup \mathcal { D } _ { \mathrm { m i d } } \right) .\tag{14}
$$

Loss functions. Let $\mathbf { s } _ { i } \in \varDelta ^ { C - 1 }$ denote the predicted class distribution obtained from the similarity scores via a softmax operator, and let $s _ { i , \tilde { y } _ { i } }$ denote the probability assigned to the current training label $\tilde { y } _ { i }$ . We optimize a weighted sum of losses over the three subsets:

$$
\begin{array} { r } { \mathcal { L } = \lambda _ { \mathrm { c } } \mathcal { L } _ { \mathrm { c l n } } + \lambda _ { \mathrm { m } } \mathcal { L } _ { \mathrm { m i d } } + \lambda _ { \mathrm { n } } \mathcal { L } _ { \mathrm { n o s } } . } \end{array}\tag{15}
$$

where $\lambda _ { \mathrm { c } } , \lambda _ { \mathrm { m } } , \lambda _ { \mathrm { n } } \geq 0$ control the relative contribution of each term. Our design follows a “strong-to-robust” principle based on subset reliability: The clean subset is expected to be highly reliable and thus benefits from the strong discriminability of cross-entropy (CE); The mid subset may contain mild corruption, so we adopt generalized cross-entropy (GCE) with $q \in ( 0 , 1 )$ as a compromise that downweights low-confidence samples while preserving learning eficacy; The noisy subset is potentially heavily corrupted, for which we use an MAE-style objective that is less sensitive to mislabeled targets and helps prevent overfitting to noise:

$$
\mathcal { L } _ { \mathrm { c l n } } = \sum _ { i \in \mathcal { D } _ { \mathrm { c l n } } } - \log s _ { i , \tilde { y } _ { i } } , \ \mathcal { L } _ { \mathrm { m i d } } = \sum _ { i \in \mathcal { D } _ { \mathrm { m i d } } } \frac { 1 - s _ { i , \tilde { y } _ { i } } ^ { q } } { q } , \ \mathcal { L } _ { \mathrm { n o s } } = \sum _ { i \in \mathcal { D } _ { \mathrm { n o s } } } ( 1 - s _ { i , \tilde { y } _ { i } } ) .\tag{16}
$$

## 5 Experiments

We conduct extensive experiments to evaluate the efectiveness and robustness of TecoPrompt under few-shot prompt learning with noisy supervision.

## 5.1 Benchmarks and Baselines

Synthetic noisy benchmarks. We use seven widely adopted visual classification datasets: Flowers102 [28], DTD [5], EuroSAT [12], OxfordPets [30], Stanford-Cars [19], UCF101 [38], and Caltech101 [7]. These datasets are originally clean; we corrupt labels only on the training split (Sec. 5.2) and keep the oficial test split unchanged for evaluation, as in [29]. For each dataset, we construct a 16-shot training set per class.

Real-world noisy benchmark. We additionally evaluate on Food101N [22], a real-world dataset with naturally noisy web-collected annotations, to assess robustness under practical noise patterns beyond controlled synthetic label flips. Baselines. We compare TecoPrompt with CoOp [53], CoOp trained with generalized cross-entropy (CoOp+GCE) [43], JoAPR [10], and NLPrompt. All methods use the same backbone and few-shot splits for fair comparison.

## 5.2 Implementation Details

We evaluate two corruption patterns following [29] and apply noise only to the training labels. Under symmetric noise (Sym), each label is independently flipped to a uniformly sampled incorrect class with probability η. Under asymmetric noise (Asym), we use a fixed class-dependent mapping, in which each class is flipped to a designated successor class with probability η, thereby producing structured corruption where noisy labels are often less distinguishable, and the setting is generally more challenging.

We adopt the same experimental setup as NLPrompt [29] for fair comparison. All experiments are conducted with the pre-trained CLIP model [34] using ResNet-50 as the image encoder. The text encoder is the CLIP text transformer, and we freeze both encoders while optimizing only the learnable prompt parameters.

We train models for 200 epochs using SGD with an initial learning rate of 0.002 and a cosine annealing schedule. For prompt design, we use 16 shared context tokens and place the class token at the end of the prompt, following the standard CoOp-style template [53]. All experiments were conducted on a cluster equipped with NVIDIA A100 GPUs using PyTorch [31]. All results are reported as the average accuracy over three random seeds, and the highest accuracy in each column is highlighted in bold.

## 5.3 Results on Synthetic Noisy Labels

Tab. 1 summarizes performance under synthetic symmetric and asymmetric label noise across seven datasets. The noise rate ranges from 12.5% to 75%, increasing in steps of 12.5%. Across most datasets and noise levels, TecoPrompt achieves strong performance, with particularly notable advantages emerging as the noise rate increases. Under severe label corruption, the performance gap becomes more pronounced, demonstrating that the proposed prompt is less afected by noisy supervision. These results indicate that TecoPrompt learns more robust prompts that maintain stable performance even under high levels of label noise.

Table 1: Classification performance (accuracy, %) under symmetric (Sym) and asymmetric (Asym) label noise. Except for TecoPrompt, all results are taken from NLPrompt [29].
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Method</td><td colspan="4">Noise Rate: Sym</td><td colspan="6">Noise Rate: Asym</td></tr><tr><td colspan="4">12.5% 25.0% 37.5% 50.0% 62.5% 75.0%</td><td colspan="6">12.5% 25.0% 37.5% 50.0% 62.5% 75.0%</td></tr><tr><td rowspan="5">Flowers102</td><td>CoOp</td><td>88.9</td><td>83.5 77.9</td><td>70.1</td><td>55.6</td><td>37.2</td><td>87.0</td><td>74.7</td><td>60.4</td><td>42.6</td><td>26.5</td><td>12.6</td></tr><tr><td>GCE</td><td>88.8</td><td>88.3 86.7</td><td>84.1</td><td>78.4</td><td>70.4</td><td>88.4</td><td>86.4</td><td>80.3</td><td>69.9</td><td>61.5</td><td>39.2</td></tr><tr><td>JoAPR</td><td>85.6</td><td>81.2 74.6</td><td>70.2</td><td>67.9</td><td>66.9</td><td>85.2</td><td>79.6</td><td>74.0</td><td>73.8</td><td>53.4</td><td>13.3</td></tr><tr><td>NLPrompt</td><td>93.9</td><td>92.6 92.7</td><td>89.9</td><td>84.8</td><td>76.8</td><td>93.8</td><td>93.4</td><td>91.8</td><td>81.1</td><td>73.6</td><td>55.3</td></tr><tr><td>TecoPrompt</td><td>94.0</td><td>93.7 92.9</td><td>90.5</td><td>86.1</td><td>82.3</td><td>94.1</td><td>93.5</td><td>92.2</td><td>89.4</td><td>80.4</td><td>66.6</td></tr><tr><td rowspan="5">DTD</td><td>CoOp GCE</td><td>56.0 61.0</td><td>49.6</td><td>43.3 34.4 50.7</td><td>27.8 43.6</td><td>17.3 33.7</td><td>55.6 60.7</td><td>47.8</td><td>38.1 52.7</td><td>29.6 44.0</td><td>20.5</td><td>11.7</td></tr><tr><td></td><td></td><td>59.8</td><td>56.8</td><td></td><td></td><td></td><td>57.6</td><td></td><td></td><td>33.4</td><td>18.2</td></tr><tr><td>JoAPR</td><td>58.1</td><td>57.7</td><td>56.3 53.0</td><td>48.1</td><td>29.9</td><td>52.4</td><td>56.6</td><td>53.1</td><td>48.9</td><td>40.2</td><td>28.3</td></tr><tr><td>NLPrompt</td><td>63.0</td><td>61.2</td><td>59.2 55.2</td><td>49.0</td><td>39.8</td><td>62.3</td><td>60.6</td><td>56.5</td><td>50.8</td><td>40.3</td><td>28.4</td></tr><tr><td>TecoPrompt</td><td>62.7</td><td>61.8</td><td>59.8 58.1</td><td>54.2</td><td>49.8</td><td>62.8</td><td>62.7</td><td>58.3</td><td>56.5</td><td>48.5</td><td>38.4</td></tr><tr><td rowspan="5">EuroSAT</td><td>CoOp GCE</td><td>76.5 82.1</td><td>69.2</td><td>61.7 52.3 63.1</td><td>37.6 49.7</td><td>26.7 31.4</td><td>76.0 78.2</td><td>66.3 72.7</td><td>53.8 63.6</td><td>41.2 45.3</td><td>28.0 22.9</td><td>17.4</td></tr><tr><td></td><td>75.1</td><td>78.6</td><td>74.7</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>12.1</td></tr><tr><td>JoAPR</td><td></td><td>61.1</td><td>60.9 63.6</td><td>39.0</td><td>27.3</td><td>69.4</td><td>67.3</td><td>59.4</td><td>47.6</td><td>33.9</td><td>17.5</td></tr><tr><td>NLPrompt</td><td>82.5</td><td>79.5</td><td>78.1 66.7</td><td>63.5</td><td>43.8</td><td>80.1</td><td>77.1</td><td>71.4</td><td>54.3</td><td>37.3</td><td>32.7</td></tr><tr><td>TecoPrompt</td><td>83.5</td><td>80.7 78.9</td><td>72.7</td><td>68.7</td><td>52.1</td><td>81.1</td><td>80.1</td><td>78.7</td><td>63.7</td><td>37.9</td><td>33.7</td></tr><tr><td rowspan="5">OxfordPets</td><td>CoOp GCE</td><td>76.5 85.6</td><td>66.7 60.3</td><td>47.0 79.2</td><td>35.8 71.4</td><td>24.6 53.2</td><td>76.1</td><td>66.2</td><td>52.5</td><td>38.7</td><td>26.6</td><td>14.9</td></tr><tr><td></td><td></td><td>84.6</td><td>83.7</td><td></td><td></td><td>85.5</td><td>83.0</td><td>76.7</td><td>68.1</td><td>50.7</td><td>32.0</td></tr><tr><td>JoAPR</td><td>84.0</td><td>83.3 83.2</td><td>83.1</td><td>82.4</td><td>74.4</td><td>82.9</td><td>83.4</td><td>79.1</td><td>75.8</td><td>52.7</td><td>43.6</td></tr><tr><td>NLPrompt</td><td>86.2</td><td>86.0 85.3</td><td>84.9</td><td>83.6</td><td>70.8</td><td>86.0</td><td>85.0</td><td>82.4</td><td>77.5</td><td>66.3</td><td>48.6</td></tr><tr><td>TecoPrompt</td><td>86.7</td><td>86.3 85.7</td><td>85.1</td><td>84.2</td><td>84.6</td><td>86.7</td><td>85.2</td><td>84.9</td><td>84.3</td><td>83.0</td><td>75.3</td></tr><tr><td rowspan="5">StanfordCars</td><td>CoOp</td><td>66.2</td><td>59.7 53.4</td><td>45.9</td><td>35.7</td><td>22.9</td><td>65.8</td><td>57.1</td><td>46.2</td><td>33.7</td><td>22.4</td><td>12.8</td></tr><tr><td>GCE</td><td>69.7</td><td>66.4</td><td>66.5 63.8</td><td>59.3</td><td>50.9</td><td>70.0</td><td>66.5</td><td>61.2</td><td>53.7</td><td>39.7</td><td>26.6</td></tr><tr><td>JoAPR</td><td>68.6</td><td>66.3 62.8</td><td>56.7</td><td>48.5</td><td>39.4</td><td>66.5</td><td>61.7</td><td>51.5</td><td>42.0</td><td>30.8</td><td>23.0</td></tr><tr><td>NLPrompt</td><td>69.4</td><td>68.8 67.2</td><td>65.6</td><td>62.8</td><td>58.3</td><td>69.8</td><td>67.5</td><td>64.2</td><td>59.0</td><td>50.9</td><td>39.5</td></tr><tr><td>TecoPrompt</td><td>69.7</td><td>68.9 68.2</td><td>66.8</td><td>64.4</td><td>60.4</td><td>70.2</td><td>68.7</td><td>66.7</td><td>63.2</td><td>58.2</td><td>51.9</td></tr><tr><td rowspan="5">UCF101</td><td>CoOp</td><td>69.0</td><td>63.4 58.2</td><td>49.7</td><td>40.8</td><td>26.3</td><td>67.2</td><td>58.1</td><td>46.5</td><td>34.4</td><td>23.7</td><td>13.2</td></tr><tr><td>GCE</td><td>74.0</td><td>73.6</td><td>72.6</td><td>69.4 66.0</td><td>57.1</td><td>73.9</td><td>71.9</td><td>68.0</td><td>62.2</td><td>52.5</td><td>36.4</td></tr><tr><td>JoAPR</td><td>72.8</td><td>71.2 70.4</td><td>67.6</td><td>65.3</td><td>57.7</td><td>72.1</td><td>69.8</td><td>64.1</td><td>59.2</td><td>56.1</td><td>47.5</td></tr><tr><td>NLPrompt</td><td>74.8</td><td>73.4 72.8</td><td>70.3</td><td>68.1</td><td>60.5</td><td>74.9</td><td>73.5</td><td>71.0</td><td>66.0</td><td>59.0</td><td>49.3</td></tr><tr><td>TecoPrompt</td><td>75.1</td><td>74.4 73.4</td><td>72.0</td><td>70.3</td><td>67.4</td><td>75.1</td><td>73.9</td><td>71.6</td><td>70.1</td><td>67.4</td><td>58.1</td></tr><tr><td rowspan="5">Caltech101</td><td>CoOp</td><td>86.4</td><td>81.0 76.7</td><td>70.9</td><td>61.3</td><td>46.9</td><td>84.9</td><td>75.2 91.2</td><td>62.9 89.7</td><td>49.4 85.8</td><td>33.6 78.2</td><td>20.3 62.1</td></tr><tr><td>GCE JoAPR</td><td>92.0 90.3</td><td>90.9 90.5</td><td>90.8 89.9</td><td>89.3 88.3</td><td>86.7 86.9</td><td>79.0 91.3 83.9 90.3</td></table>

## 5.4 Results on real-world Noisy Labels

Tab. 2 reports the results on Food101N [22], a real-world dataset with naturally noisy labels. TecoPrompt achieves the highest accuracy among all compared approaches, demonstrating that the robustness observed under synthetic noise settings efectively transfers to realistic, naturally corrupted supervision.

## 5.5 Few-shot Learning Analysis

To understand how data scarcity interacts with label corruption, we vary the number of shots per class in \ifmode lbrac \s texbraclf \i 1,2486} while fixing the noise rate at 50%. In these experiments, we set the number of training epochs to 100. The trends in Fig. 4 show that accuracy rises steadily with more shots for all compared methods, under both symmetric and asymmetric noise. Across datasets (Caltech101, DTD, OxfordPets, StanfordCars), TecoPrompt maintains an advantage throughout, with the gap most evident at very low shot counts, indicating better robustness when both supervision quality and quantity are limited.

Table 2: Test accuracy (%) on Food101N.
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>CoOp</td><td rowspan=1 colspan=1>GCE</td><td rowspan=1 colspan=1>JoAPR</td><td rowspan=1 colspan=1>NLPrompt</td><td rowspan=1 colspan=1>TecoPrompt</td></tr><tr><td rowspan=1 colspan=1>Accuracy</td><td rowspan=1 colspan=1>69.50</td><td rowspan=1 colspan=1>71.32</td><td rowspan=1 colspan=1>72.57</td><td rowspan=1 colspan=1>76.46</td><td rowspan=1 colspan=1>78.67</td></tr></table>

![](images/d5d19359169a742de8df683f06edb250643a0d0a9ab9ef96f5b3af43b6a2dc44.jpg)  
Fig. 4: Few-shot robustness under label noise with diferent shot budgets. We vary the number of shots per class over 1, 2, 4, 8, 16, while keeping the noise rate at 50%.

## 5.6 Comparison of Supervision Strategies

Tab. 3 reports test accuracy on Flowers102 after 100 epochs under symmetric and asymmetric label corruption at rates of 20%, 40%, 60%, and 80%. LS and ST denote Label Smoothing [39] and Soft Targets [14], respectively. NLPrompt performs OT-based label allocation, whereas TecoPrompt further introduces EMA-smoothed confidence and a K-epoch stability window to enable conservative hard-label rewriting. TecoPrompt consistently achieves the best performance across all noise settings, with larger margins under severe corruption, indicating that temporally verified hard rewriting provides more reliable supervision than purely OT-based soft supervision.

## 5.7 Ablation Study

Tab. 4 summarizes ablation results of TecoPrompt on OxfordPets under symmetric label noise. All models are trained for 100 epochs under the 16-shot setting following Sec. 5.2. Unless otherwise specified, the default configuration uses EMA and rewriting with K=8 and loss-CE-0.5-MAE, where CE, GCE(q=0.5), and MAE are applied to clean, mid, and noisy groups, respectively. Rows (a)–(c)

Table 3: Comparison of soft supervision and hard label rewriting strategies under diferent noise rates on Flowers102.
<table><tr><td rowspan="2">Supervision Strategy</td><td colspan="4">Noise Rate: Sym</td><td colspan="4">Noise Rate: Asym</td></tr><tr><td>20%</td><td>40%</td><td>60%</td><td>80%</td><td>20%</td><td>40%</td><td>60%</td><td>80%</td></tr><tr><td> $\overline { { \mathrm { G C E } + \mathrm { S T } } }$ </td><td>85.93</td><td>84.48</td><td>81.19</td><td>66.15</td><td>83.53</td><td>80.50</td><td>67.43</td><td>39.15</td></tr><tr><td> $\mathrm { N L P r o m p t + S T }$ </td><td>88.42</td><td>86.33</td><td>82.51</td><td>76.43</td><td>88.74</td><td>86.91</td><td>81.37</td><td>71.80</td></tr><tr><td> $\mathrm { N L P r o m p t + L S }$ </td><td>88.35</td><td>88.12</td><td>82.54</td><td>74.48</td><td>88.61</td><td>84.92</td><td>79.07</td><td>70.63</td></tr><tr><td>TecoPrompt</td><td></td><td>90.76 88.98 83.31 78.02</td><td></td><td></td><td></td><td>89.86 87.21 81.93 73.14</td><td></td><td></td></tr></table>

Table 4: Ablation studies under multiple noise ratios (%).
<table><tr><td>ID</td><td>EMA</td><td>Rewrite</td><td>K</td><td>Loss</td><td>10%</td><td>30%</td><td>50%</td><td>70%</td><td>Avg</td></tr><tr><td>(a) (b)</td><td></td><td>√</td><td>一 一</td><td></td><td>87.05 87.21</td><td>87.03 86.77</td><td>85.46 86.02</td><td>78.22 85.03</td><td>84.44 86.26</td></tr><tr><td>(c) (d)</td><td>√ √</td><td>√</td><td>一</td><td>CE-0.3-0.5</td><td>86.05</td><td>85.98</td><td>85.34</td><td>81.30</td><td>84.67</td></tr><tr><td>(e)</td><td>√</td><td>√</td><td>一 一</td><td>0.7-0.5-0.3</td><td>87.58 86.70</td><td>86.91 85.95</td><td>85.98 84.35</td><td>85.36 58.62</td><td>86.46</td></tr><tr><td>(f)</td><td>√</td><td>√</td><td>1</td><td></td><td>82.23</td><td></td><td></td><td>80.15</td><td>78.91</td></tr><tr><td>(g)</td><td>√</td><td>√</td><td>4</td><td></td><td>86.52</td><td>80.85</td><td>78.12</td><td></td><td>80.59</td></tr><tr><td></td><td>√</td><td></td><td></td><td></td><td></td><td>85.75</td><td>85.98</td><td>86.02</td><td>86.07</td></tr><tr><td>(h) (i)</td><td>√</td><td>√ √</td><td>8 16</td><td></td><td>87.52 87.70</td><td>87.19 87.61</td><td>86.40 85.87</td><td>86.72 84.74</td><td>86.96 86.48</td></tr></table>

ablate EMA and rewriting, rows (d)–(e) vary the group-wise loss design, and rows (f)–(i) study diferent K values.

The results show that rewriting is the main driver of robustness, especially under heavy noise, while EMA alone brings smaller gains but improves stability when coupled with rewriting. The sharp collapse of (e) at 70% noise highlights that an improper group-wise loss assignment can be detrimental in the high-noise regime, motivating the robust loss choice for the noisy group in our default design. Finally, varying K indicates that moderate history yields the most stable performance: small K introduces noisy corrections, whereas large K makes rewriting overly conservative and reduces the frequency of corrections. The lower error ratio under larger K can occasionally yield slightly better results than K = 8, but a moderate window provides the best overall balance.

Fig. 5 provides a closer look at how K afects rewriting dynamics. With a small K, rewriting reacts to transient fluctuations, leading to less stable corrections and a higher error ratio. Increasing K suppresses such short-term label switches and reduces harmful flips, improving correction precision in a manner consistent with Theorem 1. However, an overly large window becomes too conservative and triggers fewer efective corrections. Overall, the dynamics in Fig. 5 explain why K=8 is a suitable default on this dataset, striking a favorable balance between correction quantity and correction precision.

![](images/10a4b240b1e31c336331c2701fd0b50342d1332bc703906c6a68f3903f2d8333.jpg)  
Fig. 5: Efect of the temporal stability window size K for conservative hard label rewriting. The top row reports the per-epoch counts of rewriting outcomes, including incorrect→correct, incorrect→incorrect, and correct→incorrect. The bottom row shows the ratio of incorrect labels in the dataset over training. The dashed line denotes the warm-up stage before rewriting is enabled.

## 5.8 Qualitative Analysis of Label Rewriting

On OxfordPets under 50% label noise with 100 training epochs, TecoPrompt rewrites 234 samples, of which 222 are corrected from wrong labels to groundtruth labels, accounting for 94.87%. Since this setting contains 296 noisy labels in total, these corrections cover 75.00% of the noisy samples. In comparison, only 6 samples are changed from correct labels to wrong labels, and 6 from one wrong label to another wrong label, each accounting for 2.56%. These results indicate that the temporal stability window efectively filters unreliable OT candidates and enables accurate hard rewriting.

Fig. 6 shows the correct-to-wrong cases, where Bengal is rewritten as Egyptian Mau and Leonberger as Keeshond. These errors occur between visually similar fine-grained classes that share appearance cues such as texture, color, and facial structure. This suggests that strong inter-class similarity can occasionally lead to temporally stable but incorrect OT assignments.

## 5.9 Limitations

TecoPrompt prioritizes high-precision hard rewrites and therefore does not attempt to correct every noisy label. In practice, examples that flip frequently during training (so-called forgettable examples) typically fail the temporal verification and thus remain unmodified, which is an intentional trade-of to avoid incorrect corrections. The K-epoch stability window controls the precision-coverage balance and should be tuned for each dataset; empirically, K = 8 works well across our benchmarks. Finally, the method requires maintaining historical information and EMA updates, which introduce additional overhead. However, this cost remains modest in practice; on Caltech101 under the 16-shot setting, it requires only 0.20 seconds more per epoch than NLPrompt on a single RTX 3080 Ti.

![](images/31e920f8a9546bf17d12b30cb96af60a1ec4ebf95e43b84b42ae54c0965ab749.jpg)  
Fig. 6: Examples of rare incorrect label rewriting cases. The left column shows the original training samples, the middle column shows reference images from the revised classes, and the right column reports the corresponding label changes and rewriting outcomes. These cases illustrate that incorrect rewrites mainly occur between visually similar fine-grained categories.

## 6 Conclusion

We propose TecoPrompt, a robust prompt-learning framework for vision–language models under noisy supervision. It stabilizes OT-based label candidates through EMA confidence gating and performs conservative rewriting only when candidates remain consistent within a K-epoch temporal window, reducing confirmation bias and pseudo-label drift. A group-specific objective further applies tailored losses to clean, mid, and noisy subsets. Experiments across diverse noise types and severities demonstrate consistent robustness gains, validating temporal-conservative label correction for noise-tolerant prompt tuning.

## Acknowledgments.

This work was supported in part by the National Natural Science Foundation of China (62473276, 62573309), in part by the Natural Science Foundation of Jiangsu Province (BK20241918), and in part by the Research Fund of Horizon Robotics (H230666).

## Bibliography

[1] Arpit, D., Jastrzebski, S., Ballas, N., Krueger, D., Bengio, E., Kanwal, M.S., Maharaj, T., Fischer, A., Courville, A., Bengio, Y., Lacoste-Julien, S.: A closer look at memorization in deep networks. In: International Conference on Machine Learning (ICML) (2017)

[2] Chang, W., Shi, Y., Wang, J.: Csot: Curriculum and structure-aware optimal transport for learning with noisy labels. In: Advances in Neural Information Processing Systems. pp. 8528–8541 (2023)

[3] Chen, G., Yao, W., Song, X., Li, X., Rao, Y., Zhang, K.: Plot: Prompt learning with optimal transport for vision-language models. In: International Conference on Learning Representations (ICLR) (2023), <sub>https:</sub> //openreview.net/forum?id=b9APFSTylGT

[4] Chen, M., Cheng, H., Du, Y., Xu, M., Jiang, W., Wang, C.: Two wrongs don’t make a right: Combating confirmation bias in learning with label noise. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 37, pp. 14765–14773 (2023). <sub>https:</sub>//<sub>doi.org</sub>/<sub>10.1609</sub>/<sub>aaai.v37i12.26725</sub>

[5] Cimpoi, M., Maji, S., Kokkinos, I., Mohamed, S., Vedaldi, A.: Describing textures in the wild. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). pp. 3606–3613 (2014)

[6] Cuturi, M.: Sinkhorn distances: Lightspeed computation of optimal transport. In: Advances in Neural Information Processing Systems (2013)

[7] Fei-Fei, L., Fergus, R., Perona, P.: Learning generative visual models from few training examples: An incremental bayesian approach tested on 101 object categories. In: 2004 Conference on Computer Vision and Pattern Recognition Workshop (CVPRW). pp. 178–178. IEEE (2004)

[8] Feng, C., Ren, Y., Xie, X.: Ot-filter: An optimal transport filter for learning with noisy labels. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 16164–16174 (2023)

[9] Feng, L., Shu, S., Lin, Z., Lv, F., Li, L., An, B.: Can cross entropy loss be robust to label noise? In: Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI) (2020)

[10] Guo, Y., Gu, X.: Joapr: Cleaning the lens of prompt learning for vision language models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 28695–28705 (2024)

[11] Han, B., Yao, Q., Yu, X., Niu, G., Xu, M., Hu, W., Tsang, I., Sugiyama, M.: Co-teaching: Robust training of deep neural networks with extremely noisy labels. In: Advances in Neural Information Processing Systems (2018)

[12] Helber, P., Bischke, B., Dengel, A., Borth, D.: EuroSAT: A novel dataset and deep learning benchmark for land use and land cover classification. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing 12(7), 2217–2226 (2019)

[13] Hendrycks, D., Lee, K., Mazeika, M.: Using pre-training can improve model robustness and uncertainty. In: Proceedings of the International Conference on Machine Learning (ICML). pp. 2712–2721. PMLR (2019)

[14] Hinton, G.E., Vinyals, O., Dean, J.: Distilling the knowledge in a neural network. CoRR abs/1503.02531 (2015), <sub>https:</sub>//<sub>arxiv.org</sub>/<sub>abs</sub>/<sub>1503.</sub> 02531

[15] Jia, C., Yang, Y., Xia, Y., Chen, Y.T., Parekh, Z., Pham, H., Le, Q.V., Sung, Y.H., Li, Z., Duerig, T.: Scaling up visual and vision-language representation learning with noisy text supervision. In: Proceedings of the 38th International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 139, pp. 4904–4916. PMLR (2021), <sub>https:</sub>//<sub>proceedings.mlr.press</sub>/ v139/jia21b.html

[16] Jia, M., Tang, L., Chen, B., Cardie, C., Belongie, S.J., Hariharan, B., Lim, S.: Visual prompt tuning. In: Computer Vision – ECCV 2022. Lecture Notes in Computer Science, vol. 13693, pp. 709–727. Springer (2022). <sub>https:</sub> //doi.org/10.1007/978-3-031-19827-4\_41

[17] Khattak, M.U., Rasheed, H., Maaz, M., Khan, S., Khan, F.S.: Maple: Multi modal prompt learning. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 19113–19122 (2023)

[18] Khattak, M.U., Wasim, S.T., Naseer, M., Khan, S., Yang, M.H., Khan, F.S.: Self-regulating prompts: Foundational model adaptation without forgetting. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 15190–15200 (2023). <sub>https:</sub>//<sub>doi.org</sub>/<sub>10.</sub> 1109/ICCV51070.2023.01394

[19] Krause, J., Stark, M., Deng, J., Fei-Fei, L.: 3d object representations for finegrained categorization. In: Proceedings of the IEEE International Conference on Computer Vision Workshops (ICCVW). pp. 554–561 (2013)

[20] Lafon, M., Ramzi, E., Rambour, C., Audebert, N., Thome, N.: Gallop: Learning global and local prompts for vision-language models. In: European Conference on Computer Vision (ECCV) (2024)

[21] Laine, S., Aila, T.: Temporal ensembling for semi-supervised learning. In: International Conference on Learning Representations (ICLR) (2017)

[22] Lee, K.H., He, X., Zhang, L., Yang, L.: Cleannet: Transfer learning for scalable image classifier training with label noise. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). pp. 5447–5456 (2018)

[23] Li, J., Socher, R., Hoi, S.C.H.: Dividemix: Learning with noisy labels as semi-supervised learning. In: International Conference on Learning Representations (ICLR) (2020), <sub>https:</sub>//<sub>openreview.net</sub>/<sub>forum?id=HJgExaVtwr</sub>

[24] Li, Z., Li, X., Fu, X., Zhang, X., Wang, W., Chen, S., Yang, J.: Promptkd: Unsupervised prompt distillation for vision-language models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2024)

[25] Liang, K.J., Rangrej, S.B., Petrovic, V., Hassner, T.: Few-shot learning with noisy labels. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 9089–9098 (2022)

[26] Lu, Y., Liu, J., Zhang, Y., Liu, Y., Tian, X.: Prompt distribution learning. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 5206–5215 (June 2022)

[27] Lyu, Y., Tsang, I.W.: Curriculum loss: Robust learning and generalization against label corruption. In: International Conference on Learning Representations (ICLR) (2020), <sub>https:</sub>//<sub>openreview.net</sub>/<sub>forum?id=rkgt0REKwS</sub>

[28] Nilsback, M.E., Zisserman, A.: Automated flower classification over a large number of classes. In: 2008 Sixth Indian Conference on Computer Vision, Graphics & Image Processing. pp. 722–729. IEEE (2008)

[29] Pan, B., Li, Q., Tang, X., Huang, W., Fang, Z., Liu, F., Wang, J., Yu, J., Shi, Y.: Nlprompt: Noise-label prompt learning for vision-language models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 19963–19973 (2025)

[30] Parkhi, O.M., Vedaldi, A., Zisserman, A., Jawahar, C.V.: Cats and dogs. In: 2012 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). pp. 3498–3505. IEEE (2012)

[31] Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Kopf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J., Chintala, S.: Pytorch: An imperative style, high-performance deep learning library. In: Advances in Neural Information Processing Systems (NeurIPS). vol. 32 (2019)

[32] Patrini, G., Rozza, A., Menon, A.K., Nock, R., Qu, L.: Making deep neural networks robust to label noise: A loss correction approach. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2017)

[33] Peyré, G., Cuturi, M.: Computational optimal transport. Foundations and Trends in Machine Learning 11(5–6), 355–607 (2019). <sub>https:</sub>//<sub>doi.org</sub>/ 10.1561/2200000073

[34] Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., Sutskever, I.: Learning transferable visual models from natural language supervision. In: Proceedings of the 38th International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 139, pp. 8748–8763. PMLR (2021)

[35] Reed, S., Lee, H., Anguelov, D., Szegedy, C., Erhan, D., Rabinovich, A.: Training deep neural networks on noisy labels with bootstrapping. In: ICLR Workshop (2015)

[36] Shu, M., Nie, W., Huang, D., Yu, Z., Goldstein, T., Anandkumar, A., Xiao, C.: Test-time prompt tuning for zero-shot generalization in visionlanguage models. In: Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022 (NeurIPS 2022) (2022), <sub>http:</sub>//<sub>papers.nips.cc</sub>/<sub>paper\_files</sub>/<sub>paper</sub>/<sub>2022</sub>/<sub>hash</sub>/ 5bf2b802e24106064dc547ae9283bb0c-Abstract-Conference.html

[37] Song, H., Kim, M., Lee, J.G.: Selfie: Refurbishing unclean samples for robust deep learning. In: Proceedings of the 36th International Conference on Machine Learning (ICML). pp. 5907–5915. PMLR (2019)

[38] Soomro, K., Zamir, A.R., Shah, M.: UCF101: A dataset of 101 human action classes from videos in the wild. arXiv preprint arXiv:1212.0402 (2012)

[39] Szegedy, C., Vanhoucke, V., Iofe, S., Shlens, J., Wojna, Z.: Rethinking the inception architecture for computer vision. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). pp. 2818–2826 (2016). <sub>https:</sub>//<sub>doi.org</sub>/<sub>10.1109</sub>/<sub>CVPR.2016.308</sub>

[40] Tarvainen, A., Valpola, H.: Mean teachers are better role models: Weightaveraged consistency targets improve semi-supervised deep learning results. In: Advances in Neural Information Processing Systems (2017)

[41] Toneva, M., Sordoni, A., des Combes, R.T., Trischler, A., Bengio, Y., Gordon, G.J.: An empirical study of example forgetting during deep neural network learning. In: International Conference on Learning Representations (ICLR) (2019), <sub>https:</sub>//<sub>openreview.net</sub>/<sub>forum?id=BJlxm30cKm</sub>

[42] Wei, H., Feng, L., Chen, X., An, B.: Combating noisy labels by agreement: A joint training method with co-regularization. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 13726–13735 (2020)

[43] Wu, C.E., Tian, Y., Yu, H., Wang, H., Morgado, P., Hu, Y.H., Yang, L.: Why is prompt tuning for vision-language models robust to noisy labels? In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 15488–15497 (2023)

[44] Xia, X., Liu, T., Han, B., Gong, C., Wang, N., Ge, Z., Chang, Y.: Robust early-learning: Hindering the memorization of noisy labels. In: International Conference on Learning Representations (ICLR) (2021)

[45] Xu, T., Zhou, Y., Ji, K., Liang, Y.: When will gradient methods converge to max-margin classifier under relu models? (2018). <sub>https:</sub>//<sub>doi.org</sub>/<sub>10.</sub> 48550/arXiv.1806.04339

[46] Yao, Y., Liu, T., Han, B., Gong, M., Deng, J., Niu, G., Sugiyama, M.: Dual t: Reducing estimation error for transition matrix in label-noise learning. In: Advances in Neural Information Processing Systems (NeurIPS) (2020)

[47] Yoon, H.S., Yoon, E., Tee, J.T.J., Hasegawa-Johnson, M., Li, Y., Yoo, C.D.: C-tpt: Calibrated test-time prompt tuning for vision-language models via text feature dispersion. In: International Conference on Learning Representations (ICLR) (2024). <sub>https:</sub>//<sub>doi.org</sub>/<sub>10.48550</sub>/<sub>arXiv.2403.14119</sub>

[48] Yu, J., Wang, Z., Vasudevan, V., Yeung, L., Seyedhosseini, M., Wu, Y.: Coca: Contrastive captioners are image-text foundation models. Transactions on Machine Learning Research (2022), <sub>https:</sub>//<sub>openreview.net</sub>/<sub>forum?id=</sub> Ee277P3AYC

[49] Zhang, C., Bengio, S., Hardt, M., Recht, B., Vinyals, O.: Understanding deep learning requires rethinking generalization. In: International Conference on Learning Representations (ICLR) (2017)

[50] Zhang, Y., Niu, G., Sugiyama, M.: Learning noise transition matrix from only noisy labels via total variation regularization. In: Proceedings of the

38th International Conference on Machine Learning. pp. 12501–12512 (2021), https://proceedings.mlr.press/v139/zhang21n.html

[51] Zhang, Z., Sabuncu, M.R.: Generalized cross entropy loss for training deep neural networks with noisy labels. In: Advances in Neural Information Processing Systems (2018)

[52] Zhou, K., Yang, J., Loy, C.C., Liu, Z.: Conditional prompt learning for vision-language models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 16816–16825 (June 2022)

[53] Zhou, K., Yang, J., Loy, C.C., Liu, Z.: Learning to prompt for vision-language models. International Journal of Computer Vision 130(9), 2337–2348 (2022). https://doi.org/10.1007/s11263-022-01653-1

# TecoPrompt: Temporal-Conservative Prompt Learning for Vision-Language Models Supplementary Material

## A Additional Experiments

## A.1 Generalization of TecoPrompt

Notably, our method is not specific to CoOp, and can be readily extended to other prompt learning frameworks, such as PromptSRC and MaPLe. As shown in Tab. 5, we further validate its efectiveness on the UCF101 dataset, where integrating our method with both PromptSRC and MaPLe consistently yields excellent performance. These results suggest that our approach is generally applicable across diferent prompt learning paradigms and possesses strong robustness and transferability.

Table 5: The generalization of TecoPrompt.
<table><tr><td>Method/Noise Ratio</td><td>12.5% 25.0% 37.5%</td><td>50.0%</td><td>62.5%</td><td>75.0%</td></tr><tr><td>MaPLe</td><td>80.25 79.16 75.27</td><td>71.10</td><td>60.43</td><td>53.14</td></tr><tr><td>MaPLe+Ours PromptSRC</td><td>81.96 81.82 82.82 80.85 78.13</td><td>80.65 79.57 75.37</td><td>78.68 71.39</td><td>76.95 63.06</td></tr><tr><td>PromptSRC+Ours</td><td>83.62 82.51 81.90</td><td></td><td>80.47 79.89</td><td>77.14</td></tr></table>

## A.2 Experiments on SUN397

In Tab. 6, TecoPrompt consistently delivers the best performance on SUN397 under both symmetric and asymmetric label noise. Our method shows clear and consistent improvements across all noise rates. These results further verify the robustness and efectiveness of our approach on challenging noisy-label settings.

Table 6: Test accuracy(%) on SUN397.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Method</td><td colspan="6">Noise Rate: Sym</td><td colspan="6">Noise Rate: Asym</td></tr><tr><td>12.5% 25.0% 37.5% 50.0% 62.5% 75.0%</td><td></td><td></td><td></td><td></td><td></td><td>12.5% 25.0% 37.5% 50.0% 62.5% 75.0%</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="4">SUN397</td><td>CoOp</td><td>65.50</td><td>62.9</td><td>59.3</td><td>55.5</td><td>48.3</td><td>37.8</td><td>63.5</td><td>56.1</td><td>45.5</td><td>33.8</td><td>22.1</td><td>11.4</td></tr><tr><td>GCE</td><td>67.6</td><td>66.3</td><td>65.4</td><td>64.2</td><td>62.0</td><td>59.2</td><td>68.4</td><td>66.4</td><td>63.8</td><td>60.0</td><td>53.6</td><td>43.8</td></tr><tr><td>NLPrompt</td><td>68.4</td><td>67.5</td><td>66.4</td><td>64.8</td><td>64.1</td><td>61.7</td><td>68.7</td><td>67.5</td><td>66.1</td><td>64.0</td><td>61.4</td><td>53.0</td></tr><tr><td>TecoPrompt</td><td>68.9</td><td>68.5</td><td>68.1</td><td>67.3</td><td>66.0</td><td>65.6</td><td>69.4</td><td>68.3</td><td>67.4</td><td>66.1</td><td>65.1</td><td>60.1</td></tr></table>

## A.3 Additional Ablation Study

Table 7 further evaluates the efects of EMA, the temporal window size K, and the loss design on EuroSAT. The results show that EMA or temporal consistency alone provides limited gains, while their combination significantly improves robustness, with $\beta = 0 . 8$ and K = 8 achieving the best average accuracy. This

Table 7: Additional Ablation study on the EuroSAT dataset.
<table><tr><td>β(EMA)</td><td> $K$ </td><td>Loss</td><td>10% 30%</td><td>Noise Rate 50%</td><td>70%</td><td>Average</td></tr><tr><td rowspan="3">0.8</td><td>∞</td><td>Ours</td><td>78.23 75.84</td><td>41.15</td><td>19.29</td><td>53.63</td></tr><tr><td>∞</td><td>Ours</td><td>79.34 75.92</td><td>51.76</td><td>11.88</td><td>54.73</td></tr><tr><td>8</td><td>Ours</td><td>78.69 76.15</td><td>65.61</td><td>20.36</td><td>60.20</td></tr><tr><td rowspan="3">0.4</td><td>4</td><td>Ours</td><td>77.90 75.95</td><td>65.59</td><td>40.33</td><td>64.94</td></tr><tr><td>8</td><td>Ours</td><td>80.25 76.87</td><td>68.58</td><td>42.68</td><td>67.09</td></tr><tr><td>12</td><td>Ours</td><td>79.32 79.46</td><td>67.11</td><td>33.15</td><td>64.76</td></tr><tr><td rowspan="3">0.6</td><td>4</td><td>Ours</td><td>77.73 75.45</td><td>66.41</td><td>27.83</td><td>61.86</td></tr><tr><td>8</td><td>Ours</td><td>80.24 76.42</td><td>68.62</td><td>31.57</td><td>64.21</td></tr><tr><td>12</td><td>Ours</td><td>80.75 77.04</td><td>68.01</td><td>25.05</td><td>62.71</td></tr><tr><td rowspan="3">0.8</td><td>4</td><td>Ours</td><td>79.13 75.12</td><td>67.24</td><td>42.85</td><td>66.08</td></tr><tr><td>8</td><td>Ours</td><td>80.23 77.03</td><td>70.53</td><td>43.83</td><td>67.91</td></tr><tr><td>12</td><td>Ours</td><td>80.94 77.51</td><td>68.65</td><td>32.45</td><td>64.89</td></tr><tr><td rowspan="3">0.8</td><td>8</td><td>CE</td><td>79.28 72.33</td><td>58.76</td><td>33.32</td><td>60.92</td></tr><tr><td>8</td><td>GCE</td><td>78.89 74.47</td><td>65.67</td><td>35.61</td><td>63.66</td></tr><tr><td>8</td><td>MAE</td><td>76.33 72.03</td><td>45.37</td><td>21.46</td><td>53.80</td></tr></table>

confirms that EMA smoothing stabilizes confidence estimation and the K-epoch window helps avoid unreliable rewriting. Meanwhile, the performance drop with smaller or larger K indicates that a moderate window better balances correction precision and coverage. In addition, the proposed tri-group loss outperforms CE, GCE, and MAE, demonstrating the benefit of applying diferent losses to samples with diferent reliability levels.

## B Theoretical Analysis of Temporal-Consistency-Triggered OT Label Rewriting

In this section, we analyze when a temporally stable OT assignment can serve as a reliable rewriting candidate in TecoPrompt, and how enlarging the consistency window K improves rewrite precision.

## B.1 Notation and Definitions

Let $\mathcal { D } = \{ ( x _ { i } , \tilde { y } _ { i } ) \} _ { i = 1 } ^ { N }$ be a dataset with observed labels $\tilde { y } _ { i } \in \{ 1 , . . . , C \}$ . Each sample has an unknown ground-truth label $y _ { i } ^ { \star }$ . All probabilities are taken with respect to a uniformly random index i together with any algorithmic randomness.

Define the noise indicator and the noise rate as

$$
Z _ { i } \triangleq \mathbf { 1 } [ \widetilde { y } _ { i } \neq y _ { i } ^ { \star } ] , \qquad \eta \triangleq \mathbb { P } ( Z _ { i } = 1 ) .\tag{1}
$$

At epoch $s ,$ let $\hat { y } _ { i } ^ { ( s ) } \in \{ 1 , \ldots , C \}$ denote the OT-induced pseudo-label of sample i, and let $c _ { i } ^ { ( s ) } \in [ 0 , 1 ]$ be the corresponding confidence score.

Define the flip indicator of the OT pseudo-label trajectory by

$$
d _ { i } ^ { ( s ) } \triangleq \mathbf { 1 } \left[ \hat { y } _ { i } ^ { ( s + 1 ) } \neq \hat { y } _ { i } ^ { ( s ) } \right] .\tag{2}
$$

Fix a window $[ t _ { 0 } , t _ { 0 } + T _ { w } - 1 ]$ and define the flip rate as

$$
\phi _ { i } \triangleq \frac { 1 } { T _ { w } } \sum _ { s = t _ { 0 } } ^ { t _ { 0 } + T _ { w } - 1 } d _ { i } ^ { ( s ) } \in [ 0 , 1 ] .\tag{3}
$$

Given a threshold $\phi _ { 0 } \in ( 0 , 1 )$ , define the OT-forgettability indicator by

$$
F _ { i } \triangleq \mathbf { 1 } [ \phi _ { i } > \phi _ { 0 } ] , \qquad \tau \triangleq \mathbb { P } ( F _ { i } = 1 ) .\tag{4}
$$

We refer to samples with $F _ { i } = 0$ as OT-unforgettable and to samples with $F _ { i } = 1$ as OT-forgettable. Unlike the classical notion of forgetting, which is defined in terms of correctness with respect to $y _ { i } ^ { \star }$ , both $\phi _ { i }$ and $F _ { i }$ are directly observable from the OT pseudo-label trajectory.

The pair $( Z _ { i } , F _ { i } )$ induces the partition

$$
\begin{array} { r l r } {  { \frac { \bigl | F _ { i } = 0 , \mathrm { ~ O T - u n f o r g e t t a b l e } } { Z _ { i } = 0 , \mathrm { ~ c l e a n } } \qquad } } & { { } } & { F _ { i } = 1 , \mathrm { ~ O T - f o r g e t t a b l e } } \\ { Z _ { i } = 1 , \mathrm { ~ n o i s y } \qquad } & { { } } & { S _ { n , t } } \end{array}\tag{5}
$$

To characterize the coupling between noise and OT-forgettability, define

$$
\rho _ { u } \triangleq \mathbb { P } ( F _ { i } = 0 \mid Z _ { i } = 1 ) \in [ 0 , 1 ] \qquad ( \eta > 0 ) .\tag{6}
$$

Then the block masses are given by

$$
\pi _ { n , u } = \eta \rho _ { u } , \quad \pi _ { n , f } = \eta ( 1 - \rho _ { u } ) , \quad \pi _ { c , f } = \tau - \eta ( 1 - \rho _ { u } ) , \quad \pi _ { c , u } = 1 - \tau - \eta \rho _ { u } .\tag{7}
$$

Define the K-epoch stable-mismatch event by

$$
\mathcal E _ { K } ( i ) \triangleq \{ \hat { y } _ { i } ^ { ( t - K + 1 ) } = \cdots = \hat { y } _ { i } ^ { ( t ) } \neq \tilde { y } _ { i } \} .\tag{8}
$$

On $\mathcal { E } _ { K } ( i )$ , define the stable label as $\bar { y } _ { i } \triangleq \hat { y } _ { i } ^ { ( t ) }$

Define the K-epoch confidence gate by

$$
\mathcal { D } _ { K } ( i ) \overset { \Delta } { = } \{ c _ { i } ^ { ( t - K + 1 ) } \geq \theta , ~ . ~ . ~ . , ~ c _ { i } ^ { ( t ) } \geq \theta \} .\tag{9}
$$

Finally, define the rewrite-candidate event by

$$
\begin{array} { r } { \mathcal { A } _ { K } ( i ) \triangleq \mathcal { E } _ { K } ( i ) \cap \mathcal { D } _ { K } ( i ) . } \end{array}\tag{10}
$$

## B.2 Assumptions

Assumption 1. A one-step stability separation holds: conditional on past stability, OT-unforgettable samples retain the same OT pseudo-label with probability at least $p _ { u }$ , whereas OT-forgettable samples do so with probability at most $p _ { f }$ . Specifically, there exist constants $0 < p _ { f } < p _ { u } < 1$ such that for every $s \in \{ t - K + 1 , \ldots , t - 1 \}$ ，

$$
\mathbb { P } \Big ( \hat { y } ^ { ( s + 1 ) } = \hat { y } ^ { ( s ) } \Big | F = 0 , \ \hat { y } ^ { ( t - K + 1 ) } = \dots = \hat { y } ^ { ( s ) } \Big ) \geq p _ { u } ,\tag{11}
$$

whereas

$$
\mathbb { P } \Big ( \hat { y } ^ { ( s + 1 ) } = \hat { y } ^ { ( s ) } \Big | F = 1 , ~ \hat { y } ^ { ( t - K + 1 ) } = \cdot \cdot \cdot = \hat { y } ^ { ( s ) } \Big ) \leq p _ { f } .\tag{12}
$$

where $p _ { u }$ and $p _ { f }$ denote the corresponding one-step stability bounds.

Assumption 2. The confidence gate preserves stable candidates on the two $O T .$ unforgettable blocks with probabilities bounded below by $_ { q _ { c , u } }$ and $q _ { n , u }$ . Concretely, there exist constants $q _ { c , u } , q _ { n , u } \in ( 0 , 1 ]$ such that

$$
\mathbb { P } ( \mathcal { D } _ { K } \mid \mathcal { S } _ { c , u } , \mathcal { E } _ { K } ) \ge q _ { c , u } , \qquad \mathbb { P } ( \mathcal { D } _ { K } \mid \mathcal { S } _ { n , u } , \mathcal { E } _ { K } ) \ge q _ { n , u } .\tag{13}
$$

This assumption captures the idea that stable candidates are not frequently discarded by confidence thresholding.

Assumption 3. On the target block ${ \cal S } _ { n , u } ,$ rewritten labels are correct up to a residual error $\varepsilon _ { \star }$ . That is, there exists $\varepsilon _ { \star } \in [ 0 , 1 )$ such that

$$
\mathbb { P } ( \boldsymbol { \bar { y } } = \boldsymbol { y } ^ { \star } \mid S _ { n , u } , \mathcal { A } _ { K } ) \geq 1 - \varepsilon _ { \star } .\tag{14}
$$

where $\varepsilon _ { \star }$ denotes the residual error rate on $S _ { n , u }$

Assumption 4. The stable-mismatch event is exponentially most likely on $S _ { n , u }$ Specifically, there exist constants

$$
0 < p _ { f } < p _ { c , u } < p _ { n , u } < 1 , \qquad \varepsilon _ { c } , \varepsilon _ { n } \in [ 0 , 1 ) ,\tag{15}
$$

such that for every $K \geq 2$

$$
\mathbb { P } ( \mathcal { E } _ { K } \mid \mathcal { S } _ { n , u } ) \ge ( 1 - \varepsilon _ { n } ) p _ { n , u } ^ { K - 1 } , \qquad \mathbb { P } ( \mathcal { E } _ { K } \mid \mathcal { S } _ { c , u } ) \le \varepsilon _ { c } p _ { c , u } ^ { K - 1 } ,\tag{16}
$$

and

$$
\mathbb { P } ( \mathcal { E } _ { K } \mid \mathcal { S } _ { c , f } ) \leq p _ { f } ^ { K - 1 } , \qquad \mathbb { P } ( \mathcal { E } _ { K } \mid \mathcal { S } _ { n , f } ) \leq p _ { f } ^ { K - 1 } .\tag{17}
$$

Thus, $p _ { n , u }$ is the dominant rate.

## B.3 Main Results and Proofs

Lemma 1. The following lemma quantifies the stability gap between OT-unforgettable and OT-forgettable samples over a K-epoch window.

Under Assumption 1, for any $K \geq 2$

$$
\begin{array} { r } { \mathbb { P } ( \hat { y } ^ { ( t - K + 1 ) } = \cdot \cdot \cdot = \hat { y } ^ { ( t ) } \mid F = 0 ) \geq p _ { u } ^ { K - 1 } , } \\ { \mathbb { P } ( \hat { y } ^ { ( t - K + 1 ) } = \cdot \cdot \cdot = \hat { y } ^ { ( t ) } \mid F = 1 ) \leq p _ { f } ^ { K - 1 } . } \end{array}\tag{18}
$$

Proof. For $s \in \{ t - K + 1 , \ldots , t - 1 \}$ , let $G _ { s } \triangleq \{ \hat { y } ^ { ( s + 1 ) } = \hat { y } ^ { ( s ) } \}$ . Then

$$
\{ \hat { y } ^ { ( t - K + 1 ) } = \cdots = \hat { y } ^ { ( t ) } \} = \bigcap _ { s = t - K + 1 } ^ { t - 1 } G _ { s } .
$$

If $F = 0$ , the chain rule gives

$$
\mathbb P \left( \prod _ { s = t - K + 1 } ^ { t - 1 } G _ { s } \Bigg | F = 0 \right) = \prod _ { s = t - K + 1 } ^ { t - 1 } \mathbb P ( G _ { s } | F = 0 , \ G _ { t - K + 1 } , \dots , G _ { s - 1 } ) .\tag{19}
$$

Since $G _ { t - K + 1 } , \dots , G _ { s - 1 }$ imply $\hat { y } ^ { ( t - K + 1 ) } = \cdot \cdot \cdot = \hat { y } ^ { ( s ) }$ , each factor is at least $p _ { u }$ by (11). As there are $K - 1$ such factors, we obtain

$$
\mathbb { P } ( \hat { y } ^ { ( t - K + 1 ) } = \cdot \cdot \cdot = \hat { y } ^ { ( t ) } \mid F = 0 ) \geq p _ { u } ^ { K - 1 } .
$$

The case $F = 1$ is analogous: by (12), each factor is at most $p _ { f }$ , so

$$
\mathbb { P } ( \hat { y } ^ { ( t - K + 1 ) } = \cdot \cdot \cdot = \hat { y } ^ { ( t ) } \mid F = 1 ) \leq p _ { f } ^ { K - 1 } .
$$

Theorem 1. The following result reduces rewrite precision to the proportion of $S _ { n , u }$ within the candidate set $\boldsymbol { \mathcal { A } } _ { K }$

Under Assumption 3,

$$
\mathbb { P } ( \bar { y } = y ^ { \star } \mid \mathcal { A } _ { K } ) \ge ( 1 - \varepsilon _ { \star } ) \mathbb { P } ( S _ { n , u } \mid \mathcal { A } _ { K } ) .\tag{20}
$$

Proof. Since $S _ { c , u } , S _ { c , f } , S _ { n , u }$ , and $\mathcal { S } _ { n , f }$ form a partition, the law of total probability under $\boldsymbol { \mathcal { A } } _ { K }$ gives

$$
\mathbb { P } ( \bar { y } = y ^ { \star } \mid \mathcal { A } _ { K } ) = \sum _ { B \in \{ \mathcal { S } _ { c , u , \mathcal { S } _ { c , f } , \mathcal { S } _ { n , u } , \mathcal { S } _ { n , f } \} } } \mathbb { P } ( \bar { y } = y ^ { \star } \mid \mathcal { A } _ { K } , B ) \mathbb { P } ( B \mid \mathcal { A } _ { K } ) .\tag{21}
$$

Retaining only the nonnegative term corresponding to $S _ { n , u }$ yields

$$
\mathbb { P } ( \bar { y } = y ^ { \star } \mid \mathcal { A } _ { K } ) \ge \mathbb { P } ( \bar { y } = y ^ { \star } \mid \mathcal { A } _ { K } , \mathcal { S } _ { n , u } ) \mathbb { P } ( \mathcal { S } _ { n , u } \mid \mathcal { A } _ { K } ) .
$$

Assumption 3 bounds the first factor from below by $1 - \varepsilon _ { \star }$ , proving (20).

Theorem 2. The following theorem shows that, as K increases, the candidate set $\boldsymbol { \mathcal { A } } _ { K }$ becomes increasingly concentrated on the noisy OT-unforgettable block. Under Assumptions 4 and 2, for every $K \geq 2$

$$
\mathbb { P } ( S _ { n , u } \mid { \mathcal A } _ { K } ) \ge \frac { q _ { n , u } ( 1 - \varepsilon _ { n } ) \pi _ { n , u } } { q _ { n , u } ( 1 - \varepsilon _ { n } ) \pi _ { n , u } + \varepsilon _ { c } \left( \frac { p _ { c , u } } { p _ { n , u } } \right) ^ { K - 1 } \pi _ { c , u } + \left( \frac { p _ { f } } { p _ { n , u } } \right) ^ { K - 1 } ( \pi _ { c , f } + \pi _ { n , f } ) } .\tag{22}
$$

Moreover, the right-hand side is nondecreasing in K and converges to 1 as $K  \infty$

Proof. By Bayes’ rule and the four-block partition,

$$
\mathbb { P } ( S _ { n , u } \mid { \mathcal A } _ { K } ) = \frac { \mathbb { P } ( { \mathcal A } _ { K } \mid { \mathcal S } _ { n , u } ) \pi _ { n , u } } { \mathbb { P } ( { \mathcal A } _ { K } \mid { \mathcal S } _ { n , u } ) \pi _ { n , u } + \mathbb { P } ( { \mathcal A } _ { K } \mid { \mathcal S } _ { c , u } ) \pi _ { c , u } + \mathbb { P } ( { \mathcal A } _ { K } \mid { \mathcal S } _ { c , f } ) \pi _ { c , f } + \mathbb { P } ( { \mathcal A } _ { K } \mid { \mathcal S } _ { n , f } ) \pi _ { n , f } } .
$$

For the target block,

$$
\begin{array} { r } { \mathbb { P } ( \mathcal { A } _ { K } \mid \mathcal { S } _ { n , u } ) = \mathbb { P } ( \mathcal { D } _ { K } \mid \mathcal { S } _ { n , u } , \mathcal { E } _ { K } ) \mathbb { P } ( \mathcal { E } _ { K } \mid \mathcal { S } _ { n , u } ) \geq q _ { n , u } ( 1 - \varepsilon _ { n } ) p _ { n , u } ^ { K - 1 } , } \end{array}\tag{23}
$$

where we used Assumptions 2 and 4. For the remaining blocks, the inclusion $\mathcal { A } _ { K } \subseteq \mathcal { E } _ { K }$ gives

$$
\begin{array} { r } { \mathbb { P } ( \mathcal { A } _ { K } \mid \mathcal { S } _ { c , u } ) \leq \varepsilon _ { c } p _ { c , u } ^ { K - 1 } , \qquad \mathbb { P } ( \mathcal { A } _ { K } \mid \mathcal { S } _ { c , f } ) \leq p _ { f } ^ { K - 1 } , \qquad \mathbb { P } ( \mathcal { A } _ { K } \mid \mathcal { S } _ { n , f } ) \leq p _ { f } ^ { K - 1 } . } \end{array}
$$

Substituting these bounds into (23) yields

$$
\mathbb { P } ( S _ { n , u } \mid { \mathcal { A } } _ { K } ) \geq { \frac { q _ { n , u } ( 1 - \varepsilon _ { n } ) p _ { n , u } ^ { K - 1 } \pi _ { n , u } } { q _ { n , u } ( 1 - \varepsilon _ { n } ) p _ { n , u } ^ { K - 1 } \pi _ { n , u } + \varepsilon _ { c } p _ { c , u } ^ { K - 1 } \pi _ { c , u } + p _ { f } ^ { K - 1 } ( \pi _ { c , f } + \pi _ { n , f } ) } } .\tag{24}
$$

Dividing the numerator and denominator by $p _ { n , u } ^ { K - 1 }$ gives (22). Since

$$
0 < \frac { p _ { c , u } } { p _ { n , u } } < 1 , \qquad 0 < \frac { p _ { f } } { p _ { n , u } } < 1 ,
$$

the two contamination terms decrease monotonically to zero as K increases, whereas the leading term $q _ { n , u } ( 1 - \varepsilon _ { n } ) \pi _ { n , u }$ remains constant. Hence, the lower bound is nondecreasing in K and converges to 1.

Theorem 3. Combining the previous two results yields an explicit lower bound on rewrite precision.

Under Theorem 1 and Assumptions ${ \mathcal { Q } } - { \mathcal { 4 } } ;$

$$
\mathbb { P } ( \bar { y } = y ^ { \star } \mid A _ { K } ) \ge ( 1 - \varepsilon _ { \star } ) \frac { q _ { n , u } ( 1 - \varepsilon _ { n } ) \pi _ { n , u } } { q _ { n , u } ( 1 - \varepsilon _ { n } ) \pi _ { n , u } + \varepsilon _ { c } \left( \frac { p _ { c , u } } { p _ { n , u } } \right) ^ { K - 1 } \pi _ { c , u } + \left( \frac { p _ { f } } { p _ { n , u } } \right) ^ { K - 1 } ( \pi _ { c , f } + \pi _ { n , f } ) } .\tag{25}
$$

Hence, the lower bound on rewrite precision is nondecreasing in K and converges to $1 - \varepsilon$ <sub>⋆</sub> as $K  \infty$

Proof. Theorem 1 gives

$$
\mathbb { P } ( \bar { y } = y ^ { \star } \mid \mathcal { A } _ { K } ) \ge ( 1 - \varepsilon _ { \star } ) \mathbb { P } ( S _ { n , u } \mid \mathcal { A } _ { K } ) ,
$$

and Theorem 2 lower-bounds the second factor by the fraction in (22). Substituting this bound proves (25). The monotonicity and limiting value follow immediately from Theorem 2, since the prefactor $( 1 - \varepsilon _ { \star } )$ is independent of K.