# Quantifying Annotation Agreement in Object Detection with Optimal Transport

Kirill Prokhorov¹ and Alexandr A. Kalinin20

1 Openface.io, Vilnius, Lithuania

prokhorov@openface.me

Broad Institute of MIT and Harvard, Cambridge, MA, USA

Abstract. Reliable validation of machine learning models is a crucial step in designing robust and trustworthy systems. Yet validation relies on annotated test sets treated as ground truth, even though annotations can be noisy or ambiguous. In object detection, sources may disagree on object presence, box location, or class label. Standard agreement metrics usually binarize box overlap before computing a summary score, obscuring the degree and structure of disagreement between annotation sets. Here, we study an Optimal Transport (OT)-based metric for measuring agreement between object detection annotations. Instead of binarizing bounding box overlap, this metric assigns graded costs to matched and unmatched labeled boxes. Through theoretical analysis and experiments on real and synthetically perturbed annotations, we show that OT captures graded changes in agreement and distinguishes annotation pairs that thresholded metrics score identically. Our results suggest that OT provides a richer measure of annotation agreement in object detection, especially for domains with many small or ambiguous objects. Such agreement estimates can support more reliable validation when labels are noisy or conflicting.

Keywords: Object detection · Optimal transport · Annotator agreement · Model validation · Noisy labels

## 1 Introduction

Modern machine learning systems rely on annotated data for training and validation, typically treating a fixed test set as a gold standard. Yet a single noise-free ground truth may not exist: multiple plausible annotations can arise from task ambiguity, limited visual evidence, or differences in expert judgment. This is especially relevant in object detection, where annotators may disagree on object presence, spatial extent, or class labels [4, 14, 21]. Such disagreement contains information about the reliability of the reference data. A prediction that differs from one annotation may remain consistent with another plausible interpretation, so performance against a single reference can obscure annotation uncertainty. Measuring agreement is therefore important for assessing both dataset quality and the reliability of model validation.

Agreement in object detection is commonly assessed by matching boxes with an Intersection-over-Union (IoU) threshold and then computing a summary statistic such as F1 or Krippendorff's α [9, 22]. This approach is simple and useful, but the thresholding step discards spatial information. After overlap is converted into a binary match decision, the resulting score may no longer reflect how close the boxes are, how severe the mismatch is, or how unmatched boxes are arranged. Thus, annotation pairs with different spatial structure can receive the same agreement score.

In this work, we study an Optimal Transport (OT)-based metric for measuring agreement between object detection annotations. The formulation builds on recent OT-based work for object detection evaluation [15], but shifts the focus from comparing model predictions with a reference to comparing annotation sets themselves. Each annotation is treated as a set of labeled bounding boxes, and agreement is computed through a transport problem whose cost combines localization, label consistency, and penalties for unmatched boxes.

Our main contributions are:

We adapt an OT-based detection metric to quantify agreement between object detection annotations, jointly accounting for localization, class labels, and missing or extra boxes.

— We analyze its bounds and special cases, clarifying how object count, localization cost, and matching penalties affect the score.

— We evaluate OT on real and synthetically perturbed annotations from multiple domains, identifying cases where it captures distinctions collapsed by threshold-based agreement metrics.

Together, these results support OT-based agreement as a practical tool for assessing object detection annotations and for making model validation more sensitive to noisy or conflicting labels. The source code is publicly available at https://github.com/openface-io/ot-agreement.

## 2 Related Work

## 2.1 Uncertainty and disagreement in object detection annotations

Annotation uncertainty has been addressed by modeling label noise or learning from repeated annotations [21, 25], using richer targets such as label distributions [7, 18, 27], representing localization uncertainty probabilistically [11], and aggregating annotations into a latent ground truth [10, 25, 26]. Other work characterizes model performance under contradictory labels [23]. Our goal is complementary: we measure pairwise agreement and discrepancy directly between object detection annotations, either to characterize annotation quality or as a component of multi-reference validation.

## 2.2 Agreement and evaluation metrics for object detection

Object detection commonly uses IoU-threshold matching followed by metrics such as precision, recall, F1, or mAP. Similar matching is used when adapting agreement statistics such as Krippendorff's α to object detection [9, 22]. Although interpretable, hard thresholds discard information about the structure and severity of spatial disagreement.

Methods such as the discrepancy ratio [12] compare model predictions with multiple human annotations and therefore depend on an underlying pairwise agreement measure. A more expressive pairwise measure can improve how such methods characterize annotation variability and model performance relative to it.

## 2.3 Optimal transport for detection evaluation

Optimal Transport provides a natural framework for comparing structured sets by computing a minimum-cost assignment between elements. With dummy elements or related unbalanced formulations, it can also account for unmatched or poorly matched elements. More broadly, transport-style objectives have been used to compare distributions, align structured representations, and solve setmatching problems [3, 6, 20].

In object detection, Otani et al. [15] proposed an OT-based evaluation metric, Optimal Correction Cost (OC-cost), building on earlier work [6, 16]. OC-cost compares detector predictions with ground-truth boxes and penalizes localization errors, label mismatches, false positives, and false negatives within a single transport problem. This produces a fine-grained measure of detection quality and addresses limitations of standard mAP-style evaluation.

Our work uses the same general OT perspective but changes the object of comparison. Rather than evaluating detector predictions against a fixed ground truth with confidence scores, we compare annotation sets directly. This shift removes the need to treat either annotation as error-free and allows OT to serve as a measure of annotation agreement or discrepancy. In this setting, the transport cost captures spatial proximity, label consistency, and missing or extra boxes, providing a graded alternative to thresholded agreement metrics.

## 3 Method

## 3.1 Optimal Transport score formulation

Our formulation follows Otani et al. [15], who use OT to compare detections with ground truth. In the multi-annotator setting, we compare only bounding boxes and class labels, without detector confidence scores.

Let $\mathcal { A } = \{ b _ { 1 } ^ { A } , \ldots , b _ { M } ^ { A } \}$ and $B = \{ b _ { 1 } ^ { B } , \ldots , b _ { N } ^ { B } \}$ be two sets of M and N labeled bounding boxes. Let $l _ { i } ^ { A }$ and $l _ { j } ^ { B }$ denote their corresponding class labels. The pairwise OT cost is

$$
c _ { i j } = \lambda c _ { \mathrm { l o c } } ( b _ { i } ^ { \mathscr { A } } , b _ { j } ^ { \mathscr { B } } ) + ( 1 - \lambda ) c _ { \mathrm { l b l } } ( l _ { i } ^ { \mathscr { A } } , l _ { j } ^ { \mathscr { B } } ) ,\tag{1}
$$

where $\lambda \in [ 0 , 1 ]$ is a trade-off hyperparameter that balances localization and label costs. The localization cost is defined using Generalized IoU (GIoU) [19]:

$$
c _ { \mathrm { l o c } } ( b _ { i } ^ { A } , b _ { j } ^ { B } ) = \frac { 1 - G I o U ( b _ { i } ^ { A } , b _ { j } ^ { B } ) } { 2 } , \quad c _ { \mathrm { l o c } } ( b _ { i } ^ { A } , b _ { j } ^ { B } ) \in [ 0 , 1 ] .\tag{2}
$$

The properties of GIoU are well suited to our task, as they yield meaningful distances even for non-overlapping boxes. The label cost is given by

$$
c _ { \mathrm { l b l } } ( l _ { i } ^ { \mathcal { A } } , l _ { j } ^ { \mathcal { B } } ) = \left\{ 0 , \mathrm { i f } l _ { i } ^ { \mathcal { A } } = l _ { j } ^ { \mathcal { B } } , \right.\tag{3}
$$

We then construct the cost matrix:

$$
C = \left( \begin{array} { c c c c } { { c _ { 1 1 } } } & { { \cdots } } & { { c _ { 1 N } } } & { { \beta } } \\ { { } } & { { } } & { { } } & { { } } \\ { { \vdots } } & { { \ddots } } & { { \vdots } } & { { \vdots } } \\ { { c _ { M 1 } } } & { { \cdots } } & { { c _ { M N } } } & { { \beta } } \\ { { \beta } } & { { \cdots } } & { { \beta } } & { { \beta } } \end{array} \right) ,\tag{4}
$$

where the last row and column correspond to dummy boxes with constant cost $\beta \in [ 0 , 1 ]$ . A real box may match a dummy when no suitable real counterpart exists, representing a missing or extra object.

Within the OT framework, we formulate the task as mass transportation from one set to the other. We specify the source and target marginal distributions as weight vectors [17]:

$$
\mathbf { a } = \underbrace { [ 1 , 1 , . . . , 1 } _ { M } , N ] , \quad \mathbf { b } = \underbrace { [ 1 , 1 , . . . , 1 } _ { N } , M ] ,\tag{5}
$$

where the last entry in each vector corresponds to the dummy box. Unit capacity is assigned to each real box and an aggregate capacity to the dummy boxes. Each real box carries unit mass; because the marginals are integral, the transport problem admits an integral optimal assignment.

The optimal transport plan π is then obtained by solving the optimization problem:

$$
\begin{array} { r l } { { \pi ^ { * } = \displaystyle \mathrm { a r g m i n } \sum _ { \pi } ^ { M + 1 } \sum _ { i = 1 } ^ { N + 1 } c _ { i j } \pi _ { i j } , } } & { { } } \\ { { \mathrm { s u b j e c t ~ t o } \sum _ { j = 1 } ^ { N + 1 } \pi _ { i j } = a _ { i } , } } & { { \displaystyle \sum _ { i = 1 } ^ { M + 1 } \pi _ { i j } = b _ { j } \quad \forall i , j , } } \\ { { \displaystyle \sum _ { i = 1 } ^ { M + 1 } a _ { i } = \sum _ { j = 1 } ^ { N + 1 } b _ { j } , } } & { { } } \\ { { \pi _ { i j } \geq 0 \quad \forall i , j . } } & { { } } \end{array}\tag{6}
$$

After computing the optimal transport plan $\pi ^ { * } \ \mathrm { ( e . g } $ using solvers implemented in [5]), we ignore dummy-to-dummy mass by setting $\pi _ { M + 1 , N + 1 } ^ { * } = 0$ The final OT cost is then defined as

$$
\tilde { c } = \frac { 1 } { \sum _ { i = 1 } ^ { M + 1 } \sum _ { j = 1 } ^ { N + 1 } \pi _ { i j } ^ { * } } \sum _ { i = 1 } ^ { M + 1 } \sum _ { j = 1 } ^ { N + 1 } c _ { i j } \pi _ { i j } ^ { * } .\tag{7}
$$

Let k denote the number of real-real matches. We can then write the normalization factor as

$$
Z = \sum _ { i = 1 } ^ { M + 1 } \sum _ { j = 1 } ^ { N + 1 } \pi _ { i j } ^ { * } = \underbrace { k } _ { \mathrm { \tiny { r e a l - r e a l } } } + \underbrace { ( M - k ) } _ { \mathrm { \tiny { r e a l } } \ B \to \mathrm { d u m m y } \ A } + \underbrace { ( N - k ) } _ { \mathrm { \tiny { r e a l } } \ A \to \mathrm { d u m m y } \ B } = M + N - k .
$$

We constrain $\beta \leq 1$ , matching the range of the pairwise cost in Eq. 1. Here $\beta$ is the cost of a real-dummy match. Because dummy matching provides a feasible fallback, the optimal normalized OT cost is at most $\beta .$ Thus the OT cost lies in $[ 0 , \beta ]$ , while the corresponding score $O T _ { \mathrm { s c o r e } } = 1 - O T _ { \mathrm { c o s t } }$ lies in $[ 1 - \beta , 1 ]$

The raw OT score measures pairwise similarity but does not account for agreement expected by chance. We therefore additionally define a dataset-level, chance-corrected agreement coefficient using the $\alpha = 1 - D _ { o } / D _ { \epsilon }$ form [2]:

$$
\alpha _ { \mathrm { O T } } = 1 - \widehat { D } _ { o } / \widehat { D } _ { e } ,\tag{8}
$$

where $\widehat { D } _ { o }$ averages OT costs between annotations of the same image and $\widehat { D } _ { e }$ is estimated from Monte Carlo-sampled cross-image pairs. Thus $\alpha _ { \mathrm { O T } } = 1 , 0 ,$ and $< 0$ denote perfect, chance-level, and below-chance agreement, respectively.

## 3.2 Special cases and theoretical boundaries

We next study how the OT score behaves under simplified settings that make the transport plan interpretable. For arbitrary annotations, the optimal plan depends on the number of boxes, labels, and hyperparameters and must generally be computed numerically. To obtain analytical insight, we consider special cases in which the matching structure is known or can be approximated. These cases clarify how the score depends on object count, spatial mismatch, and the dummymatching penalty.

Case: dummy matching only for unequal box counts Consider two sets of unlabeled $( c _ { \mathrm { { l b l } } } = 0 )$ annotations $\mathcal { A } = \{ b _ { 1 } ^ { \bar { A } } , \ldots , b _ { M } ^ { A } \}$ and $B = \{ b _ { 1 } ^ { B } , \ldots , b _ { N } ^ { B } \}$ . If $\lambda \le \beta \le 1$ , real-real matches are no more costly than dummy matches. Thus dummy matches occur only when the annotations contain different numbers of boxes.

If all real boxes have to match, then $Z = \operatorname* { m a x } ( M , N )$ and upper and lower bounds for OT score have the form:

$$
1 - \frac { 1 } { Z } \left( \underbrace { \beta | M - N | } _ { \mathrm { d i f f . ~ n u m . ~ o f ~ b o x e s } } + \underbrace { \lambda \operatorname* { m i n } ( M , N ) } _ { \mathrm { m a x ~ s p a t i a l ~ m i s m a t c h } } \right) \leq 1 - O T \leq 1 - \frac { 1 } { Z } \underbrace { \beta | M - N | } _ { \mathrm { d i f f . ~ n u m . ~ o f ~ b o x e } }\tag{9}
$$

The minimum possible value depends on the box-count mismatch and the maximum spatial mismatch $( c _ { \mathrm { l o c } } = 1 )$ . The maximum is attained when matched

![](images/319170fb0f9cd9d301e34f611e9f0be5526a0b9bbc4dddb767df34b98cf89017.jpg)  
Fig. 1: OT score bounds and corresponding range of values as a function of the ratio between the numbers of boxes in two annotations. When the numbers of boxes coincide, the range equals $\lambda ,$ while $\beta$ controls the maximum (minimum) values of the cost (score). All curves asymptotically converge as log $_ { \textrm { \scriptsize 2 } } \bigg ( \frac { N } { M } \bigg )  \pm \infty$ . The sensitivity of the optimal transport metric to spatial localization gives rise to the full range of possible values.

boxes are perfectly aligned $( c _ { \mathrm { l o c } } = 0 ) . { \mathrm { I f ~ } } \beta < \lambda$ , all boxes may match to dummies in the worst case, and the lower bound becomes $1 - \beta$

We can expand $\operatorname { E q . 9 }$ for two cases:

$$
\left\{ \begin{array} { l l } { 1 - \big ( \beta ( 1 - \frac { N } { M } ) + \lambda \frac { N } { M } \big ) \leq 1 - \mathrm { O T } \leq 1 - \beta ( 1 - \frac { N } { M } ) , } & { \mathrm { i f ~ } N \leq M } \\ { 1 - \big ( \beta ( 1 - \frac { M } { N } ) + \lambda \frac { M } { N } \big ) \leq 1 - \mathrm { O T } \leq 1 - \beta ( 1 - \frac { M } { N } ) , } & { \mathrm { i f ~ } N > M } \end{array} \right.\tag{10}
$$

It is convenient to plot Eq. 10 against $\log _ { 2 } \left( { \frac { N } { M } } \right)$ so it becomes symmetric. Fig. 1 shows boundaries and range of possible values for two cases with different $\lambda$ and β. Note that when the difference in the number of boxes is small, the range of metric values is large and equal to λ when $\begin{array} { r } { \log _ { 2 } \left( \frac { N } { M } \right) = 0 } \end{array}$ . When the number of boxes differs significantly, the range is small, because the metric is poor in any case regardless of box locations. As $\log _ { 2 } \left( \frac { N } { M } \right) \to \pm \infty$ both upper and lower bounds approach $1 - \beta$ and the range of possible values shrinks to zero.

Case: dependence on $\beta$ under the noisy-box assumption. Consider two sets of unlabeled boxes $\mathcal { A } = \{ b _ { 1 } ^ { \mathcal { A } } , \ldots , b _ { M } ^ { \mathcal { A } } \}$ and $B = \{ b _ { 1 } ^ { B } , \ldots , b _ { M } ^ { B } \}$ . We analyze how $\beta$ affects whether boxes are matched to real boxes or to dummies.

Because this example includes only the spatial component of the OT cost, we set $\lambda = 1$ and assume for simplicity that corresponding boxes have the same dimensions $( w , h )$ . For each matched pair, assume that the box centers differ by noise:

$$
\left\{ { \begin{array} { l } { x _ { c 2 } = x _ { c 1 } + \varepsilon _ { x } } \\ { y _ { c 2 } = y _ { c 1 } + \varepsilon _ { y } } \end{array} } \right.\tag{11}
$$

In order to find GIoU for spatial cost term in Eq. 2, we have to define intersection $I = \operatorname* { m a x } ( 0 , w - | \varepsilon _ { x } | ) \cdot \operatorname* { m a x } ( 0 , h - | \varepsilon _ { y } | )$ , union $U = 2 w h - I$ and the area of the smallest enclosing box for two boxes $| C | = ( w + | \varepsilon _ { x } | ) ( h + | \varepsilon _ { y } | )$ . This gives the following expression for GIoU:

$$
G I o U = I o U - \frac { | C \setminus U | } { | C | } = \frac { I } { 2 w h - I } - \frac { ( w + | \varepsilon _ { x } | ) ( h + | \varepsilon _ { y } | ) - ( 2 w h - I ) } { ( w + | \varepsilon _ { x } | ) ( h + | \varepsilon _ { y } | ) } .\tag{12}
$$

A real-real match is preferred to a dummy match when $c _ { i j } < \beta .$ while $c _ { i j } > \beta$ favors matching to the dummy. Therefore, the probability of a real-real match is

$$
P ( c _ { i j } < \beta ) = F _ { c } ( \beta ) = \int _ { 0 } ^ { \beta } f _ { c } ( t ) d t = \iint _ { c ( \xi , \eta ) < \beta } f _ { | \varepsilon _ { x } | , | \varepsilon _ { y } | } ( \xi , \eta ) d \xi d \eta\tag{13}
$$

where $f _ { | \varepsilon _ { x } | , | \varepsilon _ { y } | } ( | \varepsilon _ { x } | , | \varepsilon _ { y } | )$ is the joint PDF for the noise components $\left| \varepsilon _ { x } \right|$ and $\left| \varepsilon _ { y } \right|$

To evaluate this probability, we need the distribution of the cost c. Since only the noise distribution is assumed known, we instead integrate over the joint density $f _ { | \varepsilon _ { x } | , | \varepsilon _ { y } | } ( \xi , \eta )$ . The remaining task is therefore to determine the boundary $c ( | \varepsilon _ { x } | , | \varepsilon _ { y } | ) = \ddot { \beta }$ , which defines the integration domain in Eq. 13.

To derive this boundary from Eq. 12, we consider separately the cases $I > 0$ and $I = 0$ . The former corresponds to $| \varepsilon _ { x } | < w$ and $| \varepsilon _ { y } | < h$ , whereas the latter occurs when either $\left| \varepsilon _ { x } \right| \geq w$ or $| \varepsilon _ { y } | \geq h$ . Introducing the normalized displacements $u = | \varepsilon _ { x } | / u$ and $\vartheta = | \varepsilon _ { y } | / h$ , and using Eq. 2, the condition $c ( | \varepsilon _ { x } | , | \varepsilon _ { y } | ) = \beta$ is equivalent to $G I o U = 1 - 2 \beta$ . For the case $I = 0$ , this gives:

$$
G I o U _ { I = 0 } = - \frac { ( 1 + u ) ( 1 + \vartheta ) - 2 } { ( 1 + u ) ( 1 + \vartheta ) } = 1 - 2 \beta .\tag{14}
$$

Similarly, for $I \neq 0 { : }$

$$
G I o U _ { I \ne 0 } = \frac { ( 1 - u ) ( 1 - \vartheta ) } { 2 - ( 1 - u ) ( 1 - \vartheta ) } - \frac { ( 1 + u ) ( 1 + \vartheta ) - 2 + ( 1 - u ) ( 1 - \vartheta ) } { ( 1 + u ) ( 1 + \vartheta ) } = 1 - 2 \beta .\tag{15}
$$

From Eq. 14 it is easy to find $\vartheta _ { I = 0 } ( u )$ ; Eq. 15 can be rewritten as a quadratic equation $A ( u ) \vartheta ^ { 2 } + B ( u ) \vartheta + C ( u ) = 0$ and solved in order to obtain $\vartheta _ { I \neq 0 } ( u )$ These functions determine the integration limits $a _ { 1 } ( u ) , a _ { 2 } ( u ) , a _ { 3 } ( u )$ for the three regions. Assuming independent noise components, Eq. 13 becomes

$$
\begin{array} { l } { \displaystyle F _ { c } ( \beta ) = \int _ { 0 } ^ { 1 } f _ { u } ( \xi ) F _ { \vartheta } ( a _ { 1 } ( \xi ) ) d \xi + \int _ { 0 } ^ { 1 } f _ { u } ( \xi ) \left[ F _ { \vartheta } ( a _ { 2 } ( \xi ) ) - F _ { \vartheta } ( 1 ) \right] d \xi + } \\ { \displaystyle + \int _ { 1 } ^ { \infty } f _ { u } ( \xi ) F _ { \vartheta } ( a _ { 3 } ( \xi ) ) d \xi } \end{array}\tag{16}
$$

Detailed derivation can be found in the supplementary material. Given a PDF for the noise distribution, these integrals can be evaluated numerically.

Fig. 2a shows the theoretical CDF of the box-matching cost as a function of $\beta$ under Uniform and Half-Normal perturbations, together with empirical estimates on ACNE04 [27]. For visualization, we use the annotation set from [8], which contains substantially more objects than the original annotation. Each curve corresponds to a different noise level $\gamma$ As $\beta$ increases, a perturbed box is more likely to match a real box rather than the dummy box, because $\beta$ sets the penalty for unmatched boxes. Higher noise levels shift the curves toward larger $\beta \colon$ when boxes are strongly perturbed, real-real matches require a larger dummy penalty to remain preferable. The theoretical and empirical curves agree well for small perturbations, where each perturbed box remains close to its original counterpart. The discrepancy grows at higher noise levels. In this regime, a perturbed box may become closer to a different object than to the box from which it was generated, violating the noisy-box assumption used in the derivation. This deviation is informative: it indicates that the two annotations are no longer related by simple localization noise, but differ more substantially in spatial structure.

Fig. 2b shows the same effect from another perspective, plotting the probability of real-box matching as a function of noise level $\gamma$ for fixed values of $\beta .$ Larger $\beta$ values allow boxes with higher spatial cost to remain matched to real boxes rather than to the dummy box, so the noisy-box approximation remains valid over a wider range of perturbations. For small $\beta ,$ such as $\beta = 0 . 2$ , boxes are sent to the dummy box even under modest shifts, and the assumed one-to-one noisy correspondence breaks down earlier.

## 4 Experiments

## 4.1 Datasets

\- ACNE04. We use ACNE04 because two independent annotation sets are available for the same domain, enabling direct comparison of real annotator disagreement. The dataset contains 1 457 facial images. The annotation set from the original paper [27] provides local lesion counts and global acne severity scores together with 18 983 unlabeled bounding boxes. The second annotation set [8] covers 1 204 images after removing low-quality images and contains 32 443 unlabeled bounding boxes; experts were instructed to annotate more acne lesions.

![](images/b1a43549ab79b3f72e9e06341b45170908d6852ea667a5e53ca80fce41764cba.jpg)

![](images/16f61733a4d5108e29989d7148b2cd227412adb6de5bdda8ecacae5895d72240.jpg)

(a)  
![](images/2d1d6110175991ca0d4012dc9192870871a2dfe258544c75aba597ef78f34053.jpg)

![](images/a1939ff077d109cfb4e4a898f9e03e0ecc2b1735f6a71dc66cfec1fa7c29a2c4.jpg)  
(b)  
Fig. 2: Probability of real-real matching under the noisy-box assumption. a) Dependence on $\beta$ for several noise levels γ. b) Dependence on γ for fixed β. Theoretical and empirical results diverge at higher noise levels as matches break and the noisy-box assumption becomes less accurate.

\- AcneSCU [13]. We include AcneSCU to study labeled annotations in the same application domain. It contains 276 facial images acquired with the VISIA complexion analysis system and 31 777 instance-segmentation annotations across 10 lesion categories.

TexBiG [22]. We include TexBiG to evaluate the approach in a substantially different domain with multi-annotator data. This historical-document layout dataset contains more than 52 000 expert-annotated instances across 19 classes, with multiple annotators used for quality control.

– Perturbed datasets. We additionally construct perturbed versions of these datasets using box dropping, box shifting, and label swapping. The perturbation level is controlled by a single parameter $\gamma ,$ whose interpretation depends on the perturbation type. For box drop, $\gamma \in [ 0 , 1 ]$ denotes the fraction of boxes to remove in each image. For box shift, γ scales the displacement of the box center. We shift each box proportionally to its size; for example, under Uniform noise the x-coordinate is perturbed as

![](images/4f1d0ebea02fb8e11108fbcb0b31a2f85d3ea2a9e1625d122553983c2985af8b.jpg)  
(a)

![](images/f60af2df8f9f69ad4ba7a79d8f20cc6dabe586811f5dd50d9cb3c6bf47f87ae8.jpg)  
(b)

![](images/6ef4222712f40d07dc30e56396d7d62df5acdbc9c1aecf692e481c82423a01e9.jpg)  
(c)

![](images/6d87aaab4e3932e1e30c11c55fd85a6286b76ac301f0515da4d89624d260ac9c.jpg)  
(d)  
Fig. 3: OT scores relative to their theoretical bounds. a) ACNE04 scores span much of the feasible range. b) OT scores for several TexBiG classes. $^ { \mathbf { c } , \mathbf { d } ) }$ Score distributions between the lower and upper bounds for ACNE04 and TexBiG, respectively. Because TexBiG objects are larger and easier to localize, their spatial costs are generally smaller and their scores lie closer to the upper bound.

$$
x _ { c 2 } = x _ { c 1 } + \varepsilon _ { x } , \quad \varepsilon _ { x } \sim \mathcal { U } ( - u _ { x } , u _ { x } ) , \quad u _ { x } = \gamma w ,\tag{17}
$$

similarly, under Normal noise:

$$
x _ { c 2 } = x _ { c 1 } + \varepsilon _ { x } , \quad \varepsilon _ { x } \sim \mathcal { N } ( 0 , \sigma _ { x } ) , \quad \sigma _ { x } = \gamma w ,\tag{18}
$$

and analogously for the y-coordinate. In the theoretical calculation for Eq. 16, symmetry allows us to work with the nonnegative absolute displacements $\left| \varepsilon _ { x } \right|$ and $\left| \varepsilon _ { y } \right|$ . For label swap, γ specifies the fraction of boxes in an image whose labels are to be exchanged. Selected boxes are paired at random and their labels are swapped within each pair.

## 4.2 Results and Discussion

Fig. 3 shows example OT scores comparing two annotations for the ACNE04 dataset (Fig. 3a) and for the TexBiG dataset (Fig. 3b), along with the theoretical boundaries in $\operatorname { E q . }$ 10. There are two key differences between these sets of annotations. First, acne objects generally have a smaller size relative to the image resolution than objects on paper (such as paragraphs or headings), meaning that changes in localization affect the cost metric much more strongly in the former case. Second, when an object is easy to annotate (e.g. a clearly visible paragraph in a document), it can naturally be surrounded by a rectangular box. From the figures one can clearly see that, while for ACNE04 we obtain a wide range of OT values, for TexBiG almost all objects lie close to the upper bound: localization error rarely becomes crucial. Based on these observations, we believe that the OT score can be applied to any dataset, but it is most useful and informative when there are many small or medium-sized boxes in specialized domains (e.g. acne-lesion detection), where objects can be easily missed or misclassified.

Fig. 4 shows the dependence of the metrics on the noise level $\gamma .$ We use the method for computing annotator discrepancy/similarity from [12]. If the annotation set from annotator $j$ on image i is $\mathcal { V } _ { j } ^ { i }$ and $A _ { i }$ is the number of annotators for image $i ,$ then for a pairwise similarity measure $M ( \mathcal { V } _ { j } ^ { i } , \mathcal { V } _ { j ^ { \prime } } ^ { i } )$ , such as F1, Krippendorff's $\alpha ,$ or mIoU, annotator similarity is defined as

$$
S ( \{ \mathcal { Y } _ { j } ^ { i } \} ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \frac { 1 } { A _ { i } ( A _ { i } - 1 ) } \sum _ { \substack { j , j ^ { \prime } = 1 , j \neq j ^ { \prime } } } ^ { A _ { i } } M ( \mathcal { Y } _ { j } ^ { i } , \mathcal { Y } _ { j ^ { \prime } } ^ { i } )\tag{19}
$$

In this experiment, we test how each metric behaves as the perturbation level increases by comparing the original annotation with a perturbed one. Thus, $A _ { i } = 2$ in $\operatorname { E q } .$ 19, which reduces to the mean pairwise similarity across images. For OT, we use the dataset-level chance-corrected coefficient from Eq. 8. Because these metrics are not all defined on the same chance-corrected scale, we compare their response to increasing perturbation rather than their absolute values.

Fig. 4c shows that IoU-threshold-based metrics exhibit a rapid drop as noise increases, particularly once box overlaps fall below the IoU threshold required for a true-positive match. In contrast, the $\alpha _ { \mathrm { { O T } } }$ curves with $\beta = 0 . 5$ and 0.8 decrease more gradually, reflecting their sensitivity to spatial displacement and the absence of hard matching thresholds. The $\beta = 0 . 2$ setting also exhibits a rapid decrease because the smaller dummy-matching penalty makes real-box matches less preferable even under modest perturbations. For comparison, we also report mean IoU (mIoU), which decreases more smoothly than thresholdbased metrics but becomes insensitive once boxes no longer overlap and is not commonly used as an object-detection agreement metric.

Figures 4b and 4a illustrate the behavior of the metrics under box dropping (ACNE04) and label swapping (AcneSCU), respectively. In both cases, the curves are approximately linear. For the OT-based metric, this is because each unmatched box contributes a penalty of $\beta$ in the box drop setting, while in the label swap setting a discrete label cost contributes to the pairwise cost in Eq. 3;

![](images/f66a6017fc4fc9c03621cc259392123cdf626688c2308492acc7773f7de6b2a3.jpg)  
(a)

![](images/f40588a9b314342f48dfd20c115caf5a64f963aa922657a66ec45f377d8b0f86.jpg)  
(b)

![](images/5b6e3c3b380c536411da9bcc7a729facfbc99a8bf19f4b2e6f94bdd1d8a13fb1.jpg)  
(c)  
Fig. 4: Annotator agreement between original and perturbed annotations. a) Agreement versus fraction of swapped labels on AcneSCU dataset. b) Agreement versus ratio of dropped boxes on ACNE04 dataset. c) Agreement versus shift noise level on ACNE04 dataset.

thus, the total cost scales approximately linearly with the number of such errors. An interesting extension would be to allow graded label distances or similarities rather than values restricted to {0, 1}, and to account for label imbalance, as suggested, for example, in [1]. We report several values of $\beta ;$ in practice, $\beta$ can be selected empirically or calibrated using approaches such as KαLOS [24] (see supplementary material). We repeated all experiments across several random seeds; variability was small and is omitted for clarity.

We quantify how consistently each metric responds to increasing noise. Perimage Kendall's $\tau _ { b }$ between increasing noise and decreasing agreement is 0.998 for OT and 0.993 for mIoU, compared with 0.790 for F1 and 0.790 for Krippendorff's α.

We additionally examine ordering between adjacent noise levels at the dataset level. Let $Q _ { M } ( \gamma )$ denote the dataset-level value for metric M at noise level $\gamma$

![](images/4805da5a0aea0ece2f6a3e6c33480fb8f8fab6f461bb287b123bfeac098e8705.jpg)

![](images/416c77246eadc8b5ddfd50c20ab2cce03899142adafa13d2a2adc9840853c333.jpg)  
(a)

![](images/a133d2bcf5c22e15881eed27943fb9263ebb1aa742af9cc0ba9d6b2de94fc280.jpg)

![](images/1bef22cf4e662975d09aad5bbe44f098b04f74c5fe8d77c5ba175adf56a9a4b6.jpg)  
(b)

![](images/276e263105438c983d15564b8f8e2666d71e319014424340e282b22b2ad2424e.jpg)  
(c)  
Fig. 5: Behavior of agreement metrics under increasing box-shift noise and beyond the mIoU = 0 regime. a) Ordering under increasing shift noise. b) Percentage of images with m $I o U = 0$ versus noise level. c) Distribution of OT scores for cases with $m I o U = 0$

For pairwise similarity measures, $Q _ { M }$ is the mean similarity in Eq. 19; for OT, it is $\alpha _ { \mathrm { { O T } } }$ from Eq. 8. For adjacent noise levels $\gamma _ { i } < \gamma _ { i + 1 }$ , define $\varDelta _ { M } = Q _ { M } ( \gamma _ { i } ) -$ $Q _ { M } ( \gamma _ { i + 1 } )$ . We classify each comparison as expected ordering if $\begin{array} { r } { \varDelta _ { M } \ > \ \varepsilon , } \end{array}$ a tie if $| \varDelta _ { M } | \leq \varepsilon ,$ and reverse ordering if $\varDelta _ { M } < - \varepsilon .$ with $\varepsilon = 1 0 ^ { - 3 }$ . Fig. 5a shows that OT and mIoU preserve the expected ordering in most cases, whereas the threshold-based metrics exhibit many ties: ceiling saturation dominates at low noise and floor saturation at high noise. Reversals at the largest shifts can occur when perturbed boxes move closer to neighboring reference boxes and are rematched.

Both mIoU and OT provide graded responses to localization changes while boxes overlap. Once overlap vanishes, however, mIoU is identically zero regardless of further displacement, whereas OT can retain graded spatial information when $\beta / \lambda > 1 / 2$ . On ACNE04, Fig. 5b shows that the fraction of images with m1 $\mathbf { \partial } _ { \mathbf { \theta } } \mathbf { { o } } U = 0$ rises to about 19% at 200% shift. For these images, OT scores span approximately 0.20–0.47 for $\beta \ : = \ : 0 . 8$ , as shown in Fig. 5c. Thus OT retains spatial resolution even after mIoU reaches zero.

These synthetic perturbations are controlled stress tests rather than models of human annotation noise. ACNE04 and TexBiG provide real annotation disagreements, but we do not have independent expert ratings of disagreement severity; such validation remains future work. Moreover, λ and $\beta$ are taskdependent. Supplementary Fig. 3 illustrates their effect on spatial matching, while Supplementary Fig. 4 shows data-driven calibration of $\beta$ on TexBiG.

## 5 Conclusion

In this work, we adapted an Optimal Transport (OT)-based metric for quantifying annotator agreement in object detection. By formulating agreement as a transport problem between sets of bounding boxes, our approach naturally integrates spatial and label information and allows for flexible handling of missing or extra annotations.

We analyzed several theoretical properties of the OT-based metric, deriving bounds for cases in which the transport plan is known and characterizing the possible range of OT scores under specific parameter settings. We also studied the behavior of the metric as a function of $\beta$ under a noisy-box assumption. This analysis helps to understand how many real matches remain as $\beta$ or the noise level increases, how to select $\beta$ for a given task, and when deviations from the theoretical assumptions signal that annotations have become substantially different.

Through experiments on both real-world and synthetically perturbed datasets, we demonstrated that the OT-based metric is particularly sensitive to differences in annotations, especially with respect to the spatial positions of bounding boxes. In scenarios where traditional IoU threshold-based metrics (such as F1 or Krippendorff's α) fail to distinguish between qualitatively different annotation patterns, OT still reflects meaningful variation in annotator behavior.

Our findings lead to two main conclusions. First, model performance evaluation and annotator agreement assessment are closely related but not identical: what is treated as an “error" in model evaluation does not necessarily correspond to disagreement between annotators, and OT can help to disentangle these notions. Second, because the main sensitivity of the OT cost comes from spatial localization, we believe it is especially well suited to domains with many small or medium-sized boxes and non-trivial domain knowledge (e.g. medical imaging), where objects are easily missed or misclassified.

Overall, these results indicate that the OT metric provides a richer and more informative measure of annotator agreement and discrepancy, and can support more reliable model validation in the presence of noisy or conflicting labels

## References

1. Boriah, S., Chandola, V., Kumar, V.: Similarity measures for categorical data: A comparative evaluation. In: Proceedings of the 2008 SIAM international conference on data mining. pp. 243–254. SIAM (2008)

2. Braylan, A., Alonso, O., Lease, M.: Measuring annotator agreement generally across complex structured, multi-object, and free-text annotation tasks. In: Proceedings of the ACM Web Conference 2022. pp. 1720–1730 (2022)

3. De Plaen, H., De Plaen, P.F., Suykens, J.A., Proesmans, M., Tuytelaars, T., Van Gool, L.: Unbalanced optimal transport: A unified framework for object detection. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 3198–3207 (2023)

4. Feng, D., Wang, Z., Zhou, Y., Rosenbaum, L., Timm, F., Dietmayer, K., Tomizuka, M., Zhan, W.: Labels are not perfect: Inferring spatial uncertainty in object detection. IEEE Transactions on Intelligent Transportation Systems 23(8), 9981–9994 (2021)

5. Flamary, R., Courty, N., Gramfort, A., Alaya, M.Z., Boisbunon, A., Chambon, S., Chapel, L., Corenflos, A., Fatras, K., Fournier, N., et al.: Pot: Python optimal transport. Journal of Machine Learning Research 22(78), 1–8 (2021)

6. Ge, Z., Liu, S., Li, Z., Yoshie, O., Sun, J.: Ota: Optimal transport assignment for object detection. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 303–312 (2021)

7. Geng, X.: Label distribution learning. IEEE Transactions on Knowledge and Data Engineering 28(7), 1734–1748 (2016)

8. Huynh, T.H., et al.: Acneai: A new acne severity assessment method using digital images and deep learning. In: Medical Image Computing and Computer Assisted Intervention − MICCAI 2024, pp. 68–78. Springer (2024)

9. Krippendorff, K.: Computing Krippendorff's alpha-reliability. Departmental Papers (ASC) p. 43 (2011)

10. Lampert, T.A., Stumpf, A., Gançarski, P.: An empirical study into annotator agreement, ground truth estimation, and algorithm evaluation. IEEE Transactions on Image Processing 25(6), 2557–2572 (2016)

11. Llerena, J.M., Zeni, L.F., Kristen, L.N., Jung, C.: Gaussian bounding boxes and probabilistic intersection-over-union for object detection. arXiv preprint arXiv:2106.06072 (2021)

12. Lovchinsky, I., Daks, A., Malkin, I., Samangouei, P., Saeedi, A., Liu, Y., Sankaranarayanan, S., Gafner, T., Sternlieb, B., Maher, P., et al.: Discrepancy ratio: evaluating model performance when even experts disagree on the truth. In: International conference on learning representations (2019)

13. Min, S., et al.: Learning high-quality proposals for acne detection. arXiv preprint arXiv:2207.03674 (2022)

14. Murrugarra-Llerena, J., Kirsten, L.N., Jung, C.R.: Can we trust bounding box annotations for object detection? In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops. pp. 4813–4822 (2022)

15. Otani, M., Satoh, S.: Optimal correction cost for object detection evaluation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 21107–21115 (2022)

16. Özdemir, B., Aksoy, S., Eckert, S., Pesaresi, M., Ehrlich, D.: Performance measures for object detection evaluation. Pattern Recognition Letters 31(10), 1128–1137 (2010)

17. Peyré, G., Cuturi, M.: Computational optimal transport: With applications to data science. Now Foundations and Trends (2019)

18. Prokhorov, K., Kalinin, A.A.: Improving acne image grading with label distribution smoothing. In: 2024 IEEE International Symposium on Biomedical Imaging (ISBI). pp. 1–5. IEEE (2024)

19. Rezatofighi, H., Tsoi, N., Gwak, J., Sadeghian, A., Reid, I., Savarese, S.: Generalized intersection over union: A metric and a loss for bounding box regression. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 658–666 (2019)

20. Sarlin, P.E., DeTone, D., Malisiewicz, T., Rabinovich, A.: Superglue: Learning feature matching with graph neural networks. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 4938–4947 (2020)

21. Tschirschwitz, D., Benz, C., Florek, M., Norderhus, H., Stein, B., Rodehorst, V.: Drawing the same bounding box twice? coping noisy annotations in object detection with repeated labels. In: DAGM German Conference on Pattern Recognition. pp. 605–623. Springer (2023)

22. Tschirschwitz, D., Klemstein, F., Stein, B., Rodehorst, V.: A dataset for analysing complex document layouts in the digital humanities and its evaluation with Krippendorff's alpha. In: DAGM German Conference on Pattern Recognition. pp. 354– 374. Springer (2022)

23. Tschirschwitz, D., Rodehorst, V.: Label convergence: Defining an upper performance bound in object recognition through contradictory annotations. In: IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) (2025)

24. Tschirschwitz, D., Rodehorst, V.: KαLOS finds consensus: A meta-algorithm for evaluating inter-annotator agreement in complex vision tasks. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2026)

25. Wang, C., et al.: Learn2agree: Fitting with multiple annotators without objective ground truth. arXiv preprint arXiv:2109.03596 (2021)

26. Warfield, S.K., Zou, K.H., Wells, W.M.: Simultaneous truth and performance level estimation (STAPLE): an algorithm for the validation of image segmentation. IEEE transactions on medical imaging 23(7), 903–921 (2004)

27. Wu, X., Wen, N., Liang, J., Yu, Y.J., Shen, L.L.: Joint acne image grading and counting via label distribution learning. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 10652–10660 (2019)

# Supplementary Material: Quantifying Annotation Agreement in Object Detection with Optimal Transport

Kirill Prokhorov¹ and Alexandr A. Kalinin2

1 Openface.io, Vilnius, Lithuania prokhorov@openface.me 2 Broad Institute of MIT and Harvard, Cambridge, MA, USA

## 1 On the evaluation of $P ( c _ { i j } < \beta )$

To calculate the probability that the cost $c _ { i j }$ is less than $\beta ,$ we evaluate

$$
P ( c _ { i j } < \beta ) = F _ { c } ( \beta ) = \int _ { 0 } ^ { \beta } f _ { c } ( t ) d t = \iint _ { c ( \xi , \eta ) < \beta } f _ { | \varepsilon _ { x } | , | \varepsilon _ { y } | } ( \xi , \eta ) d \xi d \eta ,\tag{1}
$$

where $f _ { c }$ is the PDF of the cost and $f _ { | \varepsilon _ { x } | , | \varepsilon _ { y } | }$ is the joint PDF of the absolute noise components.

To characterize the integration domain $c ( \xi , \eta ) < \beta$ , we use the expression for GIoU in the cases where the two boxes do and do not intersect. This leads to a decomposition of the integral in Eq. 1 into three regions $( { \mathrm { F i g . ~ } } 1 ) \colon u < 1 , \vartheta < 1$ where $I > 0$ , and two regions with $I = 0$ , namely $u < 1 , \vartheta \geq 1$ and $u \geq 1 , \vartheta > 0$ For $I = 0$ we have:

$$
G I o U _ { I = 0 } = - \frac { ( 1 + u ) ( 1 + \vartheta ) - 2 } { ( 1 + u ) ( 1 + \vartheta ) } = 1 - 2 \beta ,\tag{2}
$$

which gives

$$
\vartheta _ { I = 0 } ( u ) = \frac { 1 } { ( 1 - \beta ) ( 1 + u ) } - 1 .\tag{3}
$$

The same can be done for the case $I \neq 0 { : }$

$$
G I o U _ { I \ne 0 } = \frac { ( 1 - u ) ( 1 - \vartheta ) } { 2 - ( 1 - u ) ( 1 - \vartheta ) } - \frac { ( 1 + u ) ( 1 + \vartheta ) - 2 + ( 1 - u ) ( 1 - \vartheta ) } { ( 1 + u ) ( 1 + \vartheta ) } = 1 - 2 \beta .\tag{4}
$$

For $I \neq 0$ , Eq. 4 can be rewritten as a quadratic equation in $\vartheta ,$

$$
A ( u , \beta ) \vartheta ^ { 2 } + B ( u , \beta ) \vartheta + C ( u , \beta ) = 0 ,\tag{5}
$$

where

$$
A ( u , \beta ) = ( u - 1 ) \big [ \beta ( u + 1 ) - 2 u - 1 \big ] ,\tag{6}
$$

$$
B ( u , \beta ) = ( u + 1 ) ( u + 1 - 2 \beta ) ,\tag{7}
$$

$$
C ( u , \beta ) = - ( u + 1 ) \big [ \beta ( u + 1 ) - u \big ] .\tag{8}
$$

![](images/2e2431976dad56fe1e833a3b9d5ad516eea0968811ff03e3f15185d3f45cddc8.jpg)  
Fig. 1: Integration regions used in the calculation in Eq. 1.

We denote the relevant (larger) real root by $\vartheta _ { I \neq 0 } ( u )$ . When defining the integration limit below, we clip this root to the feasible interval $[ 0 , 1 ]$

We can now define the integration limits in each region. For $u < 1 , \vartheta < 1$ we set $a _ { 1 } ( u ) = \operatorname* { m a x } ( 0 , \operatorname* { m i n } ( 1 , \vartheta _ { I \neq 0 } ( u ) ) )$ ; for $u \ < \ 1 , \vartheta \ \geq \ 1$ , we set $a _ { 2 } ( u ) \ =$ ma $\mathfrak { c } ( 1 , \vartheta _ { I = 0 } ( u ) )$ ; and for $u \geq 1 , \vartheta > 0$ we set $a _ { 3 } ( u ) = \operatorname* { m a x } ( 0 , \vartheta _ { I = 0 } ( u ) )$

Substituting these region-specific limits into $\operatorname { E q . }$ 1 decomposes the integration domain into the three regions described above:

$$
\begin{array} { l } { { F _ { c } ( \beta ) = \displaystyle \int _ { 0 } ^ { 1 } \int _ { 0 } ^ { a _ { 1 } ( \xi ) } f _ { u , \vartheta } ( \xi , \eta ) d \eta d \xi + \displaystyle \int _ { 0 } ^ { 1 } \int _ { 1 } ^ { a _ { 2 } ( \xi ) } f _ { u , \vartheta } ( \xi , \eta ) d \eta d \xi + } } \\ { { + \displaystyle \int _ { 1 } ^ { \infty } \int _ { 0 } ^ { a _ { 3 } ( \xi ) } f _ { u , \vartheta } ( \xi , \eta ) d \eta d \xi . } } \end{array}\tag{9}
$$

If the noise components are independent, the joint distribution factorizes as

$$
f _ { u , \vartheta } ( \xi , \eta ) = f _ { u } ( \xi ) f _ { \vartheta } ( \eta ) ,\tag{10}
$$

so the inner integrals can be expressed using the CDF $F _ { \vartheta }$

$$
\begin{array} { c } { \displaystyle F _ { c } ( \beta ) = \int _ { 0 } ^ { 1 } f _ { u } ( \xi ) F _ { \vartheta } \big ( a _ { 1 } ( \xi ) \big ) d \xi + \int _ { 0 } ^ { 1 } f _ { u } ( \xi ) \big [ F _ { \vartheta } \big ( a _ { 2 } ( \xi ) \big ) - F _ { \vartheta } ( 1 ) \big ] d \xi + } \\ { \displaystyle + \int _ { 1 } ^ { \infty } f _ { u } ( \xi ) F _ { \vartheta } \big ( a _ { 3 } ( \xi ) \big ) d \xi . } \end{array}\tag{11}
$$

## 2 Examples of spatial properties of the OT score

Fig. 2 shows two annotation pairs with the same threshold-based scores but different spatial structure. In Fig. 2a, the annotations differ by a moderate spatial shift: the objects are localized approximately correctly, although the boxes are not perfectly aligned. In Fig. 2b, one annotation has been deliberately shifted so that only one pair of boxes overlaps, while the remaining boxes are far from their corresponding objects.

![](images/4a8affcb441fc463c5ab43b722426623e61e3a7b919415084c713d8c204ed472.jpg)  
(a)

![](images/db658a74ec46b41101be3a81050e85ce34545eaf17f9c2ecef80ba45f6cb369c.jpg)  
(b)  
Fig. 2: a) Two annotations with a moderate spatial shift. b) One annotation is deliberately shifted so that only one pair of boxes overlaps. F1 and Krippendorff's α assign the same scores to both examples, whereas OT distinguishes their spatial disagreement mIoU remains sensitive only while boxes overlap.

F1 and Krippendorff's α assign the same values to both cases, because after thresholded matching there is one true positive and the remaining boxes are treated as unmatched errors. This is appropriate for counting detection errors, but it loses information that matters for annotation agreement. The first pair reflects approximate agreement with localization noise, whereas the second reflects a stronger spatial disagreement. The OT score distinguishes these cases because unmatched and poorly matched boxes still contribute through their spatial costs. Mean IoU also reflects part of this difference, but only while corresponding boxes overlap; once overlap is zero, it cannot distinguish larger spatial displacements. Thus, OT provides a graded comparison of bounding-box annotations beyond the binary matched/unmatched decisions used by thresholded metrics.

In the label-free case, real matching is preferred when $c _ { i j } = \lambda ( 1 - \mathrm { G I o U } ) / 2 <$ $\beta ,$ SO $\rho = \beta / \lambda$ controls the real-to-dummy decision. Fig. 3 compares the OT possible match region of a moving probe box with the region where mIoU $> 0$ For $\rho \ \leq \ 1 / 2$ , the OT region lies inside the overlap region or even reference boxes; for $1 / 2 < \rho < 1$ , it extends beyond overlap and distinguishes cases with mIoU = 0. If $\rho > 1$ , all real edges are eligible, producing min(M, N) real matches.

![](images/01aa94f554656481bb8a91c744730f8c0d398d38f1bcb6cfcc5dcb27293b51f4.jpg)  
Fig. 3: Spatial regions in which a translated probe box is eligible for real-real matching under different $\rho = \beta / \lambda$ , compared with the region where mIoU $> 0$

![](images/99834ce0309f696ad5fe68141b4cde0b5b5daba950ee86ccac164131ad137f64.jpg)  
Fig. 4: TexBiG observed versus expected distances.

## 3 Data-driven estimation of $\beta$ using KαLOS

We apply the KαLOS data-driven $D _ { o } / D _ { e }$ calibration principle [1] to real TexBiG annotations. Using all six annotators and 2,250 images with at least two nonempty annotation sets, same-image and cross-image box-level spatial distances $d = ( 1 - \mathrm { G I o U } ) / 2$ yield $\tau ^ { * } = 0 . 2 2 2$ , providing a data-derived candidate $\beta ^ { * } =$ 0.222 for $\lambda = 1 \ ( { \mathrm { F i g . ~ 4 } } )$ . The distributions are well separated (mean $D _ { o } = 0 . 0 7 2$ VS. $D _ { e } = 0 . 4 4 0 )$ , supporting calibration from real multi-annotator data. Synthetic perturbations serve as controlled stress tests rather than models of human annotation noise; expert severity judgments would provide complementary validation.

## 4 Additional information about datasets

Fig. 5 shows the empirical CDF of bounding-box area relative to image size, highlighting differences in object size across the datasets used in this paper. For Acne04 and AcneSCU, objects are relatively small: the vast majority of boxes occupy less than 1% of the image area. In contrast, TexBiG (and in particular the three classes considered in our experiments) contains substantially larger boxes. Although many small boxes are also present, the median relative area is approximately 7%, and there is a substantial number of objects that cover 30% or more of the image. These differences in object size across datasets help illustrate how the proposed approach behaves in distinct, realistic scenarios.

Fig. 6 presents several example images from the datasets. ACNE04 is an unlabeled dataset for which we used two different annotation sets from two papers. AcneSCU includes 10 lesion categories: open comedone, closed comedone, papule, pustule, nodule, atrophic scar, hypertrophic scar, melasma, nevus, and other. TexBiG contains 19 document layout classes: paragraph, decoration, page number, column title, image, header, noise, caption, heading, editorial note, frame, footer, sub-heading, equation, author, footnote, table, logo, and advertisement. The three classes used in the main experiments (paragraph, image, and caption) were chosen because of their relatively large object sizes, serving as a counterpart to the small objects in the acne datasets. For each TexBiG class, we selected two annotators with the largest number of annotated objects to better visualize annotator differences.

![](images/acea76e6eb421fdf68424a0cd2a81d3f05952a2f25620e9bc4d6d2e8021f3e25.jpg)

![](images/dc1dfd92a562ad5228e6fddf5fc95a42ad2e4f2f1fd8d9c91fb74e97d01c9f3f.jpg)  
Fig. 5: Empirical CDF of bounding-box area relative to image area across the datasets and classes used in the experiments.

![](images/7c023f9428a455f457b1a443ceda06ba15a073a682c6370941fd26d0b672d659.jpg)  
(a)

![](images/0bf0ee1319597f7e7fc91accfe474d1b676575efdad7718853cf2c86a3476a9f.jpg)  
(b)

![](images/0c95183e8ce9af066df43be77ab24993805af9f3668181c871eb3ae924a308a4.jpg)  
(c)

![](images/b6672482d944bb9d80233b0cbb55e11d72cf67c13e487e7dfc2610f87f33d022.jpg)  
(d)

![](images/0860f86bdbf1c12d02216d6a253853515c54b3d3d4791c7f66723f5bf0901abf.jpg)  
(e)

![](images/1528092b7fb6e62efdb32b42f252a6d4a2c228bffcdd29ecaa12dac2d2a06873.jpg)  
(f)  
Fig. 6: Example images from the datasets. ${ \bf a } , { \bf b } )$ ACNE04 with two annotation sets. $^ { \mathbf { c } , \mathbf { d } ) }$ AcneSCU; colors denote lesion classes. $\mathbf { e , f } )$ TexBiG.

## References

1. Tschirschwitz, D., Rodehorst, V.: KαLOS finds consensus: A meta-algorithm for evaluating inter-annotator agreement in complex vision tasks. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2026)