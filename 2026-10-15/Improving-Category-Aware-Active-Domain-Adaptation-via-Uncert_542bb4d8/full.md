# Improving Category-Aware Active Domain Adaptation via Uncertainty-Guided Region Coverage

My Tra Nguyen Huynh<sup>1</sup> and Viet Cuong Ta<sup>1\*</sup>

<sup>1</sup>VNU University of Engineering and Technology, Hanoi, 100000, Vietnam.

\*Corresponding author(s). E-mail(s): cuongtv@vnu.edu.vn; Contributing authors: mynht@vnu.edu.vn;

## Abstract

Active domain adaptation researches focus on alleviating the annotation cost, which comprises source data and target data labelling, as well as enhancing the performance of unsupervised domain adaptation. However, optimizing the adaptation model based on the average accuracy can lead to the drop in performance of some dificult categories. To overcome this issue, a point-wise selection strategy built upon the influence function can be used to improve domain adap tation performance. In our work, we first address the sensitive and redundant aspects of point-wise selection, which can lead to overfitting of the acquisition function. We thus introduce a region-based approach that ensures the improvement of the adaptation model by maximizing a positive manifold coverage. An uncertainty score is further introduced to the selection mechanism to learn a better decision boundary for the target domain. In comprehensive experiments, our proposed method surpasses the first category-aware active domain adaptation baseline as well as existing active domain adaptation methods across multiple domain adaptation benchmarks. Our implemenation is available at: https://github.com/nhtm2702/WHO IS NEXT.

Keywords: Active Domain Adaptation, Category-Aware, Uncertainty-Aware, Region-based

## 1 Introduction

Domain Adaptation (DA) aims to leverage knowledge learned in a source domain to deploy in a related target domain [1]. In this setting, the oracle is required to annotate the whole large amount of data, which comprises a source dataset and a target dataset [2]. To make DA works more practically, the line of active learning is combined with DA to study the cost-efectiveness in transfer between related domains [3, 4]. Within the frame of DA, active learning-based methods require labels for the whole source dataset and only a subset of the target dataset, which satisfies a particular design strategy to improve the adaptation model performance. Most existing active DA works, such as CLUE [3], DUC [4] assess the efectiveness of the adaptation model through the average performance across the entire target category, which can lead to a bias phenomenon. Therefore, the learner tends to query the easier categories, while the most challenging categories seldom receive the advancements or even sufer the performance decreases after active learning rounds.

To alleviate this issue, CAADA [5] introduces a new active domain adaptation setting that trains a model for a specific category to ensure the improvement of indi vidual categories. This work adapts the influence function [6] to the selection strategy to select the samples that are considered potential to improve each category model’s performance. However, in the early rounds when the model performance is still unreliable, directly using the parameter of the logistic regression model in the point-wise selection strategy may necessitate careful design. Moreover, when using point-based selection, high influence samples may concentrate in the same local region of the feature space. This leads to redundant selections, where queried samples are mutually close and provide overlapping information, while the learned model is expected to approximate the underlying data distribution. Thus, this approach does not guarantee that it will enhance model generalization, and it may lead to overfitting of the acquisition function to the empirical training objective [7, 8].

In our work, we first discuss the redundant and parameter-dependent issues of point-based scores in the category-aware domain adaptation setting. Based on the analysis, we leverage the point-based selection to a region-based selection with a positive manifold score. The positive manifold score is designed to measure the coverability of candidate samples to the underlying target distribution through Monte-Carlo approximation. Moreover, to learn a better decision boundary, we employ the uncertainty guidance, which encourages the selection strategy to query the sample to discover more about the decision boundary regions while reducing focus on regions that are quite certain. We evaluate our proposed method with the active domain adaptation baselines across three benchmarks. The results highlight that our method enhances the average accuracy significantly with better underlying data coverages.

## 2 Background and Preliminaries

## 2.1 Related Works

Domain Adaptation methods are designed to within a transfer learning setting, that operates knowledge generalization on a source domain and then transfers this to a related target domain. According to the assumption of target data accessibility, DA settings fall into unsupervised [9, 10], semi-supervised [11, 12], weakly-supervised [13, 14], and active [3, 5, 15]. In this work, we focus on the active branch that leverages the active learning strategy to reduce annotation cost when adapting knowledge to the target domain.

Active Learning is a machine learning branch that aims to reduce the annotation cost through constructing a selection strategy to choose a subset of unlabeled data to annotate. ISAL [16] selects the most positive influence samples on model performance to annotate. [17] defines a cover region to select samples in this and assumes this region can maximize the coverage of data distribution. In real-world applications, AL can be combined with DA to reduce annotation costs while enhancing model performance on the target domain [3–5, 15]. CLUE [3] integrates both diversity and uncertainty into an uncertainty-weighted clustering framework. DUC [4] further explores the uncertainty miscalibration in DA problems, and employs a Dirichlet-based evidential model to select uncertain and informative samples.

## 2.2 Background

Active Domain Adaptation (ADA). We denote a domain adaptation model $\mathcal { G }$ trained on a labeled source domain S and need to be adapted to perform well on a related unlabled target domain T. All of the samples from the source dataset are annotated while the target dataset is divided into two parts, $\mathcal { T } _ { L }$ is the labeled target pool and $\mathcal { T } _ { U }$ is the unlabeled target pool, where $\mathcal { T } _ { L } \ll \mathcal { T } _ { U }$ . Both domains have the same set of categories $C .$ We follow the pool-based setting above to define this problem. In the initial iteration, we randomly select B samples from $\tau$ , then set $\mathcal { T } _ { \cal L } ^ { 1 } = \mathcal { S } \cup \cal B$ and $\mathcal { T } _ { U } ^ { 1 } = \mathcal { T } \backslash B$ . Next, at each iteration k with $k \in [ 2 , . . , K ]$ , the DA model $\mathcal { G }$ is retrained on a labeled dataset $\mathcal { T } _ { L } ^ { k - 1 }$ . Then, a design selection strategy is used to assess the unlabeled target domain $\mathcal { T } _ { U } ^ { k - 1 }$ to actively query and select a set of informative samples $\boldsymbol { B }$ to annotate. The labeled target pool is updated as $\mathcal T _ { L } ^ { k } = \mathcal T _ { L } ^ { k - 1 } \cup B$ and the unlabeled target pool is $\mathcal { T } _ { \boldsymbol { U } } ^ { k } = \mathcal { T } _ { \boldsymbol { U } } ^ { k - 1 } \setminus \boldsymbol { B }$

Category-aware Active Domain Adaptation. Let $\mathcal { G }$ be a pretrained domain adaptation feature extractor, which is used to extract embeddings of source and target datasets. With a specific category c, we binarized the training pool so that samples belonging to class c will be aligned to positive and the rest will be negative. Let $f _ { c } ,$ which is parameterized by $\theta ,$ be a binary logistic regression model trained on the domain-invariant representations extracted by ${ \mathcal { G } } .$ . At each iteration $k ,$ , an influence score predictor $f _ { \mathcal { T } }$ is trained to follow the Equation:

$$
\mathcal { L } _ { f _ { \mathcal { T } } } = \frac { 1 } { N _ { \mathcal { T } _ { L } } } \sum _ { x ^ { \prime } \in \mathcal { T } _ { L } } \ell _ { m s e } ( f _ { \mathcal { T } } ( x ^ { \prime } ) , \mathcal { T } ( x ^ { \prime } ) )\tag{1}
$$

where $ { \mathcal { T } } ( x ^ { \prime } )$ is the influence estimate of labeled data $x ^ { \prime }$ . This influence score is calculated based on the Hessian matrix for optimal parameter $\theta \colon$

$$
\boldsymbol { \mathcal { T } } ( \boldsymbol { x } ^ { \prime } ) = - \nabla _ { \boldsymbol { \theta } } \mathcal { L } _ { v } \mathbf { H } _ { \boldsymbol { \theta } } ^ { - 1 } \nabla _ { \boldsymbol { \theta } } \boldsymbol { \ell } ( f _ { c } ( \boldsymbol { x } ^ { \prime } ) , \boldsymbol { y } ) ,\tag{2}
$$

Then, for each unlabeled target sample $x \in \tau _ { U }$ , the influence score is estimated as $\mathcal { I } ( x ) = f _ { \mathcal { T } } ( x )$ . These influence scores are ranked in descending order and B highest samples are selected. These $\boldsymbol { B }$ selected samples are considered most beneficial for recognizing the category c. After K active rounds, the accuracy of classifier $f _ { c }$ will be reported as a performance of the domain adaptation task of category $c .$

Kernel Herding. To model the data coverage of the target distribution, we rely on the Kernel Herding formulation [18]. Let $h _ { c }$ be a non-negative function, which is defined as follows: for any $x , { \mathrm { ~ i f ~ } } | | x _ { 1 } - x | | \leq | | x _ { 2 } - x | |$ then $h _ { c } ( x , x _ { 1 } ) \geq h _ { c } ( x , x _ { 2 } )$ . In practice, the measurement $h _ { c }$ implicitly relies on the existence of a good embedding space to compute distance, which is correlated with semantic similarity, and where similar points are likely to cluster together in high density regions.

## 3 Method

## 3.1 Motivation

Category-aware active domain adaptation is a setting focused on improving the performance for the individual category without adversely afecting others. The existing method [5], which designs a point-based selection strategy, requires the Hessian value of the logistic regression $f _ { c }$ to compute the influence score for training the influence predictor $f _ { \mathbb { Z } } .$ . Thus, the influence-based acquisition relies on the fitted model $f _ { c } ,$ typically through first-order gradients and inverse-Hessian terms. The resulting scores reflect the influence of a sample under the current optimal parameter $\theta ,$ rather than an intrinsic data importance. Consequently, influence scores may become sensitive to minor changes in the model parameters.

![](images/9ee5c0253bb01f2a48e4b0f424321a7fb7093984d8fc15bef5e1def4404feda5.jpg)

![](images/03f8db217fbf2b29cc3c1cf6cbdecfd48648ee48d12928c3d6cc38a59dcbb2e2.jpg)  
Fig. 1 KDE plot for behavior of selection strategy of CAADA method in the 2nd active round with two categories, ”Batteries” and ”Trash Can”, in the Ar→Cl task of the OficeHome dataset.

On the other hand, when using point-based selection without an explicit diversity or coverage constraint, high influence samples may concentrate in the same local region of the feature space. As a result, estimating the model generalization is negatively impacted. This phenomenon can be seen in the selection strategy of the CAADA method [5]. We plot the concentration of the selected samples of CAADA in the domain adaptation active learning setting in Figure 1. The samples selected by the point-wise approach of CAADA can be significantly more compact around the category centroid compared to the full distribution. In both selected categories, the selected samples exhibit higher density peaks and narrower distributions, suggesting that in some cases, this approach mainly focuses on high density regions. This efect is particularly clear in the “Trash Can” category, where the selected distribution has a much sharper peak and smaller standard deviation, indicating stronger concentration near the category centroid.

Motivated by these issues, we aim to construct an active strategy, which rarely relies on sensitive parameters of the regression model in the query phase. Furthermore, our objective is to design a region-based selection strategy, which can eliminate the redundancy of point-based selection as well as boost the adaptation performance for each category.

![](images/9b5a3e8c46d1d56647ccf7aa8f1d3ce36f24069da4318d2972e09624490fa10e.jpg)  
Fig. 2 Illustration of diferent coverage types between CAADA, MaxHerding and our Positive Manifold method.

In Fig. 2, we visualize diferences in assessing coverage on an illustration dataset to demonstrate the behaviour of the selection strategy. CAADA (left) focuses on finding high influence samples for a specific category, which can lead to a local concentration. MaxHerding (middle) introduces a proximity measure for the whole data distribution while ignoring the category-based setting, thus, it may still select samples from negative or task-irrelevant regions. In contrast, our positive manifold selection (right) performs coverage only over the positive constraint, thus reducing redundant selections and avoiding selection in irrelevant regions.

## 3.2 Uncertainty-Guided Selection with Maximizing Region Coverage

To avoid the limitations of the point-wise selection approach, we propose region-based active learning into a domain adaptation framework. We rely on the Kernel Herding [18, 19] idea to introduce the region-based selection strategy. However, instead of only assessing the coverage of the whole data distribution, our proposed method is extended along two aspects. First, we estimate category-relevant coverage regions, which correspond to the positive manifold of each category, allowing the selection strategy to focus on category-specific target regions. Second, to ensure that the covered regions are informative for adaptation, we introduce an informative positive manifold coverage, which prioritizes regions where the model lacks confidence.

Our proposed framework is illustrated in Figure 3. We use the feature representations of both the source and target domains for training the objective learner $f _ { c }$ as well as match the target distribution to the source distribution with respect to category c. The latter objective can be achieved by optimizing the coverage gain, which is detailed by the kernel-based score $h _ { c }$ . Specifically, we compute a positive manifold coverage worth respect to the category c, by relying on $h _ { c }$ and the labeled pool $\mathcal { T } _ { L }$ . Following the region-based approaches in active learning [18, 19], our goal is to approximate the proposed coverage objective using Monte-Carlo estimation over the unlabeled target pool $\mathcal { T } _ { U }$ . However, the data coverage assumption does not directly transfer to a better decision boundary. Therefore, we further introduce an uncertainty guide process to estimate how positive manifold coverage is accumulated over the unlabeled target samples. This design enables the positive manifold coverage strategy to prioritize informative regions while still preserving the category-aware positive manifold objective.

![](images/9462947f685348d758476765848df994ed265cdb013839a9b71b7d20b11dc293.jpg)  
Fig. 3 Our proposed category-aware active domain adaptation method focuses on individual categories via uncertainty-Guided Region Coverage.

With the labeled pool $\mathcal { T } _ { L }$ , the data coverage in a category-based setting can be defined on the input feature representation and the kernel function $h _ { c }$ . We denoted $P ( \mathcal { T } _ { L } ; h _ { c } )$ as the coverage of the labeled pool $\mathcal { T } _ { L }$ compared with the whole data distribution, in the ideal scenario, our $P ( \mathcal { T } _ { L } ; h _ { c } )$ ≈ 1. With the active domain adaptation setting, where the annotation cost is expensive, we want to select a subset of samples from the unlabeled pool $\mathcal { T } _ { U }$ that maximize the $P ( \mathcal { T } _ { L } ; h _ { c } )$ with a specific budget B.

Consider a classifier model $f _ { c }$ for category c, which is trained on a c-category labeled dataset. The adapted general training loss compared to the true labeling function $\hat { f } _ { c }$ is defined as:

$$
\operatorname* { P r } _ { x } \left( \hat { f } _ { c } ( x ) \neq f _ { c } ( x ) \right) \leq \left( 1 - \operatorname { \mathbb { E } } _ { x } \left[ \operatorname* { m a x } _ { x ^ { \prime } \in \mathcal { T } _ { L } } h _ { c } ( x , x ^ { \prime } ) \right] \right) + \operatorname { \mathbb { E } } _ { x } \left[ \operatorname* { m a x } _ { x ^ { \prime } : f _ { c } ( x ) \neq f _ { c } ( x ^ { \prime } ) } h _ { c } ( x , x ^ { \prime } ) \right]\tag{3}
$$

This training error decreases when there exists a candidate sample x which is highly similar to $x ^ { \prime }$ and has the same category in the feature space, and less overlap between diferent classes of $x ^ { \prime }$ and $x .$ To optimize the first expectation on the right side of training error, we define a concept as follows:

Definition 1 (P−Coverage function) With $h _ { c }$ is a nonnegative function as denoted above, we define a coverage gain term as follows:

$$
\Delta h _ { c } ( x ; \mathcal { T } _ { L } ) = P ( \mathcal { T } _ { L } \cup \{ x \} ; h _ { c } ) - P ( \mathcal { T } _ { L } ; h _ { c } ) ,\tag{4}
$$

where $P ( \mathcal T _ { L } ; h _ { c } ) = \mathbb { E } _ { x \sim p ( x \mid y = c ) } \left[ \operatorname* { m a x } _ { x ^ { \prime } \in \mathcal T _ { L } } h _ { c } ( x , x ^ { \prime } ) \right] .$

The above definition is a straightforward extension from region-based selection of [18, 19] to the coverage of a specific category c. The following lemma states the monotonicity of the coverage function.

Lemma 1 (Monotonicity) Let $A \subseteq B$ . The coverage function $P ( \mathcal { T } _ { L } ; h _ { c } )$ is monotone nondecreasing,

$$
P ( A ; h _ { c } ) \leq P ( B ; h _ { c } ) .\tag{5}
$$

Proof For any sample $x ,$ since $A \subseteq B ,$ , we have

$$
\operatorname* { m a x } _ { x ^ { \prime } \in A } h _ { c } ( x , x ^ { \prime } ) \leq \operatorname* { m a x } _ { x ^ { \prime } \in B } h _ { c } ( x , x ^ { \prime } ) .
$$

Taking expectation with respect to $p ( x | y = c )$ on both sides,

$$
P ( A ; h _ { c } ) = \mathbb { E } _ { x \sim p ( x \mid y = c ) } \left[ \operatorname* { m a x } _ { x ^ { \prime } \in A } h _ { c } ( x , x ^ { \prime } ) \right] \leq \mathbb { E } _ { x \sim p ( x \mid y = c ) } \left[ \operatorname* { m a x } _ { x ^ { \prime } \in B } h _ { c } ( x , x ^ { \prime } ) \right] = P ( B ; h _ { c } ) .
$$

Therefore, $P ( \mathcal { T } _ { L } ; h _ { c } )$ is monotone non-decreasing.

Built upon the coverage gain definition, our objective is to quantify the utility of a queried sample by the increase in category coverage gain. To compute the coverage $P ( \mathcal { T } _ { L } ; h _ { c } )$ term, we use Monte-Carlo approximation:

$$
\begin{array} { r } { P ( \mathcal { T } _ { L } ; h _ { c } ) = \mathbb { E } _ { x \sim p ( x \mid y = c ) } \left[ \underset { x ^ { \prime } \in \mathcal { T } _ { L } } { \operatorname* { m a x } } h _ { c } ( x , x ^ { \prime } ) \right] } \\ { \approx \frac { 1 } { N _ { \mathcal { T } _ { U } } } \underset { i = 1 } { \overset { N _ { \mathcal { T } _ { U } } } { \sum } } \underset { \mathbf { x ^ { \prime } } \in \mathcal { T } _ { L } } { \operatorname* { m a x } } h _ { c } ( x _ { i } , x ^ { \prime } ) } \end{array}\tag{6}
$$

Then, we can rewrite the coverage gain in $\operatorname { E q . }$ 4 in detail as:

$$
\Delta h _ { c } ( x ; \mathcal { T } _ { L } ) \approx \frac { 1 } { N _ { \mathcal { T } _ { U } } } \sum _ { i = 1 } ^ { N _ { \mathcal { T } _ { U } } } \operatorname* { m a x } _ { { \bf x ^ { \prime } } \in \mathcal { T } _ { L } \cup \{ x \} } h _ { c } ( x _ { i } , x ^ { \prime } ) - \frac { 1 } { N _ { \mathcal { T } _ { U } } } \sum _ { i = 1 } ^ { N _ { \mathcal { T } _ { U } } } \operatorname* { m a x } _ { { \bf x ^ { \prime } } \in \mathcal { T } _ { L } } h _ { c } ( x _ { i } , x ^ { \prime } )\tag{7}
$$

Our approach considers each sample x of unlabeled target pool $\mathcal { T } _ { U } .$ , adds it to $\mathcal { T } _ { L }$ then computes the coverage gain. After that, we choose the sample that has the maximal $\Delta h _ { c } ( x ; \mathcal { T } _ { L } )$ . This approach relies on the characteristics of the extracted features rather than on the regression parameters, which are quite sensitive during the training process.

In the category-based setting, the category c can be viewed as a class-conditional location to select the next batches. This information is integrated into our selection mechanism through the positive manifold score. The positive manifold score approximates the efect of selecting a specific sample on the coverage of a given class c. We first give a definition of the positive manifold region.

Definition 2 (Positive Coverage Gain) Let c denote the target category and let

$$
y _ { c } = { \left\{ \begin{array} { l l } { 1 , } & { y = c , } \\ { 0 , } & { y \neq c , } \end{array} \right. }
$$

be the corresponding binarized label. The positive manifold associated with category c is defined as the support of the conditional distribution

$$
p ( x \mid y _ { c } = 1 ) ,
$$

which consists of all labeled samples belonging to category c. We denote this manifold by

$$
\mathcal { M } _ { c } ^ { + } = \{ x \in \mathcal { T } _ { L } : y _ { c } = 1 \} .
$$

The coverage gain of a sample x is defined as:

$$
\Delta h _ { c } ( x ; \mathcal { M } _ { c } ^ { + } ) \approx \frac { 1 } { N _ { \mathcal { T } _ { U } } } \sum _ { i = 1 } ^ { N _ { \mathcal { T } _ { U } } } \operatorname* { m a x } _ { { \bf x } ^ { \prime } \in \mathcal { M } _ { c } ^ { + } \cup \left\{ x \right\} } h _ { c } ( x _ { i } , x ^ { \prime } ) - \frac { 1 } { N _ { \mathcal { T } _ { U } } } \sum _ { i = 1 } ^ { N _ { \mathcal { T } _ { U } } } \operatorname* { m a x } _ { { \bf x } ^ { \prime } \in \mathcal { M } _ { c } ^ { + } } h _ { c } ( x _ { i } , x ^ { \prime } )\tag{8}
$$

This equation ensures the positive region coverage naturally discourages selecting redundant samples that are already well represented by the current labeled set.

To construct this positive manifold region, we compute the kernel between all label samples that have a positive category and the whole data pool. However, in the first iteration, we do not have accessibility to the label of the target pool. Therefore, we employ a warm-up phase by randomly selecting samples to add in the first labeled pool. An alternative choice is to utilize only positive labeled samples from the source domain. However, the resulting positive manifold estimation may result in bias to the source domain, which diverges from the target data distribution if the target and the source domains are distant from each other.

Although positive manifold coverage encourages the expansion of data distribution to reduce redundancy, it implicitly assumes that the informativeness of all regions of the positive manifold is equal. In particular, samples for which the current category classifier is uncertain may provide more useful supervision than samples that are already classified with high confidence. Therefore, we use the predictive uncertainty of the category classifier as an informativeness guide so that regions near the current category decision boundary receive larger emphasis while the overall coverage structure is still retained. The uncertainty measure is defined through the entropy of a predicted probability as:

$$
\mathcal { U } ( x ; f _ { c } ) = - p \log _ { 2 } p - ( 1 - p ) \log _ { 2 } ( 1 - p )\tag{9}
$$

with $p = P ( y = 1 | x )$ is the predicted probability of sample $x .$ The positive manifold score in Eq. 8 is used to combine with the uncertainty measure

$$
s ( x , \mathcal { M } _ { c } ^ { + } ; f _ { c } ) = \frac { 1 } { N _ { T _ { U } } } \sum _ { i = 1 } ^ { N _ { T _ { U } } } \mathcal { U } ( x _ { i } ; f _ { c } ) \left( \operatorname* { m a x } _ { \mathbf { x } ^ { \prime } \in \mathcal { M } _ { c } ^ { + } \cup \left\{ x \right\} } h _ { c } ( x _ { i } , x ^ { \prime } ) - \operatorname* { m a x } _ { \mathbf { x } ^ { \prime } \in \mathcal { M } _ { c } ^ { + } } h _ { c } ( x _ { i } , x ^ { \prime } ) \right)\tag{10}
$$

The resulting score $s ( x , \mathcal { M } _ { c } ^ { + } ; f _ { c } )$ combines the uncertainty of both the point under consideration and the region it covers, rather than just the point itself. Note that, although the uncertainty signal $\mathcal { U } ( x ; f _ { c } )$ is a sample-based estimation, it is not used to select a new sample directly. Instead, it serves as a guidance signal for estimating category-aware coverage gain on the positive manifold.

Our framework is summarized in Algorithm 1. The overall setup follows the general setting of category-aware active domain adaptation [5]. After the warming up phase, for each iteration, our positive manifold coverage method starts by constructing the kernel function $h _ { c } .$ Then, the coverage gain and uncertainty score of each sample are estimated by employing Eq. 6 and Eq. 9. The new sample $x _ { i }$ is then selected to maximize the score $s ( x , \mathcal { M } _ { c } ^ { + } ; f _ { c } )$ in $\operatorname { E q }$ . 10.

For the complexity estimation, the time complexity of initial $h _ { c }$ is $O ( n _ { L } n _ { \mathcal { T } _ { U } } )$ , where $n _ { L }$ is the number of labeled samples and $n \tau _ { U }$ is the number of unlabeled samples. The uncertainty scores for all unlabeled samples can be computed in $O ( n _ { \pi } )$ , excluding the model inference cost. Computing the $s ( x , \mathcal { M } _ { c } ^ { + } ; f _ { c } )$ score will take $O ( n _ { T _ { U } } ^ { 2 } )$ . The arg max operation takes $O ( n \tau _ { U } )$ . This selection strategy loops for $\boldsymbol { B }$ rounds, so the overall time complexity is $O ( B n _ { T _ { I } } ^ { 2 } )$ . Consequently, the overall complexity of our proposed method is $O ( n _ { L } n _ { \mathcal { T } _ { U } } + B n _ { \mathcal { T } _ { U } } ^ { 2 } )$

## 3.3 Theory Analysis

In this section, we analyze the point-based behavior in a category-aware setting and compare it with our proposed region-based selection mechanism. The analysis provides a guarantee on reducing the redundant selection with our proposed method. We first assume a Lipschitz Continuity on the estimation of Category-based Influence Score:

Assumption 1 (Lipschitz Continuity of Category-based Influence) For a target category $c ,$ the category-wise influence score $I _ { c } ( x )$ is L -Lipschitz continuous in the feature space. That is, for any two target samples $x _ { i } , x _ { j } \in \mathcal { T } _ { U }$

$$
| I _ { c } ( x _ { i } ) - I _ { c } ( x _ { j } ) | \leq L _ { I } d ( g _ { i } , g _ { j } ) ,
$$

where $\tau _ { U }$ is unlabeled target pool, $g _ { i } = \mathcal { G } ( x _ { i } )$ and $g _ { j } = \mathcal { G } ( x _ { j } )$ are features extracted from base DA model $\mathcal { G }$ and $d ( \cdot , \cdot )$ denotes a distance metric in the feature space.

From this assumption, we present a proposition about the local concentration induced by Lipschitz in case of influence-based selection:

Proposition 1 (Lipschitz-induced Local Influence Concentration) Let $x ^ { \star } \in \mathcal { T } _ { U }$ be a high influence sample with $I _ { c } ( x ^ { \star } ) = M$ . For any radius $\lambda > 0$ , define the local neighborhood

Algorithm 1 Uncertainty-Aware Region-based Algorithm   
1: Input: Labeled source data pool ${ \overline { { s , } } }$ unlabeled target data pool $\overline { { \tau } }$ , budget B per   
round and total rounds $K$ and a base DA model ${ \mathcal { G } } .$   
2: Output: Classification model $f _ { c }$ for category c.   
3: for $c = 1$ to $C$ do   
4: Binarized the labeled data representations.   
5: Randomly select B samples, as demonstrated in Section $3 .$   
6: Initial the labeled pool is $\mathcal { T } _ { L } ^ { 1 } = \mathcal { S } \cup \boldsymbol { B }$ and the unlabeled pool is ${ \mathcal { T } } _ { U } ^ { 1 } = { \mathcal { T } } \backslash B .$   
7: Train an initial classification model $f ^ { \prime }$ with the $\mathcal { T } _ { L } ^ { 1 }$ pool.   
8: for $k = 2$ to $K$ do   
9: ${ \mathcal { S } } _ { B } = \emptyset$ is a temporary set of B selected samples in each active round.   
10: Compute $h _ { c } \in \dot { \mathbb { R } ^ { | \mathcal { T } _ { U } ^ { k - 1 } | } }$ , where $\begin{array} { r } { h _ { c } ^ { i } = \operatorname* { m a x } _ { x ^ { \prime } \in \mathcal { M } _ { c } ^ { + } } h _ { c } ( x _ { i } , x ^ { \prime } ) } \end{array}$ , and $x _ { i } \in \mathcal { T } _ { U } ^ { k - 1 }$   
11: for $b = 1$ to $\boldsymbol { B }$ do   
12: Calculate coverage gain for each candidate sample $\Delta h _ { c } ( x _ { i } ; \mathcal { M } _ { c } ^ { + } )$ with Eq. 6   
13: Calculate uncertainty score for each sample in $\mathcal { T } _ { U } ^ { k - 1 }$ with Eq. 9.   
14: Select $x ^ { * } = \arg \operatorname* { m a x } _ { x \in \mathcal { T } _ { \mathit { t } _ { J } } ^ { k - 1 } } s ( x , \mathcal { M } _ { c } ^ { + } ; f _ { c } )$ with $s ( x , \mathcal { M } _ { c } ^ { + } ; f _ { c } )$ is computed   
following the $\operatorname { E q . }$ 10.   
15: Update $S _ { B } = S _ { B } \cup \{ x ^ { * } \}$   
16: Update $h _ { c } ^ { i } \gets \operatorname* { m a x } ( h _ { c } ( x _ { i } , x ^ { * } ) , h _ { c } ^ { i } )$   
17: end for   
18: Update $\mathcal { T } _ { U } ^ { k } = \mathcal { T } _ { U } ^ { k - 1 } \setminus { S } _ { B }$   
19: Update $\hat { \mathcal { T } _ { L } ^ { k } } = \hat { \mathcal { T } _ { L } ^ { k - 1 } } \cup \mathcal { S } _ { B }$   
20: end for   
21: Retrain classification model $f ^ { \prime }$ with $\mathcal { T } _ { L } ^ { K }$   
22: Set classification model $f _ { c } = f ^ { \prime } .$   
23: end for

$r _ { \lambda } ( x ^ { \star } ) = \{ x \in \mathcal { T } _ { U } : d ( g , g ^ { \star } ) \leq \lambda \} . \ I f \ r _ { \lambda } ( x ^ { \star } )$ contains at least B samples and ${ \cal M } - { \cal L } _ { I } \lambda >$ max $z { \in } \mathcal { T } _ { U } \backslash r _ { \lambda } ( x ^ { \star } ) \ I _ { c } ( z )$ , then the influence top-B selection satisfies

$$
S _ { B } \subseteq r _ { \lambda } ( x ^ { \star } ) .
$$

Corollary 1 (Maximum distance of top-B selected samples) For any pair of $( x _ { i } , x _ { j } )$ in the top-B selection that is located in any radius $\lambda _ { i }$

$$
\operatorname* { m a x } _ { x _ { i } , x _ { j } \in S _ { B } } d ( g _ { i } , g _ { j } ) \leq 2 \lambda .
$$

Proof By the Lipschitz continuity of $I _ { c } ,$ for any $x \in r _ { \lambda } ( x ^ { \star } )$

$$
| I _ { c } ( x ) - I _ { c } ( x ^ { \star } ) | \leq L _ { I } d ( g , g ^ { \star } ) \leq L _ { I } \lambda .
$$

Therefore,

$$
I _ { c } ( x ) \geq I _ { c } ( x ^ { \star } ) - L _ { I } \lambda = M - L _ { I } \lambda .
$$

If ${ M - L _ { I } \lambda }$ is larger than the maximum influence score outside $r _ { \lambda } ( x ^ { \star } )$ , and $r _ { \lambda } ( x ^ { \star } )$ contains at least B samples, then the top-B samples ranked by $I _ { c } ( x )$ must all lie inside $r \lambda ( x ^ { \star } )$ . Hence

$S _ { B } \subseteq r _ { \lambda } ( x ^ { \star } )$ . Since every selected sample lies within radius $\lambda$ of $x ^ { \star }$ , by the triangle inequality, for any $x _ { i } , x _ { j } \in S _ { B }$

$$
d ( g _ { i } , g _ { j } ) \leq d ( g _ { i } , g ^ { \star } ) + d ( g ^ { \star } , g _ { j } ) \leq 2 \lambda .
$$

Therefore, Lipschitz continuity provides a theoretical explanation for local influence concentration. If influence score changes smoothly across features, then high influence samples will attract close ones as well. If that neighbor is densely populated and has a higher score than the outside, the influence-only top- $\boldsymbol { . } \boldsymbol { B }$ will choose many points close together. To solve this issue, we propose a region-based approach that encourages the selection strategy to choose a sample which can approximate the underlying data distribution. We formalize this intuition with the following proposition.

Proposition 2 (Redundancy reduction) Consider an unlabeled candidate $x , i f x$ is redundant with respect to the current selected $s e t ,$

$$
h _ { c } ( x _ { i } , x ) \leq \operatorname* { m a x } _ { x ^ { \prime } \in \mathcal { M } _ { c } ^ { + } } h _ { c } ( x _ { i } , x ^ { \prime } ) \quad f o r \ m o s t \ x _ { i } ,
$$

then

$$
\Delta h _ { c } ( x ; \mathcal { M } _ { c } ^ { + } ) \approx 0 .
$$

Proof By Eq. 8, we have

$$
\Delta h _ { c } ( x ; \mathcal { M } _ { c } ^ { + } ) \approx \frac { 1 } { N _ { \mathcal { T } _ { U } } } \sum _ { i = 1 } ^ { N _ { \mathcal { T } _ { U } } } \left[ \operatorname* { m a x } _ { x ^ { \prime } \in \mathcal { M } _ { c } ^ { + } \cup \left\{ x \right\} } h _ { c } ( x _ { i } , x ^ { \prime } ) - \operatorname* { m a x } _ { x ^ { \prime } \in \mathcal { M } _ { c } ^ { + } } h _ { c } ( x _ { i } , x ^ { \prime } ) \right] .
$$

Let $\begin{array} { r } { \mathcal { H } ( \mathcal { M } _ { c } ^ { + } ) = \operatorname* { m a x } _ { x ^ { \prime } \in \mathcal { M } _ { c } ^ { + } } h _ { c } ( x _ { i } , x ^ { \prime } ) } \end{array}$ . Then, ma $\mathrm { x } _ { x ^ { \prime } \in \mathcal { M } _ { c } ^ { + } \cup \{ x \} } h _ { c } ( x _ { i } , x ^ { \prime } )$ can be expressed as follows:

$$
\operatorname* { m a x } _ { x ^ { \prime } \in \mathcal { M } _ { c } ^ { + } \cup \left\{ x \right\} } h _ { c } ( x _ { i } , x ^ { \prime } ) = \operatorname* { m a x } \left\{ \mathcal { H } ( \mathcal { M } _ { c } ^ { + } ) , h _ { c } ( x _ { i } , x ) \right\} ,
$$

We have

$$
\Delta h _ { c } ( x ; \mathcal { M } _ { c } ^ { + } ) \approx \frac { 1 } { N _ { \mathcal { T } _ { U } } } \sum _ { i = 1 } ^ { N _ { \mathcal { T } _ { U } } } \left[ \operatorname* { m a x } \left\{ \mathcal { H } ( \mathcal { M } _ { c } ^ { + } ) , h _ { c } ( x _ { i } , x ) \right\} - \mathcal { H } ( \mathcal { M } _ { c } ^ { + } ) \right] ,
$$

This is equivalent to

$$
\begin{array} { r } { \Delta h _ { c } ( x ; \mathcal { M } _ { c } ^ { + } ) \approx \left\{ \begin{array} { l l } { \frac { 1 } { N _ { \mathcal { T } _ { U } } } \sum _ { i = 1 } ^ { N _ { \mathcal { T } _ { U } } } \left[ h _ { c } ( x _ { i } , x ) - \mathcal { H } ( \mathcal { M } _ { c } ^ { + } ) \right] , } & { \mathrm { ~ i f ~ } \mathcal { H } ( \mathcal { M } _ { c } ^ { + } ) \leq h _ { c } ( x _ { i } , x ) , } \\ { 0 } & { \mathrm { ~ o t h e r w i s e } . } \end{array} \right. } \end{array}
$$

We can rewrite it in the form of

$$
\Delta h _ { c } ( x ; \mathcal { M } _ { c } ^ { + } ) \approx \frac { 1 } { N _ { \mathcal { T } _ { U } } } \sum _ { i = 1 } ^ { N _ { \mathcal { T } _ { U } } } \left[ h _ { c } ( x _ { i } , x ) - \mathcal { H } ( \mathcal { M } _ { c } ^ { + } ) \right] ^ { + }
$$

with $[ \cdot ] ^ { + }$ denotes the positive-part operator, defined as $[ z ] ^ { + } = \operatorname* { m a x } ( z , 0 )$

Thus, if $\begin{array} { r } { h _ { c } ( x _ { i } , x ) \leq \operatorname* { m a x } _ { x ^ { \prime } \in \mathcal { M } _ { c } ^ { + } } h _ { c } ( x _ { i } , x ^ { \prime } ) } \end{array}$ for most $x _ { i } ,$ the corresponding terms are zero or small. □

While our region-based approach selects the candidate sample $x ,$ which has the maximum coverage gain $\Delta h _ { c } ( x ; \mathcal { M } _ { c } ^ { + } )$ . This constraint can reduce the redundant sample that is learned by the current model, in other words, the sample that the current model covers.

Corollary 2 (Reduction in returns) For any $A \subseteq B$ and candidate x $\notin B _ { i }$

$$
\Delta h _ { c } ( x ; A ) \ge \Delta h _ { c } ( x ; B ) .
$$

Proof Since the coverage function is monotone non-decsreasing (Lemma $1 )$ , we have $\mathcal { H } ( A ) \leq$ $\mathcal { H } ( B )$ , with $\mathcal { H } ( A ) , \mathcal { H } ( B )$ are computed as $\mathcal { H } ( \mathcal { M } _ { c } ^ { + } )$ in the Proposition 2. Thus,

$$
h _ { c } ( x _ { i } , x ) - { \mathcal { H } } ( A ) \geq h _ { c } ( x _ { i } , x ) - { \mathcal { H } } ( B ) .
$$

Since the positive-part operator is monotone, we have

$$
\begin{array} { r } { [ h _ { c } ( x _ { i } , x ) - \mathcal { H } ( A ) ] ^ { + } \geq [ h _ { c } ( x _ { i } , x ) - \mathcal { H } ( B ) ] ^ { + } . } \end{array}
$$

Summing over all $x _ { i }$ samples and dividing by $N _ { \mathcal { T } _ { U } }$ , we obtain

$$
\Delta h _ { c } ( x ; A ) \ge \Delta h _ { c } ( x ; B ) .
$$

In our framework, we incorporate the uncertainty estimation of the candidate region as in $\operatorname { E q } .$ . 10. Therefore, it is required that the uncertainty weight is able to preserve the positive region coverage. To prove that, we first assume a uniform distribution on the estimation of uncertainty $\mathcal { U } \mathrm { : }$

Assumption 2 (Calibrated uncertainty signal) Let $\alpha \in ( 0 , 1 )$ , the uncertainty weights are uniformly bounded:

$$
0 < \alpha \leq \mathcal { U } ( x _ { i } ; f _ { c } ) \leq 1 , \quad \forall i \in \{ 1 , \dots , N _ { \mathcal { T } _ { U } } \} .
$$

Theorem 3 (Bounded uncertainty preserves coverage gain) For any labeled set $\mathcal { M } _ { c } ^ { + }$ and any candidate $x \in \mathcal { M } _ { c } ^ { + }$ , the uncertainty-weighted coverage gain satisfies

$$
\alpha \Delta h _ { c } ( x ; \mathcal { M } _ { c } ^ { + } ) \leq s ( x , \mathcal { M } _ { c } ^ { + } ; f _ { c } ) \leq \Delta h _ { c } ( x ; \mathcal { M } _ { c } ^ { + } ) .
$$

Therefore, uncertainty weighting preserves the positive region coverage gain up to multiplicative constants.

Proof We rewrite Eq. 10 as the form of Eq. 8 in Proposition $^ { 2 , }$

$$
s ( x , \mathcal { M } _ { c } ^ { + } ; f _ { c } ) = \frac { 1 } { N _ { \mathcal { T } _ { U } } } \sum _ { i = 1 } ^ { N _ { \mathcal { T } _ { U } } } \mathcal { U } ( x _ { i } ; f _ { c } ) \left[ h _ { c } ( x _ { i } , x ) - \mathcal { H } ( \mathcal { M } _ { c } ^ { + } ) \right] ^ { + } .
$$

Since $0 < \alpha \leq { \mathcal { U } } ( x _ { i } ; f _ { c } ) \leq 1$ , we have

$$
\alpha \left[ h _ { c } ( x _ { i } , x ) - { \mathcal { H } } ( M _ { c } ^ { + } ) \right] ^ { + } \leq { \mathcal { U } } ( x _ { i } ; f _ { c } ) \left[ h _ { c } ( x _ { i } , x ) - { \mathcal { H } } ( M _ { c } ^ { + } ) \right] ^ { + } \leq \left[ h _ { c } ( x _ { i } , x ) - { \mathcal { H } } ( M _ { c } ^ { + } ) \right] ^ { + } .
$$

Summing over all $x _ { i }$ samples and dividing by $N _ { \mathcal { T } _ { U } }$ , we obtain

$$
\alpha \Delta h _ { c } ( x ; \mathcal { M } _ { c } ^ { + } ) \leq s ( x , \mathcal { M } _ { c } ^ { + } ; f _ { c } ) \leq \Delta h _ { c } ( x ; \mathcal { M } _ { c } ^ { + } ) .
$$

In CAADA [5], the authors provide category awareness through a pointwise influence predictor but do not explicitly model redundancy among selected samples. MaxHerding [18] gives a framework which can reduce region-level redundancy through generalized coverage. MaxHerding’s coverage is defined over the entire data distribution, without distinguishing whether a sample contributes to the coverage of a particular category. Thus, these selected samples are class-agnotic. Our method bridges these two views: it inherits the region-aware coverage principle of MaxHerding, while introducing the category-based setting in active domain adaptation. Our novelty formulation restricts the coverage measure to the positive region of the target class and then guides the selection strategy by the uncertainty signal. The uncertainty-weighted selection principle in our approach is common in active learning, and has been shown to be more efective than using only one of the two signals individually in active domain adaptation [3]. For candidates with comparable category relevance, the proposed objective prefers the candidate that yields a larger positive region coverage gain. Hence, it preserves the category-aware motivation while reducing the redundancy caused by independent top-B pointwise selection.

## 4 Experiments

## 4.1 Settings

To evaluate the eficiency of our proposed method, we compare it with five active domain adaptation baselines and a category-aware active domain adaptation, which include Random, DANN [1], CLUE [3], DUC [4], ISAL [16], CAADA [5].

The experiments are carried out on three popular benchmarks in the domain adaptation field: Ofice-Home[20], DomainNet-126 [21] and VisDa-2017 [22]:

• Ofice-Home [20]: comprises four domains: Real-World, Product, Clip Art, and Artistic. Each domain includes images of 65 categories.

• DomainNet-126 [21]: We follow the same protocol as [5] to choose 126 categories and six-directions from four domains: Real, Clipart, Painting, and Sketch.

• VisDa-2017 [22]: is separated into training, validation, and testing domains and comprises 12 categories.

We use the ResNet-50 [23] is trained from scratch as the backbone feature extractor and a DANN [1] model is trained as the base DA method following CLUE [3]. The feature representations for both the labeled source domain and the unlabeled target domain are extracted from a fixed feature extractor pre-trained by the base DA method. For each category c, we binarize the source pool and train the initial model $f ^ { \prime }$ with an $L _ { 2 }$ weight of 1e − 3. We perform active learning in five iterations. With Ofice-Home, we set B = 1% of the total target dataset, and $B = 1 0 0$ for DomainNet-126 and VisDa-2017. For the kernel function, we use the Gaussian kernel $\begin{array} { r } { h _ { c } ( x , x ^ { \prime } ) = \exp ( \frac { 1 } { 2 \sigma ^ { 2 } } | | x - x ^ { \prime } | | ^ { 2 } ) } \end{array}$ , with the batch size is 512, and the kernel width $\sigma = 0 . 7$ . We apply the same training and testing protocol in the experiments for baseline methods, and then report the average per-category accuracy.

Table 1 Average per-category predicting accuracy (%) for our proposed method on Ofice-Home. Avg. column represents the average accuracy across all 6 tasks in the same table.
<table><tr><td>一</td><td>Ar→Cl</td><td> $\mathrm { A r } {  } \mathrm { P r }$ </td><td> $\mathrm { A r } {  } \mathrm { R w }$ </td><td> $\mathrm { C l } {  } \mathrm { A r }$ </td><td></td><td> $\mathrm { C l } {  } \mathrm { P r }$ </td><td>Cl→Rw</td><td>Avg.</td></tr><tr><td>DANN</td><td> $3 6 . 9 9 \pm 0 . 5 2$ </td><td> $5 1 . 8 3 \pm 0 . 2 0$ </td><td> $5 8 . 8 1 \pm 0 . 2 2$ </td><td></td><td> $3 6 . 5 4 \pm 0 . 1 2$ </td><td> $4 8 . 5 9 \pm 0 . 3 5$ </td><td> $5 9 . 1 7 \pm 0 . 4 3$ </td><td>48.65</td></tr><tr><td>Random</td><td> $4 7 . 1 9 \pm 0 . 6 0$ </td><td> $5 5 . 4 1 \pm 0 . 5 5$ </td><td> $7 0 . 1 4 \pm 0 . 3 4$ </td><td> $5 0 . 9 2 \pm 0 . 3 6$ </td><td></td><td> $6 5 . 0 8 \pm 0 . 3 6$ </td><td> $6 2 . 7 7 \pm 0 . 3 3$ </td><td>58.58</td></tr><tr><td>CLUE</td><td> $4 8 . 1 4 \pm 0 . 7 6$ </td><td> $5 9 . 3 4 \pm 0 . 7 3$ </td><td> $7 4 . 3 4 \pm 0 . 6 0$ </td><td> $5 4 . 2 3 \pm 0 . 5 2$ </td><td></td><td> $7 8 . 7 4 \pm 0 . 8 9$ </td><td> $6 6 . 9 2 \pm 0 . 8 8$ </td><td>63.62</td></tr><tr><td>DUC</td><td> $5 1 . 2 3 \pm 0 . 3 4$ </td><td> $6 1 . 8 1 \pm 0 . 1 2$ </td><td> $7 5 . 3 1 \pm 0 . 2 3$ </td><td> $5 6 . 4 1 \pm 0 . 3 3$ </td><td></td><td> $7 9 . 8 2 \pm 0 . 2 4$ </td><td> $7 0 . 0 2 \pm 0 . 4 4$ </td><td>65.76</td></tr><tr><td>ISAL</td><td> $6 4 . 8 4 \pm 0 . 2 2$ </td><td> $7 9 . 2 3 \pm 0 . 1 2$ </td><td> $7 8 . 6 6 \pm 0 . 3 2$ </td><td> $6 2 . 9 3 \pm 0 . 5 4$ </td><td></td><td> $8 1 . 2 4 \pm 0 . 5 5$ </td><td> $7 5 . 6 2 \pm 0 . 3 2$ </td><td>73.75</td></tr><tr><td>CAADA</td><td> $6 9 . 1 7 \pm 0 . 1 2$ </td><td> $8 8 . 4 2 \pm 0 . 1 0$ </td><td> $8 6 . 2 5 \pm 0 . 2 2$ </td><td> $6 8 . 8 9 \pm 0 . 1 3$ </td><td></td><td> $8 4 . 4 7 \pm 0 . 1 5$ </td><td> $8 2 . 7 4 \pm 0 . 4 3$ </td><td>81.01</td></tr><tr><td>Ours</td><td> $\mathbf { 7 4 . 2 3 \ : \pm { \ : 0 . 1 0 } }$ </td><td> ${ \bf 9 4 . 0 8 \ : \pm { \ : 0 . 0 5 } }$ </td><td> $\mathbf { 9 4 . 0 1 \ : \pm { \ : 0 . 0 6 } }$ </td><td> ${ \bf 8 0 . 1 0 \pm 0 . 1 2 }$ </td><td></td><td> $\mathbf { 9 0 . 7 7 \ : \pm { \ : 0 . 0 9 } }$ </td><td> ${ \bf 8 9 . 1 9 \pm 0 . 1 9 }$ </td><td>87.06</td></tr><tr><td></td><td>一 Pr→Ar</td><td>Pr→Cl</td><td>Pr→Rw</td><td>Rw→Ar</td><td></td><td>Rw→Cl</td><td>Rw→Pr</td><td>Avg.</td></tr><tr><td>DANN</td><td> $3 2 . 8 3 \pm 0 . 4 7$ </td><td></td><td></td><td> $5 2 . 2 8 \pm 0 . 2 7$ </td><td></td><td></td><td></td><td></td></tr><tr><td>Random</td><td> $4 8 . 4 1 \pm 0 . 2 2$ </td><td> $3 2 . 9 8 \pm 0 . 3 2$ </td><td> $5 9 . 2 6 \pm 0 . 3 3$   $7 2 . 1 1 \pm 0 . 1 8$ </td><td> $6 1 . 9 2 \pm 0 . 2 6$ </td><td></td><td> $6 8 . 9 3 \pm 0 . 2 9$ </td><td> $4 5 . 3 6 \pm 0 . 3 6$ </td><td>48.61</td></tr><tr><td>CLUE</td><td> $5 3 . 1 7 \pm 0 . 4 5$ </td><td> $4 5 . 7 7 \pm 0 . 2 7$   $5 9 . 8 3 \pm 0 . 4 6$ </td><td> $7 5 . 1 3 \pm 0 . 3 9$ </td><td> $6 5 . 1 7 \pm 0 . 5 4$ </td><td></td><td> $7 7 . 6 1 \pm 0 . 3 5$   $8 0 . 4 2 \pm 0 . 3 2$ </td><td> $5 5 . 7 8 \pm 0 . 1 3$   $6 1 . 1 4 \pm 0 . 4 7$ </td><td>60.26 65.81</td></tr><tr><td>DUC</td><td> $5 9 . 2 3 \pm 0 . 1 3$ </td><td> $6 0 . 7 3 \pm 0 . 1 6$ </td><td> $7 7 . 3 4 \pm 0 . 2 7$ </td><td> $6 7 . 3 5 \pm 0 . 3 3$ </td><td></td><td> $7 9 . 6 8 \pm 0 . 1 2$ </td><td> $6 5 . 0 2 \pm 0 . 3 0$ </td><td>68.22</td></tr><tr><td>ISAL</td><td> $6 4 . 1 7 \pm 0 . 2 3$ </td><td> $6 5 . 0 2 \pm 0 . 2 5$ </td><td> $8 3 . 5 1 \pm 0 . 3 2$ </td><td> $7 1 . 4 2 \pm 0 . 3 5$ </td><td></td><td> $8 0 . 4 1 \pm 0 . 3 6$ </td><td> $7 1 . 5 1 \pm 0 . 2 5$ </td><td>72.67</td></tr><tr><td>CAADA</td><td> $6 9 . 7 5 \pm 0 . 1 5$ </td><td> $6 7 . 8 5 \pm 0 . 1 7$ </td><td> $8 8 . 3 1 \pm 0 . 1 2$ </td><td> $7 9 . 6 1 \pm 0 . 2 1$ </td><td></td><td> $9 2 . 1 1 \pm 0 . 2 5$ </td><td> $7 4 . 2 2 \pm 0 . 3 5$ </td><td>78.64</td></tr><tr><td>Ours</td><td> $\mathbf { 8 5 . 7 0 \ : \pm { \ : 0 . 1 2 } }$ </td><td> ${ \bf 7 4 . 5 9 \pm 0 . 1 8 }$ </td><td> ${ \bf 9 3 . 8 7 \pm 0 . 0 8 }$ </td><td>87.28 ± 0.14</td><td></td><td> $\mathbf { 9 4 . 1 0 \ : \pm { \ : 0 . 0 9 } }$ </td><td>94.98 ± 0.13 88.39</td><td></td></tr></table>

<table><tr><td rowspan="2"></td><td colspan="10">DomainNet-126</td><td colspan="2">VisDA-2017</td></tr><tr><td>cl→pt</td><td></td><td>cl→rl</td><td>cl→sk</td><td></td><td>sk→cl</td><td>sk→pt</td><td></td><td>sk→rl</td><td>Avg.</td><td></td><td>S→R</td></tr><tr><td>DANN</td><td>34.18 ± 0.37</td><td></td><td> $4 2 . 1 5 \pm 0 . 3 2$ </td><td> $3 6 . 1 2 \pm 0 . 2 9$ </td><td></td><td> $3 1 . 8 2 \pm 0 . 4 1$ </td><td> $5 6 . 5 9 \pm 0 . 3 8$ </td><td></td><td>47.24 ± 0.37</td><td>41.35</td><td></td><td>47.84 ± 0.28</td></tr><tr><td>Random</td><td>36.25 ± 0.54</td><td></td><td> $4 4 . 7 5 \pm 0 . 4 3$ </td><td>40.74 ±</td><td>0.49 34.41</td><td>± 0.53</td><td>58.31 ±</td><td>0.39</td><td>49.53 ± 0.31</td><td>44.00</td><td></td><td>54.19 ± 0.33</td></tr><tr><td>CLUE</td><td>38.06 ± 0.34</td><td></td><td>45.34 ± 0.36</td><td>42.28</td><td>± 0.46 35.63</td><td>± 0.48</td><td>60.12 ±</td><td>0.44</td><td>57.37 ± 0.43</td><td>46.47</td><td></td><td>58.54 ± 0.36</td></tr><tr><td>DUC</td><td>37.53 ± 0.37</td><td></td><td>48.91 ± 0.33</td><td>39.84 ±</td><td>0.29 36.12</td><td>± 0.27</td><td>61.35 ±</td><td>0.33 58.26</td><td>± 0.39</td><td>47.02</td><td></td><td>60.12 ± 0.40</td></tr><tr><td>ISAL</td><td>45.05 ± 0.19</td><td></td><td>56.94 ± 0.21</td><td> $5 1 . 3 1 \pm \ : 0 . 1 6$ </td><td>40.34</td><td>± 0.26</td><td>62.27 ±</td><td>0.30 61.64</td><td>± 0.24</td><td>52.92</td><td></td><td>62.31 ± 0.17</td></tr><tr><td>CAADA</td><td>52.23 ± 0.13</td><td></td><td>64.21 ± 0.14</td><td> $5 6 . 3 4 \pm 0 . 1 6$ </td><td>42.49</td><td>± 0.20</td><td>67.94 ±</td><td>0.19 63.41</td><td>± 0.21</td><td>57.78</td><td></td><td>66.68 ± 0.23</td></tr><tr><td>Ours</td><td>79.53 ± 0.14</td><td></td><td>86.85</td><td>± 0.17 81.13 ± 0.13 88.17± 0.19</td><td></td><td></td><td>81.57 ±</td><td>0.20 86.30</td><td>± 0.17</td><td>83.93</td><td></td><td>79.37 ± 0.18</td></tr></table>

Table 2 Average per-category predicting accuracy (%) for Category-Aware Active Domain Adaptation on DomainNet-126 and VisDA-2017. Avg. column represents the average accuracy across all 6 tasks of DomainNet-126 in this table.

## 4.2 Main results

We report the average accuracy per category in Table 1 and Table 2. As shown in these two tables, our proposed method achieved the highest average accuracy per category in all tasks of three benchmarks, obtaining an average accuracy of 87.73% on Ofice-Home, 83.93% on DomainNet-126, and 79.37% on VisDA-2017.

In the Ofice-Home dataset, our approach outperforms the closest setting, CAADA, in all tasks with the average diference is 8.41%. While CAADA performs a competitive improvement with the other baselines, our proposed method shows notable superiority with the average diference is 26.16% in the DomainNet-126 dataset. With the adaptation from Synthetic-to-Real domain of the VisDA-2017 dataset, our proposed method outperforms the baselines with 12.69% as well.

The improvement observed across diferent source-target pairs indicates that the advantage is not restricted to a particular adaptation task. Moreover, the results observed on all benchmarks are consistent with our motivation that positive-manifold coverage can help identify more informative samples within the target category of interest, thereby making more efective use of the limited annotation budget.

![](images/6967a4db95bb7383fae9643c1ebe07bc76869eb0782679403e814ef4bcaf20d7.jpg)  
Fig. 4 KDE plot for behavior of selection strategy of ours and CAADA in the 2nd active round with two categories, ”Batteries” and ”Trash Can”, in the Ar→Cl task of the OficeHome dataset.

## 4.3 Ablation study

## 4.3.1 Analysis on the quality of selection samples

To highlight the improvement of our informative positive manifold coverage strategy, Figure 4 shows the KDE plot of the behavior of our approach compared with the CAADA method on two categories of the Ofice-Home dataset.

(b) Comparison of selected concentration of ours methoc

Consider the ”Batteries” category, the two figures exhibit noticeably diferent sample selection behaviors. As discussed in Section 3, the CAADA mainly focuses on the high-density center, while distant examples are being ignored. In contrast, our method shows a substantially broader coverage that closely matches the distribution of the full category. For the ”Trash Can” category, the diference between the two methods is even more obvious. CAADA concentrates the central region and largely overlooks the distribution tails. This observation suggests that our approach ofers broader category coverage and more eficiently maintains category-specific diversity compared to the point-wise selection behavior.

![](images/3febc13ed46fa2943d4d5faac13e4707964d1c6625b0a40fbcd564715f72e303.jpg)  
Fig. 5 Comparison data distribution between CAADA and our selection method with the ”Curtains” category in the Ar→Cl task on the Ofice-Home dataset.

In Figure 5 we choose the ”Curtains” category of the Ofice-Home benchmark to compare the quality of chosen samples. Compared with CAADA, our method selects samples that are more uniformly distributed across the target manifold, reducing sample redundancy and improving category coverage. This is reflected in the substantially lower MMD value, leading to a 4.76% performance gain and 97.62% accuracy.

![](images/c3142871ec68fd8711d14a8995e7abae4b0bac40eda50a452174e693c9cb60d1.jpg)

![](images/30686e2a8d079b2a675809a0adbac429a19e8d55d5d889d1a92c06f326f367de.jpg)

![](images/ace47ab2ca137e83406c3384e894116736bb5a2dfac363eaa94ce16942455dce.jpg)  
Fig. 6 Classification loss on the validation set and predicting accuracy on the target domain in three diferent categories in the Ar→Pr task on Ofice-Home dataset.

We demonstrate the convergence of model performance as well as the MMD difference after each query round in Fig. 6. In this experiment, we choose the budget B = 10 and run with 80 active rounds. The MMD diference varies only slightly after 5 rounds and remains nearly unchanged from round 10 onward, whereas the target accuracy steadily increases and eventually converges to its upper limit.

We further compare the coverage achieved by our proposed method with that of random sampling, where samples are selected independently and identically distributed. As shown in Fig. 7, the coverage gap gradually increases over the iterative sampling rounds, indicating that our method selects samples that provide broader coverage of the target category. This improved coverage contributes to better performance of our proposed method.

## 4.3.2 Analysis on contribution of each component

Table 3 demonstrates that each component also contributes to the improvement of our model. Compared with CAADA, our Coverage Only boosts the performance by 6.61% while adding a weighted-region scheme further improves the accuracy by 14.15% with a small standard deviation, about 0.05%. Together, these two modules increase the performance by 20.76% and reduce the deviation compared with the baseline CAADA. Moreover, using coverage for sample selection significantly reduces the runtime, with Coverage Only and our method achieving runtime reductions of about 94.20% and 87.94%, respectively, compared with CAADA.

![](images/23a9305706a693d96a7bf2174b4c350d7d3bed0487f93cf4f6ae387a8a113d2c.jpg)

![](images/20938660fd963d11f454259311fb0d0bc106135c4352acc01ebb18d630237f44.jpg)  
Fig. 7 Comparision between the coverage of our proposed method with the random sampling method on the target domain in two diferent categories in the $\mathrm { A r {  } C l }$ task on Ofice-Home dataset.

Table 3 Accuracy and runtime with diferent model components for domain adaptation task Rw→Pr on Ofice-Home dataset with three diferent seed runs.
<table><tr><td>Model</td><td>Coverage</td><td>Region Wt.</td><td>Acc. (%)</td><td>Time (s)</td></tr><tr><td>CAADA</td><td>X</td><td>X</td><td> $7 4 . 2 2 \pm 2 . 3 6$ </td><td>11407.54</td></tr><tr><td>Coverage Only</td><td>√</td><td>X</td><td> $8 0 . 8 3 \pm 0 . 0 6$ </td><td>664.75</td></tr><tr><td>Coverage + Region Wt.</td><td>√</td><td>√</td><td> $9 4 . 9 8 \pm 0 . 0 5$ </td><td>1376.05</td></tr></table>

![](images/01236d5b00203c2d3985a15b4e2b182b7bb447ec836d4a204f589d45b593cb03.jpg)

![](images/f0237c589b0a26c23783c0124d936016ec2604e401ffb667eac875d5cc71fcf8.jpg)  
Fig. 8 Comparision between the uncertainty of selected samples of our proposed method with the Coverage Only sampling method on the target domain in two diferent categories in the Rw→Pr task on Ofice-Home dataset.

To further investigate the 14.15% improvement of uncertainty weighting over Coverage Only, we analyze the uncertainty of the samples selected by the two strategies in two diferent categories, Clipboard and Computer. As shown in Fig. 8, our method consistently selects samples with substantially higher uncertainty, whereas Coverage Only predominantly selects samples with near-zero uncertainty.

## 4.3.3 Analysis on the scoring mechanism

We additionally examine whether incorporating the negative-manifold term into Eq. 10 provides complementary information for target selection. The detail of our combination can be formulated as follows:

$$
s ( x ; f _ { c } ) = \alpha \cdot s ( x , \mathcal { M } _ { c } ^ { + } ; f _ { c } ) + \beta \cdot \Delta h _ { c } ( x ; \mathcal { M } _ { c } ^ { - } )
$$

with $\Delta h _ { c } ( x ; \mathcal { M } _ { c } ^ { - } )$ is defined as

Table 4 Accuracy with diferent combinations of positive and negative manifolds for domain adaptation task Rw→Pr on the Ofice-Home dataset.
<table><tr><td>α</td><td> $\beta$ </td><td>Acc. (%)</td></tr><tr><td>1</td><td>0</td><td> $9 4 . 9 8 \pm 0 . 0 1$ </td></tr><tr><td>1</td><td>1</td><td> $9 4 . 9 0 \pm 0 . 0 1$ </td></tr><tr><td>1</td><td>-1</td><td> $9 4 . 9 7 \pm 0 . 0 2$ </td></tr><tr><td>1</td><td>-0.5</td><td> $9 4 . 9 8 \pm 0 . 0 5$ </td></tr></table>

$$
\Delta h _ { c } ( x ; \mathcal { M } _ { c } ^ { - } ) = \frac { 1 } { N _ { \tau _ { U } } } \sum _ { i = 1 } ^ { N _ { \tau _ { U } } } \operatorname* { m a x } _ { { \bf x } ^ { \prime } \in \{ \mathcal { T } _ { L } \backslash { \mathcal { M } _ { c } ^ { + } } \} \cup \{ x \} } h _ { c } ( x _ { i } , x ^ { \prime } ) - \frac { 1 } { N _ { \tau _ { U } } } \sum _ { i = 1 } ^ { N _ { \tau _ { U } } } \operatorname* { m a x } _ { { \bf x } ^ { \prime } \in \{ \mathcal { T } _ { L } \backslash { \mathcal { M } _ { c } ^ { + } } \} } h _ { c } ( x _ { i } , x ^ { \prime } )
$$

As shown in Table 4, the resulting accuracy varies by only approximately 0.07%, with no consistent improvement observed when adjusting $\beta .$ This suggests that the negative manifold provides only marginal improvement. We therefore set $\beta = 0$ and use only the positive manifold in all main experiments.

## 5 Conclusion

In this paper, we introduced a category-aware active domain adaptation method that improves individual category performance while faithfully reproducing the target distribution of each category. Motivated by the redundancy and parameter sensitivity of point-wise, influence-based selection, we reformulated the acquisition as a regionbased problem and proposed an informative positive manifold coverage score. By restricting the coverage measure to the positive manifold of the target class and approximating the coverage gain through Monte-Carlo estimation, our selection strategy queries samples that expand the represented region. We further introduced an uncertainty guidance term that biases the coverage toward regions near the decision boundary, and we proved that this weighting preserves the positive coverage gain up to a multiplicative constant. Our theoretical analysis explains the localconcentration behaviour of influence-only selection through Lipschitz continuity and formally establishes the redundancy-reduction property of our region-based objective. Empirical results demonstrate that this method outperforms existing category-aware active domain adaptation baselines and other active domain adaptation methods across various benchmarks, achieving significant performance improvements. Given the promising results, one possible extension is to include a joint fine-tuning of the extractor and the classifier. Additionally, a finer sampling method can be used to balance the uncertainty-coverage trade-of in the scoring mechanism.

## References

[1] Ganin, Y., Ustinova, E., Ajakan, H., Germain, P., Larochelle, H., Laviolette, F., Marchand, M., Lempitsky, V.: Domain-adversarial training of neural networks. Journal of Machine Learning Research 17(59), 1–35 (2016)

[2] Fang, Z., Lu, J., Liu, F., Xuan, J., Zhang, G.: Open Set Domain Adaptation: Theoretical Bound and Algorithm (2020). https://arxiv.org/abs/1907.08375

[3] Prabhu, V., Chandrasekaran, A., Saenko, K., Hofman, J.: Active domain adaptation via clustering uncertainty-weighted embeddings. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 8505–8514 (2021)

[4] Xie, M., Li, S., Zhang, R., Liu, C.H.: Dirichlet-based Uncertainty Calibration for Active Domain Adaptation (2023). https://arxiv.org/abs/2302.13824

[5] Xiao, W.-X., Gu, J.-X., Liu, H.: Category-aware active domain adaptation. In: International Conference on Machine Learning (2024). https://api.semanticscholar.org/CorpusID:271746976

[6] Cook, R.D., Weisberg, S.: Characterizations of an empirical influence function for detecting influential cases in regression. Technometrics 22, 495–508 (1980)

[7] Farquhar, S., Gal, Y., Rainforth, T.: On Statistical Bias In Active Learning: How and When To Fix It (2021). https://arxiv.org/abs/2101.11665

[8] Zhan, X., Wang, Y., Chan, A.B.: Asymptotic optimality for active learning processes. In: Conference on Uncertainty in Artificial Intelligence (2022). https://api.semanticscholar.org/CorpusID:252898970

[9] Ganin, Y., Ustinova, E., Ajakan, H., Germain, P., Larochelle, H., Laviolette, F., March, M., Lempitsky, V.: Domain-adversarial training of neural networks. Journal of Machine Learning Research 17(59), 1–35 (2016)

[10] Deng, W., Liao, Q., Zhao, L., Guo, D., Kuang, G., Hu, D., Liu, L.: Joint clustering and discriminative feature alignment for unsupervised domain adaptation. IEEE Transactions on Image Processing 30, 7842–7855 (2021) https://doi.org/10.1109/ TIP.2021.3109530

[11] Saito, K., Kim, D., Sclarof, S., Darrell, T., Saenko, K.: Semi-supervised domain adaptation via minimax entropy. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) (2019)

[12] Basak, H., Yin, Z.: SemiDAViL: Semi-supervised Domain Adaptation with Vision-Language Guidance for Semantic Segmentation (2025). https://arxiv.org/abs/ 2504.06389

[13] Okuo, T., Matsuo, S., Harada, S., Tanaka, K., Bise, R.: Weakly-Supervised Domain Adaptation with Proportion-Constrained Pseudo-Labeling (2025). https: //arxiv.org/abs/2506.22301

[14] Doan, A.-D., Nguyen, B.L., Lim, T., Jayawardhana, M., Gupta, S., Guettier, C., Reid, I., Wagner, M., Chin, T.-J.: Weakly Supervised Test-Time Domain Adaptation for Object Detection (2024). https://arxiv.org/abs/2407.05607

[15] Gavves, E., Mensink, T., Tommasi, T., Snoek, C.G.M., Tuytelaars, T.: Active Transfer Learning with Zero-Shot Priors: Reusing Past Datasets for Future Tasks (2015). https://arxiv.org/abs/1510.01544

[16] Liu, Z., Ding, H., Zhong, H., Li, W., Dai, J., He, C.: Influence selection for active learning. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 9254–9263 (2021)

[17] Yehuda, O., Dekel, A., Hacohen, G., Weinshall, D.: Active Learning Through a Covering Lens (2022). https://arxiv.org/abs/2205.11320

[18] Bae, W., Noh, J., Sutherland, D.J.: Generalized Coverage for More Robust Low-Budget Active Learning (2024). https://arxiv.org/abs/2407.12212

[19] Bae, W., Oliveira, G.L., Sutherland, D.J.: Uncertainty Herding: One Active Learning Method for All Label Budgets (2025). https://arxiv.org/abs/2412. 20644

[20] Venkateswara, H., Eusebio, J., Chakraborty, S., Panchanathan, S.: Deep hashing network for unsupervised domain adaptation. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 5018–5027 (2017)

[21] Peng, X., Bai, Q., Xia, X., Huang, Z., Saenko, K., Wang, B.: Moment matching for multi-source domain adaptation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 1406–1415 (2019)

[22] Peng, X., Usman, B., Kaushik, N., Hofman, J., Wang, D., Saenko, K.: Visda: The visual domain adaptation challenge. arXiv preprint arXiv:1710.06924 (2017)

[23] He, K., Zhang, X., Ren, S., Sun, J.: Deep Residual Learning for Image Recognition (2015). https://arxiv.org/abs/1512.03385