# FedBST: Bilateral Self-Training for Unsupervised Federated Classification

Aijing Yu<sup>1,2</sup>, Kuangpu Guo<sup>4,5</sup>, Jian Liang<sup>3,4</sup>, and Xiao-Yu Zhang<sup>1,2,∗</sup>

<sup>1</sup>School of Cyber Security, University of Chinese Academy of Sciences, Beijing, China

<sup>2</sup>Institute of Information Engineering, Chinese Academy of Sciences, Beijing, China

<sup>3</sup>School of Artificial Intelligence, University of Chinese Academy of Sciences, Beijing, China

<sup>4</sup>NLPR & MAIS, Institute of Automation, Chinese Academy of Sciences, Beijing, China

<sup>5</sup>University of Science and Technology of China, Hefei, China

Email: zhangxiaoyu@iie.ac.cn

Abstract—Unsupervised federated learning with Vision-Language Models (e.g., CLIP) offers a promising solution for label-free classification, yet its performance often degrades under severe non-IID shifts. A pioneering method, FedCoPL, typically relies on self-training strategies that exclusively prioritize highconfidence predictions for pseudo-labeling. However, this selective approach neglects the information of ambiguous samples (often referred to as “waist” samples), leading to the underutilization of data and biased decision boundaries. To address this limitation, we propose FedBST, a bilateral self-training framework that effectively exploits both high-confidence and ambiguous data. Specifically, for local optimization, we introduce uncertaintyaware confidence penalization (UACP). UACP targets ambiguous samples by penalizing overconfident predictions to implicitly promote higher uncertainty, thereby preventing erroneous pseudo-labeling. At inference, we employ a visual-text ensemble strategy. This approach aggregates uncertainty-aware local visual prototypes and ensembles them with textual prototypes, achieving lightweight adaptation to local distributions via prototype-based calibration. Experiments demonstrate that FedBST achieves consistent improvements over existing methods with standard federated aggregation protocols.

Index Terms—unsupervised federated learning, visionlanguage models, uncertainty estimation, pseudo labeling

## I. INTRODUCTION

Federated Learning (FL) has established itself as a de facto paradigm for privacy-preserving collaborative training, allowing multiple decentralized clients to train a global model without sharing raw private data [1], [2]. While conventional FL relies heavily on high-quality annotated data [3], [4], obtaining such supervision is often impractical in real-world edge scenarios due to the prohibitive cost of manual labeling and strict privacy constraints [5]. Consequently, Unsupervised Federated Learning (UFL) has emerged as a critical research direction [6], [7]. Early UFL approaches primarily focused on representation learning or clustering [8], [9], which limits their applicability to direct classification tasks.

Recently, Large-scale Vision-Language Models (VLMs), such as CLIP [10], have significantly advanced this field by providing rich semantic supervision. By leveraging the zeroshot capability of VLMs, a pioneering work, FedCoPL [11] has explored UFL to Unsupervised Federated Classification (UFC) settings through cooperative pseudo-labeling, establishing strong baselines. This method employs a selective strategy that prioritizes high-confidence predictions as positive pseudolabels while discarding uncertain ones.

However, this selective paradigm exhibits two limitations under Non-IID shifts. First, the exclusion of ambiguous samples leads to biased decision boundaries. By ignoring informative “waist” samples near the boundaries, local models tend to overfit to high-confidence easy patterns, failing to learn the robust decision surfaces required for complex Non-IID data. Second, the exclusive reliance on global textual prompts limits inference adaptability. Such generic prompts, while semantically correct, lack the granularity to capture clientspecific visual variations. Relying on this global prior neglects the local visual distribution, thereby degrading performance when local data diverges from the generic textual description.

To address these challenges, we introduce FedBST, a Bilateral Self-Training framework designed to maximize data utilization and inference reliability. Unlike previous methods that rely solely on positive learning from high-confidence data, FedBST employs a bilateral optimization strategy.

For local training, we propose uncertainty-aware confidence penalization (UACP). Instead of discarding ambiguous samples, UACP treats them as valuable signals for regularization. Specifically, it explicitly suppresses the maximum predicted probability of these uncertain candidates. This operation penalizes overconfident predictions, thereby promoting higher predictive entropy to reduce overconfident early assignments to incorrect classes. For inference, we employ a visual-text ensemble strategy. This mechanism dynamically aggregates uncertainty-aware local visual prototypes and ensembles them with generic textual prototypes. This allows the model to leverage the stability of pre-trained textual knowledge while adaptively correcting predictions based on local visual distributions, all without additional parameter optimization. Our contributions are summarized as follows:

• We propose FedBST, a robust UFC framework that integrates bilateral self-training mechanisms. It enables the model to learn simultaneously from high-confidence data while regularizing with ambiguous samples, ensuring robust performance under heterogeneous settings.

• We design two complementary components: the uncertainty-aware confidence penalization (UACP) module to regularize decision boundaries using ambiguous data, and a visual-text ensemble (VTE) strategy to bridge generic textual semantics with local visual distributions for reliable inference.

• Extensive experiments on multiple benchmarks demonstrate that FedBST outperforms state-of-the-art methods in accuracy and robustness with standard server-side aggregation protocols.

## II. RELATED WORK

Unsupervised Federated Learning. Existing UFL approaches primarily focus on representation learning rather than direct classification. Methods like FedU [12] and FedEMA [13] address heterogeneity via divergence-aware updates, while others employ knowledge distillation [5], [6] or clustering [14] to transfer structural knowledge. However, these representationfocused methods cannot be directly applied to downstream classification tasks without labels.

Federated Learning with VLMs. VLMs like CLIP have been integrated into FL to reduce communication costs. Works such as PromptFL [15] and FedCLIP [16] optimize prompts but require supervision. Recent personalization methods [17]–[20] improve global-local alignment, but still depend on groundtruth, rendering them unsuitable for fully label-free scenarios. Unsupervised Federated Classification. FedCoPL [11] recently addressed this challenge by introducing a cooperative pseudo-labeling framework. It utilizes CLIP’s zero-shot predictions to generate pseudo-labels and mitigates label skew through global budget allocation. As a state-of-the-art baseline in this setting, FedCoPL demonstrates the feasibility of labelfree federated classification. However, it primarily relies on selecting high-confidence samples for training, which inherently neglects the informative value of ambiguous (“waist”) samples. Building upon this foundation, our work introduces a bilateral self-training strategy to effectively exploit both high-confidence and ambiguous data, further enhancing performance and robustness.

## III. METHOD

## A. Preliminaries

We consider the UFC setting with K distributed clients. Each client k holds a private unlabeled dataset $\mathcal { D } _ { k }$ and shares a pre-trained CLIP backbone comprising a fixed image encoder $E _ { v } ( \cdot )$ and a text encoder $E _ { t } ( \cdot )$ . To adapt CLIP to downstream tasks, we introduce learnable visual prompts $\mathbf { P } ^ { v }$ and textual prompts $\mathbf { P } _ { k } ^ { t }$ . The probability of sample x belonging to class $c ,$ denoted as $p ( y = c | x )$ , is computed via the cosine similarity:

$$
p ( y = c | x ) = \frac { \exp ( \sin ( f _ { v } ( x ) , f _ { t } ( c ) ) / \tau ) } { \sum _ { j = 1 } ^ { C } \exp ( \sin ( f _ { v } ( x ) , f _ { t } ( j ) ) / \tau ) } ,\tag{1}
$$

where $f _ { v }$ and $f _ { t }$ are ℓ -normalized visual and textual features extracted using $\mathbf { P } ^ { v }$ and $\mathbf { P } _ { k } ^ { t }$ , respectively. sim(·, ·) denotes the cosine similarity (implemented as a dot product for normalized vectors), and $\tau$ is the temperature parameter. Let $\mathbf { p } _ { i } = [ p ( y =$ $1 | x _ { i } ) , \ldots , p ( y = C | x _ { i } ) | ^ { \top } \in \mathbb { R } ^ { C }$ denote the predictive probability vector of sample $x _ { i } .$ , and $\hat { y } _ { i } = \arg \operatorname* { m a x } _ { c } p ( y = c | x _ { i } )$ be the generated pseudo-label. The goal is to collaboratively optimize these prompts without accessing ground-truth labels.

To mitigate non-IID shifts, a pioneering approach [11] utilizes the Cooperative Pseudo-Labeling (CoPL) paradigm, which dynamically updates the training set via two protocols:

• Global Budget Allocation: The server calculates a balanced budget $\tilde { U } _ { k }$ for each client to rectify class imbalance.

• Confidence-Based Selection: Based on this budget, client k constructs a filtered dataset $\tilde { \mathcal { D } } _ { k }$ by selecting the top-ranked samples with the highest confidence.

The standard optimization objective minimizes the crossentropy loss solely on this filtered high-confidence subset:

$$
\mathcal { L } _ { b a s e } = \sum _ { k = 1 } ^ { K } \mathbb { E } _ { ( \boldsymbol { x } , \hat { \boldsymbol { y } } ) \in \tilde { \mathcal { D } } _ { k } } \ell _ { c e } \big ( g ( \mathbf { P } ^ { v } , \mathbf { P } _ { k } ^ { t } ; \boldsymbol { x } ) , \hat { \boldsymbol { y } } \big ) .\tag{2}
$$

While CoPL ensures global balance by prioritizing highconfidence “head” samples $( \tilde { \mathcal { D } } _ { k } )$ , it excludes samples outside the selection budget. Among them, the ambiguous “waist” samples, which lie close to the decision boundary, are not utilized during optimization, despite exhibiting high predictive uncertainty. As a result, the model lacks explicit constraints on ambiguous predictions, which can encourage overly confident decision boundaries. To address this, we propose FedBST, which extends the base framework by incorporating an uncertainty-aware confidence penalization (UACP) term applied to these previously ignored samples. The overall framework of FedBST is illustrated in Figure 1.

## B. Uncertainty-Aware Confidence Penalization

The budget-based selection strategies discard a substantial portion of local data. Although these discarded samples are unsuitable for positive pseudo-labeling, they provide information about regions where the model exhibits high predictive uncertainty. To leverage this signal, we introduce uncertaintyaware confidence penalization (UACP). Rather than assigning these ambiguous samples to specific classes, which may introduce noise, UACP uses them to regulate model confidence and mitigate confirmation bias.

Identifying Ambiguous Candidates. We partition the local dataset $\mathcal { D } _ { k }$ according to the selection protocol. Let $\tilde { \mathcal { D } } _ { k }$ denote the selected subset of samples with high-confidence positive predictions. The remaining samples, defined as $\mathcal { R } _ { k } \ : = \ : \mathcal { D } _ { k } \ : \backslash$ $\tilde { \mathcal { D } } _ { k }$ , are retained for further consideration, as they may still provide useful regularization signals. However, $\mathcal { R } _ { k }$ inevitably contains both informative, ambiguous samples and low-quality noise. To disentangle these two cases, we utilize a dual-filter mechanism to isolate the most informative “waist” samples:

1) Symmetric Confidence Interval: We filter samples based on the confidence of their tentative predictions. For sample $x _ { i } ,$ let $\hat { y } _ { i } = \arg \operatorname* { m a x } _ { c } p ( y = c | x _ { i } )$ be the predicted class and $\kappa _ { i } = p ( y = \hat { y } _ { i } | x _ { i } )$ be the corresponding confidence score. We define a symmetric acceptance interval controlled by a threshold $\tau _ { c o n }$ . A sample is retained if $\kappa _ { i }$ lies within $[ \tau _ { c o n } , 1 - \tau _ { c o n } ] .$ . This excludes both low-confidence noise $( \kappa _ { i } ~ < ~ \tau _ { c o n } )$ and high-confidence samples $( \kappa _ { i } > 1 - \tau _ { c o n } )$ covered by the positive set.

![](images/0c7a88e6081a2043ca49874836c9848ca403aeb1f87508a851a60c47977ec716.jpg)  
Fig. 1. Overview of the proposed FedBST framework. The method consists of two phases: (Left) Local Bilateral Self-Training: Rather than using only high-confidence samples (blue), FedBST additionally leverages ambiguous “waist” samples (orange) for regularization. An Uncertainty-Aware Confidence Penalization (UACP) term suppresses the maximum predicted probability of these samples to reduce confirmation bias. (Right) Visual-Text Ensemble: At inference, global textual semantics are combined with local visual prototypes. The visual stream is restricted to locally observed classes (hollow dots indicate absent classes), while unobserved classes are handled by the textual stream, ensuring coverage of the full label space

2) Entropy Verification: To ensure meaningful uncertainty, we further require the predictive entropy to exceed a specific threshold, i.e., $\mathcal { H } ( \mathbf { p } _ { i } ) > \tau _ { e n t } .$

The final ambiguous set is defined as:

$$
S _ { a m b } = \{ x _ { i } \in \mathcal { R } _ { k } \mid \tau _ { c o n } \leq \kappa _ { i } \leq 1 - \tau _ { c o n } , \mathcal { H } ( \mathbf { p } _ { i } ) > \tau _ { e n t } \} .\tag{3}
$$

Penalization Objective. For the high-confidence set $\tilde { \mathcal { D } } _ { k }$ , we retain the standard cross-entropy loss $\mathcal { L } _ { b a s e }$ to encourage class separation. For the ambiguous set $\boldsymbol { S } _ { a m b }$ , we introduce a confidence penalization term. As ground-truth labels are unavailable, no class-specific supervision is imposed. Instead, we penalize overconfident predictions on these uncertain inputs by minimizing the maximum predicted class probability:

$$
\mathcal { L } _ { u a c p } = \frac { 1 } { \vert S _ { a m b } \vert } \sum _ { x _ { i } \in S _ { a m b } } p ( y = \hat { y } _ { i } \mid x _ { i } ) .\tag{4}
$$

Minimizing $\mathcal { L } _ { u a c p }$ reduces overly peaked predictive distribu tions on ambiguous samples. Although this objective does not explicitly compute Shannon entropy, it acts as a lightweight surrogate that discourages overconfident predictions, particularly near decision boundaries.

Total Objective. The overall local objective for client k is:

$$
\begin{array} { r } { \mathcal { L } _ { t o t a l } = \mathcal { L } _ { b a s e } + \lambda \mathcal { L } _ { u a c p } , } \end{array}\tag{5}
$$

where $\lambda$ acts as a trade-off parameter, balancing discriminative learning on high-confidence samples with explicit confidence control on ambiguous ones.

Global Aggregation. Following FedCoPL [11], we adopt partial prompt aggregation. Specifically, let $\mathbf { P } _ { k } ^ { v }$ denote the locally updated visual prompts of client k. The server aggregates these prompts globally as $\begin{array} { r } { \mathbf { P } _ { g l o b a l } ^ { v }  \sum _ { k } \frac { n _ { k } } { N } \mathbf { P } _ { k } ^ { v } } \end{array}$ , while textual prompts $\mathbf { P } _ { k } ^ { t }$ remain client-specific to preserve local semantic adaptation.

## C. Visual-Text Ensemble

To complement the training-time regularization, we employ an ensemble strategy at inference time. Although global textual prompts encode category-level semantics, they are often insufficient to capture the client-specific visual variations that arise under non-IID local distributions. Accordingly, we incorporate locally constructed visual prototypes to augment the global textual semantics, enabling effective local adaptation while mitigating overfitting.

Local Visual Prototypes. After local training, visual prototypes are constructed from the high-confidence set $\tilde { \mathcal { D } } _ { k }$ . For simplicity and robustness, we aggregate reliable samples to represent the local data distribution. Specifically, for each class c, we sum the visual features of samples predicted as class c:

$$
\mathbf { v } _ { c } = \sum _ { x _ { i } \in \tilde { \mathcal { D } } _ { k } } \mathbb { I } ( \hat { y } _ { i } = c ) \cdot f _ { v } ( x _ { i } ) .\tag{6}
$$

The prototype $\mathbf { z } _ { c }$ is obtained by $\ell _ { 2 }$ -normalizing $\mathbf { v } _ { c } .$ To handle non-IID settings, where a client typically observes only a subset of the global categories (i.e., label skew), we utilize a Class Existence Mask $m _ { c } \in \{ 0 , 1 \}$ . The mask is defined as $m _ { c } = 1 \ \mathrm { i f } \ \| \mathbf { v } _ { c } \| _ { 2 } > 0$ and 0 otherwise.

Ensembled Inference. At inference time, predictions are obtained by combining a textual stream with a visual stream. To ensure the visual stream provides reliable signals, we perform probability normalization strictly within the subset of observed classes. The two streams are defined as:

• Textual Stream: $\mathbf { p } _ { t e x t } \in \mathbb { R } ^ { C }$ , computed via Eq. (1), serves as a global fallback covering the entire label space.

• Visual Stream: We compute the similarity between the test sample and local prototypes. We apply the mask $m _ { c }$ within the Softmax formulation to restrict the probability distribution to locally existing classes:

$$
p _ { v i s u a l } ( y = c | x ) = \frac { m _ { c } \cdot \exp ( \sin ( f _ { v } ( x ) , \mathbf { z } _ { c } ) / \tau _ { v } ) } { \sum _ { j = 1 } ^ { C } m _ { j } \cdot \exp ( \sin ( f _ { v } ( x ) , \mathbf { z } _ { j } ) / \tau _ { v } ) } .\tag{7}
$$

TABLE I  
ACCURACY (%) OF DIFFERENT METHODS UNDER DIRICHLET-BASED LABEL SKEWS WITH VIT/B-32 AS THE BACKBONE. FPL [21], CPL [22], AND COPL [11] ARE ADOPTED AS BASELINE PSEUDO LABELING (PL) METHOD. BOLD AND UNDERLINE REPRESENT THE BEST AND SECOND-BEST RESULTS.
<table><tr><td rowspan="2">Method</td><td rowspan="2">PL</td><td colspan="2">DTD</td><td colspan="2">RESISC45</td><td colspan="2">CUB</td><td colspan="2">UCF101</td><td colspan="2">CIFAR10</td><td colspan="2">CIFAR100</td><td colspan="2">Average</td></tr><tr><td> $\beta = 0 . 1 0 ~ \beta = 0 . 0 5 ~ \beta = 0 . 1 0 ~ \beta = 0 . 0 5 ~ \beta = 0 . 1 0 ~ \beta = 0 . 0 5 ~ \beta = 0 . 1 0 ~ \beta = 0 . 0 5 ~ \beta = 0 . 1 0 ~ \beta = 0 . 0 5 ~ \beta = 0 . 1 0 ~ \beta = 0 . 0 5 ~ \beta = 0 . 1 0 ~ \beta = 0 . 0 5 ~ \beta = 0 . 1 0 ~ \beta = 0 . 1 0 ~ \beta = 0 . 0 0 ~ 0$ </td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CLIP</td><td></td><td>43.24</td><td>43.24</td><td>54.51</td><td>54.51</td><td>51.28</td><td>51.28</td><td>61.00</td><td>61.00</td><td>86.93</td><td>86.93</td><td>64.17</td><td>64.17</td><td>60.19</td><td>60.19</td></tr><tr><td>PromptFL [15]</td><td>FPL</td><td>45.79</td><td>44.62</td><td>59.76</td><td>58.05</td><td>47.29</td><td>46.16</td><td>64.39</td><td>62.96</td><td>87.51</td><td>85.02</td><td>63.26</td><td>62.56</td><td>61.33</td><td>59.90</td></tr><tr><td>PromptFL [15]</td><td>CPL</td><td>44.84</td><td>46.32</td><td>62.52</td><td>60.99</td><td>48.72</td><td>48.86</td><td>63.86</td><td>64.97</td><td>88.90</td><td>89.14</td><td>66.04</td><td>65.14</td><td>62.48</td><td>62.57</td></tr><tr><td>Promptprox [2]</td><td>FPL</td><td>45.15</td><td>43.88</td><td>59.36</td><td>58.59</td><td>47.04</td><td>47.01</td><td>62.91</td><td>61.27</td><td>87.13</td><td>85.62</td><td>62.77</td><td>64.12</td><td>60.73</td><td>60.08</td></tr><tr><td>Promptprox [2]</td><td>CPL</td><td>43.51</td><td>45.85</td><td>59.99</td><td>62.35</td><td>49.25</td><td>48.65</td><td>64.55</td><td>63.94</td><td>90.44</td><td>90.07</td><td>65.86</td><td>66.15</td><td>62.27</td><td>62.84</td></tr><tr><td>pFedPrompt [17]</td><td>FPL</td><td>44.56</td><td>46.19</td><td>65.95</td><td>60.52</td><td>48.48</td><td>44.42</td><td>64.37</td><td>64.18</td><td>87.45</td><td>90.72</td><td>65.08</td><td>65.83</td><td>62.65</td><td>61.98</td></tr><tr><td>pFedPrompt [17]</td><td>CPL</td><td>44.22</td><td>47.59</td><td>61.76</td><td>66.79</td><td>47.23</td><td>50.97</td><td>65.59</td><td>65.44</td><td>90.26</td><td>89.10</td><td>65.63</td><td>67.86</td><td>62.45</td><td>64.63</td></tr><tr><td>FedOPT [19]</td><td>FPL</td><td>51.97</td><td>48.71</td><td>57.76</td><td>55.62</td><td>47.99</td><td>47.05</td><td>61.83</td><td>64.56</td><td>91.32</td><td>90.73</td><td>67.51</td><td>66.50</td><td>63.06</td><td>62.20</td></tr><tr><td>FedOPT [19] FedCoPL [11]</td><td>CPL CoPL</td><td>37.43 60.89</td><td>39.33 66.37</td><td>51.10 75.76</td><td>48.47 80.26</td><td>46.61 56.09</td><td>45.07</td><td>58.93</td><td>57.12</td><td>91.08</td><td>91.36</td><td>57.87</td><td>59.81</td><td>57.17</td><td>56.86 74.06</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>54.80</td><td>73.20</td><td>74.97</td><td>95.38</td><td>95.11</td><td>73.59</td><td>72.84</td><td>72.49</td><td></td></tr><tr><td>FedBST</td><td>UACP</td><td>61.46</td><td>67.77</td><td>77.88</td><td>81.16</td><td>57.25</td><td>57.08</td><td>72.58</td><td>75.04</td><td>95.47</td><td>95.46</td><td>75.01</td><td>73.92</td><td>73.28</td><td>75.07</td></tr></table>

TABLE II

ACCURACY (%) OF DIFFERENT METHODS UNDER QUANTITY-BASED LABEL SKEWS WITH VIT/B-32 AS THE BACKBONE. FPL [21], CPL [22], AND COPL [11] ARE ADOPTED AS BASELINE PSEUDO LABELING (PL) METHOD. BOLD AND UNDERLINE REPRESENT THE BEST AND SECOND-BEST RESULTS.
<table><tr><td>Method</td><td>PL</td><td>DTD</td><td>RESISC45</td><td>CUB</td><td>UCF101</td><td>CIFAR10</td><td>CIFAR100</td><td>Average</td></tr><tr><td>CLIP</td><td>=</td><td>43.24</td><td>54.51</td><td>51.28</td><td>61.00</td><td>86.93</td><td>64.17</td><td>60.19</td></tr><tr><td>PromptFL [15]</td><td>FPL</td><td>43.53</td><td>57.53</td><td>46.66</td><td>62.35</td><td>88.29</td><td>63.94</td><td>60.38</td></tr><tr><td>PromptFL [15]</td><td>CPL</td><td>43.82</td><td>61.11</td><td>47.31</td><td>64.23</td><td>88.70</td><td>67.23</td><td>62.07</td></tr><tr><td>Promptprox [2]</td><td>FPL</td><td>44.89</td><td>56.71</td><td>48.74</td><td>61.80</td><td>85.69</td><td>64.01</td><td>60.31</td></tr><tr><td>Promptprox [2]</td><td>CPL</td><td>45.53</td><td>60.77</td><td>47.61</td><td>63.86</td><td>89.21</td><td>66.49</td><td>62.25</td></tr><tr><td>pFedPrompt [17]</td><td>FPL</td><td>45.09</td><td>60.46</td><td>47.71</td><td>65.39</td><td>87.54</td><td>62.82</td><td>61.50</td></tr><tr><td>pFedPrompt [17]</td><td>CPL</td><td>45.14</td><td>66.68</td><td>50.45</td><td>65.57</td><td>87.98</td><td>65.45</td><td>63.55</td></tr><tr><td>FedOPT [19]</td><td>FPL</td><td>47.65</td><td>56.04</td><td>46.61</td><td>62.95</td><td>90.74</td><td>66.76</td><td>61.79</td></tr><tr><td>FedOPT [19]</td><td>CPL</td><td>40.24</td><td>47.09</td><td>44.00</td><td>57.24</td><td>89.21</td><td>57.98</td><td>55.96</td></tr><tr><td>FedCoPL [11]</td><td>CoPL</td><td>56.18</td><td>81.06</td><td>56.31</td><td>72.03</td><td>94.86</td><td>73.39</td><td>72.31</td></tr><tr><td>FedBST</td><td>UACP</td><td>57.63</td><td>82.04</td><td>56.79</td><td>73.01</td><td>95.24</td><td>74.85</td><td>73.26</td></tr></table>

Implementation Details. We use a frozen CLIP (ViT-B/32 by default). Prompts are optimized via SGD (lr=0.1, momentum=0.9) with cosine decay. We report the mean accuracy of three trials. Default hyperparameters are set as $\tau _ { \mathrm { c o n } } = 0 . 2 .$ $\tau _ { \mathrm { e n t } } = 0 . 5 , \lambda = 0 . 0 0 1$ , and $\alpha = 0 . 8$ . Other settings are the same as FedCoPL [11].

This formulation ensures that for missing classes $( m _ { c } = 0 )$ the visual probability is zero, while for existing classes, the probabilities sum to 1 (i.e., locally normalized).

The final prediction is obtained by a linear combination of the two streams:

$$
\mathbf { p } _ { f i n a l } = \alpha \cdot \mathbf { p } _ { t e x t } + ( 1 - \alpha ) \cdot \mathbf { p } _ { v i s u a l } .\tag{8}
$$

Through this ensemble, predictions for locally observed classes are refined by specific visual prototypes, while predictions for missing classes implicitly rely on the global textual knowledge (scaled by α), ensuring the model remains functional across the full label space.

## IV. EXPERIMENTS

## A. Experimental Setup

Datasets and Baselines. We conduct evaluations on six benchmarks: DTD [23], RESISC45 [24], UCF101 [25], CUB [26], CIFAR-10, and CIFAR-100 [27]. Following [11], we simulate two non-IID scenarios: (1) Dirichlet-based skew, where class distributions are sampled from a Dirichlet distribution (β); and (2) Quantity-based skew, where samples are grouped by class and distributed via heterogeneous shards. We compare with state-of-the-art methods including Fed-CoPL [11], FedOPT [19], pFedPrompt [17], PromptProx [2], and PromptFL [15]. Supervised baselines are combined with pseudo-labeling strategies FPL [21] or CPL [22] for unsupervised adaptation.

## B. Experimental Results

Results under Dirichlet-based label skew with various datasets. Table I reports the accuracy under Dirichlet-based skew with $\beta \in \{ 0 . 1 , 0 . 0 5 \}$ . FedBST consistently achieves the best performance across all settings. Notably, on fine-grained datasets like CUB, baselines relying on FPL or CPL often succumb to noisy pseudo-labels, leading to error accumulation and occasionally underperforming zero-shot CLIP. While Fed-CoPL provides a strong unsupervised federated classification baseline, FedBST yields further consistent improvements. This indicates that effective utilization of ambiguous samples and the visual-text ensemble is critical for robust UFC with CLIP. Results under quantity-based label skew with various datasets. Table II presents results where clients observe only a limited subset of classes $( s ~ \ : = ~ C \times 0 . 2 )$ This severe class scarcity significantly challenges the generation of reliable pseudo-labels, causing several baselines to underperform zeroshot CLIP. In contrast, FedBST consistently achieves the best results, outperforming the strongest baseline FedCoPL by a clear margin (Average +0.95%). This demonstrates that our bilateral self-training strategy effectively mines useful signals

![](images/631d3a3015513110c6b04e12436586d791d0d2f14261e74c6bf3062526ba24bf.jpg)  
Fig. 2. Results of experiments with various client numbers and different client joining rates under Dirichlet-based label skews $( \beta = 0 . 1 )$

TABLE III  
EXPERIMENTAL RESULTS (%) USING CLIP VIT-B/16 AS BACKBONE UNDER DIRICHLET-BASED LABEL SKEW $( \beta = 0 . 1 )$
<table><tr><td>Method</td><td>PL</td><td>DTD</td><td>RESISC45</td><td>CUB</td><td>UCF101</td></tr><tr><td>CLIP</td><td>-</td><td>42.87</td><td>56.61</td><td>55.16</td><td>65.13</td></tr><tr><td>PromptFL [15]</td><td>FPL</td><td>44.36</td><td>61.36</td><td>51.95</td><td>64.60</td></tr><tr><td>PromptProx [2]</td><td>FPL</td><td>47.12</td><td>61.01</td><td>49.44</td><td>65.31</td></tr><tr><td>pFedPrompt [17]</td><td>FPL</td><td>47.65</td><td>63.18</td><td>50.19</td><td>65.19</td></tr><tr><td>FedOPT [19]</td><td>FPL</td><td>46.52</td><td>58.94</td><td>49.67</td><td>65.68</td></tr><tr><td>FedCoPL [11]</td><td>CoPL</td><td>56.83</td><td>77.50</td><td>59.78</td><td>78.29</td></tr><tr><td>FedBST</td><td>UACP</td><td>58.72</td><td>79.81</td><td>61.40</td><td>78.93</td></tr></table>

TABLE IV

EXPERIMENTAL RESULTS (%) USING CLIP RN50 AS BASE MODEL UNDER DIRICHLET-BASED LABEL SKEWS $( \beta = 0 . 1 )$ ACROSS FOUR DATASETS.
<table><tr><td>Method</td><td>PL</td><td>DTD</td><td>RESISC45</td><td>UCF101</td><td>CIFAR10</td></tr><tr><td>CLIP</td><td>一</td><td>41.62</td><td>52.12</td><td>65.13</td><td>74.80</td></tr><tr><td>PromptFL [15]</td><td>FPL</td><td>42.36</td><td>56.16</td><td>62.90</td><td>78.01</td></tr><tr><td>PromptProx [2]</td><td>FPL</td><td>43.12</td><td>56.54</td><td>64.37</td><td>78.77</td></tr><tr><td>pFedPrompt [17]</td><td>FPL</td><td>43.65</td><td>57.13</td><td>64.97</td><td>79.69</td></tr><tr><td>FedOPT [19]</td><td>FPL</td><td>45.52</td><td>58.94</td><td>65.54</td><td>80.13</td></tr><tr><td>FedCoPL [11]</td><td>CoPL</td><td>51.83</td><td>70.78</td><td>76.48</td><td>84.85</td></tr><tr><td>FedBST</td><td>UACP</td><td>53.40</td><td>72.55</td><td>77.01</td><td>85.21</td></tr></table>

from “waist” samples even when high-confidence data is scarce, ensuring robustness under extreme non-IID conditions.

## C. Analysis

Results under different client numbers. Fig. 2 (Top) illustrates the performance under varying federation scales. As client numbers increase, data heterogeneity intensifies, causing performance degradation across all methods due to the difficulty of reliable pseudo-labeling. However, FedBST exhibits the slowest degradation rate and maintains the highest accuracy. This demonstrates its superior scalability, attributed to the UACP module, which effectively prevents noise accumulation even when local data becomes sparse.

Impact of client joining rates. Fig. 2 (Bottom) reports results under varying participation rates. Lower participation reduces information aggregation and increases local update variance, causing performance drops across all methods. Baselines are particularly sensitive to this scarcity, showing substantial degradation on challenging datasets. In contrast, FedBST consistently outperforms competitors, demonstrating that our visual-text ensemble strategy ensures stability even when active clients are sparse.

TABLE V  
PERFORMANCE (%) OF FEDBST UNDER DIFFERENT VALUES OF HYPERPARAMETER $\tau _ { c o n }$ AND λ WITH DIRICHLET-BASED LABEL SKEWS.
<table><tr><td> $\lambda ( \times e ^ { - 3 } )$ </td><td>0.4</td><td>0.6</td><td>0.8</td><td>1.0</td><td>1.2</td><td>1.4</td><td>1.6</td></tr><tr><td>DTD</td><td>60.24</td><td>60.10</td><td>60.71</td><td>61.46</td><td>60.73</td><td>60.57</td><td>60.79</td></tr><tr><td>RESICS45</td><td>78.23</td><td>78.01</td><td>78.20</td><td>77.88</td><td>77.60</td><td>77.91</td><td>78.05</td></tr><tr><td>α</td><td>0.60</td><td>0.65</td><td>0.70</td><td>0.75</td><td>0.80</td><td>0.85</td><td>0.90</td></tr><tr><td>DTD</td><td>61.31</td><td>61.26</td><td>61.15</td><td>61.31</td><td>61.46</td><td>61.36</td><td>61.19</td></tr><tr><td>RESICS45</td><td>78.08</td><td>77.97</td><td>77.91</td><td>77.93</td><td>77.88</td><td>77.75</td><td>77.71</td></tr></table>

TABLE VI

PERFORMANCE (%) OF FEDBST UNDER DIFFERENT VALUES OF HYPERPARAMETER $\tau _ { c o n }$ AND $\tau _ { e n t } .$
<table><tr><td>Tcon</td><td>0.10</td><td>0.15</td><td>0.20</td><td>0.25</td><td>0.30</td></tr><tr><td>DTD</td><td>61.21</td><td>61.45</td><td>61.46</td><td>61.62</td><td>61.60</td></tr><tr><td>RESICS45</td><td>78.56</td><td>78.10</td><td>77.88</td><td>77.39</td><td>77.30</td></tr><tr><td> $\tau _ { e n t }$ </td><td>0.40</td><td>0.45</td><td>0.50</td><td>0.55</td><td>0.60</td></tr><tr><td>DTD</td><td>61.39</td><td>61.50</td><td>61.46</td><td>61.55</td><td>61.21</td></tr><tr><td>RESICS45</td><td>78.33</td><td>78.04</td><td>77.88</td><td>77.39</td><td>76.87</td></tr></table>

Results under different image encoder backbones. Tables III and IV present results under Dirichlet-based label skews using ViT-B/16 and RN50 image encoders. FedBST consistently outperforms zero-shot CLIP and federated baselines across both Transformer-based and CNN-based architectures. This confirms that our framework is architectureagnostic, maintaining superior performance and practical ap plicability without requiring backbone-specific modifications. Parameter Sensitivity.

Tables V and VI report performance under varying λ, α, $\tau _ { \mathrm { c o n } } .$ , and $\tau _ { \mathrm { e n t } }$ . FedBST exhibits stable performance across a broad range of values, with accuracy fluctuating minimally on DTD and RESISC45 around the default settings. This resilience suggests that FedBST is practical for real-world scenarios where extensive hyperparameter tuning is infeasible. Effectiveness of each component. Table VII presents an ablation study on Uncertainty-Aware Confidence Penalization (UACP) and Visual-Text Ensemble (VTE) under Dirichletbased label skew $( \beta ~ = ~ 0 . 1 )$ . Each component provides individual gains, yet their combination performs best. This confirms their synergy: UACP regularizes training, while VTE calibrates inference.

TABLE VII  
ABLATION RESULTS (%) COMPARING THE EFFECTS OF UNCERTAINTY-AWARE CONFIDENCE PENALIZATION (UACP) AND VISUAL-TEXT ENSEMBLE (VTE).
<table><tr><td>UACP</td><td>VTE</td><td>DTD</td><td>RESISC45</td><td>CUB</td><td>CIFAR100</td></tr><tr><td>=</td><td></td><td>60.89</td><td>75.76</td><td>56.09</td><td>73.59</td></tr><tr><td>√</td><td>=</td><td>61.22</td><td>77.34</td><td>56.89</td><td>74.87</td></tr><tr><td>=</td><td>√</td><td>61.10</td><td>76.77</td><td>56.54</td><td>74.41</td></tr><tr><td>√</td><td>√</td><td>61.46</td><td>77.88</td><td>57.25</td><td>75.01</td></tr></table>

## V. CONCLUSION

This work introduces FedBST, a bilateral self-training framework for unsupervised federated classification that is explicitly designed to operate under severe non-IID conditions. Rather than discarding ambiguous data, FedBST makes systematic use of both high-confidence and uncertain samples, allowing a more complete exploitation of local data. Uncertain predictions are incorporated as regularization signals via uncertainty-aware confidence penalization, while inference is further strengthened through a visual-text ensemble, which together help alleviate confirmation bias and improve the alignment between global semantics and client-specific distributions. Experimental results across multiple benchmarks indicate that these design choices lead to stable performance improvements, with the largest gains observed in settings characterized by strong label skew and limited data.

## ACKNOWLEDGMENT

This work was supported by the National Natural Science Foundation of China (NSFC) (Grant 62376265). It was also supported by the National Natural Science Foundation of China under Grants (62276256, U2441251).

## REFERENCES

[1] Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas, “Communication-efficient learning of deep networks from decentralized data,” in Proc. AISTATS, 2017, pp. 1273– 1282.

[2] Tian Li, Anit Kumar Sahu, Manzil Zaheer, Maziar Sanjabi, Ameet Talwalkar, and Virginia Smith, “Federated optimization in heterogeneous networks,” in Proc. MLSys, 2020, pp. 429–450.

[3] Debora Caldarola, Pietro Cagnasso, Barbara Caputo, and Marco Ciccone, “Beyond local sharpness: Communication-efficient global sharpness-aware minimization for federated learning,” in Proc. CVPR, 2025, pp. 25187–25197.

[4] Xinting Liao, Weiming Liu, Jiaming Qian, Pengyang Zhou, Jiahe Xu, Wenjie Wang, Chaochao Chen, Xiaolin Zheng, and Tat-Seng Chua, “Focoop: Enhancing out-of-distribution robustness in federated prompt learning for vision-language models,” in Proc. ICML, 2025, pp. 37528– 37554.

[5] Sungwon Han, Sungwon Park, Fangzhao Wu, Sundong Kim, Chuhan Wu, Xing Xie, and Meeyoung Cha, “Fedx: Unsupervised federated learning with cross knowledge distillation,” in Proc. ECCV, 2022, pp. 691–707.

[6] Hansol Kim, Youngjun Kwak, Minyoung Jung, Jinho Shin, Youngsung Kim, and Changick Kim, “Protofl: Unsupervised federated learning via prototypical distillation,” in Proc. ICCV, 2023, pp. 6470–6479.

[7] Hao Yan and Yuhong Guo, “Lightweight unsupervised federated learning with pretrained vision language model,” in International Workshop on Trustworthy Federated Learning, 2024, pp. 148–163.

[8] Xinting Liao, Weiming Liu, Chaochao Chen, Pengyang Zhou, Fengyuan Yu, Huabin Zhu, Binhui Yao, Tao Wang, Xiaolin Zheng, and Yanchao Tan, “Rethinking the representation in federated unsupervised learning with non-iid data,” in Proc. CVPR, 2024, pp. 22841–22850.

[9] Farhad Pourpanah, Mahdiyar Molahasani, Milad Soltany, Michael Greenspan, and Ali Etemad, “Federated unsupervised domain generalization using global and local alignment of gradients,” in Proc. AAAI, 2025, pp. 19948–19958.

[10] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al., “Learning transferable visual models from natural language supervision,” in Proc. ICML, 2021, pp. 8748–8763.

[11] Kuangpu Guo, Lijun Sheng, Yongcan Yu, Jian Liang, Zilei Wang, and Ran He, “Cooperative pseudo labeling for unsupervised federated classification,” in Proc. ICCV, 2025, pp. 3326–3336.

[12] Weiming Zhuang, Xin Gan, Yonggang Wen, Shuai Zhang, and Shuai Yi, “Collaborative unsupervised visual representation learning from decentralized data,” in Proc. ICCV, 2021, pp. 4912–4921.

[13] Weiming Zhuang, Yonggang Wen, and Shuai Zhang, “Divergence-aware federated self-supervised learning,” in Proc. ICLR, 2022, pp. 1930–1948.

[14] Ekdeep Lubana, Chi Ian Tang, Fahim Kawsar, Robert Dick, and Akhil Mathur, “Orchestra: Unsupervised federated learning via globally consistent clustering,” in Proc. ICML, 2022, pp. 14461–14484.

[15] Tao Guo, Song Guo, Junxiao Wang, Xueyang Tang, and Wenchao Xu, “Promptfl: Let federated participants cooperatively learn prompts instead of models–federated learning in age of foundation model,” IEEE Transactions on Mobile Computing, vol. 23, no. 5, pp. 5179–5194, 2023.

[16] Wang Lu, HU Xixu, Jindong Wang, and Xing Xie, “Fedclip: Fast generalization and personalization for clip in federated learning,” in Proc. ICLR 2023 Workshop on Trustworthy and Reliable Large-Scale Machine Learning Models, 2023.

[17] Tao Guo, Song Guo, and Junxiao Wang, “Pfedprompt: Learning personalized prompt for vision-language models in federated learning,” in Proc. WWW, 2023, pp. 1364–1374.

[18] Fu-En Yang, Chien-Yi Wang, and Yu-Chiang Frank Wang, “Efficient model personalization in federated learning via client-specific prompt generation,” in Proc. ICCV, 2023, pp. 19159–19168.

[19] Hongxia Li, Wei Huang, Jingya Wang, and Ye Shi, “Global and local prompts cooperation via optimal transport for federated learning,” in Proc. CVPR, 2024, pp. 12151–12161.

[20] Yuguang Zhang, Kuangpu Guo, Yunbo Wang, and Jian Liang, “Personalized federated learning via dual-prompt optimization and cross fusion,” in Proc. PRCV, 2026.

[21] Cristina Menghini, Andrew Delworth, and Stephen Bach, “Enhancing clip with clip: Exploring pseudolabeling for limited-label prompt tuning,” in Proc. NeurIPS, 2023, pp. 60984–61007.

[22] Jiahan Zhang, Qi Wei, Feng Liu, and Lei Feng, “Candidate pseudolabel learning: Enhancing vision-language models by prompt tuning with unlabeled data,” in Proc. ICML, 2024, pp. 60004–60020.

[23] Mircea Cimpoi, Subhransu Maji, Iasonas Kokkinos, Sammy Mohamed, and Andrea Vedaldi, “Describing textures in the wild,” in Proc. CVPR, 2014, pp. 3606–3613.

[24] Gong Cheng, Junwei Han, and Xiaoqiang Lu, “Remote sensing image scene classification: Benchmark and state of the art,” Proceedings of the IEEE, vol. 105, no. 10, pp. 1865–1883, 2017.

[25] Khurram Soomro, Amir Roshan Zamir, and Mubarak Shah, “Ucf101: A dataset of 101 human actions classes from videos in the wild,” arXiv preprint arXiv:1212.0402, 2012.

[26] Catherine Wah, Steve Branson, Peter Welinder, Pietro Perona, and Serge Belongie, “The caltech-ucsd birds-200-2011 dataset,” Technical Report CNS-TR-2011-001, 2011.

[27] Alex Krizhevsky, Geoffrey Hinton, et al., “Learning multiple layers of features from tiny images,” 2009.