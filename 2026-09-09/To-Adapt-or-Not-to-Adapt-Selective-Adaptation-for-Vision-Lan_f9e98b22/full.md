# To Adapt or Not to Adapt? Selective Adaptation for Vision-Language Models

Siru Jiang1,2†0, Yuwei Liang2,3†, Jian Liang2,3\*0, Ran He2,30, and Tieniu Tan1,2,4

1 School of Advanced Interdisciplinary Sciences, University of Chinese Academy of Sciences, China

2 NLPR & MAIS, Institute of Automation, Chinese Academy of Sciences, China 3 School of Artificial Intelligence, University of Chinese Academy of Sciences, China 4 Nanjing University, China {sirujiang324, liangjian92}@gmail.com

Abstract. Test-time adaptation (TTA) has emerged as a prominent strategy for adapting vision-language models to distribution shifts during inference. We conduct a per-sample analysis of model predictions before and after adaptation, and observe two failure modes in existing TTA methods that echo previous work. Adaptations are frequently negligible, yielding no change in the model's predictions, and more severely, they can be detrimental by flipping previously correct predictions to incorrect ones. This naturally raises a question: Can we identify and skip such negligible or harmful adaptations? In this work, we introduce a new problem of selective adaptation, which aims to determine whether a given test sample should undergo adaptation or be skipped. To this end, we propose Cross-Augmentation Similarity (CAS), a simple baseline that performs adaptation only when predictions across augmented views exhibit low similarity. Notably, CAS not only preserves but in some cases improves overall accuracy, even when skipping nearly 85% of the adaptation process. We hope other researchers will explore this new direction and surpass the performance of our baseline. Our code is available at https://github.com/sirujiang/selective-adaptation.

Keywords: Test-time Adaptation ·Selective Adaptation · Vision-Language Models

## 1 Introduction

Vision-language models (VLMs), such as CLIP [46], ALIGN [28], Flamingo [2], and LLaVA [37], are pretrained on large-scale image-text pairs and have demonstrated strong generalization across diverse vision tasks [61, 74, 77, 78]. Among them, CLIP [46] aligns visual and textual representations in a shared embedding space, enabling impressive zero-shot performance. However, VLMs remain sensitive to distribution shifts and often suffer from performance degradation when the test distribution differs from that of pretraining [31, 36, 48, 69].

![](images/ad11e995281d5c834f34129ca2c325cddd0a82190694c430a362356560f8dd5b.jpg)

![](images/3fb4b734ca9f75e99f7cf13a12f013ae4862edbe31455b6a937150ad7db51205.jpg)  
(b) Accuracy versus Skip Ratio  
Fig. 1: (a) Given a test image $x ,$ a score α(x) is computed. Samples with low scores undergo adaptation, while those with high scores are skipped, with the zero-shot prediction used instead. (b) The proposed CAS maintains or slightly improves accuracy across a wide range of skip ratios (the proportion of skipped adaptations) on ImageNet.

In recent years, test-time adaptation (TTA) has emerged as an effective paradigm for mitigating distribution shifts by adapting VLMs with unlabeled test data. In the context of image classification, existing TTA methods mainly focus on improving prediction accuracy [6, 10, 53] or enhancing model calibration [1,50,68]. Despite their effectiveness, they implicitly assume that adaptation is beneficial for all test samples. To better understand this issue, we conduct a preliminary per-sample analysis of the adaptation process under the classic TPT framework [53].

Consistent with observations in prior work [10], we find that a large proportion of predictions remain unchanged before and after adaptation, leading to unnecessary computational overhead. More critically, some originally correct predictions flip to incorrect after adaptation, leading to performance degradation. Notably, these negligible or even harmful adaptations account for more than 90% of all adaptation processes. Motivated by these observations, we propose to identify and skip ineffective adaptations, thereby improving efficiency while preserving accuracy. Unlike previous TTA approaches that enhance efficiency through parameter-free retrieval [7,29] or lightweight parameter optimization [6, 27], our method improves efficiency at the per-sample level by adapting only when necessary.

In this work, we introduce a new problem, termed selective adaptation, which formulates a binary detection task to identify whether the adaptation for a given test sample should be performed or skipped. An illustration of this problem is shown in Fig. 1 (a). Similar to out-of-distribution (OOD) detection [17,21,24,38, 39, 41,66], it relies on a scoring function to identify ineffective adaptations. Specifically, samples with higher scores tend to correspond to negligible or harmful cases and use zero-shot predictions directly, while those with lower scores continue to undergo adaptation. Generally, the goal of selective adaptation is to skip as many ineffective adaptations as possible while maintaining or even improving overall accuracy. To evaluate this problem, we adopt the standard Area Under the ROC Curve (AUC) [8, 11] for detection quality, and introduce Accuracy Expectation with a Triangular Prior (AEP) to measure expected accuracy under different skip ratios.

Test-time augmentation [10, 34, 49, 53] has been used in many TTA methods [53, 75] to generate multiple views for an input sample. We argue that the prediction similarity between the test sample and its augmented views may be closely correlated with adaptation effectiveness. This motivates our simple baseline Cross-Augmentation Similarity (CAS), which computes a score based on the prediction similarity across multiple augmented views. We compare CAS against random skipping and established OOD detection methods [38,41] under representative TTA frameworks [6, 10, 52, 53]. Overall, CAS achieves an AUC of around 90%. More importantly, it preserves and even improves the accuracy of full adaptation while skipping 85% of adaptations on ImageNet, its variants, and multiple fine-grained benchmarks. The result of ImageNet is shown in Fig. 1 (b). Beyond classification accuracy, CAS also maintains calibration performance in calibration-oriented TTA methods [50,68]. Our contributions are summarized as follows:

• We introduce selective adaptation, an underexplored direction to improve TTA efficiency by detecting whether a test sample can benefit from adaptation.

• We provide Cross-Augmentation Similarity (CAS), a simple baseline based on prediction similarity across test-time augmented views.

• Extensive experiments validate that CAS maintains and even improves TTA performance, providing a baseline for future research to advance.

## 2 Related Work

Test-time adaptation (TTA). TTA aims to mitigate performance degradation caused by distribution shifts by adapting models to unlabeled test data [36, 51]. TTA approaches can be broadly categorized into two paradigms based on how they process test data. Online TTA [57,59,63,70,71,79] processes streaming data and updates model parameters by leveraging historical knowledge from previous test samples. In contrast, episodic TTA, such as MEMO [75] and TTT [56], treats each test sample independently, making adaptation more challenging. Throughout this work, we focus exclusively on the episodic paradigm.

With the rise of VLMs [2, 28, 46], increasing attention has been devoted to applying TTA [50, 52, 53] to CLIP [46]. A line of work explores training-based TTA methods [13, 34, 53], which adapt models at test time by optimizing a subset of parameters using unlabeled test samples. The pioneering work TPT [53] adapts the model by optimizing learnable prompts through entropy minimization with confidence selection. Alternatively, another line of work explores trainingfree paradigms, such as ZERO [10], MTA [72], and TPS [55], which enable fast test-time adaptation without requiring gradient updates. In addition to accuracy improvement, recent studies have also explored other aspects of performance, including calibration [1, 50, 68] and adversarial robustness [52, 60, 64].

Efficiency in TTA. In addition to improving adaptation performance, several studies [6, 29, 44, 80] focus on enhancing efficiency. In episodic TTA, most existing work [6, 27] focuses on making the optimization process more efficient. TTL [27] improves efficiency by optimizing low-rank adapters and STS [6] adapts only a small number of parameters. Instead of refining the optimization algorithm, we propose selective skipping, which identifies and skips ineffective adaptations to improve efficiency without sacrificing performance. Notably, a line of work in online TTA [29, 44, 76, 80] has also explored efficiency improvements. For example, EATA [44 improves efficiency by selecting informative samples for adaptation based on entropy, while other methods reduce computational overhead via key–value cache retrieval [7,29, 76,80]. These approaches differ fundamentally from our selective adaptation approach.

Out-of-distribution (OOD) detection and selective classification. OOD detection [17,21, 41, 65] aims to identify test samples that differ from the training distribution to ensure model reliability. Early work initially introduced MSP [24] for detecting misclassified or OOD samples, followed by Energy [38], MCM [41] and Doctor [17]. We leverage the scoring functions provided by these methods as a comparison strategy and employ AUC to evaluate the effectiveness of our selective adaptation baseline. Selective classification [5, 14, 15, 35], also known as classification with a rejection option, is a machine learning framework that allows a model to abstain from making a prediction when it is uncertain. While selective classification typically abstains from making predictions after inference [16], selective adaptation introduced in this paper instead focuses on rejecting samples before performing adaptation.

## 3 Method

## 3.1 Preliminaries

CLIP [46] is a widely used VLM due to its strong zero-shot generalization capability. It consists of two parts, a visual encoder $f _ { v } ( \cdot )$ and a text encoder $f _ { t } ( \cdot )$ . For a K-class classification task with a label space $\mathcal { Y } = \{ y _ { 1 } , y _ { 2 } , \dots , y _ { K } \}$ , let x denote an input image and $y \in \mathcal { V }$ denote its ground-truth label. The visual encoder extracts visual features from the input image $v = f _ { v } ( x )$ . For the text encoder, each class label $y _ { k } \in \mathcal { V }$ is converted into a textual prompt $c _ { k } \ { \mathrm { ( e . g . } }$ , using a template such as $^ { 6 6 } \mathrm { a }$ photo of a [class]") and encoded into a textual feature $t _ { k } = f _ { t } ( c _ { k } )$ The prediction probability is then computed as

$$
p ^ { k } ( x ) = p ( y = k \mid x ) = \frac { \exp { ( \cos ( v , t _ { k } ) / \tau ) } } { \sum _ { j = 1 } ^ { K } \exp { ( \cos ( v , t _ { j } ) / \tau ) } } ,\tag{1}
$$

where τ is a temperature parameter and cos $( \cdot , \cdot )$ denotes cosine similarity. Given a test image $x , y _ { \mathrm { z s } } ( x ) =$ arg maxk $p _ { \mathrm { z s } } ^ { k } ( x )$ and $y _ { \mathrm { a d a p t } } ( x ) =$ arg maxk $p _ { \mathrm { a d a p t } } ^ { k } ( x )$ 2 where $p _ { \mathrm { z s } } ( x )$ and $p _ { \mathrm { a d a p t } } ( x )$ denote the zero-shot and adapted probability vectors, and $y _ { \mathrm { z s } } ( x )$ and $y _ { \mathrm { a d a p t } } ( x )$ are their corresponding predicted labels.

![](images/59cca166f075e09e756a39dfd8dada6bf4d5390bbe40ff09fe46c4f9c7076f2e.jpg)  
Fig. 2: (a) We categorize the adaptation process into four cases (A-D). Checkmarks and crosses indicate prediction correctness, with zero-shot prediction shown above and TTA prediction shown below. The arrow denotes the adaptation process. (b) Proportion of each case on ImageNet and EuroSAT under TPT [53] framework. The percentage of the beneficial-only case is highlighted.

## 3.2 Problem Formulation: Selective Adaptation

Motivation. Consistent with ZERO [10], we find that a large proportion of adaptations are negligible or even harmful. As illustrated in Fig. 2 (a), the outcomes of adaptation can be categorized into four cases based on prediction changes between the zero-shot and adapted models. Based on their impact on performance, we further group these cases into three types: negligible adaptation, harmful adaptation, and beneficial adaptation.

• Harmful adaptation. The zero-shot prediction is correct, but becomes incorrect after adaptation $( y _ { \mathrm { z s } } ( x ) = y , y _ { \mathrm { a d a p t } } ( x ) \neq y )$ . These adaptations will reduce the overall performance.

• Negligible adaptation. The zero-shot prediction remains unchanged after adaptation $( y _ { \mathrm { z s } } ( x ) = y , y _ { \mathrm { a d a p t } } ( x ) = y ) o r ~ ( y _ { \mathrm { z s } } ( x ) \neq y , y _ { \mathrm { a d a p t } } ( x ) \neq y )$ Adapting these adaptations incurs unnecessary computational overhead.

• Beneficial adaptation. The zero-shot prediction is incorrect but is adapted to correct successfully $( y _ { \mathrm { z s } } ( x ) \neq y , \ y _ { \mathrm { a d a p t } } ( x ) = y )$ . These adaptations are the only ones to improve the performance.

Notably, negligible and harmful adaptations dominate the test set in many datasets. Fig. 2 (b) further shows that these cases account for over 90% of test samples on datasets such as ImageNet and EuroSAT, highlighting the importance of identifying and skipping ineffective adaptations.

Formulation. We formulate the selective adaptation problem as a binary detection task. Given a test sample x, we decide whether to perform adaptation or skip it. Following the paradigm used in OOD detection, we define a decision function $G _ { \gamma } ( x )$ based on a scoring function $\alpha ( x )$ and a threshold $\gamma \colon$

$$
G _ { \gamma } ( x ) = { \left\{ \begin{array} { l l } { { \mathrm { N o t ~ A d a p t ~ } } } & { { \mathrm { i f ~ } } \alpha ( x ) \geq \gamma } \\ { { \mathrm { A d a p t ~ } } } & { { \mathrm { o t h e r w i s e } } } \end{array} \right. }\tag{2}
$$

where $\gamma$ is designed to control the trade-off between performance and efficiency, and $\alpha ( x )$ is defined based on prediction similarity across augmented views. A larger α(x) indicates a higher likelihood of skipping adaptation for the input image x. Given a fixed threshold $\gamma ,$ a unique skip ratio s is determined, where s denotes the proportion of test samples for which adaptation is skipped.

Evaluation metrics. To evaluate the selective skipping strategy comprehensively, we introduce the following metrics:

• Area under the ROC curve (AUC). We formulate the identification of ineffective adaptations as a binary detection task. AUC [8, 11] measures how well a scoring function distinguishes ineffective adaptations from beneficial ones. An AUC of 0.5 corresponds to random guessing, while higher values indicate stronger discriminative capability.

• Accuracy expectation with triangular prior (AEP). Since the optimal skip ratio $s ^ { * }$ may vary across deployment scenarios, we propose a metric to evaluate overall performance by computing the expected accuracy under a prior distribution of s. Specifically, we adopt a triangular prior with probability density function $f ( s ) = 2 ( 1 - s )$ for $s \in [ 0 , 1 ]$ . As s increases, efficiency gains become more significant, and slight accuracy degradation becomes more acceptable. Therefore, performance is assigned a lower weight at larger skip ratios. The metric is defined as

$$
{ \mathrm { A E P } } = \int _ { 0 } ^ { 1 } \operatorname { a c c } ( s ) \cdot 2 ( 1 - s ) d s ,\tag{3}
$$

where $\operatorname { a c c } ( s )$ denotes the model accuracy under a skip ratio s. By integrating the skip-accuracy curve, AEP provides a comprehensive evaluation of performance under different computational budgets. Notably, acc(s) can be replaced with other performance metrics (e.g., ECE [18]) to evaluate different aspects of model behavior.

## 3.3 Cross-Augmentation Similarity as a Simple Baseline

Test-time augmentation [30,49], derived from data augmentation techniques [67], applies random transformations to test samples during inference. It has been widely adopted in TTA methods such as MEMO [75] and TPT [53]. Specifically, TPT [53] generates $( N - 1 )$ augmented views for a test image x using AugMix [25]. Let $\{ \mathcal { A } _ { i } ( x ) \} _ { i = 0 } ^ { N - 1 }$ denote the augmented views, where $\boldsymbol { \mathcal { A } } _ { 0 } ( \boldsymbol { x } )$ is the original image. A cutoff percentile $\rho \in [ 0 , 1 ]$ is applied over the N augmented views to select high-confidence samples. Views whose prediction entropy is lower than the threshold $\beta$ are retained to form the high-confidence set S:

$$
S = \{ i \mid H ( p _ { \mathrm { z s } } ( A _ { i } ( x ) ) ) \leq \beta , i \in [ 0 , N - 1 ] \} ,\tag{4}
$$

![](images/a4630f657c4502dc479648ffa2348ed4b18f394a56d94e0f4c30a761348bdd15.jpg)  
Fig. 3: Pipeline of the proposed selective adaptation framework. Augmented views and class prompts are encoded to obtain zero-shot predictions. CAS measures cross-view prediction agreement. Samples with low CAS scores undergo adaptation, while those with high scores directly use zero-shot predictions, enabling efficient adaptation.

where $\beta$ represents the ρ-percentile entropy threshold, and $H ( \cdot )$ denotes Shannon entropy. To promote cross-view consistency, TPT optimizes textual prompts by minimizing the entropy of the averaged predictions over the selected set S:

$$
{ \cal L } _ { \mathrm { T P T } } ( x ) = { \cal H } \left( \frac { 1 } { | S | } \sum _ { i \in S } p _ { \mathrm { z s } } ( A _ { i } ( x ) ) \right) ,\tag{5}
$$

The adapted prediction $p _ { \mathrm { a d a p t } } ( x )$ is then obtained from the original view $\boldsymbol { \mathcal { A } } _ { 0 } ( \boldsymbol { x } )$ using the updated model. However, when predictions in S are consistent with $p _ { \mathrm { z s } } ( { \mathcal { A } } _ { 0 } ( x ) )$ , they provide little informative supervision for model updates. We therefore argue that the similarity among selected predictions is closely related to the effectiveness of subsequent adaptation. To measure this, we define a prediction consistency score $\begin{array} { r } { \alpha _ { \mathrm { C o n } } ( x ) = \sum _ { i \in S } \mathbb { I } \big ( p _ { \mathtt { z s } } ( \mathcal { A } _ { i } ( x ) ) = p _ { \mathtt { z s } } ( \mathcal { A } _ { 0 } ( x ) ) \big ) } \end{array}$ , which measures how many augmented views produce the same prediction as the original view. Higher consistency suggests that the prediction is already stable across different views, implying limited benefit from further adaptation.

Hard prediction consistency can be improved by cross-augmentation similarity. Samples may share the same predicted label while exhibiting different probability distributions. In such cases, adaptation may still be beneficial and should therefore not be skipped. To address this issue, we introduce CAS, a scoring function for selective adaptation that jointly considers prediction consistency and distribution similarity:

$$
\alpha _ { \mathrm { C A S } } ( x ) = \sum _ { i \in S } \widetilde { \mathrm { c o s } } ( p _ { \mathbb { Z } \mathrm { s } } ( A _ { i } ( x ) ) , p _ { \mathbb { Z } \mathrm { s } } ( \mathcal { A } _ { 0 } ( x ) ) ) \cdot \mathbb { I } ( y _ { \mathbb { Z } \mathrm { s } } ( \mathcal { A } _ { i } ( x ) ) = y _ { \mathbb { Z } \mathrm { s } } ( \mathcal { A } _ { 0 } ( x ) ) ) .\tag{6}
$$

For each augmented view $\mathcal { A } _ { i } ( x )$ , a reweighting factor is computed via cosine similarity and normalized across selected views. Specifically, letting $p _ { i }$ and $p _ { 0 }$ denote $p _ { \mathrm { z s } } ( { \mathcal { A } } _ { i } ( x ) )$ and $p _ { \mathrm { z s } } ( { \mathcal { A } } _ { 0 } ( x ) )$ respectively, the normalized similarity is given by

Table 1: Performance comparison of different selection strategies under various TTA methods with ViT-B/16. We report AUC and AEP on ImageNet and its variants. The best results are highlighted in bold.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Strategy</td><td colspan="2">ImageNet</td><td colspan="2">ImageNet-A</td><td colspan="2">ImageNet-V</td><td colspan="2">ImageNet-R</td><td colspan="2">ImageNet-K</td><td colspan="2"> $\operatorname { A v g } .$ </td></tr><tr><td>AUC↑ AEP↑</td><td></td><td>AUC↑</td><td>AEP↑</td><td>AUC↑</td><td>AEP↑</td><td>AUC↑</td><td>AEP↑</td><td>AUC↑</td><td>AEP↑</td><td>AUC↑</td><td>AEP↑</td></tr><tr><td rowspan="4">TPT [53]</td><td>Random</td><td>50.77</td><td>68.19</td><td>51.08</td><td>52.54</td><td></td><td>51.38 62.62</td><td></td><td>48.54</td><td>75.97</td><td>49.46</td><td>47.23 50.25</td><td>61.31</td></tr><tr><td>Energy [38]</td><td>57.27</td><td>68.41</td><td>52.50</td><td>52.70</td><td>57.70</td><td>62.95</td><td>64.31</td><td>76.64</td><td>54.04</td><td>47.41</td><td>57.16</td><td>61.62</td></tr><tr><td>MCM [41]</td><td>63.82</td><td>68.50</td><td>53.73</td><td>52.73</td><td>61.44</td><td>62.95</td><td>71.09</td><td>76.77</td><td>54.33</td><td>47.36</td><td>60.88</td><td>61.66</td></tr><tr><td>CAS</td><td>91.10</td><td>69.00</td><td>88.62</td><td>54.70</td><td>89.69</td><td>63.45</td><td>93.45</td><td>77.14</td><td>84.24</td><td>47.93</td><td>89.42</td><td>62.44</td></tr><tr><td rowspan="4">R-TPT [51]</td><td>Random</td><td>50.63</td><td>68.50</td><td>50.67</td><td>54.55</td><td>51.71</td><td>63.09</td><td>48.27</td><td>75.87</td><td>49.85</td><td>47.14</td><td>50.23</td><td>61.83</td></tr><tr><td>Energy [38]</td><td>58.17</td><td>68.78</td><td>52.22</td><td>54.82</td><td>58.69</td><td>63.45</td><td>64.03</td><td>76.56</td><td>54.11</td><td>47.35</td><td>57.44</td><td>62.19</td></tr><tr><td>MCM [41]</td><td>65.35</td><td>68.90</td><td>53.33</td><td>54.75</td><td>63.44</td><td>63.46</td><td>71.37</td><td>76.68</td><td>55.17</td><td>47.27</td><td>61.73</td><td>62.21</td></tr><tr><td>CAS</td><td>91.54</td><td>69.45</td><td>89.77</td><td>57.76</td><td>89.55</td><td>64.03</td><td>93.52</td><td>77.13</td><td>84.86</td><td>48.02</td><td>89.85</td><td>63.28</td></tr><tr><td rowspan="4">STS [6]</td><td>Random</td><td>50.23</td><td>68.17</td><td>50.84</td><td>57.21</td><td>51.41</td><td>63.17</td><td>48.30</td><td>75.94</td><td>50.11</td><td></td><td>50.18</td><td></td></tr><tr><td>Energy [38]</td><td>58.21</td><td>68.41</td><td>53.69</td><td>57.78</td><td>59.03</td><td>63.58</td><td>64.71</td><td>76.78</td><td>53.78</td><td>47.45</td><td></td><td>62.39</td></tr><tr><td>MCM [41]</td><td>65.62</td><td>68.45</td><td>55.68</td><td>57.75</td><td>64.29</td><td>63.55</td><td>71.67</td><td>76.78</td><td>54.17</td><td>47.55 47.38</td><td>57.88</td><td>62.82</td></tr><tr><td>CAS</td><td>94.51</td><td>68.89</td><td>91.02</td><td>61.23</td><td>93.03</td><td>64.14</td><td>94.96</td><td>77.11</td><td>88.10</td><td>48.14</td><td>62.29 92.32</td><td>62.78 63.90</td></tr><tr><td rowspan="4">ZERO [10]</td><td>Random</td><td>51.08</td><td>68.57</td><td>50.71</td><td>55.92</td><td>51.42</td><td>63.22</td><td>48.59</td><td>76.10</td><td>49.80</td><td>47.69</td><td>50.32</td><td>62.30</td></tr><tr><td>Energy [38]</td><td>57.04</td><td>68.72</td><td>51.89</td><td>56.29</td><td>57.66</td><td>63.53</td><td>63.79</td><td>76.78</td><td>54.09</td><td>47.83</td><td>56.89</td><td>62.63</td></tr><tr><td>MCM [41]</td><td>64.22</td><td>68.83</td><td>52.92</td><td>56.16</td><td>62.71</td><td>63.57</td><td>70.64</td><td>76.86</td><td>54.82</td><td>47.74</td><td>61.06</td><td>62.63</td></tr><tr><td>CAS</td><td>94.58</td><td>69.36</td><td>91.71</td><td>59.58</td><td>93.35</td><td>64.22</td><td>95.05</td><td>77.27</td><td>88.41</td><td>48.50</td><td>92.62</td><td>63.79</td></tr></table>

$\begin{array} { r } { \widetilde { \cos } ( p _ { i } , p _ { 0 } ) = \frac { \cos ( p _ { i } , p _ { 0 } ) } { \sum _ { j \in S } \cos ( p _ { j } , p _ { 0 } ) } } \end{array}$ . As illustrated in Fig. 3, augmentations with higher consistency with the original prediction are assigned larger weights, thereby contributing more to the final score. The pseudo-code is provided in the Appendix.

## 4 Experiment

## 4.1 Experimental Setup

Datasets. To comprehensively evaluate the efficiency and performance of CAS, we conduct experiments across a range of benchmarks, including ImageNet and its variants, as well as fine-grained datasets. We first use ImageNet [9] and its four variants: ImageNet-A (natural adversarial examples) [26], ImageNet-V (recollected images) [47], ImageNet-R (artistic renditions) [22], and ImageNet-K (sketch-style images with domain shifts) [58]. We further evaluate our method on fine-grained datasets to assess cross-domain generalization, including Flowers102 [43], DTD [4], Pets [45], UCF101 [54], Caltech101 [12], Aircraft [40], EuroSAT [20], Cars [32], Food101 [3], and SUN397 [62]. In the episodic TTA setting, no training data is available at test time, and all experiments are conducted strictly in a zero-shot manner.

Baselines. To validate the generalizability of our method, we integrate CAS into four representative TTA methods: TPT [53], ZERO [10], R-TPT [52], and STS [6]. We compare CAS with three sample selection strategies (Random Skipping, Energy [38], and MCM [41]) to demonstrate its effectiveness. Random skipping serves as a lower-bound baseline for comparison. As representative OOD detection approaches, Energy [38] utilizes a logit-based energy score, while

Table 2: Performance comparison of different selection strategies under various TTA methods with ViT-B/16. We report AUC and AEP on fine-grained datasets. The best results are highlighted in bold.
<table><tr><td>Method</td><td>Strategy</td><td>Metric</td><td>Flow.</td><td>DTD</td><td>Pets</td><td>UCF</td><td>Cal.</td><td>Air.</td><td>Euro.</td><td>Cars</td><td>Food</td><td>SUN</td><td>Avg.</td></tr><tr><td rowspan="4">TPT [53]</td><td>Random</td><td>AUC AEP</td><td>49.04 68.23</td><td>49.06 45.98</td><td>55.59 87.60</td><td>49.33 67.08</td><td>46.03 94.12</td><td>53.72 23.89</td><td>49.55 42.55</td><td>49.94 66.28</td><td>50.63 84.35</td><td>50.12 64.48</td><td>50.30 64.46</td></tr><tr><td>Energy [38]</td><td>AUC AEP</td><td>63.48 68.42</td><td>53.52 46.26</td><td>69.46 87.36</td><td>60.01 67.62</td><td>63.46 94.39</td><td>46.97 23.36</td><td>75.99 43.97</td><td>54.58 66.28</td><td>69.82 84.54</td><td>59.56 64.95</td><td>61.69 64.72</td></tr><tr><td>MCM [41]</td><td>AUC AEP</td><td>68.47 68.53</td><td>69.48 46.78</td><td>75.21 87.30</td><td>65.47 67.64</td><td>73.69 94.33</td><td>47.83 23.09</td><td>70.17 43.38</td><td>63.01 66.47</td><td>79.74 84.60</td><td>65.92 65.08</td><td>67.90 64.72</td></tr><tr><td>CAS</td><td>AUC AEP</td><td>89.64 86.47 68.65</td><td>47.25</td><td>95.61 87.30</td><td>89.67 68.09</td><td>94.42 94.22</td><td>65.06 23.86</td><td>82.57 43.19</td><td>89.45 66.68</td><td>96.50 84.68</td><td></td><td>90.34 87.97 65.58 64.95</td></tr><tr><td rowspan="4">R-TPT [51]</td><td>Random</td><td>AUC AEP</td><td>47.42 67.91</td><td>50.76 45.53</td><td>53.97 87.38</td><td>49.31 66.66</td><td>49.28 94.01</td><td>55.72 24.42</td><td>49.08 36.84</td><td>50.93 66.47</td><td>50.50 84.06</td><td>50.10 64.58</td><td>50.71 63.79</td></tr><tr><td>Energy [38]</td><td>AUC AEP</td><td>62.55 68.14</td><td>56.79 45.72</td><td>68.78 87.02</td><td>59.34 67.15</td><td>62.93 94.26</td><td>46.64 23.67</td><td>69.58 38.26</td><td>53.17 66.29</td><td>70.36 84.20</td><td>59.60 65.05</td><td>60.97 63.98</td></tr><tr><td>MCM [41]</td><td>AUC AEP</td><td>66.63 68.25</td><td>69.75 46.10</td><td>73.97 86.94</td><td>65.86 67.12</td><td>73.75 94.14</td><td>48.17 23.55</td><td>65.63 37.49</td><td>63.70 66.54</td><td>80.36 84.19</td><td>66.48</td><td>67.43</td></tr><tr><td>CAS</td><td>AUC AEP</td><td>89.14 68.48</td><td>87.20 46.66</td><td>97.44 86.92</td><td>89.31 67.52</td><td>95.35 94.01</td><td>64.73 24.20</td><td>78.31 37.26</td><td>90.12 66.88</td><td>96.47 84.26</td><td>65.17 90.59 87.87</td><td>63.95</td></tr><tr><td rowspan="4">STS [6]</td><td>Random</td><td>AUC AEP</td><td>49.91 66.37</td><td>49.81 45.40</td><td>52.74 87.14</td><td>48.41 65.93</td><td>51.65 93.87</td><td>54.23 24.59</td><td>50.33 39.41</td><td>51.31 66.82</td><td>50.16 83.34</td><td>50.18 64.12</td><td>65.71 64.19 50.87 63.70</td></tr><tr><td>Energy [38]</td><td>AUC AEP</td><td>60.70 66.10</td><td>58.16 45.79</td><td>68.22 86.71</td><td>57.45 66.41</td><td>63.41 94.05</td><td>47.38 24.32</td><td>69.11 39.86</td><td>53.17 66.63</td><td>70.39 83.31</td><td>59.50 64.47</td><td>60.75 63.77</td></tr><tr><td>MCM [41]</td><td>AUC AEP</td><td>64.91 66.05</td><td>69.78 45.96</td><td>74.22 86.60</td><td>65.83 66.32</td><td>76.92 93.93</td><td>48.47 24.18</td><td>66.03 39.57</td><td>63.23 66.91</td><td>79.69 83.17</td><td>66.54 64.54</td><td>67.56 63.72</td></tr><tr><td>CAS</td><td>AUC AEP</td><td>93.07 66.03</td><td>89.97 46.18</td><td>97.68 86.49</td><td>93.24 66.59</td><td>98.94 93.63</td><td>76.44 24.56</td><td>81.97 39.26</td><td>92.29 67.20</td><td>96.85 83.17</td><td>93.70 91.42</td><td>64.88 63.80</td></tr><tr><td rowspan="4">ZERO [10]</td><td>Random</td><td>AUC AEP</td><td>46.49 66.80</td><td>50.30 44.96</td><td>54.13 87.68</td><td>47.44 65.76</td><td>49.82 93.93</td><td>56.01 24.90</td><td>49.12 38.73</td><td>50.85 66.89</td><td>50.87 83.71</td><td>49.98 64.59</td><td>50.50 63.80</td></tr><tr><td>Energy [38]</td><td>AUC AEP AUC</td><td>56.72 66.71 62.91</td><td>56.22 45.16</td><td>67.86 87.44</td><td>57.99 66.26</td><td>66.72 94.23 79.23</td><td>47.31 24.36 48.10</td><td>69.19 39.17 64.73</td><td>53.62 66.71 62.39</td><td>69.98 83.76 79.73</td><td>59.01 65.02 65.93</td><td>60.46 63.88 67.02</td></tr><tr><td>MCM [41]</td><td>AEP</td><td>66.84</td><td>67.47 45.26</td><td>73.80 87.35</td><td>65.89 66.19</td><td>94.11</td><td>24.41</td><td>38.59</td><td>66.96</td><td>83.70</td><td>65.15</td><td>63.86</td></tr><tr><td>CAS</td><td>AUC AEP</td><td>91.95 66.99</td><td>90.93 45.46</td><td>97.58 87.30</td><td>93.37 66.44</td><td>98.29 93.90</td><td>76.71 24.81</td><td>81.48 38.69</td><td>91.60 67.32</td><td>96.87 83.73</td><td>93.75</td><td>91.25 65.58 64.02</td></tr></table>

MCM [41] measures confidence by evaluating the alignment between visual features and textual concepts.

Metrics. To comprehensively evaluate the effectiveness of selective adaptation, we introduce two complementary metrics that assess both detection quality and overall performance. AUC measures the selection strategy's ability to distinguish between beneficial and ineffective adaptations, independent of the decision threshold. Meanwhile, AEP evaluates the strategy's performance across varying skip ratios. It computes the expected accuracy under a predefined prior distribution of skip ratios (e.g., a triangular prior), thereby reflecting the overall efficiency-accuracy trade-off of the selection strategy.

Table 3: Performance comparison of different selection strategies across various TTA methods with ViT-B/16. We report AEP, ECE expectation with triangular prior (EEP), and AUC on ImageNet and its variants.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Strategy</td><td colspan="3">ImageNet</td><td colspan="3">ImageNet-A</td><td colspan="3">ImageNet-V</td><td colspan="3">ImageNet-R</td><td colspan="3">ImageNet-K</td><td colspan="3"> $\operatorname { A v g } .$ </td></tr><tr><td>AEP↑ EEP↓ AUC↑</td><td></td><td></td><td>AEP↑</td><td>EEP↓</td><td>AUC↑</td><td>AEP↑EEP↓ AUC↑</td><td></td><td></td><td>AEP↑ EEP↓</td><td></td><td>AUC↑</td><td>AEP↑</td><td>EEP↓</td><td>AUC↑</td><td>AEP↑ EEP↓</td><td></td><td>AUC↑</td></tr><tr><td rowspan="4">TPT [53]</td><td>Random</td><td>68.19</td><td>7.40</td><td>50.77</td><td>52.54</td><td>12.62</td><td>51.08</td><td>62.62</td><td>8.52</td><td>51.38</td><td>75.97</td><td>3.25</td><td>48.54</td><td>47.23</td><td>12.07</td><td>49.46</td><td>61.31</td><td>8.77</td><td>50.25</td></tr><tr><td>Energy [38]</td><td>68.41</td><td>7.68</td><td>57.27</td><td>52.70</td><td>13.23</td><td>52.50</td><td>62.95</td><td>9.01</td><td>57.70</td><td>76.64</td><td>3.27</td><td>64.31</td><td>47.41</td><td>12.12</td><td>54.04</td><td>61.62</td><td>9.06</td><td>57.16</td></tr><tr><td>MCM [41]</td><td>68.50</td><td>7.91</td><td>63.82</td><td>52.73</td><td>13.66</td><td>53.73</td><td>62.95</td><td>9.15</td><td>61.44</td><td>76.77</td><td>3.54</td><td>71.09</td><td>47.36</td><td>12.17</td><td>54.33</td><td>61.66</td><td>9.29</td><td>60.88</td></tr><tr><td>CAS</td><td>69.00</td><td>7.33</td><td>91.10</td><td>54.70 12.02</td><td></td><td>88.62</td><td>63.45</td><td>8.54</td><td>89.69</td><td>77.14</td><td>3.89</td><td>93.45</td><td>47.93 11.24 84.24</td><td></td><td></td><td>62.44</td><td>8.60</td><td>89.42</td></tr><tr><td rowspan="4">C-TPT [68]</td><td>Random</td><td>67.88</td><td>3.88</td><td>50.83</td><td>50.14</td><td>7.95</td><td>50.04</td><td>62.03</td><td>5.02</td><td>51.21</td><td>75.20</td><td>2.05</td><td>49.29</td><td>47.03</td><td>8.34</td><td>50.57</td><td>60.46</td><td>5.45</td><td>50.39</td></tr><tr><td>Energy [38]</td><td>68.14</td><td>4.26</td><td>58.26</td><td>50.66</td><td>8.20</td><td>55.21</td><td>62.31</td><td>5.42</td><td>60.21</td><td>75.63</td><td>1.79</td><td>64.36</td><td>47.15</td><td>8.61</td><td>54.40</td><td>60.78</td><td>5.66</td><td>58.49</td></tr><tr><td>MCM [41]</td><td>68.21</td><td>4.31</td><td>64.98</td><td>50.63</td><td>8.56</td><td>56.72</td><td>62.28</td><td>5.48</td><td>63.78</td><td>75.73</td><td>1.80</td><td>71.42</td><td>47.16</td><td>8.62</td><td>55.57</td><td>60.80</td><td>5.75</td><td>62.49</td></tr><tr><td>CAS</td><td>68.59</td><td>4.01</td><td>86.50</td><td>051.74</td><td>7.53</td><td>84.29</td><td>62.68</td><td>5.05</td><td></td><td>86.40 76.02</td><td>2.91</td><td>90.22</td><td>47.60</td><td>7.67</td><td>80.95</td><td>61.33</td><td>5.43</td><td>85.67</td></tr><tr><td rowspan="4">0-TPT [50]</td><td>Random</td><td>67.18</td><td>1.95</td><td>51.35</td><td>48.11</td><td>7.22</td><td>51.65</td><td>61.32</td><td>3.02</td><td>52.08</td><td>74.02</td><td>3.87</td><td>50.15</td><td>46.51</td><td>5.36</td><td>50.96</td><td>59.43</td><td>4.28</td><td>51.24</td></tr><tr><td>Energy [38]</td><td>67.28</td><td>2.05</td><td>57.35</td><td>48.26</td><td>7.05</td><td>55.41</td><td>61.35</td><td>2.97</td><td>57.80</td><td>74.02</td><td>3.81</td><td>62.41</td><td>46.49</td><td>5.59</td><td>52.00</td><td>59.48</td><td>4.29</td><td>56.99</td></tr><tr><td>MCM [41]</td><td>67.35</td><td>2.03</td><td>63.50</td><td>48.22</td><td>7.18</td><td>57.28</td><td>61.32</td><td>2.97</td><td>61.37</td><td>74.10</td><td>3.83</td><td>69.81</td><td>46.50</td><td>5.64</td><td>53.12</td><td>59.50</td><td>4.33</td><td>61.02</td></tr><tr><td>CAS</td><td>67.70</td><td>2.25</td><td>76.05</td><td>49.29</td><td>6.57</td><td>78.18</td><td>61.71</td><td>3.16</td><td></td><td>75.91 74.54</td><td>4.24</td><td></td><td>83.31 46.93</td><td>5.27</td><td></td><td>69.70 60.03</td><td>4.30</td><td>76.63</td></tr></table>

Implementation details. We use CLIP-ViT-B/16 [46] as the backbone model and follow the standard TPT setting. The text prompt is initialized with the template $^ { 6 6 } \mathrm { a }$ photo of $\mathrm { a } ^ { \dag } .$ For each image, we generate N = 64 augmented views via AugMix [25]. The confidence threshold $\rho$ is set to 0.1 and kept fixed across all datasets. Experiments are conducted with multiple random seeds. All baseline results are reproduced following the previous benchmark [51].

## 4.2 Results

Performance on ImageNet and its variants. We evaluate the performance of CAS on ImageNet and its variants. As demonstrated in Table 1, CAS consistently outperforms all competing selection strategies across diverse TTA methods [6, 10, 52, 53] with respect to both AEP and AUC. Specifically, when integrated into the ZERO [10], CAS achieves state-of-the-art performance on ImageNet, attaining an AEP of 69.36% and an AUC of 94.58%. The latter represents an outperformance of 30.36% over the second-best baseline, MCM [41] On ImageNet-A, CAS improves the AEP of R-TPT [52] to 57.76%, validating its superior ability in filtering out ineffective adaptations. Across the evaluated datasets, CAS maintains an AUC around 90%, peaking at 92.62% when combined with ZERO [10]. These results show that CAS effectively distinguishes beneficial from harmful adaptations, achieving a good balance between efficiency and accuracy under distribution shifts.

Performance on fine-grained datasets. We evaluate CAS on fine-grained benchmarks, with results summarized in Table 2. Across all evaluated strategies, CAS consistently delivers the highest average performance. Under the TPT framework [53], CAS achieves a peak average AUC of 87.97% and an AEP of 64.95%. It outperforms MCM [41] and Energy [38] by margins of 20.07% and 26.28% in AUC, respectively, demonstrating a superior ability to identify ineffective adaptations. Specifically, when integrated into ZERO [10], CAS demonstrates highly competitive results, achieving an AUC of 98.29% on Caltech101 and 96.87% on Food101.

![](images/15c3bc838ba5b2807ece42c173f136d3c23c13eb1492a8b962462159b9b2781f.jpg)  
(a) TPT [53]  
(b) R-TPT [52]  
(c) STS [6]  
(d) ZERO [10]  
Fig. 4: Accuracy (%) versus skip ratio (%) under different skipping strategies across four TTA methods on ImageNet, ImageNet-A, and DTD. CAS consistently maintains higher accuracy even when skipping a large proportion of samples. Compared with other strategies, CAS demonstrates a superior efficiency-accuracy trade-off.

## 4.3 Impact on Calibration

While our main results focus on TTA methods [6, 10, 52,53] that aim to improve classification accuracy, several prior works emphasize calibration, including C-TPT [68] and O-TPT [50]. To evaluate the generalization of our approach, we further incorporate CAS into these calibration-oriented methods. We additionally report the ECE expectation with a triangular prior (EEP), which is computed in the same manner as AEP in Eq. (3). The results are summarized in Table 3. Across all methods, CAS consistently achieves the highest AEP and AUC, while maintaining lower EEP. Under TPT [53], CAS maintains the lowest average EEP of 8.60% across ImageNet and its variants. This indicates that selective skipping guided by CAS does not amplify overconfidence or destabilize prediction margins. For calibration-oriented methods such as C-TPT [68] and O-TPT [50], CAS continues to generalize effectively. In C-TPT [68], CAS achieves the highest average AEP of 61.33% and AUC of 85.67% while preserving competitive calibration performance with an EEP of 5.43%. Similarly, under O-TPT [50], CAS yields the best AEP of 60.03% and AUC of 76.63%, while the other three show comparable performance in EEP. Overall, these results suggest that our method is not limited to accuracy-oriented TTA methods. It also generalizes effectively to calibration-oriented methods, consistently improving robustness and reliability without sacrificing calibration performance.

Table 4: Ablation study of CAS across various TTA methods on both ImageNet and its variants and fine-grained benchmarks, evaluated using AUC and AEP.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Strategy</td><td colspan="2">TPT [53]</td><td colspan="2">R-TPT [51]</td><td colspan="2">STS [6]</td><td colspan="2">ZERO [10]</td></tr><tr><td>AUC↑</td><td>AEP↑</td><td>AUC↑</td><td>AEP↑</td><td>AUC↑</td><td>AEP↑</td><td>AUC↑</td><td>AEP↑</td></tr><tr><td rowspan="4">ImageNet &amp; its variants</td><td>MCM [41]</td><td>60.88</td><td>61.66</td><td>61.73</td><td>62.21</td><td>62.29</td><td>62.78</td><td>61.06</td><td>62.63</td></tr><tr><td>Similarity</td><td>82.84</td><td>62.31</td><td>85.50</td><td>63.17</td><td>88.41</td><td>63.84</td><td>87.78</td><td>63.73</td></tr><tr><td>Consistency</td><td>89.14</td><td>62.44</td><td>89.89</td><td>63.28</td><td>92.65</td><td>63.90</td><td>92.59</td><td>63.79</td></tr><tr><td>CAS</td><td>89.42</td><td>62.44</td><td>89.85</td><td>63.28</td><td>92.32</td><td>63.90</td><td>92.62</td><td>63.79</td></tr><tr><td rowspan="4">Fine-grained</td><td>MCM [41]</td><td>81.08</td><td>64.70</td><td>80.70</td><td>63.95</td><td>79.54</td><td>63.70</td><td>79.60</td><td>63.86</td></tr><tr><td>Similarity</td><td>81.04</td><td>64.81</td><td>81.44</td><td>63.93</td><td>83.61</td><td>63.60</td><td>82.91</td><td>63.79</td></tr><tr><td>Consistency</td><td>87.65</td><td>64.93</td><td>87.62</td><td>64.13</td><td>91.27</td><td>63.84</td><td>90.82</td><td>63.97</td></tr><tr><td>CAS</td><td>87.97</td><td>64.95</td><td>87.87</td><td>64.19</td><td>91.42</td><td>63.80</td><td>91.25</td><td>64.02</td></tr></table>

Table 5: Accuracy (%) of TTA methods on ImageNet with ViT-B/16 at an 85% skip ratio. Total inference time is reported in hours (h). CLIP\* denotes CLIP with 64 augmentations.
<table><tr><td rowspan="2">Metric</td><td colspan="2">CLIP*</td><td colspan="2">TPT [53]</td><td colspan="2">R-TPT [52]</td><td colspan="2">STS [6]</td><td colspan="2">ZERO [10]</td></tr><tr><td>Base</td><td>CAS</td><td>Base</td><td>CAS</td><td>Base</td><td>CAS</td><td>Base</td><td>CAS</td><td>Base</td><td>CAS</td></tr><tr><td>Acc. (%)</td><td>66.72</td><td>一</td><td>68.88</td><td>69.03</td><td>69.36</td><td>69.43</td><td>68.83</td><td>69.20</td><td>69.28</td><td>69.32</td></tr><tr><td>Time (h)</td><td>1.65</td><td>一</td><td>7.76</td><td>2.55</td><td>6.73</td><td>2.40</td><td>2.03</td><td>1.68</td><td>5.42</td><td>4.96</td></tr><tr><td>Speedup</td><td colspan="2"></td><td colspan="2">3.04×</td><td colspan="2">2.81×</td><td colspan="2">1.21×</td><td colspan="2">1.09×</td></tr></table>

## 4.4 In-depth Analysis

Trade-off between efficiency and accuracy. To evaluate the trade-off between efficiency and performance, we show the accuracy-skip ratio curves for four TTA methods [6, 10, 52, 53], varying the skip ratio s from 0 to 1 with a step size of 0.05. The results demonstrate that CAS consistently maintains or improves accuracy compared to the full-adaptation baseline (s = 0), even when skipping up to 85% of samples. This suggests that standard TTA may over-adapt certain samples, while CAS selectively identifies and bypasses harmful adaptations, thereby mitigating performance degradation. Notably, R-TPT [52] achieves an accuracy above 69.60% when skipping nearly 80% of samples on the ImageNet dataset, while STS [6] reaches 69.10% with a skip ratio of 90%. In contrast, selection strategies based on Energy [38] and MCM [41] exhibit steep accuracy declines as the skip ratio increases. In general, Figure 4 shows that CAS can effectively improve computational overhead without sacrificing accuracy, serving as a good baseline for the selective adaptation problem.

Ablation study. CAS consists of augmentation prediction consistency and similarity reweighting, which correspond to the consistency and similarity components, respectively. To analyze their individual contributions, we evaluate each component independently as a scoring function. The results are reported in Table 4. Both consistency and similarity serve as effective skipping strategies compared to the classical MCM criterion [41], consistently yielding an improvement of about 30% in AUC on ImageNet and its variants. However, the integrated CAS baseline achieves the most stable and competitive performance overall, attaining the best or near-best AUC and AEP across all methods. While consistency yields slightly higher gains than CAS in a few isolated cases, CAS demonstrates more stable improvements, particularly on fine-grained datasets. Notably, CAS improves the AUC from 90.82% to 91.25% under ZERO [10], further validating the effectiveness of combining both components.

Table 6: Performance comparison of different AEP metric functions under TPT [53] framework. We report AUC and AEP on ImageNet and its variants.
<table><tr><td>Method</td><td>Strategy</td><td></td><td></td><td></td><td></td><td>ImageNet ImageNet-A ImageNet-V ImageNet-R ImageNet-K</td><td>Avg.</td></tr><tr><td rowspan="4">f(s) = 1</td><td>Random</td><td>67.83</td><td>51.41</td><td>62.23</td><td>75.43</td><td>46.93</td><td>60.77</td></tr><tr><td>Energy [38]</td><td>68.03</td><td>51.49</td><td>62.53</td><td>76.13</td><td>47.13</td><td>61.06</td></tr><tr><td>MCM [41]</td><td>68.16</td><td>51.45</td><td>62.52</td><td>76.26</td><td>47.07</td><td>61.09</td></tr><tr><td>CAS</td><td>68.91</td><td>53.99</td><td>63.32</td><td>76.96</td><td>47.80</td><td>62.19</td></tr><tr><td rowspan="4">f(s) = 2s</td><td>Random</td><td>67.47</td><td>50.27</td><td>61.85</td><td>74.88</td><td>46.63</td><td>60.22</td></tr><tr><td>Energy [38]</td><td>67.66</td><td>50.28</td><td>62.12</td><td>75.62</td><td>46.84</td><td>60.50</td></tr><tr><td>MCM [41]</td><td>67.82</td><td>50.17</td><td>62.10</td><td>75.76</td><td>46.78</td><td>60.53</td></tr><tr><td>CAS</td><td>68.81</td><td>53.28</td><td>63.19</td><td>76.78</td><td>47.66</td><td>61.94</td></tr></table>

Table 7: Performance using resized crops/horizontal flip [10] as data augmentation strategy under ZERO [10] with ViT-B/16. We report AUC and AEP on ImageNet and its variants.
<table><tr><td rowspan="3">Method</td><td rowspan="3">Strategy</td><td colspan="2">ImageNet</td><td colspan="2">ImageNet-A</td><td colspan="2">ImageNet-V</td><td colspan="2">ImageNet-R</td><td colspan="2">ImageNet-K</td><td colspan="2">Avg.</td></tr><tr><td>AUC↑</td><td>AEP↑</td><td>AUC↑</td><td>AEP↑</td><td>AUC↑</td><td>AEP↑</td><td>AUC↑</td><td>AEP↑</td><td>AUC↑</td><td>AEP↑</td><td>AUC↑</td><td>AEP↑</td></tr><tr><td rowspan="4">ZERO [10]</td><td>Random</td><td>51.18</td><td>68.50</td><td>50.66</td><td>55.91</td><td>51.33</td><td>63.18</td><td>48.78</td><td>76.15</td><td>49.99</td><td>47.70</td><td>50.39</td><td>62.29</td></tr><tr><td>Energy [38]</td><td>57.12</td><td>68.67</td><td>51.98</td><td>56.30</td><td>57.93</td><td>63.51</td><td>63.68</td><td>76.82</td><td>53.94</td><td>47.81</td><td>56.93</td><td>62.62</td></tr><tr><td>MCM [41]</td><td>64.30</td><td>68.77</td><td>52.96</td><td>56.15</td><td>62.89</td><td>63.52</td><td>70.56</td><td>76.93</td><td>54.80</td><td>47.72</td><td>61.10</td><td>62.62</td></tr><tr><td>CAS</td><td>94.59</td><td>69.27</td><td>91.89</td><td>59.54</td><td>93.33</td><td>64.16</td><td>95.08</td><td>77.34</td><td>88.37</td><td>48.49</td><td></td><td>92.65 63.76</td></tr></table>

Computation cost. To validate the efficiency of our baseline, we measure the total inference time of different TTA methods [6, 10,52,53] with and without CAS, as shown in Table 5. Specifically, we report the total inference time on ImageNet using ViT-B/16. By integrating CAS, the overall adaptation time of TPT [53] decreases from 7.76 to 2.55 hours. This delivers a 3.04× speedup over full adaptation while marginally improving accuracy from 68.88% to 69.03%. For training-free TTA methods such as ZERO [10] and computationally efficient methods like STS [6], CAS further reduces computational overhead by approximately 20% and 10%, respectively, while simultaneously improving accuracy. These results demonstrate that CAS effectively accelerates the overall adaptation process without sacrificing TTA performance.

Different AEP measuring functions. We choose a decreasing linear function because lower skip ratios may be more preferred in real-world deployments as they sacrifice less performance. And $f ( s ) \ : = \ : 2 ( 1 - s )$ was explicitly chosen because its integral evaluates exactly to 1. To verify that our method is not dependent on a specific AEP metric function, we further evaluate CAS with two alternative AEP functions, including $f ( s ) = 1$ and $f ( s ) = 2 s$ . As shown in Table 6, CAS consistently achieves the best performance under both alternative settings. Specifically, CAS obtains the highest average AEP of 62.19% with $f ( s ) = 1$ and 61.94%with $f ( s ) = 2 s$ , outperforming other strategies. These results indicate that CAS remains robust across different AEP measuring functions.

![](images/b7c3f9c23c2a94a2d1e2fd76abd2d39a55c93a7e82c682f87d1c9d376f4bf64e.jpg)  
(a) ImageNet

![](images/59c77a67580ff84038b4a5ba7e81e653ccd7e7a0281b1a5efcb51fb2d4b77ef3.jpg)  
(b) ImageNet-A

![](images/82d48034b451186a466610f2d1835d375c29e143bef940a1e2664322e2548dc4.jpg)  
(c) Flowers102

![](images/7626d1ffbc3535e74716a37364b94ae44619b69e3d53508c0edc9027401bd6af.jpg)  
(d) EuroSAT

![](images/7d5a95bd7e776ab1e13278af2644ef12dc7f98740651a2b024d47394036603d0.jpg)  
(e) ImageNet

![](images/a32b17fd64be186add3c20bedfd359b7c2d2f7e53d95a4fba0853f67e0df482c.jpg)  
(f) ImageNet-A

![](images/09fb5939f242e2108c86d8d986491b1e6c9c1a6bc38cfe0ed0c95b10bc355c4c.jpg)  
(g) Flowers102

![](images/489295e490dff549811af0e7fe1bec672422cb3ec68c7fd074cc1c1960de3cda.jpg)  
(h) EuroSAT  
Fig. 5: Sensitivity analysis of CAS under different cutoff percentile $\rho$ ranging from 0.05 to 0.50 under TPT with ViT-B/16. (a)-(d) demonstrate the AUC performance on ImageNet, ImageNet-A, Flowers102, and EuroSAT, respectively, while (e)-(h) show the AEP results for the same datasets.

Different data augmentations. To examine whether CAS depends on a specific augmentation strategy, we replace the AugMix [25] augmentation with the random resized crops $/$ horizontal flips used in ZERO [10]. As shown in Table 7, CAS still achieves the best performance on ImageNet and its variants. Specifically, CAS obtains the highest average AUC of 92.65% and average AEP of 63.76%, outperforming other strategies by a clear margin, indicating that CAS remains effective under different data augmentation settings.

Sensitivity to cutoff percentile $\rho .$ To evaluate the sensitivity of CAS to cutoff percentile $\rho ,$ we vary it from 0.05 $( | S | = 3 )$ to 0.50 $( | S | = 3 2 )$ across four methods [6, 10,52,53] . As illustrated in Figure 5, the AUC remains robust across different $\rho .$ Initially, increasing the number of selected augmentations improves performance. For example, on ImageNet, increasing $\rho$ by 20% improves AUC by about $1 \%$ However, when $\rho$ becomes too large, additional views with high entropy are introduced, which may harm performance and lead to slight degradation. Similar trends are observed on ImageNet-A and fine-grained datasets. In contrast, AEP varies only marginally across different $\rho ,$ indicating that larger ratios bring limited overall benefit. Notably, on EuroSAT, AEP even declines as $\rho$ increases, suggesting that excessive augmentations may negatively affect performance. Based on these results, we set the cutoff percentile $\rho$ to 0.1 as the default setting, as this configuration maintains high detection quality while minimizing computational overhead.

## 4.5 Case Study

For correct-to-correct samples, predictions are stable across augmentations, indicating that the model has learned robust and invariant representations. Conversely, wrong-to-wrong samples yield consistently incorrect predictions, suggesting stable but biased representations that augmentation alone cannot rectify. Meanwhile, wrong-to-correct samples lie near decision boundaries, where augmentations provide consistent corrective signals. In contrast, correct-to-wrong samples are overly sensitive: perturbations disrupt originally correct cues, leading to performance degradation. Overall, stable samples offer limited adaptation gains, boundary samples benefit the most from adaptation, and highly sensitive samples risk degradation from improper updates.

![](images/a69f028ae62bc1ddc7acf835363100b8a07db91f897c1c99b6dc3da29fbb757f.jpg)

![](images/203429ae7c54f38a49edcdfdd116c9415097bfba30593a68e102d08cc3fd8ce5.jpg)

![](images/4dcdfcfa3e72b481c21a9c07babfec43cf0b5d3a84a017acd426732c82b0c117.jpg)

(d) Correct -> Wrong  
![](images/e88b5385dbec952d413da565eaaa2ca5c0a7a8608b804de30097a2dda8421ddd.jpg)  
Fig. 6: Visualization of augmented views and their corresponding predictions.

## 5 Conclusion

While existing TTA methods generally prioritize overall performance gains, this paper shifts the focus toward adaptation efficiency at the per-sample level. Our main contribution is the introduction of a new selective adaptation problem, which aims to determine whether a given test sample should undergo adaptation or be skipped. We also introduce CAS as a simple baseline that maintains performance with reduced computational overhead, while improving end performance serves as an added benefit. We hope this work inspires the community to further investigate this problem and build upon our baseline. Additionally, we encourage the exploration of new directions, such as extending selective adaptation to tasks beyond image classification.

## Acknowledgements

We thank Dr. Lijun Sheng for his critical discussions, and the anonymous reviewers for their constructive comments and helpful suggestions that improved this paper. This work was funded by the National Natural Science Foundation of China under Grants 62276256 and U2441251, Beijing Natural Science Foundation Z260008, and National Key Research and Development Program of China 2026ZD1500301.

## References

1. Ahamed, S.A., Thanthrige, U.S.K.P.M., Rodrigo, R., Khan, M.H.: A-TPT: Angular diversity calibration properties for test-time prompt tuning of vision-language models. In: ICLR (2026)

2. Alayrac, J.B., Donahue, J., Luc, P., Miech, A., Barr, I., Hasson, Y., Lenc, K., Mensch, A., Millican, K., Reynolds, M., et al.: Flamingo: a visual language model for few-shot learning. In: NeurIPS. pp. 23716–23736 (2022)

3. Bossard, L., Guillaumin, M., Van Gool, L.: Food-101-mining discriminative components with random forests. In: ECCV. pp. 446–461 (2014)

4. Cimpoi, M., Maji, S., Kokkinos, I., Mohamed, S., Vedaldi, A.: Describing textures in the wild. In: CVPR. pp. 3606–3613 (2014)

5. Cortes, C., DeSalvo, G., Mohri, M.: Learning with rejection. In: ALT. pp. 67–82 (2016)

6. Dafnis, K.M., Metaxas, D.N.: Test-time spectrum-aware latent steering for zeroshot generalization in vision-language models. In: NeurIPS. pp. 151169–151194 (2025)

7. Dastmalchi, H., An, A., et al.: Etta: Efficient test-time adaptation for visionlanguage models through dynamic embedding updates. In: BMVC (2025)

8. Davis, J., Goadrich, M.: The relationship between precision-recall and roc curves. In: ICML. pp. 233–240 (2006)

9. Deng, J., Dong, W., Socher, R., Li, L.J., Li, K., Fei-Fei, L.: Imagenet: A large-scale hierarchical image database. In: CVPR. pp. 248–255 (2009)

10. Farina, M., Franchi, G., Iacca, G., Mancini, M., Ricci, E.: Frustratingly easy testtime adaptation of vision-language models. In: NeurIPS. pp. 129062–129093 (2024)

11. Fawcett, T.: An introduction to roc analysis. Pattern Recognition Letters 27(8), 861–874 (2006)

12. Fei-Fei, L., Fergus, R., Perona, P.: Learning generative visual models from few training examples: An incremental bayesian approach tested on 101 object categories. In: CVPRW. pp. 178–178 (2004)

13. Feng, C.M., Yu, K., Liu, Y., Khan, S., Zuo, W.: Diverse data augmentation with diffusions for effective test-time prompt tuning. In: ICCV. pp. 2704–2714 (2023)

14. Fisch, A., Jaakkola, T.S., Barzilay, R.: Calibrated selective classification. TMLR (2022)

15. Geifman, Y., El-Yaniv, R.: Selective classification for deep neural networks. In: NeurIPS. p. 4885–4894 (2017)

16. Geifman, Y., El-Yaniv, R.: Selectivenet: A deep neural network with an integrated reject option. In: ICML. pp. 2151–2159 (2019)

17. Granese, F., Romanelli, M., Gorla, D., Palamidessi, C., Piantanida, P.: Doctor: A simple method for detecting misclassification errors. In: NeurIPS. pp. 5669–5681 (2021)

18. Guo, C., Pleiss, G., Sun, Y., Weinberger, K.Q.: On calibration of modern neural networks. In: ICML. pp. 1321–1330 (2017)

19. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: CVPR. pp. 770–778 (2016)

20. Helber, P., Bischke, B., Dengel, A., Borth, D.: Introducing eurosat: A novel dataset and deep learning benchmark for land use and land cover classification. In: IGARSS. pp. 204–207 (2018)

21. Hendrycks, D., Basart, S., Mazeika, M., Zou, A., Kwon, J., Mostajabi, M., Steinhardt, J., Song, D.: Scaling out-of-distribution detection for real-world settings. In: ICML. pp. 8759–8773 (2022)

22. Hendrycks, D., Basart, S., Mu, N., Kadavath, S., Wang, F., Dorundo, E., Desai, R., Zhu, T., Parajuli, S., Guo, M., et al.: The many faces of robustness: A critical analysis of out-of-distribution generalization. In: ICCV. pp. 8340–8349 (2021)

23. Hendrycks, D., Dietterich, T.: Benchmarking neural network robustness to common corruptions and perturbations. In: ICLR (2019)

24. Hendrycks, D., Gimpel, K.: A baseline for detecting misclassified and out-ofdistribution examples in neural networks. In: ICLR (2017)

25. Hendrycks, D., Mu, N., Cubuk, E.D., Zoph, B., Gilmer, J., Lakshminarayanan, B.: Augmix: A simple data processing method to improve robustness and uncertainty. In: ICLR (2020)

26. Hendrycks, D., Zhao, K., Basart, S., Steinhardt, J., Song, D.: Natural adversarial examples. In: CVPR. pp. 15262–15271 (2021)

27. Imam, R., Gani, H., Huzaifa, M., Nandakumar, K.: Test-time low rank adaptation via confidence maximization for zero-shot generalization of vision-language models. In: WACV. pp. 5449–5459 (2025)

28. Jia, C., Yang, Y., Xia, Y., Chen, Y.T., Parekh, Z., Pham, H., Le, Q., Sung, Y.H., Li, Z., Duerig, T.: Scaling up visual and vision-language representation learning with noisy text supervision. In: ICML. pp. 4904–4916 (2021)

29. Karmanov, A., Guan, D., Lu, S., El Saddik, A., Xing, E.: Efficient test-time adaptation of vision-language models. In: CVPR. pp. 14162–14171 (2024)

30. Kim, I., Kim, Y., Kim, S.: Learning loss for test-time augmentation. In: NeurIPS. pp. 4163–4174 (2020)

31. Kim, Y., Cho, D., Han, K., Panda, P., Hong, S.: Domain adaptation without source data. IEEE Transactions on Artificial Intelligence 2(6), 508–518 (2021)

32. Krause, J., Stark, M., Deng, J., Fei-Fei, L.: 3d object representations for finegrained categorization. In: ICCVW. pp. 554–561 (2013)

33. Krizhevsky, A., Hinton, G.: Learning multiple layers of features from tiny images. Master's thesis, Department of Computer Science, University of Toronto (2009)

34. Li, X., Zhang, D., Du, Z., Zhu, L., Chen, Z., Li, J.: Pataug: Augmentation of augmentation for test-time adaptation. In: ACM MM. pp. 5080–5089 (2025)

35. Liang, H., Peng, L., Sun, J.: Selective classification under distribution shifts. TMLR (2024)

36. Liang, J., He, R., Tan, T.: A comprehensive survey on test-time adaptation under distribution shifts. IJCV 133(1), 31–64 (2025)

37. Liu, H., Li, C., Wu, Q., Lee, Y.J.: Visual instruction tuning. In: NeurIPS. pp. 34892–34916 (2023)

38. Liu, W., Wang, X., Owens, J., Li, Y.: Energy-based out-of-distribution detection. In: NeurIPS. pp. 21464–21475 (2020)

39. Lu, S., Wang, Y., Sheng, L., He, L., Zheng, A., Liang, J.: Out-of-distribution detection: A task-oriented survey of recent advances. ACM Computing Surveys 58(2), 1–39 (2025)

40. Maji, S., Rahtu, E., Kannala, J., Blaschko, M., Vedaldi, A.: Fine-grained visual classification of aircraft. arXiv preprint arXiv:1306.5151 (2013)

41. Ming, Y., Cai, Z., Gu, J., Sun, Y., Li, W., Li, Y.: Delving into out-of-distribution detection with vision-language representations. In: NeurIPS. pp. 35087–35102 (2022)

42. Miyai, A., Yu, Q., Irie, G., Aizawa, K.: Gl-mcm: Global and local maximum concept matching for zero-shot out-of-distribution detection. IJCV (2025)

43. Nilsback, M.E., Zisserman, A.: Automated flower classification over a large number of classes. In: ICVGIP. pp. 722–729 (2008)

44. Niu, S., Wu, J., Zhang, Y., Chen, Y., Zheng, S., Zhao, P., Tan, M.: Efficient testtime model adaptation without forgetting. In: ICML. pp. 16888–16905 (2022)

45. Parkhi, O.M., Vedaldi, A., Zisserman, A., Jawahar, C.: Cats and dogs. In: CVPR. pp. 3498–3505 (2012)

46. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: ICML. pp. 8748–8763 (2021)

47. Recht, B., Roelofs, R., Schmidt, L., Shankar, V.: Do imagenet classifiers generalize to imagenet? In: ICML. pp. 5389–5400 (2019)

48. Saenko, K., Kulis, B., Fritz, M., Darrell, T.: Adapting visual category models to new domains. In: ECCV. pp. 213–226 (2010)

49. Shanmugam, D., Blalock, D., Balakrishnan, G., Guttag, J.: Better aggregation in test-time augmentation. In: ICCV. pp. 1214–1223 (2021)

50. Sharifdeen, A., Munir, M.A., Baliah, S., Khan, S., Khan, M.H.: O-tpt: Orthogonality constraints for calibrating test-time prompt tuning in vision-language models. In: CVPR. pp. 19942–19951 (2025)

51. Sheng, L., Liang, J., He, R., Wang, Z., Tan, T.: The illusion of progress? a critical look at test-time adaptation for vision-language models. In: NeurIPS (2025)

52. Sheng, L., Liang, J., Wang, Z., He, R.: R-tpt: Improving adversarial robustness of vision-language models through test-time prompt tuning. In: CVPR. pp. 29958- 29967 (2025)

53. Shu, M., Nie, W., Huang, D.A., Yu, Z., Goldstein, T., Anandkumar, A., Xiao, C.: Test-time prompt tuning for zero-shot generalization in vision-language models. In: NeurIPS. pp. 14274–14289 (2022)

54. Soomro, K., Zamir, A.R., Shah, M.: Ucf101: A dataset of 101 human actions classes from videos in the wild. arXiv preprint arXiv:1212.0402 (2012)

55. Sui, E., Wang, X., Yeung-Levy, S.: Just shift it: Test-time prototype shifting for zero-shot generalization with vision-language models. In: WACV. pp. 825–835 (2025)

56. Sun, Y., Wang, X., Liu, Z., Miller, J., Efros, A., Hardt, M.: Test-time training with self-supervision for generalization under distribution shifts. In: ICML. pp. 9229–9248 (2020)

57. Wang, D., Shelhamer, E., Liu, S., Olshausen, B., Darrell, T.: Tent: Fully test-time adaptation by entropy minimization. In: ICLR (2021)

58. Wang, H., Ge, S., Lipton, Z., Xing, E.P.: Learning robust global representations by penalizing local predictive power. In: NeurIPS. pp. 10506–10518 (2019)

59. Wang, Q., Fink, O., Van Gool, L., Dai, D.: Continual test-time domain adaptation. In: CVPR. pp. 7201–7211 (2022)

60. Wang, X., Chen, K., Zhang, J., Chen, J., Ma, X.: Tapt: Test-time adversarial prompt tuning for robust inference in vision-language models. In: CVPR. pp. 19910–19920 (2025)

61. Wu, W., Luo, H., Fang, B., Wang, J., Ouyang, W.: Cap4video: What can auxiliary captions do for text-video retrieval? In: CVPR. pp. 10704–10713 (2023)

62. Xiao, J., Hays, J., Ehinger, K.A., Oliva, A., Torralba, A.: Sun database: Large-scale scene recognition from abbey to zoo. In: CVPR. pp. 3485–3492 (2010)

63. Xiao, Z., Yan, S., Hong, J., Cai, J., Jiang, X., Hu, Y., Shen, J., Wang, C., Snoek, C.G.M.: Dynaprompt: Dynamic test-time prompt tuning. In: ICLR (2025)

64. Xing, S., Zhao, Z., Sebe, N.: Clip is strong enough to fight back: Test-time counterattacks towards zero-shot adversarial robustness of clip. In: CVPR. pp. 15172– 15182 (2025)

65. Yang, J., Zhou, K., Li, Y., Liu, Z.: Generalized out-of-distribution detection: A survey. IJCV 132(12), 5635–5662 (2024)

66. Yang, P., Liang, J., Cao, J., He, R.: Auto: Adaptive outlier optimization for online test-time ood detection. arXiv preprint arXiv:2303.12267 (2023)

67. Yin, D., Gontijo Lopes, R., Shlens, J., Cubuk, E.D., Gilmer, J.: A fourier perspective on model robustness in computer vision. In: NeurIPS. vol. 32 (2019)

68. Yoon, H.S., Yoon, E., Tee, J.T.J., Hasegawa-Johnson, M.A., Li, Y., Yoo, C.D.: C-TPT: Calibrated test-time prompt tuning for vision-language models via text feature dispersion. In: ICLR (2024)

69. Yu, Y., Sheng, L., He, R., Liang, J.: Benchmarking test-time adaptation against distribution shifts in image classification. arXiv preprint arXiv:2307.03133 (2023)

70. Yu, Y., Sheng, L., He, R., Liang, J.: Stamp: Outlier-aware test-time adaptation with stable memory replay. In: ECCV. pp. 375–392 (2024)

71. Yuan, L., Xie, B., Li, S.: Robust test-time adaptation in dynamic scenarios. In: CVPR. pp. 15922–15932 (2023)

72. Zanella, M., Ben Ayed, I.: On the test-time zero-shot generalization of visionlanguage models: Do we really need prompt learning? In: CVPR. pp. 23783–23793 (2024)

73. Zhai, X., Mustafa, B., Kolesnikov, A., Beyer, L.: Sigmoid loss for language image pre-training. In: ICCV. pp. 11975–11986 (2023)

74. Zhang, J., Huang, J., Jin, S., Lu, S.: Vision-language models for vision tasks: A survey. IEEE TPAMI 46(8), 5625–5644 (2024)

75. Zhang, M.M., Levine, S., Finn, C.: Memo: Test time robustness via adaptation and augmentation. In: NeurIPS (2022)

76. Zhang, T., Wang, J., Guo, H., Dai, T., Chen, B., Xia, S.T.: Boostadapter: Improving vision-language test-time adaptation via regional bootstrapping. In: NeurIPS. pp. 67795–67825 (2024)

77. Zhou, K., Yang, J., Loy, C.C., Liu, Z.: Conditional prompt learning for visionlanguage models. In: CVPR. pp. 16816–16825 (2022)

78. Zhou, K., Yang, J., Loy, C.C., Liu, Z.: Learning to prompt for vision-language models. IJCV 130(9), 2337–2348 (2022)

79. Zhou, L., Ye, M., Li, S., Li, N., Zhu, X., Deng, L., Liu, H., Lei, Z.: Bayesian testtime adaptation for vision-language models. In: CVPR. pp. 29999–30009 (2025)

80. Zhou, S., Yin, M., Sun, L., Yang, S., Xie, D., Zhu, J.: Training-free test-time adaptation via shape and style guidance for vision-language models. In: NeurIPS. pp. 152968–152982 (2025)

Jiang et al.

## A Algorithm

We provide the pseudo-code for the proposed Cross-Augmentation Similarity (CAS) in Algorithm 1. For a given test image x, the CAS score is computed by evaluating the prediction consistency and distribution similarity across its highquality augmented views. Samples with a high CAS score can skip the adaptation process and rely directly on zero-shot predictions.

Algorithm 1 CAS Algorithm   
Input: Test image x, pretrained VLM $f _ { \theta } ,$ augmentation function A(·), augmentation   
number $( N - 1 )$ , cutoff percentile $\rho ,$ threshold $\gamma .$   
Output: y(x)   
1: $p _ { \mathrm { z s } } ( \mathcal { A } _ { i } ( \boldsymbol { x } ) )  \mathrm { s o f t m a x } ( f _ { \theta } ( \mathcal { A } _ { i } ( \boldsymbol { x } ) ) )$   
2: $H ( \mathcal { A } _ { i } ( x ) ) \gets - p _ { \mathrm { z s } } ( \mathcal { A } _ { i } ( x ) ) \log p _ { \mathrm { z s } } ( \mathcal { A } _ { i } ( x ) )$   
3: $k \gets \lfloor N \rho \rfloor$   
4: $s $ indices of k smallest $H ( A _ { i } ( x ) )$   
5: $c _ { i } \gets \mathbb { I } ( y _ { z s } ( \mathcal { A } _ { i } ( x ) ) = y _ { z s } ( \mathcal { A } _ { 0 } ( x ) ) )$   
6: $s _ { i } \gets \cos ( p _ { z s } ( \mathcal { A } _ { i } ( x ) ) , p _ { z s } ( \mathcal { A } _ { 0 } ( x ) ) )$   
7: $\textstyle W \gets \sum _ { i \in S } s _ { i }$   
8: $\begin{array} { r } { \alpha _ { \mathrm { C A S } } ( x ) \gets \frac { 1 } { W } \sum _ { i \in S } c _ { i } \cdot s _ { i } } \end{array}$   
9: if $\alpha _ { \mathrm { C A S } } ( x ) < \gamma$ then   
10: Perform adaptation   
11: return yadapt(x)   
12: else   
13: return $y _ { \mathbf { z } \mathbf { s } } ( x )$   
14: end if

## B Results on ResNet-50

Performance on ImageNet and its variants We evaluate the performance of CAS on ImageNet and its variants using the ResNet-50 [19] backbone, as summarized in Table 8. As demonstrated, CAS consistently outperforms all competing selection strategies across various TTA methods with respect to both AUC and AEP metrics. Specifically, when integrated into the ZERO [10], CAS achieves the best overall performance, attaining an average AUC of 87.96% and an AEP of 47.54%. This represents a notable improvement over the second-best baseline, MCM [41], which achieves an average AUC of 54.88%, demonstrating stronger detection ability of CAS for selective adaptation. On the ImageNet dataset, CAS integrated with ZERO reaches an AUC of 91.86%, surpassing MCM's 61.44%. Even across challenging datasets like ImageNet-A and ImageNet-R, CAS maintains robust performance, peaking at an AUC of 83.28% under TPT and 89.36% under STS. These results show that CAS generalizes well to other architectures, effectively separating beneficial from harmful adaptations.

Performance on fine-grained datasets. We further evaluate CAS on fine-grained datasets, with the results summarized in Table 9. CAS consistently achieves the highest average performance across all evaluated TTA methods, outperforming Random Energy [38] and MCM [41] strategies. For instance, CAS attains an average AUC of 87.64% under STS [6], which represents a notable margin over MCM [41] (58.56%) and Energy [38] (52.30%), illustrating the superior capability of CAS in filtering out ineffective adaptations. Furthermore, CAS demonstrates remarkable robustness across diverse benchmarks, achieving peak AUCs of 97.08% on Caltech101 and 93.30% on Food101 when integrated with STS. Overall, CAS maintains consistently high average performance across all baselines, highlighting its strong adaptability and superior balance between efficiency and accuracy on fine-grained datasets.

Table 8: Performance comparison of different selection strategies under various TTA methods with RN50. We report AUC and AEP on ImageNet and its variants. The best results under each TTA method are highlighted in bold.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Strategy</td><td colspan="2">ImageNet</td><td colspan="2">ImageNet-A</td><td colspan="2">ImageNet-V</td><td colspan="2">ImageNet-R</td><td colspan="2">ImageNet-K</td><td colspan="2">Avg.</td></tr><tr><td>AUC↑</td><td>AEP↑</td><td>AUC↑</td><td>AEP↑</td><td>AUC↑</td><td>AEP↑</td><td>AUC↑</td><td>AEP↑</td><td>AUC↑</td><td>AEP↑</td><td>AUC↑</td><td>AEP↑</td></tr><tr><td rowspan="4">TPT [53]</td><td>Random</td><td>50.04</td><td>59.90</td><td>50.26</td><td>24.90</td><td>49.87</td><td>53.58</td><td>50.96</td><td>58.17</td><td>50.04</td><td>34.54</td><td>50.23</td><td>46.22</td></tr><tr><td>Energy [38]</td><td>56.37</td><td>60.13</td><td>50.55</td><td>24.92</td><td>54.68</td><td>53.99</td><td>59.38</td><td>58.51</td><td>51.94</td><td>34.63</td><td>54.58</td><td>46.44</td></tr><tr><td>MCM [41]</td><td>61.19</td><td>60.22</td><td>47.49</td><td>24.76</td><td>58.05</td><td>53.98</td><td>62.30</td><td>58.54</td><td>49.35</td><td>34.56</td><td>55.68</td><td>46.41</td></tr><tr><td>CAS</td><td>89.45</td><td>60.72</td><td>83.28</td><td>26.28</td><td>88.05</td><td>54.59</td><td>88.81</td><td>59.04</td><td>80.85</td><td>35.12</td><td>86.09</td><td>47.15</td></tr><tr><td rowspan="4">R-TPT [51]</td><td>Random</td><td>50.64</td><td>59.97</td><td>51.24</td><td>26.22</td><td>50.11</td><td>53.62</td><td>50.80</td><td>57.31</td><td>49.94</td><td>33.85</td><td>50.55</td><td>46.19</td></tr><tr><td>Energy [38]</td><td>56.59</td><td>60.16</td><td>48.67</td><td>25.87</td><td>55.71</td><td>54.12</td><td>59.42</td><td>57.58</td><td>52.91</td><td>34.02</td><td>54.66</td><td>46.35</td></tr><tr><td>MCM [41]</td><td>62.02</td><td>60.26</td><td>46.95</td><td>25.78</td><td>59.74</td><td>54.09</td><td>61.61</td><td>57.52</td><td>49.48</td><td>33.85</td><td>55.96</td><td>46.30</td></tr><tr><td>CAS</td><td>89.50</td><td>60.82</td><td>82.87</td><td>28.10</td><td>87.98</td><td>54.72</td><td>88.24</td><td>58.04</td><td>80.31</td><td>34.56</td><td>85.78</td><td>47.25</td></tr><tr><td rowspan="4">STS [6]</td><td>Random</td><td>50.26</td><td>59.23</td><td>49.08</td><td>28.64</td><td>49.89</td><td>53.08</td><td>50.52</td><td>57.10</td><td>50.39</td><td>34.36</td><td>50.03</td><td>46.48</td></tr><tr><td>Energy [38]</td><td>57.15</td><td>59.39</td><td>49.40</td><td>28.64</td><td>55.53</td><td>53.50</td><td>59.98</td><td>57.48</td><td>52.41</td><td>34.42</td><td>54.89</td><td>46.69</td></tr><tr><td>MCM [41]</td><td>61.99</td><td>59.33</td><td>47.42</td><td>28.41</td><td>58.74</td><td>53.34</td><td>61.37</td><td>57.18</td><td>47.87</td><td>34.11</td><td>55.48</td><td>46.47</td></tr><tr><td>CAS</td><td>91.43</td><td>59.72</td><td>83.29</td><td>31.56</td><td>89.70</td><td>53.96</td><td>89.36</td><td>57.64</td><td>83.22</td><td>34.91</td><td>87.40</td><td>47.56</td></tr><tr><td rowspan="4">ZERO [10]</td><td>Random</td><td>50.23</td><td>59.69</td><td>50.45</td><td>27.58</td><td>50.55</td><td>53.48</td><td>51.08</td><td>57.32</td><td>50.39</td><td>34.35</td><td>50.54</td><td>46.48</td></tr><tr><td>Energy [38]</td><td>56.24</td><td>59.84</td><td>48.72</td><td>27.37</td><td>54.36</td><td>53.86</td><td>59.58</td><td>57.58</td><td>52.05</td><td>34.42</td><td>54.19</td><td>46.61</td></tr><tr><td>MCM [41]</td><td>61.44</td><td>59.86</td><td>46.12</td><td>27.09</td><td>57.76</td><td>53.79</td><td>61.34</td><td>57.38</td><td>47.73</td><td>34.12 34.93</td><td>54.88</td><td>46.45</td></tr><tr><td>CAS</td><td>91.86</td><td>60.38</td><td>84.33</td><td>29.98</td><td>90.36</td><td>54.52</td><td>89.80</td><td>57.89</td><td>83.44</td><td></td><td>87.96</td><td>47.54</td></tr></table>

## C Results on Other TTA Methods

The main paper focuses on training-based TTA methods for CLIP [46]. To further examine the generality of CAS, we apply it to the standard model-based TTA method MEMO [75], as well as the training-free method MTA [72]. As shown in the following part, CAS remains effective under both methods, further demonstrating its broad applicability as a selective adaptation strategy.

## C.1 CAS for MEMO [75]

To further demonstrate the generality of our baseline, we incorporate CAS as a selective adaptation strategy into MEMO [75]. We evaluate it on CIFAR-10 [33] and CIFAR-10-C [23] with ResNet-26 [19], and on ImageNet-R [22] with ResNet-50 [19]. As summarized in Table 10, CAS consistently outperforms baseline strategies, including Random, Energy [38], and MCM [41], across both AUC and AEP metrics. On CIFAR-10, CAS achieves an AUC of 97.98%, significantly surpassing the second strongest baseline, MCM [41] (87.22%). A similar pattern is observed on CIFAR-10-C, where CAS again obtains the best results, achieving the highest AUC of 95.10% and AEP of 80.40%. On the more challenging ImageNet-R dataset, CAS maintains its advantage, achieving an AUC of 81.86%, outperforming the second-best method by 30.09%. These results further show that CAS generalizes well across different TTA frameworks, highlighting its broad applicability to selective adaptation.

Table 9: Performance comparison of different selection strategies under various TTA methods with RN50. We report AUC and AEP on fine-grained and downstream datasets. The best results under each TTA method are highlighted in bold.
<table><tr><td>Method</td><td>Strategy</td><td>Metric</td><td>Flow.</td><td>DTD</td><td>Pets</td><td>UCF</td><td>Cal.</td><td>Air.</td><td>Euro.</td><td>Cars</td><td>Food</td><td>SUN</td><td>Avg.</td></tr><tr><td rowspan="4">TPT [53]</td><td>Random</td><td>AUC AEP</td><td>49.05 62.15</td><td>47.29 40.92</td><td>50.85 84.09</td><td>47.07 59.99</td><td>49.54 87.18</td><td>47.08 16.77</td><td>49.70 26.66</td><td>48.99 57.43</td><td>50.23 74.67</td><td>51.13 60.59</td><td>49.09 57.05</td></tr><tr><td>Energy [38]</td><td>AUC AEP</td><td>56.70 62.00</td><td>58.58 41.42</td><td>59.40 83.99</td><td>51.22 60.16</td><td>65.13 88.02</td><td>46.87 16.81</td><td>18.87 21.93</td><td>51.02 57.53</td><td>62.63 74.85</td><td>56.24 60.71</td><td>52.67 56.74</td></tr><tr><td>MCM [41]</td><td>AUC AEP</td><td>59.77 62.10</td><td>62.59 41.37</td><td>65.40 84.07</td><td>57.53 60.11</td><td>76.43 88.09</td><td>41.22 16.38</td><td>31.60 24.62</td><td>56.96 57.77</td><td>70.62 74.89</td><td>63.18 60.86</td><td>58.53 57.03</td></tr><tr><td>CAS</td><td>AUC AEP</td><td>84.58 62.41</td><td>82.78 41.48</td><td>96.22 84.52</td><td>89.57 60.67</td><td>96.95 88.07</td><td>66.91 17.41</td><td>72.64 27.84</td><td>83.55 58.37</td><td>91.87 75.06</td><td>89.05 61.37</td><td>85.41 57.72</td></tr><tr><td rowspan="4">R-TPT [51]</td><td>Random</td><td>AUC AEP</td><td>47.50 61.44</td><td>46.52 40.53</td><td>49.78 83.88</td><td>46.79 59.08</td><td>52.54 86.08</td><td>48.74 17.06</td><td>49.63 21.86</td><td>49.29 57.22</td><td>50.38 73.64</td><td>51.10 60.19</td><td>49.23 56.10</td></tr><tr><td>Energy [38]</td><td>AUC AEP</td><td>56.27 61.03</td><td>56.38 40.86</td><td>60.95 83.74</td><td>52.54 59.21</td><td>64.51 86.55</td><td>47.27 17.13</td><td>18.30 16.05</td><td>51.37 57.18</td><td>62.53 73.56</td><td>55.95 60.23</td><td>52.61 55.55</td></tr><tr><td>MCM [41]</td><td>AUC AEP</td><td>60.56 61.05</td><td>63.16 41.04</td><td>66.79 83.79</td><td>60.06</td><td>77.07</td><td>40.93</td><td>31.47</td><td>57.62</td><td>71.52</td><td>63.39</td><td>59.26</td></tr><tr><td>CAS</td><td>AUC AEP</td><td>83.83 61.33</td><td>82.42 41.09</td><td>96.36</td><td>59.09 88.57</td><td>86.50 96.73</td><td>16.47 68.04</td><td>19.65 70.00</td><td>57.39 83.85</td><td>73.52 92.35</td><td>60.33 88.96</td><td>55.88 85.11</td></tr><tr><td rowspan="4">STS [6]</td><td>Random</td><td>AUC AEP</td><td>47.33 59.15</td><td>48.75</td><td>84.20 49.07</td><td>59.63 46.56</td><td>86.49 51.40</td><td>17.71 49.85</td><td>22.99 49.10</td><td>58.15 48.32</td><td>73.58 49.65</td><td>60.81 50.76</td><td>56.60 49.08</td></tr><tr><td>Energy [38]</td><td>AUC AEP</td><td>55.87</td><td>39.43 56.95</td><td>83.29 60.46</td><td>59.01 52.02</td><td>86.41 66.18</td><td>16.93 44.29</td><td>22.60 17.34</td><td>56.82 52.14</td><td>72.22 62.08</td><td>59.51 55.69</td><td>55.54 52.30</td></tr><tr><td>MCM [41]</td><td>AUC AEP</td><td>58.29 58.82 58.19</td><td>39.58 62.36</td><td>83.06 65.82</td><td>59.20 60.21</td><td>87.02 78.22</td><td>16.66 38.00</td><td>16.70 31.86</td><td>56.94 58.45</td><td>71.91 69.66</td><td>59.51 62.20</td><td>54.89 58.56</td></tr><tr><td>CAS</td><td>AUC AEP</td><td>89.97 58.29</td><td>39.61 87.75 39.44</td><td>83.05 96.76 83.38</td><td>59.00 90.22 59.50</td><td>86.98 97.08 86.80</td><td>16.18 72.34 17.47</td><td>20.28 69.96</td><td>56.98 88.03</td><td>71.64 93.30</td><td>59.54 91.00 59.82</td><td>55.15 87.64 55.70</td></tr><tr><td rowspan="4">ZERO [10]</td><td>Random</td><td>AUC AEP</td><td>47.78 59.78</td><td>50.69 39.41</td><td>48.81 83.78</td><td>45.69 58.77</td><td>50.49 86.21</td><td>49.58</td><td>23.25 49.56</td><td>57.53 48.55</td><td>71.53 50.31</td><td>50.76</td><td>49.22 55.75</td></tr><tr><td>Energy [38]</td><td>AUC AEP</td><td>57.17 59.16</td><td>56.79 39.56</td><td>61.01 83.81</td><td>50.89 58.98</td><td>66.40 86.85</td><td>16.98 46.70 16.94</td><td>22.44 18.70 16.81</td><td>57.21 52.48 57.38</td><td>72.83 62.30 72.62</td><td>60.04 55.20 60.05</td><td>52.76 55.22</td></tr><tr><td>MCM [41]</td><td>AUC AEP</td><td>59.51 58.99</td><td>60.18 39.59</td><td>66.29 83.80</td><td>59.24 58.83</td><td>77.86 86.79</td><td>38.34 16.40</td><td>32.36 20.17</td><td>57.80 57.44</td><td>70.24 72.43</td><td>61.84 60.12</td><td>58.37 55.46</td></tr><tr><td>CAS</td><td>AUC AEP</td><td>89.82 59.15</td><td>85.65 39.37</td><td>96.34 84.14</td><td>90.46 59.25</td><td>97.05 86.57</td><td>73.41 17.55</td><td>71.15 23.49</td><td>87.16 58.10</td><td>93.19 72.38</td><td>91.12 60.52</td><td>87.54 56.05</td></tr></table>

To evaluate the trade-off between efficiency and accuracy, we plot the accuracyskip ratio curves of different selection strategies under MEMO [75], which is illustrated in Fig. 7. CAS consistently outperforms Random, MCM, and Energy across almost the entire skip range, maintaining accuracy close to the fulladaptation baseline even when substantial samples are skipped. By contrast, the other strategies show much larger performance drops as the skip ratio increases. These results indicate that CAS identifies samples whose adaptation can be skipped more reliably, leading to better efficiency without sacrificing accuracy.

## C.2 CAS for MTA [72]

Performance on ImageNet and its variants. We evaluate the performance of CAS under the training-free method MTA [72] framework on ImageNet and its variants, as summarized in Table 11. CAS consistently outperforms all selection strategies across all evaluation metrics. Specifically, CAS achieves an average AUC of 93.36%, providing a substantial improvement over the second strongest baseline, MCM, which attains an AUC of 59.80%. On the challenging ImageNet-A dataset, CAS reaches an AUC of 92.12%, demonstrating its strong capability in identifying ineffective adaptations even in training-free scenarios. Across all evaluated variants, its performance remains highly stable, highlighting its ability to effectively distinguish beneficial updates from harmful ones.

Table 10: Comparison of AUC and AEP under MEMO [75] with different skipping strategies on CIFAR-10 and ImageNet-R. Bold numbers indicate the best performance.
<table><tr><td>Dataset</td><td>Metric</td><td>Random</td><td>MCM</td><td>Energy</td><td>CAS</td></tr><tr><td colspan="6">Backbone: ResNet-26 [19]</td></tr><tr><td>CIFAR-10</td><td>AUC</td><td>52.12</td><td>87.22</td><td>84.38</td><td>97.98</td></tr><tr><td rowspan="3">CIFAR-10-C</td><td>AEP</td><td>92.09</td><td>92.55</td><td>92.51</td><td>92.66</td></tr><tr><td>AUC</td><td>50.24</td><td>76.52</td><td>73.69</td><td>95.10</td></tr><tr><td>AEP</td><td>79.43</td><td>79.94</td><td>79.84</td><td>80.40</td></tr><tr><td colspan="2">Backbone: ResNet-50 [19]</td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="2">ImageNet-R</td><td>AUC</td><td>49.93</td><td>51.77</td><td>50.52</td><td>81.86</td></tr><tr><td>AEP</td><td>39.59</td><td>39.73</td><td>39.74</td><td>41.08</td></tr></table>

![](images/f64a4fb3ce320ef09aaec5120b6b96e366893435049fc2f0ea572f08574a8be6.jpg)  
(a) CIFAR-10

![](images/38ea1bb2d740b4d7794f0ab5bb5d00af7756b1181c0153267366e5d8d2ef0936.jpg)  
(b) CIFAR-10-C

![](images/9b1822dfa430aba031710a55f13c0c168487527fa6fe0712467e66f651e793e9.jpg)  
(c) ImageNet-R  
Fig. 7: Accuracy (%) versus skip ratio (%) under different skipping strategies for MEMO. CAS consistently maintains higher accuracy even when skipping a large proportion of adaptation processes. Compared with Random, MCM, and Energy, CAS demonstrates a superior efficiency-accuracy trade-off.

Performance on fine-grained datasets. We further evaluate CAS on finegrained datasets under MTA [72], with results summarized in Table 12. CAS consistently achieves the highest average performance across all datasets. Under the MTA framework, CAS attains a peak average AUC of 92.30%, significantly surpassing MCM (66.50%) and Energy (61.03%) by a wide margin. Notably, CAS demonstrates robustness across different domains, achieving a high AUC of 99.13% on Caltech101 and 97.85% on Pets. Overall, CAS delivers consistently robust performance across fine-grained classification tasks, demonstrating its superior trade-off between efficiency and accuracy.

Table 11: Performance comparison of different selection strategies under the trainingfree MTA framework on ImageNet and its variants. We report AUC and AEP. The best results are highlighted in bold.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Strategy</td><td colspan="2">ImageNet</td><td colspan="2">ImageNet-A</td><td colspan="2">ImageNet-V</td><td colspan="2">ImageNet-R</td><td colspan="2">ImageNet-K</td><td colspan="2">Avg.</td></tr><tr><td>AUC↑</td><td>AEP↑</td><td>AUC↑</td><td>AEP↑</td><td>AUC↑</td><td>AEP↑</td><td>AUC↑</td><td>AEP↑</td><td>AUC↑</td><td>AEP↑</td><td>AUC↑</td><td>AEP↑</td></tr><tr><td rowspan="4">MTA</td><td>Random</td><td>50.85</td><td>68.44</td><td>49.20</td><td>53.84</td><td>51.71</td><td>62.79</td><td>49.05</td><td>75.93</td><td>50.45</td><td>47.73</td><td>50.25</td><td>61.75</td></tr><tr><td>Energy [38]</td><td>57.43</td><td>68.64</td><td>50.51</td><td>54.18</td><td>57.52</td><td>62.99</td><td>63.11</td><td>76.47</td><td>53.06</td><td>47.79</td><td>56.33</td><td>62.01</td></tr><tr><td>MCM [41]</td><td>63.98</td><td>68.74</td><td>50.51</td><td>54.00</td><td>62.57</td><td>63.04</td><td>69.18</td><td>76.55</td><td>52.77</td><td>47.73</td><td>59.80</td><td>62.01</td></tr><tr><td>CAS</td><td>95.08</td><td>69.24</td><td>92.12</td><td>56.83</td><td>94.01</td><td>63.58</td><td>95.59</td><td>76.96</td><td>89.98</td><td>48.47</td><td>93.36</td><td>63.02</td></tr></table>

Table 12: Performance comparison of different selection strategies under the trainingfree MTA framework on fine-grained and downstream datasets. We report AUC and AEP. The best results are highlighted in bold.
<table><tr><td>Method</td><td>Strategy</td><td>Metric</td><td>Flow.</td><td>DTD</td><td>Pets</td><td>UCF</td><td>Cal.</td><td>Air.</td><td>Euro.</td><td>Cars</td><td>Food</td><td>SUN</td><td>Avg.</td></tr><tr><td rowspan="6">MTA</td><td rowspan="2">Random</td><td>AUC</td><td>50.46 67.41</td><td>51.68</td><td>55.67 88.11</td><td>49.73</td><td>48.59</td><td>53.04</td><td>49.88</td><td>50.74</td><td>50.38</td><td>50.09</td><td>51.03</td></tr><tr><td>AEP</td><td></td><td>45.32</td><td></td><td>66.75</td><td>94.18</td><td>24.60</td><td>42.35</td><td>67.01</td><td>84.19</td><td>64.36</td><td>64.43</td></tr><tr><td rowspan="3">Energy [38]</td><td>AUC</td><td>60.38</td><td>54.51</td><td>67.94</td><td>60.63</td><td>69.36</td><td>47.94</td><td>67.74</td><td>54.43</td><td>68.64</td><td>58.68</td><td>61.03</td></tr><tr><td>AEP</td><td>67.35</td><td>45.42</td><td>87.99</td><td>67.17</td><td>94.47</td><td>24.34</td><td>42.82</td><td>67.07</td><td>84.36</td><td>64.65</td><td>64.56</td></tr><tr><td>AUC</td><td>63.78</td><td>67.73</td><td>73.06</td><td>68.02</td><td>79.01</td><td>47.91</td><td>58.97</td><td>63.21</td><td>78.63</td><td>64.64</td><td>66.50</td></tr><tr><td rowspan="3">MCM [41]</td><td>AEP</td><td>67.40</td><td>45.71</td><td>87.97</td><td>67.19</td><td>94.41</td><td>24.21</td><td>42.71</td><td>67.28</td><td>84.38</td><td>64.78</td><td></td><td>64.60</td></tr><tr><td></td><td>93.62</td><td></td><td></td><td>94.15</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>AUC AEP</td><td>67.52</td><td>90.97 45.91</td><td>97.85 87.94</td><td>67.55</td><td></td><td>99.13 94.32</td><td>79.32 24.66</td><td>82.80 42.56</td><td>93.14 67.60</td><td>97.36 84.43</td><td>94.67 65.21</td><td>92.30 64.77</td></tr></table>

## D More Baselines

We explored additional baseline strategies, including Max Logits, Max Softmax, and GL-MCM [42]. However, we only reported Energy [38] and MCM [41] in the main text due to their comparable performance and space constraints. Table 13 summarizes the detailed results for these previously evaluated baselines. Specifically, CAS achieves an average AUC of 89.42%, outperforming the second-best baseline by a margin of 22.63%. Furthermore, CAS yields an average AEP of 62.44%, further validating its superior capability in maintaining TTA performance in the selective adaptation problem.

Table 13: Performance comparison of different baselines under TPT [53] with ViT-B/16. We report AUC and AEP on ImageNet and its variants.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Strategy</td><td colspan="2">ImageNet</td><td colspan="2">ImageNet-A</td><td colspan="2">ImageNet-V</td><td colspan="2">ImageNet-R</td><td colspan="2">ImageNet-K</td><td colspan="2">Avg.</td></tr><tr><td>AUC↑</td><td>AEP↑</td><td>AUC↑</td><td>AEP↑</td><td>AUC↑</td><td>AEP↑</td><td>AUC↑</td><td>AEP↑</td><td>AUC↑</td><td>AEP↑</td><td>AUC↑ AEP↑</td><td></td></tr><tr><td rowspan="4">TPT [53]</td><td>Max Logits</td><td>59.24</td><td>68.44</td><td>52.71</td><td>52.69</td><td>59.20</td><td>62.96</td><td>65.85</td><td>76.67</td><td>54.94</td><td>47.42</td><td>58.39</td><td>61.64</td></tr><tr><td>Max Softmax</td><td>72.85</td><td>68.64</td><td>57.47</td><td>52.88</td><td>68.85</td><td>63.02</td><td>75.07</td><td>76.75</td><td>59.72</td><td>47.38</td><td>66.79</td><td>61.73</td></tr><tr><td>GL-MCM [42]</td><td>62.67</td><td>68.50</td><td>56.82</td><td>53.03</td><td>60.90</td><td>62.94</td><td>68.23</td><td>76.69</td><td>54.29</td><td>47.47</td><td>60.58</td><td>61.73</td></tr><tr><td>CAS</td><td>91.10</td><td>69.00</td><td>88.62</td><td>54.70</td><td>89.69</td><td>63.45</td><td>93.45</td><td>77.14</td><td></td><td>84.24 47.93 8</td><td></td><td>89.42 62.44</td></tr></table>

## E Different VLMs

To demonstrate the generalizability of our approach across different VLMs, we further report the performance of CAS on the SigLIP [73] backbone. As shown in Table 14, CAS consistently maintains strong detection capabilities, achieving an average AUC of 93.03% across ImageNet and its OOD variants.

Table 14: Performance comparison of different VLMs under TPT [53] framework. We report AUC and AEP on ImageNet and its variants.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Strategy</td><td colspan="2">ImageNet</td><td colspan="2">ImageNet-A</td><td colspan="2">ImageNet-V</td><td colspan="2">ImageNet-R</td><td colspan="2">ImageNet-K</td><td colspan="2">Avg.</td></tr><tr><td>AUC↑</td><td>AEP↑</td><td>AUC↑</td><td>AEP↑</td><td>AUC↑</td><td>AEP↑</td><td>AUC↑</td><td>AEP↑</td><td>AUC↑</td><td>AEP↑</td><td>AUC↑</td><td>AEP↑</td></tr><tr><td rowspan="4">SigLIP [73]</td><td>Random</td><td>50.77</td><td>76.23</td><td>50.05</td><td>46.15</td><td>51.00</td><td>68.98</td><td>50.54</td><td>89.73</td><td>48.30</td><td>66.89</td><td>50.13</td><td>69.60</td></tr><tr><td>Energy [38]</td><td>60.20</td><td>76.26</td><td>51.56</td><td>46.17</td><td>57.60</td><td>69.00</td><td>61.34</td><td>89.75</td><td>53.42</td><td>66.94</td><td>56.82</td><td>69.63</td></tr><tr><td>MCM [41]</td><td>67.16</td><td>76.32</td><td>51.77</td><td>46.20</td><td>62.48</td><td>69.07</td><td>74.68</td><td>89.82</td><td>62.48</td><td>66.99</td><td>63.71</td><td>69.68</td></tr><tr><td>CAS</td><td>95.45</td><td>76.46</td><td>85.78</td><td>46.70</td><td>93.30</td><td>69.23</td><td>97.12</td><td>89.90</td><td>93.52</td><td>67.07</td><td>93.03</td><td>69.87</td></tr></table>

## F Impact of the Number of Test-Time Augmentations

To assess the sensitivity of CAS to the number of test-time augmentations, we examine how its performance changes under different augmentation budgets. While existing TTA methods typically rely on a default of 64 views to ensure stable prediction, this high volume incurs significant computational overhead. In this study, we evaluate the stability of CAS by reducing the number of augmentation views from the original 64 to smaller values of 6. Figure 8 illustrates the impact of view reduction on AUC and AEP across various datasets. We observe that while performance initially increases with the number of views, it rapidly plateaus at a relatively low view count. Notably, both AUC and AEP remain largely stable even when using fewer augmentations compared to the default setting. This trend indicates that the proposed selective adaptation mechanism is highly robust. It does not rely on an excessive augmentation budget to maintain performance.

## G Adaptation Behaviors across Different TTA Methods

To examine the similarity of selective adaptation behaviors across different TTA methods, we analyze the detection ground truth of effective and ineffective cases in TPT [53], R-TPT [52], STS [6], and ZERO [10] using Hamming distance. For each method, Wrong to Correct is labeled as beneficial (0), while Correct to Correct, Wrong to Wrong, and Correct to Wrong are treated as ineffective (1). Given two methods $M _ { 1 }$ and $M _ { 2 }$ , the Hamming similarity is defined as the percentage of test samples for which the two methods assign the same binary label. The corresponding Hamming Similarity is defined as 1 – Distance. As shown in Fig. 9, the Hamming similarity across all evaluated benchmarks consistently exceeds

![](images/97c2a3b3d3a3a2fe17ca482f8e77797eaf8119a3f61b3b9e2637553ffdaaeff1.jpg)

(a) ImageNet  
![](images/bfc197f3be500f88dce44896cc83a11c5ead7ef226cad090808c8328b47f4307.jpg)

![](images/cc9dcec78f48d9c64ad5d3c78e1504f1ff1be580d1ff9e48ab3be00d3a009d2b.jpg)

![](images/b51f2fbae8eac41b51b6eea82ea4dab646391b39ce06d97f36b5a6f458f30631.jpg)  
(b) ImageNet-A

![](images/c3d02b34cbb1d2223b7ef77224abf4370ef041a1424a3b5d40cba10cb7a07c8a.jpg)

![](images/5e7a826c414162c60f117ba1333ec3efc89017372ef41a2806c7d6b014a8a910.jpg)  
(c) ImageNet-V2

![](images/c442cd93cc2a7bf99628e347e201967da3e5d1431c0f7dcd563d697f68fab7a2.jpg)

![](images/019f538c98fa9e58e3b5ba04bfc53228ec68287f22980fb0f11c66424dc8f1b7.jpg)  
(d) ImageNet-R

![](images/807db749ef2dcc240ee2d31939dccf95b855986af4fbb024e9ea1313fbda6aef.jpg)

![](images/14d1254fa96c203f04b0dc6a0797198e48cdc4c493af533c5a5e312a424661b1.jpg)  
(e) ImageNet-Sketch

![](images/ca7e9e1b79a7c3ecd33327facffc880268821cd13446bea230507e512bd06de7.jpg)

![](images/1042e19cb8e4e722a3e71fc95a7f41c83b0237d628f2d23a618a27686a131bd7.jpg)  
(f) Average  
Fig. 8: Impact of reducing augmentation views on ImageNet and its variants. Even when using substantially fewer augmentations than the default 64 views, both AUC and AEP remain largely stable across datasets, indicating that our method does not rely on excessive augmentations.

90%, indicating that different TTA methods exhibit highly consistent selective adaptation behaviors. In particular, the ineffective cases are largely shared across methods, meaning that most samples are consistently identified as not requiring adaptation. This observation suggests that the necessity of adaptation is largely determined by the sample itself rather than the specific TTA algorithm. Consequently, many adaptation operations performed by existing methods are redundant, highlighting the importance of selectively applying adaptation only to samples that can truly benefit from it.

## H Statistics of Adaptation Cases

To further understand the behavior of test-time adaptation, we analyze the distribution of four adaptation cases across different datasets under TPT. Figure 10 presents the distribution of four adaptation cases across 15 benchmarks. Cases #A (Correct to Correct) and #B (Wrong to Wrong) are categorized as negligible cases, where adaptation does not change the prediction outcome. Case #C

![](images/f579d65a39e5282de6f1242fabd603b43ab0ad38365b3a833a7e35e0c3466673.jpg)  
(a) All datasets

![](images/5f08d33592da43c45233ee64f3512687413af3a1acdc33f0fc894df749dc39bc.jpg)  
(b) ImageNet

![](images/6c3d6a4f6cfe7e71464af343cb51e580bd7515bce077def7b166c04f1e892ffc.jpg)  
(c) ImageNet-A

![](images/bc88fd1a138923b52b5b2e0af7a813db6295760288ee995cf44d6eaa403d87cc.jpg)  
(d) ImageNet-R

![](images/8fbfef995ac4088ce369134a0a38fb0e7b240b425f10987109d7bfdc5a43e1fe.jpg)  
(e) ImageNet-V

![](images/48885527a201d5ac28b3c81a748ab82f2542b2670551760f2397f7f3fe7fde5a.jpg)  
(f) ImageNet-K

![](images/d4cd45d78633de059614e4d664d433e0c623599af358785838c707e253b6eb33.jpg)  
(g) Aircraft

![](images/25f4cc8230df796c9e98a03faa12110c5c8f2e880064079576a6d3ecec2da14b.jpg)

![](images/67757ba3591473f5f2da23fb9fc7cbf019fcba8ab6f99d65e8597d20ce746c9c.jpg)

![](images/340be635d2d708606ff67e4c8aa7fccb5b70c37a687531da63e6e56599f9807c.jpg)  
(i) Caltech101  
(j) DTD

(h) Cars  
![](images/3e57c30d3421fd8d7b08b0b1a7fc61b2fc033660d3e1e2fc2c76477e911ca728.jpg)

![](images/576aaf0f14b5aaf213dfc67d5d02c965f010bc5441003d501966c3cbba0ff40f.jpg)  
(k) EuroSAT  
(1) Flower102

![](images/2bf2aaa228def87cca4a736874bb873dab1524f538dba07895a31db98bd0e00b.jpg)  
(m) Food101

![](images/8e3c1786ccf0f9e6f7d91b444cef1f848da15b7fc1cc1f447e702dc09a803bf5.jpg)  
(n) Pets

![](images/994985cfdfcf2a2850e42112875cf92b705cc6e7b6ccb6ee0e923b6d914049da.jpg)  
(o) SUN397

![](images/9be2fd3ee52252132d8b6ccb44d5f5cf00dadc94e2a4c407071346b627401e97.jpg)  
(p) UCF101

Fig. 9: The figure presents the Hamming similarity of effective and ineffective adaptation cases across different TTA methods. Higher similarity indicates stronger agreement among methods on whether a sample requires adaptation.

(Correct to Wrong) represents harmful adaptation, while Case #D (Wrong to Correct) corresponds to beneficial adaptation. Notably, only Case #D reflects genuinely effective adaptation, whereas the remaining three cases are ineffective and do not require adaptation. As shown in the figure, the majority of samples fall into the negligible cases (#A and #B) across all datasets, while harmful and beneficial transitions occur only in a small fraction of instances. In particular, beneficial adaptations account for only a small percentage, typically around 5% or lower, indicating that most adaptation processes are unnecessary and contribute little to performance improvement.

![](images/f70db81cc3ea2ceffaedb01297f4888c3b79a0d4342b3fbd7c80bd08a4fda680.jpg)  
(a) ImageNet

![](images/e5da81ac1283cc5694ef59fac5fec941ab011f483bdd0c0c7369aa8886601f9a.jpg)  
(b) ImageNet-A

![](images/354deb19da7f75afe99fb473eb27ba373b7b45b7f2c5bb12ad8358a7d31f5832.jpg)  
(c) ImageNet-K

![](images/b4a396f493f465432e2888a28bcf98ecaa5bcc84365973f4af86d378851d7b79.jpg)  
(d) ImageNet-R

![](images/1f1d87ecbec6caeb9cb26bcbe22dbdcf7aedda38b33aa71362aba1605128ab03.jpg)  
(e) ImageNet-V

![](images/ddd08e7c17ed4d556a215049428672bcb04708cd5e0e35084fa20ff3943c9822.jpg)  
(f) Aircraft

![](images/fb04e293ecba61acdf596aca5abe5ea410ba4e40588b9a7a779a3977865de164.jpg)  
(g) Caltech101

![](images/8eb77a28857370da65d85438ce59458a38114e0c85f3e465a2ad1f433790fcf5.jpg)

![](images/77fe681597540b2c4bb1217bdbc62f29d4862600be3f98bbb1fa42759c5e4a6b.jpg)

![](images/9af9302b6165d333003f750d2235b1f131c365882f9e38833a0d3fe5c773ca32.jpg)  
(i) DTD

(h) Cars  
![](images/a3f2a8963df10303415bc1abf743797c80270a010c299f2c182badea52822801.jpg)  
(j) EuroSAT

![](images/0363f6408f563ac5c57c7aa02973f309765b087cc58ae3e74eb4ea2b038eebd0.jpg)

![](images/78c856d36391c737bc86ca6a380dc0eaf11d4803e2a6aa5cb83ac40a6c018ada.jpg)  
(m) Pets

(1) Food101  
(k) Flowers102  
![](images/539d676562f438c203b5a44049f38b975e2d93954345a1cabbdfb0546faa92fa.jpg)  
(n) SUN397

![](images/211643139dca37c92591b7ab8d9b4657922aca9e5eebd38559728307f93dd0db.jpg)  
(o) UCF101

![](images/a1ab89f44d3be843e4c4379ab85ace13e13b5e39b8f90163437bbf0c2fcf2587.jpg)  
(p) Average

Fig. 10: The figure illustrates the distribution of four adaptation cases across various datasets under TPT [53], including #A (Correct to Correct), #B (Wrong to Wrong), $\# \mathrm { C }$ (Correct to Wrong), and #D (Wrong to Correct). The percentage of the beneficial case #D is highlighted.

![](images/58d148c8cc2f8952eb4538df473cce33499491aacc01fcf76e28dee51aa27ad6.jpg)  
(a) ImageNet

![](images/fca770c3e0a4c862285800699a28f5d75c6234846a9409df4221e7484f7b72dc.jpg)  
(b) ImageNet

![](images/e89793ad02d41b4c773f85159c1b72e4fdc3788aca14c94067c53efccd9bc08c.jpg)  
(c) DTD

![](images/f701fe603a3c3a1e75124e046d00c231bbc19aa3107c3ac44271527ed431af3b.jpg)  
(d) DTD

![](images/122bd88a1a4dee4ca005c679ad5950bd7d2bb484c20036fe8650af28b8d58dfb.jpg)  
(e) Caltech101

![](images/d75d3f14e62266deeef64dbb2e690c33bcca15477fa88e60993c6cae0bc4c333.jpg)  
(f) Caltech101  
Fig. 11: This figure shows CAS distribution for different adaptation cases. The top row presents the distributions of negligible, harmful, and beneficial cases, while the bottom row groups them into ineffective and effective updates.

## I CAS Distribution

To examine whether CAS can identify adaptation effectiveness, we visualize the distribution of CAS values for different cases across multiple datasets. As shown in Fig. 11, the CAS values exhibit clear separation among the three categories. Beneficial cases (i.e., effective adaptations) are concentrated around small CAS values, whereas negligible and harmful cases (i.e., ineffective adaptations) tend to produce significantly larger CAS values. In particular, harmful and negligible updates show similar distributions with peaks at high CAS regions, indicating that ineffective adaptations generally correspond to large CAS scores. This consistent pattern across datasets suggests that CAS serves as an effective indicator to distinguish effective and ineffective adaptation cases.

Table 15: Standard deviations comparison of different selection strategies under various TTA methods with ViT-B/16 on ImageNet and its variants.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Strategy</td><td colspan="2">ImageNet</td><td colspan="2">ImageNet-A</td><td colspan="2">ImageNet-V</td><td colspan="2">ImageNet-R</td><td colspan="2">ImageNet-K</td><td colspan="2">Avg.</td></tr><tr><td>AUC↓</td><td>AEP↓</td><td>AUC↓</td><td>AEP↓</td><td>AUC↓</td><td>AEP↓</td><td>AUC↓</td><td>AEP↓</td><td>AUC↓</td><td>AEP↓</td><td>AUC↓</td><td>AEP↓</td></tr><tr><td rowspan="4">TPT [53]</td><td>Random</td><td>0.0272</td><td>0.0000</td><td>0.0240</td><td>0.0006</td><td>0.7056</td><td>0.0066</td><td>0.0121</td><td>0.0006</td><td>0.1849</td><td>0.0015</td><td>0.1908</td><td>0.0019</td></tr><tr><td>Energy [38]</td><td>0.0016</td><td>0.0001</td><td>0.2401</td><td>0.0090</td><td>0.3025</td><td>0.0083</td><td>0.0441</td><td>0.0000</td><td>0.0625</td><td>0.0012</td><td>0.1302</td><td>0.0037</td></tr><tr><td>MCM [41]</td><td>0.0064</td><td>0.0001</td><td>0.3136</td><td>0.0053</td><td>0.4489</td><td>0.0000</td><td>0.0506</td><td>0.0018</td><td>0.1980</td><td>0.0001</td><td>0.2035</td><td>0.0015</td></tr><tr><td>CAS</td><td>0.0132</td><td>0.0000</td><td>0.0169</td><td>0.0002</td><td>0.1156</td><td>0.0012</td><td>0.0012</td><td>0.0013</td><td>0.0156</td><td>0.0004</td><td>0.0325</td><td>0.0006</td></tr><tr><td rowspan="4">R-TPT [51]</td><td>Random</td><td>0.0342</td><td>0.0012</td><td>0.0210</td><td>0.0159</td><td>0.1056</td><td>0.0002</td><td>0.0144</td><td>0.0003</td><td>0.0002</td><td>0.0008</td><td>0.0351</td><td>0.0037</td></tr><tr><td>Energy [38]</td><td>0.0380</td><td>0.0000</td><td>0.2401</td><td>0.0003</td><td>0.0441</td><td>0.0021</td><td>0.0049</td><td>0.0000</td><td>0.0004</td><td>0.0005</td><td>0.0655</td><td>0.0006</td></tr><tr><td>MCM [41]</td><td>0.0272</td><td>0.0005</td><td>0.1024</td><td>0.0261</td><td>0.0090</td><td>0.0062</td><td>0.0156</td><td>0.0000</td><td>0.0001</td><td>0.0036</td><td>0.0309</td><td>0.0073</td></tr><tr><td>CAS</td><td>0.0002</td><td>0.0002</td><td>0.0196</td><td>0.0240</td><td>0.0020</td><td>0.0042</td><td>0.0036</td><td>0.0000</td><td>0.0225</td><td>0.0034</td><td>0.0096</td><td>0.0064</td></tr><tr><td rowspan="4">STS [6]</td><td>Random</td><td>0.0100</td><td>0.0019</td><td>0.0049</td><td>0.0000</td><td>0.0870</td><td>0.0001</td><td>0.0256</td><td>0.0009</td><td>0.0225</td><td>0.0006</td><td>0.0300</td><td>0.0007</td></tr><tr><td>Energy [38]</td><td>0.0006</td><td>0.0022</td><td>0.0110</td><td>0.0000</td><td>0.0006</td><td>0.0000</td><td>0.0182</td><td>0.0001</td><td>0.0064</td><td>0.0001</td><td>0.0074</td><td>0.0005</td></tr><tr><td>MCM [41]</td><td>0.0000</td><td>0.0018</td><td>0.0004</td><td>0.0013</td><td>0.0064</td><td>0.0006</td><td>0.0090</td><td>0.0010</td><td>0.0156</td><td>0.0000</td><td>0.0063</td><td>0.0009</td></tr><tr><td>CAS</td><td>0.0012</td><td>0.0018</td><td>0.0020</td><td>0.0028</td><td>0.0240</td><td>0.0001</td><td>0.0000</td><td>0.0004</td><td>0.0169</td><td>0.0002</td><td>0.0088</td><td>0.0011</td></tr><tr><td rowspan="4">ZERO [10]</td><td>Random</td><td>0.0100</td><td>0.0050</td><td>0.0030</td><td>0.0001</td><td>0.0090</td><td>0.0023</td><td>0.0380 0.0121</td><td>0.0024 0.0019</td><td>0.0342 0.0225</td><td>0.0003 0.0001</td><td>0.0188</td><td>0.0020 0.0029</td></tr><tr><td>Energy [38]</td><td>0.0072 0.0000</td><td>0.0051 0.0056</td><td>0.0081</td><td>0.0040</td><td>0.0729</td><td>0.0032</td><td>0.0001</td><td>0.0033</td><td>0.0016</td><td>0.0002</td><td>0.0246</td><td></td></tr><tr><td>MCM [41]</td><td>0.0000</td><td>0.0082</td><td>0.0110 0.0342</td><td>0.0001 0.0019</td><td>0.0506 0.0006</td><td>0.0025 0.0030</td><td>0.0012</td><td>0.0052</td><td>0.0016</td><td>0.0002</td><td>0.0127 0.0075</td><td>0.0023 0.0037</td></tr><tr><td>CAS</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 16: Standard deviations comparison of different selection strategies under various TTA methods with ViT-B/16 on fine-grained datasets.
<table><tr><td>Method</td><td>Strategy</td><td>Metric</td><td>Flow.</td><td>DTD</td><td>Pets</td><td>UCF</td><td>Cal.</td><td>Air.</td><td>Euro.</td><td>Cars</td><td>Food</td><td>SUN</td><td>Avg.</td></tr><tr><td rowspan="4">TPT [53]</td><td>Random</td><td>AUC AEP</td><td>0.0342 0.0005</td><td>0.1482 0.0008</td><td>0.0182 0.0009</td><td>0.4624 0.0300</td><td>0.0009 0.0041</td><td>1.0000 0.0007</td><td>0.0380 0.0025</td><td>0.1369 0.0018</td><td>0.0042 0.0000</td><td>0.0100 0.0020</td><td>0.1853 0.0043</td></tr><tr><td>Energy [38]</td><td>AUC AEP</td><td>0.1640 0.0001</td><td>0.1056 0.0005</td><td>0.0552 0.0001</td><td>0.0306 0.0699</td><td>4.6440 0.0002</td><td>0.0484 0.0280</td><td>0.0420 0.0014</td><td>1.8632 0.0117</td><td>0.1122 0.0000</td><td>0.0225 0.0001</td><td>0.7088 0.0112</td></tr><tr><td>MCM [41]</td><td>AUC AEP</td><td>0.0484 0.0017</td><td>0.7656 0.0108</td><td>1.1990 0.0000</td><td>0.0306 0.0772</td><td>3.2942 0.0021</td><td>1.2656 0.0046</td><td>0.0306 0.0004</td><td>0.0240 0.0001</td><td>0.1444 0.0001</td><td>0.0702 0.0002</td><td>0.6873 0.0097</td></tr><tr><td>CAS</td><td>AUC AEP</td><td>0.6889 0.0011</td><td>0.4290 0.0022</td><td>0.3969 0.0019</td><td>0.7482 0.0594</td><td>0.0306 0.0116</td><td>0.6724 0.0207</td><td>0.0009 0.0029</td><td>0.0900 0.0047</td><td>0.0004 0.0001</td><td>0.0016 0.0004</td><td>0.3059 0.0105</td></tr><tr><td rowspan="4">R-TPT [51]</td><td>Random</td><td>AUC AEP</td><td>0.7921 0.0020</td><td>1.1342 0.0179</td><td>0.0144 0.0002</td><td>0.0784 0.0537</td><td>0.0240 0.0011</td><td>2.5760 0.0001</td><td>0.0324 0.0293</td><td>0.3844 0.0005</td><td>0.0441 0.0004</td><td>0.0600 0.0012</td><td>0.5140 0.0106</td></tr><tr><td>Energy [38]</td><td>AUC AEP</td><td>0.0400 0.0084</td><td>0.0420 0.0522</td><td>0.7744 0.0000</td><td>0.1444 0.0034</td><td>0.6480 0.0020</td><td>0.0009 0.0098</td><td>0.0072 0.0336</td><td>0.0169 0.0145</td><td>0.0784 0.0018</td><td>0.0042 0.0015</td><td>0.1756 0.0127</td></tr><tr><td>MCM [41]</td><td>AUC AEP</td><td>0.5852 0.0044</td><td>1.5500 0.1001</td><td>0.8372 0.0000</td><td>0.0009 0.0144</td><td>2.8900 0.0029</td><td>0.6480 0.0392</td><td>0.0361 0.0263</td><td>0.2209 0.0161</td><td>0.2756 0.0032</td><td>0.0121 0.0009</td><td>0.7056 0.0208</td></tr><tr><td>CAS</td><td>AUC AEP</td><td>0.6889 0.0098</td><td>0.7832 0.0651</td><td>0.0210 0.0002</td><td>0.1521 0.0325</td><td>0.0006</td><td>5.7121</td><td>0.0012</td><td>0.0110</td><td>0.0006</td><td>0.0016</td><td>0.7372</td></tr><tr><td rowspan="4">STS [6]</td><td>Random</td><td>AUC</td><td>0.5041</td><td>1.4762</td><td>0.7744</td><td>0.3721</td><td>0.0001 0.0870</td><td>0.0930 2.2350</td><td>0.0218 0.2209</td><td>0.0072 0.6006</td><td>0.0023 0.0784</td><td>0.0018 0.0992</td><td>0.0234 0.6448</td></tr><tr><td>Energy [38]</td><td>AEP AUC</td><td>0.0075 0.0016</td><td>0.0000 0.0030</td><td>0.0044 0.0169</td><td>0.0079 1.7161</td><td>0.0008 0.0576</td><td>0.0687 0.2862</td><td>0.0011 0.0144</td><td>0.0003 0.0182</td><td>0.0005 0.0182</td><td>0.0013 0.0144</td><td>0.0093 0.2147</td></tr><tr><td>MCM [41]</td><td>AEP AUC</td><td>0.0228 0.0961</td><td>0.0056 1.3806</td><td>0.0108 0.0002</td><td>0.0881 0.4970</td><td>0.0178 0.9216</td><td>0.1492 1.2432</td><td>0.0001 0.0420</td><td>0.0024 0.1521</td><td>0.0001 0.0400</td><td>0.0022 0.0042</td><td>0.0299 0.4377</td></tr><tr><td>CAS</td><td>AEP AUC</td><td>0.0239 0.0342</td><td>0.0024 0.0090</td><td>0.0086 0.0020</td><td>0.0584 0.0030</td><td>0.0157 0.0004</td><td>0.1277 0.1560</td><td>0.0020 0.0036</td><td>0.0165 0.0020</td><td>0.0003 0.0001</td><td>0.0035 0.0049</td><td>0.0259 0.0215</td></tr><tr><td rowspan="4">ZERO [10]</td><td>Random</td><td>AEP AUC</td><td>0.0176 0.2116</td><td>0.0057 0.8836</td><td>0.0120 1.5252</td><td>0.0481 1.3225</td><td>0.0263 0.0225</td><td>0.1517 0.0196</td><td>0.0009 0.0002</td><td>0.0056 0.0064</td><td>0.0009 0.2916</td><td>0.0022 0.0484</td><td>0.0271 0.4332</td></tr><tr><td>Energy [38]</td><td>AEP AUC</td><td>0.0128 2.7225</td><td>0.0023 0.8649</td><td>0.0006 0.0400</td><td>0.0287 0.0132</td><td>0.0007 0.0020</td><td>0.0002 0.0156</td><td>0.0063 0.0484</td><td>0.0001 0.0400</td><td>0.0015 0.0132</td><td>0.0008 0.0056</td><td>0.0054 0.3765</td></tr><tr><td>MCM [41]</td><td>AEP AUC</td><td>0.0138 0.8372</td><td>0.0000 0.4290</td><td>0.0005 0.5929</td><td>0.0421 0.1681</td><td>0.0045 0.0900</td><td>0.0001 0.0441</td><td>0.0060 0.0009</td><td>0.0003 0.5329</td><td>0.0035 0.0036</td><td>0.0002 0.0289</td><td>0.0071 0.2728</td></tr><tr><td>CAS</td><td>AEP AUC AEP</td><td>0.0172 0.0121 0.0179</td><td>0.0083 0.0056 0.0064</td><td>0.0002 0.0121 0.0000</td><td>0.0328 0.0324 0.0283</td><td>0.0089 0.7482 0.0034</td><td>0.0004 4.3681 0.0035</td><td>0.0005 0.0380 0.0018</td><td>0.0061 0.0552 0.0018</td><td>0.0037 0.0000 0.0034</td><td>0.0000 0.0100 0.0007</td><td>0.0078 0.5282 0.0067</td></tr></table>

## J Standard Deviations

We detail the standard deviations of selection strategies on ImageNet and its variants in Table 15. Furthermore, the results on fine-grained and downstream

benchmarks are provided in Table 16, which consistently show the stable performance of CAS across different domains.