# FLOW: Optimal Transport-Driven Feature Warping for Generalized Remote Physiological Measurement

Bo Zhao<sup>1∗</sup>, Junzhe Cao<sup>1,5∗</sup>, Dan Guo <sup>2</sup>, Dongmin Huang<sup>4</sup>, Wenjin Wang <sup>4</sup>, Tao Tan<sup>3</sup> Yue Sun<sup>3‡</sup>, Zitong Yu<sup>1,6‡</sup> <sup>1</sup>Great Bay University <sup>2</sup>Hefei University of Technology, <sup>3</sup>Macao Polytechnic University , <sup>4</sup>Southern University of Science and Technology, <sup>5</sup>Harbin Institute of Technology, Shenzhen, <sup>6</sup>Dongguan Key Laboratory for Intelligence and Information Technology

bozhao@link.cuhk.edu.cn, zitong.yu@ieee.org Equal contribution, <sup>‡</sup> Corresponding author

## Abstract

Remote photoplethysmography (rPPG) enables noncontact physiological measurement but remains vulnerable to domain shifts from illumination, motion, and sensors. We propose FLOW (Feature-Level Optimal Warping), an optimal transport–driven framework for domain-generalized rPPG. FLOW integrates a Temporal Refinement Module (TRM) to stabilize temporal dynamics and a Prototypebased Cross-Temporal Optimal Transport (PCOT) module to achieve domain-invariant alignment via learnable prototypes. Beyond feature alignment, FLOW employs soft cross-temporal correspondence modeling that aligns temporal features in a flexible manner, allowing the model to respect and preserve the intrinsic rhythmic patterns of physiological signals. Moreover, the lightweight design of our modules allows seamless integration into existing end-to-end rPPG architectures without additional preprocessing. Two regularization terms further enforce source consistency and identity preservation. Theoretically, we derive a generalization bound under conditional optimal transport. Extensive experiments across four rPPG benchmarks show that FLOW achieves state-of-the-art cross-domain performance with lightweight design and strong physiologicalfidelity.

## 1. Introduction

Remote photoplethysmography (rPPG) is a non-contact technique for estimating physiological signals such as heart rate and blood volume pulse (BVP) from facial videos. Compared with traditional contact-based photoplethysmography (PPG), rPPG offers a more convenient and hygienic alternative, making it highly attractive for applications in telemedicine, emotion recognition, and fitness monitoring. With the advent of deep learning, recent studies have demonstrated that end-to-end neural networks can directly learn complex spatial-temporal patterns from raw videos for robust rPPG signal prediction [3, 4, 28, 44, 49].

However, despite promising results, the generalization ability of rPPG models—particularly end-to-end rPPG models—remains severely limited. These models often suffer significant performance drops when applied to new domains with different illumination, camera sensors, skin tones, or motion patterns. This issue is primarily due to domain shifts, i.e., the distributional differences between training and deployment environments. In real-world scenarios, such shifts are inevitable, and collecting labeled data from every possible target domain is impractical [15, 58]. Hence, domain generalization (DG) [13, 36], which aims to train models that generalize well to unseen domains without accessing any target data, becomes a critical challenge for making rPPG truly deployable at scale.

While domain generalization has been extensively studied in image classification and other computer vision tasks, its application to end-to-end rPPG learning remains largely unexplored. Most existing efforts in rPPG generalization either rely on data-level preprocessing like spatialtemporal map (STMap) construction followed by handcrafted pipeline tuning [18, 40, 42, 43], or focus on architectural innovations within single-source settings [10, 31, 47, 53]. These approaches, however, face two major limitations: (1) They do not address generalization for fully end-to-end pipelines, where raw video is directly mapped to physiological signals—an increasingly adopted paradigm in modern rPPG; (2) They often lack a principled theoretical understanding of how to align or unify diverse source domain representations.

To fill this gap, we propose FLOW (Feature-Level Optimal Warping), a novel Optimal Transport–driven framework for multi-source domain generalization in end-to-end rPPG learning. Our key insight is to interpret inter-domain variations as structured transport problems and leverage the geometry of Optimal Transport (OT) [6, 29] to achieve principled feature alignment. Specifically, FLOW introduces a plug-and-play feature-level warping module that aligns feature distributions across multiple source domains in a shared latent space. Unlike adversarial or statistical alignment approaches, this OT-based formulation provides an interpretable and mathematically grounded mechanism for domain-invariant representation learning while maintaining compatibility with various rPPG backbones [33, 49].

To enhance stability and physiological fidelity, we design FLOW as a unified framework that integrates temporal refinement, prototype-based alignment, and consistency regularization. A lightweight Temporal Refinement Module (TRM) first produces temporally coherent features by suppressing high-frequency motion artifacts and spatial entanglement. The Prototype-based Cross-Temporal Optimal Transport (PCOT) module then performs soft temporal alignment by establishing correspondences between features and a learnable prototype bank, yielding domain invariant yet physiologically meaningful representations. To further stabilize learning, we introduce two regularization terms: a source-consistency constraint encouraging uniform prototype usage across domains and an identitypreservation constraint maintaining proximity between original and aligned features. Together, these components form a coherent OT-driven objective that reduces interdomain discrepancies while preserving temporal and spectral integrity crucial for accurate physiological estimation.

From a theoretical perspective, we also derive a multisource generalization bound under the conditional optimal transport discrepancy, formally linking alignment quality to prediction risk on unseen domains. This provides a principled explanation for FLOW’s robustness in crossdomain temporal regression tasks.

In summary, our main contributions are as follows:

• We propose FLOW, a architecture-agnostic framework for domain-generalized physiological signal estimation.

• We introduce a Temporal Refinement Module (TRM) that stabilizes heterogeneous temporal patterns across subjects and domains. The module learns to capture local temporal dependencies, suppressing motion artifacts and inconsistent rhythm dynamics.

• We propose a Prototype-based Cross-Temporal Optimal Transport (PCOT) module to align domain-specific features with a prototype bank.PCOT formulates temporal correspondence as an OT problem with task-aware constraints, allowing flexible alignment across domains.

## 2. Related Work

## 2.1. Remote Photoplethysmography (rPPG)

rPPG measurement aims to estimate physiological signals such as heart rate and blood volume pulse (BVP) using only video recordings of human faces. Early rPPG methods mainly relied on signal processing techniques to extract pulse signals from color fluctuations in skin regions [21, 30, 32, 39]. Later, more robust pipelines emerged by introducing spatial-temporal maps (STMaps) [24, 27, 43], which convert pixel-level temporal signals into images for use with convolutional neural networks (CNNs) [28, 42]. Recently, end-to-end deep learning approaches have gained attention for their ability to directly regress physiological signals from raw videos without relying on handcrafted preprocessing. Models such as DeepPhys [4], PhysNet [49], and PhysFormer [51] learn joint spatial-temporal features from video sequences in an end-to-end manner. These methods demonstrate improved performance and robustness compared to traditional pipelines, yet their generalization across domains remains a major challenge.

## 2.2. Domain Generalization (DG)

Domain generalization (DG) aims to train models that perform well on unseen domains without access to any targetdomain data during training. A wide variety of DG approaches have been proposed in the fields of image classification, segmentation, and medical imaging. These include data augmentation-based strategies [55], feature alignment using adversarial learning [14, 17], and regularizationbased methods [11, 56]. In the rPPG literature, domain shift—caused by differences in lighting, motion, skin tone, or recording hardware—has been widely observed to degrade model performance [22, 25, 57]. Some recent works attempt to address this by using domain-aware training protocols or specialized network architectures [42, 51]. However, these studies mostly focus on single-source settings or rely heavily on STMap preprocessing, limiting their applicability to end-to-end pipelines. Moreover, they often lack theoretical grounding, making it hard to generalize their findings across tasks and architectures. To the best of our knowledge, there is currently no dedicated study focusing on domain generalization for end-to-end rPPG models, despite their increasing importance in real-world applications.

## 2.3. Optimal Transport for Domain Adaptation and Generalization

Optimal Transport (OT) has emerged as a powerful tool for comparing and aligning probability distributions. In machine learning, OT has been successfully applied to unsupervised domain adaptation, particularly for image classification tasks [5, 8]. The Wasserstein distance [12], in particular, offers a meaningful and geometry-aware metric to measure distributional divergence. Some studies have extended OT to domain generalization by aligning multiple source domains in a shared feature space [16, 23, 26]. However, these methods are mostly limited to classification problems and are not directly applicable to time-series regression tasks like rPPG measurement. Furthermore, OT has not yet been explored in the context of rPPG, especially in plug-and-play modules for end-to-end networks. In contrast, our work is the first to apply OT-based alignment to multi-source domain generalization for end-to-end rPPG models, offering both algorithmic contributions and theoretical guarantees.

## 3. Methodology

Our goal is to learn temporally stable and physiologically coherent representations that remain robust across subjects and recording conditions. To achieve this, we propose a unified architecture in which temporal refinement precedes cross-domain alignment. As illustrated in Figure 1, intermediate representations are first processed by a lightweight Temporal Refinement Module (TRM), which unifies heterogeneous spatiotemporal features into a consistent temporal form. The refined features are then aligned via the Prototype-based Cross-Temporal Optimal Transport (PCOT) module, which establishes soft correspondences between temporal features and a shared prototype bank. Finally, a set of regularization objectives ensures temporal consistency, cross-domain coherence, and physiological interpretability.

## 3.1. Prototype-based Cross-Temporal Optimal Transport (PCOT)

Temporal signals from different domains often exhibit heterogeneous rhythms and domain-specific distortions. To achieve domain-invariant alignment, PCOT models each temporal step as a distribution over shared prototypes and aligns these representations via an entropic optimal transport (OT) formulation. This yields soft correspondences between time-varying features and domain-agnostic prototypes.

Prototype construction. We maintain a learnable set of prototypes $\mathcal { P } = \{ p _ { k } \} _ { k = 1 } ^ { K }$ and their associated physiological anchors $\mathcal { H } = \{ h _ { k } \} _ { k = 1 } ^ { K }$ . Given a batch of temporal features $X = \{ x _ { t } \} _ { t = 1 } ^ { \bar { T } } \stackrel { \cdot \cdot } { ( x _ { t } \in \mathbb { R } ^ { C } ) }$ , each temporal step is softly matched to prototypes based on both feature similarity and physiological coherence. The transport cost is defined as:

$$
C _ { t , k } = \| W ( x _ { t } - p _ { k } ) \| _ { 2 } ^ { 2 } + \lambda _ { \mathrm { h r } } \left( 1 - \exp \left[ - \frac { ( h _ { t } - h _ { k } ) ^ { 2 } } { 2 \sigma ^ { 2 } } \right] \right) ,\tag{1}
$$

where W is a learnable diagonal weighting matrix, and $h _ { t }$ is the estimated heart rate from an auxiliary head Head<sub>HR</sub>. The first term enforces semantic similarity, while the second term penalizes physiological inconsistency, encouraging prototypes to encode domain-invariant but physiologically coherent features.

Optimal transport formulation. Let $\mu$ and ν denote the empirical distributions of temporal features and prototypes, where $\mu _ { t } = 1 / T$ and $\nu _ { k } = 1 / K$ . The matching is formalized as an entropic OT problem:

$$
S _ { \varepsilon } ( \mu , \nu ) = \operatorname* { m i n } _ { \Pi \in \mathcal { U } ( \mu , \nu ) } \langle C , \Pi \rangle + \varepsilon H ( \Pi ) ,\tag{2}
$$

where $\mathcal { U } ( \mu , \nu ) = \{ \Pi \geq 0 \ \vert \ \Pi \mathbf { 1 } = \mu , \Pi ^ { \top } \mathbf { 1 } = \nu \}$ and $\begin{array} { r } { H ( \Pi ) = - \sum _ { t , k } \pi _ { t , k } ( \log \pi _ { t , k } - 1 ) } \end{array}$ denotes the entropy regularizer. The regularization coefficient ε controls smoothness and ensures differentiability.

Sinkhorn normalization. We compute the optimal coupling Π<sup>⋆</sup> using the Sinkhorn algorithm [7]:

$$
K = \exp \left( - \frac { C } { \varepsilon } \right) , \quad \Pi ^ { \star } = \mathrm { D i a g } ( u ) K \mathrm { D i a g } ( v ) ,\tag{3}
$$

with iterative updates:

$$
u ^ { ( m + 1 ) } = \frac { a } { K v ^ { ( m ) } } , \quad v ^ { ( m + 1 ) } = \frac { b } { K ^ { \top } u ^ { ( m + 1 ) } } ,\tag{4}
$$

where $\begin{array} { r } { a _ { t } \ = \ \frac { 1 } { T } } \end{array}$ and $\begin{array} { r } { \begin{array} { r } { b _ { k } \ = \ \frac { 1 } { K } } \end{array} } \end{array}$ . This normalization ensures marginal consistency $( \Pi ^ { \star } \mathbf { 1 } = a , ( \Pi ^ { \star } ) ^ { \top } \mathbf { 1 } = b )$ and provides a stable and differentiable transport plan.

Barycentric projection and alignment. The optimal coupling $\Pi ^ { \star }$ defines soft correspondences between temporal steps and prototypes. Aligned representations are obtained by barycentric projection:

$$
\tilde { x } _ { t } = \sum _ { k = 1 } ^ { K } \pi _ { t , k } ^ { \star } p _ { k } , \quad \tilde { X } = \{ \tilde { x } _ { t } \} _ { t = 1 } ^ { T } .\tag{5}
$$

This projection re-expresses temporal features in the prototype manifold, removing domain-specific variations and improving temporal smoothness. To mitigate entropy bias, we adopt the debiased Sinkhorn divergence as the alignment loss:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { O T } } = S _ { \varepsilon } ( \mu , \nu ) - \frac { 1 } { 2 } S _ { \varepsilon } ( \mu , \mu ) - \frac { 1 } { 2 } S _ { \varepsilon } ( \nu , \nu ) . } \end{array}\tag{6}
$$

This symmetric and unbiased discrepancy measure stabilizes alignment and enhances generalization.

Intuitively, PCOT constructs a compact and domainagnostic prototype manifold, where temporal features are softly aligned via differentiable OT. The resulting mapping disentangles domain-specific appearance factors while preserving intrinsic physiological rhythms.

## 3.2. Temporal Refinement Module (TRM)

While PCOT aligns temporal dynamics across domains, intermediate features may still contain spatial entanglement or high-frequency noise. The Temporal Refinement Module (TRM) addresses this issue by converting heterogeneous feature tensors into a unified temporal form and refining local dynamics through lightweight depthwiseseparable convolutions.

![](images/4c2008c09e44a77b06f628c80b1929e20783e07a8b535190f9f9a096182f7acf.jpg)  
Figure 1. Overall architecture of FLOW. Given multi-domain video inputs, a shared backbone extracts intermediate features that are temporally refined by the TRM to suppress noise and unify dynamics. The refined sequences are then aligned through the PCOT module, which computes an optimal transport plan between temporal features and a learnable prototype bank. The framework is trained with the OT-based alignment loss ${ \mathcal { L } } _ { O T }$ , the source-consistency loss $\mathcal { L } _ { s c } ,$ the identity-preservation loss $\mathcal { L } _ { i d }$ and the task loss $\mathcal { L } _ { t a s k }$ . During training, the ground truth heart rate signals are used to guide and refine the transport plan for better prototype alignment. During inference, the model directly applies the learned transport plan to predict heart rates, without using any ground truth signals.

Unified temporal representation. Given intermediate features F of arbitrary shape $( \mathbf { e . g . } , B \times C \times H \times W$ or $B \times C \times T \times H \times W )$ , TRM performs spatial global pooling to produce a consistent temporal sequence:

$$
X = { \mathrm { P o o l } } _ { \mathrm { s p a t i a l } } ( F ) \in \mathbb { R } ^ { B \times T \times C } .\tag{7}
$$

This ensures all temporal tokens share a unified semantic basis for subsequent modeling.

Local temporal modeling. TRM refines temporal sequences using stacked depthwise-separable 1D convolutional blocks:

$$
Y ^ { ( l + 1 ) } = \mathrm { N o r m } \Big ( X ^ { ( l ) } + \mathcal { F } _ { \mathrm { T R M } } \Big ( X ^ { ( l ) } \Big ) \Big ) , \quad Y ^ { ( 0 ) } = X .\tag{8}
$$

Each block comprises:

(1) Depthwise temporal filtering:

$$
Z _ { t , c } = \sum _ { i = 1 } ^ { k } w _ { c , i } ^ { ( d ) } X _ { t + i , c } ^ { ( l ) } ,\tag{9}
$$

which captures localized rhythmic dependencies.

(2) Pointwise channel fusion:

$$
\tilde { Z } _ { t } = \phi ( W _ { p } \ : Z _ { t } + b _ { p } ) ,\tag{10}
$$

where $W _ { p } \in \mathbb { R } ^ { C \times C }$ and $\phi ( \cdot )$ denotes a nonlinearity $( \mathrm { e . g . }$ GELU). The combined transformation is:

$$
\mathcal { F } _ { \mathrm { T R M } } ( \boldsymbol { X } ^ { ( l ) } ) = \phi \big ( W _ { p } * \big ( W _ { d } * \boldsymbol { X } ^ { ( l ) } \big ) \big ) ,\tag{11}
$$

balancing local temporal filtering and inter-channel fusion with complexity $\mathcal { O } ( B T C k )$ . Stacking multiple layers progressively suppresses short-term noise and refines rhythmic patterns:

$$
X ^ { \mathrm { T R M } } = \mathrm { T R M } ( X ) \in \mathbb { R } ^ { B \times T \times C } .\tag{12}
$$

From a signal-processing perspective, TRM acts as a lowpass temporal filter that enhances phase stability and rhythmic coherence.

## 3.3. Regularization for Stable Cross-Domain Alignment

Although OT provides a principled alignment mechanism, large domain gaps can lead to unstable or over-smoothed transport plans. We introduce two complementary regularization terms to enhance robustness and preserve physiological interpretability.

Source-consistency regularization. For each source domain $D _ { j }$ , we compute the mean prototype-assignment histogram based on the optimal transport plan $\Pi ^ { \star } \in \mathbb { R } ^ { T \times K }$

$$
\bar { \mathbf { h } } _ { j } = \frac { 1 } { | D _ { j } | } \sum _ { i \in D _ { j } } \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \pi _ { t , k } ^ { \star } .\tag{13}
$$

To ensure consistent prototype utilization across domains, we minimize the variance among these mean histograms:

$$
\mathcal { L } _ { \mathrm { s r c } } = \frac { 1 } { M } \sum _ { j = 1 } ^ { M } \Vert \bar { \mathbf { h } } _ { j } - \bar { \mathbf { h } } \Vert _ { 2 } ^ { 2 } , \quad \bar { \mathbf { h } } = \frac { 1 } { M } \sum _ { j = 1 } ^ { M } \bar { \mathbf { h } } _ { j } .\tag{14}
$$

Table 1. Multi-domain generalization evaluation. Best results are marked in bold. ‘+’ means domain generalization methods are based on ‘Baseline’.
<table><tr><td rowspan="2">Model</td><td colspan="3">Others→U</td><td colspan="3">Others→P</td><td colspan="3">Others→B</td><td colspan="3">Others→M</td><td colspan="3">Average</td></tr><tr><td>MAE↓</td><td>RMSE↓</td><td>R↑</td><td>MAE↓</td><td>RMSE↓</td><td>R↑</td><td>MAE↓</td><td>RMSE↓</td><td>R↑</td><td>MAE↓</td><td>RMSE↓</td><td>R↑</td><td>MAE↓</td><td>RMSE↓</td><td>R↑</td></tr><tr><td>Green [39]</td><td>19.73</td><td>31.00</td><td>0.37</td><td>10.09</td><td>23.85</td><td>0.34</td><td>6.89</td><td>10.39</td><td>0.60</td><td>21.68</td><td>27.69</td><td>-0.01</td><td>14.10</td><td>23.73</td><td>0.33</td></tr><tr><td>CHROM [9]</td><td>7.23</td><td>8.92</td><td>0.51</td><td>9.79</td><td>12.76</td><td>0.37</td><td>6.09</td><td>8.29</td><td>0.51</td><td>13.66</td><td>18.76</td><td>0.08</td><td>9.69</td><td>12.68</td><td>0.37</td></tr><tr><td>POS [41]</td><td>7.35</td><td>8.04</td><td>0.49</td><td>9.82</td><td>13.44</td><td>0.34</td><td>5.04</td><td>7.12</td><td>0.63</td><td>12.36</td><td>17.71</td><td>0.18</td><td>8.64</td><td>11.58</td><td>0.41</td></tr><tr><td>EfficientPhys [19]</td><td>12.87</td><td>18.80</td><td>0.19</td><td>7.15</td><td>15.04</td><td>0.23</td><td>32.30</td><td>34.00</td><td>-0.03</td><td>12.87</td><td>18.80</td><td>0.19</td><td>16.80</td><td>21.66</td><td>0.14</td></tr><tr><td>PhysFormer [52]</td><td>10.29</td><td>18.13</td><td>0.60</td><td>19.75</td><td>24.30</td><td>0.24</td><td>22.09</td><td>26.21</td><td>0.03</td><td>13.90</td><td>19.30</td><td>0.06</td><td>16.51</td><td>21.98</td><td>0.23</td></tr><tr><td>PhysNet [49]</td><td>13.83</td><td>23.66</td><td>0.35</td><td>33.23</td><td>35.25</td><td>-0.15</td><td>12.75</td><td>16.37</td><td>0.08</td><td>13.37</td><td>16.64</td><td>0.29</td><td>18.30</td><td>22.98</td><td>0.14</td></tr><tr><td>RhythmFormer [59]</td><td>14.71</td><td>22.49</td><td>0.43</td><td>21.11</td><td>25.76</td><td>0.04</td><td>6.04</td><td>10.84</td><td>0.42</td><td>16.14</td><td>20.50</td><td>-0.11</td><td>14.50</td><td>19.90</td><td>0.20</td></tr><tr><td>NEST [24]</td><td>12.24</td><td>10.56</td><td>0.36</td><td>19.26</td><td>26.11</td><td>-0.06</td><td>9.19</td><td>12.38</td><td>0.19</td><td>13.97</td><td>18.20</td><td>0.15</td><td>13.67</td><td>16.81</td><td>0.16</td></tr><tr><td>Greip [54]</td><td>17.50</td><td>20.42</td><td>0.21</td><td>5.07</td><td>14.50</td><td>0.78</td><td>7.94</td><td>10.93</td><td>-0.03</td><td>13.02</td><td>17.11</td><td>0.14</td><td>10.88</td><td>15.74</td><td>0.28</td></tr><tr><td>Baseline [48]</td><td>9.92</td><td>13.92</td><td>0.64</td><td>15.97</td><td>26.61</td><td>0.23</td><td>6.02</td><td>8.61</td><td>0.63</td><td>12.23</td><td>15.51</td><td>0.15</td><td>11.04</td><td>16.16</td><td>0.41</td></tr><tr><td>Coral+ [36]</td><td>11.47</td><td>14.42</td><td>0.54</td><td>14.18</td><td>23.04</td><td>0.26</td><td>3.06</td><td>4.42</td><td>0.95</td><td>9.18</td><td>14.96</td><td>0.43</td><td>9.97</td><td>14.21</td><td>0.55</td></tr><tr><td>MMD+ [13]</td><td>9.18</td><td>12.14</td><td>0.60</td><td>16.56</td><td>22.93</td><td>0.28</td><td>2.80</td><td>3.67</td><td>0.95</td><td>8.87</td><td>14.39</td><td>0.45</td><td>9.35</td><td>13.28</td><td>0.57</td></tr><tr><td>FLOW (Ours)</td><td>6.89</td><td>10.12</td><td>0.69</td><td>10.86</td><td>16.47</td><td>0.64</td><td>2.23</td><td>3.36</td><td>0.97</td><td>7.38</td><td>13.12</td><td>0.51</td><td>6.84</td><td>10.75</td><td>0.70</td></tr></table>

This encourages all domains to share a uniform prototype occupancy pattern, reinforcing domain-invariant semantics.

Identity-preservation regularization. To prevent excessive feature deformation, we constrain the distance between original and aligned representations:

$$
\mathcal { L } _ { \mathrm { i d } } = \frac { 1 } { B T } \sum _ { b = 1 } ^ { B } \sum _ { t = 1 } ^ { T } \Vert \tilde { x } _ { b , t } - x _ { b , t } \Vert _ { 2 } ^ { 2 } .\tag{15}
$$

This penalty stabilizes alignment while preserving subject identity and intrinsic rhythm.

Final objective. The overall training objective integrates OT alignment with the two regularization terms:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { t a s k } } + \lambda _ { \mathrm { O T } } \mathcal { L } _ { \mathrm { O T } } + \lambda _ { \mathrm { s r c } } \mathcal { L } _ { \mathrm { s r c } } + \lambda _ { \mathrm { i d } } \mathcal { L } _ { \mathrm { i d } } .\tag{16}
$$

Here, $\lambda _ { \mathrm { s r c } }$ and $\lambda _ { \mathrm { i d } }$ balance alignment flexibility and representation stability. The detailed model training and inference process can be found in the Appendix D.

Together, TRM, PCOT, and the regularization objectives form a coherent pipeline: TRM unifies and refines temporal dynamics, PCOT aligns features through prototype-guided optimal transport, and the regularizations ensure both global domain consistency and local physiological preservation. This design enables smooth temporal representations without adversarial training.

## 4. Experiments

We conduct multi-source domain generalization experiments on four public datasets: UBFC-rPPG (U) [1], PURE (P) [34], BUAA-MIHR (B) [45], and MMPD (M) [38]. We report Mean Absolute Error (MAE), Root Mean Square Error (RMSE), and Pearson correlation coefficient (R), where lower MAE/RMSE and higher R indicate better performance. All experimental settings and implementation details are included in Appendix B.

## 4.1. Experimental Results

## 4.1.1. Multi-source Domain Generalization Results

Comparison with standard baselines. We first compare FLOW with conventional signal-processing and learningbased approaches that lack explicit domain generalization mechanisms. Traditional handcrafted methods— Green [13], CHROM [9], and POS [41]—show strong domain sensitivity, exhibiting large performance fluctuations across datasets. For instance, although CHROM achieves a reasonable MAE of 6.09 on BUAA-MIHR [45], its error sharply increases to 13.66 on MMPD [38] and 7.23 on UBFC-rPPG [1], highlighting its vulnerability to illumination and motion changes.

Among deep-learning-based baselines, PhysNet [49] and EfficientPhys [19] also exhibit poor cross-domain robustness, particularly when the target distribution deviates from training sources. Notably, PhysNet even produces a negative correlation on PURE [34] (R = -0.15), indicating a failure to capture meaningful physiological dynamics. Phys-Former [52] demonstrates partial robustness but still performs inconsistently across domains (e.g., R = 0.03 on BUAA-MIHR and R = 0.06 on MMPD). Overall, these results confirm that, without explicit domain-invariant modeling, existing rPPG frameworks struggle to maintain stability under distribution shifts.

Comparison with DG baselines. We further evaluate FLOW against representative domain generalization (DG) methods—CORAL [36] and MMD [13]—implemented on the same backbone for a fair comparison. Coral [36] improves correlation on PURE [34] and BUAA-MIHR [45] (e.g., R = 0.64 and R = 0.95, respectively), yet its MAE and RMSE remain relatively high, suggesting that secondorder feature alignment alone is insufficient for robust temporal generalization. MMD [13] yields more balanced performance across metrics but still suffers from instability on UBFC-rPPG and MMPD. In contrast, FLOW achieves consistently superior results across all unseen domains. On

Table 2. Limited-source domain generalization performance on MMPD. Best results are in bold.
<table><tr><td rowspan="2">Model</td><td colspan="3">P+B</td><td colspan="3">P+U</td><td colspan="3">B+U</td><td colspan="3">Average</td></tr><tr><td>MAE↓</td><td>RMSE↓</td><td>R↑</td><td>MAE↓</td><td>RMSE↓</td><td>R↑</td><td>MAE↓</td><td>RMSE↓</td><td>R↑</td><td>MAE↓</td><td>RMSE↓</td><td>R↑</td></tr><tr><td>PhysNet [49]</td><td>13.20</td><td>16.70</td><td>0.23</td><td>11.00</td><td>17.30</td><td>0.28</td><td>13.50</td><td>17.00</td><td>0.09</td><td>12.57</td><td>17.00</td><td>0.20</td></tr><tr><td>PhysFormer [52]</td><td>13.90</td><td>18.60</td><td>0.21</td><td>11.40</td><td>17.50</td><td>0.23</td><td>13.20</td><td>16.50</td><td>0.12</td><td>12.83</td><td>17.53</td><td>0.19</td></tr><tr><td>EfficientPhys [19]</td><td>11.90</td><td>18.50</td><td>0.21</td><td>11.80</td><td>18.90</td><td>0.22</td><td>15.50</td><td>20.80</td><td>0.03</td><td>13.07</td><td>19.40</td><td>0.15</td></tr><tr><td>RhythmFormer [59]</td><td>13.98</td><td>19.46</td><td>0.12</td><td>10.50</td><td>16.72</td><td>0.28</td><td>11.70</td><td>16.56</td><td>0.18</td><td>12.06</td><td>17.58</td><td>0.19</td></tr><tr><td>NEST [20]</td><td>10.85</td><td>15.45</td><td>0.33</td><td>9.90</td><td>14.82</td><td>0.36</td><td>10.64</td><td>15.12</td><td>0.31</td><td>10.46</td><td>15.13</td><td>0.33</td></tr><tr><td>Greip [54]</td><td>10.60</td><td>15.12</td><td>0.35</td><td>9.78</td><td>14.63</td><td>0.38</td><td>10.42</td><td>14.85</td><td>0.33</td><td>10.27</td><td>14.87</td><td>0.35</td></tr><tr><td>Baseline [48]</td><td>11.90</td><td>15.30</td><td>0.26</td><td>9.95</td><td>14.96</td><td>0.31</td><td>12.10</td><td>15.20</td><td>0.21</td><td>11.32</td><td>15.15</td><td>0.26</td></tr><tr><td>Coral+ [36]</td><td>10.88</td><td>15.47</td><td>0.29</td><td>11.62</td><td>15.89</td><td>0.25</td><td>10.95</td><td>15.34</td><td>0.28</td><td>11.15</td><td>15.57</td><td>0.27</td></tr><tr><td>MMD+ [13]</td><td>9.96</td><td>15.01</td><td>0.34</td><td>11.49</td><td>16.02</td><td>0.24</td><td>10.78</td><td>15.42</td><td>0.27</td><td>10.74</td><td>15.48</td><td>0.28</td></tr><tr><td>FLOW (Ours)</td><td>8.90</td><td>13.64</td><td>0.47</td><td>8.34</td><td>12.79</td><td>0.51</td><td>8.70</td><td>13.36</td><td>0.45</td><td>8.65</td><td>13.26</td><td>0.48</td></tr></table>

Table 3. Limited-source domain generalization performance on BUAA-MIHR. Best results are in bold.
<table><tr><td rowspan="2">Model</td><td colspan="3">P+M</td><td colspan="3">M+U</td><td colspan="3">P+U</td><td colspan="3">Average</td></tr><tr><td>MAE↓</td><td>RMSE↓</td><td>R↑</td><td>MAE↓</td><td>RMSE↓</td><td>R↑</td><td>MAE↓</td><td>RMSE↓</td><td>R↑</td><td>MAE↓</td><td>RMSE↓</td><td>R↑</td></tr><tr><td>PhysNet [49]</td><td>20.97</td><td>24.75</td><td>0.01</td><td>11.40</td><td>16.72</td><td>0.14</td><td>15.34</td><td>21.48</td><td>-0.29</td><td>15.90</td><td>20.98</td><td>-0.05</td></tr><tr><td>PhysFormer [52]</td><td>14.86</td><td>18.26</td><td>0.03</td><td>10.87</td><td>16.20</td><td>0.08</td><td>18.23</td><td>22.17</td><td>0.07</td><td>14.65</td><td>18.88</td><td>0.06</td></tr><tr><td>EfficientPhys [19]</td><td>4.15</td><td>7.14</td><td>0.77</td><td>3.00</td><td>5.18</td><td>0.89</td><td>3.00</td><td>5.18</td><td>0.89</td><td>3.38</td><td>5.83</td><td>0.85</td></tr><tr><td>RhythmFormer [59]</td><td>3.55</td><td>5.35</td><td>0.90</td><td>6.20</td><td>11.23</td><td>0.49</td><td>3.90</td><td>6.51</td><td>0.82</td><td>4.55</td><td>7.70</td><td>0.74</td></tr><tr><td>NEST [20]</td><td>3.32</td><td>5.04</td><td>0.92</td><td>3.71</td><td>5.67</td><td>0.87</td><td>3.15</td><td>4.88</td><td>0.93</td><td>3.39</td><td>5.20</td><td>0.91</td></tr><tr><td>Greip [54]</td><td>3.21</td><td>4.92</td><td>0.93</td><td>3.82</td><td>5.86</td><td>0.85</td><td>3.08</td><td>4.73</td><td>0.94</td><td>3.37</td><td>5.17</td><td>0.91</td></tr><tr><td>Baseline [48]</td><td>3.06</td><td>3.99</td><td>0.95</td><td>4.46</td><td>8.85</td><td>0.60</td><td>3.64</td><td>8.04</td><td>0.71</td><td>3.72</td><td>6.96</td><td>0.75</td></tr><tr><td>Coral+ [36]</td><td>3.20</td><td>4.30</td><td>0.92</td><td>4.80</td><td>9.00</td><td>0.58</td><td>3.54</td><td>6.12</td><td>0.78</td><td>3.85</td><td>6.47</td><td>0.76</td></tr><tr><td>MMD+ [13]</td><td>2.98</td><td>3.92</td><td>0.95</td><td>4.83</td><td>9.15</td><td>0.57</td><td>3.89</td><td>6.72</td><td>0.72</td><td>3.90</td><td>6.60</td><td>0.75</td></tr><tr><td>FLOW (Ours)</td><td>2.56</td><td>3.35</td><td>0.98</td><td>2.78</td><td>2.96</td><td>0.95</td><td>2.33</td><td>3.10</td><td>0.97</td><td>2.56</td><td>3.14</td><td>0.97</td></tr></table>

BUAA-MIHR, FLOW attains an MAE of 2.23 and R of 0.97, surpassing the best DG baseline (MMD [13], MAE = 2.80, R = 0.95). On UBFC-rPPG [1] and PURE [34], FLOW improves the Pearson correlation by more than 0.4 compared with MMD [13], demonstrating strong robustness against appearance and motion variability.

These results highlight that effective domain generalization in rPPG requires not only statistical distribution alignment but also semantic and physiological consistency—objectives explicitly enforced in FLOW through its prototype-based optimal-transport alignment and temporal refinement modules.

## 4.1.2. Limited-source Domain Generalization Results

To further evaluate the robustness of our method under limited training data scenarios, we perform cross-domain experiments where only two datasets were used as source domains. We selected two target datasets, MMPD [38] and BUAA-MIHR [45], due to their challenging domain characteristics. Table 2 and Table 3 reports detailed comparisons.

Performance on MMPD [38]. Across all source-domain combinations, our method consistently achieves the lowest MAE and RMSE. When trained on PURE [34]+UBFCrPPG [1], our model attains an MAE of 9.06 and an RMSE of 14.23, outperforming both traditional approaches (e.g., Green [39]) and domain generalization baselines (e.g., Coral [36], MMD [13]). This demonstrates the model’s capability to generalize to domains with complex motion patterns and severe illumination variations, even when the diversity of training data is limited.

Performance on BUAA-MIHR [45]. Our approach also exhibits strong generalization to BUAA-MIHR [45]. Particularly under the PURE [34]+MMPD [38] training setup, it achieves an MAE of 2.56 and R = 0.98—the best among all methods. Such results highlight our framework’s ability to learn semantically consistent, domain-invariant temporal features from disjoint source domains, effectively bridging large inter-domain gaps.

Comparison with DG baselines. Although Coral [36] and MMD [13] improve over the naive baseline, their gains are often inconsistent and sensitive to domain combinations. For instance, Coral [36] performs relatively well on BUAA-MIHR [45] but degrades noticeably on MMPD [38], while MMD [13] struggles when the domain discrepancy is large (e.g., PURE [34]+BUAA-MIHR [45] → MMPD [38]). In contrast, our method maintains stable performance across all configurations, indicating better scalability and robustness under data-constrained scenarios.

Table 4. Comparison of different domain-alignment modules applied to the PhysFormer backbone (values are MAE↓/RMSE↓). “MMD” and “CORAL” are standard alignment baselines, and “FLOW” is our proposed method.  
(a) Limited-source domain generalization performance on BUAA-MIHR
<table><tr><td></td><td>P+M</td><td>U+M</td><td>P+U</td></tr><tr><td>PhysFormer</td><td>14.86/18.26</td><td>10.87/16.20</td><td>8.23/22.17</td></tr><tr><td>PhysFormer (MMD)</td><td>33.12/35.26</td><td>8.14/11.23</td><td>18.41/21.10</td></tr><tr><td>PhysFormer (CORAL)</td><td>13.25/20.03</td><td>10.69/15.24</td><td>14.35/19.31</td></tr><tr><td>PhysFormer (FLOW) 10.38/14.14</td><td></td><td>7.18/11.14</td><td>9.59/13.85</td></tr></table>

<table><tr><td colspan="4">(b) Limited-source domain generalization performance on MMPD</td></tr><tr><td></td><td>P+B</td><td>P+U</td><td>B+U</td></tr><tr><td>PhysFormer</td><td>13.91/18.63</td><td>11.41/17.52</td><td>13.20/16.50</td></tr><tr><td>PhysFormer (MMD)</td><td>15.31/19.01</td><td>22.12/26.41</td><td>13.26/16.53</td></tr><tr><td>PhysFormer (CORAL)</td><td>13.24/18.17</td><td>10.31/16.20</td><td>12.76/18.14</td></tr><tr><td>PhysFormer (FLOW) 10.34/17.05</td><td></td><td>9.18/15.50</td><td>10.01/13.72</td></tr></table>

Overall, these findings confirm that our model generalizes effectively to unseen domains even when the source diversity is limited. We attribute this robustness to our prototype-based feature alignment and temporal refinement, which jointly preserve semantic structures while reducing domain-specific bias. Consequently, our framework provides a more reliable foundation for real-world crossdomain rPPG deployment.

## 4.1.3. Generalization of FLOW

To assess the generalization ability of FLOW beyond a specific architecture, we integrate it into multiple representative rPPG backbones, including RhythmFormer [59], EfficientPhys [19], PhysFormer [51], and PhysNet [49]. As shown in Figure 2, FLOW consistently improves performance across diverse architectures and datasets , reducing MAE by a large margin. These results demonstrate that FLOW is architecture-agnostic and can be seamlessly integrated into various temporal models to enhance temporal stability and physiological consistency.

Furthermore, we compare FLOW with conventional domain-alignment techniques—MMD [13] and Coral [36]—under the same PhysFormer backbone. As presented in Table 4, FLOW achieves the lowest MAE/RMSE across multiple source–target settings (e.g., PM-B, MU-B, and BU-M). While MMD [13] and Coral [36] only perform statistical alignment on global feature distributions, FLOW introduces temporal refinement and prototype-based optimal transport, enabling more stable and physiologically meaningful cross-domain adaptation.

Performance across source combinations. In all configurations, incorporating FLOW consistently improves over the baseline PhysFormer and its DG variants. For instance, under the PURE [34]+MMPD [38] training setup, FLOW reduces MAE from 14.86 to 10.38 and RMSE from 18.26 to 14.14, achieving a clear performance gain. Similarly, in the PURE [34]+BUAA-MIHR [45] setting (targeting MMPD [38]), FLOW improves the correlation from 0.21 to 0.42. In contrast, both MMD [13] and Coral [36] exhibit unstable results in several cases (e.g., negative R under PURE [34]+MMPD [38] or PURE [34]+UBFC-rPPG [1]), indicating that simple statistical alignment fails to preserve meaningful temporal dynamics.

Table 5. Ablation results on model components. ”FLOW” denotes the full FLOW model with both TRM and PCOT enabled.
<table><tr><td></td><td colspan="2">BUAA</td><td colspan="2">MMPD</td></tr><tr><td></td><td>MAE↓</td><td>RMSE↓</td><td>MAE↓</td><td>RMSE↓</td></tr><tr><td>FLOW</td><td>2.23</td><td>3.36</td><td>7.38</td><td>13.12</td></tr><tr><td>FLOW(w/o TRM)</td><td>3.12</td><td>4.61</td><td>8.16</td><td>14.10</td></tr><tr><td>FLOW(w/o PCOT)</td><td>4.67</td><td>6.13</td><td>10.24</td><td>14.94</td></tr></table>

Architecture-level transferability. The consistent improvement across PhysFormer variants validates the nature of FLOW, confirming that its prototype-based temporal alignment does not depend on a specific model architecture.

## 4.2. Ablation Study

## 4.2.1. Ablation on Model Components

To investigate the contribution of each component in FLOW, we perform an ablation study by removing the Temporal Relation Module (TRM) and the Prototype-based Cross-domain Optimal Transport (PCOT) module, respectively. Experiments are conducted on BUAA-MIHR [45] and MMPD [38] datasets under the same training setup as the multi-source domain generalization setting. The results are summarized in Table 5.

Efficacy of TRM. Removing the Temporal Relation Module leads to a noticeable drop in performance. Specifically, MAE increases from 2.23 to 3.12 on BUAA-MIHR [45] and from 7.38 to 8.16 on MMPD [38], indicating that the TRM effectively captures temporal dependencies and mitigates temporal noise during motion variations. Efficacy of PCOT. Eliminating the PCOT module causes the largest degradation among all variants, showing an MAE of 4.67 on BUAA-MIHR [45] and 10.24 on MMPD [38]. This demonstrates that cross-domain prototype alignment is crucial for learning semantically consistent and domain-invariant representations, enabling the model to generalize effectively to unseen environments.

## 4.2.2. Ablation on Loss Functions

We further examine the contribution of each loss component in FLOW, including the optimal transport loss $( L _ { \mathrm { O T } } ) .$ the prototype alignment loss $( L _ { \mathrm { a l i g n } } )$ , and the semantic consistency loss $( L _ { \mathrm { s c } } )$ . As shown in Table $^ { 6 , }$ using only a single objective yields limited improvement, since each focuses on a different aspect of domain adaptation. The optimal transport term facilitates temporal correspondence between prototype distributions, while $L _ { \mathrm { a l i g n } }$ enhances cross-domain feature alignment. Adding $L _ { \mathrm { s c } }$ further preserves semantic relationships among prototypes, ensuring that inter-domain structural consistency is maintained. When all three losses are combined, the model achieves the best performance (MAE = 2.23 on BUAA-MIHR [45], 7.38 on MMPD [38]), indicating that the joint optimization effectively balances global distribution alignment and semantic-level regularization, leading to stronger domain generalization .

![](images/3d3b60e677cee0c4cdd7baf6e400214e9b209cbb4ae87af4545a646d25e8f877.jpg)

Table 6. Ablation study on loss functions.
<table><tr><td rowspan="2"> $L _ { \mathrm { O T } }$ </td><td rowspan="2"> $L _ { \mathrm { i d } }$ </td><td rowspan="2"> $L _ { \mathrm { s c } }$ </td><td colspan="2">BUAA</td><td colspan="2">MMPD</td></tr><tr><td>MAE↓</td><td>RMSE↓</td><td>MAE↓</td><td>RMSE↓</td></tr><tr><td>√</td><td></td><td></td><td>4.63</td><td>6.32</td><td>9.73</td><td>14.32</td></tr><tr><td></td><td>√</td><td></td><td>4.57</td><td>5.89</td><td>9.82</td><td>13.96</td></tr><tr><td>√</td><td>√</td><td></td><td>2.44</td><td>4.12</td><td>8.23</td><td>14.42</td></tr><tr><td>√</td><td></td><td>√</td><td>3.63</td><td>5.61</td><td>7.98</td><td>13.17</td></tr><tr><td></td><td>√</td><td>√</td><td>3.77</td><td>5.72</td><td>8.87</td><td>14.71</td></tr><tr><td>√</td><td>√</td><td>√</td><td>2.23</td><td>3.36</td><td>7.38</td><td>13.12</td></tr></table>

Figure 2. Visualization of FLOW’s effectiveness across different backbones and cross-domain settings.

## 4.2.3. Ablation on Backbone

To further illustrate the effectiveness and generalization capability of FLOW, we visualize its impact on different backbone models. As shown in Fig. 2, we compare the mean absolute error (MAE) before and after integrating FLOW into four representative architectures, RhythmFormer [59], EfficientPhys [19], Phys-Former [52], and PhysNet [49]—under four cross-domain settings (Others→UBFC-rPPG [1], Others→PURE, Others→BUAA-MIHR [45], and Others→MMPD [38]).

Across all backbones and datasets, incorporating FLOW consistently reduces the prediction error. Notably, Phys-Former and EfficientPhys exhibit substantial improvements on challenging domains such as BUAA-MIHR [45] and MMPD [38], where domain shifts due to illumination and motion are most severe. This confirms that FLOW effectively captures domain-invariant temporal dependencies and semantic consistency without relying on specific architectural designs. The consistent downward trend across all models highlights FLOW’s universality as a plug-and-play module for cross-domain physiological signal estimation.

![](images/684a9da755a76cc0e38184646d26cc2b2b5e852c728a3e64380588bac6cf03a7.jpg)  
(a) Effect of the number of prototypes K in the PCOT module on MAE/MSE performance.

![](images/e08cbf11ca2a27a0e2668bd1df861b8fc91e1b805a3e4bfb8c68a9f3edadcf3b.jpg)  
(b) Effect of the number of aligner layers in the TRM block on MAE/MSE performance.  
Figure 3. Ablations on the structure of PCOT and TRM modules.

## 4.2.4. Analysis of Structural Hyperparameters

We further investigate the impact of two key structural hyperparameters in FLOW: the number of prototypes K in the PCOT module and the number of aligner layers in the TRM block. As shown in Fig. 3a and Fig. 3b, both factors influence performance but exhibit stable trends, demonstrating the robustness of our framework.

For the prototype number K, the MAE gradually decreases as K increases from 16 to 64, reaching the optimal performance (2.23 bpm) at $K \ : = \ : 6 4$ . Beyond this point, the improvement saturates and slightly declines, likely due to over-fragmentation of prototype distributions that weakens semantic compactness. For the number of aligner layers, increasing layers improves representation alignment up to three layers, where the MAE reaches its minimum (2.23 bpm). Further stacking yields marginal gains, indicating that moderate depth is sufficient to capture cross-domain temporal dependencies without overfitting.

These results confirm that FLOW maintains consistent generalization across reasonable hyperparameter ranges, highlighting its stability and scalability under different architectural configurations.

## 5. Conclusion

In this paper, we propose the FLOW, a unified framework for domain-generalized physiological signal estimation. By combining the Temporal Refinement Module (TRM) for temporal coherence and the Prototype-based Cross-domain Optimal Transport (PCOT) for semantic alignment, FLOW effectively learns domain-invariant yet physiologically consistent representations. Extensive experiments demonstrate that FLOW achieves state-of-the-art generalization across diverse datasets and architectures. In the future, we plan to extend FLOW to multi-modal sensing and real-world deployment, advancing reliable physiological estimation under unconstrained conditions.

## References

[1] Serge Bobbia, Richard Macwan, Yannick Benezeth, Alamin Mansouri, and Julien Dubois. Unsupervised skin tissue segmentation for remote photoplethysmography. Pattern Recognition Letters, 124:82–90, 2017. 5, 6, 7, 8, 1

[2] Serge Bobbia, Richard Macwan, Yannick Benezeth, Alamin Mansouri, and Julien Dubois. Unsupervised skin tissue segmentation for remote photoplethysmography. Pattern recognition letters, 124:82–90, 2019. 1

[3] Junzhe Cao, Bo Zhao, Zhiyi Niu, Dan Guo, Yue Sun, Haochen Liang, Yong Xu, and Zitong Yu. Physnext: Next-generation dual-branch structured attention fusion network for remote photoplethysmography measurement. arXiv preprint arXiv:2603.19752, 2026. 1

[4] Wei-Ting Chen, Daniel McDuff, Javier Hernandez, and Rosalind W Picard. Deepphys: Video-based physiological measurement using convolutional attention networks. In Proceedings of the European Conference on Computer Vision (ECCV), pages 349–365, 2018. 1, 2

[5] Nicolas Courty, Remi Flamary, Devis Tuia, and Alain Rako-´ tomamonjy. Joint distribution optimal transportation for domain adaptation. In Advances in Neural Information Processing Systems (NeurIPS), 2017. 2

[6] Nicolas Courty, Remi Flamary, Devis Tuia, and Alain Rako-´ tomamonjy. Optimal transport for domain adaptation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 39(9):1853–1865, 2017. 2

[7] Marco Cuturi. Sinkhorn distances: Lightspeed computation of optimal transport. In Advances in Neural Information Processing Systems (NeurIPS), pages 2292–2300, 2013. 3

[8] Bharath Bhushan Damodaran, Benjamin Kellenberger, Remi´ Flamary, Devis Tuia, and Nicolas Courty. Deepjdot: Deep joint distribution optimal transport for unsupervised domain adaptation. In European Conference on Computer Vision (ECCV), pages 447–463, 2018. 2

[9] Gerard De Haan and Vincent Jeanne. Robust pulse rate from chrominance-based rppg. IEEE transactions on biomedical engineering, 60(10):2878–2886, 2013. 5

[10] Jingda Du, Si-Qi Liu, Bochao Zhang, and Pong C. Yuen. Dual-bridging with adversarial noise generation for domain adaptive rppg estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), page 10355–10364, 2023. 1

[11] Abhishek Dubey, Raja Giryes, and Lior Wolf. Adaptive risk minimization: A meta-learning approach for domain generalization. In IEEE International Conference on Computer Vision (ICCV), pages 1090–1099, 2021. 2

[12] Wilfrid Gangbo and Robert J McCann. The geometry of optimal transportation. 1996. 2

[13] Arthur Gretton, Karsten M Borgwardt, Malte J Rasch, Bernhard Scholkopf, and Alexander Smola. A kernel two-sample¨ test. The Journal of Machine Learning Research, 13(1):723– 773, 2012. 1, 5, 6, 7

[14] Libo Huang, Lu Gan, and Bingo Wing-Kuen Ling. A unified optimization model of feature extraction and clustering for spike sorting. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 29:750–759, 2021. 2

[15] Libo Huang, Yan Zeng, Chuanguang Yang, Zhulin An, Boyu Diao, and Yongjun Xu. etag: Class-incremental learning via embedding distillation and task-oriented generation. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 12591–12599, 2024. 1

[16] Zhexian Huang, Bo Zhao, Hui Ma, Zhishu Liu, Jie Zhang, Ruixin Zhang, Shouhong Ding, and Zitong Yu. Complementarity-supervised spectral-band routing for multimodal emotion recognition. arXiv preprint arXiv:2603.13340, 2026. 2

[17] Da Li, Yongxin Yang, Yuwei Song, and Timothy M Hospedales. Domain generalization via conditional invariant adversarial networks. Advances in Neural Information Processing Systems (NeurIPS), 31, 2018. 2

[18] Haodong Li, Hao Lu, and Ying-Cong Chen. Bi-tta: Bidi rectional test-time adapter for remote physiological measurement. In European Conference on Computer Vision, pages 356–374. Springer, 2024. 1

[19] Xin Liu, Brian Hill, Ziheng Jiang, Shwetak Patel, and Daniel McDuff. Efficientphys: Enabling simple, fast and accurate camera-based cardiac measurement. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 5008–5017, 2023. 5, 6, 7, 8

[20] Xin Liu, Girish Narayanswamy, Akshay Paruchuri, Xiaoyu Zhang, Jiankai Tang, Yuzhe Zhang, Roni Sengupta, Shwetak Patel, Yuntao Wang, and Daniel McDuff. rppg-toolbox: Deep remote ppg toolbox. Advances in Neural Information Processing Systems, 36:68485–68510, 2023. 6, 1

[21] Yunfan Liu, Qi Li, and Zhenan Sun. One-shot face reenactment with dense correspondence estimation. Machine Intelligence Research, 21(5):941–953, 2024. 2

[22] Zhishu Liu, Kaishen Yuan, Bo Zhao, Yong Xu, and Zitong Yu. Au-llm: Micro-expression action unit detection via enhanced llm-based feature fusion. In Chinese Conference on Biometric Recognition, pages 355–365. Springer, 2025. 2

[23] Zhishu Liu, Kaishen Yuan, Bo Zhao, Hui Ma, and Zitong Yu. Aullm++: Structural reasoning with large language models for micro-expression recognition. arXiv preprint arXiv:2603.08387, 2026. 2

[24] Hao Lu, Zitong Yu, Xuesong Niu, and Ying-Cong Chen. Neuron structure modeling for generalizable remote physiological measurement. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18589–18599, 2023. 2, 5

[25] Daniel McDuff. A survey of remote optical photoplethysmographic imaging methods. IEEE Transactions on Biomedical Engineering, 66(4):1–16, 2018. 2

[26] Caio Montesuma, Simone Scardapane, Lamberto Ballan, and Pier Luigi Dragotti. Wasserstein domain generaliza tion: theoretical foundations and algorithms. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2021. 2

[27] Xuesong Niu, Shiguang Shan, Hu Han, and Xilin Chen. Rhythmnet: End-to-end heart rate estimation from face via spatial-temporal representation. IEEE Transactions on Im age Processing, 29:2409–2423, 2019. 2

[28] Xiaobai Niu, Hui Han, Shiguang Shan, and Xilin Chen. Video-based physiological measurement via cross-verified

feature disentangling. In European Conference on Computer Vision (ECCV), pages 545–561, 2020. 1, 2

[29] Gabriel Peyre and Marco Cuturi. Computational optimal ´ transport. Foundations and Trends in Machine Learning, 11 (5-6):355–607, 2019. 2

[30] Ming-Zher Poh, Daniel J McDuff, and Rosalind W Picard. Non-contact, automated cardiac pulse measurements using video imaging and blind source separation. In International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC), pages 3130–3133, 2010. 2

[31] Marko Savic and Guoying Zhao. Oulu remotephotoplethysmography physical domain attacks database (orpdad). In Computer Vision – ECCV 2024, Lecture Notes in Computer Science, Vol. 15131, page 51–68. Springer, Cham, 2024. 1

[32] Guanghui Shi, Shasha Mao, Shuiping Gou, Dandan Yan, Licheng Jiao, and Lin Xiong. Adaptively enhancing facial expression crucial regions via a local non-local joint network. Machine Intelligence Research, 21(2):331–348, 2024. 2

[33] Jang-Han Song, Youngbin Kim, Sangsoo Lee, and Seungmoon Lee. Hr-cnn: Deep learning for remote heart rate estimation from face videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 772–781, 2021. 2

[34] Ronny Stricker, Steffen Mueller, and Horst-Michael Gross. Non-contact video-based pulse rate measurement on a mobile service robot. In IEEE Int. Symposium on Robot and Human Interactive Communication (RO-MAN), pages 1056– 1062. IEEE, 2014. 5, 6, 7, 1

[35] Ronny Stricker, Steffen Muller, and Horst-Michael Gross.¨ Non-contact video-based pulse rate measurement on a mobile service robot. In The 23rd IEEE International Symposium on Robot and Human Interactive Communication, pages 1056–1062. IEEE, 2014. 1

[36] Baochen Sun, Jiashi Feng, and Kate Saenko. Return of frustratingly easy domain adaptation. arXiv preprint arXiv:1511.05547, 2016. 1, 5, 6, 7

[37] Zhaodong Sun and Xiaobai Li. Contrast-phys: Unsupervised video-based remote physiological measurement via spatiotemporal contrast. In European Conference on Computer Vision, pages 492–510. Springer, 2022. 1

[38] Jiankai Tang, Kequan Chen, Yuntao Wang, Yuanchun Shi, Shwetak Patel, Daniel McDuff, and Xin Liu. Mmpd: Multidomain mobile video physiology dataset. In 2023 45th Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC), pages 1–5. IEEE, 2023. 5, 6, 7, 8, 1

[39] Wim Verkruysse, Lars O Svaasand, and J Stuart Nelson. Remote plethysmographic imaging using ambient light. Optics express, 16(26):21434–21445, 2008. 2, 5, 6

[40] Jiyao Wang, Hao Lu, Ange Wang, Xiao Yang, Yingcong Chen, Dengbo He, and Kaishun Wu. Physmle: Generalizable and priors-inclusive multi-task remote physiological measurement. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025. 1

[41] Wenjin Wang, Albertus C Den Brinker, Sander Stuijk, and Gerard De Haan. Algorithmic principles of remote ppg.

IEEE Transactions on Biomedical Engineering, 64(7):1479– 1491, 2016. 5

[42] Weixuan Wang, Kunlin Peng, Yao Lu, and Min Wu. Domain adaptation for remote photoplethysmography under inconsistent light conditions. In IEEE Transactions on Instrumentation and Measurement, pages 1–13, 2021. 1, 2

[43] Yin Wang, Hao Lu, Ying-Cong Chen, Li Kuang, Mengchu Zhou, and Shuiguang Deng. rppg-hiba: Hierarchical balanced framework for remote physiological measurement. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 2982–2991, 2024. 1, 2

[44] Zheng Wu, Yiping Xie, Bo Zhao, Jiguang He, Fei Luo, Ning Deng, and Zitong Yu. Cardiacmamba: A multimodal rgb-rf fusion framework with state space models for remote physiological measurement. arXiv preprint arXiv:2502.13624, 2025. 1

[45] Lin Xi, Weihai Chen, Changchen Zhao, Xingming Wu, and Jianhua Wang. Image enhancement for remote photoplethysmography in a low-light environment. In 2020 15th IEEE International Conference on Automatic Face and Gesture Recognition (FG 2020), pages 1–7. IEEE, 2020. 5, 6, 7, 8, 1

[46] Lin Xi, Weihai Chen, Changchen Zhao, Xingming Wu, and Jianhua Wang. Image enhancement for remote photoplethysmography in a low-light environment. In 2020 15th IEEE International Conference on Automatic Face and Gesture Recognition (FG 2020), pages 1–7. IEEE, 2020. 1

[47] Yiping Xie, Zitong Yu, Bingjie Wu, Weicheng Xie, and Linlin Shen. Sfda-rppg: Source-free domain adaptive re mote physiological measurement with spatio-temporal con sistency. arXiv preprint, 2024. arXiv:2409.12040. 1

[48] Yiping Xie, Bo Zhao, Mingtong Dai, Jian-Ping Zhou, Yue Sun, Tao Tan, Weicheng Xie, Linlin Shen, and Zitong Yu. Physllm: Harnessing large language models for cross-modal remote physiological sensing. arXiv preprint arXiv:2505.03621, 2025. 5, 6, 1

[49] Zhenghan Yu, Geetha Balakrishnan, Xuhua Zhao, Qiang Li, Senem Velipasalar, and Min Wu. Remote photoplethysmograph signal measurement from facial videos using spatiotemporal networks. In British Machine Vision Conference (BMVC), 2019. 1, 2, 5, 6, 7, 8

[50] Zitong Yu, Xiaobai Li, and Guoying Zhao. Remote photo plethysmograph signal measurement from facial videos using spatio-temporal networks. In BMVC, 2019. 1

[51] Zhenghan Yu, Xuhua Zhao, Geetha Balakrishnan, and et al. Physformer: Facial video-based physiological measurement with temporal difference transformer. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV), pages 135–144, 2021. 2, 7

[52] Zitong Yu, Yuming Shen, Jingang Shi, Hengshuang Zhao, Philip HS Torr, and Guoying Zhao. Physformer: Facial video-based physiological measurement with temporal dif ference transformer. In Proceedings of the IEEE/CVF con ference on computer vision and pattern recognition, pages 4186–4196, 2022. 5, 6, 8

[53] Yan Zeng, Ruichu Cai, Fuchun Sun, Libo Huang, and Zhifeng Hao. A survey on causal reinforcement learning. IEEE Transactions on Neural Networks and Learning Sys tems, 2024. 1

[54] Yuting Zhang, Hao Lu, Xin Liu, Yingcong Chen, and Kaishun Wu. Advancing generalizable remote physiological measurement through the integration of explicit and implicit prior knowledge. IEEE Transactions on Image Processing, 2025. 5, 6

[55] Kaiyang Zhou, Yongxin Yang, Yu Qiao, and Tao Xiang. Domain generalization: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2021. 2

[56] Yijie Zhu, Yibo Lyu, Zitong Yu, Rui Shao, Kaiyang Zhou, and Liqiang Nie. Emosym: A symbiotic framework for unified emotional understanding and generation via latent reasoning. In Proceedings of the 33nd ACM International Conference on Multimedia, 2025. 2

[57] Yijie Zhu, Lingsen Zhang, Zitong Yu, Rui Shao, Tao Tan, and Liqiang Nie. Uniemo: Unifying emotional understanding and generation with learnable expert queries. arXiv preprint arXiv:2507.23372, 2025. 2

[58] Yijie Zhu, Rui Shao, Ziyang Liu, Jie He, Jizhihui Liu, Jiuru Wang, and Zitong Yu. H-gar: A hierarchical interaction framework via goal-driven observation-action refinement for robotic manipulation. In Proceedings of the AAAI Conference on Artificial Intelligence, 2026. 1

[59] Bochao Zou, Zizheng Guo, Jiansheng Chen, Junbao Zhuo, Weiran Huang, and Huimin Ma. Rhythmformer: Extracting patterned rppg signals based on periodic sparse attention. Pattern Recognition, 164:111511, 2025. 5, 6, 7, 8

# FLOW: Optimal Transport-Driven Feature Warping for Generalized Remote Physiological Measurement

Supplementary Material

## A. Introduction to the Datasets

UBFC-rPPG [1] contains 42 RGB facial videos from 42 distinct subjects. Each video is captured at 640×480 pixel resolution and 30 frames per second (fps). Recordings take place under varied lighting conditions, including natural sunlight and indoor artificial illumination. Ground-truth physiological signals are recorded via a CMS50E pulse oximeter at 60 Hz, ensuring precise temporal alignment for evaluation.

PURE [34] comprises 60 high-quality RGB videos collected from 10 subjects performing six different head movement scenarios (static, talking, translation movements, etc.). Videos are recorded at 30 fps under consistent indoor lighting and controlled background settings, minimizing external interference. Synchronized physiological measurements are obtained using a CMS50E oximeter sampling at 60 Hz. PURE is particularly valuable for evaluating rPPG performance during facial movements.

BUAA-MIHR [45] is designed to assess algorithmic robustness across varying illumination intensities. The dataset features video sequences recorded under a range of controlled lighting conditions, from low-light (below 10 lux) to normal brightness. In our experiments, we only utilize videos captured under illumination levels ≥10 lux, as extremely dim lighting introduces significant image degradation requiring specialized enhancement techniques beyond this study’s scope.

MMPD [38] comprises 660 videos, each lasting one minute, collected from 33 subjects with diverse skin tones and gender distributions. Each video is recorded at 30 fps with a resolution of 320×240 pixels, under four distinct lighting conditions (bright, warm, dim, and colored lighting). Subjects perform various daily activities, introducing intra-subject variability and further increasing dataset complexity.

## B. Experimental Settings

Datasets and Evaluation Metrics We evaluate our method on four widely used remote photoplethysmography (rPPG) datasets: UBFC-rPPG [2], PURE [35], BUAA-MIHR [46] and MMPD [38]. Following prior works [37, 50], we adopt three standard evaluation metrics: mean absolute error (MAE), root mean square error (RMSE), and Pearson’s correlation coefficient (R), to assess the accuracy of predicted heart rates (HRs). For both MAE and RMSE, lower values indicate smaller prediction errors, while higher values of R (closer to 1.0) indicate stronger linear correlation with the ground-truth HRs. MAE and RMSE are reported in beats per minute (bpm); for brevity, we omit these units in subsequent tables and discussions.

Implementation Details Our experiments are implemented in PyTorch, primarily based on the rPPG-Toolbox [20]. For preprocessing, we detect and crop the face region from the first frame of each video clip and apply a fixed bounding box across subsequent frames. Each video is resampled to a consistent frame rate of 30 fps, and a random chunk of 128 frames is selected, resized to $1 2 8 \times 1 2 8$ pixels. We adopt the PhysLLM [48] as the baseline. The hyperparameters $\alpha = 0 . 8$ and $l _ { t a r g e t } = 3 2$ are set by default. The LLM is trained using the Adam optimizer with an initial learning rate of $1 \times 1 0 ^ { - 4 }$ and a weight decay of $5 \times 1 0 ^ { - 5 }$ . The entire model is trained for 20 epochs on an NVIDIA H100 GPU with a batch size of 4.

## C. Theoretical Proof of the Generalization Bound

In this section, we provide a detailed derivation of the proposed multi-source generalization bound under the conditional optimal transport (OT) geometry used in FLOW. Our goal is to connect the quality of cross-domain alignment— measured under a task-aware OT cost—to the prediction risk on an unseen target domain.

## Preliminaries

Data and hypothesis space. Let $\{ D _ { s } \} _ { s = 1 } ^ { m }$ be the source domains with mixing coefficients $\alpha _ { s } \mathrm { ~  ~ { ~ \geq ~ } ~ 0 ~ }$ such that $\textstyle \sum _ { s = 1 } ^ { m } \alpha _ { s } \ = \ 1$ , and let T denote the (target) test domain. Each sample is denoted by $z = ( x , h r )$ , where $x \in \mathbb { R } ^ { d }$ is the visual-temporal feature and hr $\in \mathbb { R }$ is the continuous heart rate label. A hypothesis $h \in \mathcal H$ produces a prediction $\hat { y }$ from $x .$ We assume a bounded loss $\ell ( h ( x ) , y ) \in [ 0 , 1 ]$ For a distribution $P ,$ the expected risk is

$$
R _ { P } ( h ) = \mathbb { E } _ { ( z , y ) \sim P } [ \ell ( h ( x ) , y ) ] .\tag{17}
$$

Task-driven conditional cost. Following our formulation in the main paper, we define a conditional ground cost c that jointly accounts for feature distance and physiological consistency:

$$
\begin{array} { r } { c \bigl ( ( x , h r ) , ( x ^ { \prime } , h r ^ { \prime } ) \bigr ) = \| x - x ^ { \prime } \| _ { W } ^ { 2 } + \lambda _ { h r } \Bigl ( 1 - \exp \bigl ( - \frac { ( h r - h r ^ { \prime } ) ^ { 2 } } { 2 \sigma ^ { 2 } } \bigr ) \Bigr ) , } \end{array}\tag{18}
$$

where $\begin{array} { c c l } { { W } } & { { = } } & { { \mathrm { d i a g } ( w _ { 1 } , \ldots , w _ { d } ) } } \end{array}$ is the learnable frequency/feature weighting matrix, and $\lambda _ { h r } , \sigma > 0$ control the influence and bandwidth of the heart-rate kernel term. This cost emphasizes both semantic similarity in feature space and coherence in underlying physiological signals. Conditional OT distance. Given two distributions $P , Q$ on $( x , h r )$ , we define the conditional OT distance:

$$
W _ { c } ( P , Q ) = \operatorname* { i n f } _ { \pi \in \Pi ( P , Q ) } \int c ( z , z ^ { \prime } ) d \pi ( z , z ^ { \prime } ) ,\tag{19}
$$

where $\Pi ( P , Q )$ is the set of couplings with marginals $P$ and $Q .$ Throughout, we assume c induces a valid (or pseudo-)metric structure compatible with the OT geometry.

OT barycenter over sources. We consider the (conditional) OT barycenter $B$ of $\{ D _ { s } \} _ { s = 1 } ^ { m } \mathrm { . }$

$$
B = \arg \operatorname* { m i n } _ { \nu } \sum _ { s = 1 } ^ { m } \alpha _ { s } W _ { c } ( D _ { s } , \nu ) .\tag{20}
$$

Intuitively, B captures a geometry-aware “anchor” distribution that balances all source domains under $W _ { c }$

Regularized OT and residual terms. In practice, we employ a debiased Sinkhorn divergence to approximate $W _ { c } ,$ , together with structure-preserving regularization terms (e.g., identity-preserving constraints) used in FLOW. We collect these deviations into:

$\Delta _ { \mathrm { s i n k } }$ : the residual bias between the ideal $W _ { c }$ and its debiased Sinkhorn approximation;

$\Delta _ { \mathrm { i d } } \mathrm { : }$ the residual bias introduced by identity-preserving regularization (and related structure-preserving penalties) that slightly perturb the ideal OT geometry.

Both $\Delta _ { \mathrm { s i n k } }$ and $\Delta _ { \mathrm { i d } }$ are treated as small non-negative constants controlled by regularization strength and optimization accuracy.

## Risk Discrepancy under Conditional OT

We first relate the risk difference between two domains to their conditional OT distance.

Lemma 1 (Task-related discrepancy bound). Let $g ( z ) =$ $\mathbb { E } [ \ell ( h ( x ) , y ) \mid z ]$ . Assume $g$ is $L _ { c ^ { - 1 } }$ Lipschitz with respect to the metric induced by $c , \mathrm { i . e . }$

$$
| g ( z ) - g ( z ^ { \prime } ) | \leq L _ { c } d _ { c } ( z , z ^ { \prime } ) \leq L _ { c } c ( z , z ^ { \prime } ) .\tag{21}
$$

Then for any distributions $P , Q$

$$
| R _ { P } ( h ) - R _ { Q } ( h ) | \leq L _ { c } W _ { c } ( P , Q ) + \Delta _ { \mathrm { i d } } .\tag{22}
$$

Proof. Let $\pi ^ { \star } \in \Pi ( P , Q )$ be an optimal coupling for $W _ { c }$ Then

$$
R _ { P } ( h ) - R _ { Q } ( h ) = \mathbb { E } _ { P } [ g ( z ) ] - \mathbb { E } _ { Q } [ g ( z ^ { \prime } ) ] = \int \int ( g ( z ) - g ( z ^ { \prime } ) )\tag{23}
$$

By the Lipschitz property, $| g ( z ) - g ( z ^ { \prime } ) | \leq L _ { c } c ( z , z ^ { \prime } )$ , thus

$$
| R _ { P } ( h ) - R _ { Q } ( h ) | \leq L _ { c } \iint c ( z , z ^ { \prime } ) d \pi ^ { \star } ( z , z ^ { \prime } ) = L _ { c } W _ { c } ( P , Q ) + \Delta _ { \mathrm { i d } } ,\tag{24}
$$

where $\Delta _ { \mathrm { i d } }$ accounts for the slight distortion introduced by identity-preserving regularization in the practical alignment. □

## From Sources to Barycenter and Target

Using Lemma 1, we control the risk at the barycenter B and then transfer it to the target domain T.

Lemma 2 (Multi-source to barycenter). For any $h \in \mathcal H$

$$
R _ { B } ( h ) \leq \sum _ { s = 1 } ^ { m } \alpha _ { s } R _ { D _ { s } } ( h ) + L _ { c } \sum _ { s = 1 } ^ { m } \alpha _ { s } W _ { c } ( D _ { s } , B ) + \Delta _ { \mathrm { i d } } .\tag{25}
$$

Proof. Apply Lemma 1 with $( P , Q ) = ( D _ { s } , B )$ and then average with weights $\alpha _ { s } \mathrm { : }$

$$
R _ { B } ( h ) \leq \sum _ { s } \alpha _ { s } R _ { D _ { s } } ( h ) + L _ { c } \sum _ { s } \alpha _ { s } W _ { c } ( D _ { s } , B ) + \Delta _ { \mathrm { i d } } .\tag{26}
$$

□

Lemma 3 (Barycenter to target). For any $h \in \mathcal H$

$$
R _ { T } ( h ) \leq R _ { B } ( h ) + L _ { c } W _ { c } ( B , T ) + \Delta _ { \mathrm { i d } } .\tag{27}
$$

Proof. Apply Lemma 1 with $( P , Q ) = ( B , T )$ directly. □

Lemma 4 (Two-hop barycenter inequality). Assume $W _ { c }$ satisfies the triangle inequality and $B$ is the OT barycenter defined above. Then

$$
\sum _ { s = 1 } ^ { m } \alpha _ { s } W _ { c } ( D _ { s } , B ) + W _ { c } ( B , T ) \leq 2 \sum _ { s = 1 } ^ { m } \alpha _ { s } W _ { c } ( D _ { s } , T ) .\tag{28}
$$

Sketch. By barycenter optimality, for any ν (in particular $\nu = T )$ ,

$$
\sum _ { s } \alpha _ { s } W _ { c } ( D _ { s } , B ) \leq \sum _ { s } \alpha _ { s } W _ { c } ( D _ { s } , T ) .\tag{29}
$$

By triangle inequality, $W _ { c } ( D _ { s } , T ) ~ \leq ~ W _ { c } ( D _ { s } , B ) ~ +$ $W _ { c } ( B , { T } )$ , hence

$$
W _ { c } ( B , T ) \leq W _ { c } ( D _ { s } , T ) - W _ { c } ( D _ { s } , B ) + ( \mathrm { n o n - n e g a t i v e ~ t e r m } ) .\tag{30}
$$

Aggregating over s and combining the two relations yields an upper bound of the two-hop quantity by a constant factor of the direct discrepancies $\{ W _ { c } ( D _ { s } , T ) \}$ ; the above inequality is a convenient sufficient form. A fully rigorous derivation can be obtained by exploiting convexity of OT distances and the characterization of Wasserstein barycenters; we omit the routine details for brevity. □

## Conditional OT Generalization Bound

We now combine the above lemmas to obtain the main bound.

Theorem 1 (Conditional OT barycenter generalization bound). For any $h \in \mathcal H$

$$
\begin{array} { r l r } {  { R _ { T } ( h ) \leq \sum _ { s = 1 } ^ { m } \alpha _ { s } R _ { D _ { s } } ( h ) + L _ { c } \Big ( \displaystyle \sum _ { s = 1 } ^ { m } \alpha _ { s } W _ { c } ( D _ { s } , B ) + W _ { c } ( B , T ) \Big ) } } \\ & { } & { + \Delta _ { \mathrm { s i n k } } + \Delta _ { \mathrm { i d } } + \Delta _ { \mathrm { e s t } } . \qquad ( 3 1 ) \qquad \quad } \end{array}
$$

where $\Delta _ { \mathrm { e s t } }$ denotes standard statistical estimation errors from finite samples.

Proof. Starting from Lemma 3,

$$
R _ { T } ( h ) \leq R _ { B } ( h ) + L _ { c } W _ { c } ( B , T ) + \Delta _ { \mathrm { i d } } ,\tag{32}
$$

then substitute Lemma 2 for $R _ { B } ( h )$ to obtain

$$
\begin{array} { l } { { \displaystyle R _ { T } ( h ) \leq \sum _ { s } \alpha _ { s } R _ { D _ { s } } ( h ) + L _ { c } \sum _ { s } \alpha _ { s } W _ { c } ( D _ { s } , B ) } } \\ { { \displaystyle ~ + ~ L _ { c } W _ { c } ( B , T ) + \Delta _ { \mathrm { i d } } . } } \end{array}\tag{33}
$$

Approximating $W _ { c }$ with the debiased Sinkhorn divergence contributes an additional $\Delta _ { \mathrm { s i n k } }$ , and empirical estimation of risks contributes $\Delta _ { \mathrm { { e s t } } } .$ , leading to the stated inequality. □

## Comparison with Traditional Geometric Bounds

We compare our conditional OT-based discrepancy with the standard Euclidean Wasserstein-1 bound for multi-source domain generalization.

Classical bound. A typical geometric bound using the Wasserstein-1 distance $W _ { 1 }$ has the form:

$$
R _ { T } ( h ) \leq \sum _ { s = 1 } ^ { m } \alpha _ { s } R _ { D _ { s } } ( h ) + L _ { 1 } \sum _ { s = 1 } ^ { m } \alpha _ { s } W _ { 1 } ( D _ { s } , T ) + \lambda ^ { \star } + \tilde { \Delta } ,\tag{34}
$$

where $L _ { 1 }$ is a Lipschitz constant under the Euclidean metric, $\lambda ^ { \star }$ denotes an irreducible joint error term, and $\tilde { \Delta }$ collects statistical/approximation errors.

Theorem 2 (Relative tightness under conditional geometry). Assume: (i) bounded feature domain $\| x \| \leq M _ { x }$ (ii) bounded weight matrix $\| W \| _ { \mathrm { o p } } \le \Lambda _ { W } ;$ (iii) heart-rate kernel term is $L _ { h r ^ { - } } \mathrm { I }$ ipschitz; and (iv) the conditional metric removes irrelevant (non-physiological) directions so that there exists $0 < r < 1$ with $L _ { c } \leq r L _ { 1 }$ . Then there exists a constant $\kappa > 0$ such that

$$
W _ { c } ( P , Q ) \leq \kappa W _ { 1 } ( P , Q ) ,\tag{35}
$$

and, combining Lemma 4 with Theorem 1, our discrepancy term satisfies

$$
\begin{array} { r c l } { \displaystyle \mathrm { D i f f _ { o u r s } } } & { \lesssim } & { \displaystyle 2 L _ { c } \sum _ { s = 1 } ^ { m } \alpha _ { s } W _ { c } ( D _ { s } , T ) } \\ & & { \leq } & { \displaystyle 2 r \kappa L _ { 1 } \sum _ { s = 1 } ^ { m } \alpha _ { s } W _ { 1 } ( D _ { s } , T ) . } \end{array}\tag{36}
$$

up to $( \Delta _ { \mathrm { s i n k } } + \Delta _ { \mathrm { i d } } + \Delta _ { \mathrm { e s t } } )$ . Under standard boundedness nd normalization assumptions, both r and κ remain moderate constants. In particular, when $2 r \kappa < 1$ , which is a mild condition that can be encouraged in practice via feature normalization and regularization, the conditional OT geometry yields a strictly tighter or comparable discrepancy term than the Euclidean Wasserstein-1 counterpart. □

## D. Training and Inference Procedure of FLOW

In this section, we provide additional details regarding the training and inference pipeline of FLOW. During training, TRM first stabilizes intermediate representations, after which PCOT computes soft cross-temporal correspondences to align features across domains, as summarized in Algorithm 1. The regularization terms jointly ensure consistent prototype usage and preserve feature identity throughout the alignment process. At inference time, FLOW operates without requiring domain labels or additional preprocessing, producing temporally coherent and domain-invariant representations that support robust physiological signal prediction, as illustrated in Algorithm 2. For clarity, we summarize the full procedure in the following subsections.

Algorithm 1: Training of FLOW   
Input: Multi-source videos $\{ X _ { i } \} ,$ domains $\{ d _ { i } \}$   
rPPG signals $\{ s _ { i } \} ;$ hyper-parameters   
${ \lambda _ { O T } } , { \lambda _ { s c } } , { \lambda _ { \mathrm { i d } } } , { \bar { \alpha } } , \mathrm { \bar { \beta } } , \eta , E , B .$   
Output: Parameters $\theta _ { b } , \theta _ { t } , \theta _ { p } , \theta _ { r } ;$ prototype bank   
$P = \{ P _ { d } \} .$   
Initialize f<sub>backbone</sub>, $f _ { \mathrm { T R M } } , \mathrm { P C O T } , h _ { \mathrm { r e g } } ,$ optimizer, $P ;$   
for epoch = 1 to $E$ do   
sample minibatch $\{ ( X _ { i } , d _ { i } , s _ { i } ) \} _ { i = 1 } ^ { B } ;$   
// 1. HR labels from $\tt F F T$   
$y _ { i } \gets \mathrm { F F T } \mathbf { \mathrm { . H R } } ( s _ { i } )$ for all $i ;$   
// 2. Backbone + TRM   
$F _ { i } \gets f _ { \mathrm { b a c k b o n e } } ( X _ { i } ; \theta _ { b } ) ;$   
$\begin{array} { r } { C _ { i } \gets f _ { \mathrm { T R M } } ( F _ { i } ; \theta _ { t } ) ; } \end{array}$   
flatten {C } to $F _ { \mathrm { f l a t } } = \{ F _ { n } \} _ { n = 1 } ^ { N }$ , repeat $d _ { i } , y _ { i }$ to   
d<sub>flat</sub>, y<sub>flat</sub>;   
$/ / 3$ PCOT with label-aware   
cost   
for $n = 1$ to N do   
select domain prototypes $P _ { n }$ from $P _ { d _ { \mathrm { f l a t } } [ n ] } ;$   
for $j = 1$ to $K$ do   
${ \dot { C } } [ n , j ] =$   
α dist $( F _ { n } , P _ { n } [ j ] ) + \beta | y _ { \mathrm { f l a t } } [ n ] - \bar { y } _ { P _ { n } [ j ] } | ;$   
compute transport plan $\Pi ^ { \star } = \mathrm { O T } .$ Solver(C);   
$\begin{array} { r } { \mathcal { L } _ { O T } = \sum _ { n , j } \Pi _ { n , j } ^ { \star } C [ n , j ] ; } \end{array}$   
$\begin{array} { r } { \mathcal { L } _ { s c } = \frac { 1 } { N } \sum _ { n } } \end{array}$ min<sub>j</sub> dist $( F _ { n } , P _ { n } [ j ] ) ^ { 2 } ;$   
$\begin{array} { r } { A _ { n } = \ddot { \sum _ { j } } \Pi _ { n , j } ^ { \star } P _ { n } [ j ] } \end{array}$ for all n;   
$/ / 4 .$ Regression loss   
aggregate $\left\{ A _ { n } \right\}$ by sample to get $A _ { i } ;$   
$\begin{array} { r } { \hat { y } _ { i } = h _ { \mathrm { r e g } } ( A _ { i } ; \theta _ { r } ) ; } \end{array}$   
$\begin{array} { r } { \mathcal { L } _ { \mathrm { t a s k } } = \frac { 1 } { B } \sum _ { i } ( \hat { y } _ { i } - y _ { i } ) ^ { 2 } ; } \end{array}$   
$/ / 5$ Identity-preserving loss   
$\begin{array} { r } { \mathcal { L } _ { \mathrm { i d } } = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \lVert A _ { n } - F _ { n } \rVert _ { 2 } ^ { 2 } ; } \end{array}$   
// 6. Total loss and update   
$\mathcal { L } = \mathcal { L } _ { \mathrm { t a s k } } + \lambda _ { O T } \mathcal { L } _ { O T } + \lambda _ { s c } \mathcal { L } _ { s c } + \lambda _ { \mathrm { i d } } \mathcal { L } _ { \mathrm { i d } } ;$   
update $\theta _ { b } , \theta _ { t } , \theta _ { p } , \theta _ { r }$ by SGD on $\mathcal { L } ;$   
P ← UpdatePrototypes $( P , \{ A _ { n } \} , d _ { \mathrm { f l a t } } ) ;$   
return $\theta _ { b } , \theta _ { t } , \theta _ { p } , \theta _ { r } , P ;$

Algorithm 2: Inference of FLOW   
Input: Trained $\theta _ { b } , \theta _ { t } , \theta _ { p } , \theta _ { r } ,$ prototype bank $P ,$ test   
video $X .$   
Output: Predicted heart rate $\hat { y } .$   
$/ / \perp$ Backbone + TRM   
$F \gets f _ { \mathrm { b a c k b o n e } } ( X ; \theta _ { b } ) ;$   
$C \gets f _ { \mathrm { T R M } } ( F ; \theta _ { t } ) ;$   
flatten $C$ to $F _ { \mathrm { f l a t } } = \{ F _ { n } \} _ { n = 1 } ^ { N } ;$   
$/ / 2$ PCOT   
for $n = 1$ to $N$ do   
choose prototypes $P _ { n } \left( \mathbf { e . g . } \right.$ , corresponding   
domain or all domains);   
for $j = 1$ to K do   
$\big \lfloor \overrightarrow { C } [ n , j ] = \mathrm { d i s t } ( F _ { n } , P _ { n } [ j ] ) ;$   
Π = OT Solver(C) (or nearest-prototype);   
$\begin{array} { r } { A _ { n } = \sum _ { j } \Pi _ { n , j } P _ { n } [ j ] } \end{array}$ for all $n ;$   
$/ / 3$ Regression   
aggregate $\left\{ A _ { n } \right\}$ to video-level $A ;$   
$\begin{array} { r } { \hat { y } = h _ { \mathrm { r e g } } ( A ; \theta _ { r } ) ; } \end{array}$   
return $\hat { y } ;$