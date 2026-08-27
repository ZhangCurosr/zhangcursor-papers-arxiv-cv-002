# Semi-Supervised Adaptation of Vision-Language Models for Image Classification

Mohamed L. Mekhalfi, Senior Member, IEEE, Mohamad M. Al Rahhal, Senior Member, IEEE, Yakoub Bazi, Senior Member, IEEE, Salah E. Khenfer, Mingdeng Shi, Hua Zou, and Mansour Zuair, Member, IEEE

Abstract—Vision-language models like CLIP have shown significant potential in handling natural images, yet their performance is often limited by the distinct characteristics of satellite imagery. While parameter-efficient adaptation techniques exist, their efficacy is frequently limited by the scarcity of annotated samples. In this letter, we propose Self-Evolutionary CLIP (SE-CLIP), a semi-supervised framework designed for recursive label mining in scene classification. The approach follows a dual-phase pipeline, where an initial warm-up on a few annotated seeds is followed by a recursive discovery phase that iteratively identifies high-confidence samples from unlabeled pools. To maintain the integrity of the evolving support set, we employ a class-balanced selection strategy that prevents the model from being dominated by easily learned categories. Results on the UCM and NWPU benchmarks indicate that SE-CLIP significantly outperforms existing semi-supervised approaches. The framework provides a viable solution for adapting VLMs to the remote sensing domain with minimal human intervention.

Index Terms—Vision-language models, semi-supervised learning, satellite imagery, image classification.

## I. INTRODUCTION

[1], has revolutionized visual representation learning. By establishing a unified latent space, these models enable zeroshot classification in Remote Sensing (RS) through semantic alignment. However, a critical performance gap persists due to the significant domain discrepancy between general-purpose pre-training and the specialized spectral-spatial geometries of satellite imagery.

To mitigate this domain discrepancy, parameter-efficient fine-tuning techniques, such as Low-Rank Adaptation (LoRA), have been introduced [2]. By updating only a small subset of additional parameters while keeping the backbone weights frozen. These methods allow for domain specialization without the computational burden of full fine-tuning. Yet, the efficacy of such adaptation remains heavily dependent on the availability of high-quality, expert-annotated labels. In the RS domain, where data labeling requires specialized geographical knowledge and is notoriously labor-intensive, this dependence creates a critical bottleneck for large-scale monitoring applications.

Semi-supervised learning (SSL) offers a potential solution by leveraging vast amounts of unlabeled data. However, traditional SSL frameworks often struggle with the noisy nature of satellite imagery, leading to confirmation bias or the reinforcement of majority-class errors during pseudo-labeling.

Existing semi-supervised frameworks, ranging from holistic consistency regularization to curriculum-based pseudolabeling [3], [4], [5], [6], [7], [8], [9], [10], [11], primarily rely on heuristic or self-adaptive thresholding to filter unlabeled samples. While effective in closed-set scenarios, these methods encounter three critical bottlenecks in the RS domain. First, thresholding-based selection is highly sensitive to the initial model confidence. Precisely, in the presence of a domain gap, low zero-shot accuracy leads to the exclusion of informative samples or the inclusion of high-confidence errors. Second, the reliance on absolute thresholds in these methods introduces a significant risk of including erroneous labels. For instance, samples from easily distinguishable categories frequently exceed the confidence threshold, while complex or visually ambiguous classes are systematically excluded. This imbalance creates a self-reinforcing bias that biases the adapted feature space toward majority-class characteristics, ultimately degrading the model’s sensitivity to the full diversity of land-use semantics. Third, these works remain confined to uni-modal architectures with standard linear heads, failing to exploit the rich, high-dimensional semantic priors available in Vision-Language Models (VLMs). By ignoring the cross-modal alignment capabilities of VLMs like CLIP, traditional SSL methods overlook a robust initialization source that can guide the discovery of unlabeled samples even before extensive fine-tuning.

Our framework addresses these gaps by replacing rigid thresholding with a recursive discovery protocol anchored in CLIP’s semantic latent space. This framework, termed SE-CLIP (Self-Evolutionary CLIP), transforms the adaptation process into a dual-phase pipeline that initiates from a sparse set of expert-labeled seeds. Unlike traditional methods that rely on static pseudo-labeling, SE-CLIP utilizes fixed textual anchors to guide the label discovery and self-training of the model. By parameterizing the similarity scaling factor as a learned multiplicative variable, the framework adaptively calibrates its alignment confidence, enabling the robust identification of high-confidence samples even under significant domain shifts. This process is governed by a class-balanced migration strategy that enforces an equidistribution of mined samples across the label space, ensuring a high-purity expansion of the support set and preventing dominance by easy classes.

The remainder of this letter is organized as follows. Section II details the SE-CLIP framework, including the architectural adaptation and the recursive discovery protocol. Section III presents and discusses the results. Finally, Section IV concludes the letter and suggests potential research directions.

## II. METHODOLOGY

## A. Problem Formulation

The proposed SE-CLIP framework establishes a label mining pipeline designed to bridge the domain discrepancy between general-purpose vision-language models and the specific spectral-spatial requirements of remote sensing (RS) scene classification. By integrating parameter-efficient lowrank adaptation with a cross-modal alignment objective, the proposed framework enables the transition from a zero-shot foundation model to a domain-specialized classifier.

In the context of semi-supervised remote sensing imagery classification, we consider a dataset D partitioned into a labeled support set $\mathcal { L }$ and a significantly larger unlabeled pool U. The support set is defined as $\bar { \mathcal { L } } \ = \ \backslash ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N _ { L } }$ where $x _ { i }$ denotes an image sample and $y _ { i } \in \{ 1 , \ldots , C \}$ represents its corresponding categorical label for C distinct classes. The unlabeled pool is represented as $\mathcal { U } = \{ x _ { i } \} _ { i = 1 } ^ { N _ { U } }$ where $N _ { L } \ll N _ { U }$

The primary objective is to optimize a classification function $f ( \cdot )$ that minimizes the empirical risk on L while effectively exploiting the structural information in U to improve generalization on unseen test data. We achieve this by iteratively expanding L with high-confidence samples discovered in U through a recursive alignment process.

## B. Architectural Adaptation via LoRA

The framework utilizes the Contrastive Language-Image Pre-training (CLIP) foundation model as the primary feature extractor [12]. CLIP consists of a visual encoder $f _ { \phi } ( \cdot )$ and a text encoder $g _ { \psi } ( \cdot )$ . To bridge the domain gap between natural imagery and the spectral-spatial characteristics of satellite sensors, we use Low-Rank Adaptation (LoRA) [2].

For a given frozen weight matrix $W _ { 0 } \in \mathbb { R } ^ { d \times l }$ within the Transformer layers, we inject trainable bypass matrices $A \in$ $\mathbb { R } ^ { r \times l }$ and $B \in \mathbb { R } ^ { d \times r }$ (see Fig. 1). The updated forward pass is formulated as:

$$
h _ { i } = W _ { 0 } x _ { i } + \Delta W x _ { i } = W _ { 0 } x _ { i } + B A x _ { i }\tag{1}
$$

where $r$ is the intrinsic rank, typically $r \ll \operatorname* { m i n } ( d , l )$ . This constraint ensures that the adaptation process remains focused on the most relevant features while preventing overfitting to the sparse initial labels.

![](images/108f331f53b9007d229f57896331f38d088aeb5ab31e5d2121ce7023be5e8168.jpg)  
Fig. 1: Overview of the Low-Rank Adaptation (LoRA) mechanism. Trainable matrices A and B are integrated into the frozen Transformer architecture to efficiently specialize the foundation model for remote sensing scene classification.

## C. Semantic Anchoring and Alignment

Rather than employing a traditional linear classification head, the framework utilizes fixed textual anchors $T _ { c } \in \mathbb { R } ^ { d }$ for each class c. These anchors are generated by encoding class-specific prompts through the frozen text encoder. The visual representation of an image $x _ { i }$ is mapped to a vector $V _ { i } = f _ { \phi + \Delta W } ( x _ { i } )$ , and the model is trained to align $V _ { i }$ with its respective anchor $T _ { c }$

The learning objective is defined by a cross-entropy loss over the temperature-scaled cosine similarities:

$$
\mathcal { L } = - \log \frac { \exp ( \cos ( V _ { i } , T _ { G T } ) \cdot e ^ { \tau } ) } { \sum _ { j = 1 } ^ { C } \exp ( \cos ( V _ { i } , T _ { j } ) \cdot e ^ { \tau } ) }\tag{2}
$$

where $T _ { G T }$ represents the ground-truth textual anchor, and $e ^ { \tau }$ denotes the constant log-parameterized temperature scale inherited directly from the pre-trained CLIP baseline to preserve calibrated cross-modal alignment.

## D. Recursive Discovery

The training follows a recursive protocol designed to eliminate manual labeling requirements. Initially, the model undergoes a warm-up phase where the LoRA parameters are updated exclusively on the initial expert seeds to stabilize the gradients.

Upon completion of the warm-up, the model initiates the discovery phase. For each class $c ,$ the system ranks all samples in U according to their similarity to $T _ { c }$ . A budget of k samples is then migrated to the support set and never re-evaluated:

$$
{ \mathcal { M } } _ { c } = \{ x _ { i } \in { \mathcal { U } } \mid \operatorname { s o r t } ( \cos ( V _ { i } , T _ { c } ) ) \leq k \}\tag{3}
$$

where sort(·) sorts the unlabeled pool U in descending order based on their similarity to the text anchor $T _ { c }$

The discovered set $\mathcal { M } _ { c }$ is appended to ${ \mathcal { L } } ,$ and the process repeats. This recursive expansion ensures a balanced and highpurity growth of the labeled pool, facilitating the continuous evolution of the adapted visual encoder. By enforcing a uniform budget k across all categories, the framework ensures a strictly class-balanced expansion of the support set. This equidistribution prevents the optimization process from being dominated by high-frequency or easily discriminable classes. The flowchart of the proposed method is given in Fig. 2.

![](images/aa654306acd94a89e8b048bac054533eb138763c900702ec012326aa5ce29d55.jpg)  
Fig. 2: The SE-CLIP recursive discovery pipeline. The model iteratively synchronizes visual-textual representations on labeled seeds before mining the most similar unlabeled samples to expand the support set.

## III. EXPERIMENTS

## A. Datasets and Settings

Evaluations are carried out on two benchmarks, namely UC Merced (UCM) [13] and NWPU-RESISC45 (NWPU) [14]. The UCM dataset contains 2100 aerial images evenly distributed across 21 land-use categories, whereas the large-scale NWPU dataset contains 31,500 images across 45 classes (700 per class). All scene images are resized to $2 5 6 \times 2 5 6$ pixels. The framework is implemented in PyTorch utilizing OpenAI’s pre-trained ViT-B/32 CLIP model as the core multimodal backbone. Optimization is executed via Adam with a learning rate of $1 0 ^ { - 4 } .$ . The warm up iterations were set to 10, and the total training iterations to 200. All the experiments have been run five times with different random seeds, and the mean ± standard deviation is reported in each case.

Following prior works regarding data division [11], 80% of the dataset is allocated for training, where only five seeds per class are initially labeled, and the remainder is treated as unlabeled data, while the remaining 20% is strictly reserved for testing. Classification accuracy is adopted as a metric.

## B. Quantitative Analysis

To evaluate the sensitivity of the proposed framework to the pseudo-label mining budget, Table I reports the classification scores across various k values with a fixed LoRA rank of $r = 4 .$ . For the UCM dataset, the performance remains highly stable, achieving a peak accuracy of 98.81% at $k = 1$ before experiencing a marginal degradation as k increases. This suggests that a smaller, high-confidence mining budget is sufficient for less complex scene distributions. Conversely, on the more challenging and large-scale NWPU dataset, performance scales directly with the mining budget, rising from 88.56% at k = 1 to a peak of 94.92% at $k = 5$ . This improvement underscores that complex, multi-class remote sensing environments benefit from a larger pool of mined pseudo-labels per iteration to effectively align the visual embeddings with the text anchors. However, increasing k further to 7 introduces a slight performance drop (94.77%), suggesting the inclusion of noisy or misclassified samples into the training pool.

TABLE I: SCORES (%) ON UCM AND NWPU DATASETS WITH RANK $r = 4$ ACROSS VARIOUS k VALUES
<table><tr><td rowspan="2">Dataset</td><td colspan="5">Top-k</td></tr><tr><td>1</td><td>2</td><td>5</td><td>7</td><td></td></tr><tr><td>UCM</td><td> $\mathbf { \overline { { 9 8 . 8 1 \pm 0 . 6 9 } } }$ </td><td> $\overline { { 9 8 . 6 2 \pm 0 . 3 9 } }$ </td><td> $\overline { { 9 8 . 6 2 \pm 0 . 4 9 } }$ </td><td></td><td> $9 8 . 3 8 \pm 0 . 7 8$ </td></tr><tr><td>NWPU</td><td> $8 8 . 5 6 \pm 1 . 3 2$ </td><td> $9 1 . 5 2 \pm 0 . 4 8$ </td><td> $\mathbf { 9 4 . 9 2 \ : \pm { \ : 0 . 3 1 } }$ </td><td></td><td> $9 4 . 7 7 \pm 0 . 4 8$ </td></tr></table>

In the second experiment, Table II evaluates the impact of the LoRA rank r under the values established previously $( k = 1$ for UCM and $k = 5$ for NWPU). The results demonstrate a monotonic performance improvement on both datasets as the rank scales. Specifically, increasing r from 2 to 16 improves the classification accuracy from 98.71% to 99.09% on UCM, and from 94.41% to 95.07% on NWPU. This slight gain indicates that a higher low-rank parameter space provides more learning capacity to adapt the underlying visual encoder to the complex spatial textures and highly correlated class boundaries inherent in remote sensing imagery.

The ablation results presented in Table III validate the necessity of the full recursive discovery mechanism. Relying solely on the warm-up phase causes significant performance drops on both benchmarks. This confirms that, while the warmup initialization provides a stable starting point, continuous recursive optimization is essential for the model to successfully discover and assimilate new samples without collapsing.

The evaluation in Table IV (for k = 5 and r = 4) highlights the critical role of our per-class balanced mining strategy over a standard global mining baseline (i.e., aggregating all the samples and selecting the global top ones without perclass constraints). Without class-specific constraints, global mining suffers catastrophic performance degradation on both datasets, alongside elevated standard deviations. Enforcing local per-class balance completely mitigates this instability, yielding superior accuracy and verifying that uniform category distribution is essential for robust semi-supervised adaptation.

To further study the effect of the proposed balanced perclass mining mechanism, we calculate the cumulative standard deviation in comparison to using a standard global mining baseline. As illustrated in Fig. 3, the global selection baseline exhibits an exponential surge in cumulative standard deviation immediately following the 10-iteration warm-up phase, suggesting easy class dominance. Conversely, our local selection strategy maintains a strict standard deviation of zero across all optimization iterations on both datasets, providing empirical evidence that enforcing per-class constraints successfully stabilizes pseudo-label distribution.

We further analyze the computational efficiency of SE-CLIP. As detailed in Table V, our framework optimizes only 1.16% of the CLIP model volume via LoRA. Moreover, because the recursive label-mining modules are completely discarded during deployment, our framework achieves a rapid inference latency of just 10.26 ms per batch (size=16).

TABLE II: SCORES (%) ON UCM AND NWPU DATASETS ACROSS VARIOUS LoRA RANK (r) CONFIGURATIONS
<table><tr><td rowspan="2">Dataset</td><td colspan="3">LoRA Rank (r)</td></tr><tr><td>2</td><td>4</td><td>16</td></tr><tr><td>UCM</td><td> $\overline { { 9 8 . 7 1 \pm 0 . 2 7 } }$ </td><td> $\overline { { 9 8 . 8 1 \pm 0 . 6 9 } }$ </td><td> $\overline { { { \bf 9 9 . 0 9 \pm 0 . 5 4 } } }$ </td></tr><tr><td>NWPU</td><td> $9 4 . 4 1 \pm 0 . 3 4$ </td><td> $9 4 . 9 2 \pm 0 . 3 1$ </td><td> ${ \bf 9 5 . 0 7 \pm 0 . 9 5 }$ </td></tr></table>

TABLE III: ABLATION STUDY COMPARING THE FULL RECURSIVE DISCOVERY MECHANISM AGAINST WARM-UP ONLY
<table><tr><td>Configuration</td><td>UCM</td><td>NWPU</td></tr><tr><td>Full recursive discovery</td><td> $\overline { { { \bf 9 9 . 0 9 \pm 0 . 5 4 } } }$ </td><td> $\mathbf { \overline { { 9 5 . 0 7 \ : \pm { \ : 0 . 9 5 } } } }$ </td></tr><tr><td>Warm-up only</td><td> $8 7 . 8 1 \pm 2 . 5 0$ </td><td> $7 6 . 3 4 \pm 0 . 8 7$ </td></tr></table>

TABLE IV: ABLATION STUDY COMPARING PER-CLASS BALANCED MINING AGAINST GLOBAL MINING
<table><tr><td>Configuration</td><td>UCM</td><td>NWPU</td></tr><tr><td>Per-class balanced mining</td><td> $\mathbf { \overline { { 9 8 . 6 2 \ : \pm { 0 . 4 9 } } } }$ </td><td> $\mathbf { \sigma } \mathbf { 9 4 . 9 2 \pm 0 . 3 1 }$ </td></tr><tr><td>Global mining</td><td> $8 3 . 1 9 \pm 2 . 5 5$ </td><td> $4 2 . 3 3 \pm 4 . 6 9$ </td></tr></table>

![](images/c5f1c944350dc9151ea9329222689dfbc3588640524299c31de6823276f3dd27.jpg)

![](images/665e0e280b59b419562a7516772c4f52668ef19d381c62bb64659d89d3235c62.jpg)  
Fig. 3: Cumulative pseudo-label standard deviation across iterations for global vs. local selection on UCM and NWPU.

TABLE V: COMPUTATIONAL EFFICIENCY ANALYSIS
<table><tr><td>Metric Description</td><td>Value / Footprint</td></tr><tr><td>Hardware environment</td><td>NVIDIA Tesla V100 GPU (32 GB)</td></tr><tr><td>Total parameters</td><td>152.54 M</td></tr><tr><td>Trainable parameters  $( r = 1 6 )$ </td><td>1,769,472 (1.16%)</td></tr><tr><td>GPU memory consumption</td><td>1.12 GB (batch size = 16)</td></tr><tr><td>FLOPs UCM total training time</td><td>2.95 GFLOPs/image</td></tr><tr><td>NWPU total training time</td><td>26,933.52 sec (7.48 hours) 52,649.98 sec (14.62 hours)</td></tr><tr><td>Inference latency (per batch, size=16)</td><td>10.26 ms</td></tr></table>

![](images/f8c7b13d58d8986cb3df9b4bbd7f8f6e651767de32c27f807ef5acd90bc543a4.jpg)  
Fig. 4: Accuracy (%) evolution of SE-CLIP across 200 iterations on the UCM and NWPU benchmarks. The grey dashed vertical line marks the mining onset.

TABLE VI: COMPARISON WITH STATE-OF-THE-ART SEMI-SUPERVISED METHODS
<table><tr><td>Method</td><td>UCM (%)</td><td>NWPU (%)</td></tr><tr><td>Flexmatch [3]</td><td>82.54</td><td>69.16</td></tr><tr><td>MSmatch [4]</td><td>85.70</td><td>60.00</td></tr><tr><td>RSmatch [5]</td><td>84.10</td><td>58.30</td></tr><tr><td>Freematch [6]</td><td>80.71</td><td>69.10</td></tr><tr><td>Softmatch [7]</td><td>87.38</td><td>70.09</td></tr><tr><td>Fixmatch [8]</td><td>84.84</td><td>65.89</td></tr><tr><td>SemiRS-COC [9]</td><td>85.00</td><td>59.10</td></tr><tr><td>CPL-PL [10]</td><td>87.20</td><td>61.30</td></tr><tr><td>DARP [11]</td><td>95.87</td><td>88.86</td></tr><tr><td>SE-CLIP (OURS)</td><td>99.09</td><td>95.07</td></tr></table>

## C. Qualitative analysis

The performance trajectories across optimization iterations are shown in Fig. 4. Following the 10-iteration warm-up, the activation of top-k label mining triggers a smooth accuracy ascent on both benchmarks. Due to varying dataset scale and complexity, the performance stabilizes into a plateau around iteration 60 for UCM, whereas the larger NWPU dataset requires a prolonged discovery trajectory to reach its steadystate plateau between iterations 160 and 170. As illustrated by the t-SNE visualizations [15] in Fig. 5 the zero-shot CLIP baseline exhibits highly overlapping class distributions with poorly defined boundaries. By contrast, our proposed SE-CLIP successfully reorganizes the visual embedding space into tightly bound, well-separated semantic clusters, confirming its superior capability in maximizing inter-class separability.

To verify the competitive advantage of the proposed framework, Table VI compares SE-CLIP against several state-ofthe-art semi-supervised learning methods on both benchmarks. The results show that SE-CLIP outperforms representative state-of-the-art semi-supervised frameworks. Specifically, on the UCM dataset, our method achieves an accuracy of 99.09%, outperforming the strongest baseline, DARP [11], by 3.22%. On the more challenging NWPU benchmark, SE-CLIP reaches 95.07%, with a substantial 6.21% improvement over the strongest baseline. blue The results confirm that using CLIP with parameter-efficient low-rank adaptation provides superior bimodal representation capabilities in few-shot remote sensing scenarios compared to standard semi-supervised frameworks.

## IV. CONCLUSION

In this letter, we presented SE-CLIP, a semi-supervised scene classification framework that leverages the pre-trained capabilities of the CLIP foundation model through parameterefficient fine-tuning. This is achieved by applying a balanced pseudo-label mining strategy based on fixed textual anchors. The experiments show that adapting CLIP, pretrained on natural images, via target-directed low-rank mining offers a more effective alternative for handling complex remote sensing data distributions than traditional unimodal classifiers.

Regarding operational constraints, the framework requires expressive text vocabularies to construct discriminative anchors, and performance remains limited in fine-grained scenarios with high inter-class structural similarity where semantic descriptions cannot fully resolve visual overlap. To this end, future work will focus on developing robust VLMdriven paradigms to improve cross-modal alignment for image classification using only class names [16]. ACKNOWLEDGMENT

![](images/1832790d2e38b26711d023204202597f3d30168fa8e949c6819ae11144d34adb.jpg)  
Fig. 5: Qualitative t-SNE visualization [15] of the visual embedding space comparing the pre-trained zero-shot CLIP baseline (left panels) against our proposed SE-CLIP method (right panels) across all categories for both UCM (top) and NWPU (bottom) datasets. The proposed SE-CLIP successfully adapts and reorganizes the visual embeddings into tightly bound, wellseparated semantic clusters. Distinct marker shapes and color schemes are assigned uniquely to each category to highlight cluster separability. Best viewed in color.

This research was supported by Ongoing Research Funding program (ORF-2026-995), King Saud University, Riyadh, Saudi Arabia.

## REFERENCES

[1] J. Li, Y. Li, Y. Fu, J. Liu, Y. Liu, M. Yang, and I. King, “Clip-powered domain generalization and domain adaptation: A comprehensive survey,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2026.

[2] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen et al., “Lora: Low-rank adaptation of large language models.” Iclr, vol. 1, no. 2, p. 3, 2022.

[3] B. Zhang, Y. Wang, W. Hou, H. Wu, J. Wang, M. Okumura, and T. Shinozaki, “Flexmatch: Boosting semi-supervised learning with curriculum pseudo labeling,” Advances in neural information processing systems, vol. 34, pp. 18 408–18 419, 2021.

[4] P. Gomez and G. Meoni, “Msmatch: Semisupervised multispectral scene´ classification with few labels,” IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 14, pp. 11 643– 11 654, 2021.

[5] W. Lin, J. Ma, X. Tang, X. Zhang, and L. Jiao, “Rsmatch: Semisupervised learning with adaptive category-related pseudo labeling for remote sensing scene classification,” in International Conference on Intelligence Science. Springer, 2022, pp. 220–227.

[6] Y. Wang, H. Chen, Q. Heng, W. Hou, Y. Fan, Z. Wu, J. Wang, M. Savvides, T. Shinozaki, B. Raj et al., “Freematch: Self-adaptive thresholding for semi-supervised learning,” arXiv preprint arXiv:2205.07246, 2022.

[7] H. Chen, R. Tao, Y. Fan, Y. Wang, J. Wang, B. Schiele, X. Xie, B. Raj, and M. Savvides, “Softmatch: Addressing the quantity-quality trade-off in semi-supervised learning,” arXiv preprint arXiv:2301.10921, 2023.

[8] X. Zhang, L. Huang, J. Lv, and M. Yang, “Self adaptive threshold pseudo-labeling and unreliable sample contrastive loss for semisupervised image classification,” in International Conference on Artificial Neural Networks. Springer, 2024, pp. 61–75.

[9] Q. Liu, J. Yue, Y. Kuang, W. Xie, and L. Fang, “Semirs-coc: Semisupervised classification for complex remote sensing scenes with crossobject consistency,” IEEE Transactions on Image Processing, vol. 33, pp. 3855–3870, 2024.

[10] G. Swetha, R. Datla, S. Babu, and C. K. Mohan, “Cpl-pl: Contrapositive learning-based pseudo-labeling for semi-supervised scene classification in remote sensing images,” IEEE Geoscience and Remote Sensing Letters, 2025.

[11] L. Ye, F. Shang, and H. Liu, “Semi-supervised remote sensing imagery scene classification based on probabilistic selection,” IEEE Geoscience and Remote Sensing Letters, 2025.

[12] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual models from natural language supervision,” in International Conference on Machine Learning. PMLR, 2021, pp. 8748–8763.

[13] Y. Yang and S. Newsam, “Bag-of-visual-words and spatial extensions for land-use classification,” in Proceedings of the 18th ACM SIGSPA-TIAL International Conference on Advances in Geographic Information Systems, 2010, pp. 270–279.

[14] G. Cheng, J. Han, and X. Lu, “Remote sensing image scene classification: Benchmark and state of the art,” Proceedings of the IEEE, vol. 105, no. 10, pp. 1865–1883, 2017.

[15] L. Van der Maaten and G. Hinton, “Visualizing data using t-sne,” Journal of Machine Learning Research, vol. 9, no. 11, pp. 2579–2605, 2008.

[16] O. Chakraborty, J. Dolz, and I. Ben Ayed, “Orion: Orthonormal text encoding for universal vlm adaptation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026, pp. 31 556–31 565.