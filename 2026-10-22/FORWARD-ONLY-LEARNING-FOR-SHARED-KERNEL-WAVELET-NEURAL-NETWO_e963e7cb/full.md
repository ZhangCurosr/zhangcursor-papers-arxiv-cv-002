# FORWARD-ONLY LEARNING FOR SHARED-KERNEL WAVELET NEURAL NETWORKS WITHOUT BACKPROPAGATION

Yongfei Guo<sup>1,2</sup>, Xuan Sun<sup>1,2</sup>, Ang Li<sup>1,3</sup>, Zhao Zhang<sup>1,3</sup>, Yuanhao Gong<sup>1,3</sup>

<sup>1</sup>Changchun Institute of Optics, Fine Mechanics and Physics, Chinese Academy of Sciences <sup>2</sup>University of Chinese Academy of Sciences, Beijing, China <sup>3</sup>Chinese Academy of Sciences, Beijing, China

## ABSTRACT

Backpropagation remains the dominant training paradigm, using backward gradients and local updates that can suffer from vanishing or exploding gradients in deep networks. Although the Forward-Forward algorithm avoids conventional backpropagation, it remains mainly local or layer-wise, motivating global whole-network learning using forward evaluations only. For shared-kernel wavelet neural networks, we therefore propose a CMA-ES-based global forward-only learning framework. We further introduce structure-aware parameter decomposition and channel-specific credit assignment, factorizing the search into three subspaces. On BSDS500, our method reduces test MSE from 4.1783 to 4.0130 and improves PSNR from 42.621 to 42.787 dB over Adam/BP without backward propagation. Beyond these gains, training requires only 6 training-set coverages, compared with 1500 epochs for BP. Ablations show that structural decomposition and aligned credit assignment are more important than increasing population size, demonstrating the feasibility of whole-network forward-only learning for compact reconstruction networks. These results suggest a potential route toward forward-only optimization of compact reconstruction models for resource-constrained imaging and edge or near-sensor systems.

Index Terms— Forward-only learning, CMA-ES, sharedkernel networks, image reconstruction

## 1. INTRODUCTION

Image reconstruction is a fundamental problem in computational imaging and low-level vision, where neural methods recover high-quality images from degraded or transformed measurements [1–3]. Practical systems increasingly require not only reconstruction accuracy but also low memory and computational costs. Multiscale and wavelet-inspired architectures are well suited to this goal because they capture image structures across resolutions while introducing structured analysis–synthesis operations [4–6].

![](images/e6934bc7b7ff2e04df26e25bd0d68b2c61585470bd0c571c2f6bd9ffeb12c2c5.jpg)  
Fig. 1. Local gradient-based optimization versus global forward-only population search. BP uses backward gradients for local updates. In contrast, our method evaluates complete parameter candidates through forward losses and performs population-based global search with CMA-ES.

Shared-kernel wavelet networks combine multiscale analysis–synthesis with cross-scale parameter sharing, yielding a compact trainable parameterization while retaining reconstruction capability [7]. Beyond model efficiency, such a low-dimensional parameter space can make optimization strategies impractical for large networks tractable.

Backpropagation remains the dominant neural-network training paradigm, using local first-order gradient information for iterative parameter updates [8,9]. Recent backpropagationfree methods, including the Forward-Forward algorithm, avoid conventional backward propagation but remain predominantly based on layer-local objectives [9–11]. In contrast, evolutionary and derivative-free methods evaluate populations of complete candidate solutions, enabling global exploration without gradients [12, 13]. Here, global refers to population-based search over the complete network parameterization rather than guaranteed convergence to a global optimum. Although such whole-network search is generally impractical for large neural networks, the compact parameterization of shared-kernel wavelet networks makes it tractable.

In this work, we propose a global forward-only learning framework for shared-kernel wavelet neural networks. As illustrated in Fig. 1, conventional BP relies on backward gradient propagation and local gradient-based updates, whereas our approach evaluates complete network candidates through forward reconstruction losses and adapts a population distribution using CMA-ES [13]. Compactness alone, however, does not make naive joint search effective. We therefore introduce structure-aware parameter decomposition and channelspecific credit assignment to align the search with the intrinsic network structure. The resulting framework optimizes the complete network through forward evaluations only, without automatic differentiation or backward propagation.

The main contributions are summarized as follows:

• We introduce global forward-only learning for compact shared-kernel wavelet networks, replacing backward gradients with population-based search.

• We develop structure-aware optimization with parameter decomposition and channel-specific credit assignment for effective whole-network learning.

• We validate the framework through BP comparisons, structural ablations, and multi-seed evaluations, demonstrating stable reconstruction without backpropagation.

## 2. RELATED WORK

We review two lines of work relevant to our framework: compact multiscale reconstruction networks for parameterefficient architectures and backpropagation-free learning as an alternative to conventional gradient-based training.

## 2.1. Compact Multiscale Reconstruction Networks

Multiscale reconstruction networks exploit hierarchical representations to recover image structures across resolutions. Uformer, Restormer, and NAFNet demonstrate effective multiscale or efficient restoration designs [1, 2, 4], while waveletinspired models such as WINNet and PIRNet explicitly incorporate multiresolution signal-processing structures [5, 6]. Recent lightweight models further reduce complexity through parameter-efficient transformations [14]. Shared-kernel wavelet networks extend this direction by reusing a small set of learnable kernels across scales [7]. Whereas existing studies primarily exploit compactness for architectural efficiency, we investigate whether extreme parameter sharing can also enable a different learning regime.

## 2.2. Backpropagation-Free Neural Learning

Backpropagation-free methods replace end-to-end gradient propagation with alternative learning signals. Forward-

Forward and PEPITA use layer-wise objectives or errordriven forward modulation [10, 15], with subsequent extensions remaining predominantly local or layer-wise [9, 11, 16–19]. MeZO performs zeroth-order optimization using forward evaluations [20], while evolutionary methods evaluate populations of complete candidates [12]. CMA-ES adapts the search mean, covariance, and step size for continuous black-box optimization [13], but whole-network search becomes difficult in high dimensions. We instead exploit extreme network compactness to make such search tractable and structure-aware.

## 3. GLOBAL FORWARD-ONLY LEARNING

Our framework combines a compact shared-kernel wavelet network with CMA-ES-based forward-only search. We first define the shared-kernel reconstruction model, then introduce population-based search and the structure-aware decomposition with channel-specific credit assignment.

## 3.1. Shared-Kernel Wavelet Network

We consider the shared-kernel wavelet network in Fig. 2 for reconstructing an image from a Laplacian field y with given per-channel DC information [7]. Let P denote zero padding and $\mu ( \cdot )$ the per-image, per-channel spatial mean. Let $\mathbf { d } =$ $\mu ( \mathcal P ( \mathbf x ) ) \in \mathbb { R } ^ { 3 }$ denote the per-channel DC vector of the zeropadded reference image, where $\mu ( \cdot )$ is the spatial mean. For uncorrected output $a _ { \theta } ( \mathbf { y } )$ , the final reconstruction is

$$
\hat { \mathbf { x } } = F _ { \pmb { \theta } } ( \mathbf { y } , \mathbf { d } ) = \mathcal { C } [ a _ { \pmb { \theta } } ( \mathbf { y } ) - \mu ( a _ { \pmb { \theta } } ( \mathbf { y } ) ) + \mathbf { d } ] ,\tag{1}
$$

where $\mathcal { C }$ removes the padded boundary. The same three-scalar DC information is provided to BP and CMA-ES in all stages.

The multiscale analysis–synthesis network reuses three learnable kernels, H, G, and $K .$ , across all resolutions, making its trainable parameter space independent of the number of scales while preserving a highly compact parameterization.

The learnable kernels are implemented as bias-free channelwise grouped convolutions, with $H ~ \in ~ \mathbb { R } ^ { 3 \times 1 \times 5 \times 5 }$ $G \in$ $\mathbb { R } ^ { 3 \times 1 \times 3 \times 3 }$ , and $K \in \mathbb { R } ^ { 3 \times 1 \times 5 \times 5 }$ . Thus,

$$
d _ { c } = 5 ^ { 2 } + 3 ^ { 2 } + 5 ^ { 2 } = 5 9 , \qquad \pmb \theta \in \mathbb R ^ { 3 d _ { c } } = \mathbb R ^ { 1 7 7 } ,\tag{2}
$$

where $d _ { c }$ is the number of learnable parameters per RGB channel. Hence, $\pmb { \theta } = [ \pmb { \theta } _ { R } , \pmb { \theta } _ { G } , \pmb { \theta } _ { B } ]$ with ${ \bf \bar { \boldsymbol { \theta } } } _ { c } \in \mathbb { R } ^ { 5 9 }$

## 3.2. Population-Based Forward-Only Search

Given training pairs $\{ ( \mathbf { y } _ { i } , \mathbf { x } _ { i } ) \} _ { i = 1 } ^ { N }$ with corresponding DC information $\mathbf { d } _ { i } = \mu ( \mathcal { P } ( \mathbf { x } _ { i } ) )$ , the reconstruction objective is

$$
\mathcal { L } ( \pmb \theta ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathrm { M S E } ( F _ { \pmb \theta } ( \mathbf { y } _ { i } , \mathbf { d } _ { i } ) , \mathbf { x } _ { i } ) .\tag{3}
$$

![](images/801da68d5c20da28c42544729c6c061199488e16cbb4c25f013d82ab363fedbb.jpg)  
Fig. 2. Shared-kernel wavelet network. The learnable kernels H, G, and K are reused across all scales, while all intermediate states remain three-channel representations. The nonlearned per-channel DC correction is omitted for clarity.

Conventional BP minimizes this objective using gradients with respect to θ. In contrast, we optimize complete parameter candidates using the Covariance Matrix Adaptation Evolution Strategy (CMA-ES) [13]. At each generation, candidates are sampled as

$$
\pmb { \theta } ^ { ( k ) } = \mathbf { m } + \sigma \mathbf { C } ^ { 1 / 2 } \mathbf { z } ^ { ( k ) } , \qquad \mathbf { z } ^ { ( k ) } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } ) ,\tag{4}
$$

where m, C, and σ denote the search mean, covariance matrix, and global step size, respectively. Candidates are ranked using forward reconstruction losses, after which CMA-ES adapts its search distribution. The learning loop therefore requires no automatic differentiation or backward propagation.

## 3.3. Structure-Aware Search and Credit Assignment

Although the overall parameter space is compact, directly searching $\pmb \theta \in \mathbb { R } ^ { 1 7 7 }$ ignores the intrinsic RGB separability of the grouped network. We therefore factorize the search into three channel-specific distributions,

$$
\begin{array} { r } { q _ { c } ( \pmb { \theta } _ { c } ) = \mathcal { N } \left( \mathbf { m } _ { c } , \sigma _ { c } ^ { 2 } \mathbf { C } _ { c } \right) , \qquad \pmb { \theta } _ { c } \in \mathbb { R } ^ { 5 9 } , \quad c \in \{ R , G , B \} . } \end{array}\tag{5}
$$

At each population index, one candidate from each channel distribution is assembled into a network candidate and evaluated through the same RGB forward pass. Because all learnable convolutions are channel-wise grouped and the remaining operations preserve channels, each output channel depends only on its corresponding parameter block.

For consistent ranking, each coverage partitions all N samples into non-overlapping subsets of size M. At generation t, all candidates and three CMA-ES instances share $S _ { t }$ ; the permutation is refreshed every N/M generations.

We further use channel-specific credit assignment so that each parameter block receives its directly corresponding optimization signal. For channel c, the fitness is

$$
\mathcal { L } _ { c } = \frac { 1 } { M } \sum _ { i \in S _ { t } } \mathrm { M S E } ( F _ { \pmb \theta } ( \mathbf { y } _ { i } , \mathbf { d } _ { i } ) _ { c } , \mathbf { x } _ { i , c } ) , \quad c \in \{ R , G , B \} .\tag{6}
$$

The R, G, and B CMA-ES instances are ranked and updated using $\mathcal { L } _ { R } , \mathcal { L } _ { G }$ , and $\mathcal { L } _ { B }$ , respectively. This combination of parameter-space decomposition and aligned credit assignment allows population-based search to exploit the intrinsic RGB-separable structure while retaining a fully forward-only learning procedure.

## 4. EXPERIMENTS

We evaluate the framework through comparisons with backpropagation, structure and credit-assignment ablations, and computational analysis. We first describe the common protocol used across comparisons and multi-seed experiments.

## 4.1. Experimental Setup

Experiments use a BSDS500-derived reconstruction dataset. We extract 256 × 256 patches from the original training, validation, and test splits and apply eight-way rotation-and-flip augmentation, yielding 51,200, 1,280, and 12,800 samples, respectively, without source-image overlap. MSE, PSNR, and SSIM are used for evaluation. All methods share the same 177-parameter network, Xavier initialization, data splits, and forward implementation; validation selects checkpoints and the test set remains isolated. The BP baseline uses Adam with a learning rate of $1 0 ^ { - 5 }$ , batch size 32, and 1,500 epochs over ten data-order seeds. CMA-ES uses $\sigma _ { 0 } = 0 . 2 5 , M = 6 4$ samples per generation, and 4,800 generations, corresponding to six training-set coverages. We use $\lambda = 3 2$ , 64 for the main comparison and $\lambda \in \{ 1 9 , 3 2 , 4 8 , 6 4 \}$ for the population study. The choice $M = 6 4$ follows a preliminary ranking-stability analysis, balancing reliability and forward-evaluation cost.

## 4.2. Comparison with Backpropagation

Table 1 compares the proposed forward-only learner with Adam/BP. Values are means over ten runs. Without backward propagation, both forward-only configurations outperform BP in reconstruction quality. In particular, λ = 64 reduces the average test MSE from 4.1783 to 4.0130, corresponding to a 3.95% reduction, while also improving PSNR and SSIM.

The results are also highly consistent across random runs. Across ten runs, the test-MSE standard deviations of Adam/BP, Ours $( \lambda = 3 2 )$ , and Ours (λ = 64) are 0.0010, 0.0121, and 0.0008, respectively. The test-MSE coefficient of variation is 0.0241% for BP and 0.0209% for the $\lambda = 6 4$ configuration. Moreover, all ten runs of both forward-only configurations obtain lower test MSE than all BP runs, indicating that the gain is not driven by isolated favorable runs.

Table 1. Comparison with Adam/BP on the BSDS500 test set. Reconstruction metrics are means over ten runs. Forward evaluations denote training-image evaluations.
<table><tr><td>Method</td><td>MSE↓</td><td>PSNR↑</td><td>SSIM↑</td><td>Fwd. evals</td></tr><tr><td>Adam/BP</td><td>4.1783</td><td>42.6211</td><td>0.994875</td><td>76.80M</td></tr><tr><td>Ours  $( \lambda = 3 2 )$ </td><td>4.0431</td><td>42.7583</td><td>0.995019</td><td>9.83M</td></tr><tr><td>Ours (λ = 64)</td><td>4.0130</td><td>42.7870</td><td>0.995035</td><td>19.66M</td></tr></table>

![](images/21693d134af4f56cd38da77ac878d71c231f3758a7c69e9f0c6ccdd2df4cfa07.jpg)  
Fig. 3. Mechanism ablation at $\lambda = 3 2 .$ . Validation-MSE trajectories compare joint 177-D search, structured $3 \times 5 9 - \mathrm { D }$ search with global fitness, and channel-specific credit assignment. Curves show median and min–max envelopes over five matched seeds; all runs use $M = 6 4$ and 4,800 generations.

## 4.3. Structure and Credit Assignment

We examine how parameter-space structure and fitness assignment affect forward-only optimization. Figure 3 compares three formulations at $\lambda = 3 2$ : joint 177-D search with global RGB fitness, $ { 3 \times 5 9 - \mathrm { D } }$ structured search with the same fitness, and $3 \times 5 9 - \mathrm { D }$ search with channel-specific credit assignment. The joint search converges slowly and remains at high validation error. Decomposing the search space substantially improves convergence, while assigning each parameter block its channel-specific fitness produces a further large improvement and reaches the low-error regime.

The test MSE decreases from 940.23 for joint 177-D search to 30.81 after $3 ~ \times ~ 5 9 { - } \mathrm { D }$ decomposition, and further to 4.04 with channel-specific credit assignment. Thus, compactness alone is insufficient for whole-network search: parameter-space decomposition is decisive, while channelspecific credit assignment is critical for reaching BP-level reconstruction quality under the tested optimization budget. Together, the two components align the optimizer with the intrinsic RGB-separable structure of the network.

We next study the effect of population size. Figure 4 compares joint 177-D and structured $3 \times 5 9 - \mathrm { D }$ optimization for $\lambda \in \{ 1 9 , 3 2 , 4 8 , 6 4 \}$ . Increasing λ improves both formulations, but even the joint 177-D search at $\lambda = 6 4$ remains far from the BP regime. In contrast, the structured formulation achieves orders-of-magnitude lower error. As highlighted by the inset, the largest improvement occurs from $\lambda = 1 9 \mathrm { t o } 3 2 .$ after which the gain saturates. This confirms that parameterspace structure is the dominant factor in forward-only optimization, whereas increasing population provides only secondary refinement after structured search is established.

![](images/480c642c481427eb11c29cb47212ef2c9e5259bb670461f1e8bd3bdcdc9065a5.jpg)  
Fig. 4. Population-size response of joint 177-D and structured $3 \times 5 9 - \mathrm { D }$ forward-only search. The inset magnifies the structured regime with Adam/BP as a reference. Points show mean±standard deviation over five matched seeds; all runs use $M = 6 4$ and 4,800 generations.

## 4.4. Computational Characteristics

As shown in Table 1, CMA-ES requires only forward evaluations, whereas BP additionally propagates gradients through all training samples. These counts are not a strict equalcompute comparison: larger λ increases candidate evaluations, and BP and CMA-ES use different GPU parallelism. We therefore make no claim of wall-clock superiority.

## 5. CONCLUSION

This work presented a global forward-only learning framework for compact shared-kernel wavelet neural networks. By combining CMA-ES with structure-aware parameter decomposition and channel-specific credit assignment, the proposed method enables whole-network optimization using forward evaluations only, without automatic differentiation or backward propagation. Experiments on image reconstruction show that the resulting learner consistently outperforms the Adam/BP baseline while revealing that network structure and aligned credit assignment are more important than simply increasing the population size. These results demonstrate that extreme parameter sharing can make global forward-only learning practical for compact neural networks. Future work will investigate its extension to larger architectures and more general vision tasks.

## 6. REFERENCES

[1] Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang, “Restormer: Efficient transformer for highresolution image restoration,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 5728–5739.

[2] Liangyu Chen, Xiaojie Chu, Xiangyu Zhang, and Jian Sun, “Simple baselines for image restoration,” in Proceedings of the European Conference on Computer Vision, 2022, pp. 17–33.

[3] Vaishnav Potlapalli, Syed Waqas Zamir, Salman Khan, and Fahad Shahbaz Khan, “PromptIR: Prompting for all-in-one image restoration,” in Advances in Neural Information Processing Systems, 2023.

[4] Zhendong Wang, Xiaodong Cun, Jianmin Bao, Wengang Zhou, Jianzhuang Liu, and Houqiang Li, “Uformer: A general u-shaped transformer for image restoration,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 17683–17693.

[5] Jun-Jie Huang and Pier Luigi Dragotti, “WINNet: Wavelet-inspired invertible network for image denoising,” IEEE Transactions on Image Processing, vol. 31, pp. 4377–4392, 2022.

[6] Xin Deng, Chao Gao, and Mai Xu, “PIRNet: Privacypreserving image restoration network via wavelet lifting,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 22368– 22377.

[7] Yuanhao Gong, Tan Tang, and Qianyan Liu, “Sharedkernel wavelet networks for near-sensor poisson image reconstruction,” IEEE Sensors Journal, vol. 26, no. 14, 2026.

[8] Adityanarayanan Radhakrishnan, Daniel Beaglehole, Parthe Pandit, and Mikhail Belkin, “Mechanism for feature learning in neural networks and backpropagationfree machine learning models,” Science, vol. 383, no. 6690, pp. 1461–1467, 2024.

[9] Guy Lorberbom, Itai Gat, Yossi Adi, Alexander Schwing, and Tamir Hazan, “Layer collaboration in the forward-forward algorithm,” Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, no. 13, pp. 14141–14148, 2024.

[10] Geoffrey Hinton, “The forward-forward algorithm: Some preliminary investigations,” 2022.

[11] Mengye Ren, Simon Kornblith, Renjie Liao, and Geoffrey Hinton, “Scaling forward gradient with local losses,” in International Conference on Learning Representations, 2023.

[12] Nan Li, Lianbo Ma, Tiejun Xing, Guo Yu, Chen Wang, Yingyou Wen, Shi Cheng, and Shangce Gao, “Automatic design of machine learning via evolutionary computation: A survey,” Applied Soft Computing, vol. 143, pp. 110412, 2023.

[13] Nikolaus Hansen, “The cma evolution strategy: A tutorial,” arXiv preprint arXiv:1604.00772, 2016.

[14] Zhijian Wu, Jun Li, Yang Hu, and Dingjiang Huang, “Compacter: A lightweight transformer for image restoration,” in Proceedings of the 32nd ACM International Conference on Multimedia, 2024, pp. 3094–3103.

[15] Giorgia Dellaferrera and Gabriel Kreiman, “Errordriven input modulation: Solving the credit assignment problem without a backward pass,” in Proceedings of the 39th International Conference on Machine Learning, 2022, pp. 4937–4955.

[16] Andreas Papachristodoulou, Christos Kyrkou, Stelios Timotheou, and Theocharis Theocharides, “Convolutional channel-wise competitive learning for the forward-forward algorithm,” Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, no. 13, pp. 14536–14544, 2024.

[17] Ravi Srinivasan, Francesca Mignacco, Martino Sorbaro, Maria Refinetti, Avi Cooper, Gabriel Kreiman, and Giorgia Dellaferrera, “Forward learning with top-down feedback: Empirical and analytical characterization,” in International Conference on Learning Representations, 2024.

[18] Xing Chen, Dongshu Liu, Jer´ emie Laydevant, and Julie´ Grollier, “Self-contrastive forward-forward algorithm,” Nature Communications, vol. 16, pp. 5978, 2025.

[19] Liang Sun, Yang Zhang, Weizhao He, Jiajun Wen, Linlin Shen, and Weicheng Xie, “Deeperforward: Enhanced forward-forward training for deeper and better performance,” in International Conference on Learning Representations, 2025.

[20] Sadhika Malladi, Tianyu Gao, Eshaan Nichani, Alex Damian, Jason D. Lee, Danqi Chen, and Sanjeev Arora, “Fine-tuning language models with just forward passes,” in Advances in Neural Information Processing Systems, 2023, vol. 36, pp. 53038–53075.