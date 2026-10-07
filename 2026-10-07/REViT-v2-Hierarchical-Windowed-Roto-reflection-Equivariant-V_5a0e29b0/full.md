# REViT-v2: Hierarchical Windowed Roto-reflection Equivariant ViT for Equivariant Feature Extraction

Sheir A. Zaheer

sheir@kc-ml2.com

KC Machine Learning Lab, Rep. of Korea

Jihwan Moon<sup>∗</sup>

jimoon1215@gmail.com

NFOCZ Inc and Seoul National University, Rep. of Korea

Chan Y. Park

chan.y.park@kc-ml2.com

KC Machine Learning Lab, Rep. of Korea

## Abstract

We propose a scalable roto-reflection-group-equivariant vision transformer based on windowed group-convolutional self-attention and a hierarchical feature architecture. We demonstrate that our approach can be scaled to group-equivariant vision transformers (ViTs) with millions of parameters and large datasets with practically sized images, i.e., ImageNet. The code and pretrained weights for the proposed Hierarchical Windowed Roto-reflection Equivariant ViTs (REViT-v2) are available at <sup>§</sup> https://github.com/kc-ml2/revit.

## 1. Introduction

Standard vision transformers (ViTs) (Dosovitskiy et al., 2021) do not explicitly encode common image symmetries. A conventional model must learn from data that a visual pattern retains its identity after a rotation or reflection. Group-equivariant neural networks instead incorporate this relationship into the architecture. Group-equivariant convolutional networks introduced group convolutions as a systematic means of sharing filters across translations, rotations, and reflections (Cohen and Welling, 2016; Weiler and Cesa, 2019).

Romero and Cordonnier (2021) extended the same principle to attention and showed that attention can be made equivariant to symmetry groups by constructing positional encodings that are invariant under the corresponding group action. This provides a general theoretical formulation, but it does not scale beyond low-resolution images. Moreover, group-relative positional terms must be repeatedly evaluated in all attention layers. Zaheer et al. (2026) subsequently introduced rot-reflection equivariant convolutional vision transformer (REViT) with group-convolutional self-attention (G-CSA), in which convolutional projections remove the dependence on explicit positional encoding. This makes equivariant transformers considerably simpler. Nevertheless, when group-convolutional attention is evaluated globally, its spatial attention matrix remains quadratic with respect to input size. The cost is especially important for equivariant models because regular group representations typically allocate several orientation channels to each feature field.

Several developments in eficient vision transformers suggest a path toward addressing this limitation. CvT introduced convolutional token embedding and convolutional attention projections showing that convolutional structure can reduce the need for explicit positional encoding (Wu et al., 2021). Swin transformer restricted attention to local windows and organized the network into a multiscale pyramid (Liu et al., 2021). Multiscale Vision Transformers similarly demonstrated the value of progressively reducing token resolution while increasing channel capacity (Fan et al., 2021).

In this paper, we propose REViT-v2. REViT-v2 combines hierarchical vision transformers with local group-equivariant convolutional self attention. REViT-v2 introduces two central components. First, it utilizes Windowed Group Convolutional Self-Attention (wG-CSA), which performs attention within local spatial windows while preserving complete group-representation fields in each attention head. Second, it constructs a multi-stage equivariant feature pyramid using group-convolutional tokenization and downsampling. Consequently, REViT-v2 reduces the spatial complexity of group-equivariant attention from quadratic to linear in the image area. This design targets large-scale image recognition while maintaining equivariance throughout the feature extractor. We demonstrate its efectiveness through classification results on ImageNet-1K (Deng et al., 2009) and a complexity comparison with global G-CSA.

## 2. Windowed Group Convolutional Self-Attention

wG-CSA is a local window extension of G-CSA (Zaheer et al., 2026). The aim is to reduce the quadratic complexity of G-CSA in image size to linear complexity.

## 2.1. Equivariant Projections

Let the input to an attention block be a feature map. The channels of input x are organized according to a representation $\rho$ of a finite planar symmetry group $G ,$ such as the cyclic rotation group $p N$ or the dihedral group $p N m$ . Queries, keys, and values are produced by group-equivariant convolutional projections Zaheer et al. (2025):

$$
Q = \phi _ { Q } ( x ) , \qquad K = \phi _ { K } ( x ) , \qquad V = \phi _ { V } ( x ) ,\tag{1}
$$

The attention output is followed by another group-equivariant projection $\phi _ { O }$

## 2.2. Local Window Attention

The spatial lattice is partitioned into non-overlapping windows, $\mathcal { W } = \{ W _ { 1 } , \ldots , W _ { K } \}$ , where each window contains $M ^ { 2 }$ positions. Let $W ( p )$ denote the window containing position $p .$ For head h, the attention score between $p$ and $q \in W ( p )$ is

$$
s _ { p q } ^ { h } = \frac { \left. Q ^ { h } ( p ) , K ^ { h } ( q ) \right. } { \sqrt { d _ { h } } } ,\tag{2}
$$

where $Q ^ { h }$ and $K ^ { h }$ are query and key projections for head h. The normalized attention coeficient and output are

$$
a _ { p q } ^ { h } = \frac { \exp ( s _ { p q } ^ { h } ) } { \sum _ { r \in W ( p ) } \exp ( s _ { p r } ^ { h } ) } ,\tag{3}
$$

$$
Y ^ { h } ( p ) = \sum _ { q \in W ( p ) } a _ { p q } ^ { h } V ^ { h } ( q ) ,\tag{4}
$$

where $V ^ { h }$ is the value projection. Outputs from all heads are concatenated along complete representation fields and projected using ϕ<sub>O</sub>:

$$
\operatorname { w G - C S A } ( x ) = \phi _ { O } \biggl ( \operatorname { C o n c a t } _ { h = 1 } ^ { H } Y ^ { h } \biggr ) .\tag{5}
$$

No absolute positional embedding is introduced. Spatial structure is instead preserved by the equivariant convolutional stem and the convolutional query, key, and value projections. We demonstrate the equivariance of wG-CSA in Appendix A.

## 2.3. Linear Complexity of Windowed G-CSA

For x of size $H \times W$ ， $N = H W$ denote the number of positions in a feature map. Global attention computes $N ^ { 2 }$ pairwise scores per head and has approximate complexity $O ( N ^ { 2 } C )$ (Dosovitskiy et al., 2021). Windowed attention with $M \times M$ sized windows, however, contains $N / M ^ { 2 }$ windows, each requiring $M ^ { 4 }$ pairwise comparisons. Its total complexity is therefore

$$
{ \cal O } \big ( ( N / M ^ { 2 } ) M ^ { 4 } C \big ) = { \cal O } ( N M ^ { 2 } C ) .\tag{6}
$$

For fixed M, the cost is linear in image area.

## 3. REViT-v2

REViT-v2 arranges the proposed wG-CSA in a multi-stage feature hierarchy. The early layers need to preserve suficient spatial resolution to identify local structures, whereas the deeper layers need greater representation capacity and wider efective receptive fields.

## 3.1. Equivariant Patching and Lifting

The input image is interpreted as a field of trivial representations because rotating an image changes spatial positions but does not permute its RGB channels. A group-equivariant convolutional stem lifts this input into regular group representations. Two strided groupequivariant convolutions reduce the image resolution by a factor of four.

## 3.2. Equivariant Transformer Block

Each block contains a pre-normalized wG-CSA module and an equivariant feed-forward network. Given input $x _ { l } .$ , the block is

$$
z _ { l } = x _ { l } + \mathrm { D r o p P a t h } ( \alpha _ { l } \mathrm { w G - C S A } ( \mathcal { N } _ { 1 } ( x _ { l } ) ) ) ,\tag{7}
$$

$$
x _ { l + 1 } = z _ { l } + \mathrm { D r o p P a t h } ( \beta _ { l } \mathcal { M } _ { G } ( \mathcal { N } _ { 2 } ( z _ { l } ) ) ) .\tag{8}
$$

Here, $\mathcal { N } _ { 1 }$ and ${ \mathcal { N } } _ { 2 }$ are representation-compatible normalization operations, while $\alpha _ { l }$ and $\beta _ { l }$ are trainable residual scaling parameters. The equivariant MLP $\mathcal { M } _ { G }$ replaces ordinary fully connected layers with $1 \times 1$ group convolutions. DropPath implements stochastic depth. Stochastic depth regularizes the residual branches without independently dropping orientation channels, which would violate their representation structure.

Table 1: Performance Comparison on ImageNet-1K.
<table><tr><td>Model</td><td>Top-1 Accuracy (%)</td><td>Top-5 Accuracy (%)</td><td>Params.</td></tr><tr><td>ViT-S w/ aug</td><td>72.08</td><td>89.54</td><td>22 M</td></tr><tr><td>RE-ResNet</td><td>77.37</td><td>93.74</td><td>11M</td></tr><tr><td>REViT-v2-T</td><td>72.58</td><td>90.88</td><td>5M</td></tr><tr><td>REViT-v2-S</td><td>79.27</td><td>94.45</td><td>18 M</td></tr><tr><td>REViT-v2-S</td><td>80.9</td><td>95.1</td><td>47M</td></tr></table>

Table 2: Comparison with global G-CSA on Rotated MNIST.
<table><tr><td rowspan=1 colspan=1>Attention</td><td rowspan=1 colspan=1>Params.</td><td rowspan=1 colspan=1>Forward FLOPs</td><td rowspan=1 colspan=1>Latency vs global</td><td rowspan=1 colspan=1>Peak Training Memory</td><td rowspan=1 colspan=1>Accuracy (%)</td></tr><tr><td rowspan=1 colspan=1>Global</td><td rowspan=1 colspan=1>97.7 K</td><td rowspan=1 colspan=1>1.69 GFLOPs</td><td rowspan=1 colspan=1>1×</td><td rowspan=1 colspan=1>429.7 MB</td><td rowspan=1 colspan=1>98.23</td></tr><tr><td rowspan=1 colspan=1>Global w/downsampling</td><td rowspan=1 colspan=1>97.7 K</td><td rowspan=1 colspan=1>333.6 MFLOPs</td><td rowspan=1 colspan=1>0.75×</td><td rowspan=1 colspan=1>50.98 MB</td><td rowspan=1 colspan=1>98.28</td></tr><tr><td rowspan=1 colspan=1>Windowed</td><td rowspan=1 colspan=1>102.9 K</td><td rowspan=1 colspan=1>80.1 MFLOPs</td><td rowspan=1 colspan=1>0.61×</td><td rowspan=1 colspan=1>26.47 MB</td><td rowspan=1 colspan=1>98.26</td></tr></table>

## 3.3. Multistage Hierarchy

After the blocks in each of the first three stages, a strided group convolution reduces the spatial resolution while the number of representation fields is increased. Later stages can therefore use more channels and attention heads without exceeding the cost. At deeper stages, a fixed M × M window also corresponds to a progressively larger region of the original image. The final stage can therefore model high-level interactions over a substantial portion of the input without constructing a global high-resolution attention matrix.

The hierarchy and local windows address diferent parts of the scaling problem: windowing eliminates the quadratic dependence on the number of tokens, while downsampling prevents the high-resolution representation from being propagated through the complete network. Together, they provide a practical route toward symmetry-aware ViT backbones for ImageNet-scale classification and, with suitable interfaces, dense prediction tasks.

## 4. Experimental Results

Table 1 compares the performance of a p4m-equivariant REViT-v2 with two baselines: a vanilla ViT trained with rotation and flip augmentations and a p4m-equivariant RE-ResNet model. REViT-v2 outperforms both. Additional details are provided in Appendix B.

## Complexity Comparison with Global G-CSA

Table 2 compares wG-CSA with global G-CSA on Rotated MNIST (Larochelle et al., 2007) trained for 200 epochs and a batch size of 128. The proposed windowed architecture substantially reduces model size, forward FLOPs, latency, and peak memory while maintaining comparable accuracy. Since REViT-v2 combines a downsampling stem with hierarchical windowed attention, we also evaluate global attention after the same downsampling stem to distinguish the efect of early downsampling from that of windowed attention.

## 5. Conclusion

In this paper, we demonstrate the efectiveness of REViT-v2 as scalable group-equivariant vision transformers that can serve as backbone feature extractors in the same manner as conventional ViTs. We also provide code and pretrained weights to support further research.

## References

Gabriele Cesa, Leon Lang, and Maurice Weiler. A program to build E(N)-equivariant steerable CNNs. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=WE4qe9xlnQw.

Taco S. Cohen and Max Welling. Group equivariant convolutional networks. In Proceedings of the 33rd International Conference on Machine Learning, volume 48 of Proceedings of Machine Learning Research, pages 2990–2999. PMLR, 2016. URL https: //proceedings.mlr.press/v48/cohenc16.html.

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A largescale hierarchical image database. In 2009 IEEE Conference on Computer Vision and Pattern Recognition, pages 248–255, 2009. doi: 10.1109/CVPR.2009.5206848.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=YicbFdNTTy.

Haoqi Fan, Bo Xiong, Karttikeya Mangalam, Yanghao Li, Zhicheng Yan, Jitendra Malik, and Christoph Feichtenhofer. Multiscale vision transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 6824– 6835, 2021. URL https://openaccess.thecvf.com/content/ICCV2021/html/Fan\_ Multiscale\_Vision\_Transformers\_ICCV\_2021\_paper.html.

H. Larochelle, D. Erhan, Aaron C. Courville, James Bergstra, and Yoshua Bengio. An empirical evaluation of deep architectures on problems with many factors of variation. In International Conference on Machine Learning, 2007. URL https://api.semanticscholar. org/CorpusID:14805281.

Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 10012–10022, 2021. URL https://openaccess.thecvf.com/content/ICCV2021/ html/Liu\_Swin\_Transformer\_Hierarchical\_Vision\_Transformer\_Using\_Shifted\_ Windows\_ICCV\_2021\_paper.html.

David W. Romero and Jean-Baptiste Cordonnier. Group equivariant stand-alone selfattention for vision. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=JkfYjnOEo6M.

Maurice Weiler and Gabriele Cesa. General E(2)-equivariant steerable CNNs. In Advances in Neural Information Processing Systems, volume 32, 2019. URL https://proceedings.neurips.cc/paper\_files/paper/2019/hash/ 45d6637b718d0f24a237069fe41b0db4-Abstract.html.

Haiping Wu, Bin Xiao, Noel Codella, Mengchen Liu, Xiyang Dai, Lu Yuan, and Lei Zhang. CvT: Introducing convolutions to vision transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22– 31, 2021. URL https://openaccess.thecvf.com/content/ICCV2021/html/Wu\_CvT\_ Introducing\_Convolutions\_to\_Vision\_Transformers\_ICCV\_2021\_paper.html.

S. A. Zaheer, A. C. Holston, and C. Y. Park. REViT: Roto-reflection equivariant convolutional vision transformer. arXiv preprint arXiv:2606.25318, 2026. URL https: //arxiv.org/abs/2606.25318.

Sheir A. Zaheer, Alexander C. Holston, and Chan Y. Park. Group convolutional selfattention for roto-translation equivariance in vits. In NeurIPS 2025 Workshop on Symmetry and Geometry in Neural Representations, 2025. URL https://openreview.net/ forum?id=fOy5s3Gr35.

## Appendix A. Equivariance Analysis

## A.1. Theoretical Analysis

Assume that the group representations within a head are orthogonal, as is the case for permutation-based regular representations (Cesa et al., 2022) used in this work:

$$
\rho _ { h } ( g ) ^ { \top } \rho _ { h } ( g ) = I .\tag{9}
$$

Under a transformation g, the projected features satisfy

$$
Q ^ { h ^ { \prime } } ( g p ) = \rho _ { h } ( g ) Q ^ { h } ( p ) , \qquad K ^ { h ^ { \prime } } ( g q ) = \rho _ { h } ( g ) K ^ { h } ( q ) .\tag{10}
$$

Their dot product is therefore unchanged:

$$
\begin{array} { r l } & { \left. Q ^ { h ^ { \prime } } ( g p ) , K ^ { h ^ { \prime } } ( g q ) \right. = Q ^ { h } ( p ) ^ { \top } \rho _ { h } ( g ) ^ { \top } \rho _ { h } ( g ) K ^ { h } ( q ) } \\ & { \qquad = \left. Q ^ { h } ( p ) , K ^ { h } ( q ) \right. . } \end{array}\tag{11}
$$

(12)

The attention coeficients are consequently permuted consistently with the transformed spatial positions. Since the values transform through the same representation, the output satisfies

$$
Y ^ { h ^ { \prime } } ( g p ) = \rho _ { h } ( g ) Y ^ { h } ( p ) .\tag{13}
$$

Thus,

$$
\mathrm { w G - C S A } ( T _ { g } x ) = T _ { g } \mathrm { w G \mathrm { - } C S A } ( x ) ,\tag{14}
$$

provided that the transformation maps complete windows to complete windows:

$$
W ( g p ) = g W ( p ) .\tag{15}
$$

This condition is satisfied by square windows under $9 0 ^ { \circ }$ rotations and axis-aligned reflections when the square feature-map dimensions are divisible by the window size.

Table 3: Equivariance error analysis
<table><tr><td>Group</td><td>Lifting Stem Equivariance Err.↓</td><td>Pre-Class Equivariance Err.↓</td></tr><tr><td>p4</td><td> $\overline { { 0 . 0 0 0 1 4 2 \pm 0 . 0 0 0 0 1 7 } }$ </td><td> $\overline { { 0 . 0 0 2 1 8 \pm 0 . 0 0 0 4 7 1 } }$ </td></tr><tr><td> $p 4 m$ </td><td> $0 . 0 0 1 3 1 6 \pm 0 . 0 0 0 6 1 4$ </td><td> $0 . 0 0 0 0 8 \pm 0 . 0 0 0 0 3 4$ </td></tr><tr><td>p8</td><td> $0 . 0 0 1 4 7 4 \pm \ : 0 . 0 0 0 3 6 6$ </td><td> $0 . 0 0 0 1 0 2 \pm 0 . 0 0 0 0 1 3$ </td></tr></table>

## A.2. Quantitative Analysis

We also evaluated the architectural equivariance of REViT-v2s on the ImageNet validation set through equivariance error analysis.

Definition: For a feature map $f ( \cdot )$ , the equivariance error is defined as

$$
\mathcal { E } _ { \mathrm { e q } } = \| f ( g \cdot x ) - g \cdot f ( x ) \| _ { 1 } ,\tag{16}
$$

where $\lVert \cdot \rVert _ { 1 }$ denotes the mean absolute diference over all spatial locations, channels, and batch elements.

This metric is evaluated at two stages: 1) after the lifting layer (lifting equivariance error) and 2) at the final feature representation after all the G-CSA transformer blocks and immediately preceding the classification head (pre-class equivariance error ). In this setting, no trained checkpoint is loaded; the model is randomly initialized, so the experiment isolates equivariance induced by the network design rather than invariance learned through optimization. 1024 images were taken from ImageNet validation set and processed with standard ImageNet evaluation transforms: resize to 256, center cropping to 224x224, and normalization with ImageNet mean and standard deviation.

For each sampled image and each non-identity element of the p4, p4m, and p8 groups, we compared the network response to the transformed input against the transformed response to the original input. Equivariance was measured at the stem output and immediately before the classification head (after all G-CSA blocks). Table 3 summarizes the results of these equivariance tests. For the $p 8$ group, errors accumulate starting at the lifting layer, which we attribute to interpolation artifacts introduced by rotations at angles such as $4 5 ^ { \circ }$ and 135<sup>◦</sup>, where pixel locations fall between the original grid. This results in approximation error at the input level.

## Appendix B. Training Setup

REViT-v2 was trained on ImageNet using distributed data parallelism on four Nvidia GeForce RTX 4090 GPUs, with a per-GPU batch size of 128, giving an efective batch size of 512. We optimize the model for 300 epochs with AdamW, using an initial learning rate of 3e-4, weight decay of 0.05, and a learning-rate schedule consisting of 20 epochs of linear warmup followed by cosine decay. The model is instantiated with a window size of $^ { 7 , }$ and a 3x3 equivariant kernel for the $\mathrm { Q / K / V }$ projections in the attention layers.