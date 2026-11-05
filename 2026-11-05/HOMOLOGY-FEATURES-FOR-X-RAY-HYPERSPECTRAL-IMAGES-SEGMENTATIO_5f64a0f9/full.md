# HOMOLOGY FEATURES FOR X-RAY HYPERSPECTRAL IMAGES SEGMENTATION

Fethi Harkat<sup>\*</sup> <sup>†</sup>, Guillaume Gey<sup>\*</sup>, Valérie Perrier<sup>†</sup>, Kévin Polisano<sup>†</sup>,

<sup>\*</sup>Detection Technology Plc, A Grid, Otakaari 5A, 02150 Espoo, Finland

<sup>†</sup> Univ. Grenoble Alpes, CNRS, Grenoble INP, LJK, 38000 Grenoble, France

## ABSTRACT

Characterizing Multi-Energy X-ray Transmission (XRT) hyperspectral images presents significant challenges for traditional classification frameworks due to high-dimensional redundancy and non-linear spectral overlaps. Although manifold learning techniques such as UMAP enhance spectral representation by preserving non-linear correlations, they do not explicitly utilize the geometric structure of the embedded manifold for segmentation. This study introduces a hybrid architecture that integrates topological features derived from persistent homology with convolutional and graph neural networks. The approach computes Vietoris-Rips persistent homology on UMAP-projected superpixel point clouds to capture H cyclic structures that encode spectral heterogeneity. These topological descriptors are combined with pixel-level CNN features through a broadcasting mechanism that propagates superpixel-level graph representations to full spatial resolution. Experimental results on XRT multi-energy imagery demonstrate that the proposed hybrid model significantly outperforms both CNN-only and superpixel-pooling fusion baselines, suggesting that topological features provide complementary discriminative information to local spectralspatial descriptors.

Index Terms— Hyperspectral imaging, X-ray detection, Topology, Manifold learning, Dimensionality reduction, Deep learning, Feature extraction, Computer Vision.

## 1. INTRODUCTION

Linear projection techniques such as Principal Component Analysis (PCA) [1] do not capture the non-linear geometry of Hyperspectral Images (HSI) [2], and our previous work showed that they also fail to represent the intrinsic geometry of XRT datasets [3]. The Uniform Manifold Approximation and Projection (UMAP) algorithm [4, 5] was shown to improve optical HSI classification over PCA [6], and our previous work established that the manifold underlying XRT data exhibits non-linear correlations that can be characterized through topological approximation [3]. These results indicate that the topology of the data contains discriminative information that is not captured by standard spectral-spatial feature extractors.

Building upon this foundation, we propose a hybrid architecture that integrates tools from Topological Data Analysis (TDA), specifically persistent homology [7, 8], with Graph Neural Networks (GNNs) [9, 10] and Convolutional Neural Networks (CNNs). The high dimensionality and spectral redundancy of HSI data have motivated a shift from pixel-wise classification to region-based reasoning through graph representations [11]. In this framework, the image is abstracted as a graph where nodes correspond to image regions and edges encode spatial or spectral adjacency. Early approaches constructed graphs at the pixel level, treating each spectral vector as a node and connecting k-nearest neighbors in spectral space [12]. While preserving fine-grained spatial resolution, this approach suffers from computational intractability and susceptibility to spectral noise, as the graph connectivity becomes unreliable in high-dimensional, noisy spectral spaces [11]. Graph Convolutional Networks (GCNs) [13] operate on this structure through message-passing mechanisms that propagate information across the graph, enabling the integration of local spectral features with global spatial context.

Computing persistent homology requires a sufficient number of data points to form meaningful topological structures. Operating at the pixel scale is impractical because individual pixels lack adequate spectral information to construct stable persistence diagrams. We therefore adopt a superpixel-based strategy. Following our previous methodology in [3], we first perform a UMAP projection on the image before grouping pixels into perceptually coherent superpixels using Simple Linear Iterative Clustering (SLIC) [14]. Each superpixel’s spectral point cloud constitutes an independent metric space on which Vietoris-Rips persistent homology is computed via Ripser [15]. The resulting persistence statistics, combined with spatial descriptors, form node features for a GCN operating on the superpixel adjacency graph. A broadcasting mechanism propagates superpixel-level graph features back to individual pixels, where they are concatenated with pixel-level CNN features for segmentation.

Prior work has integrated topological summaries into machine learning pipelines through persistence images [16], landscapes [17], kernel embeddings [18], and direct graph architectures [19, 20]. However, the application of localized persistent homology to superpixel-based representations of hyperspectral data remains unexplored. Furthermore, existing hybrid CNN-GNN frameworks treat the two branches of the two neural networks as independent feature extractors whose outputs are combined only at the final layer, missing opportunities for intermediate information exchange [11, 12]. None of these approaches explicitly incorporate topological features into the GNN branch.

The key contributions of this paper are threefold. First, we demonstrate that persistent homology features computed on UMAP-projected superpixel point clouds provide complementary discriminative signal to standard spectral-spatial descriptors. Second, through systematic ablation, we identify a minimal subset of $H _ { 1 }$ persistence statistics that optimizes the trade-off between feature dimensionality and segmentation performance. Third, we propose a broadcasting fusion strategy that preserves pixel-level spatial resolution and outperforms superpixel-pooling alternatives by a substantial margin. Experiments on XRT multi-energy imagery validate the proposed framework.

The remainder of this paper is organized as follows. Section 2 details the proposed methodology, including UMAP projection, superpixel segmentation, topological feature extraction, and the hybrid CNN-GNN architecture. Section 3 describes the experimental setup and datasets, then presents quantitative and qualitative results. Section 4 concludes with directions for future work.

## 2. METHODOLOGY

## 2.1. Overview

The proposed framework, illustrated in Fig. 1, consists of four stages. First, raw XRT hyperspectral cubes are projected onto a low-dimensional manifold using UMAP [4], fitted on a sample of training pixels, following the approach in [3]. Second, the projected image is segmented into superpixels. Third, topological and spatial features are extracted from each superpixel’s spectral point cloud. Fourth, a dual-branch hybrid architecture fuses pixel-level CNN features with superpixellevel GNN features via a broadcasting mechanism, followed by a final classification layer. The first three stages are preprocessing steps; only the two branches and the classifier are trained. We detail below these different steps.

## 2.2. Superpixel Segmentation and Graph Construction

Raw ME-XRT images are acquired as three-dimensional cubes of shape [H, W, C], where H and W denotes spatial dimensions, the detector pixel axis, and C the energy band axis [3]. UMAP [4], fitted on a sample of training pixels, projects each pixel’s C-dimensional spectral signature onto a D-dimensional manifold $( D \ll C ;$ ; here $C = 1 6$ and $D = 5 )$

![](images/1c00a7809883740355c9e70c4ab086e12d7687e12faa73069c3f78bb710ba134.jpg)  
Fig. 1: Pipeline of the proposed hybrid CNN+GNN method. The CNN branch (bottom) processes the UMAP-projected image at full pixel resolution, while the GNN branch (top) operates on superpixel-level topological and spatial features. A broadcasting mechanism fuses both representations before the final classification layer.

yielding projected images of shape [H, W, D] that preserve non-linear spectral correlations.

The UMAP-projected image is then segmented into N perceptually coherent superpixels using SLIC [14], operating on the D manifold channels. The resulting superpixels form the nodes of a graph $\mathcal { G } = ( \nu , \mathcal { E } )$ , where edges connect spatially adjacent regions. Thus, the graph comprises N nodes instead of the original $H \times W$ pixels, reducing the spatial complexity from O(HW) to O(N) while preserving spatial coherence. The spectral dimension is embedded in the $D _ { - }$ dimensional UMAP coordinates and subsequently encoded in the topological descriptors extracted per superpixel.

## 2.3. Node Feature Extraction

For each superpixel $s _ { i } ,$ let $\mathcal { P } _ { i } ~ \subset ~ \mathbb { R } ^ { D }$ denote its UMAPprojected spectral point cloud. Vietoris–Rips persistent homology is computed via Ripser [15], yielding persistence diagrams $\mathcal { D } _ { 0 } ( \mathcal { P } _ { i } )$ and $\mathcal { D } _ { 1 } ( \mathcal { P } _ { i } )$ . Diagram $\mathcal { D } _ { 0 }$ captures connected components (0-dimensional features), while $\mathcal { D } _ { 1 }$ captures 1- dimensional loops. From these diagrams we extract a pool of 30 candidate topological descriptors.

Topological descriptors (30 candidates). For each birth–death pair $( b _ { j } , d _ { j } )$ in a persistence diagram, the persistence is $\ell _ { j } = d _ { j } - b _ { j }$ . Let $n _ { k } = | \mathcal D _ { k } ( \mathcal P _ { i } ) | \left( \bar { k } = 0 , 1 \right)$ denote the number of points in $\mathcal { D } _ { k }$ . The $H _ { 1 }$ -statistics are defined as:

$$
\begin{array} { r l } & { \mathrm { M a x P e r s = m a x \ } \ell _ { j } , } \\ & { \mathrm { M e a n P e r s = \frac { 1 } { n _ { 1 } } \sum _ { j } } \ell _ { j } , \qquad \mathrm { S t a P e r s = \sqrt { \frac { 1 } { n _ { 1 } } \sum _ { j } ( \ell _ { j } - \bar { \ell } ) ^ { 2 } } } , } \\ & { \mathrm { S u m P e r s = \sum _ { \vec { \jmath } } } \ell _ { j } , \qquad \mathrm { M e a n B i r t h = \frac { 1 } { n _ { 1 } } \sum _ { \vec { \jmath } } } b _ { j } , } \\ & { \mathrm { M e a n D e a t h = \frac { 1 } { n _ { 1 } } \sum _ { \vec { \jmath } } } d _ { j } , \qquad \mathrm { M a x B i r t h = m _ { \vec { \jmath } } } b _ { j } , } \\ & { \mathrm { M a x D e a t h = m _ { \vec { \jmath } } } d _ { j } , \qquad \mathrm { E n t r o p y = - \sum _ { \vec { \jmath } } } p _ { j } \log p _ { j } , } \end{array}
$$

where $p _ { j } = \ell _ { j } / \sum _ { k } \ell _ { k }$ , and $\mathrm { P o i n t s C o u n t } = n _ { 1 }$ is the 11th feature. The same 11 statistics are computed on $\mathcal { D } _ { 0 }$ (denoted $H _ { 0 - } { } ^ { * } )$ . Additionally, Betti numbers $\beta _ { 0 }$ and $\beta _ { 1 }$ are recorded at three fixed filtration scales $\epsilon _ { 1 } < \epsilon _ { 2 } < \epsilon _ { 3 }$ (6 features), and two composite scores (Total\_Complexity, Topology\_Score) combine multi-scale topological indicators, yielding 30 candidates in total.

Spectral descriptors (25 dimensions). For each UMAP channel $d \in \{ 1 , \ldots , D \}$ (here $D = 5 )$ , the mean, standard deviation, and quartiles $( \mathrm { Q } 1 , \mathrm { Q } 2 , \mathrm { Q } 3 )$ of pixel intensities within the superpixel are computed, giving $5 \times 5 = 2 5$ dimensions.

Geometric descriptors (5 dimensions). A superpixel $s _ { i }$ is a connected set of pixels; let $\mathcal { R } _ { i } = \{ ( h , w ) \in [ H ] \times [ W ] \ |$ pixel $( h , w )$ belongs to $s _ { i } \}$ denote the collection of pixel coordinates belonging to superpixel $s _ { i } .$ From this set we extract area $\left| \mathcal { R } _ { i } \right|$ (the pixel count), perimeter, and:

$$
\begin{array} { r l } & { \mathrm { E c c e n t r i c i t y } = \sqrt { 1 - \frac { \lambda _ { \operatorname* { m i n } } } { \lambda _ { \operatorname* { m a x } } } } , } \\ & { \quad \quad \mathrm { S o l i d i t y } = \frac { | \mathcal { R } _ { i } | } { | \mathrm { C o n v } ( \mathcal { R } _ { i } ) | } , } \\ & { \quad \quad \mathrm { O r i e n t a t i o n } = \frac { 1 } { 2 } \arctan \left( \frac { 2 \Sigma _ { 1 2 } } { \sum _ { 1 1 } - \sum _ { 2 2 } } \right) , } \end{array}
$$

where $\lambda _ { \operatorname* { m a x } } , \lambda _ { \operatorname* { m i n } }$ are the eigenvalues of the coordinate covariance matrix Σ of the points in $\mathcal { R } _ { i }$ , and $\operatorname { C o n v } ( \mathcal { R } _ { i } )$ denotes the convex hull of $\mathcal { R } _ { i }$ (the smallest convex polygon containing all pixels of the superpixel). Thus $\vert \mathrm { C o n v } ( \mathcal { R } _ { i } ) \vert$ | is the area of that convex hull, and the ratio $| \mathcal { R } _ { i } | / | \operatorname { C o n v } ( \mathcal { R } _ { i } )$ | measures boundary regularity (1 for a convex shape, $< 1$ for irregular boundaries).

Feature vector. Each superpixel node is initially described by a 60-dimensional vector (30 topological + 25 spectral + 5 geometric). The procedure that selects the final 11 $H _ { 1 }$ topological features retained for the GNN is detailed in the ablation study (Section 3.2).

## 2.4. Hybrid CNN-GNN Architecture

As shown in Fig. 1, the architecture comprises two parallel branches whose outputs are fused before the final classification layer.

CNN Branch. The UMAP-projected image is processed by a fully convolutional network with no pooling, preserving spatial resolution. The architecture comprises three stages: a $3 \times 3$ convolution expanding the $D = 5 ~ \mathrm { U M A P }$ channels (where D is the output dimension of the UMAP reduction from $C ~ = ~ 1 6$ spectral bands) to 32 feature maps; a $3 \times 3$ convolution with a dilation factor of $^ { 2 , }$ achieving a $5 \times 5$ effective receptive field at 64 channels; and a $1 \times 1$ convolution projecting to 64 output channels. Each convolution is followed by batch normalization, and by a ReLU activation for the first two.

GNN Branch. The superpixel graph is constructed with 41-dimensional node features: 11 topological descriptors derived from $H _ { 1 }$ persistence statistics (see Section 3.2; maximum, mean, standard deviation, sum, and minimum persistence; mean and maximum birth and death scales; entropy; and point count) combined with 30 spatial descriptors (perchannel spectral statistics and geometric shape features). These node features are processed by a 3-layer GCN [13] $( 4 1 \ \to \ 6 4 \ \to \ 6 4 \ \to \ 2 ) .$ , with batch normalization, ReLU activation and dropout $( p = 0 . 3 )$ after the first two layers, producing a 2-dimensional output per superpixel that encodes the topological and structural signature of each region. The source code is publicly available<sup>1</sup>.

## 2.5. Fusion

The CNN branch outputs pixel-level features $\mathbf { F } _ { \mathrm { c n n } } \in \mathbb { R } ^ { H \times W \times 6 4 }$ To combine these with the GNN’s superpixel-level outputs $\mathbf { Z } ^ { ( 3 ) } ~ \in ~ \mathbb { R } ^ { N \times 2 }$ , where the superscript (3) indicates the third (final) GCN layer and N is the number of superpixels $( N \ \leq \ 1 5 0 \colon$ SLIC is run with 150 target segments and superpixels of fewer than 5 pixels are discarded), we broadcast each superpixel’s feature vector to all its constituent pixels. Let $S : [ H ] \times [ W ] \to [ N ]$ map each pixel to its superpixel index. The broadcasted feature map $\mathbf { \bar { F } } _ { \mathrm { b r o a d } } ~ \in ~ \mathbb { R } ^ { H \times W \times 2 } ~ \mathrm { i s }$ defined by:

$$
\mathbf { F } _ { \mathrm { b r o a d } , h , w , k } = \mathbf { Z } _ { \mathcal { S } ( h , w ) , k } ^ { ( 3 ) } ,
$$

and set to zero for the pixels of discarded superpixels. The GNN output dimension is set to 2, the number of classes (background and target) of the binary segmentation task; these two channels are latent features learned jointly with the final classifier. The CNN and broadcasted GNN features are concatenated channel-wise and classified by a final $1 \times 1$ convolution:

$$
\widehat { \mathbf { Y } } = \mathbf { C o n v } _ { 6 6  2 } ^ { 1 \times 1 } ( \mathbf { C o n c a t } [ \mathbf { F } _ { \mathrm { c n n } } , \mathbf { F } _ { \mathrm { b r o a d } } ] ) \in \mathbb { R } ^ { H \times W \times 2 } ,
$$

where $\widehat { \mathbf Y }$ contains the per-pixel logits of the two classes. The two branches and the final classifier are trained jointly, with gradients flowing through both branches simultaneously, whereas UMAP, SLIC and persistent homology are fixed preprocessing steps (see Fig. 1).

Training We employ weighted cross-entropy at the pixel level to handle class imbalance:

$$
\begin{array} { r } { \mathcal { L } = - \frac { \sum _ { h , w } \omega _ { Y _ { h , w } } \log \sigma ( \widehat { \mathbf { Y } } _ { h , w } ) _ { Y _ { h , w } } } { \sum _ { h , w } \omega _ { Y _ { h , w } } } , } \end{array}\tag{1}
$$

where σ is the softmax over the two classes, $Y _ { h , w } \in \{ 0 , 1 \}$ is the ground-truth label of pixel $( h , w )$ , and the class weights $\omega _ { 0 } = 1 . 0$ and $\omega _ { 1 } = 5 . 0$ penalize false negatives on the target class.

## 3. EXPERIMENTAL SETUP AND RESULTS

## 3.1. Dataset and Metrics

We evaluate on the cigarette dataset from our prior work [3], consisting of ME-XRT image patches depicting cigarette packs placed over luggage backgrounds, with binary groundtruth masks distinguishing target (cigarette) from background. This dataset is particularly suited for evaluating topological descriptors in XRT imagery because the partial superposition of two distinct materials creates spectral mixing regions at their interfaces, where $H _ { 1 }$ cyclic structures in the UMAP-projected manifold encode material boundaries. Evaluation uses Accuracy and F1-score [21] at both the pixel and the superpixel level (a superpixel is positive $\mathrm { i f } \geq 7 0 \%$ of its pixels are target).

## 3.2. Feature Selection and Ablation

The 11 topological descriptors retained in Section 2.3 were selected by ablating the GNN branch in isolation: the CNN branch was disabled during this phase, so the reported scores reflect pure graph-based classification without pixel-level CNN features. We kept the 30 spatial features (25 spectral statistics + 5 geometric descriptors) fixed and progressively added candidate topological features from the pool of 30 defined in Section 2.3, tracking the F1-score [21] on the test set over 10 runs per configuration (the data set is split beforehand into a training and a test set, same as in [3]).

Table 1 reports the results. Topological features alone (30 candidates on raw HSI) yield the lowest performance (F1 = 0.283), because persistent homology is invariant to pixel permutations within a superpixel and thus discards absolute spectral intensities and spatial coordinates. On raw HSI, adding the 11 selected $H _ { 1 }$ statistics to the spatial features raises F1 from 0.685 to 0.764, whereas expanding to 19 or 30 topological features degrades it to 0.752 and 0.725, respectively: the additional candidates $( H _ { 0 }$ statistics, Betti numbers at fixed scales [7], composite scores) increase node dimensionality from 41 to 60 without yielding extra discriminative power, diluting class-specific signals during graph convolution. The same 11 features bring a similar gain on UMAP-projected data, where F1 rises from 0.778 to 0.823, confirming a significant complementary signal.

Table 1: GNN ablation on raw HSI (16 bands) and UMAPprojected data (5 components). F1 (Target) denotes the F1- score on the positive (target) class; OA denotes Overall Accuracy. All scores are obtained with the GNN branch alone (no CNN fusion).
<table><tr><td>Input</td><td>Node Features</td><td>F1 (Target)</td><td>OA</td></tr><tr><td>Raw HSI Raw HSI</td><td>Topo only (30)</td><td> $\overline { { 0 . 2 8 3 \pm 0 . 0 1 1 } }$ </td><td> $\overline { { 0 . 3 9 1 \pm 0 . 0 0 3 } }$ </td></tr><tr><td rowspan="4">Raw HSI Raw HSI Raw HSI</td><td>Spatial only</td><td> $0 . 6 8 5 \pm 0 . 0 0 5$ </td><td> $0 . 9 1 2 \pm 0 . 0 0 6$ </td></tr><tr><td> $\mathrm { S p a t i a l } + 3 \mathrm { T o p o }$ </td><td> $0 . 6 8 3 \pm 0 . 0 0 8$ </td><td> $0 . 9 4 7 \pm 0 . 0 0 6$ </td></tr><tr><td>Spatial + 11 Topo</td><td> $\mathbf { 0 . 7 6 4 \pm 0 . 0 0 7 }$ </td><td> $0 . 9 3 4 \pm 0 . 0 1 0$ </td></tr><tr><td>Spatial + 19 Topo</td><td> $0 . 7 5 2 \pm 0 . 0 0 3$ </td><td> $0 . 9 4 8 \pm 0 . 0 0 2$ </td></tr><tr><td>UMAP UMAP</td><td>Raw HSI  $\mathrm { S p a t i a l } + 3 0 \mathrm { T o p o }$  Spatial only Spatial + 11 Topo</td><td> $0 . 7 2 5 \pm 0 . 0 0 5$   $\overline { { 0 . 7 7 8 \pm 0 . 0 1 1 } }$   $\mathbf { 0 . 8 2 3 \pm 0 . 0 0 3 }$ </td><td> $0 . 9 3 3 \pm 0 . 0 0 3$   $\overline { { 0 . 9 3 5 \pm 0 . 0 0 9 } }$ </td></tr></table>

This pattern reflects the distinct nature of the descriptors. $H _ { 0 }$ statistics correlate with superpixel size and density rather than material identity, whereas the selected $H _ { 1 }$ descriptors quantify non-trivial cyclic structures in the spectral point cloud arising from spectral mixing in heterogeneous regions. The compact 11-feature subset thus captures the essential topological structure without redundant dimensionality. With and without UMAP. Table 1 also compares persistent homology computed directly in the original 16-band spectral space (raw HSI) and in the 5-dimensional UMAP space. The 11 $H _ { 1 }$ features improve the spatial-only baseline in both spaces (+0.079 in F1 on raw HSI, +0.045 on UMAP), which shows that the cyclic structures they capture are already present in the original spectra and are not created by the embedding. UMAP nevertheless gives the best results, with F1 rising from 0.764 to 0.823 for the same 11-feature configuration: by unfolding the spectral manifold into a lower-dimensional metric space, it removes redundant and noisy dimensions and yields more stable persistence diagrams.

## 3.3. Hybrid Architecture: Quantitative Results

Table 2 compares the two fusion strategies against CNNonly baselines. The broadcasting hybrid achieves the highest scores across all metrics, surpassing the best CNN baseline by +0.086 in superpixel F1 and +0.070 in pixel F1. Even the pooling hybrid outperforms both CNN variants, though it trails broadcasting by −0.061 in superpixel F1.

Table 2: Hybrid CNN+GNN: fusion strategies vs. CNN baselines.
<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>Superpixel</td><td rowspan=1 colspan=2>Pixel</td></tr><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>Acc.</td><td rowspan=1 colspan=1>F1</td><td rowspan=1 colspan=1>Acc.</td><td rowspan=1 colspan=1>F1</td></tr><tr><td rowspan=1 colspan=1>CNN (Pixel-wise)</td><td rowspan=1 colspan=1>0.9605±0.003</td><td rowspan=1 colspan=1>0.8467±0.007</td><td rowspan=1 colspan=1>0.9433±0.004</td><td rowspan=1 colspan=1>0.8245±0.008</td></tr><tr><td rowspan=1 colspan=1>CNN (Superpixel-wise)</td><td rowspan=1 colspan=1>0.9387±0.005</td><td rowspan=1 colspan=1>0.8201±0.009</td><td rowspan=1 colspan=1>0.9317±0.006</td><td rowspan=1 colspan=1>0.8144±0.009</td></tr><tr><td rowspan=1 colspan=1>Hybrid (Pooling)</td><td rowspan=1 colspan=1>0.9589±0.003</td><td rowspan=1 colspan=1>0.8720±0.006</td><td rowspan=1 colspan=1>0.9496±0.004</td><td rowspan=1 colspan=1>0.8569±0.007</td></tr><tr><td rowspan=1 colspan=1>Hybrid (Broadcasting)</td><td rowspan=1 colspan=1>0.9802±0.002</td><td rowspan=1 colspan=1>0.9329±0.003</td><td rowspan=1 colspan=1>0.9640±0.003</td><td rowspan=1 colspan=1>0.8945±0.005</td></tr></table>

These gains show that persistent homology features carry useful information for XRT segmentation. The CNN captures local spectral-spatial patterns in the UMAP embedding, but it does not characterize the global structure of the spectral manifold. The GNN supplies this through $H _ { 1 }$ statistics that quantify cyclic structures in each superpixel’s spectral point cloud. These cycles encode spectral mixing, the patterns that appear when the X-ray beam crosses multiple materials, and are not visible to convolutional filters alone. The hybrid model therefore outperforms both CNN baselines because topological descriptors capture patterns that local features miss.

The gap between broadcasting and pooling comes from spatial resolution. Broadcasting sends each superpixel’s topological descriptor to all its pixels, letting the final classifier mix local CNN features with global topological information at full resolution. Under pooling, the CNN map is collapsed to superpixel averages before fusion, destroying fine spatial detail. Broadcasting is also more robust to segmentation errors: a bad superpixel poisons all its pixels under pooling, but under broadcasting the wrong topological vector only affects 2 of the 66 channels, which the CNN can partially correct.

These results extend our previous work [3], which relied on the UMAP projection alone: adding $H _ { 1 }$ statistics through the GNN branch yields clear further gains, confirming that the topological structure of the spectral manifold carries discriminative information beyond the UMAP coordinates themselves.

Figure 2 supports this: superpixels with high $H _ { 1 }$ maximum persistence (warm colors) correspond to regions where the spectral point cloud has complex structure, while homogeneous regions show low persistence (cool colors). The topological signal is strongest where spectral mixing is most pronounced.

## 4. CONCLUSION

This study investigated the integration of persistent homology features into a hybrid CNN-GNN framework for multienergy XRT hyperspectral image segmentation. Our findings demonstrate that 1-dimensional persistent homology statistics derived from UMAP-projected superpixel point clouds effectively encode the non-linear structure and spatial complexity of the spectral manifold. In particular, these topological descriptors capture local spectral mixing at material interfaces, providing complementary discriminative information that standard spectral-spatial features miss. Future works may focus on a deeper theoretical and empirical exploration of topological descriptors in X-ray hyperspectral imaging, investigating why persistent homology is particularly suited to modeling material boundaries, extending the framework to complex multi-class scenarios, and exploring adaptive superpixel partitioning strategies.

![](images/4f40b0f82fcf4cad023da808e644e633e594ea0798d1dd689257dad51eb77595.jpg)  
Fig. 2: $H _ { 1 }$ maximum persistence per superpixel overlaid on UMAP intensity. Warm colors indicate high persistence in heterogeneous regions; cool colors indicate homogeneous regions.

## 5. REFERENCES

[1] Ian T. Jolliffe and Jorge Cadima, “Principal component analysis: A review and recent developments,” Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences, vol. 374, no. 2065, pp. 20150202, 2016.

[2] Charles M Bachmann, Thomas L Ainsworth, and Robert A Fusina, “Exploiting manifold geometry in hyperspectral imagery,” IEEE Transactions on Geoscience and Remote Sensing, vol. 43, no. 3, pp. 441–454, 2005.

[3] Fethi Harkat, Guillaume Gey, Valérie Perrier, Kévin Polisano, and Tiphaine Debeuret, “Manifold learning for hyperspectral images,” in 2025 15th Workshop on Hyperspectral Imaging and Signal Processing: Evolution in Remote Sensing (WHISPERS). IEEE, 2025, pp. 1–7.

[4] Leland McInnes, John Healy, and James Melville, “Umap: Uniform manifold approximation and projection for dimension reduction,” arXiv preprint arXiv:1802.03426, 2018.

[5] John Healy and Leland McInnes, “Uniform manifold approximation and projection,” Nature Reviews Methods Primers, vol. 4, no. 1, pp. 82, 2024.

[6] Chen Lou, Mohammed AA Al-qaness, Sike Ni, Dalal Al-Alimi, Robertas Damaševicius, and Saeed Hamoodˇ Alsamhi, “Hyperspectral image classification using uniform manifold approximation and projection with fusion deep learning network,” Applied Soft Computing, vol. 180, pp. 113371, 2025.

[7] Herbert Edelsbrunner and John Harer, Computational topology: an introduction, American Mathematical Soc., 2010.

[8] Gunnar Carlsson, “Topology and data,” Bulletin of the American Mathematical Society, vol. 46, no. 2, pp. 255– 308, 2009.

[9] Zonghan Wu, Shirui Pan, Fengwen Chen, Guodong Long, Chengqi Zhang, and Philip S. Yu, “A comprehensive survey on graph neural networks,” IEEE Transactions on Neural Networks and Learning Systems, vol. 32, no. 1, pp. 4–24, 2020.

[10] Jie Zhou, Ganqu Cui, Shengding Hu, Zhengyan Zhang, Cheng Yang, Zhiyuan Liu, Lifeng Wang, Changcheng Li, and Maosong Sun, “Graph neural networks: A review of methods and applications,” AI Open, vol. 1, pp. 57–81, 2020.

[11] Qinghui Liu, Liangpei Xiao, Jie Yang, and Zhongyi Wei, “CNN-enhanced graph convolutional network with pixel- and superpixel-level feature fusion for hyperspectral image classification,” IEEE Transactions on Geoscience and Remote Sensing, vol. 59, no. 10, pp. 5966– 5978, 2020.

[12] Danfeng Hong, Lianru Gao, Naoto Yokoya, Jing Yao, Jocelyn Chanussot, Qian Du, and Bing Zhang, “Graph convolutional networks for hyperspectral image classification,” IEEE Transactions on Geoscience and Remote Sensing, vol. 59, no. 10, pp. 5966–5978, 2020.

[13] Thomas N. Kipf and Max Welling, “Semi-supervised classification with graph convolutional networks,” in Proceedings of the 5th International Conference on Learning Representations (ICLR), Toulon, France, 2017.

[14] Radhakrishna Achanta, Appu Shaji, Kevin Smith, Aurelien Lucchi, Pascal Fua, and Sabine Süsstrunk, “SLIC superpixels compared to state-of-the-art superpixel methods,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 34, no. 11, pp. 2274– 2282, 2012.

[15] Ulrich Bauer, “Ripser: Efficient computation of vietoris-rips persistence barcodes,” Journal of Applied and Computational Topology, vol. 5, no. 3, pp. 391–423, 2021.

[16] Henry Adams, Tegan Emerson, Michael Kirby, Rachel Neville, Chris Peterson, Patrick Shipman, Sofya Chepushtanova, Eric Hanson, Francis Motta, and Lori Ziegelmeier, “Persistence images: A stable vector representation of persistent homology,” Journal ofMachine Learning Research, vol. 18, no. 8, pp. 1–35, 2017.

[17] Peter Bubenik, “Statistical topological data analysis using persistence landscapes,” Journal ofMachine Learning Research, vol. 16, no. 1, pp. 77–102, 2015.

[18] Jan Reininghaus, Stefan Huber, Ulrich Bauer, and Roland Kwitt, “A stable multi-scale kernel for topological machine learning,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Boston, MA, USA, 2015, pp. 3071–3078.

[19] Mathieu Carriëre, Theo Lacombe, Martin Glisse, and Steve Oudot, “Perslay: A neural network layer for persistence diagrams and new graph topological signatures,” in Proceedings ofthe 23rd International Conference on Artificial Intelligence and Statistics (AISTATS), Palermo, Sicily, Italy, 2020, vol. 108 of Proceedings of Machine Learning Research, pp. 2786–2796.

[20] Max Horn, Edward De Brouwer, Michael Moor, Yves Moreau, Bastian Rieck, and Karsten Borgwardt, “Topological graph neural networks,” ICLR 2022, 2022.

[21] D. M. W. Powers, “Evaluation: from precision, recall and F-measure to ROC, informedness, markedness and correlation,” Journal of Machine Learning Technologies, vol. 2, no. 1, pp. 37–63, 2011.