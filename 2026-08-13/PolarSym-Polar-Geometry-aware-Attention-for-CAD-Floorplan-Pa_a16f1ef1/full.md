# PolarSym: Polar Geometry-aware Attention for CAD Floorplan Parsing

Kerui Chen<sup>∗1,2</sup>, Yiqing Wang<sup>†1</sup>, Kangzhou Xin<sup>‡2</sup>, Qinghan Zhang<sup>§2</sup>, and Songyang Ding<sup>¶1,2</sup>

<sup>1</sup>School of Computer and Information Engineering, Henan University of Economics and Law, Zhengzhou, China

<sup>2</sup>Third Dimension (Henan) Software Technology, Zhengzhou, China

August 13, 2026

## Abstract

CAD plan parsing constitutes a fundamental task in Building Information Modeling (BIM), whose objective is to automatically recover architectural elements such as walls, doors, windows, and furniture from 2D engineering drawings. Although existing Transformer-based methods can capture global semantic dependencies through self-attention mechanisms, they primarily infer spatial relationships from semantic features without explicitly modeling the inherent geometric symmetry of building layouts. Consequently, these methods are vulnerable to incorrect correspon dences when handling long-range matching and complex symmetric spatial configurations. To address these issues, this paper proposes a polar-coordinate geometry-aware attention framework named PolarSym for CAD plan parsing. The framework explicitly decouples architectural geometric relationships into two complementary factors—direction and distance—and models them separately. It enhances structural consistency through directional constraints and establishes longrange symmetric correspondences via distance constraints. Additionally, a dynamic gating mechanism is employed to achieve synergistic fusion of the two geometric information streams while preserving the core transformer architecture. This design significantly improves geometric modeling capabilities with minimal computational overhead. Experimental results on a public CAD plan parsing dataset demonstrate that PolarSym achieves a performance improvement of 1.73% PQ, 1.54% RQ, and 4.31% mIoU over the reproduced SymPoint V2 baseline under identical training configurations. Furthermore, PolarSym exhibits faster convergence and more stable optimization dynamics. Ablation studies further validate the complementary role of directional and distance modeling. The results indicate that PolarSym efectively enhances the geometric perception capability of transformers with low computational overhead, providing an eficient geometric modeling solution for CAD plan parsing tasks.

Keywords: CAD floorplan parsing; Geometry-aware attention; Transformer; Positional encoding; Building information modeling(BIM)

## 1 Introduction

Computer-Aided Design (CAD) planar diagram parsing aims to automatically identify building components such as walls, windows, doors, and furniture from architectural drawings and reconstruct their structured semantic representations. As a foundational task for Building Information Modeling (BIM), digital twins, and indoor scene understanding, this problem holds significant application value in intelligent buildings, automated architectural design, and indoor navigation. With the advancement of deep learning, neural network-based methods have gradually replaced traditional rule-driven algorithms, substantially improving the automation level and reconstruction accuracy of CAD planar diagram parsing.

In recent years, Transformers have demonstrated superior modeling capabilities over convolutional neural networks in CAD planar diagram parsing tasks due to their global self-attention mechanisms. The SymPoint series represents a representative attempt to incorporate Point Transformer into architectural floor plan parsing, where point-wise feature interactions enable competitive instance-level parsing performance. However, existing Transformer methods primarily construct attention based on semantic similarity, while spatial geometric relationships are often implicitly represented via positional encoding without explicit modeling of architectural geometric structures. When confronted with repeated layouts, mirror structures, or long-range symmetric targets, attention mechanisms relying solely on semantic similarity tend to degenerate into local associations, making it dificult to establish stable long-range structural correspondences.

Architectural plans exhibit distinct geometric patterns that difer significantly from natural images, with numerous building components arranged in regular axisymmetric or repetitive distributions along the building’s primary axis. The structural relationships of such components must satisfy not only semantic consistency but also be constrained by direction consistency (Direction Consistency) and radial correspondence (Radial Correspondence). Here, direction determines whether two components undergo the same symmetric transformation, while distance dictates their spatial correspondence. Current position encodings or distance biases typically employ unified Cartesian coordinates, mixing diferent geometric attributes or assuming monotonic relationships between distance and relevance. Such modeling approaches fail to accurately describe the prevalent long-range symmetric structures in architectural plans, thereby limiting Transformers’ capability to model global geometric relationships.

To illustrate the aforementioned issue, Figure 1 presents the attention visualization results of the baseline model and the proposed method under identical CAD inputs and query point (Query) conditions. It can be observed that the baseline model concentrates its attention primarily around the query point, establishing only local spatial associations while struggling to connect distant windows with identical structural attributes. In contrast, the proposed method not only accurately localizes the current target but also establishes long-range structural correspondences between multiple symmetric windows. This demonstrates that relying solely on semantic features is insuficient to fully capture the geometric patterns of component arrangement in architectural plane figures, whereas explicitly incorporating geometric constraints efectively enhances the Transformer’s capability to model repetitive structures and symmetric relationships.

To address the aforementioned challenges, this paper proposes a polar-coordinate geometry-aware attention framework for CAD plane figure parsing—PolarSym. Unlike existing methods, PolarSym does not treat spatial information as a unified positional encoding. Instead, it explicitly decouples architectural geometric relationships during attention computation into two complementary factors—direction and distance—and models them separately. The direction branch enhances globa directional consistency, while the distance branch establishes long-range structural correspondences. These two branches are jointly regulated by a unified dynamic gating mechanism to modulate the Transformer attention, thereby enabling synergistic semantic information and geometric prior modeling without altering the original network architecture.

The main contributions of this work are summarized as follows:

• Proposal of a polar-coordinate geometry-aware attention framework, PolarSym. Unlike conventional positional encodings, PolarSym explicitly models geometric relationships among architectural components in the attention layer by decoupling directional and distance information, providing a unified geometric modeling framework for CAD plane figure parsing.

• Design of a direction–distance collaborative modeling mechanism. We propose a direction modeling method based on SF-RoPE (Split-Feature Rotary Position Embedding) and a radial modeling method based on RDB (Resonant Distance Bias), and leverage a dynamic gating strategy to achieve adaptive fusion of the two geometric modalities, thereby efectively enhancing the Transformer’s capability to represent long-range symmetric structures.

• Empirical validation on public CAD plane figure parsing datasets. Under identical training configurations, PolarSym outperforms the SymPoint V2 baseline across multiple evaluation metrics including PQ, RQ, SQ, and mIoU, while demonstrating faster convergence and more stable optimization dynamics, thereby validating the efectiveness of explicit geometric modeling for architectural plane figure parsing tasks.

![](images/f8260e72b86302c89b86cc9aae99615271c0d873e99c576ab25417d78f8b7de2.jpg)  
Figure 1: Visual comparison of baseline models and our method.

## 2 Related Work

## 2.1 Planar CAD Diagram Parsing

In recent years, the advancement of technologies such as Building Information Modeling (BIM), Digita Twin, intelligent construction, and indoor navigation has positioned automatic CAD diagram parsing as an emerging research direction within computer vision, graphics, and architectural intelligence. CAD diagrams not only encompass rich geometric primitives but also encode complex spatial topological relationships and architectural semantics, making the accurate modeling of spatial relationships between primitives a core challenge in CAD diagram understanding. Rezvanifar et al. conducted a systematic review of CAD diagram parsing methods, classifying existing approaches based on input data representation paradigms into four categories: Raster-based, Vector-based, Point-based, and Multimodal-based [RCBA19]. Difering data representation paradigms determine the types of information models can leverage, thereby directly influencing the subsequent development trajectories of transformer architectures.

## 2.1.1 Raster-based Methods

Raster-based methods first convert CAD drawings into two-dimensional images before leveraging convolutional neural networks (CNNs) to learn visual features. Given that CNNs inherently possess local receptive fields and translation invariance, early architectural drawing recognition predominantly employed such approaches.

In recent years, multimodal fusion has further advanced raster-based methods. Xing et al. proposed a multimodal aggregation framework that represents architectural symbols as visual, geometric, and textual modalities [XGZ<sup>+</sup>26]. This framework designs a hybrid sampling strategy while simultaneously employing boundary dense sampling (boundary dense sampling) and region-coarse sampling (region-coarse sampling) to extract symbol contour details and internal contextual information, respectively. Additionally, to enhance geometric representation capabilities, the method introduces normalized shape encoding (normalized shape encoding), which encodes geometric statistics—including area, perimeter, orientation angle, aspect ratio—into the feature space and integrates CAD textua information, thereby improving architectural symbol detection performance.

Although this approach efectively utilizes multimodal information, its underlying architecture remains within the CNN framework. Consequently, the model requires multiple rounds of sampling and feature fusion to acquire global contextual information, resulting in slow training convergence. Furthermore, the prevalence of repetitive elements in large-scale architectural drawings significantly burdens network computation, making it challenging to meet the demands of high-throughput automatic review in engineering scenarios.

## 2.1.2 Vector-Based Methods

Given that CAD inherently represents vector graphics, an increasing number of studies have directly leveraged CAD primitives for representation learning. Liu et al. first employed ResNet-152 to predict junction layers (Junction Layer) in architectural drawings, where topological nodes are recovered through intersection detection [LWKF17]. Subsequently, an image partition (IP) mechanism is utilized to reconstruct the primitive layer, encompassing primitives such as line segments, boundaries, and loca contours. The junction layer fundamentally encodes node connectivity relationships within architectural spaces, while the primitive layer records geometric constituents of CAD drawings. Compared to direct pixel classification, this approach recovers structural information from architectural drawings. However, since IP primarily relies on regional segmentation and local structural constraints, it recovers only low-level geometric relationships, lacking higher-level topological representations between architectural components. Consequently, this method exhibits limited capability for long-range association modeling in complex architectural spaces.

Subsequently, Jiang et al. proposed YOLaT (You Only Look at Text), which first performed direct object detection on vector graphics [JLS<sup>+</sup>21]. This method represents CAD primitives as Bezier curves and constructs a bi-stream graph neural network to jointly learn geometric and spatial relationships, enabling architectural component detection. Compared to raster-based methods, YOLaT avoids information loss caused by rasterization and directly leverages connectivity relationships between CAD primitives, thereby significantly improving detection accuracy. To further enhance vector graphics understanding, Dou et al. proposed YOLaT++, which not only expanded the VG-DCU large-scale vector graphics dataset but also established a comprehensive training dataset integrating vector graphics, rasterized images, annotations, and original CAD data, providing a unified foundation for transformer and graph network models [DJL<sup>+</sup>24]. However, regardless of YOLaT or YOLaT++, both methods depend on Bezier curves to construct complex graph structures. As architectural scale increases, the number of Bezier curves grows exponentially, leading to heightened computational complexity in graph neural network message propagation and consequently reduced training stability and inference eficiency.

Concurrently, ReGroup employs recursive neural networks for hierarchical grouping of CAD primitives [CLC21]. The authors leverage contrastive learning to learn high-dimensional representations of primitive subsets and perform building component clustering based on cosine similarity. By eliminating the need for complex graph neural networks, ReGroup achieves substantially improved computational eficiency. However, this approach is limited to CAD drawings with simple hierarchical structures, no occlusions, and no layer overlaps—making it inefective for handling complex spatial relationships in real-world architectural drawings.

PanCADNet introduces a CNN-GCN fusion network that jointly models geometric features (length, position), visual features, and type features (line segments, arcs) of CAD primitives [FZL<sup>+</sup>21]. By leveraging graph convolutional networks (GCNs) to learn topological relationships between primitives, PanCADNet achieves unified identification of things and stuf. Nevertheless, the GCN-based graph construction and message-passing process incurs substantial computational overhead, particularly when architectural drawings contain numerous entities (e.g., complex commercial buildings), where the proliferation of nodes and edges may degrade both training and inference eficiency. Additionally, if annotations contain noise or entity boundaries are ambiguous, the GCN’s topological feature learning may be compromised.

Subsequently, GAT-CADNet further introduced relative spatial encoding (Relative Spatial Encoding, RSE) and cascaded edge encoding (Cascaded Edge Encoding, CEE) [ZLZ<sup>+</sup>22]. Specifically, RSE establishes spatial encodings using relative coordinates of primitives, while CEE enhances information propagation within graph neural networks through multi-level edge representations, enabling joint optimization of semantic and instance recognition.

Following this, VectorFloorSeg proposed a bi-stream GNN attention network, where one stream learns line boundary information and the other stream learns regional information [YJPX23]. A crossstream modulation mechanism was designed to facilitate inter-stream interactions, thereby achieving more complete planar floor segmentation. However, this approach exhibits several limitations: it is prone to misclassification in minimal regions (e.g., narrow corridors) due to insuficient sampling points leading to low feature embedding quality; it struggles to distinguish morphologically similar categories (e.g., wall panels versus railings), necessitating additional shape priors (e.g., window and door features); it currently only supports straight line segments, requiring polygonal approximation for complex curves (e.g., B-splines, Bezier curves), which may incur geometric precision loss; and the bi-stream graph neural network with cross-stream modulation increases model complexity, potentially resulting in slower inference speeds compared to lightweight image segmentation methods.

## 2.1.3 Point-based Methods

CADSpotting enables symbol recognition in large-scale architectural CAD drawings by representing each basic geometric shape through dense sampling points and describing its coordinate and color attributes [YMZ<sup>+</sup>24]. This approach addresses challenges such as symbol diversity, scale variations, and overlapping elements in CAD design, achieving integrated learning of semantic, instance, and panoptic segmentation. To achieve accurate segmentation in large and complex drawings, the authors further propose the Sliding Window Aggregation (SWA) technique, which combines weighted voting and non-maximum suppression (NMS). However, this method exhibits poor stability and lacks globa topological characteristics.

SymPoint represents an initial attempt to address the panoptic symbol localization task in CAD drafting using point-set representations [LYW<sup>+</sup>24]. Despite achieving significant success, SymPoint neglects geometric layer information and exhibits very slow convergence during training. To resolve these issues, SymPoint-V2 was introduced, building upon SymPoint by proposing a Layer Feature Enhancement Module (LFE) to encode geometric layer information into raw features and a Position-Guided Training (PGT) method to accelerate model convergence in early training stages [LYYZ24]. In current benchmark tests, SymPoint-V2 demonstrates state-of-the-art performance among existing models.

## 2.1.4 Multimodal-based Methods

FloorNet employs a three-branch hybrid network that cross-fuses 3D point data and RGB data to fully leverage local spatial information and partial global layout information [LWF18]. This approach is computationally complex, exhibits low computational eficiency, and achieves low extraction accuracy.

Im2Vec is an end-to-end variational autoencoder (VAE) architecture where the encoder maps raster images to a global latent vector, and the decoder generates path latent vectors via recurrent neural networks (RNNs) [RGLM21]. These path latent vectors are then decoded into closed B´ezier curves through a path decoder, supporting variable path counts and topologies. However, this method relies on image-space losses that may cause fine-grained component loss, necessitating either increased training resolution or the design of more sophisticated loss functions for mitigation. Additionally, although it supports dynamic pruning of degenerate paths, the maximum path count T must be pre-specified, limiting its capability for generating highly complex geometries.

RendNet constructs hypergraphs by converting vector graphics (VG) primitives (curves and surfaces) into hypergraphs, where nodes represent curve intersections or endpoints, edges correspond to curve segments, and hyperedges denote surfaces [SJS<sup>+</sup>22]. This architecture implements a dual-stream network: (i) a vector stream (hypergraph neural network (HGNN) for learning topological features) and (ii) a raster stream (via latent space rasterization (LSR), which renders VG into point clouds followed by PointNet for spatial feature extraction). The dual-stream features are fused through residual blocks before global aggregation to generate the target representation. Nevertheless, hypergraph construction relies on prior knowledge, with node selection (curve intersections and endpoints) and hyperedge definition (surface parameters) requiring manual design. Consequently, this approach may exhibit limited adaptability to complex or non-standard VG formats (e.g., custom curve types).

## 2.2 Geometric Perception Attention

In recent years, Transformer-based architectures have become a widely adopted solution in visual understanding tasks, benefiting from their ability to capture global spatial-contextual dependencies through self-attention mechanisms.

DeepSVG initially leveraged the hierarchical structure of transformers to learn vector graphic representation, partitioning SVG graphics into two levels: Shape-level and Command-level [CDAT20]. These levels encode high-level shape semantics and Bezier command sequences, respectively. The transformer first independently encodes each shape token before utilizing global self-attention to learn relationships between diferent shapes, thereby obtaining a global representation of the entire SVG graphic. DeepSVG fully exploits the arrangement invariance characteristic of SVG, enabling transformers to overcome the local receptive field limitations of CNNs and achieve generation and editing of complex vector graphics. However, this approach requires global attention computation over all shape tokens, resulting in a time complexity that becomes computationally prohibitive when CAD drawings contain tens of thousands of elements. Additionally, its spatial representation remains predominantly dependent on shape positions, lacking explicit modeling of directional relationships, distance constraints, and architectural topological constraints—thus hindering direct transfer to architectura CAD drawing parsing tasks.

Fan et al. proposed CADTransformer, decomposing architectural CAD drawings into primitive elements and leveraging HRNet to extract raster image features, which are then combined with element coordinates to generate primitive tokens [FCWW22]. To enhance local geometric relationship modeling, the authors designed neighborhood-aware self-attention (Neighborhood-aware Self-Attention), enabling information propagation from local to global scales through progressively expanding neighborhoods. They further introduced hierarchical feature fusion to integrate features across diferent scales and designed graphic entity position encoding for CAD element positioning. CADTransformer ultimately established Semantic Heads and Instance Heads to achieve architectural symbol category prediction and instance center ofset prediction.

Compared to CNN and GCN methods, CADTransformer first demonstrated that transformers can efectively learn long-range contextual relationships within CAD drawings. However, its token features still rely on raster image features extracted by HRNet, fundamentally failing to escape raster representation. Additionally, the position encoding primarily uses absolute coordinate encoding, which only reflects element positions and cannot express the more critical directional and distance relationships between architectural components. When architectural drawings contain numerous repetitive room layouts, reliance on coordinate positions often leads to feature confusion across diferent instances.

The self-attention mechanism in transformers exhibits quadratic computational complexity, causing training and inference resource demands to escalate sharply as sequence length increases. To address this, researchers have proposed linear transformers that aim to improve attention computation eficiency. However, when sequence length exceeds the model dimension, linear transformers sufer from memory collisions, impairing precise retrieval of critical information. To resolve this, the Gated DeltaNet architecture was introduced, combining gating mechanisms with delta update rules to enhance linear transformers’ performance in long-sequence modeling and information retrieval tasks [YKH25]. Nevertheless, this method incurs additional storage overhead and has not been experimentally validated on vector graphics data, making it incapable of capturing fine-grained topologica relationships within diagrams.

Representative methods such as the SymPoint series have introduced Point Transformers into CAD planar drawing parsing, significantly improving instance-level recognition performance. By representing architectural primitives as point entities, these methods efectively capture semantic interactions between diferent building components and achieve state-of-the-art performance on public benchmarks.

Despite these advances, existing transformer-based methods still primarily construct attention based on feature similarity. The geometric regularity of architectural layouts is implicitly encoded through positional embeddings, making it challenging to establish reliable correspondences between components that are structurally symmetric but distant. Consequently, parsing performance degrades in scenarios involving repetitive layouts, mirror structures, and long-range architectural dependencies.

To enhance spatial representation learning, numerous studies have incorporated geometric information into transformer attention mechanisms. Position encoding is the most widely adopted strategy, including absolute position encoding, relative position encoding, and rotational position embedding (RoPE) [SAL<sup>+</sup>24]. RoPE preserves relative positional relationships through feature rotation and has demonstrated exceptional performance in visual transformers and large language models. Beyond position encoding, some research integrates pairwise spatial information into attention computation to introduce geometric perception attention. Typical approaches employ relative positional ofsets or distance-aware attention to improve local geometric consistency. These methods have achieved promising results in point cloud understanding, object detection, and 3D scene analysis.

However, existing geometric perception attention mechanisms remain insuficient for architectural planar drawing parsing. First, most position encoding methods represent geometry in Cartesian coordinates, embedding both directional and distance information into a single representation. This implicit encoding hinders the network’s ability to distinguish distinct geometric factors during attention learning. Second, conventional distance-aware attention typically assumes a monotonic relationship between spatial distance and feature correlation. While this assumption holds for natural images and point clouds, it conflicts with architectural drawings, where symmetric structures are often separated by larger spatial distances. Consequently, traditional geometric attention mechanisms tend to underestimate long-range symmetric correspondences.

## 2.3 Discussion

The aforementioned review demonstrates that current CAD planar diagram parsing methods achieve robust semantic representations via Transformer architectures, while existing geometry-aware attention mechanisms enhance spatial modeling through positional or distance constraints. However, neither direction nor distance has been explicitly modeled as independent geometric priors, and the intrinsic symmetry of architectural layouts remains underutilized.

Inspired by this, we propose PolarSym, a polar coordinate geometry-aware attention framework that explicitly decomposes architectural geometry into two complementary components: direction and distance. Directional consistency is modeled via Split Feature Rotation Position Embedding (SF-RoPE), while long-range radial correspondences are captured by Resonance Distance Deviation (RDB). Combined with a dynamic gating mechanism, the proposed framework integrates geometric priors directly into Transformer attention, thereby enabling more reliable structural correspondences for CAD planar diagram parsing.

## 3 Methods

## 3.1 Overall Framework

The overall architecture of the proposed PolarSym framework is illustrated in Figure 2.

Given a set of point features extracted by the Point Transformer backbone, our objective is to enhance the attention mechanism through explicit geometric reasoning while preserving the original Transformer architecture.

Let ${ \bf F } = \{ f _ { i } \} _ { i = 1 } ^ { N } \in \mathbb { R } ^ { N \times D }$ represent the semantic feature embeddings of N points, where D denotes the feature dimension. Each point is associated with two-dimensional spatial coordinates. The data $\mathbf { P } = \{ p _ { i } = ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N } \in \mathbb { R } ^ { N \times \hat { 2 } }$ is first fed into the backbone network, which generates query, key, and value representations via linear projection: $\mathbf { Q } = \mathbf { F W } _ { Q } , \quad \mathbf { K } = \mathbf { F W } _ { K } , \quad \mathbf { V } = \mathbf { F W } _ { V }$

Subsequently, the layer ID features are fused into the Layer Feature Enhanced Module (LFEM), followed by the PolarSym decoder module. PolarSym does not compute attention scores based on semantic similarity directly but instead enhances attention scores using two complementary geometric priors. Specifically, the proposed framework explicitly decomposes geometric relationships into direction and distance, and models them through two parallel branches: Direction branch: Responsible for capturing global directional consistency via Split Feature Rotation Position Embedding (SF-RoPE); Radial branch: Responsible for modeling long-range geometric correspondences via Resonance Distance Bias (RDB).

The outputs of the two branches are integrated into the original attention scores through a lightweight dynamic gating mechanism to produce the final geometric-aware attention map. Therefore, the proposed framework retains the Transformer’s global semantic modeling capability while explicitly incorporating architectural geometric priors into the attention computation. Unlike existing approaches that implicitly embed spatial information into feature representations via positional encodings, PolarSym directly models geometry in the attention domain. This decoupled formulation enables the collaborative optimization of semantic representations, directional consistency, and radial correspondences without modifying the backbone architecture.

## 3.2 PolarSym Geometric Attention

Transformer attention estimates the correlation between two points based on semantic similarity:

$$
S ^ { \mathrm { s e m } } = { \frac { \mathbf { Q } \mathbf { K } ^ { \top } } { \sqrt { d } } }\tag{1}
$$

![](images/28b3b3689ca87a5453e9160de336d79d1a5a599beebcf894f5d0e6eae3901a62.jpg)  
Figure 2: PolarSym framework diagram.

where d denotes the feature dimension per attention head.

Although semantic attention efectively captures contextual relationships, it does not explicitly encode geometric regularities inherent in building layouts. In CAD floor plans, distant building elements may exhibit weak semantic relationships while still belonging to the same underlying geometric symmetry pattern. Consequently, relying solely on semantic similarity often leads to inaccurate long-range correspondences.

To address this, we propose PolarSym geometric attention, which enhances semantic attention through explicit geometric reasoning. As shown in Figure 3.

PolarSym geometric attention does not represent geometry as a unified positional embedding, but decomposes geometric information into two complementary components: Direction consistency: describes whether two points satisfy the same symmetric transformation; Radial correspondence: quantifies whether two points exhibit compatible geometric distances under building symmetry.

Compared with traditional geometric perception attention, PolarSym introduces two critical distinctions. First, direction and radial information are independently modeled rather than jointly encoded into positional embeddings, enabling each geometric factor to focus on specific aspects of building component symmetry within the planar graph. Second, geometric information is injected via additive attention modulation rather than feature transformation, facilitating seamless integration into existing Transformer architectures with negligible computational overhead.

![](images/f0549517e2c3817d4e108c4ad169bf0b7f6324776adc729dc437e200c1132f57.jpg)  
Figure 3: PolarSym geometric attention computation.

## 3.2.1 Directional Modeling

The directional branch aims to enhance attention through explicit orientation awareness. In architectural floor plans, symmetric structures typically exhibit consistent directional relationships relative to the building’s primary axis. However, conventional positional encoding implicitly preserves relative positions without explicitly distinguishing distinct symmetric orientations.

To address this issue, we propose Split Feature Rotation Position Embedding $\left( { \mathrm { S F } } { \mathrm { - R o P E } } \right)$ . As illustrated in Figure 4.

Unlike conventional Rotation Position Embedding (RoPE), which performs rotation on adjacent feature pairs, SF-RoPE divides feature dimensions into two complementary halves and executes crosshalf feature rotation. This design enables long-range interactions between distant feature channels, thereby improving global directional representation.

Given a query (or key) feature vector $\pmb { x } = [ x ^ { ( 1 ) } , x ^ { ( 2 ) } ]$ , SF-RoPE first splits the feature into two equal parts: $x ^ { ( 1 ) } = x _ { 1 : D / 2 } , \ x ^ { ( 2 ) } = x _ { D / 2 + 1 : D }$ . For each feature pair, its rotated representation is computed as:

$$
\left[ x _ { k + D / 2 } ^ { \prime } \right] = \left[ \cos \theta _ { k } \quad - \sin \theta _ { k } \right] \left[ x _ { k + D / 2 } \right] .\tag{2}
$$

where the rotation angle $\theta _ { k }$ is determined by physical coordinates: $\theta _ { k } = \lambda \phi ( { \pmb p } _ { i } )$ . Here, $\phi ( \pmb { p } )$ denotes an angle mapping function that takes positional coordinates as input, and λ is a learnable frequency scaling factor. Diferent attention heads alternately adopt horizontal and vertical coordinates to encode directional information along the two primary axes.

The rotated query and key are then used to compute directional attention: $\begin{array} { r } { S ^ { \mathrm { d i r } } = \frac { \mathbf { Q } ^ { \prime } \mathbf { K } ^ { \prime \top } } { \sqrt { d } } } \end{array}$ . The directional enhancement term is obtained via:

$$
\Delta ^ { \mathrm { d i r } } = S ^ { \mathrm { d i r } } - S ^ { \mathrm { s e m } }\tag{3}
$$

The directional branch does not replace semantic attention but solely estimates geometric-induced attention increments. Consequently, semantic representations and directional geometry remain decoupled, enabling the network to explicitly learn architectural symmetry while preserving original semantic relationships.

In contrast to conventional RoPE, which establishes local adjacent channel rotations, SF-RoPE facilitates global channel interactions. This design enables more efective modeling of long-range directional consistency prevalent in CAD floor plans.

![](images/516c64daad10ef5af826f1db437f0a00526d1f7ee8b14db42e91a6de3e3be7bf.jpg)  
Figure 4: Split feature rotation position embedding representation.

## 3.2.2 Radial Modeling

Although directional branches can establish global directional consistency, relying solely on directional information is insuficient to determine whether two points exhibit reasonable spatial correspondence. Therefore, we further introduce a radial branch to model pairwise distance relationships, thereby enhancing long-range geometric correlation capabilities.

Existing studies demonstrate that directly incorporating spatial distance information into attention computation can efectively improve geometric modeling capabilities, such as through distance bias or relative spatial encoding modulation of attention weights. However, such methods generally perform well in local spatial modeling, while symmetric components in architectural floor plans often span extensive spatial ranges. Sole reliance on local distance constraints hinders the accurate establishment of long-range structural correspondences.

To address this, we propose the Resonant Distance Bias (RDB) module. This module learns a nonmonotonic response function in the distance space to assign higher attention responses to distant point pairs with structural correspondence, without assuming a fixed decay relationship between distance and correlation. As illustrated in Figure 5.

For two points located at $\pmb { p } _ { i } = ( x _ { i } , y _ { i } )$ and $\pmb { p } _ { j } = ( x _ { j } , y _ { j } )$ , their Euclidean distance is first computed: $d _ { i j } = \lVert \pmb { p } _ { i } - \pmb { p } _ { j } \rVert _ { 2 }$ . The RDB module does not directly utilize the scalar distance d but transforms it into a Fourier feature representation:

$$
\gamma ( d _ { i j } ) = \bigl [ d _ { i j } , \sin ( 2 ^ { 0 } \pi d _ { i j } ) , \cos ( 2 ^ { 0 } \pi d _ { i j } ) , \ldots , \sin ( 2 ^ { L - 1 } \pi d _ { i j } ) , \cos ( 2 ^ { L - 1 } \pi d _ { i j } ) \bigr ]\tag{4}
$$

where L denotes the number of frequency bands. The encoded distance features are then processed by a lightweight multilayer perceptron (MLP):

$$
B ^ { \mathrm { r a d } } = \mathrm { M L P } \big ( \gamma ( d _ { i j } ) \big )\tag{5}
$$

This network predicts the radial attention bias.

Unlike linear distance bias, Fourier encoding enables the network to learn periodic distance responses with multiple local maxima. Consequently, the radial branch assigns higher attention values to distant but symmetric structures while suppressing geometrically inconsistent point pairs. This resonant mechanism significantly improves the establishment of long-range architectural component correspondences without introducing artificial distance priors.

![](images/052b121b38b7ae0287b8eb8edde036fd6307a2deedc4ff7de86a2b0af596f0cd.jpg)  
Figure 5: Computation process of resonant distance bias.

## 3.2.3 Dynamic Gating Strategy

The directional branch and radial branch capture complementary geometric properties but exhibit distinct optimization characteristics. Directional consistency represents a globally stable structura prior throughout training, whereas radial relationships depend on the learning of nonlinear distance

functions, rendering them more sensitive to parameter initialization. To balance their optimization processes, PolarSym introduces a lightweight dynamic gating strategy that adaptively controls the contribution of each geometric branch.

Consequently, the final attention score is formulated as:

$$
\begin{array} { r } { { \pmb S } = S ^ { \mathrm { s e m } } + \alpha \Delta ^ { \mathrm { d i r } } + \beta B ^ { \mathrm { r a d } } } \end{array}\tag{6}
$$

where $S ^ { \mathrm { s e m } }$ denotes the original semantic attention, $\Delta ^ { \mathrm { d i r } }$ represents the directional enhancement term generated by the SF-RoPE branch, $B ^ { \mathrm { r a d } }$ corresponds to the radial distance deviation learned by the RDB branch, and $\alpha , \beta$ are the physically interpretable learnable gating coeficients that control the contributions of the two geometric branches.

To stabilize optimization, both gating coeficients are parameterized as: $\alpha = \operatorname { t a n h } ( g _ { \alpha } ) , \beta =$ tanh(g<sub>β</sub>), where $g _ { \alpha }$ and $g _ { \beta }$ are trainable parameters.

The hyperbolic tangent function constrains the coeficients within the interval (−1, 1), preventing excessive geometric responses during early optimization. Experiments initially adopted asymmetric initialization: the directional gate was set to a larger value $g _ { \alpha } ~ = ~ 2 . 0 , ~ \alpha _ { \mathrm { i n i t } } ~ = ~ \mathrm { t a n h } ( 2 . 0 ) ~ \approx ~ 0 . 9 6$ to ensure directional geometry participates in attention computation from the outset. The radial gate was initialized to $g _ { \beta } = 0 , \beta _ { \mathrm { i n i t } } = \operatorname { t a n h } ( 0 ) = 0$ , suppressing the radial branch during the initial phase, while its internal parameters were optimized via backpropagation. As training progressed, the network gradually increased $\beta$ to incorporate radial geometry only after establishing reliable semantic and directional representations.

## 4 Experiments

## 4.1 Experimental Setup

The PolarSym framework was evaluated on publicly available CAD planar diagram parsing benchmarks. In this work, four standard metrics for panoptic segmentation and semantic segmentation tasks were employed: Panoptic Quality (PQ), Recognition Quality (RQ), Segmentation Quality (SQ), and Mean Intersection over Union (mIoU). The core notations are defined as follows: TP denotes the number of instances where predicted results successfully match ground-truth labels (true positives); FP denotes the number of predicted instances without corresponding ground-truth label matches (false positives); FN denotes the number of ground-truth instances not predicted by the model (false negatives); IoU denotes the intersection-over-union (IoU) for matched pairs.

SQ quantifies the pixel-level segmentation accuracy of matched instances and is defined as the mean intersection-over-union (IoU) across all successfully matched instances, calculated as:

$$
\mathrm { S Q } = \frac { \sum _ { i \in \mathrm { T P } } \mathrm { I o U } _ { i } } { | \mathrm { T P } | }\tag{7}
$$

RQ measures the completeness of instance-level recognition and matching, equivalent to a weighted F1 score, efectively characterizing the model’s capability for target instance detection and recognition. Its calculation formula is:

$$
{ \mathrm { R Q } } = { \frac { \left| \mathrm { T P } \right| } { \left| \mathrm { T P } \right| + { \frac { 1 } { 2 } } | \mathrm { F P } | + { \frac { 1 } { 2 } } | \mathrm { F N } | } }\tag{8}
$$

PQ serves as a comprehensive evaluation metric for panoptic segmentation, integrating both pixellevel segmentation quality and instance-level recognition quality. It is derived from the product of SQ and RQ, with the overall calculation formula expressed as:

$$
\mathrm { P Q } = \mathrm { S Q } \times \mathrm { R Q } = { \frac { \sum _ { i \in \mathrm { T P } } \mathrm { I o U } _ { i } } { | \mathrm { T P } | + { \frac { 1 } { 2 } } | \mathrm { F P } | + { \frac { 1 } { 2 } } | \mathrm { F N } | } }\tag{9}
$$

mIoU is a classical evaluation metric for semantic segmentation, computed as the average intersectionover-union (IoU) across all classes to holistically assess the model’s global segmentation performance.

Given $N _ { c }$ as the total number of classes in the dataset and $\mathrm { I o U } _ { c }$ as the intersection-over-union for class $c ,$ the calculation formula is:

$$
\mathrm { I o U } _ { c } = \frac { \mathrm { T P } _ { c } } { \mathrm { T P } _ { c } + \mathrm { F P } _ { c } + \mathrm { F N } _ { c } } , \quad \mathrm { m I o U } = \frac { 1 } { N _ { c } } \sum _ { c = 1 } ^ { N _ { c } } \mathrm { I o U } _ { c }\tag{10}
$$

All experiments were conducted on two NVIDIA V100-32GB GPUs using identical training configurations. Following the constrained resource setup adopted in this work, the network was trained for 50 epochs using the AdamW optimizer.

## 4.2 Comparison with State-of-the-Art Methods

Table 1 compares PolarSym with representative planar graph parsing methods. Although PolarSym achieves competitive performance with a significantly smaller computational budget than previous state-of-the-art methods, it still demonstrates superior results. Specifically, PolarSym attains 87.15% PQ, significantly outperforming PanCADNet, GAT-CADNet, and SymPoint V1. Compared to the oficial report of SymPoint V2, our model achieves comparable performance in just 50 training epochs, whereas SymPoint V2 was trained for 250 epochs under conditions of 8×GPUs and batch size 16, while SymPoint V1 required up to 1,000 epochs. These results demonstrate that explicit geometric modeling significantly enhances optimization eficiency. PolarSym does not rely solely on prolonged optimization but enables the Transformer to establish reliable structural correspondences in early training stages, thereby improving data utilization eficiency within constrained computational resources.

To eliminate the influence of varying training budgets, we further compare PolarSym with the re-implemented SymPoint V2 benchmark under identical constrained settings. As shown in Table 2, PolarSym consistently outperforms the re-implemented benchmark across all evaluation metrics, with improvements of +1.73% in $\mathrm { P Q } , + 1 . 5 4 \%$ in $\mathrm { R Q } , + 0 . 3 0 \%$ in SQ, and +4.31% in mIoU.

The most significant improvement is observed in mIoU, indicating that the proposed geometryaware attention mechanism substantially enhances semantic consistency. Furthermore, the improvement in RQ demonstrates that explicit geometric reasoning facilitates the establishment of more reliable long-range correspondences, thereby improving instance recognition.

Overall, these results demonstrate that incorporating explicit directional and radial geometric priors into the Transformer attention mechanism can efectively improve planar graph parsing without increasing network complexity.

Table 1: Comparison between PolarSym and representative planar graph parsing methods.
<table><tr><td>Method</td><td>Backbone</td><td>Epochs</td><td>PQ (%)</td><td>RQ (%)</td><td>SQ (%)</td><td>mIoU</td></tr><tr><td>PanCADNet GAT-CADNet</td><td>CNN + GCN GAT (Graph At-</td><td></td><td>59.5 73.7</td><td>82.6</td><td>66.9</td><td></td></tr><tr><td></td><td>tention Network)</td><td></td><td></td><td>91.4</td><td>80.7</td><td></td></tr><tr><td>SymPoint V1</td><td>PointNet++</td><td>1000</td><td>83.3</td><td>91.4</td><td>91.1</td><td>69.7</td></tr><tr><td>SymPoint V2</td><td>PointT</td><td>250</td><td>90.1</td><td>96.3</td><td>93.6</td><td>74.0</td></tr><tr><td>PolarSym (Ours)</td><td>PointT</td><td>50</td><td>87.15</td><td>91.48</td><td>95.28</td><td>70.87</td></tr></table>

Table 2: Comparison between PolarSym and SymPoint V2 on the benchmark.
<table><tr><td>Method</td><td>GPU / Batch</td><td>PQ (%)</td><td>RQ (%)</td><td>SQ (%)</td><td>mIoU</td></tr><tr><td>SymPoint V2 (Reproduction)</td><td>2/8</td><td>85.42</td><td>89.92</td><td>94.98</td><td>66.56</td></tr><tr><td>PolarSym (Ours)</td><td>2/6</td><td>87.15</td><td>91.46</td><td>95.28</td><td>70.87</td></tr><tr><td>Relative Improvement</td><td></td><td>+1.73</td><td>+1.54</td><td>+0.30</td><td>+4.31</td></tr></table>

## 4.3 Ablation Experiments

To investigate the contributions of individual components, we progressively introduced the directional and radial branches into the baseline network.

As shown in Table 3, the baseline performance corresponds to the model without the PolarSym decoder module. Introducing the SF-RoPE module alone increased the PQ from 85.42% to 86.83% and improved the mIoU by 2.60%. This result demonstrates that explicit directional modeling effectively establishes global geometric consistency and provides reliable directional cues for attention computation.

Further introduction of the RDB module elevated the performance to 87.15% PQ and 70.87% mIoU. Although the absolute gain in PQ is relatively modest, the RDB consistently improved all evaluation metrics. This observation indicates that directional modeling and radial modeling provide complementary geometric information. The directional branch determines the global structural orientation, while the radial branch refines long-range geometric correspondences.

We further evaluated the robustness of the proposed geometric modeling strategy under diferent initialization settings. Table 4 compares PolarSym geometric modeling with a conventional combination of standard axial RoPE and linear distance bias.

When intentionally constructing weaker baselines via suboptimal initialization, the conventional geometric modeling achieved significant performance improvements: the baseline model’s PQ increased by 4.2 percentage points, and the final model’s PQ increased by 4.42 percentage points. Experimenta results demonstrate that explicit geometric modeling facilitates Transformer optimization. Nevertheless, the proposed PolarSym consistently achieves significantly higher final performance. This result confirms that independent modeling of direction and distance outperforms direct concatenation of conventional positional encoding with linear distance bias.

Table 3: Performance comparison of PolarSym components.
<table><tr><td>Setting</td><td>PQ (%)</td><td>RQ (%)</td><td>SQ (%)</td><td>mIoU</td></tr><tr><td>Baseline</td><td>85.420</td><td>89.927</td><td>94.988</td><td>66.562</td></tr><tr><td>+ SF-RoPE</td><td>86.825</td><td>91.208</td><td>95.195</td><td>69.159</td></tr><tr><td>+ RDB</td><td>87.155</td><td>91.468</td><td>95.284</td><td>70.870</td></tr></table>

Table 4: Performance comparison of PolarSym under diferent experimental settings.
<table><tr><td>Setting</td><td>Geometry Implementation</td><td>Baseline PQ</td><td>Final PQ</td><td>Relative Gain (∆)</td></tr><tr><td>Weak Baseline</td><td>Standard Axial + Linear Dist</td><td>81.019</td><td>83.57</td><td>+2.55%</td></tr><tr><td>Strong Baseline</td><td>PolarSym (SF-RoPE + RDB)</td><td>85.21</td><td>87.15</td><td>+1.94%</td></tr></table>

## 4.4 Convergence Analysis

Beyond final performance, PolarSym demonstrates significantly faster convergence during training.   
Figure 6 illustrates the PQ curves of the reproduced baseline and PolarSym.

In the first 20 epochs of training, PolarSym consistently outperforms the baseline in convergence speed. By the 20th epoch, PolarSym achieves 74.6% PQ, whereas the baseline reaches only 58.1% PQ. This acceleration primarily stems from explicit geometric modeling. Since the directional branch participates in attention computation from the outset, the network establishes global structural correspondences prior to learning fine-grained semantic representations. Subsequently, the radial branch progressively refines long-range geometric relationships, enabling stable optimization and improved convergence. The faster convergence rate further demonstrates that PolarSym enhances optimization eficiency rather than merely increasing model capacity.

![](images/4d5d7022992ff2b7b4a9651a7a58bc5edece6bbc7037b40faba84ea272e7d54c.jpg)  
Figure 6: PolarSym convergence process.

## 4.5 Qualitative Results

Figure 7 illustrates the qualitative comparison between PolarSym and the reproduced SymPoint V2 benchmark on several challenging planar examples. Compared to the benchmark, PolarSym generates more complete structural predictions while reducing false positives in complex architectural regions. Notably, the proposed method achieves more accurate predictions for spatially separated yet geometrically correlated symmetric walls, doors, and furniture.

The visual improvements align with the previously described quantitative analysis. By explicitly modeling directional consistency and radial correspondence, PolarSym establishes more reliable longrange attention mechanisms, resulting in more coherent architectural layouts and reduced structural inconsistencies.

![](images/1d09be9f9722db8f028b4c64ada7c6bde31860189d91316448fb11655f1b8f89.jpg)  
Figure 7: Qualitative comparison of PolarSym and SymPoint V2 benchmarks on several challenging planar graph examples.

## 5 Conclusion

This work addresses the issue of transformer attention mechanisms lacking explicit geometric modeling in CAD floor plan parsing by proposing PolarSym, a polar coordinate geometry attention framework. Unlike existing methods that uniformly encode spatial information into positional representations, PolarSym explicitly decouples architectural geometric relationships into two complementary factors—direction and distance—and models them separately using SF-RoPE and RDB, respectively. Additionally, a lightweight dynamic gating mechanism is introduced to enable collaborative modeling of semantic features and geometric priors while preserving the original transformer architecture unchanged.

Extensive experimental results demonstrate the efectiveness of the proposed method. Under a unified constrained training environment, PolarSym outperforms the reproduced SymPoint V2 baseline across multiple metrics including PQ, RQ, SQ, and mIoU. Furthermore, it exhibits faster convergence speed and a more stable optimization process. Ablation studies further confirm the complementary nature of direction modeling and distance modeling: the former establishes global directional consistency, while the latter optimizes long-range structural correspondences. Together, they enhance the transformer’s geometric perception capability for building symmetry structures. Due to its lightweight extension to attention computation, PolarSym can be seamlessly integrated into existing transformer architectures without requiring modifications to the backbone network.

Although PolarSym achieves promising performance in CAD floor plan parsing, its geometric modeling remains primarily constrained to 2D planar structures. Future work will explore more generalizable geometric attention mechanisms, including unified modeling for 3D building models, cross-modal building data, and more complex spatial topological relationships, to further advance the transformer’s geometric reasoning capability in building scene understanding.

## References

[CDAT20] Alexandre Carlier, Martin Danelljan, Alexandre Alahi, and Radu Timofte. Deepsvg: A hierarchical generative network for vector graphics animation. Advances in Neural Information Processing Systems, 33:16351–16361, 2020.

[CLC21] Sumit Chaturvedi, Michal Luk´aˇc, and Siddhartha Chaudhuri. Regroup: Recursive neural networks for hierarchical grouping of vector graphic primitives. arXiv preprint arXiv:2111.11759, 2021.

[DJL<sup>+</sup>24] Shuguang Dou, Xinyang Jiang, Lu Liu, Lu Ying, Caihua Shan, Yifei Shen, Xuanyi Dong, Yun Wang, Dongsheng Li, and Cairong Zhao. Hierarchically recognizing vector graphics and a new chart-based vector graphics dataset. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(12):7556–7573, 2024.

[FCWW22] Zhiwen Fan, Tianlong Chen, Peihao Wang, and Zhangyang Wang. Cadtransformer: Panoptic symbol spotting transformer for cad drawings. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10976–10986. IEEE, 2022.

[FZL<sup>+</sup>21] Zhiwen Fan, Lingjie Zhu, Honghua Li, Xiaohao Chen, Siyu Zhu, and Ping Tan. Floorplancad: A large-scale cad drawing dataset for panoptic symbol spotting. In 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pages 10108–10117. IEEE, 2021.

[JLS<sup>+</sup>21] Xinyang Jiang, Lu Liu, Caihua Shan, Yifei Shen, Xuanyi Dong, and Dongsheng Li. Recognizing vector graphics without rasterization. Advances in Neural Information Processing Systems, 34:24569–24580, 2021.

[LWF18] Chen Liu, Jiaye Wu, and Yasutaka Furukawa. Floornet: A unified framework for floorplan reconstruction from 3d scans. In European Conference on Computer Vision, pages 203– 219. Springer, 2018.

[LWKF17] Chen Liu, Jiajun Wu, Pushmeet Kohli, and Yasutaka Furukawa. Raster-to-vector: Revisiting floorplan transformation. In 2017 IEEE international conference on computer vision (ICCV), pages 2214–2222. IEEE, 2017.

[LYW<sup>+</sup>24] Wenlong Liu, Tianyu Yang, Yuhan Wang, Qizhi Yu, and Lei Zhang. Symbol as points: Panoptic symbol spotting via point-based representation. In International Conference on Learning Representations, volume 2024, pages 50797–50817, 2024.

[LYYZ24] Wenlong Liu, Tianyu Yang, Qizhi Yu, and Lei Zhang. Sympoint revolutionized: boosting panoptic symbol spotting with layer feature enhancement. arXiv preprint arXiv:2407.01928, 2024.

[RCBA19] Alireza Rezvanifar, Melissa Cote, and Alexandra Branzan Albu. Symbol spotting for architectural drawings: state-of-the-art and new industry-driven developments. IPSJ Transactions on Computer Vision and Applications, 11(1):2, 2019.

[RGLM21] Pradyumna Reddy, Michael Gharbi, Michal Lukac, and Niloy J Mitra. Im2vec: Synthesizing vector graphics without vector supervision. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 7342–7351, 2021.

[SAL<sup>+</sup>24] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.

[SJS<sup>+</sup>22] Ruoxi Shi, Xinyang Jiang, Caihua Shan, Yansen Wang, and Dongsheng Li. Rendnet: Unified 2d/3d recognizer with latent space rendering. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 5398–5407. IEEE, 2022.

[XGZ<sup>+</sup>26] Jici Xing, Guowei Gao, Tianyi Zeng, Jianga Shang, Yu Han, Zhenghua Tao, and Guoying Liu. Multimodal integration for advanced floor plan symbol spotting. Automation in Construction, 181:106659, 2026.

[YJPX23] Bingchen Yang, Haiyong Jiang, Hao Pan, and Jun Xiao. Vectorfloorseg: Two-stream graph attention network for vectorized roughcast floorplan segmentation. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 1358–1367. IEEE, 2023.

[YKH25] Songlin Yang, Jan Kautz, and Ali Hatamizadeh. Gated delta networks: Improving mamba2 with delta rule. In International Conference on Learning Representations, volume 2025, pages 29687–29707, 2025.

[YMZ<sup>+</sup>24] Fuyi Yang, Jiazuo Mu, Yanshun Zhang, Mingqian Zhang, Junxiong Zhang, Yongjian Luo, Lan Xu, Jingyi Yu, Yujiao Shi, and Yingliang Zhang. Cadspotting: Robust panoptic symbol spotting on large-scale cad drawings. arXiv preprint arXiv:2412.07377, 2024.

[ZLZ<sup>+</sup>22] Zhaohua Zheng, Jianfang Li, Lingjie Zhu, Honghua Li, Frank Petzold, and Ping Tan. Gat-cadnet: Graph attention network for panoptic symbol spotting in cad drawings. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 11737–11746. IEEE, 2022.