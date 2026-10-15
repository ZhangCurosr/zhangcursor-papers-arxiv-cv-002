# Q-MAGIC: Quad-only Mesh Autoregressive Generation via Singularity-Aware Circular Relative Offset Tokenization

Yuming Zhao<sup>1†</sup>, Zezeng Li<sup>2,†</sup>, Weimin Wang<sup>1‡</sup>, Xiang Wu<sup>1</sup>, Na Lei<sup>1,‡</sup>

<sup>1</sup>Dalian University of Technology, China <sup>2</sup>École Centrale de Lyon, France

## Abstract

Autoregressive models have proven effective for triangular mesh generation by transforming it into predicting the next face converting the non-differentiable connectivity prediction into a sequence of token predictions. However, generating quad-only meshes remains challenging because quad faces impose stricter topological constraints than triangles. The model must not only serialize quad faces into a valid token sequence, but also support unambiguous connectivity reconstruction during deserialization. To address this, we propose Q-MAGIC, a novel autoregressive modelfor quad mesh generation. Wefirst introduce a tailored serialization schemefor quad meshes, which requires at least two new verticesfor the nextface and is lessflexible than triangles due to strict wiring constraints. To meet these requirements and reduce serialization length, we propose a singularityaware serialization that prioritizes wiring from singular points, leading to more coherent traversals. Additionally, we propose a circular relative offset tokenization that encodes geometry mainly via local relative offsets. Built upon this representation, an autoregressive Transformer learns the underlying distribution of quad meshes via next-token prediction, enabling effective quad mesh generation. Extensive experiments demonstrate that Q-MAGIC produces more compact sequences while consistentl improving geometricfidelity and mesh quality compared to state-of-the-art methods.

• Computing methodologies → Shape modeling; Computer vision; Neural networks;

## 1. Introduction

Quad meshes are a widely used surface representation in production workflows such as animation, CAD, and physical simulation, where pure quadrilateral meshes are particularly favored for their compatibility with standard modeling pipelines and support for downstream applications, including computational mechanics and geometric analysis. Consequently, developing robust and practical methods for generating high-quality pure quad meshes remains an important problem in both academia and industry.

Most current methods generate quad meshes via a two-stage pipeline. These methods first create a triangle mesh and then apply quadrangulation or remeshing as post-processing. This intermediate triangle mesh can be produced either from continuous 3D representations [RMA<sup>∗</sup>24, MHS<sup>∗</sup>21, PFS<sup>∗</sup>19, WLL<sup>∗</sup>21, HTC<sup>∗</sup>24, XCG<sup>∗</sup>24] or from triangle-based autoregressive mesh generators [WZL<sup>∗</sup>25,CWL<sup>∗</sup>25]. However, this conversion adds extra stages and heuristics, and it does not reliably guarantee quadonly outputs, often requiring additional cleanup or refinement.

Inspired by triangular mesh generation, Liu et al. [LWG<sup>∗</sup>25] proposed a mixed triangle and quad face mesh generative model via a united hybrid polygon mesh serialization approach. While this serialization method is capable of handling both triangular and quadrilateral faces simultaneously, it suffers from two key limitations. First, it fails to account for the unique mesh wiring constraints and singular structures inherent to quadrilaterals. Second, the excessively redundant sequence length substantially increases the difficulty of model training and optimization.

Compared to triangular meshes, quad meshes are more imbalanced in structure. Most faces consist of regular valence-4 connectivity, yet the global layout is controlled by a small set of singular vertices and their neighborhoods. These singular structures steer the overall edge flow and indicate where regularity must break. Learning a quad mesh as a uniform sequence over-emphasizes the regular patterns and underfits these rare singularities. As a result, errors in early singularity decisions can propagate along loops, producing inconsistent layouts and reduced editability.

To address these issues, we propose Q-MAGIC, a quad-only autoregressive model that follows a layout-first design. We partition a mesh into patches and serialize them in a singularity-aware order, so topology-critical regions are generated early. Within each patch, our method predicts geometry using circular relative offsets. This representation reduces dynamic range and redundancy, improves robustness to local mesh-density variations, leading to more stable training and higher-quality quad generation. In summary, our contributions are as follows:

• We propose Q-MAGIC, to our knowledge, the first autoregressive Transformer framework for quad-only mesh generation.

• We propose a singularity-aware traversal to prioritize topologycritical wiring, shorten serialization sequences, and improve mesh validity.

• We introduce a circular relative offset tokenization to encode geometry and handle mesh-density variation robustly.

• Extensive experiments show that Q-MAGIC improves mesh validity and reconstruction quality over state-of-the-art methods.

## 2. Related Works

## 2.1. Autoregressive Model for Triangle or Mixed Mesh

Following MeshGPT [SAA<sup>∗</sup>24], recent autoregressive mesh generation methods have moved towards stronger conditional controllability and better scalability. Some methods introduce richer shape conditions or control signals to impose stronger constraints on the output without changing the autoregressive paradigm [CHH<sup>∗</sup>25, HRLL24]. Other methods address the bottleneck of training and inference caused by long sequences, proposing scalable network architecture and training designs for meshes with more patches and higher resolution [HRLL24, CCP<sup>∗</sup>24, WWZ<sup>∗</sup>25b].

Another important direction is mesh serialization, which directly affects sequence length, local connectivity modeling, and generation stability. EdgeRunner [TLH 25] borrows the idea of connectivity compression from EdgeBreaker [Ros99] to achieve lossless patch compression and reduce long-range dependencies. MeshAnythingV2 [CWL<sup>∗</sup>25] uses adjacent-face encoding to reduce vertex redundancy. OCTGPT [WWZ<sup>∗</sup>25a] uses a multi-scale structure for scalable 3D shape serialization. BPT [WZL 25] compresses meshes at both vertex and face levels via blocked and patchified representations to reduce sequence length. MeshRipple [LLG<sup>∗</sup>26] aligns the generation order with surface topology through frontier-aware traversal, and Mesh Silksong $[ \mathrm { S } \mathrm { Z } \bar { \mathrm { W } } ^ { * } 2 6 ]$ emphasizes topology-preserved serialization. FACE [WGL 26] represents each face as a compact token, while SATO [XQQ<sup>∗</sup>26] introduces strip-based tokenization for unified triangle and quadrilateral representation.

Deepmesh and Mesh-RFT [ZYW<sup>∗</sup>25, LXG<sup>∗</sup>25] further explore preference-based post-training (e.g., DPO [RSM<sup>∗</sup>23]) to improve the generation quality. Researchers are also exploring accelerated decoding and progressive level-of-detail generation to achieve a better balance between inference speed and output quality [SZT<sup>∗</sup>25, SLWR25, CQL<sup>∗</sup>25, ZSA<sup>∗</sup>25].

Recently, autoregressive modeling has been extended to quadrilateral meshes. QuadGPT $[ \mathrm { L W G } ^ { * } 2 5 ]$ and Mesh-Pro [ZLL<sup>∗</sup>26] present end-to-end quadrilateral dominant mesh generation frameworks that employ unified hybrid polygon serialization and rewardbased refinement strategies. QuadLink [ZZS 26] formulates quaddominant mesh generation as a point-relation learning problem, predicting vertices and face centroids and assembling polygonal faces through centroid-conditioned vertex links and a quadfirst strategy. In general, existing autoregressive methods are either dominated by triangular meshes or produce hybrid meshes, whereas native quad-only mesh generation remains underexplored.

## 2.2. Quad Mesh Generation

Quad mesh generation has long been studied in computer graphics, with traditional methods mainly formulated as remeshing, layout extraction or field-aligned optimization. For example, Marinov and Kobbelt [MK04] proposed anisotropic quad-dominant remeshing, while Pietroni et al. $\boldsymbol { [ \mathrm { P N A } ^ { * } 2 1 ] }$ extracted field-coherent quad layouts from cross fields. Although these methods can produce highquality quad structures, they typically rely on carefully constructed inputs, such as clean surfaces, reliable directional fields, or wellinitialized layouts. Moreover, since quad connectivity is mainly obtained through hand-designed remeshing or layout extraction procedures, these methods are less flexible in modeling diverse quadonly meshes in a data-driven generative setting.

Beyond traditional optimization-based pipelines, recent learning-based methods have also explored quad mesh generation, although most of them still rely on intermediate fields, templates, or local assembly procedures. Many works first predict a local directional frame field for quadrangulation and then generate quad meshes via field-aligned parameterization or subsequent remeshing. This pipeline imposes strong requirements on the predicted field, which should be globally smooth for stable parameterization, while still aligning with geometric features to support reliable quad extraction [YLGL25, DWX<sup>∗</sup>25b, DWX<sup>∗</sup>25a, LLL<sup>∗</sup>25b, QQX<sup>∗</sup>25, CLZ<sup>∗</sup>26, DSS<sup>∗</sup>26]. Another line of work builds on PDE-constrained formulations, e.g., incorporating PDE residuals into the learning objective to learn mappings from a parametric domain to the physical domain for structured mesh generation [CLW<sup>∗</sup>22, PCL25]. MeshONet $[ \mathrm { X C P ^ { * } } 2 6 ]$ further formulates structured mesh generation as an operator-learning problem for mapping computational domains to physical domains. Other approaches use flow or ODE-based template deformations to fit geometry while largely preserving topology [GC20, LSCdG<sup>∗</sup>21, ZLW24]. Meanwhile, some approaches cast meshing as a sequential decision-making problem and use reinforcement learning to optimize element placement or advancing-front updates, with rewards designed around element quality and topological preferences [PHCZ22,TQHZ23,LFWN20]. Overall, these methods tend to work best in simple planar geometries or topologies with clean inputs, and their applicability remains limited for complex shapes.

In addition, researchers have explored directly generating quad meshes from point clouds. Point2Quad $[ \mathrm { L Q W } ^ { * } 2 5 ]$ directly produces quad-only surface meshes by predicting and selecting local quadrilateral faces for mesh assembly. However, it follows a candidate-face prediction paradigm. Therefore, the recoverable quad connectivity is bounded by whether valid faces are included in this candidate set. Our method instead represents quad meshes as autoregressive sequences, where quad faces are generated through the learned sequence and the corresponding de-serialization rule, reducing the reliance on pre-enumerated local face candidates.

![](images/2199aec57dcb9bff851d8dbd064c44da4cad4b73dabfabe0937b3757dbf544db.jpg)  
Figure 1: Overview of the Q-MAGIC. We adopt a singularity-aware serialization strategy that prioritizes critical singularities in the sequence. In our token representation, geometric information is mainly encoded by circular relative offsets, and block tokens are inserted only when necessary to re-anchor the spatial location, enabling a compact representation. During inference, Q-MAGIC conditions on point clouds and autoregressively generates quad meshes.

## 3. Method

As shown in Fig. 1, the proposed Q-MAGIC centers on a compact serialization for quad-only meshes, which transforms mesh geometry and connectivity into a token sequence for autoregressive generation. While Q-MAGIC adopts the block–offset indexing introduced by BPT [WZL<sup>∗</sup>25], BPT is designed for triangular meshes, whose local serialization exploits the fact that every pair of vertices within a triangle is directly connected by an edge. This property does not hold for quadrilateral faces, where diagonally opposite vertices are not edge-connected. Consequently, directly extending BPT-style traversal to quad meshes becomes insufficient for preserving valid connectivity, particularly around singular regions with non-consecutive local face configurations. The serialization follows a singularity-aware traversal and uses circular relative offset tokens within local patches to produce shorter sequences while retaining topology-critical structures.

## 3.1. Naive Quad Mesh Serialization

We formulate the quad mesh generation task as a sequence prediction problem, transforming a mesh M to a one-dimensional sequence of discrete tokens. Following triangular mesh serialization, a naive quad mesh serialization represents a mesh as an ordered collection of faces together with their vertex coordinates. Let $\mathcal { M } = \{ f _ { i } \} _ { i = 1 } ^ { N }$ denote a mesh with N quadrilateral faces, where the ith face is $f _ { i } = ( \nu _ { i 1 } , \nu _ { i 2 } , \nu _ { i 3 } , \nu _ { i 4 } )$ and each vertex $\nu _ { i j } = ( x _ { i j } , y _ { i j } , z _ { i j } ) \in$ $\mathbb { R } ^ { 3 }$ . To make the continuous geometry amenable to autoregressive modeling, all vertex coordinates are first rescaled to the fixed range $[ - 1 , 1 ]$ , denoted by $\hat { \nu } _ { i j } = ( \hat { x } _ { i j } , \hat { y } _ { i j } , \hat { z } _ { i j } )$ , and each axis is uniformly quantized with resolution r. Specifically, the quantized coordinate is computed by

$$
\tilde { x } _ { i j } = \left\lfloor \frac { \hat { x } _ { i j } + 1 } { 2 } ( r - 1 ) + \frac { 1 } { 2 } \right\rfloor \in \{ 0 , 1 , . . . , r - 1 \} ,\tag{1}
$$

and the same operation is applied to $\hat { y } _ { i j }$ and $\hat { z } _ { i j }$ . This yields quantized vertices $\tilde { \nu } _ { i j } = ( \tilde { x } _ { i j } , \tilde { y } _ { i j } , \tilde { z } _ { i j } )$ .

All integer vertices are then sorted in the lexicographic order of Z, Y, X, and assigned global indices accordingly, and quadrilateral faces are sorted by their vertex indices to obtain a globally ordered face sequence $F = ( f _ { 1 } , f _ { 2 } , \dots , f _ { N } )$

Based on this ordering, faces are traversed in sequence. Within each face, the four vertices are visited in a fixed order. The three quantized coordinates of every vertex are listed consecutively, resulting in a length-12N discrete token sequence

$$
S ( \mathcal { M } ) = ( \tilde { x } _ { 1 1 } , \tilde { y } _ { 1 1 } , \tilde { z } _ { 1 1 } , \tilde { x } _ { 1 2 } , \tilde { y } _ { 1 2 } , \tilde { z } _ { 1 2 } , \dots , \tilde { x } _ { N 4 } , \tilde { y } _ { N 4 } , \tilde { z } _ { N 4 } ) .
$$

For a quadrilateral mesh with N faces, this naive scheme yields a sequence of length $L _ { \mathrm { n a i v e } } = 1 2 N$ over a vocabulary size of r. This highly redundant and lengthy sequence substantially elevates the complexity of model learning tasks.

## 3.2. Singularity-aware Serialization

Block–offset indexing. Although the naive quad mesh serialization described above is simple and directly compatible with autoregressive models, it treats each quantized coordinate as an independent token and repeats shared vertices for every incident face. Existing methods, such as QuadGPT [LWG<sup>∗</sup>25], adopt similar coordinatebased tokenizations, which can result in long and redundant sequences and a vocabulary size that grows cubically with the coordinate resolution. To address these issues, we follow BPT [WZL 25] and convert quantized coordinates into a block–offset representation, as shown in Fig. 2.

Given the vertex sequence S, we discretize each 3D position into a block-offset index. Along each coordinate axis, we partition the coordinate range into B blocks and subdivide each block into O offsets. So the resolution per axis is B · O. For a quantized vertex $\tilde { \nu } _ { i j } = ( \tilde { x } _ { i j } , \tilde { y } _ { i j } , \tilde { z } _ { i j } )$ with each integer coordinate in $\{ 0 , \ldots , B \cdot O - 1 \}$ , the block-wise indexing $\left( b _ { i } , o _ { i } \right)$ is computed as

![](images/c1914850e82e7a9b4b5272a2f04ec61ea07e30e6322241846dd6e5d1afa25126.jpg)  
Figure 2: Overview of circular relative offset tokenization. Left: each vertex is represented by a block–offset pair (B, O). Right: For intra-block moves, we emit only ∆o<sub>n</sub> (wrapping to the bounded interval when needed); when crossing a block boundary, we emit the new block index B and then continue with relative offsets. Bottom: the resulting token sequence with BOS/EOS, consisting of block tokens and circular relative offset tokens.

Algorithm 1 Singularity-aware Quad Mesh Tokenization   
1: procedure Tokenization(M)   
2: P ← ∅ ▷ List of patches   
3: while $\mathcal { M } \neq \emptyset$ do   
4: v<sub>c</sub> ← SELECTCENTER $( \mathcal { V } _ { \mathrm { i r r } } \succ \mathcal { V } _ { \mathrm { r e g } } , \mathcal { M } )$   
5: P ← AGGREGATE(v )   
6: P<sub>c</sub> ← BLOCKWISE-INDEX(P<sub>c</sub>) ▷ Eq. 2   
7: P<sub>c</sub> ← RELATIVE-OFFSET(P<sub>c</sub>) ▷ Eq. 3   
8: P<sub>c</sub> ← DUAL-BLOCK(P<sub>c</sub>)   
9: $\mathcal { P }  \mathcal { P } \cup { P } _ { c } .$ $\mathcal { M }  \mathcal { M } - P _ { c }$   
10: end while   
11: return $\mathcal { P }$

$$
\begin{array} { r l } & { b _ { i } = ( \tilde { x } _ { i } \mid O ) \cdot B ^ { 2 } + ( \tilde { y } _ { i } \mid O ) \cdot B + ( \tilde { z } _ { i } \mid O ) , } \\ & { o _ { i } = ( \tilde { x } _ { i } \% O ) \cdot O ^ { 2 } + ( \tilde { y } _ { i } \% O ) \cdot O + ( \tilde { z } _ { i } \% O ) , } \end{array}\tag{2}
$$

where | denotes integer division and % denotes the modulo operation. Here $b _ { i } \in [ 0 , B ^ { 3 } - 1 ]$ is the block index and $o _ { i } \in [ 0 , O ^ { 3 } - 1 ]$ is the absolute offset inside that block. Together, the block-offset pair $( b _ { i } , o _ { i } )$ uniquely encodes a quantized 3D coordinate with per-axis resolution $r = B \cdot O$ . The overall tokenization process is summarized in Algorithm 1.

Singularity-aware patch construction. Local topological complexity can be characterized by the valence of a vertex. We refer to vertices whose valence is not equal to 4 as singular vertices, and denote the set of such vertices as $V _ { \mathrm { i r r } }$ . During traversal, patch centers are first selected from $V _ { \mathrm { i r r } }$ until all singular vertices are covered.

After the center vertex c is determined, we collect all quadrilateral faces incident to c and aggregate them into a local patch $\mathcal { P } ( c )$ Each face in the patch is written as

$$
f _ { i } = ( c , u _ { i } , \nu _ { i } , w _ { i } )
$$

where $u _ { i }$ and $w _ { i }$ are the two vertices connected to c by real mesh edges, and $\nu _ { i }$ is the vertex diagonal to c. In the traversal sequence of $\mathcal { P } ( c )$ , the vertex immediately following the center c, which we call the starting vertex, is selected only from $\{ u _ { i } \} \cup \{ w _ { i } \}$

Fig. 3 illustrates three representative local quad-face layouts and their corresponding serialized vertex sequences. Case (a) shows a regular four-valent configuration, where the incident quad faces form a consecutive layout around the center vertex. In this case, the BPT-style local traversal can be naturally extended to quad meshes by recording the non-center vertices in order.

![](images/0190bd3e38dccb6a76c38dfb80f67701342d74b0129b764120bb354514578ec8.jpg)  
Figure 3: De-serialization in Q-MAGIC. (a) Consecutive records recover the shared vertex from the last vertex of the previous face. (b,c) Directly connecting adjacent tokens for non-consecutive records may generate invalid red dashed quads. We resolve this issue via a novel de-serialization scheme. Quads are reconstructed from the recorded non-center vertices under the face of the current local patch, ensuring consistent local topology.

However, cases (b) and (c) show two typical singular configurations with non-consecutive local layouts. The gray faces indicate quad faces that have been assigned to other center vertices and should not be serialized under the current center vertex. After excluding these faces, the remaining incident faces around the current center vertex no longer form a simple consecutive sequence. In particular, the last vertex of the previous face may not coincide with the first vertex of the next face. Therefore, directly applying the triangle-based BPT serialization rule to quad meshes is insufficient.

To handle such cases, we first identify all vertices that share an edge with the center vertex in the current local patch. Among these adjacent vertices, we select the first vertex encountered along the specified traversal direction as the starting vertex. From this vertex, we traverse the valid quad faces around the center vertex clockwise. For consecutive faces, if the first non-center vertex of the current face matches the last non-center vertex of the previous face, only the remaining two non-center vertices are appended; otherwise, a new local record is started with all three non-center vertices of the current face.

```perl
Algorithm 2 Quad Mesh De-tokenization
1: procedure De-serialization(S)
2: M ← ∅
3: {S<sub>c</sub>} ← SPLIT(s, sp)
4: while $S _ { c } \neq \emptyset$ do
5: $( \nu _ { c } , P _ { c } ) \gets \mathrm { P A R S E } ( S _ { c } )$
6: $P _ { c } \gets \mathrm { I N V D U A L } ( P _ { c } )$
7: P<sub>c</sub> ← RECOVERO(P<sub>c</sub>) ▷ Eq. 4
8: P<sub>c</sub> ← INVINDEX(P<sub>c</sub>) ▷ Eq. 2
9: $\mathcal { M }  \mathcal { M } \cup \mathrm { R E B U I L D Q U A D } ( \nu _ { c } , P _ { c } )$
10: end while
11: return $\mathcal { M }$
```

Similarly, when deserializing consecutive records, the shared vertex is recovered from the last vertex of the previous face. For a non-consecutive record, a new quad face should be initialized from its three recorded non-center vertices. If one directly follows the original BPT decoding rule and simply connects locally adjacent tokens, the decoder may incorrectly reconstruct the red dashed quads shown in Fig. 3 (b) and (c). These quads should not exist under the current center vertex. Therefore, our quad-aware de-serialization follows the inverse of the serialization rule while respecting the face ownership of the current local patch. The corresponding de-tokenization procedure is summarized in Algorithm 2.

## 3.3. Circular Relative Offset Tokenization

As illustrated in Fig. 2, we encode offsets as relative moves on a discrete circle, mapping raw differences into a bounded interval.

We reparameterize the offset sequence by replacing absolute offsets $\left\{ o _ { n } \right\}$ with wrapped differences between consecutive offsets. Let the offset space size be $| \Omega | = O ^ { 3 }$ and define

$$
\Delta o _ { n } = o _ { n } - o _ { n - 1 } , \qquad R = \bigl \lfloor \Omega / 2 \bigr \rfloor .
$$

The differences $\Delta o _ { n }$ are folded into a symmetric interval:

$$
\Delta o _ { n } ^ { \prime } = \left\{ \begin{array} { l l } { \Delta o _ { n } - \Omega , } & { \mathrm { i f } \Delta o _ { n } > R , } \\ { \Delta o _ { n } + \Omega , } & { \mathrm { i f } \Delta o _ { n } < - R , } \\ { \Delta o _ { n } , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{3}
$$

ensuring that $\Delta o _ { n } ^ { \prime }$ falls within the range $[ - R , R ]$ . The absolute offsets can be recovered recursively as

$$
o _ { n } = o _ { n - 1 } + \Delta o _ { n } ^ { \prime } { \pmod { \Omega } } .\tag{4}
$$

Equivalently, for any reference index $i \leq n ,$

$$
o _ { n } = o _ { i } + \sum _ { k = i + 1 } ^ { n } \Delta o _ { k } ^ { \prime } \quad ( \mathrm { m o d } \ \Omega ) .\tag{5}
$$

This circular relative offset representation has two immediate properties. Firstly, under the relative encoding, each token only needs to specify a local displacement $\Delta o _ { n } ^ { \prime }$ within the bounded interval $[ - R , R ]$ , rather than selecting an absolute index from the entire offset space $\{ 0 , \ldots , \Omega - 1 \}$ . Secondly, under this circular relative offset parametrization, the geometry of an intra-block vertex sequence is mainly determined by the sequence of relative offsets $\Delta o _ { n } ^ { \prime }$ , and is independent of the specific value of the initial offset. If the starting offset is changed, all subsequent vertices are translated consistently, while their relative configuration remains unchanged

Since consecutive vertices often fall into the same block, we adopt a run-based circular relative offset encoding that keeps an explicit block index only when necessary. As illustrated in Fig. 2, the encoding covers three typical cases.

Intra-block and representable offset. When two consecutive vertices $\nu _ { n }$ and $\nu _ { n - 1 }$ lie in the same block with $\left| \Delta o _ { n } \right| \leq R ,$ we compute $\Delta o _ { n } = o _ { n } - o _ { n - 1 }$ and append it directly as a circular relative offset token, without emitting a block token or applying circular wrapping. In the Intra-block example of Fig. 2 (c), vertices 1 and 2 stay in the same block and the displacement falls within the preset radius. Thus, we directly append the circular relative offset token $\Delta o _ { 2 }$ to the sequence.

Intra-block but out-of-range. Even when $b _ { n } = b _ { n - 1 }$ holds, the raw intra-block displacement $\Delta o _ { n } = o _ { n } - o _ { n - 1 }$ can still exceed the preset radius R. As shown by vertices 3 and 4 in Fig. 2 (c), the two vertices lie in the same block, yet their offset difference is larger than R. We therefore apply the circular mapping to wrap this difference back into the representable range of radius R, obtaining a bounded displacement $\Delta o _ { 4 } ^ { \prime } ,$ and append the resulting circular relative offset token to the sequence.

Cross-block. When the traversal crosses a block boundary, an explicit block index is required to re-anchor the location. In the Cross Block example of Fig. 2 (c), vertices 2 and 3 fall into different blocks $( b _ { 3 } \neq b _ { 2 } )$ . Therefore, we append the block index $b _ { 3 }$ together with its bounded offset $\Delta o _ { 3 } ^ { \prime }$ to the sequence, since the block index cannot be omitted across blocks.

Formally, given the original block–offset representation $( b _ { i } , o _ { i } , b _ { i + 1 } , o _ { i + 1 } , \dots , b _ { N } , o _ { N } )$ , we rewrite it as

$$
\begin{array} { r } { S ( \mathcal { M } ) = ( b _ { i } , o _ { i } , \tau _ { i + 1 } , \tau _ { i + 2 } , \ldots , \tau _ { N } ) , } \end{array}\tag{6}
$$

where each emitted token $\tau _ { k }$ follows a single rule:

$$
\begin{array} { r } { \bar { \tau } _ { k } = \left\{ \begin{array} { l l } { \Delta o _ { k } ^ { \prime } , } & { \mathrm { i f } \ b _ { k } = b _ { k - 1 } } \\ { ( b _ { k } , \Delta o _ { k } ^ { \prime } ) , } & { \mathrm { o t h e r w i s e } . } \end{array} \right. } \end{array}\tag{7}
$$

As shown in Fig. 2 (e), we encode the fine-grained location using circular relative offset tokens. When the relative offset exceeds the radius R, we wrap it back to $[ - R , R ]$ on the discrete circle. Only when the traversal enters a new block do we insert an explicit block token $b _ { k }$ to re-anchor the coarse location.

To quantify the compactness of our serialization scheme, we calculate the average sequence length produced by different serialization schemes. We apply the serialization scheme of each method to the same set of meshes and report the resulting mean sequence length. Our serialization produces the shortest token sequence on average among all compared methods, as shown in Table 1. We further provide an analysis of the sequence composition in the appendix, which helps explain the observed length differences.

## 4. Experiments

In this section, we conduct comprehensive experiments to evaluate Q-MAGIC for quad-mesh generation. Our evaluation emphasizes (i) geometric fidelity and quad-mesh quality and (ii)

![](images/d9d76878f72258f3952ea5804a2c092b4e9a1bec1f91f2e3c8ba9e7fdc27688e.jpg)  
Figure 4: Visualization results and close-up views of generated quad meshes. Our method can better preserve continuous boundary loops and structured quad layouts in detail-sensitive regions, producing more regular and stable quad meshes.

Table 1: Average sequence length comparison. QGPT, MAV2, EG, and DM denote QuadGPT, MeshAnythingV2, EdgeRunner and DeepMesh. Our serialization yields the lowest token length on the same mesh set.
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>QGPT</td><td rowspan=1 colspan=1>MAV2</td><td rowspan=1 colspan=1>EG</td><td rowspan=1 colspan=1>DM</td><td rowspan=1 colspan=1>BPT</td><td rowspan=1 colspan=1>Ours</td></tr><tr><td rowspan=1 colspan=1>Avg. Len.</td><td rowspan=1 colspan=1>43106</td><td rowspan=1 colspan=1>29000</td><td rowspan=1 colspan=1>28497</td><td rowspan=1 colspan=1>20675</td><td rowspan=1 colspan=1>14104</td><td rowspan=1 colspan=1>12655</td></tr><tr><td rowspan=1 colspan=1>Comp. Ratio</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0.67</td><td rowspan=1 colspan=1>0.66</td><td rowspan=1 colspan=1>0.48</td><td rowspan=1 colspan=1>0.33</td><td rowspan=1 colspan=1>0.29</td></tr></table>

topological correctness. We compare Q-MAGIC against strong baselines, including BPT [WZL<sup>∗</sup>25], TreeMeshGPT [LLL25a], QuadGPT [LWG 25], and Point2Quad [LQW 25], and further validate the contributions of key components via ablation studies. Specifically, Sec. 4.1 introduces the experimental settings, including the datasets, baselines, metrics, and implementation details. Sec. 4.2 presents the quantitative results for comprehensive comparison, followed by Sec. 4.3, which provides qualitative results and visual analyses. Finally, we perform ablation studies in Sec. 4.4 to assess each core component.

## 4.1. Experiment Settings

Datasets. We construct our experimental dataset with the obj files in Thing10K dataset [PNA 21] and apply a fixed, automated filtering pipeline to ensure mesh quality and reliable evaluation. We retain only meshes that satisfy the following criteria: all-quadrilateral faces, valid vertex coordinates, no degenerate faces, and no nonmanifold edges. Since some meshes have excessively high resolution and an excessive number of faces, we retopologize and simplify them using Quad Remesher [Max24], limiting the face count to under 4,700 to guarantee stable training and manageable computational overhead. After filtering and simplification, we obtain 2,100 quad meshes. We randomly split the dataset into 80% for training and 20% for testing. To improve robustness to object orientation, we apply rotation augmentation to the training data.

Baselines. We compare Q-MAGIC with two types of methods. The first type consists of autoregressive triangle or hybrid mesh generators, including BPT [WZL<sup>∗</sup>25], TreeMeshGPT [LLL25a], and QuadGPT [LWG<sup>∗</sup>25]. For a fair evaluation, we convert the outputs of the autoregressive baselines into quad meshes using the Blender Quad Remesher [Max24], an automatic remeshing tool that converts an input surface into a quad-based mesh, and we apply the same post-processing settings across methods. During this quad remeshing process, the algorithm may automatically fill small holes, which can lead to the presence of a limited number of triangular faces in the vicinity of the filled regions. The second type is a direct quad-only reconstruction method, Point2Quad [LQW<sup>∗</sup>25]. All point cloud conditioned methods take 4,096 uniformly sampled surface points as input. As QuadGPT does not provide a pretrained model, we use the official API (https://3d.hunyuan.tencent.com/) to produce its results. The generation process can be viewed as resembling a topology-optimization-style pipeline.

Metrics. We assess geometric fidelity using Chamfer distance (CD) and Hausdorff distance (HD) computed on uniformly sampled 1024 points from the generated and ground-truth meshes, where lower is better. We evaluate topological validity using Watertightness and Manifoldness, where higher is better. We further report the mesh-averaged per-face Scaled Jacobian as a measure of quad element quality, where higher is better.

Table 2: Quantitative comparison. ↑<sup>a</sup> and ↓<sup>a</sup> denote that larger and smaller values are better, respectively. Here, a and b represent the upper and lower bounds.SJ, WT, MF, CD, HD, SR and US denote Scaled Jacobian, Watertightness, Manifoldness, Chamfer distance, Hausdorff distance, singularity rate and user study.
<table><tr><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>Point2Quad</td><td rowspan=1 colspan=1>TMGPT</td><td rowspan=1 colspan=1>QuadGPT</td><td rowspan=1 colspan=1>BPT</td><td rowspan=1 colspan=1>Ours</td></tr><tr><td rowspan=1 colspan=1> $\overline { { \mathbf { S J } \uparrow _ { - 1 } ^ { 1 } } }$ </td><td rowspan=1 colspan=1>0.9726</td><td rowspan=1 colspan=1>0.8879</td><td rowspan=1 colspan=1>0.9727</td><td rowspan=1 colspan=1>0.9762</td><td rowspan=2 colspan=1>0.98290.9996</td></tr><tr><td rowspan=1 colspan=1> $\mathbf { W T } \uparrow _ { 0 } ^ { 1 }$ </td><td rowspan=1 colspan=1>0.9995</td><td rowspan=1 colspan=1>0.8921</td><td rowspan=1 colspan=1>0.9836</td><td rowspan=1 colspan=1>0.9873</td></tr><tr><td rowspan=1 colspan=1>MF↑0</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0.9989</td><td rowspan=1 colspan=1>0.9998</td><td rowspan=1 colspan=1>0.9998</td><td rowspan=3 colspan=1>0.99990.06840.1058</td></tr><tr><td rowspan=1 colspan=1> $\mathrm { C D } \downarrow _ { 0 } ^ { \infty }$ </td><td rowspan=1 colspan=1>0.0717</td><td rowspan=1 colspan=1>0.1704</td><td rowspan=1 colspan=1>0.0854</td><td rowspan=1 colspan=1>0.0719</td></tr><tr><td rowspan=1 colspan=1> $\mathrm { H D } \downarrow _ { 0 } ^ { \infty }$ </td><td rowspan=1 colspan=1>0.1069</td><td rowspan=1 colspan=1>0.3255</td><td rowspan=1 colspan=1>0.1374</td><td rowspan=1 colspan=1>0.1299</td></tr><tr><td rowspan=1 colspan=1>SR↓0</td><td rowspan=1 colspan=1>0.3050</td><td rowspan=1 colspan=1>0.8715</td><td rowspan=1 colspan=1>0.2164</td><td rowspan=1 colspan=1>0.4000</td><td rowspan=1 colspan=1>0.2310</td></tr><tr><td rowspan=1 colspan=1> $\mathrm { U S } \uparrow _ { 0 } ^ { 1 0 }$ </td><td rowspan=1 colspan=1>6.95</td><td rowspan=1 colspan=1>5.88</td><td rowspan=1 colspan=1>7.48</td><td rowspan=1 colspan=1>7.88</td><td rowspan=1 colspan=1>8.84</td></tr></table>

Implementation Details. Q-MAGIC is a compact autoregressive Transformer with 8 decoder layers and a hidden size of 1024. For mesh tokenization, we set the block size to B = 8 and the offset size to $O = 1 6 ,$ resulting in a per-axis spatial resolution of $B \cdot O = 1 2 8$ . The maximum sequence length is about 20K tokens.

For point-cloud conditioning, we employ a frozen pretrained Michelangelo Encoder to extract 768-dimensional point-cloud features. The extracted features are projected into the 1024- dimensional latent space of the Transformer and incorporated into the autoregressive decoder through cross-attention.

For remeshing, we convert the triangular meshes generated by BPT into quad meshes using Blender Quad Remesher. We set the target face count to 4,500 and the adaptive size to 30%, allowing the remesher to adjust the local face density according to the geometric characteristics of each mesh.

We first sample 16,384 surface points per shape and randomly subsample 4,096 points during training. All experiments are trained on a single NVIDIA A100 GPU for approximately 7 days. The model is trained using the standard next-token cross-entropy loss. The model is optimized with AdamW using $\beta _ { 1 } = 0 . 9$ and $\beta _ { 2 } =$ 0.95, with a base learning rate of $5 \times 1 0 ^ { - 5 }$ and a batch size of 4. We adopt a LambdaLR schedule with a linear warmup for the first 10% of training steps, followed by a cosine decay to a minimum learning rate ratio of 0.5. During inference, we autoregressively generate mesh tokens using multinomial sampling with top-k sampling (k = 50) and a temperature of 0.7.

## 4.2. Quantitative Results

Table 2 reports the quantitative comparison between Q-MAGIC and Point2Quad, TreeMeshGPT, QuadGPT, and BPT. Overall, Q-MAGIC achieves the best results in geometric fidelity, as measured by CD and HD, and element quality, as reflected by the Scaled Jacobian, while preserving high mesh validity in terms of Watertightness and Manifoldness. These results demonstrate that Q-MAGIC produces more regular quad meshes with more accurate surface fitting, without sacrificing closedness or manifold consistency.

Geometric Fidelity. We evaluate geometric fidelity using CD and HD, where lower values are better. Q-MAGIC achieves the lowest errors on both metrics, showing that direct quad mesh generation preserves geometric accuracy. Although some baselines obtain competitive results by first generating triangle meshes and then remeshing them into quads, their final quality remains limited by the intermediate triangle representation. Errors in the generated triangle mesh can propagate through remeshing, as observed in TreeMeshGPT. By directly predicting quad structures, Q-MAGIC avoids this propagation and improves geometric fidelity.

Mesh Validity. Watertightness and manifoldness are used to assess the topological validity of generated meshes, where higher values indicate more closed and manifold structures. As shown in Table 2, Q-MAGIC achieves near-perfect scores on both metrics, confirming that direct quad generation preserves strong structural validity. Compared with Point2Quad, Q-MAGIC obtains better geometric fidelity and element quality while maintaining comparable watertightness and manifoldness. This suggests that Q-MAGIC achieves a more favorable balance between surface accuracy, mesh regularity, and topological consistency.

Element Quality. We evaluate element quality using the Scaled Jacobian, where higher values indicate better-shaped elements. This metric reflects the distortion and regularity of quad faces. Q-MAGIC achieves the highest score, suggesting that its generated quads are closer to ideal elements with lower distortion. This improvement is important for downstream applications such as simulation and geometric processing, since element quality directly affects both practical usability and numerical stability.

User Study. We conducted a user study to evaluate perceptual mesh quality. For each test case, the outputs of all compared methods were presented in a randomized order with method names hidden, forming a blind evaluation protocol. Testers rated the outputs of all methods on a 0–10 scale, where higher scores indicate better visual quality. Considering the cost of participant involvement, we performed the study on a limited set of ten cases with 50 testers. Although the number of cases is limited, the observed preferences provide supporting evidence for the effectiveness of Q-MAGIC. As shown in Table 2, our method achieves the highest mean score, indicating a perceptual preference for our results. This is consistent with the qualitative comparisons, where Q-MAGIC produces more continuous boundary loops, fewer unnecessary openings and holes, and more regular quad layouts with coherent local structures in detail-sensitive regions. These advantages lead to higher subjective ratings.

## 4.3. Qualitative Results

Fig. 4 compares Q-MAGIC with Point2Quad, BPT, TreeMeshGPT, and QuadGPT, with the ground-truth quad meshes shown in the first column. For each example, we zoom in on representative regions to examine mesh structure differences at a fine scale. Near boundary openings, Q-MAGIC consistently forms continuous boundary loops and stable quad layouts, with smoother variations in element size and orientation. In contrast, some baselines exhibit broken edge-loop continuity, locally distorted elements, or uneven layouts in the same regions. On large curved surfaces and transition areas, Q-MAGIC produces a more coherent quad flow, with reduced local shear distortion and elements closer to ideal shapes. In the enlarged region of the fourth row, our method avoids unnecessary boundary singularities and yields a more regular local valence pattern and connectivity structure. Overall, our method preserves both geometric adherence and structured quad topology in detail-sensitive regions, resulting in more regular and stable quad meshes.

Table 3: Ablation study on patch serialization policies under circular relative offset tokenization (Random vs. singularity-aware).
<table><tr><td>Metrics</td><td>Random</td><td>Singularity-aware</td></tr><tr><td>Scaled Jacobian  $\uparrow _ { - 1 } ^ { 1 }$ </td><td>0.8862</td><td>0.9829</td></tr><tr><td>Watertightness  $\uparrow _ { 0 } ^ { 1 }$ </td><td>0.9429</td><td>0.9996</td></tr><tr><td>Manifoldness  $\uparrow _ { 0 } ^ { 1 }$ </td><td>0.9986</td><td>0.9999</td></tr><tr><td>Chamfer distance  $\downarrow _ { 0 } ^ { \infty }$ </td><td>0.2202</td><td>0.0684</td></tr><tr><td>Hausdorff distance  $\downarrow _ { 0 } ^ { \infty }$ </td><td>0.3906</td><td>0.1058</td></tr></table>

Table 4: Ablation study on serialization scheme under the default singularity-aware traversal (Circular relative offset vs. Block-offset).
<table><tr><td>Metrics</td><td>Block-offset</td><td>Circ. relative offset</td></tr><tr><td>Scaled Jacobian  $\uparrow _ { - 1 } ^ { 1 }$ </td><td>0.7442</td><td>0.9829</td></tr><tr><td>Watertightness  $\uparrow _ { 0 } ^ { 1 }$ </td><td>0.6635</td><td>0.9996</td></tr><tr><td>Manifoldness  $\uparrow _ { 0 } ^ { 1 }$ </td><td>0.9993</td><td>0.9999</td></tr><tr><td>Chamfer distance  $\downarrow _ { 0 } ^ { \infty }$ </td><td>0.3161</td><td>0.0684</td></tr><tr><td>Hausdorff distance  $\downarrow _ { 0 } ^ { \infty }$ </td><td>0.5487</td><td>0.1058</td></tr></table>

![](images/c40898faaff0d8fdb9684d18d81cd187315e4c0dfbc59299f047632518b43ea8.jpg)

![](images/c95e5275235573d4acfe1b11c5a265159400e39cbd54a52772a386994b9bee60.jpg)  
Figure 5: Qualitative ablation on serialization schemes.

## 4.4. Ablation Study

To validate the effectiveness of the core designs in our Q-MAGIC pipeline, we conduct ablation studies on the center vertex selection and the serialization scheme, with results reported in Table 3 and Table 4. Qualitative visualizations are provided in Figure 5 and Figure 6

Table 3 shows that singularity-aware traversal consistently improves mesh validity and element quality over random traversal. Since singularity regions determine local valence distributions and edge-loop organization, errors introduced there are difficult to correct later. Random traversal often establishes connectivity in regular regions first, forcing the model to satisfy singularity constraints within an already generated structure, which can lead to disordered layouts and broken loops. In contrast, singularity-aware traversal anchors critical connections early, providing stronger structural guidance and reducing error accumulation.

![](images/56ad7e8921e5994267dd6a8c5338fb1ec9c7007078bb2f96af3eee3980be8771.jpg)

![](images/53a4912d03699b9966d243c26b4501314c36b04b2eafe309399799ba95e4737a.jpg)  
Figure 6: Qualitative ablation on patch traversal order.

Table 4 shows that the circular relative offset representation is more robust than the block-offset representation. By shifting prediction from global positions to local displacements, circular relative offsets operate within a bounded value range and avoid the large jumps introduced by global discretization. In contrast, absolute offsets are more susceptible to drift across regions, and the resulting errors can be amplified during de-serialization, leading to reduced mesh validity and geometric accuracy.

We further analyze the computational sensitivity to the block size B and offset size O. Increasing (B, O) from the default (8, 16) to (16, 32) raises the per-axis spatial resolution from 128 to 512, while substantially enlarging the corresponding token spaces. Under the same training setting, the peak GPU memory increases and the training time increases by approximately 53%. Therefore, we adopt $( B , O ) = ( 8 , 1 6 )$ as a practical configuration that provides sufficient spatial resolution while maintaining manageable memory and training costs.

## 5. Limitations and Future Work

Despite the promising results, Q-MAGIC still has several limitations. As an autoregressive framework, its computational cost increases with sequence length, which may limit scalability to highly complex or high-resolution meshes. In addition, global connectivity across locally generated regions is not always guaranteed for complex shapes. Our current evaluation demonstrates good generalization to diverse shapes within the Thing10K distribution; however, cross-domain generalization to CAD-oriented or artistdesigned quad meshes with different topology and edge-flow patterns remains to be further explored. Finally, the singularity-aware traversal currently relies primarily on vertex valence.

In future work, we plan to explore hierarchical decoding for better scalability, explicit global connectivity constraints for improved topological consistency, more diverse cross-domain training data for stronger generalization, and geometry-aware singularity selection incorporating curvature, feature lines, and boundary information.

## 6. Conclusion

In this paper, we introduce Q-MAGIC, an autoregressive Transformer framework for quad-only mesh generation, which takes the undifferentiable connectivity prediction as the next face token prediction task. To shorten token sequences and enhance the overall regularity of the global mesh, we introduce a singularity-aware serialization to prioritize topology-critical wiring and improve quad mesh validity. Furthermore, to compactly encode geometry and handle mesh-density variation, we propose circular relative offset tokenization, which encodes geometry via local relative offsets into discrete tokens, achieving mesh-density adaptation. Experimental results demonstrate that our Q-MAGIC outperforms baseline methods in both mesh validity and reconstruction quality.

## Acknowledgment

This research was supported by the National Key R&D Program of China under Grant No. 2021YFA1003003, the Natural Science Foundation of China under Grant No. T2225012, 12494554, 12494550.

## References

[CCP<sup>∗</sup>24] CHEN S., CHEN X., PANG A., ZENG X., CHENG W., FU Y., YIN F., WANG Z., YU J., YU G., FU B., CHEN T.: MeshXL: Neural coordinate field for generative 3d foundation models. In The Thirtyeighth Annual Conference on Neural Information Processing Systems (2024). 2

[CHH<sup>∗</sup>25] CHEN Y., HE T., HUANG D., YE W., CHEN S., TANG J., CAI Z., YANG L., YU G., LIN G., ZHANG C.: Meshanything: Artistcreated mesh generation with autoregressive transformers. In The Thirteenth International Conference on Learning Representations (2025). 2

[CLW<sup>∗</sup>22] CHEN X., LI T., WAN Q., HE X., GONG C., PANG Y., LIU J.: Mgnet: a novel differential mesh generation method based on unsupervised neural networks. Eng. Comput. 38, 5 (2022), 4409–4421. doi:10.1007/S00366-022-01632-7. 2

[CLZ<sup>∗</sup>26] CHEN Y., LIU X., ZHU X., ZHU Y., CHEN Z., ZHANG D., GUO C.: Topgen: Learning structural layouts and cross-fields for quadrilateral mesh generation, 2026. arXiv:2603.10606. 2

[CQL<sup>∗</sup>25] CHEN D., QU Y., LI X., LI M., ZHANG S.: Xspecmesh: Quality-preserving auto-regressive mesh generation acceleration via multi-head speculative decoding, 2025. arXiv:2507.23777. 2

[CWL<sup>∗</sup>25] CHEN Y., WANG Y., LUO Y., WANG Z., CHEN Z., ZHU J., ZHANG C., LIN G.: Meshanything v2: Artist-created mesh generation with adjacent mesh tokenization. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) (October 2025), pp. 13922–13931. 1, 2

[DSS<sup>∗</sup>26] DECATUR D., SERFATY J., STEIN O., VAXMAN A., HANOCKA R.: Look both ways before you cross: Lifting cross fields from 2d visual priors, 2026. URL: https://arxiv.org/abs/ 2605.26062, arXiv:2605.26062. 2

[DWX<sup>∗</sup>25a] DONG Q., WANG J.-C., XU R., LIN C., LIU Y., XIN S., ZHONG Z., LI X., TU C., KOMURA T., KOBBELT L., SCHAEFER S., WANG W.: Crossgen: Learning and generating cross fields for quad meshing. ACM Transactions on Graphics (TOG) 44 (2025), 1 – 15. 2

[DWX<sup>∗</sup>25b] DONG Q., WEN H., XU R., CHEN S., ZHOU J., XIN S., TU C., KOMURA T., WANG W.: Neurcross: A neural approach to computing cross fields for quad mesh generation. ACM Trans. Graph. 44, 4 (2025), 108:1–108:17. 2

[GC20] GUPTA K., CHANDRAKER M.: Neural mesh flow: 3d manifold mesh generation via diffeomorphic flows. In Advances in Neural Information Processing Systems (2020), Larochelle H., Ranzato M., Hadsell R., Balcan M., Lin H., (Eds.), vol. 33, Curran Associates, Inc., pp. 1747– 1758. 2

[HRLL24] HAO Z., ROMERO D. W., LIN T.-Y., LIU M.-Y.: Meshtron: High-fidelity, artist-like 3d mesh generation at scale. ArXiv abs/2412.09548 (2024). 2

[HTC<sup>∗</sup>24] HONG F., TANG J., CAO Z., SHI M., WU T., CHEN Z., WANG T., PAN L., LIN D., LIU Z.: 3dtopia: Large text-to-3d generation model with hybrid diffusion priors. ArXiv abs/2403.02234 (2024). 1

[LFWN20] LIN C., FAN T., WANG W., NIESSNER M.: Modeling 3d shapes by reinforcement learning. In European Conference on Computer Vision (2020). 2

[LLG<sup>∗</sup>26] LIN J., LONG H., GUO H., ZHANG J., YANG J., GUO T., YANG Y., LI J., ZHANG W., NIESSNER M., YANG W.: Meshripple: Structured autoregressive generation of artist-meshes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (June 2026), pp. 12706–12718. 2

[LLL25a] LIONAR S., LIANG J., LEE G. H.: Treemeshgpt: Artistic mesh generation with autoregressive tree sequencing. 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2025), 26608–26617. 6

[LLL<sup>∗</sup>25b] LIU Y.-T., LI J., LIU Y.-T., YU X., GUO Y.-C., CAO Y.- P., LIANG D., SHAMIR A., ZHANG S.-H.: Neuframeq: Neural frame fields for scalable and generalizable anisotropic quadrangulation. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV) (October 2025), pp. 28000–28009. 2

[LQW<sup>∗</sup>25] LI Z., QI Z., WANG W., WANG Z., DUAN J., LEI N.: Point2quad: Generating quad meshes from point clouds via face prediction. IEEE Transactions on Circuits and Systems for Video Technology 35 (2025), 8586–8597. 2, 6

[LSCdG<sup>∗</sup>21] LEBRAT L., SANTA CRUZ R., DE GOURNAY F., FU D., BOURGEAT P., FRIPP J., FOOKES C., SALVADO O.: Corticalflow: A diffeomorphic mesh transformer network for cortical surface reconstruction. In Advances in Neural Information Processing Systems (2021), Ranzato M., Beygelzimer A., Dauphin Y., Liang P., Vaughan J. W., (Eds.), vol. 34, Curran Associates, Inc., pp. 29491–29505. 2

[LWG<sup>∗</sup>25] LIU J., WANG C., GUO S., WENG H., ZHOU Z., LI Z., YU J., ZHU Y., XU J., LEI B., CHEN Z., GUO C.: Quadgpt: Native quadrilateral mesh generation with autoregressive models. ArXiv abs/2509.21420 (2025). 1, 2, 3, 6

[LXG<sup>∗</sup>25] LIU J., XU J., GUO S., LI J., GUOJINGFENG, YU J., WENG H., LEI B., YANG X., CHEN Z., ZHU F., HAN T., GUO C.: Mesh-RFT: Enhancing mesh generation via fine-grained reinforcement finetuning. In The Thirty-ninth Annual Conference on Neural Information Processing Systems (2025). 2

[Max24] MAXIME: Quad remesher, 2024. Exoside. 6

[MHS<sup>∗</sup>21] MUNKBERG J., HASSELGREN J., SHEN T., GAO J., CHEN W., EVANS A., MÜLLER T., FIDLER S.: Extracting triangular 3d models, materials, and lighting from images. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2021), 8270–8280. 1

[MK04] MARINOV M., KOBBELT L.: Direct anisotropic quad-dominant remeshing. 12th Pacific Conference on Computer Graphics and Applications, 2004. PG 2004. Proceedings. (2004), 207–216. 2

[PCL25] PENG J., CHEN X., LIU J.: 3dmeshnet: A three-dimensional differential neural network for structured mesh generation. Graph. Model. 139 (2025), 101257. doi:10.1016/J.GMOD.2025. 101257. 2

[PFS<sup>∗</sup>19] PARK J. J., FLORENCE P., STRAUB J., NEWCOMBE R., LOVEGROVE S.: Deepsdf: Learning continuous signed distance functions for shape representation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (June 2019). 1

[PHCZ22] PAN J., HUANG J., CHENG G., ZENG Y.: Reinforcement learning for automatic quadrilateral mesh generation: a soft actor-critic approach. CoRR abs/2203.11203 (2022). arXiv:2203.11203, doi:10.48550/ARXIV.2203.11203. 2

[PNA<sup>∗</sup>21] PIETRONI N., NUVOLI S., ALDERIGHI T., CIGNONI P., TARINI M., ET AL.: Reliable feature-line driven quad-remeshing. TOG 40, 4 (2021), 1–17. 2, 6

[QQX<sup>∗</sup>25] QI L., QIU J., XU G., LIU Y., XU J., GU R., LU F., PANG Y.: Rff-meshing: a parallel and anisotropic quad-dominant mesh generation framework based on riemann frame field. Engineering with Computers 41 (2025), 2267–2289. doi:10.1007/ s00366-024-02096-7. 2

[RMA<sup>∗</sup>24] RAKOTOSAONA M.-J., MANHARDT F., ARROYO D. M., NIEMEYER M., KUNDU A., TOMBARI F.: Nerfmeshing: Distilling neural radiance fields into geometrically-accurate 3d meshes. In 2024 International Conference on 3D Vision (3DV) (2024), pp. 1156–1165. doi:10.1109/3DV62453.2024.00093. 1

[Ros99] ROSSIGNAC J.: Edgebreaker: connectivity compression for triangle meshes. IEEE Transactions on Visualization and Computer Graphics 5, 1 (1999), 47–61. doi:10.1109/2945.764870. 2

[RSM<sup>∗</sup>23] RAFAILOV R., SHARMA A., MITCHELL E., MANNING C. D., ERMON S., FINN C.: Direct preference optimization: Your language model is secretly a reward model. In Advances in Neural Information Processing Systems (2023), Oh A., Naumann T., Globerson A., Saenko K., Hardt M., Levine S., (Eds.), vol. 36, Curran Associates, Inc., pp. 53728–53741. 2

[SAA<sup>∗</sup>24] SIDDIQUI Y., ALLIEGRO A., ARTEMOV A., TOMMASI T., SIRIGATTI D., ROSOV V., DAI A., NIESSNER M.: Meshgpt: Generating triangle meshes with decoder-only transformers. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2024), pp. 19615–19625. doi:10.1109/CVPR52733. 2024.01855. 2

[SLWR25] SUN X., LIANG C., WANG Q., RAKITA D.: Mesh rag: Retrieval augmentation for autoregressive mesh generation, 2025. arXiv: 2511.16807. 2

[SZT<sup>∗</sup>25] SHEN T., ZHANG Y., TANG C., PING C., ZHAO Z., WAN L., WANG Y., WANG R., HE S.: Flashmesh: Faster and better autoregressive mesh synthesis via structured speculation, 2025. arXiv: 2511.15618. 2

[SZW<sup>∗</sup>26] SONG G., ZHAO Z., WENG H., ZENG J., JIA R., GAO S.: Topology-preserved auto-regressive mesh generation in the manner of weaving silk. In The Fourteenth International Conference on Learning Representations (2026). 2

[TLH<sup>∗</sup>25] TANG J., LI Z., HAO Z., LIU X., ZENG G., LIU M.-Y., ZHANG Q.: Edgerunner: Auto-regressive auto-encoder for artistic mesh generation. In The Thirteenth International Conference on Learning Representations (2025). 2

[TQHZ23] TONG H.-S., QIAN K., HALILAJ E., ZHANG Y. J.: Srlassisted afm: Generating planar unstructured quadrilateral meshes with supervised and reinforcement learning-assisted advancing front method. J. Comput. Sci. 72 (2023), 102109. 2

[WGL<sup>∗</sup>26] WANG H., GUO Y.-C., LIU Y.-T., ZOU Z.-X., ZHANG B., QUAN W., LIANG D., CAO Y.-P., YAN D.-M.: Face: A face-based autoregressive representation for high-fidelity and efficient mesh generation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (June 2026), pp. 12719–12729. 2

[WLL<sup>∗</sup>21] WANG P., LIU L., LIU Y., THEOBALT C., KOMURA T., WANG W.: Neus: Learning neural implicit surfaces by volume rendering for multi-view reconstruction. In Advances in Neural Information Processing Systems (2021), Beygelzimer A., Dauphin Y., Liang P., Vaughan J. W., (Eds.). 1

[WWZ<sup>∗</sup>25a] WEI S.-T., WANG R.-H., ZHOU C.-Z., CHEN B., WANG P.-S.: Octgpt: Octree-based multiscale autoregressive models for 3d shape generation. Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers (2025). 2

[WWZ<sup>∗</sup>25b] WENG H., WANG Y., ZHANG T., CHEN C. L. P., ZHU J.: Pivotmesh: Generic 3d mesh generation via pivot vertices guidance. In The Thirteenth International Conference on Learning Representations (2025). 2

[WZL<sup>∗</sup>25] WENG H., ZHAO Z., LEI B., YANG X., LIU J., LAI Z., CHEN Z., LIU Y., JIANG J., GUO C., ZHANG T., GAO S., PHILIP CHEN C. L.: Scaling mesh generation via compressive tokenization. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2025), pp. 11093–11103. doi:10.1109/ CVPR52734.2025.01036. 1, 2, 3, 6

[XCG<sup>∗</sup>24] XU J., CHENG W., GAO Y., WANG X., GAO S., SHAN Y.: Instantmesh: Efficient 3d mesh generation from a single image with sparse-view large reconstruction models, 2024. arXiv:2404.07191. 1

[XCP<sup>∗</sup>26] XIAO J., CHEN X., PENG J., WANG Q., LIU J.: Meshonet: A generalizable and efficient operator learning method for structured mesh generation. Neural Networks 199 (2026), 108746. URL: http://dx.doi.org/10.1016/j.neunet. 2026.108746, doi:10.1016/j.neunet.2026.108746. 2

[XQQ<sup>∗</sup>26] XU R., QIN D., QIAO K., DONG Q., PI H., ZHANG Q., ZHANG L., XU L., YU J., WANG W., KOMURA T.: Strips as tokens: Artist mesh generation with native uv segmentation, 2026. arXiv: 2604.09132. 2

[YLGL25] YU Y., LI N., GONG G., LIN X.: A neural poly-vector based non-orthogonal frame field generation method for quad meshing. Scientific Reports 15, 1 (2025), 33595. doi:10.1038/ s41598-025-18823-z. 2

[ZLL<sup>∗</sup>26] ZHOU Z., LIU J., LEI B., XU J., WENG H., ZHU Y., CHEN Z., FAN J., MA Y., DU D., GUO S., JING F., GUO C.: Mesh-pro: Asynchronous advantage-guided ranking preference optimization for artiststyle quadrilateral mesh generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (June 2026), pp. 34248–34258. 2

[ZLW24] ZHANG Z., LIN C., WANG B.: Physics-informed shape optimization using coordinate projection. Scientific Reports 14, 1 (2024), 6537. 2

[ZSA<sup>∗</sup>25] ZHANG X., SIDDIQUI Y., AVETISYAN A., XIE C., ENGEL J., HOWARD-JENKINS H.: Vertexregen: Mesh generation with continuous level of detail. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) (October 2025), pp. 12570–12580. 2

[ZYW<sup>∗</sup>25] ZHAO R., YE J., WANG Z., LIU G., CHEN Y., WANG Y., ZHU J.: Deepmesh: Auto-regressive artist-mesh creation with reinforcement learning. In Proceedings of the IEEE/CVF International Conference on Computer Vision (2025), pp. 10612–10623. 2

[ZZS<sup>∗</sup>26] ZHANG Y., ZHU Z., SHEN T., CAI Z., LI T., ZHAO Z., DONG Q., DOU Z., WANG J., WAN L., WANG Y., WANG W., LIU Y., LIN C.: Quadlink: Autoregressive quad-dominant mesh generation via pointrelation learning, 2026. arXiv:2605.16813. 2