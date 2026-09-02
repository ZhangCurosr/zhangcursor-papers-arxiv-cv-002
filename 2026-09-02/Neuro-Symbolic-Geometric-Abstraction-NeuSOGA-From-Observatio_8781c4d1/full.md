# Neuro-Symbolic Geometric Abstraction (NeuSOGA): From Observations to Symbolic Mathematical Representations

Qingde Li<sup>1</sup>, Qingqi Hong<sup>2</sup>, Jie Tian<sup>3</sup>, Fellow, IEEE

<sup>1</sup>Computer Science, School of Digital and Physical Sciences, University of Hull, HU6 7RX, UK

<sup>2</sup>The Institute of Artificial Intelligence, Xiamen University, Xiamen, China

<sup>3</sup> Chinese Academy of Sciences, Institute of Automation, Beijing, China

A fundamental challenge in artificial intelligence is the transformation of perceptual observations into explicit symbolic representations suitable for abstraction, reasoning, and knowledge formation. While modern AI systems achieve remarkable perceptual performance through large-scale statistical learning, the resulting knowledge is typically encoded within latent parameters that are difficult to interpret, manipulate, or analyse mathematically. Inspired by Neuro-Symbolic AI and theories of human abstraction, this paper investigates how symbolic mathematical representations can emerge from geometric observations. We propose NeuSOGA (Neuro-Symbolic Geometric Abstraction), a framework that implements the transformation O → T → G → S, where observations (O) are progressively transformed into topological abstractions (T), geometric abstractions (G), and ultimately symbolic mathematical representations (S). The architecture combines topology-guided structural discovery using Euclidean Distance Transforms, foundation-model perception using Segment Anything, adaptive multi-scale geometric abstraction based on scalespace theory, and symbolic synthesis through Implicit Area Splines. Unlike conventional learning-based reconstruction approaches, structural knowledge is derived primarily through topological and geometric reasoning rather than task-specific geometric training. The resulting representation is an analytical implicit model possessing arbitrary-order smoothness (C<sup>n</sup> continuity), strict additivity, and closed-form evaluation. Unlike neural latent encodings, the generated Area Spline representation remains interpretable, editable, compositional, and mathematically explicit. Experiments on ModelNet40 point-cloud data, arbitrary-view projections, and segmented optical observations demonstrate that NeuSOGA consistently transforms diverse observations into compact symbolic representation while preserving essential topological and geometric structure across sensing modalities and viewing directions. The principal contribution of this work is the introduction of a neuro-symbolic mechanism for symbolic representation formation, establishing a computational bridge between perception and mathematical representation. By demonstrating the transformation from observation to symbol through topology-guided geometric abstraction, NeuSOGA provides an interpretable and explainable foundation for future systems capable of higher-level geometric reasoning, concept formation, and machine-assisted mathematical discovery.

Index Terms—Neuro-Symbolic AI, Symbolic Representation Formation, Mathematical Abstraction, Geometric Abstraction, Topological Reasoning, Topology-Guided Perception, Computational Geometry, Implicit Area Splines, Symbolic Geometry, Shape Representation, Explainable AI, Machine Reasoning

## I. INTRODUCTION

Recent advances in deep learning have produced powerful models capable of perception, pattern recognition, and geometric inference across a wide range of domains, including computer vision, shape analysis, and computer-aided design reconstruction [5], [28], [38], [12], [20]. These systems typically derive their capabilities from large-scale statistical learning, where neural networks are optimized using vast collections of training examples. While this paradigm has achieved remarkable practical success, the resulting representations are generally encoded within latent neural parameters whose structure is difficult to interpret, manipulate, or analyse directly. Furthermore, generalization remains challenging when observations differ substantially from the geometries, structures, or topologies encountered during training [15], [23]. The computational and energy demands associated with large-scale neural systems have also motivated growing interest in alternative approaches that combine learning with more explicit forms of representation and reasoning [34], [9]. Human mathematical cognition appears to operate in a fundamentally different manner. Rather than relying solely on large numbers of examples, humans can often infer general geometric principles from relatively few observations and subsequently construct abstract symbolic representations that extend beyond immediate sensory experience [15], [36]. Developmental cognitive research suggests that humans possess innate forms of “core knowledge” concerning objects, space, geometry, and topological relationships, providing a foundation for higher-level abstraction and reasoning [33], [11]. Lake et al. argue that human intelligence is characterized not only by pattern recognition but also by its ability to construct causal, compositional, and abstract representations supporting systematic generalization [15]. Such capabilities allow mathematicians and scientists to derive idealized geometric structures and analytical models from incomplete and noisy observations of the physical world. This observation motivates a broader view of intelligence as a hierarchical process of abstraction. At the perceptual level, observations are acquired from an uncertain and noisy environment. At the structural level, invariant geometric and topological relationships are extracted from these observations. At the symbolic level, the resulting structures are represented analytically in forms that support interpretation, manipulation, and subsequent reasoning [27]. From this perspective, mathematical abstraction may be viewed as a process of transforming observations into explicit symbolic representations through successive layers of structural organization. The growing interest in neuro-symbolic artificial intelligence reflects the recognition that neither purely neural nor purely symbolic approaches are independently sufficient for supporting this form of abstraction [10], [1], [8], [2], [9]. Neural systems provide robustness to noise, ambiguity, and incomplete information, while symbolic systems offer explicit representations, interpretability, compositionality, and analytical structure [27]. A neuro-symbolic architecture therefore provides a promising framework for investigating how symbolic representations can emerge from perceptual observations. The Scan-to-CAD problem provides a particularly useful testbed for exploring this question. The objective is to transform unstructured geometric observations into compact, editable, and mathematically meaningful representations. Recent approaches such as DeepCAD, CAD-SIGNet, Point2CAD, and related CAD reconstruction systems have demonstrated impressive performance through the learning of geometric priors from large collections of synthetic CAD models [38], [12], [20], [22]. However, these methods remain fundamentally dependent upon learned statistical representations and training data distributions. Consequently, generalization to previously unseen topologies, sparse observations, and structurally novel geometries remains an active challenge. We argue that a key limitation of existing approaches is the absence of an explicit abstraction layer that separates topological structure, geometric form, and symbolic representation. A point cloud contains not only geometric measurements but also latent structural information that reflects the organization of the observed object. Rather than learning this structure exclusively through statistical inference, we propose to extract it directly through deterministic geometric operators. The Medial Axis Transform (MAT) introduced by Blum provides a compact description of shape topology through skeletal representations [4], while Euclidean Distance Transforms (EDT) provide an efficient mechanism for computing intrinsic spatial relationships within geometric domains [25]. Together, these methods reveal topological abstractions that are largely independent of specific geometric parameterizations. Building upon this foundation, we introduce NeuSOGA (Neuro-Symbolic Geometric Abstraction), a framework that implements the transformation

![](images/b72604fb8febc65a268f6af59fe114468a885881ec2242da2fffc51eb8242df9.jpg)

![](images/04c95e59fe6db5c458974a80f19271ddcdf079cdc86a8c56bffdcea7fc4062b1.jpg)

![](images/0dc90151b1a95a9634ccc563a7835fc8577c9e370f12cff55b406968006e367f.jpg)

![](images/935bbf721040c8b3300cf4e8513470aea036298ae8b09a0010b8583e9be84c9d.jpg)

![](images/990fc31ac3758d859476354cf27b7cd03f53a5e4a8ab181e5277db234c547add.jpg)

![](images/ce5cecb2299f817762dafb87d8736776d82b904c96c647923de56c65d31e272a.jpg)  
Fig. 1. Overview of the proposed NeuSOGA (Neuro-Symbolic Geometric Abstraction) pipeline: $\mathcal { O } \to \mathcal { T } \to \mathcal { G } \to \mathcal { S } .$ Starting from a raw point-cloud observation, the framework extracts topology-aware structural cores using Euclidean Distance Transforms, employs these abstractions to guide foundationmodel perception, performs adaptive multi-scale geometric abstraction, and finally constructs an explicit symbolic mathematical representation through Implicit Area Spline synthesis. Unlike conventional learning-based reconstruction methods that encode geometry within latent neural parameters, the proposed framework produces an analytical representation $F ( x , y ) = { \check { 0 } }$ that is interpretable, editable, additive, and mathematically explicit.

$$
\mathcal { O } \to \mathcal { T } \to \mathcal { G } \to \mathcal { S } ,
$$

where observations (O) are progressively transformed into topological abstractions (T), geometric abstractions (G), and ultimately symbolic mathematical representations (S). Within NeuSOGA, modern foundation models such as SAM and SAM2 [13], [30] serve as perceptual front ends for isolating geometric structures from observations, while structural understanding emerges through topology-guided abstraction rather than additional task-specific geometric training. Global geometric organization is captured using scale-space theory [37], [17], [19], and sparse geometric abstractions are constructed through adaptive multi-scale simplification based on the Douglas-Peucker principle [7]. The resulting geometric abstractions are subsequently projected into an analytical symbolic representation using Implicit Area Splines [16]. Unlike conventional parametric spline representations such as NURBS [29], which primarily describe boundary trajectories, the Area Spline formulation represents spatial regions directly through implicit analytical fields. This representation supports arbitrary-order smoothness, closed-form evaluation, and additive composition, making it particularly suitable as a symbolic representation layer within the proposed neurosymbolic architecture. More broadly, NeuSOGA explores a computational mechanism by which symbolic mathematical representations may emerge from observations through topology-guided geometric abstraction. The contribution of this work is therefore not merely a new reconstruction pipeline, but a neuro-symbolic framework for symbolic representation formation. By establishing an explicit observation-to-symbol transformation,

$$
\mathcal { O } \to \mathcal { T } \to \mathcal { G } \to \mathcal { S } ,
$$

NeuSOGA provides an interpretable and explainable substrate upon which future mechanisms for symbolic reasoning, concept formation, and machine-assisted mathematical discovery may be constructed.

To formalize this perspective, we introduce NeuSOGA (Neuro-Symbolic Geometric Abstraction), a computational framework for transforming observations into explicit symbolic mathematical representations. NeuSOGA implements the abstraction hierarchy

$$
\mathcal { O } \to \mathcal { T } \to \mathcal { G } \to \mathcal { S } ,
$$

where observations (O) are progressively transformed into topological abstractions (T ), geometric abstractions (G), and ultimately symbolic mathematical representations (S). The central premise of NeuSOGA is that symbolic representation formation constitutes the critical bridge between perception and reasoning. Rather than encoding geometric knowledge within latent neural parameters, NeuSOGA progressively constructs explicit representations whose structure remains accessible to interpretation, manipulation, analysis, and future symbolic reasoning. The main contributions of this work are as follows:

1) NeuSOGA: A neuro-symbolic framework for observation-to-symbol transformation. We introduce a computational abstraction hierarchy $\mathcal { O } \to \mathcal { T } \to \mathcal { G } \to \mathcal { S }$ that transforms observations into explicit symbolic mathematical representations through topology-guided geometric abstraction.

2) Topology-guided integration of perception and reasoning. We develop a topology-driven prompting strategy that couples Euclidean Distance Transform analysis with foundation-model perception, enabling structural discovery and object isolation without task-specific geometric training.

3) Adaptive multi-scale geometric abstraction. We present an error-driven coarse-to-fine abstraction mechanism that selectively preserves geometrically significant features while compressing dense observations into sparse and structurally meaningful control polygons.

4) Implicit Area Splines as a symbolic representation layer. We demonstrate that Implicit Area Splines provide an explicit, interpretable, and mathematically accessible symbolic representation. The resulting models support arbitrary-order smoothness, additive composition, and closed-form evaluation, while remaining directly available for inspection, editing, and future symbolic reasoning.

5) Modality- and viewpoint-independent symbolic representation formation. Through experiments on ModelNet40 point clouds, arbitrary-view projections, and segmented optical observations from the COCO dataset, we demonstrate that NeuSOGA consistently transforms diverse observations into compact symbolic mathematical representations while preserving essential topological and geometric structure.

6) An explainable pathway from perception to symbol. By maintaining explicit topological, geometric, and symbolic representations throughout the abstraction hierarchy, NeuSOGA provides explainability by construction, offering a transparent alternative to purely latent representation learning approaches.

Figure 1 provides an overview of the NeuSOGA architecture and illustrates the complete observation-to-symbol transformation implemented by the proposed framework.

Figure 1 summarizes the proposed neuro-symbolic abstraction pipeline. The framework progressively transforms observations into symbolic mathematical representations through topological discovery, geometric abstraction, and analytical spline synthesis.

## II. RELATED WORK

A. Foundation Models, Deep Learning, and the Limits of Statistical Learning

Recent advances in deep learning and foundation models have transformed computer vision, natural language processing, and geometric understanding through large-scale datadriven learning [5]. Vision foundation models such as Seg ment Anything (SAM) [13] and SAM2 [30] demonstrate remarkable zero-shot generalization capabilities and provide highly effective mechanisms for extracting semantic information from visual data. Despite these successes, concerns remain regarding the ability of purely statistical learning systems to perform robust abstraction, reasoning, and knowledge discovery. Lake et al. argue that human intelligence depends upon compositionality, causal reasoning, and model-based abstraction, capabilities that are difficult to achieve through pattern recognition alone [15]. Similarly, Marcus identifies systematic generalization, explainability, and reasoning as fundamental challenges for purely connectionist systems [23]. Furthermore, scaling modern deep learning systems increasingly requires substantial computational resources and energy consumption [34]. These limitations have motivated significant interest in hybrid paradigms capable of combining the strengths of learning and reasoning.

## B. Neuro-Symbolic AI and Machine Abstraction

The integration of neural learning and symbolic reasoning has been an active research area for more than two decades. Early neural-symbolic systems sought to combine the adaptability of neural networks with the formal expressiveness and inferential capabilities of symbolic logic [10]. Bader and

Hitzler classified neural-symbolic architectures according to their mechanisms for knowledge representation, learning, and reasoning, providing one of the earliest systematic frameworks for integrating symbolic and connectionist computation [1]. More recent work has expanded this vision into the broader field of Neuro-Symbolic AI. Garcez et al. argued that future AI systems require the principled integration of learning, reasoning, and knowledge representation [8]. Besold et al. further identified neural-symbolic learning as a mechanism for bridging low-level representation learning and high-level cognitive reasoning [2]. This perspective has culminated in the concept of the “third wave” of AI, in which neural computation and symbolic reasoning are treated as complementary components of a unified intelligent system [9]. The motivation for neuro-symbolic AI is closely related to findings from cognitive science. Spelke and Kinzler’s Core Knowledge theory suggests that humans possess structured priors concerning objects, geometry, space, and topology that support rapid abstraction from limited experience [33]. Similarly, Lake et al. propose that human intelligence emerges through the construction of compositional and reusable abstract models capable of supporting systematic generalization [15]. These observations suggest that mathematical cognition relies on more than statistical learning alone; it requires mechanisms for extracting invariant structures and transforming them into symbolic representations. Our work adopts this perspective and investigates a topology-driven neuro-symbolic architecture in which perception, abstraction, and symbolic synthesis are explicitly separated. Unlike conventional learning-based approaches, abstraction is performed through deterministic geometric analysis rather than data-driven optimization.

## C. Learning-Based Scan-to-CAD Reconstruction

The reconstruction of editable CAD models from point clouds remains a central problem in reverse engineering and geometric modeling. Benchmark initiatives such as the SHARP Challenge highlight the difficulty of recovering CAD design intent, construction history, and parametric information from real-world scans [22]. Early work such as CSGNet formulated reconstruction as the inference of constructive solid geometry programs from observed shapes [32]. Subsequently, DeepCAD represented CAD reconstruction as a sequence-generation problem, learning parametric modeling operations from large CAD repositories [38]. Recent methods have introduced increasingly sophisticated architectures. CAD-SIGNet employs sketch-guided attention mechanisms to infer feature histories from point clouds [12], while Point2CAD combines neural predictions with geometric fitting procedures to improve reconstruction accuracy [20]. KP-RED further utilizes semantically meaningful keypoints to establish geometric correspondence and improve shape recovery [39]. Although these approaches differ substantially in their architectures, they share a common assumption: geometric understanding is acquired through statistical learning from large collections of synthetic examples. Consequently, their ability to generalize remains dependent upon the diversity and completeness of the training distribution. In contrast, the proposed framework seeks to derive geometric structure directly from topological and geometric principles rather than from learned statistical correlations.

## D. Topological Shape Abstraction

A fundamental requirement for mathematical abstraction is the discovery of invariant structure. Within geometric modeling, topology provides such invariants by describing structural relationships that remain unchanged under continuous deformation. Blum’s Medial Axis Transform (MAT) represents one of the most influential approaches to topological shape analysis [4]. By reducing a shape to its skeletal structure and associated radius function, MAT provides a compact representation of object organization while preserving essential topological information. Euclidean Distance Transforms (EDTs) complement this representation by providing efficient computation of intrinsic geometric relationships across arbitrary dimensions [25]. Unlike learned latent representations, which often encode topology implicitly within network parameters, MAT and EDT provide explicit and interpretable descriptions of geometric structure. In the proposed framework, these methods serve as mechanisms for invariant extraction, transforming raw observations into symbolic topological descriptions that can subsequently support analytical reasoning.

## E. Explicit and Implicit Geometric Representations

The representation of geometry has long been a central concern within CAD and solid modeling. Classical geometric systems rely primarily on explicit representations such as boundary representations (B-Reps), constructive solid geometry (CSG), and spline-based surface models [31]. Among these, NURBS have become the dominant industrial standard due to their geometric flexibility, precision, and compatibility with manufacturing processes [29]. However, explicit representations often require sophisticated numerical procedures to maintain topological consistency during Boolean operations, intersection calculations, and geometric editing [31]. To overcome these limitations, implicit representations describe shapes as level sets of continuous scalar fields, enabling topological operations to be expressed naturally. Recent neural implicit approaches such as DeepSDF have demonstrated the representational power of learned continuous distance fields [28]. Nevertheless, geometric knowledge in such systems is encoded within network weights, limiting interpretability and direct integration with conventional design workflows. In contrast, Piecewise Algebraic Splines provide a fully analytical implicit representation in which geometry remains explicitly controllable through editable parameters while preserving the topological robustness of implicit modeling [16].

## F. Scale-Space Theory and Multi-Scale Geometric Reasoning

A central problem in geometric abstraction is the separation of meaningful structural information from noise and small-scale variations. Scale-space theory provides a rigorous mathematical framework for addressing this challenge by representing signals and geometric structures across a continuum of observation scales [37], [14]. The fundamental principle underlying scale-space analysis is that meaningful structures should persist as the observation scale increases, while insignificant details gradually disappear. The mathematical foundations of scale-space theory were formalized by Lindeberg, who established a comprehensive framework for multi-scale representation, feature extraction, and scale invariance [17], [19]. One of the key insights of this framework is that no single observation scale is sufficient to capture all meaningful information. Instead, stable geometric structures emerge through comparisons across multiple scales. Lindeberg further demonstrated that automatic scale selection can identify characteristic structures directly from data, eliminating the need for manually chosen smoothing parameters [18]. Multi scale representations have had a profound impact on computer vision and shape analysis. For example, Lowe’s scaleinvariant feature transform (SIFT) demonstrated that robust image features can be identified as extrema within scale-space representations [21]. Similarly, Mokhtarian and Mackworth developed a curvature-based scale-space framework in which meaningful shape features correspond to structures that persist across multiple levels of smoothing [26]. These studies suggest that scale-space representations provide a natural mechanism for extracting invariant geometric information while suppressing noise. In geometric modeling and surface reconstruction, multi-scale analysis has also played an important role in shape fairing and smoothing. Taubin’s signal-processing framework demonstrated how geometric surfaces could be filtered across scales while preserving essential structural properties [35]. Such methods illustrate the broader principle that geometric abstraction should be viewed as a hierarchical process rather than a single-stage operation. The proposed framework adopts this perspective by combining scale-space analysis with topology-guided abstraction. Rather than applying a fixed smoothing level, multiple scale-space representations are evaluated simultaneously and integrated through an adaptive errordriven subdivision strategy. Global shape characteristics are extracted from coarse scales, while local engineering features are preserved through fine-scale analysis and Douglas-Peucker simplification [7]. From a neuro-symbolic perspective, this process performs an abstraction operation analogous to human cognition, in which coarse structural concepts emerge before progressively refined detail. Consequently, scale-space theory provides the mathematical foundation for the transition from perceptual geometry to symbolic geometric representation within the proposed architecture.

## III. EMULATING THE HUMAN MATHEMATICIAN

The proposed architecture is motivated by the hypothesis that mathematical intelligence emerges through the interaction of perception, abstraction, and symbolic reasoning. Cognitive studies suggest that humans do not construct mathematical knowledge purely through statistical learning. Instead, they exploit structured priors concerning objects, geometry, space, and topology to extract invariant relationships from sensory observations and subsequently transform these relationships into symbolic concepts [33], [15]. This view aligns closely with the objectives of Neuro-Symbolic AI, which seeks to integrate data-driven perception with explicit knowledge representation and reasoning [10], [1], [9]. Inspired by these observations, our framework decomposes mathematical cognition into two complementary stages: structural discovery and symbolic invention. The first stage identifies invariant geometric structure directly from observations, while the second constructs symbolic mathematical representations that can support reasoning, editing, and generalization.

## A. Structural Discovery: Extracting Invariants from Observation

Human cognition appears to possess powerful mechanisms for discovering stable structures within noisy sensory environments. Core Knowledge theory suggests that humans reason about objects, boundaries, containment, and spatial relation ships using highly structured internal representations [33]. Similarly, Lake et al. argue that abstraction arises from identifying reusable structural regularities rather than merely memorizing examples [15]. Within the proposed architecture, this capability is realized through topology-driven geometric analy sis. Given an unstructured point cloud, the system first employs visual foundation models as perceptual front ends for isolating meaningful observations. These observations are subsequently transformed through the Euclidean Distance Transform (EDT) [25] and Medial Axis Transform (MAT) [4]. Together, these operations reveal intrinsic geometric relationships that remain largely invariant under boundary deformation. Importantly, the MAT does not function as a learned representation. Instead, it acts as an analytical mechanism for extracting topological structure directly from geometric data. This distinguishes the proposed approach from learning-based reconstruction systems that encode geometric knowledge implicitly within network parameters [32], [38], [12], [20]. The resulting skeletal representation serves as an explicit structural description of the observed object and forms the basis for subsequent symbolic processing. From a neuro-symbolic perspective, this stage corresponds to the transformation of perceptual observations into invariant symbolic primitives. Rather than learning geometric concepts from examples, the system derives structure directly from first principles. In this sense, topology functions as an abstraction layer connecting perception to reasoning.

## B. Symbolic Invention: Constructing Mathematical Reality

Observation alone does not constitute mathematical reasoning. Human mathematicians routinely transform imperfect and incomplete observations into idealized mathematical objects possessing properties that may not exist exactly in physical reality. Lines become infinitely thin, circles become perfectly smooth, and geometric relationships become exact. This process may be interpreted as a transition from observed structure to symbolic representation [15], [8], [2]. The second stage of our framework emulates this transition through symbolic geometric synthesis. Following topological abstraction, the extracted skeletal structure is analyzed across multiple scales using scale-space theory [37], [17], [19]. Multiscale analysis enables the system to distinguish persistent structural features from noise and local measurement artefacts. Global organizational characteristics are encoded at coarse scales, while local engineering details are retained through fine-scale representations. The resulting control points form an explicit symbolic description of the underlying geometry. These symbols are subsequently processed through the Implicit Area Spline framework [16], yielding a continuous analytical representation of the reconstructed object. Unlike neural implicit representations such as DeepSDF [28], the generated model is fully interpretable, analytically defined, and directly editable through symbolic geometric parameters. From the perspective of Neuro-Symbolic AI, the spline formulation serves as a reasoning substrate rather than merely a geometric representation. Control points, spline segments, and algebraic constraints constitute explicit symbolic entities that can be manipulated, combined, and analyzed independently of the original observations. The resulting geometry therefore represents not only a reconstruction of the observed object but also a mathematical hypothesis about its idealized structure.

## C. Toward Machine Mathematical Abstraction

The proposed architecture suggests a broader interpretation of mathematical intelligence. Rather than viewing intelligence solely as statistical prediction, we view it as a process of transforming observations into invariants and invariants into symbolic representations. This progression can be expressed conceptually as

$$
\mathcal { O } \xrightarrow { \mathrm { P e r c e p t i o n } } \mathcal { T } \xrightarrow { \mathrm { A b s t r a c t i o n } } \mathcal { S } \xrightarrow { \mathrm { R e a s o n i n g } } \mathcal { C }
$$

where O denotes observations, $\tau$ denotes topological invariants, $s$ denotes symbolic representations, and $\mathcal { C }$ denotes mathematical concepts. Under this interpretation, Scan-to-CAD reconstruction becomes a specific instance of a more general cognitive process. The objective is not merely to recover geometry but to construct symbolic models from empirical observations. This aligns with the broader goals of Neuro-Symbolic $\mathrm { A I } ,$ namely the integration of perception, abstraction, knowledge representation, and reasoning within a unified computational framework [9]. The proposed system therefore serves as a concrete demonstration of how topologydriven abstraction may support machine mathematical reasoning and concept formation.

## IV. THE IMPLICIT AREA SPLINE

The final stage of the proposed architecture transforms topological abstractions into explicit mathematical representations. Within the neuro-symbolic framework, this stage plays the role of symbolic synthesis, converting geometric observations into analytical expressions that can be manipulated independently of the original sensory data. Unlike learned latent representations, which encode geometric knowledge implicitly within network parameters, the proposed representation remains entirely analytical, interpretable, and mathematically tractable.

## A. Closed-Form Symbolic Representation

Classical CAD systems rely predominantly on explicit boundary representations such as B-Reps and NURBS [31], [29]. These representations provide intuitive geometric control and have become the industrial standard for engineering design. However, many geometric operations, including intersection calculations, trimming, and Boolean evaluation, require sophisticated numerical procedures and topology management algorithms. More recently, neural implicit approaches such as DeepSDF represent geometry through learned signed distance functions encoded within deep neural networks [28]. Although highly expressive, these representations embed geometric knowledge within high-dimensional weight spaces and therefore lack direct symbolic interpretability and editability. In contrast, the proposed framework adopts the Implicit Area Spline formulation introduced by Li and Tian [16]. Geometry is represented as the zero level set of a continuous scalar field

$$
F ( x , y ) = 0 ,\tag{1}
$$

constructed from a sparse control polygon

$$
V = \{ v _ { 1 } , v _ { 2 } , \ldots , v _ { n } \} .\tag{2}
$$

Unlike numerical approximations or learned implicit functions, the Area Spline is defined entirely through closed-form analytical expressions. The resulting representation is directly evaluable, differentiable, and symbolically manipulable. Furthermore, Li and Tian proved that the Piecewise Algebraic Spline framework supports arbitrary-order continuity, yielding implicit representations with guaranteed $C ^ { n }$ smoothness while remaining completely analytical [16]. Consequently, the Area Spline combines four properties that rarely coexist within a single geometric representation:

• closed-form analytical evaluation,

• arbitrary-order continuity $( C ^ { n } )$

• explicit geometric editability through control points,

• implicit topological robustness.

Within the proposed neuro-symbolic architecture, the control points, spline segments, and algebraic constraints constitute explicit symbolic entities that encode geometric knowledge. Geometric reasoning is therefore performed through analytical manipulation of symbolic expressions rather than adaptation of learned model parameters.

## B. Additivity and Symbolic Composition

The defining mathematical property of Area Splines is their strict additivity. Let a polygon $P$ be partitioned into a collection of sub-polygons

$$
P = \bigcup _ { i = 1 } ^ { m } P _ { i } .\tag{3}
$$

The Area Spline field satisfies

$$
F _ { P } ( x , y ) = \sum _ { i = 1 } ^ { m } F _ { P _ { i } } ( x , y ) .\tag{4}
$$

This property follows directly from the additive nature of the underlying area formulation [16], [6]. Shared internal boundaries contribute equal and opposite signed terms, causing their influence to cancel exactly. Consequently, only the external boundary contributes to the final field representation. From a geometric perspective, Eq. (4) implies that shape composition can be performed through simple analytical accumulation rather than explicit reconstruction of boundary topology. Unlike conventional boundary representations, the mathematical description of an object is invariant with respect to how it has been partitioned during modeling. This property is of particular significance because it introduces a compositional structure analogous to symbolic reasoning. Complex geometric configurations can be constructed by combining simpler analytical components while preserving mathematical consistency. The representation therefore supports compositional operations directly at the symbolic level, a capability widely regarded as a defining characteristic of human abstraction and reasoning [15], [9].

## C. Implications for Constructive Solid Geometry

Constructive Solid Geometry (CSG) traditionally represents objects through Boolean operations applied to explicit geometric primitives [31]. Although elegant in theory, practical implementations often require numerically intensive procedures for intersection detection, trimming, reparameterization, and topology maintenance. These operations become increasingly fragile as geometric complexity increases. The Area Spline formulation provides an alternative viewpoint. Since objects are represented as analytical scalar fields, geometric composition can be performed directly through field operations. Internal boundaries vanish automatically through symbolic cancellation, eliminating the need for many of the topology-repair procedures commonly required by explicit geometric representations. Consequently, Boolean composition is transformed from a topology-management problem into an algebraic operation on symbolic field expressions. This significantly reduces geometric complexity while preserving analytical consistency. The resulting representation remains smooth, continuous, and topologically coherent throughout the modeling process.

## D. Area Splines as a Neuro-Symbolic Representation

Within the broader cognitive architecture proposed in this paper, the Area Spline serves as the final symbolic representation layer. Topological abstraction extracts invariant structural descriptions from sensory observations, while the spline formulation projects these structures into an explicit mathematical domain. This process closely mirrors a fundamental aspect of mathematical cognition. Human mathematicians routinely transform imperfect observations into idealized symbolic objects that can be manipulated independently of the physical world. Lines become infinitely thin, curves become perfectly smooth, and geometric relationships become analytically exact. Similarly, the proposed system transforms noisy point-cloud observations into topological invariants and subsequently into closed-form algebraic representations. The resulting model is therefore more than a geometric reconstruction. It constitutes a symbolic mathematical hypothesis describing the underlying structure of the observed object. In this sense, the Area Spline provides the mechanism through which geometric perception is converted into mathematical representation, completing the transition from observation to abstraction envisioned by the proposed neuro-symbolic framework.

It is important to distinguish the proposed symbolic representation from conventional parametric spline representations such as Bezier curves, B-splines, or NURBS. Parametric´ splines describe geometric boundaries through a mapping

$$
\mathbf { C } ( t ) = ( x ( t ) , y ( t ) ) ,
$$

where geometry is recovered by traversing a parameter domain. In contrast, the Area Spline represents the object directly as an implicit field

$$
F ( x , y ) = 0 ,
$$

thereby encoding the spatial region itself rather than merely its boundary. This distinction is particularly significant for symbolic reasoning because spatial relationships, containment, composition, and topological structure become intrinsic properties of the representation. Furthermore, the additive property of Area Splines enables geometric composition to be expressed through algebraic field operations, providing a form of compositional symbolic representation that is generally not available in conventional parametric spline formulations.

## V. METHODOLOGY: A NEURO-SYMBOLIC PIPELINE FOR GEOMETRIC ABSTRACTION

The NeuSOGA architecture is designed as a computational model of mathematical abstraction. Rather than learning a direct mapping from observations to CAD parameters, the system progressively transforms observations into symbolic mathematical representations through four stages: topological abstraction, perceptual isolation, geometric abstraction, and symbolic synthesis. Formally, the architecture implements the transformation

$$
\mathcal { O } \to \mathcal { T } \to \mathcal { G } \to \mathcal { S } ,\tag{5}
$$

where

• O denotes observational data,

$\tau$ denotes topological abstractions,

• $\mathcal { G }$ denotes geometric abstractions,

• S denotes symbolic mathematical representations.

Unlike conventional learning-based Scan-to-CAD systems that encode geometric knowledge within neural network parameters, the proposed framework derives structural representations explicitly from topological and geometric principles before projecting them into an analytical symbolic representation.

## A. Stage 1: Topological Abstraction from Observation

Given an observed point cloud, the system first generates a two-dimensional projection onto a principal viewing plane. Let the resulting occupancy mask be represented as

$$
{ \mathcal { M } } ( x , y ) .\tag{6}
$$

To recover intrinsic structural information, an exact Euclidean Distance Transform (EDT) is computed:

$$
D ( p ) = \operatorname* { m i n } _ { q \in \partial \mathcal { M } } \| p - q \| _ { 2 } ,\tag{7}
$$

where ∂M denotes the boundary of the projected shape [25]. The local maxima of the distance field,

$$
P _ { \mathrm { c o r e } } = \{ p _ { 1 } , p _ { 2 } , . . . , p _ { n } \} ,\tag{8}
$$

correspond to centres of maximal inscribed regions and provide a topology-aware description of the object’s structural organization. Inspired by the principles underlying Blum’s Medial Axis Transform [4], these points function as topological primitives that summarize the object’s intrinsic spatial structure. Unlike conventional object detectors, these topological abstractions are derived analytically and require no taskspecific training.

## B. Stage 2: Topology-Guided Perceptual Isolation

The discovered structural cores are subsequently used to guide foundation-model perception. For each topological primitive p<sub>i</sub>, a segmentation is generated using Segment Anything (SAM/SAM2):

$$
\mathcal { P } _ { i } = \mathcal { F } _ { S A M } ( \mathcal { M } , p _ { i } ) ,\tag{9}
$$

where $\mathcal { F } _ { S A M }$ denotes the segmentation operator [13], [30]. This differs fundamentally from conventional prompting strategies. Rather than relying on user interaction or arbitrary seeds, segmentation prompts are generated automatically from topological analysis. Consequently, topology guides perception rather than perception discovering topology. The output of this stage is a collection of coherent object components

$$
\mathcal { P } = \{ \mathcal { P } _ { 1 } , \mathcal { P } _ { 2 } , \ldots , \mathcal { P } _ { m } \} ,\tag{10}
$$

which form the basis for subsequent geometric abstraction. From a neuro-symbolic perspective, this stage combines neural perception with deterministic geometric reasoning while maintaining a clear separation between the two processes.

## C. Stage 3: Adaptive Multi-Scale Geometric Abstraction

For each segmented component, the boundary contour is analysed across multiple scales using Gaussian scale-space theory [37], [17], [19]. Two complementary contour representations are generated:

$$
C _ { \mathrm { c o a r s e } } ( t ) = G _ { \sigma _ { c } } * C ( t ) ,\tag{11}
$$

$$
C _ { \mathrm { f i n e } } ( t ) = G _ { \sigma _ { f } } * C ( t ) ,\tag{12}
$$

where $\sigma _ { c } = 6 . 0$ and $\sigma _ { f } = 1 . 5$ in the current implementation. The coarse representation captures the dominant global structure of the object, while the fine representation retains local geometric detail. Their discrepancy is measured as

$$
\begin{array} { r } { \epsilon ( t ) = \| C _ { \mathrm { c o a r s e } } ( t ) - C _ { \mathrm { f i n e } } ( t ) \| _ { 2 } . } \end{array}\tag{13}
$$

Regions exhibiting large deviations indicate locations where significant geometric detail would be lost through excessive smoothing. An importance mask is therefore defined as

$$
M ( t ) = { \left\{ \begin{array} { l l } { 1 , } & { \epsilon ( t ) > \tau , } \\ { 0 , } & { { \mathrm { o t h e r w i s e } } , } \end{array} \right. }\tag{14}
$$

where τ is a deviation threshold. To ensure spatial coherence, the binary importance mask is smoothed using a Gaussian kernel, producing a continuous blending function

$$
W ( t ) = G _ { \sigma _ { w } } * M ( t ) .\tag{15}
$$

The final hybrid contour is constructed by adaptive feature fusion:

$$
C _ { \mathrm { h y b r i d } } ( t ) = ( 1 - W ( t ) ) C _ { \mathrm { c o a r s e } } ( t ) + W ( t ) C _ { \mathrm { f i n e } } ( t ) .\tag{16}
$$

Consequently, coarse-scale geometry is preserved throughout most of the contour, while fine-scale structure is selectively reintroduced only in regions containing significant geometric detail. This produces a contour that simultaneously preserves global smoothness and local fidelity. Finally, the hybrid contour is compressed using a Douglas-Peucker polygonal approximation [7],

$$
V = \{ v _ { 1 } , v _ { 2 } , \ldots , v _ { k } \} ,\tag{17}
$$

yielding a sparse control polygon that serves as the symbolic geometric abstraction of the original observation.

## D. Stage 4: Symbolic Mathematical Synthesis

The final stage projects geometric abstractions into an explicit symbolic mathematical representation. For each control polygon $V _ { i } ,$ , an implicit Area Spline representation is generated according to the Piecewise Algebraic Spline formulation of Li and Tian [16]. The resulting geometry is represented analytically as

$$
F _ { i } ( x , y ) = 0 .\tag{18}
$$

The complete object is then synthesized through additive field composition

$$
F ( x , y ) = \sum _ { i = 1 } ^ { m } F _ { i } ( x , y ) .\tag{19}
$$

Because the Area Spline formulation satisfies strict additivity, composite shapes can be constructed through algebraic field accumulation without explicit boundary merging or topological repair [16]. Unlike neural implicit representations, the resulting symbolic representation possesses explicit mathematical structure, arbitrary-order smoothness $( C ^ { n }$ continuity), exact analytical evaluation, and direct editability through control points. The output of the pipeline is therefore not merely a reconstructed shape but an explicit symbolic mathematical model of the observed object.

## E. Interpretation as Mathematical Abstraction

The complete architecture implements a hierarchy of abstraction:

$$
\mathrm { O b s e r v a t i o n }  \mathrm { T o p o l o g y }  \mathrm { G e o m e t r y }  \mathrm { S y m b o l } .\tag{20}
$$

Raw observations are transformed into topology-aware structural cores, topological structures guide neural perception, perceptual components are compressed into geometric primitives, and geometric primitives are projected into analytical symbolic representations through Area Splines. The central hypothesis of this work is that symbolic representation constitutes the critical bridge between perception and mathematical reasoning. The present framework therefore focuses on the transformation

$$
\mathcal { O } \to \mathcal { T } \to \mathcal { G } \to \mathcal { S } ,
$$

establishing a neuro-symbolic mechanism by which observations are converted into explicit mathematical representations suitable for subsequent reasoning, editing, and knowledge generation.

## F. Implicit Spline Evaluation

The extracted control points are ordered counter-clockwise via the Shoelace Formula [6]. Driven by mathematical additivity [16], the overlapping semantic parts are arithmetically summed into a single, flawless Boolean CSG union $\boldsymbol { \mathcal { U } } ( \boldsymbol { x } , \boldsymbol { y } ) ;$

$$
\mathcal { U } ( x , y ) = \sum _ { k = 1 } ^ { K } F _ { k } ( x , y )\tag{21}
$$

## VI. SYMBOLIC REPRESENTATION AS THE OUTCOME OFABSTRACTION

The central hypothesis of this work is that symbolic representation constitutes the essential bridge between perception and mathematical reasoning. Modern neural systems are highly effective at extracting information from sensory data, but their knowledge is typically encoded within latent parameters that are not directly accessible to analytical manipulation. In contrast, human mathematical reasoning depends upon explicit symbolic objects such as equations, functions, geometric constructions, and logical statements. Such representations possess semantics that are independent of the observations from which they were derived and can therefore support reasoning, generalization, and knowledge composition. The proposed architecture is designed explicitly to perform the transformation

$$
O \to T \to G \to S ,
$$

where observations are converted into topological invariants, topological invariants into geometric abstractions, and geometric abstractions into symbolic mathematical representations. Within this framework, the Implicit Area Spline representation constitutes the symbolic layer. Given a set of control points

$$
V = \{ v _ { 1 } , v _ { 2 } , \ldots , v _ { n } \} ,
$$

the geometry is represented analytically by a scalar field

$$
F ( x , y ) = 0 .
$$

Unlike latent neural representations, the resulting symbolic object possesses explicit mathematical structure, supports closedform evaluation, exhibits arbitrary-order continuity, and satisfies additive composition properties [16]. The output of the proposed system is therefore not merely a geometric reconstruction but an explicit mathematical model whose structure is available for subsequent symbolic manipulation and reasoning.

## VII. EXPERIMENTAL EVALUATION

The objective of the present experiments is not to benchmark reconstruction accuracy against large-scale learningbased Scan-to-CAD systems. Instead, the experiments are designed to evaluate the central hypothesis of this work: that a neuro-symbolic pipeline can transform raw observations into explicit symbolic mathematical representations without task-specific geometric training. Specifically, we evaluate the proposed transformation

$$
\mathcal { O } \to \mathcal { T } \to \mathcal { G } \to \mathcal { S } ,
$$

where observations are converted into topological abstractions, geometric abstractions, and finally symbolic mathematical representations.

## A. Experimental Setup

Experiments were conducted using point-cloud models from the ModelNet40 dataset, a widely used benchmark for geometric learning and shape analysis. Three representative object categories were selected to evaluate the robustness of the proposed framework across substantially different structural configurations:

• Airplane,

• Chair,

• Lamp.

Each object was normalized and projected onto three principal orthographic views:

• Top view (XY plane),

• Front view (XZ plane),

• Side view (YZ plane).

For each projection, the complete neuro-symbolic pipeline was executed:

Projection → EDT Core Discovery → SAM Segmentation

$$
 \mathrm { A d a p t i v e \ A b s t r a c t i o n }  \mathrm { A r e a \ S p l i n e \ S y n t h e s i s }
$$

No category-specific training, CAD templates, geometric priors, or reconstruction supervision were used.

## B. Evaluation Criteria

Because the proposed framework is designed to generate symbolic representations rather than merely geometric reconstructions, evaluation focuses on four qualitative criteria:

1) Structural preservation,

2) Topological consistency,

3) Representation compactness,

4) Symbolic expressiveness.

Structural preservation evaluates whether major geometric structures remain recognizable following abstraction. Topological consistency evaluates whether the generated implicit representation preserves the connectivity and organization of the original object. Representation compactness evaluates the degree to which complex boundaries can be represented using sparse control polygons. Symbolic expressiveness evaluates whether the final output constitutes an explicit mathematical representation rather than a numerical approximation or latent encoding.

![](images/4a5bd42adc6fa21acba90b7649d45c3ff2f66f92592b592a38dd24c8ed59a2d4.jpg)

![](images/2ae6eea623aaa6cb1f6dffb4e84c0f4bb3e9c671c08396bfd33939875d39715d.jpg)

![](images/0ca555d48d8142dca4e43dc5f18da03fe2dca59601261b4e3a6bc9f197e6d701.jpg)

![](images/6b44332c2ba65f12b0aeec5aaa5eb8fe24b651b6763863268a7494fe18ef2687.jpg)

![](images/12e7e6b84b83d425528eddce0643aa1f0489af6fbc6c41874a50d5a5e483faf2.jpg)

![](images/3e9350543fb4867dbdc42975ef531c2c44ff4d11f3ca0ff487106a447535f977.jpg)

![](images/6d5f3206527d4bddac67fedff86765bc99da89d9bf157a8ed59c01ea032027bb.jpg)

![](images/649f336cae60de97216a30111c485bc981325bd5423d75e659512da0f7a5ba3f.jpg)

![](images/daa7e2049aed567b3a8c4774d84706a9910c17405f9359fb27982d4b44a1a256.jpg)

![](images/b97ec8381254df7118037cc93d65c732c204fe0f1ac83a37c50b24265e46da52.jpg)

![](images/e304de5a7a6ac07418711dfdb0119b558e0856ec2d75ae6122e61e021cfcad53.jpg)

![](images/00f508269c28312f4e8a18e70c227055c80ef1ca159614f9c57eb39c86c814cd.jpg)

![](images/b62582fcc8afce6c00e3df798102e98cff697709fa3470c566ca7f881d08f03a.jpg)

![](images/5c1294d66566c983bd5538f87bc5be70274e8b2d308235bb8b329e6870bbf611.jpg)

![](images/b8e02654d9cf0e68fd5011256f535e46f759a8ed36a5f405a8f69a4674a71dca.jpg)

![](images/d5ad1e17df175c66d8dda1fb6a6a80d44109b6e17a5b7f1267d1a4be985bacde.jpg)

![](images/47c71629b6b9a90b2cfeda6d2b7c31eb12565c19dd8495a67576cbf467d49efb.jpg)

![](images/6cbb70182a9aa04606d8ebdabd3f96e41f421fdf95c2d4a247a89714db118871.jpg)

![](images/28ba6b1c7fb7b2ca602ed68af3131f1481a0ece10a2ad14cc9539b4f113efc34.jpg)

![](images/815f20af8de6c5d91e307da5144d89faa3f228cb239f08abdb1fe21579c13802.jpg)

![](images/b87cc27ca2cf1d961e4d7c4f7c92325525a383918f9c45d112a09f94faa64e9f.jpg)

![](images/d951ca7d0bf5b7491ccedc5a46886933fd94ade17d4f11a3eb4418f799203270.jpg)

![](images/5eb9b3cc1105721d3c16cb3e2d9bfff0d5aae90fad066f5db48e266d45cf571e.jpg)

![](images/7891e152660fd254dc27f25e1a87e400726bef32e75a3ac9d5fe2a20807c9691.jpg)  
Fig. 2. Application of NeuSOGA to an airplane point-cloud model. Rows correspond to the top (XY), front (XZ), and side (YZ) projections of the original 3D object. For each projection, the proposed neuro-symbolic framework progressively transforms the observation into topology-aware structural abstractions, segmented object regions, adaptive geometric abstractions, and finally an explicit symbolic mathematical representation using Implicit Area Splines. The resulting representations preserve the dominant aircraft morphology while significantly reducing geometric complexity, demonstrating the complete transformation from observation to symbol $( { \mathcal { O } } \to { \mathcal { T } } \to { \mathcal { G } } \to S )$

## C. Airplane Reconstruction

Figure 2 presents the results obtained for the airplane model. The topological abstraction stage successfully identified the major structural components of the aircraft despite substantial variations between viewing directions. The adaptive coarse-tofine abstraction mechanism preserved high-curvature features such as wing tips, stabilizers, and engine structures while simultaneously recovering the dominant fuselage geometry. The resulting polygonal abstraction required between 97 and 132 control points depending upon projection direction, representing a substantial reduction in geometric complexity relative to the original point cloud. Despite this reduction, the resulting Area Spline representation accurately preserved the characteristic aircraft morphology. Particularly noteworthy is the preservation of global symmetry in both the top and front views. The symbolic representation reconstructed through the Area Spline formulation produces a smooth analytical boundary while retaining fine-scale structural details introduced through adaptive scale-space refinement.

![](images/01be16820c629bc4e33bd4c8e38b7a6c46c3db6fe484a17771a954f548819a02.jpg)

![](images/7fcc9c691e1f300b8ddfd4bce090f78f68d2a74cfb4c05b8ef292f66081a432d.jpg)

![](images/899b28e8ccb11d3624a5cb33d04d6d9ab643036ddfaa4b4408e3d09ec60d2127.jpg)

![](images/f9ed29db6438b0d01648d4a15128a42a9afb66971d06c1b99aaec57e1ed59b40.jpg)

![](images/a2af22eb60fad7378a7dea74ba0094bb1dbdb1dbd56c7dcef74843de2117bedd.jpg)

![](images/b2752c4d72dd229b6e2c6dfc5e31fcd69d72c04fa633c450a787e90d373cbf2a.jpg)

![](images/3d41a966cc0e470584d33a0d99a03117d3c47fb43d154bf82f32f632dbcc2017.jpg)

![](images/f67abed5ea20c9f64235b84cba866c1cffc66e65fb8d04a4c6f945e5cb66fea4.jpg)

![](images/3745c1848168fa76aa388a77b6c356c5c26a6cf664aa0f65b7613b36a9b6d0ab.jpg)

![](images/4960b61cd2e6a441eee8392f14b27307aad1c59e9931e3f974955d23f38d0e1e.jpg)

![](images/ddd4a5110e747c51149d9d8de115c29b572f5487c58d142706f111a2a36c459e.jpg)

![](images/af5a7cf1601d51029bdc56fba6c7d1a05fca42292b2b3d0ce96228a985b8f8a9.jpg)

![](images/4ba7ca70fc9b82cd8236dc72c4f34cc418953ef6417fa3f6291a22dc8ddffe6d.jpg)

![](images/eb9f56a609c9b32f173205fafef4b22d6c91e143c73bf9e32e18f07192f22aff.jpg)

![](images/279893f4aa742bab10fb5320ca93ae6a9ac2bd14ddab66b226feb35ed9e7fed4.jpg)

![](images/a7bdb61c42318a5f06b14e6b72154ca387cc9cc65ec5c3c283deb9a158d71f9b.jpg)

![](images/1d14e561ca384bc8a6c3b67646e262f6a89ee877fb1cd5692db3b2b060ae8bbe.jpg)

![](images/3eaec0af507278eb07560c19f376916f428f2301dd125eb5149f5b45876f2927.jpg)

![](images/d149afa4d1e6017f87939097e1dcdf051574b7597cc839490f6ab7f679185f18.jpg)

![](images/24262cc7871c3674f26213b2d256ea7a9205b3d3af075f150dcc44715a5ba533.jpg)

![](images/4fd55dafe1e2d5032fed1913dcd16fe5d9a4cc95773971233a9faf1980e66216.jpg)

![](images/6beb6f07234841930ff44acc8c463d6082465155a3736655561685bbb1dc4c23.jpg)

![](images/cf7a1af914fb9aad99ae198d0a6eede74663c70880c2c401b3efaccd4a6ace8b.jpg)

![](images/9076d02da1c47e4c01d1ac323c1a8efb93b55e7ff6a8f9b6ff9266feabab1e21.jpg)  
Fig. 3. Application of NeuSOGA to an chair point-cloud model. Rows correspond to the top (XY), front (XZ), and side (YZ) projections. Despite the presence of sharp corners, support structures, and complex connectivity, NeuSOGA successfully extracts topology-aware primitives, performs adaptive geometric abstraction, and synthesizes explicit symbolic representations through Implicit Area Splines. The resulting symbolic models preserve the essential seat, backrest, and support-leg structures while providing a compact analytical representation.

## D. Chair Reconstruction

Figure 3 illustrates results for a structurally different object category. Unlike the airplane, the chair contains pronounced rectilinear structures, sharp transitions, and disconnected geometric support elements. These characteristics represent a challenging test for smoothing-based reconstruction methods. The proposed abstraction process successfully preserved the dominant seat, backrest, and support-leg structures. Although substantial geometric simplification was performed, the resulting symbolic representations maintain clear correspondence with the original object topology. The side-view reconstruction is particularly significant because it demonstrates that the adaptive abstraction mechanism preserves engineering-style geometric features without excessive smoothing. The resulting symbolic representation remains interpretable and structurally faithful despite substantial reduction in boundary complexity.

## E. Lamp Reconstruction

Figure 4 presents results for the lamp model. This example exhibits strong rotational symmetry and smooth curved surfaces, providing a complementary test case to the highly articulated airplane and chair geometries. Across all three projections, the proposed framework reconstructed clean implicit boundaries that faithfully captured the dominant geometric features of the lamp. The hemispherical shade, cylindrical support structure, and circular base were all preserved through the abstraction process. The front projection is particularly illustrative because the generated Area Spline representation recovers a nearly perfect circular profile using a sparse symbolic representation. This demonstrates the ability of the framework to transform noisy point-cloud observations into smooth analytical geometric descriptions.

![](images/3ed37b69d09b5a899eaf935634b6289aeab984381906e906636ac858cff9a939.jpg)

![](images/19f5d85bb843b04c942c28655a19b56eb295d256cd786cd0654e7e77e3247a56.jpg)

![](images/d26ade526231fcc049a38c6aa11d808be390e9f9d1f95fd917c1301b5b022524.jpg)

![](images/f4ad1e9290c6b743aa435b672004c90670770f820b200b36042aa44ad760ad47.jpg)

![](images/3429afd7b69b0513ec355b4e880be943f2fc5044cc86e3ca12ffaff2174ecdc5.jpg)

![](images/f180425ccb130310ddb27e61ccb445ea71dbd35d6cda72f1f7718c235eed19fc.jpg)

![](images/bd5a8e6e0241e363edf2686dd272ab274d9b8aafe86cf8ed9684a502aa3ab609.jpg)

![](images/72ca04d4f2843859e9c51b6a5ff849eabc0badca1779481820480b7ac8bc38b4.jpg)

![](images/7780fbfb7d7eaae46e178b922de778f3bcec79e871d09854435c2f341ec3f223.jpg)

![](images/4ec5a7fa8630f17656673124b67a6178ebb175a4c02310a0a1e01f69a3a8b953.jpg)

![](images/4f5ebd9d63715b4ff8905d79b786021f5bd00ff09d9334bbda325a9e8bf85c51.jpg)

![](images/725adc97e70f289915ba868e3c207b8e9585b69b59229ff8e04c5035ea050f24.jpg)

![](images/34af31f672ca06674573e9124ae4af2a2e16d807ecf40b4ffbffed4dee4e8cf4.jpg)

![](images/40455b16e5e61cec7448739b6d49355a4fb81876dd7a01c01dc812bedb7ab745.jpg)

![](images/4f17fcb9d021632c6891bd8b7b02e4ed365b6fe08ed6e1b9ddb38c909e656b3b.jpg)

![](images/5b6cada84c481679478c51e62f1284066dd99e6b1c7a5ab1e10206844396f437.jpg)

![](images/e695fa80cbfd06c57b31637dab8132934afb9805f6a41eb274f974a8424ee9cc.jpg)

![](images/32f916c90473c20358e4be12dd095327e691953620e1362b05d60b0e09761158.jpg)

![](images/952145387352c181575407fbfeeeafd20ba59d1bd1cb9a9bd8717df56d2e6ee4.jpg)

![](images/dfd1d53fb249ccbdeac54ee6b16627ef12847ceb4d4eaab0db722d4cd6329fdb.jpg)

![](images/55eab817186f165a935d1336cf1a80fbcc2b3b43c9cb3db8adaf8222933c3f3b.jpg)

![](images/859860ce2c3e80f8fc7721a8f34f331f13796c04d6f7dcb728a2f6d3792991b7.jpg)

![](images/85576bad5ea786f21265e4df4e00bc5ff2840904b72cb9ee60d0a9f3f88d3231.jpg)

![](images/09325ec28bf027dc899ab9cdbf42baf4c0dd514da52e4305cb54e4f9fd092ca5.jpg)  
Fig. 4. Application of NeuSOGA to an lamp point-cloud model. Rows correspond to the top (XY), front (XZ), and side (YZ) projections. The proposed pipeline identifies topology-aware structural cores, isolates coherent object regions through foundation-model segmentation, and generates smooth symbolic representations using adaptive abstraction and Implicit Area Splines. The resulting analytical models accurately capture both the global shape and curved geometric features of the lamp, illustrating the robustness of the observation-to-symbol transformation across different object categories.

## F. Robustness Across Modalities and Viewing Directions

To evaluate the generality of the proposed neuro-symbolic architecture, we investigated its behaviour across both different sensing modalities and different viewing conditions. Specifically, we performed two complementary robustness studies. First, we applied the complete $O \to T \to G \to S$ pipeline to segmented binary observations extracted from the COCO 2017 dataset. Second, we evaluated the framework using non-canonical projections of ModelNet40 point-cloud objects generated along an arbitrary viewing direction (1, 1, 1). Together, these experiments assess whether the abstraction process depends upon a particular input modality or viewing configuration.

A central objective of the proposed neuro-symbolic architecture is to perform abstraction independently of the sensing modality used to generate the observation. To evaluate this property, we extended the experimental analysis beyond projected point clouds and applied the complete $O  T $ $G  S$ pipeline to segmented optical observations. Specifically, binary object masks were extracted from the COCO 2017 dataset and injected directly into the Observation layer (O). These masks contain substantially different characteristics from the projected point-cloud observations used in previous experiments. While point clouds are sparse and incomplete, binary masks provide dense occupancy information with sharply defined boundaries. Successful processing of both modalities would therefore indicate that the architecture is responding to abstract spatial structure rather than modality-specific characteristics. Unlike conventional perception pipelines that often require separate processing pathways for different data sources, the proposed framework applies the same abstraction sequence regardless of the underlying representation:

$$
\mathcal { O } \to \mathcal { T } \to \mathcal { G } \to \mathcal { S } .
$$

No modality-specific modifications were introduced during evaluation. For binary observations, the Euclidean Distance Transform first generates a spatial distance field from which topology-aware structural cores and internal void nodes are extracted. These topological abstractions subsequently guide foundation-model perception and contour isolation. Adaptive multi-scale abstraction then converts the resulting contours into sparse geometric control polygons, which are finally synthesized into symbolic mathematical representations through the Area Spline formulation. An interesting observation emerges when applying the framework to clean binary masks. Because the object has already been segmented, the foundation-model layer frequently behaves as a nearidentity mapping. While this may appear computationally redundant in idealized settings, retaining the perceptual layer provides several important advantages. First, it preserves a modality-independent architecture. Point clouds, segmented images, occupancy grids, medical images, and other spatial observations can be processed using the same computational pathway without introducing heuristic branching logic. Second, the perceptual layer provides a mechanism for handling imperfect observations. Real-world binary masks often contain segmentation artifacts, disconnected regions, missing boundaries, aliasing effects, and spurious noise. Foundation-model perception offers an additional abstraction layer capable of improving structural consistency before symbolic processing. Third, topology-guided prompting introduces a level of semantic awareness that is not available through conventional contour extraction alone. Multiple structural cores and internal void nodes can be used to distinguish disconnected regions, preserve internal cavities, and guide segmentation of complex overlapping structures. Figure 5 illustrates representative results obtained from bicycle and chair objects containing multiple internal voids and thin structural features. Despite substantial topological complexity, NeuSOGA successfully identifies both positive structural regions and negative interior spaces. The resulting control polygons capture the dominant geometric organization of the objects, while the symbolic Area Spline representation preserves fine structural details such as bicycle-frame openings, wheel cavities, and chair support gaps. Most importantly, the experiments demonstrate that the symbolic representation layer remains unchanged across modalities. Whether the input originates from a projected point cloud or a segmented optical observation, the final output consists of an explicit analytical representation

$$
F ( x , y ) = 0 ,
$$

possessing arbitrary-order smoothness, additive composition, and direct symbolic interpretability. This result provides evidence that the proposed architecture is learning neither object categories nor sensing modalities. Rather, it operates on topological and geometric abstractions that remain invariant across different forms of observation. Consequently, the experiments support the broader hypothesis that symbolic representation formation can be achieved through a modality-independent abstraction process, reinforcing the role of the proposed architecture as a general neuro-symbolic framework rather than a domain-specific reconstruction pipeline.

In addition to modality generalization, we investigated viewpoint robustness using arbitrary projections of threedimensional point-cloud models from the ModelNet40 dataset. Whereas previous experiments focused on canonical orthographic views (top, front, and side), the additional evaluation employed projections along the (1, 1, 1) viewing direction. This configuration simultaneously exposes multiple object surfaces and typically produces more complex contour structures than standard orthographic projections. The resulting observations represent a particularly challenging test case because self-occlusion, overlapping structural components, and non-axis-aligned features generate significantly more intricate topological and geometric configurations. Despite these challenges, NeuSOGA successfully identified topology-aware structural cores, extracted meaningful geometric abstractions, and synthesized continuous Area Spline representations. The resulting symbolic models preserved the dominant structural organisation of the original objects while maintaining compact analytical descriptions. These experiments demonstrate that the transformation

$$
\mathcal { O } \to \mathcal { T } \to \mathcal { G } \to \mathcal { S }
$$

is not dependent upon canonical viewpoints. Instead, the abstraction process operates upon structural and topological information that remains largely invariant across viewing directions, further supporting the hypothesis that the proposed architecture learns abstract geometric organisation rather than view-specific representations.

The significance of these results extends beyond the recovery of geometric shape. The proposed framework demonstrates a complete transformation from observation to symbol, converting raw sensory measurements into explicit mathematical representations. Viewed from this perspective, the central contribution is not simply reconstruction accuracy, but the formation of symbolic structures from perceptual data.

![](images/b95d5a186f366932fb18d3de437b0d5b1327d8bb7d30d3025b9fc8fdd9ac3316.jpg)  
(b) Cycle  
Fig. 5. Robustness evaluation of the proposed neuro-symbolic abstraction framework on segmented optical observations from the COCO dataset. The upper two rows correspond to a chair object, while the lower two rows correspond to a bicycle object. For each object, the eight columns illustrate the complete observation-to-symbol transformation $( { \mathcal { O } } \to { \mathcal { T } } \to { \mathcal { G } } \to S )$ from left to right: (1) segmented binary observation (O), (2) Euclidean Distance Transform, (3) topology-aware structural cores and internal void nodes (T), (4) topology-guided foundation-model segmentation, (5) adaptive multi-scale contour extraction, (6) sparse geometric control polygon (G), (7) Implicit Area Spline field representation, and (8) the final symbolic mathematical boundary F(x, y) = 0 (S). The results demonstrate that the proposed architecture preserves complex topological structures, internal voids, and thin mechanical features while generating explicit symbolic representations directly from segmented optical observations.

This raises two fundamental questions: why are symbolic representations essential for abstraction, and how can they provide a foundation for explainable machine intelligence? The following sections explore these questions from both cognitive and computational perspectives.

## G. Why Symbolic Expressiveness in 2D Matters

At first glance, the use of two-dimensional symbolic representations may appear to be a simplification of the broader three-dimensional reconstruction problem. However, from the perspective of both cognitive science and computer vision, symbolic representation in 2D occupies a fundamentally important position within the abstraction hierarchy. Marr’s influential theory of vision proposes that visual understanding proceeds through a sequence of intermediate representations beginning with a two-dimensional primal sketch, followed by a 2.5D sketch, and ultimately a full three-dimensional model [24]. Within this framework, geometric understanding does not emerge directly from sensory measurements. Instead, the visual system first extracts structured symbolic descriptions of contours, boundaries, edges, and local geometric organization before constructing higher-level spatial representations.

Similarly, Biederman’s Recognition-by-Components theory argues that human object recognition relies upon structural information recoverable from two-dimensional contour configurations [3]. Curvature, symmetry, collinearity, concavity, and boundary relationships provide sufficient information for robust human recognition even when images are degraded, incomplete, or viewed from novel orientations. These findings suggest that contour-level symbolic abstractions are not merely low-level visual features but constitute essential cognitive representations supporting higher-level reasoning. The proposed framework follows a similar philosophy. Rather than attempting direct reconstruction of complete three-dimensional CAD models, the system first transforms observations into explicit two-dimensional symbolic representations through topological abstraction and Area Spline synthesis. The resulting implicit functions

$$
F ( x , y ) = 0 
$$

are mathematically explicit objects possessing semantic structure, analytical properties, and compositional behaviour. Unlike latent neural representations, these symbols can be directly interpreted, manipulated, and reasoned about. From this perspective, the objective of the current experiments is not simply shape reconstruction. Instead, the experiments validate the ability of the proposed architecture to perform the transformation

$$
\mathcal { O } \to \mathcal { T } \to \mathcal { G } \to \mathcal { S } ,
$$

where symbolic representations emerge directly from observations. The successful generation of compact Area Spline representations therefore demonstrates the formation of explicit geometric symbols, which may subsequently serve as the foundation for higher-level reasoning, concept formation, and symbolic manipulation.

It is important to note that the symbolic significance of the resulting representations does not arise solely from their ability to reconstruct object boundaries. Conventional parametric spline formulations can also represent smooth geometric contours. However, the objective of the proposed framework is not merely boundary recovery but symbolic representation formation. By representing geometry as an analytical implicit field

$$
F ( x , y ) = 0 ,
$$

the Area Spline formulation encodes spatial regions, topological relationships, and compositional structure within a single mathematical representation. Combined with its additive field formulation, this allows geometric entities to be composed and manipulated symbolically through algebraic operations, making the representation particularly suitable as the symbolic layer of the proposed neuro-symbolic architecture.

## H. Explainability Through Symbolic Representation

Unlike conventional deep learning systems, whose internal representations are largely encoded within latent model parameters, the proposed architecture maintains explicit representations throughout the abstraction hierarchy. Topological primitives are observable through the EDT field, geometric abstractions are represented by sparse control polygons, and symbolic representations are expressed through closed-form analytical Area Spline formulations. As a result, the complete transformation

$$
\mathcal { O } \to \mathcal { T } \to \mathcal { G } \to \mathcal { S }
$$

remains fully interpretable. This transparency provides a form of intrinsic explainability rather than post-hoc explanation. The resulting symbolic representations can be inspected, edited, differentiated, composed, and mathematically analysed without reference to hidden neural parameters. Consequently, the proposed framework suggests a pathway toward explainable geometric intelligence through symbolic representation formation.

## I. Discussion

The experimental results demonstrate the feasibility of the proposed neuro-symbolic abstraction framework NeuSOGA across multiple object classes and viewing directions. Several observations are particularly important. First, the framework operates without task-specific geometric training. Structural understanding emerges through topological abstraction rather than through statistical learning of object categories. Second, topology-guided prompting consistently produces coherent object segmentations that are subsequently refined through deterministic geometric analysis. Third, adaptive multi-scale abstraction successfully balances global smoothness and local geometric detail. The resulting control polygons remain compact while preserving visually significant structural features. Most importantly, the final output of the pipeline is an explicit symbolic mathematical representation. Unlike neural implicit representations, whose geometry is encoded within network weights, the proposed approach generates analytical Area Spline fields possessing exact mathematical structure, arbitrary-order smoothness, and additive compositional properties. These results therefore support the central hypothesis of this work: that observations can be transformed into symbolic mathematical representations through a neurosymbolic pipeline that combines topology-driven abstraction, foundation-model perception, geometric simplification, and analytical spline synthesis.

The experiments also highlight the importance of the chosen symbolic representation. While alternative representations such as Bezier curves, B-splines, or NURBS could be fitted to´ the extracted contours, these formulations primarily describe boundary parameterizations. In contrast, the proposed Area Spline formulation represents spatial regions directly through an analytical implicit field and supports additive symbolic composition. Consequently, the final representation acts not only as a geometric model but also as an explicit symbolic object suitable for subsequent reasoning and manipulation.

## J. Limitations and Future Directions

The experiments presented in this work demonstrate the feasibility of transforming observations into symbolic mathematical representations through the proposed neuro-symbolic abstraction pipeline. However, the current framework focuses primarily on symbolic representation formation and therefore addresses only one stage of a broader cognitive abstraction process. Within the proposed architecture, the Area Spline formulation serves as the symbolic endpoint of the transformation

$$
\mathcal { O } \to \mathcal { T } \to \mathcal { G } \to \mathcal { S } .
$$

The resulting representations are analytical, interpretable, and compositional, but the current system does not yet perform higher-level reasoning directly on these symbols. In particular, automatic discovery of geometric properties, structural invariants, symbolic relations, and conceptual hierarchies remains an open problem. Future work will therefore investigate mechanisms operating on top of the symbolic representation layer. Potential directions include automatic symmetry discovery, repeated motif identification, structural decomposition, shape grammar formation, and symbolic concept learning from collections of geometric observations. Such capabilities would extend the present framework from symbolic representation formation toward the broader objectives of concept formation and machine-assisted mathematical reasoning. A second limitation concerns experimental scale. Although the current results demonstrate robustness across multiple object categories, viewing directions, and sensing modalities, future evaluation should investigate larger and more diverse datasets. Particular attention will be given to assessing the stability of symbolic representations under changes in observation quality, sensing modality, occlusion, and geometric complexity. More broadly, the present work establishes a symbolic representation layer for geometric intelligence. Future research will explore how such representations may serve as the foundation for explainable reasoning systems capable of constructing increasingly abstract geometric knowledge from observations.

## VIII. CONCLUSION

This paper introduced NeuSOGA, a neuro-symbolic framework for transforming observations into explicit symbolic mathematical representations. Inspired by theories of human abstraction and symbolic reasoning, the proposed architecture implements a hierarchy of representation formation

$$
\mathcal { O } \to \mathcal { T } \to \mathcal { G } \to \mathcal { S } ,
$$

where observations are progressively transformed into topological abstractions, geometric abstractions, and ultimately symbolic mathematical representations. Unlike conventional learning-based reconstruction systems that encode geometric knowledge within latent neural parameters, the proposed framework separates perception, abstraction, and symbolic representation into distinct computational stages. Topologyaware structural primitives are first extracted through Euclidean Distance Transforms and subsequently used to guide foundation-model perception. Adaptive multi-scale geometric abstraction then converts dense observations into sparse structural representations, which are finally projected into analytical symbolic representations through the Implicit Area Spline formulation. The experimental results demonstrated that this abstraction process remains effective across multiple object categories, sensing modalities, and viewing directions.

Evaluations on ModelNet40 point-cloud data, arbitrary-view projections, and segmented optical observations confirmed that the same computational architecture can consistently generate compact symbolic representations while preserving essential geometric and topological structure. These findings suggest that the proposed framework operates on abstract spatial organization rather than modality-specific characteristics. A central contribution of this work is the identification of the Implicit Area Spline as a symbolic representation layer. Unlike conventional parametric boundary representations, the Area Spline formulation represents spatial regions directly through analytical implicit fields, supports arbitrary-order smoothness, and admits additive symbolic composition. The resulting representation is not merely a reconstructed shape but an explicit mathematical object that can be inspected, manipulated, and analysed without reference to hidden neural parameters. From a broader perspective, the proposed framework contributes to Neuro-Symbolic AI by demonstrating a pathway through which symbolic representations can emerge from perceptual observations. Because each stage of the abstraction hierarchy remains observable and interpretable, the architecture also provides a form of explainability by construction, in contrast to the largely latent representations employed by modern deep learning systems. The present work focuses on symbolic representation formation as a fundamental prerequisite for higher levels of machine intelligence. Future research will investigate mechanisms capable of operating directly on the generated symbolic representations to discover geometric properties, structural invariants, compositional relationships, and higherlevel concepts. From this perspective, the proposed framework establishes a symbolic foundation for future systems that move beyond perception toward explainable reasoning, concept formation, and machine-assisted mathematical discovery.

## REFERENCES

[1] Sebastian Bader and Pascal Hitzler. Dimensions of neural-symbolic integration: A structured survey. arXiv preprint cs/0511042, 2005.

[2] Tarek R. Besold, Artur d’Avila Garcez, Sebastian Bader, Howard Bowman, Pedro Domingos, Pascal Hitzler, Kai-Uwe Kuhnberger, Lu¨ ´ıs C. Lamb, Daniel Lowd, Leo de Penning, Gadi Pinkas, Hoifung Poon, and Gerson Zaverucha. Neural-symbolic learning and reasoning: A survey and interpretation. arXiv preprint arXiv:1711.03902, 2017.

[3] Irving Biederman. Recognition-by-components: A theory of human image understanding. Psychological Review, 94(2):115–147, 1987.

[4] Harry Blum. A transformation for extracting new descriptors of shape. Models for the Perception of Speech and Visual Form, 19(5):362–380, 1967.

[5] Rishi Bommasani et al. On the opportunities and risks of foundation models. arXiv preprint arXiv:2108.07258, 2021.

[6] Bart Braden. The surveyor’s area formula. The College Mathematics Journal, 17(4):326–337, 1986.

[7] David H Douglas and Thomas K Peucker. Algorithms for the reduction of the number of points required to represent a digitized line or its caricature. Cartographica, 10(2):112–122, 1973.

[8] Artur d’Avila Garcez, Tarek R. Besold, Luc De Raedt, Peter Foldiak,¨ Pascal Hitzler, Thomas Icard, Kai-Uwe Kuhnberger, Lu¨ ´ıs C. Lamb, Risto Miikkulainen, and Daniel L. Silver. Neural-symbolic learning and reasoning: Contributions and challenges. In AAAI Spring Symposium Series, 2015.

[9] Artur d’Avila Garcez and Lu´ıs C Lamb. Neurosymbolic ai: The 3rd wave. Artificial Intelligence Review, 56(11):12387–12406, 2022.

[10] Artur S. d’Avila Garcez, Krysia Broda, and Dov M. Gabbay. Neural-Symbolic Learning Systems: Foundations and Applications. Springer, London, UK, 2002.

[11] David C Geary. From infancy to adulthood: The development of numerical abilities. European Child & Adolescent Psychiatry, 9:11–16, 2000.

[12] Muhammad Sami Khan, Emmanuel Dupont, Sk Aziz Ali, Kseniya Cherenkova, Anis Kacem, and Djamila Aouada. Cad-signet: Cad language inference from point clouds using layer-wise sketch instance guided attention. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4713–4722, 2024.

[13] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4015– 4026, 2023.

[14] Jan J. Koenderink. The structure of images. Biological Cybernetics, 50(5):363–370, 1984.

[15] Brenden M Lake, Tomer D Ullman, Joshua B Tenenbaum, and Samuel J Gershman. Building machines that learn and think like people. Behavioral and Brain Sciences, 40:e253, 2017.

[16] Qiang Li and Jie Tian. 2d piecewise algebraic splines for implicit modeling. ACM Transactions on Graphics (TOG), 28(2):1–19, 2009.

[17] Tony Lindeberg. Scale-space theory: A basic tool for analysing structures at different scales. Journal of Applied Statistics, 21(1-2):225–270, 1994.

[18] Tony Lindeberg. Feature detection with automatic scale selection. International Journal of Computer Vision, 30(2):79–116, 1998.

[19] Tony Lindeberg. Scale-Space Theory in Computer Vision. Kluwer Academic Publishers, 1998.

[20] Yu Liu, Anton Obukhov, Jan D Wegner, and Konrad Schindler. Point2cad: Reverse engineering cad models from 3d point clouds. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3763–3772, 2024.

[21] David G. Lowe. Distinctive image features from scale-invariant keypoints. International Journal of Computer Vision, 60(2):91–110, 2004.

[22] Dimitrios Mallis, Sk Aziz Ali, Emmanuel Dupont, Kseniya Cherenkova, Ali Sahin Karadeniz, Muhammad Sami Khan, Anis Kacem, Gleb Gusev, and Djamila Aouada. Sharp challenge 2023: Solving cad history and parameters recovery from point clouds and 3d scans. In Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops, pages 1778–1787, 2023.

[23] Gary Marcus. The next decade in ai: Four steps towards robust artificial intelligence. arXiv preprint arXiv:2002.06177, 2020.

[24] David Marr. Vision: A Computational Investigation into the Human Representation and Processing of Visual Information. W. H. Freeman, San Francisco, 1982.

[25] Calvin Maurer, Rensheng Qi, and Vijay Raghavan. A linear time algorithm for computing exact euclidean distance transforms of binary images in arbitrary dimensions. IEEE Transactions on Pattern Analysis and Machine Intelligence, 25(2):265–270, 2003.

[26] Farzin Mokhtarian and Alan K. Mackworth. A theory of multiscale, curvature-based shape representation for planar curves. IEEE Transactions on Pattern Analysis and Machine Intelligence, 14(8):789–805, 1992.

[27] Allen Newell and Herbert A Simon. Computer science as empirical inquiry: Symbols and search. Communications of the ACM, 19(3):113– 126, 1976.

[28] Jeong Joon Park, Peter Florence, Julian Straub, Richard Newcombe, and Steven Lovegrove. Deepsdf: Learning continuous signed distance functions for shape representation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 165– 174, 2019.

[29] Les Piegl and Wayne Tiller. The NURBS Book. Springer Science & Business Media, 1997.

[30] Nikhila Ravi et al. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024.

[31] Aristides AG Requicha. Representations for rigid solids: Theory, methods, and systems. ACM Computing Surveys (CSUR), 12(4):437– 464, 1980.

[32] Gopal Sharma, Rishabh Goyal, Donglai Liu, Evangelos Kalogerakis, and Subhransu Maji. Csgnet: Neural shape parser for constructive solid geometry. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5515–5523, 2018.

[33] Elizabeth S Spelke and Katherine D Kinzler. Core knowledge. Developmental Science, 10(1):89–96, 2007.

[34] Emma Strubell, Ananya Ganesh, and Andrew McCallum. Energy and policy considerations for deep learning in nlp. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 3645–3650, 2019.

[35] Gabriel Taubin. A signal processing approach to fair surface design. Proceedings of SIGGRAPH, pages 351–358, 1995.

[36] Joshua B Tenenbaum, Charles Kemp, Thomas L Griffiths, and Noah D Goodman. How to grow a mind: Statistics, structure, and abstraction. Science, 331(6022):1279–1285, 2011.

[37] Andrew P Witkin. Scale-space filtering. In Proceedings of the 8th International Joint Conference on Artificial Intelligence, pages 1019– 1022, 1983.

[38] Rundi Wu, Chang Xiao, and Changxi Zheng. Deepcad: A deep generative network for computer-aided design models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 6772–6782, 2021.

[39] Ruida Zhang, Chenyang Zhang, Yan Di, Fabian Manhardt, Xingyu Liu, Federico Tombari, and Xiangyang Ji. Kp-red: Exploiting semantic keypoints for joint 3d shape retrieval and deformation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20540–20550, 2024.

![](images/386d1cf51f20088b1212363325db5ac1a3ef781c63460052ac300645dc6e4d86.jpg)  
Fig. 6. Viewpoint robustness evaluation using arbitrary projections of ModelNet40 point-cloud objects. The figure illustrates the complete observation-tosymbol transformation $( { \mathcal { O } } \to { \mathcal { T } } \to { \mathcal { G } } \to S )$ for airplane, car, chair, guitar, monitor, plant, desk, person, stairs, and stool models. Each object is processed through the stages of projected observation, Euclidean Distance Transform, topology-aware structural core extraction, topology-guided segmentation, adaptive contour abstraction, sparse geometric control polygon generation, Area Spline field construction, and final symbolic representation $F ( x , y ) = 0$ . Unlike canonical top, front, and side projections, the arbitrary viewing direction introduces more complex contour interactions, overlapping structures, and mixed surface visibility. The successful generation of compact symbolic representations demonstrates that the proposed abstraction framework remains stable across substantial viewpoint variations.